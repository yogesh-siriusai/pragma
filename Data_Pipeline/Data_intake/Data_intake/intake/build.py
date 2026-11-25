"""Pass 2: turn staged partitions into events.parquet and profiles.parquet.

Work goes one customer range at a time. For each range we read its staged
rows, melt them into key/value rows, apply the snapshot "changes only" rule,
sort, and group the rows into events - on the GPU when there is one - and
then build the same customers' profiles. Ranges are sized so the whole step
fits under the RAM cap (and the GPU cap), and the size is re-measured after
every range, so a heavier-than-expected client shrinks the next one.

Ranges follow customer order, so the files come out sorted by customer id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .customers import Customers
from .engine import Engine, GpuOutOfMemory, collect_where, probe_steps
from .report import FieldTally
from .resources import Budget, PeakMemory, process_rss
from .shapes import UID_SHIFT, changes_only, master_attributes, master_latest, \
    master_lifelong, melt_table, sort_events
from .stage import GROWTH, TablePlan, first_code, read_partition

# the contract, imported from fluxtra when it is installed so the two can
# never drift; the copy keeps this package usable on its own
try:
    from fluxtra.core.schema import EVENTS_SCHEMA, PROFILES_SCHEMA
except ImportError:  # pragma: no cover - exercised only without fluxtra
    EVENTS_SCHEMA = pa.schema([
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("ts", pa.timestamp("us"), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("fields", pa.map_(pa.string(), pa.string()), nullable=False),
    ])
    PROFILES_SCHEMA = pa.schema([
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("as_of", pa.timestamp("us"), nullable=False),
        pa.field("attributes", pa.map_(pa.string(), pa.string()), nullable=False),
        pa.field("lifelong", pa.list_(pa.struct([
            pa.field("key", pa.string()),
            pa.field("ts", pa.timestamp("us"))])), nullable=False),
    ])

# rows per parquet row group in the output. Fixed, not tied to how ranges
# were cut, so the same input gives byte-identical files on any machine.
# Well under fluxtra's 4M-row limit for parallel tokenizing.
ROW_GROUP_ROWS = 131_072

# share of the free RAM (and of the GPU cap) one range may use
RAM_SHARE = 0.7
GPU_SHARE = 0.8

# first guesses for the memory one staged value costs while a range is built.
# Deliberately high; the measured figure replaces them after the first range.
START_RAM_BYTES = 256.0
START_GPU_BYTES = 160.0
MIN_BYTES = 32.0

# Arrow's plain string and map columns use 32-bit offsets, so no single array
# may pass 2**31 values or 2**31 bytes of text. Events and profiles are built
# in pieces well under both, cut on event and customer boundaries.
PIECE_ROWS = 1 << 24            # key/value rows per piece
PIECE_BYTES = 1 << 29           # bytes of value text per piece
MAX_UNITS_PER_RANGE = 1 << 30


@dataclass
class BuildStats:
    """What pass 2 did."""

    ranges: int = 0
    events_written: int = 0
    profiles_written: int = 0
    events_by_table: dict[int, int] = field(default_factory=dict)    # ordinal -> events
    no_fields_by_table: dict[int, int] = field(default_factory=dict) # rows with every field empty
    customers_with_events: int = 0
    peak_rss: int = 0
    over_cap: int = 0
    seconds: float = 0.0


# ---------------------------------------------------------------- output files

class RowGroupWriter:
    """Write a parquet file in row groups of exactly ROW_GROUP_ROWS rows.

    Rows arrive in pieces of whatever size a range produced; they are
    regrouped here so the file layout depends only on the data.
    """

    def __init__(self, path: Path, schema: pa.Schema):
        self.path = path
        self.schema = schema
        self.writer = pq.ParquetWriter(str(path), schema, compression="zstd")
        self.pending: list[pa.Table] = []
        self.pending_rows = 0
        self.rows = 0

    def add(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self.pending.append(table)
        self.pending_rows += table.num_rows
        while self.pending_rows >= ROW_GROUP_ROWS:
            self._flush(ROW_GROUP_ROWS)

    def _flush(self, n: int) -> None:
        joined = pa.concat_tables(self.pending)
        # one contiguous chunk per column, so page boundaries cannot depend
        # on where the pieces happened to join
        group = joined.slice(0, n).combine_chunks()
        self.writer.write_table(group, row_group_size=n)
        self.rows += n
        rest = joined.slice(n)
        # a slice is a view, and would keep the whole previous range's buffers
        # alive while the next range is built; copy the few rows left over
        if rest.num_rows:
            rest = rest.take(pa.array(np.arange(rest.num_rows)))
        self.pending = [rest] if rest.num_rows else []
        self.pending_rows = rest.num_rows

    def close(self) -> None:
        if self.pending_rows:
            self._flush(self.pending_rows)
        self.writer.close()


# ---------------------------------------------------------------- choosing ranges

class RangePlanner:
    """Cut customer ranges whose build fits in memory.

    The cost of a customer is the staged rows plus field values they hold,
    known exactly from pass 1. Bytes per unit of cost start as a cautious
    guess and are replaced by what each finished range actually used.

    A range is always sized so it also fits on the CPU. If the GPU later
    fails or runs out of memory, the same range is rebuilt on the CPU, and
    that must not be what pushes the machine out of RAM. So the RAM estimate
    is learned from CPU runs, and a GPU run may only raise it, never lower it.
    """

    def __init__(self, cost_by_code: np.ndarray, n_partitions: int,
                 budget: Budget, engine: Engine):
        self.prefix = np.concatenate(([0], np.cumsum(cost_by_code, dtype=np.int64)))
        self.n_customers = len(cost_by_code)
        self.n_partitions = n_partitions
        self.budget = budget
        self.engine = engine
        # measured against a floor, with the same three safety rules as the
        # pass 1 batch sizer (see stage.BatchSizer): lowest size between
        # ranges, no-growth readings ignored, at most GROWTH times bigger
        self.floor = process_rss()
        self.ram_bytes = START_RAM_BYTES
        self.gpu_bytes = START_GPU_BYTES
        self.last_units = 0

    def units(self, lo: int, hi: int) -> int:
        return int(self.prefix[hi] - self.prefix[lo])

    def target(self) -> float:
        """How many units of cost the next range may hold."""
        ram = max(self.budget.ram_cap - self.floor, 0) * RAM_SHARE / self.ram_bytes
        if self.engine.is_gpu and self.budget.gpu_cap:
            ram = min(ram, self.budget.gpu_cap * GPU_SHARE / self.gpu_bytes)
        # the first range is a probe; after that, grow a step at a time
        step = ram / 4 if not self.last_units else self.last_units * GROWTH
        return max(min(ram, step, MAX_UNITS_PER_RANGE), 1.0)

    def part_of(self, code: int) -> int:
        return code * self.n_partitions // self.n_customers

    def next_range(self, lo: int) -> int:
        """The end (exclusive) of the range that starts at code ``lo``."""
        target = self.target()
        p = self.part_of(lo)
        first = first_code(p, self.n_customers, self.n_partitions)

        # starting on a partition boundary: take as many whole partitions as fit
        if lo == first:
            end_p = p
            while end_p < self.n_partitions:
                nxt = first_code(end_p + 1, self.n_customers, self.n_partitions)
                if self.units(first, nxt) > target:
                    break
                end_p += 1
            if end_p > p:
                return first_code(end_p, self.n_customers, self.n_partitions)

        # one partition is more than fits: cut it by customer, never splitting one
        limit = first_code(p + 1, self.n_customers, self.n_partitions)
        hi = int(np.searchsorted(self.prefix, self.prefix[lo] + target, side="right")) - 1
        return min(max(hi, lo + 1), limit)

    def learn(self, lo: int, hi: int, peak: int, where: str) -> None:
        units = self.units(lo, hi)
        if units <= 0:
            return
        self.last_units = units
        self.floor = min(self.floor, process_rss())
        used = peak - self.floor
        if used <= 0:
            return
        seen = max(used / units * 1.2, MIN_BYTES)
        if seen > self.ram_bytes:
            # a heavier range takes effect at once, whatever ran it
            self.ram_bytes = seen
        elif where == "cpu":
            # only a CPU run may ease the estimate down: it is the fallback
            self.ram_bytes = 0.8 * self.ram_bytes + 0.2 * seen

    def gpu_too_small(self) -> None:
        """The GPU ran out of memory: plan smaller ranges for it from now on."""
        self.gpu_bytes *= 1.5


# ---------------------------------------------------------------- reading a range

def _read_range(folder: Path, parts: range, lo: int, hi: int,
                aligned: bool) -> pl.DataFrame | None:
    frames = []
    for p in parts:
        df = read_partition(folder, p)
        if df is not None and df.height:
            frames.append(df)
    if not frames:
        return None
    df = pl.concat(frames, how="vertical", rechunk=False)
    if not aligned:
        df = df.filter((pl.col("code") >= lo) & (pl.col("code") < hi))
    return df


# ---------------------------------------------------------------- events

def _events_plan(frames: list[tuple[TablePlan, pl.DataFrame]], key_dtype,
                 engine: Engine, planner: RangePlanner, label: str) -> pl.LazyFrame:
    """The melt-diff-sort for one range.

    Two steps, so each can run wherever it is supported: the snapshot diff
    first, then the melt and sort of everything together, which is the heavy
    part and the one we most want on the GPU.
    """
    longs = []
    snapshots = []
    for plan, df in frames:
        long = melt_table(df.lazy(), plan.staged, plan.key_ids, key_dtype, plan.is_snapshot)
        if plan.is_snapshot and plan.tmap.emit == "changes_only":
            snapshots.append(changes_only(long))
        elif plan.is_snapshot:
            longs.append(long.drop("_subject"))
        else:
            longs.append(long)

    if snapshots:
        diffed, _where = _run_step(pl.concat(snapshots, how="vertical"), engine, planner,
                                   f"{label} snapshots", "snapshots")
        longs.append(diffed.lazy())
    return sort_events(pl.concat(longs, how="vertical"))


def _run_step(lf: pl.LazyFrame, engine: Engine, planner: RangePlanner,
              label: str, step: str) -> tuple[pl.DataFrame, str]:
    """Run one step, rebuilding the range on the CPU if the GPU is too small.

    Ranges are sized to fit on the CPU (see RangePlanner), so this fallback
    cannot be the thing that runs the machine out of RAM.
    """
    try:
        return collect_where(lf, engine, label, step)
    except GpuOutOfMemory as exc:
        planner.gpu_too_small()
        print("    %s did not fit on the GPU (%s); this range runs on the cpu"
              % (label, str(exc).splitlines()[0][:120]))
        t0 = time.time()
        df = lf.collect()
        engine.runs["cpu"] += 1
        print("    %-34s %10s rows   %6.2fs  [cpu]" % (label, format(df.height, ","),
                                                     time.time() - t0))
        return df, "cpu"


def _event_pieces(long: pl.DataFrame) -> list[tuple[int, int]]:
    """Row ranges of ``long`` that each start and end on an event boundary.

    Each piece stays under PIECE_ROWS rows and PIECE_BYTES of value text, so
    its map column fits Arrow's 32-bit offsets however big the range was.
    """
    n = long.height
    if n == 0:
        return []
    uid = long.get_column("uid").to_numpy()
    starts = np.flatnonzero(np.concatenate(([True], uid[1:] != uid[:-1])))
    text = np.cumsum(long.get_column("value").str.len_bytes().fill_null(0).to_numpy(),
                     dtype=np.int64)

    pieces, a = [], 0
    while a < n:
        # the furthest event start that keeps the piece under both limits
        row_limit = a + PIECE_ROWS
        byte_limit = np.searchsorted(text, (text[a - 1] if a else 0) + PIECE_BYTES, side="right")
        limit = min(row_limit, byte_limit, n)
        i = np.searchsorted(starts, limit, side="right") - 1
        b = int(starts[i]) if limit < n else n
        if b <= a:
            # a single event bigger than a piece: give it a piece of its own
            nxt = np.searchsorted(starts, a, side="right")
            b = int(starts[nxt]) if nxt < starts.size else n
        pieces.append((a, b))
        a = b
    return pieces


def _to_events(long: pl.DataFrame, customer_ids: pa.Array, key_names: pa.Array,
               source_names: pa.Array) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    """Sorted key/value rows (one piece) to one row per event, with a real Arrow map.

    Returns the table plus each event's customer code and table ordinal, for
    the counts.
    """
    uid = long.get_column("uid").to_numpy()
    n = uid.size
    if n == 0:
        empty = pa.table({name: pa.array([], type=f.type)
                          for name, f in zip(EVENTS_SCHEMA.names, EVENTS_SCHEMA)},
                         schema=EVENTS_SCHEMA)
        return empty, np.empty(0, np.int64), np.empty(0, np.int64)
    if n >= (1 << 31):
        raise ValueError(f"{n:,} values in one piece overflow a 32-bit map offset")

    # an event is a run of rows sharing one id; the sort keeps each run together
    starts = np.flatnonzero(np.concatenate(([True], uid[1:] != uid[:-1])))
    offsets = np.concatenate((starts, [n])).astype(np.int32)

    codes = long.get_column("code").to_numpy()[starts].astype(np.int64)
    ordinals = (uid[starts] >> UID_SHIFT).astype(np.int64)
    key_ids = pa.array(long.get_column("key_id").to_numpy().astype(np.int64))

    fields = pa.MapArray.from_arrays(
        pa.array(offsets, pa.int32()),
        key_names.take(key_ids),
        long.get_column("value").to_arrow().cast(pa.string()))

    ts = long.get_column("ts").to_arrow().cast(pa.timestamp("us")).take(pa.array(starts))
    table = pa.table({
        "user_id": customer_ids.take(pa.array(codes)).cast(pa.string()),
        "ts": ts,
        "source": source_names.take(pa.array(ordinals)),
        "fields": fields,
    }, schema=EVENTS_SCHEMA)
    return table, codes, ordinals


# ---------------------------------------------------------------- profiles

def _profile_parts(masters: list[tuple[TablePlan, pl.DataFrame]], event_codes: np.ndarray,
                   now: datetime) -> tuple[np.ndarray, pl.DataFrame, pl.DataFrame, pl.Series]:
    """(customer codes, attributes, milestones, as_of) for one range, all sorted by code.

    Every master customer gets a row, and so does every customer with events.
    A later master wins when two carry the same attribute; within a master the
    later source row wins. Milestones are ordered by time, then name. as_of
    comes from the latest master that has one, then the latest source row.
    """
    attrs_parts, life_parts, as_of_parts, code_parts = [], [], [], []
    for master_ord, (plan, df) in enumerate(masters):
        lf = master_latest(df.lazy(), plan.tmap)
        code_parts.append(lf.select("code"))
        attrs_parts.append(master_attributes(lf, plan.staged, plan.keys, master_ord))
        life_cols = [f"l{i}" for i in range(len(plan.lifelong))]
        life_parts.append(master_lifelong(lf, life_cols, plan.lifelong))
        if plan.tmap.as_of:
            as_of_parts.append(lf.select(pl.col("code"),
                                         pl.lit(master_ord, dtype=pl.Int32).alias("_master"),
                                         pl.col("_row"), pl.col("_as_of"))
                                 .filter(pl.col("_as_of").is_not_null()))

    # who gets a profile row
    users = pl.Series("code", event_codes, dtype=pl.UInt32)
    if code_parts:
        users = pl.concat([users, pl.concat(code_parts).collect().get_column("code")])
    users = users.unique().sort()

    # attributes: later master, then later row, wins per key
    if attrs_parts:
        attrs = (pl.concat(attrs_parts).collect()
                   .sort(["code", "key", "_master", "_row"])
                   .unique(subset=["code", "key"], keep="last", maintain_order=True))
    else:
        attrs = pl.DataFrame(schema={"code": pl.UInt32, "key": pl.String, "value": pl.String})

    if life_parts:
        life = pl.concat(life_parts).collect().sort(["code", "ts", "key"])
    else:
        life = pl.DataFrame(schema={"code": pl.UInt32, "key": pl.String, "ts": pl.Datetime("us")})

    # as_of: sorted on every column that could tie, so the pick is the same every run
    as_of = pl.DataFrame({"code": users}).lazy()
    if as_of_parts:
        known = (pl.concat(as_of_parts).sort(["code", "_master", "_row", "_as_of"])
                   .unique(subset=["code"], keep="last", maintain_order=True)
                   .select("code", "_as_of"))
        as_of = as_of.join(known, on="code", how="left", maintain_order="left")
    else:
        as_of = as_of.with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("_as_of"))
    as_of = as_of.select(pl.col("_as_of").fill_null(pl.lit(now, dtype=pl.Datetime("us")))) \
                 .collect().get_column("_as_of")

    return users.to_numpy().astype(np.int64), attrs, life, as_of


def _profile_pieces(user_codes: np.ndarray, a_starts: np.ndarray,
                    l_starts: np.ndarray) -> list[tuple[int, int]]:
    """User ranges whose attributes and milestones each fit 32-bit offsets."""
    n = user_codes.size
    pieces, a = [], 0
    while a < n:
        b = min(n, np.searchsorted(a_starts, a_starts[a] + PIECE_ROWS, side="right") - 1,
                np.searchsorted(l_starts, l_starts[a] + PIECE_ROWS, side="right") - 1)
        b = max(int(b), a + 1)
        pieces.append((a, b))
        a = b
    return pieces


def _profile_tables(masters: list[tuple[TablePlan, pl.DataFrame]], event_codes: np.ndarray,
                    customer_ids: pa.Array, now: datetime) -> list[pa.Table]:
    """Profiles for one range, as one or more tables in customer order."""
    user_codes, attrs, life, as_of = _profile_parts(masters, event_codes, now)
    if user_codes.size == 0:
        return []

    a_codes = attrs.get_column("code").to_numpy()
    l_codes = life.get_column("code").to_numpy()
    a_starts = np.concatenate((np.searchsorted(a_codes, user_codes, side="left"), [attrs.height]))
    l_starts = np.concatenate((np.searchsorted(l_codes, user_codes, side="left"), [life.height]))

    tables = []
    for a, b in _profile_pieces(user_codes, a_starts, l_starts):
        a_lo, a_hi = int(a_starts[a]), int(a_starts[b])
        l_lo, l_hi = int(l_starts[a]), int(l_starts[b])
        piece_attrs = attrs.slice(a_lo, a_hi - a_lo)
        piece_life = life.slice(l_lo, l_hi - l_lo)

        attributes = pa.MapArray.from_arrays(
            pa.array((a_starts[a:b + 1] - a_lo).astype(np.int32), pa.int32()),
            piece_attrs.get_column("key").to_arrow().cast(pa.string()),
            piece_attrs.get_column("value").to_arrow().cast(pa.string()))
        milestones = pa.StructArray.from_arrays(
            [piece_life.get_column("key").to_arrow().cast(pa.string()),
             piece_life.get_column("ts").to_arrow().cast(pa.timestamp("us"))],
            names=["key", "ts"])
        lifelong = pa.ListArray.from_arrays(
            pa.array((l_starts[a:b + 1] - l_lo).astype(np.int32), pa.int32()), milestones)

        tables.append(pa.table({
            "user_id": customer_ids.take(pa.array(user_codes[a:b])).cast(pa.string()),
            "as_of": as_of.slice(a, b - a).to_arrow().cast(pa.timestamp("us")),
            "attributes": attributes,
            "lifelong": lifelong,
        }, schema=PROFILES_SCHEMA))
    return tables


# ---------------------------------------------------------------- the whole pass

def build(event_plans: list[TablePlan], master_plans: list[TablePlan],
          stage_dir: Path, customers: Customers, n_partitions: int,
          cost_by_code: np.ndarray, key_names: list[str], key_dtype,
          now: datetime, engine: Engine, budget: Budget,
          events_path: Path, profiles_path: Path,
          tally: FieldTally) -> BuildStats:
    """Run pass 2 over every customer range and write both contract files."""
    stats = BuildStats()
    t_start = time.time()
    n_customers = customers.count

    # all ids together can pass 2 GB of text, so hold them with 64-bit offsets
    # and narrow each piece to the contract's plain strings as it is taken
    customer_ids = customers.ids.to_arrow().cast(pa.large_string())
    names = pa.array(key_names, pa.string())
    # source name by table ordinal; ordinal 0 is never used by an event table
    by_ordinal = [""] * (max([p.ordinal for p in event_plans], default=0) + 1)
    for p in event_plans:
        by_ordinal[p.ordinal] = p.tmap.source_name
    sources = pa.array(by_ordinal, pa.string())

    # find out once, up front, which steps the GPU can really run
    if engine.is_gpu:
        for step, (ok, detail) in probe_steps(engine).items():
            print("  gpu check     %-10s %s" % (step, "gpu" if ok else "cpu - " + detail))

    events_out = RowGroupWriter(events_path, EVENTS_SCHEMA)
    profiles_out = RowGroupWriter(profiles_path, PROFILES_SCHEMA)
    planner = RangePlanner(cost_by_code, n_partitions, budget, engine)
    events_per_customer = np.zeros(n_customers, dtype=np.int64)

    try:
        lo = 0
        while lo < n_customers:
            hi = planner.next_range(lo)
            parts = range(planner.part_of(lo), planner.part_of(hi - 1) + 1)
            aligned = (lo == first_code(parts[0], n_customers, n_partitions)
                       and hi == first_code(parts[-1] + 1, n_customers, n_partitions))
            label = "customers %s-%s" % (format(lo, ","), format(hi - 1, ","))
            where = "cpu"

            with PeakMemory() as mem:
                # read this range's staged rows, table by table
                frames = []
                for plan in event_plans:
                    df = _read_range(stage_dir / plan.name, parts, lo, hi, aligned)
                    if df is None:
                        continue
                    # rows whose every field is empty make no event; count them
                    if plan.staged:
                        empty = df.select(~pl.any_horizontal(
                            [pl.col(c).is_not_null() for c in plan.staged])).to_series().sum()
                        stats.no_fields_by_table[plan.ordinal] = \
                            stats.no_fields_by_table.get(plan.ordinal, 0) + int(empty)
                    frames.append((plan, df))

                masters = []
                for plan in master_plans:
                    df = _read_range(stage_dir / plan.name, parts, lo, hi, aligned)
                    if df is not None:
                        masters.append((plan, df))

                # melt, diff, sort and group into events, a piece at a time
                range_codes = []
                if frames:
                    lf = _events_plan(frames, key_dtype, engine, planner, label)
                    long, where = _run_step(lf, engine, planner, label, "events")
                    del frames, lf
                    for a, b in _event_pieces(long):
                        piece = long.slice(a, b - a)
                        events, event_codes, ordinals = _to_events(piece, customer_ids,
                                                                   names, sources)
                        tally.add(piece, ordinals_per_pair=(piece.get_column("uid").to_numpy()
                                                            >> UID_SHIFT))
                        events_out.add(events)
                        stats.events_written += events.num_rows
                        np.add.at(events_per_customer, event_codes, 1)
                        range_codes.append(np.unique(event_codes))
                        for o, n in zip(*np.unique(ordinals, return_counts=True)):
                            stats.events_by_table[int(o)] = \
                                stats.events_by_table.get(int(o), 0) + int(n)
                        del piece, events
                    del long

                # and the same customers' profiles
                event_codes = (np.unique(np.concatenate(range_codes)) if range_codes
                               else np.empty(0, np.int64))
                for profiles in _profile_tables(masters, event_codes, customer_ids, now):
                    profiles_out.add(profiles)
                    stats.profiles_written += profiles.num_rows
                del masters

            planner.learn(lo, hi, mem.peak, where)
            stats.peak_rss = max(stats.peak_rss, mem.peak)
            if mem.peak > budget.ram_cap:
                stats.over_cap += 1
                print("    %s went over the RAM cap (%.1f GB > %.1f GB); later ranges shrink"
                      % (label, mem.peak / 2**30, budget.ram_cap / 2**30))
            stats.ranges += 1
            lo = hi
    finally:
        events_out.close()
        profiles_out.close()

    stats.customers_with_events = int(np.count_nonzero(events_per_customer))
    tally.events_per_customer = events_per_customer
    stats.seconds = time.time() - t_start
    return stats
