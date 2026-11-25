"""Pass 1: split every table into customer partitions on disk.

Each source table is read in bounded batches. For every row we work out its
customer code, its UTC timestamp and its event id, decide whether it has to
be dropped (and count why), write its field values as text, and append it to
the partition that holds its customer.

The rows stay wide here - one row per source row, not one per field - which
is 10 to 20 times smaller than the key/value form, so the disk copy is small
and this pass is mostly reading and writing.

Partitions are contiguous ranges of customer codes. Partition p holds codes
c with c * P // N == p, so concatenating partitions in order keeps customers
in id order, and pass 2 can cut any range of customers it likes.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

from .customers import Customers, dimension_frame, in_slice, key_expr, lookup_slices, \
    parent_lookup
from .mapping import Mapping, MappingError, TableMap
from .resources import Budget, PeakMemory, process_rss
from .shapes import MAX_ROWS_PER_TABLE, UID_SHIFT, entity_expr, time_expr, value_expr
from .tables import TableFiles, iter_batches

# share of the free room one batch may use in this pass. The rest covers the
# open partition files and what our measurement misses.
BATCH_SHARE = 0.5

# never plan on fewer bytes per row than this, however small a probe looks
MIN_ROW_BYTES = 64

# the first batch of a table is a probe: small, so we learn its real size in
# memory before committing to a big one
PROBE_ROWS = 65_536
MIN_BATCH_ROWS = 4_096
MAX_BATCH_ROWS = 4_000_000

# a batch may be at most this many times the one before it
GROWTH = 4

# the reasons a row can be dropped, in the order they are tested
NO_CUSTOMER = "no customer id"
NO_PARENT = "no matching parent"
NO_TIME = "no timestamp"
FUTURE = "timestamp after the run clock"


@dataclass
class TablePlan:
    """Everything pass 1 and pass 2 need to know about one mapped table."""

    tmap: TableMap
    table: TableFiles
    kind: str                       # "event" or "master"
    ordinal: int                    # event tables 1.., masters 0..
    fields: list[str]               # source columns that become fields
    keys: list[str]                 # the names the model sees for them
    dtypes: dict                    # source column -> dtype, enrichment included
    key_ids: list[int] = field(default_factory=list)   # events: ids in the key dictionary
    lifelong: list[str] = field(default_factory=list)  # masters: milestone columns
    enrich: list[tuple[TableMap, list[str]]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)   # columns whose type cannot be a field

    @property
    def name(self) -> str:
        return self.tmap.name

    @property
    def staged(self) -> list[str]:
        """Column names of the fields once staged: f0, f1, ..."""
        return [f"f{i}" for i in range(len(self.fields))]

    @property
    def is_snapshot(self) -> bool:
        return self.tmap.shape == "periodic_snapshot"


@dataclass
class TableStage:
    """What pass 1 did to one table."""

    rows_read: int = 0
    staged: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    batches: int = 0
    seconds: float = 0.0
    peak_rss: int = 0


# ---------------------------------------------------------------- partition files

class PartitionWriter:
    """One Arrow IPC stream per customer partition, opened when first needed.

    IPC appends a batch with no re-encoding and reads straight back into
    Polars. lz4 keeps the files small at almost no CPU cost.
    """

    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        codec = "lz4" if pa.Codec.is_available("lz4") else None
        self.options = ipc.IpcWriteOptions(compression=codec)
        self.schema: pa.Schema | None = None
        self.open: dict[int, tuple] = {}

    def write(self, part: int, table: pa.Table) -> None:
        if self.schema is None:
            self.schema = table.schema
        elif not table.schema.equals(self.schema):
            table = table.cast(self.schema)
        if part not in self.open:
            sink = pa.OSFile(str(self.folder / f"part-{part:05d}.arrows"), "wb")
            self.open[part] = (sink, ipc.new_stream(sink, self.schema, options=self.options))
        self.open[part][1].write_table(table)

    def close(self) -> None:
        for sink, writer in self.open.values():
            writer.close()
            sink.close()
        self.open.clear()


def partition_of(codes: pl.Expr, n_customers: int, n_partitions: int) -> pl.Expr:
    """Contiguous ranges of customer codes, one per partition."""
    return (codes.cast(pl.Int64) * n_partitions // max(n_customers, 1)).cast(pl.Int32)


def first_code(part: int, n_customers: int, n_partitions: int) -> int:
    """The smallest code partition ``part`` holds."""
    return -(-part * n_customers // n_partitions)


def read_partition(folder: Path, part: int) -> pl.DataFrame | None:
    """One staged partition of one table, or None when it holds no rows."""
    path = folder / f"part-{part:05d}.arrows"
    if not path.exists():
        return None
    return pl.read_ipc_stream(path)


# ---------------------------------------------------------------- batch sizing

class BatchSizer:
    """Pick each batch's row count from the memory the last one really used.

    Three rules keep the estimate on the safe side:

      * memory is measured against a floor - the lowest the process has been
        between batches - not against its size just before a batch. The
        allocator keeps freed memory and hands it out again, so growth during
        one batch understates what a batch costs;
      * a reading that shows no growth tells us nothing and changes nothing
        (the allocator returning memory mid-batch once made a 4M-row batch
        look free, and it went straight over the cap);
      * the next batch may be at most GROWTH times the last, so a wrong
        estimate is caught after a small step instead of a giant one.
    """

    def __init__(self, budget: Budget, table: TableFiles):
        self.budget = budget
        self.floor = process_rss()
        # parquet's own size figures understated the real in-memory cost about
        # tenfold on the sample data, so the first guess is deliberately high
        self.bytes_per_row = max(table.raw_bytes_per_row * 10, MIN_ROW_BYTES)
        room = max(self.budget.ram_cap - self.floor, 0)
        probe = int(room * 0.1 / self.bytes_per_row)
        self.rows = min(PROBE_ROWS, max(probe, MIN_BATCH_ROWS), max(table.rows, 1))

    def next(self) -> int:
        return self.rows

    def learn(self, rows: int, peak: int) -> None:
        if rows <= 0:
            return
        # memory the allocator gave back lowers the floor
        self.floor = min(self.floor, process_rss())
        used = peak - self.floor
        if used > 0:
            seen = max(used / rows, MIN_ROW_BYTES)
            # trust a measured rise at once; let a measured fall pull us down slowly
            self.bytes_per_row = seen if seen > self.bytes_per_row else \
                0.7 * self.bytes_per_row + 0.3 * seen
        room = max(self.budget.ram_cap - self.floor, 0)
        fit = int(room * BATCH_SHARE / (self.bytes_per_row * 1.25))
        self.rows = min(max(fit, MIN_BATCH_ROWS), MAX_BATCH_ROWS,
                        max(rows * GROWTH, MIN_BATCH_ROWS))


# ---------------------------------------------------------------- one table

def _count_drops(df: pl.DataFrame, stage: TableStage) -> None:
    counts = df.group_by("__reason").agg(pl.len().alias("n")).filter(
        pl.col("__reason").is_not_null())
    for reason, n in counts.iter_rows():
        stage.dropped[reason] = stage.dropped.get(reason, 0) + int(n)


def _account(kept: pl.DataFrame, pairs: pl.Expr, rows_by_code: np.ndarray,
             pairs_by_code: np.ndarray) -> None:
    """Add this batch's rows and field values to the per-customer totals."""
    per = kept.group_by("code").agg(pl.len().alias("rows"), pairs.sum().alias("pairs"))
    codes = per.get_column("code").to_numpy().astype(np.int64)
    np.add.at(rows_by_code, codes, per.get_column("rows").to_numpy())
    np.add.at(pairs_by_code, codes, per.get_column("pairs").to_numpy())


def _write_partitions(kept: pl.DataFrame, writer: PartitionWriter,
                      n_customers: int, n_partitions: int) -> None:
    """Append each partition's slice of the batch to its file."""
    if kept.height == 0:
        return
    kept = kept.with_columns(partition_of(pl.col("code"), n_customers, n_partitions)
                             .alias("__part")).sort("__part", maintain_order=True)
    parts = kept.get_column("__part").to_numpy()
    # plain (large) strings, not string views: a slice of a string-view column
    # still drags the whole batch's text along, and writing one slice per
    # partition then wrote the batch ~30 times over - measured, not guessed
    body = kept.drop("__part").to_arrow()
    edges = np.flatnonzero(np.diff(parts)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(parts)]))
    for s, e in zip(starts, ends):
        writer.write(int(parts[s]), body.slice(int(s), int(e - s)))


def _source_columns(plan: TablePlan) -> list[str]:
    """The columns pass 1 reads from the source table."""
    t = plan.tmap
    cols: list[str] = []
    for c in (t.entity, t.time, t.snapshot_of, t.latest_by, t.as_of):
        if c and c not in cols:
            cols.append(c)
    if t.shape == "child_of_event":
        cols.append(t.resolve_entity_via["key"])
    for _dim, _added in plan.enrich:
        if _dim.enrich_on not in cols:
            cols.append(_dim.enrich_on)
    enriched = {c for _dim, added in plan.enrich for c in added}
    for c in plan.fields + plan.lifelong:
        if c not in cols and c not in enriched:
            cols.append(c)
    return [c for c in cols if c in plan.table.schema]


def _event_batch(df: pl.DataFrame, plan: TablePlan, mapping: Mapping,
                 now: datetime) -> pl.DataFrame:
    """Customer, time, event id and drop reason for a batch of an event table."""
    t = plan.tmap
    if t.shape == "child_of_event":
        no_code = NO_PARENT
    else:
        no_code = NO_CUSTOMER

    df = df.with_columns(time_expr(t.time, plan.dtypes[t.time], mapping.timezone, "__ts"))
    reason = (pl.when(pl.col("__code").is_null()).then(pl.lit(no_code))
                .when(pl.col("__ts").is_null()).then(pl.lit(NO_TIME)))
    if mapping.drop_future_events:
        reason = reason.when(pl.col("__ts") > pl.lit(now, dtype=pl.Datetime("us"))) \
                       .then(pl.lit(FUTURE))
    return df.with_columns(reason.otherwise(None).alias("__reason"))


def _event_stage_frame(kept: pl.DataFrame, plan: TablePlan) -> pl.DataFrame:
    """The columns pass 2 needs from an event table, values already text."""
    t = plan.tmap
    cols = [pl.col("__code").alias("code"),
            pl.col("__ts").alias("ts"),
            (pl.lit(plan.ordinal << UID_SHIFT, dtype=pl.Int64) + pl.col("__row")).alias("uid")]
    if plan.is_snapshot:
        subject = t.snapshot_of or t.entity
        cols.append(pl.col(subject).cast(pl.String).alias("_subject"))
    for src, staged in zip(plan.fields, plan.staged):
        cols.append(value_expr(src, plan.dtypes[src], staged))
    return kept.select(cols)


def _master_stage_frame(kept: pl.DataFrame, plan: TablePlan,
                        mapping: Mapping) -> pl.DataFrame:
    """The columns pass 2 needs from a master table."""
    t = plan.tmap
    cols = [pl.col("__code").alias("code"), pl.col("__row").alias("_row")]
    if t.latest_by:
        cols.append(time_expr(t.latest_by, plan.dtypes[t.latest_by], mapping.timezone, "_latest"))
    if t.as_of:
        cols.append(time_expr(t.as_of, plan.dtypes[t.as_of], mapping.timezone, "_as_of"))
    for i, c in enumerate(plan.lifelong):
        cols.append(time_expr(c, plan.dtypes[c], mapping.timezone, f"l{i}"))
    for src, staged in zip(plan.fields, plan.staged):
        cols.append(value_expr(src, plan.dtypes[src], staged))
    return kept.select(cols)


def stage_table(plan: TablePlan, mapping: Mapping, tables: dict[str, TableFiles],
                customers: Customers, n_partitions: int, now: datetime,
                budget: Budget, folder: Path) -> tuple[TableStage, np.ndarray, np.ndarray]:
    """Run pass 1 for one table into ``folder``.

    Returns the counts, plus how many rows and field values each customer
    got - pass 2 sizes its batches from those.
    """
    t = plan.tmap
    table = plan.table
    if table.rows > MAX_ROWS_PER_TABLE:
        raise MappingError(f"{t.name}: {table.rows:,} rows is more than an event id can number")

    n_customers = customers.count
    stage = TableStage(rows_read=table.rows)
    rows_by_code = np.zeros(n_customers, dtype=np.int64)
    pairs_by_code = np.zeros(n_customers, dtype=np.int64)
    t_start = time.time()

    # small lookups that every batch joins against
    lookup_frame = customers.frame().rename({"_id": "__id", "code": "__code"})
    dims = []
    for dim_map, added in plan.enrich:
        dim_table = tables[dim_map.name]
        dim = dimension_frame(dim_table, dim_map, added, PROBE_ROWS, budget.room())
        dims.append((dim_map.enrich_on, dim))

    # a child table may need its parent lookup cut into slices to fit
    n_slices = 1
    if t.shape == "child_of_event":
        parent_map = mapping.tables[t.resolve_entity_via["parent"]]
        n_slices = lookup_slices(tables[parent_map.name], budget.room())
        if n_slices > 1:
            print("    %s: parent lookup is large; processing it in %d slices"
                  % (t.name, n_slices))

    columns = _source_columns(plan)
    if plan.kind == "event":
        pairs = (pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int64) for c in plan.staged])
                 if plan.staged else pl.lit(1, dtype=pl.Int64))
    else:
        named = plan.staged + [f"l{i}" for i in range(len(plan.lifelong))]
        pairs = (pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int64) for c in named])
                 if named else pl.lit(0, dtype=pl.Int64))

    writer = PartitionWriter(folder)
    sizer = BatchSizer(budget, table)
    try:
        for slice_index in range(n_slices):
            lookup = None
            if t.shape == "child_of_event":
                key = t.resolve_entity_via["key"]
                lookup = parent_lookup(tables[parent_map.name], parent_map, key, customers,
                                       PROBE_ROWS, n_slices, slice_index)
                lookup = lookup.rename({"_key": "__key", "code": "__code"})

            for df, offset in iter_batches(table, columns, sizer.next):
                with PeakMemory() as mem:
                    df = df.with_columns((pl.int_range(pl.len(), dtype=pl.Int64) + offset)
                                         .alias("__row"))

                    # bring in dimension columns, keyed one row per fact row
                    for on, dim in dims:
                        df = df.join(dim, on=on, how="left", maintain_order="left")

                    # find each row's customer code
                    if t.shape == "child_of_event":
                        df = df.with_columns(key_expr(t.resolve_entity_via["key"], "__key"))
                        if n_slices > 1:
                            # a row with no key belongs to slice 0 only, so it is
                            # counted once. The null check comes first on purpose:
                            # the hash of a null is not null and lands in some slice.
                            has_key = pl.col("__key").is_not_null()
                            mine = has_key & in_slice(n_slices, slice_index, "__key")
                            if slice_index == 0:
                                mine = mine | ~has_key
                            df = df.filter(mine)
                        df = df.join(lookup, on="__key", how="left", maintain_order="left")
                    else:
                        df = df.with_columns(entity_expr(t.entity, "__id"))
                        df = df.join(lookup_frame, on="__id", how="left", maintain_order="left")

                    # decide what is dropped and why
                    if plan.kind == "event":
                        df = _event_batch(df, plan, mapping, now)
                    else:
                        df = df.with_columns(
                            pl.when(pl.col("__code").is_null()).then(pl.lit(NO_CUSTOMER))
                              .otherwise(None).alias("__reason"))
                    _count_drops(df, stage)
                    kept = df.filter(pl.col("__reason").is_null())

                    # the staged form, then out to the partitions
                    if plan.kind == "event":
                        staged = _event_stage_frame(kept, plan)
                    else:
                        staged = _master_stage_frame(kept, plan, mapping)
                    _account(staged, pairs, rows_by_code, pairs_by_code)
                    _write_partitions(staged, writer, n_customers, n_partitions)
                    stage.staged += staged.height
                    stage.batches += 1
                    batch_rows = df.height
                    del df, kept, staged

                sizer.learn(batch_rows, mem.peak)
                stage.peak_rss = max(stage.peak_rss, mem.peak)
    finally:
        writer.close()

    stage.seconds = time.time() - t_start
    return stage, rows_by_code, pairs_by_code


# ---------------------------------------------------------------- saving progress

def save_table(folder: Path, stage: TableStage, rows_by_code: np.ndarray,
               pairs_by_code: np.ndarray) -> None:
    """Record a finished table, so a resumed run can skip it."""
    np.save(folder / "rows_by_code.npy", rows_by_code)
    np.save(folder / "pairs_by_code.npy", pairs_by_code)
    doc = {"rows_read": stage.rows_read, "staged": stage.staged, "dropped": stage.dropped,
           "batches": stage.batches, "seconds": stage.seconds, "peak_rss": stage.peak_rss}
    (folder / "stage.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load_table(folder: Path) -> tuple[TableStage, np.ndarray, np.ndarray] | None:
    """A finished table from an earlier run, or None if it never finished."""
    meta = folder / "stage.json"
    if not meta.exists():
        return None
    doc = json.loads(meta.read_text(encoding="utf-8"))
    stage = TableStage(rows_read=doc["rows_read"], staged=doc["staged"],
                       dropped=doc["dropped"], batches=doc["batches"],
                       seconds=doc["seconds"], peak_rss=doc["peak_rss"])
    return (stage, np.load(folder / "rows_by_code.npy"),
            np.load(folder / "pairs_by_code.npy"))


def stage_into(plan: TablePlan, final: Path, **kw) -> tuple[TableStage, np.ndarray, np.ndarray]:
    """Stage one table into a temp folder and move it into place when done.

    A crash halfway leaves only the temp folder, which the next run deletes,
    so a half-written table can never be mistaken for a finished one.
    """
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    stage, rows_by_code, pairs_by_code = stage_table(plan, folder=tmp, **kw)
    save_table(tmp, stage, rows_by_code, pairs_by_code)
    shutil.rmtree(final, ignore_errors=True)
    move_into_place(tmp, final)
    return stage, rows_by_code, pairs_by_code


def move_into_place(tmp: Path, final: Path, attempts: int = 8) -> None:
    """Rename a finished file or folder into place, retrying briefly when held.

    On Windows a virus scanner or the search indexer can hold a file it just
    saw being written, and a rename then fails with "access denied" for a
    moment. Linux never does this, but the code has to survive both.
    """
    for attempt in range(attempts):
        try:
            os.replace(tmp, final)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1 * (attempt + 1))


def partitions_for(total_cost: float, target: float, n_customers: int,
                   file_limit: int) -> int:
    """How many customer partitions pass 1 should write.

    Aim for partitions about a quarter of a pass-2 batch, so a batch is a few
    whole partitions and can still shrink to one if the data is heavier than
    estimated. Bounded by the customer count and by how many files we may
    keep open at once.
    """
    want = math.ceil(4 * total_cost / max(target, 1.0))
    ceiling = max(min(n_customers, file_limit - 64, 8192), 1)
    return int(min(max(want, 1), ceiling))
