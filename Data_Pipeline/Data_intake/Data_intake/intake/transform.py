"""The heavy lifting: client tables in, events.parquet and profiles.parquet out.

The shape of the work, in order:

  0. plan     measure the machine, read the table footers, number every
              customer (sorted ids -> codes), pick how many partitions to cut
  1. stage    read each table in bounded batches, clean each row, count every
              drop, and write the rows to customer partitions on disk
  2. build    one customer range at a time: melt, diff snapshots, sort, group
              into events and profiles, append to the two output files

Every row that comes in is accounted for at the end: it became an event, or
it was dropped for a named reason. The counts are checked to add up.

A run keeps its staged work in a work folder. If it dies, `--resume` picks up
from the last fully staged table instead of starting over.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from . import build as _build
from .customers import Customers, build_customers
from .engine import Engine
from .mapping import Mapping, MappingError
from .report import FieldTally, build_report, write_report
from .resources import GIB, Budget, disk_free, make_budget, measure, raise_open_file_limit
from .shapes import MARKER_KEY, field_columns, output_keys
from .stage import (TablePlan, TableStage, load_table, move_into_place, partitions_for,
                    stage_into)
from .tables import TableFiles, open_table

# bump when the staged format changes, so an old work folder is never reused
STAGE_FORMAT = 2

# re-exported for callers that used to import them from here
EVENTS_SCHEMA = _build.EVENTS_SCHEMA
PROFILES_SCHEMA = _build.PROFILES_SCHEMA

# batch size for small reads: customer ids, lookups, dimensions
SMALL_READ_ROWS = 262_144

# the file that marks a folder as intake's own work folder, safe to delete
WORK_MARKER = ".fluxtra-intake-work"
WORK_MARKER_TEXT = "fluxtra-intake work folder - safe to delete"

# free disk wanted, as multiples of the source size: plan on 5x, refuse under 2x
DISK_PLAN = 5
DISK_REFUSE = 2

EMPTY_FIELDS = "every field empty"
UNCHANGED = "unchanged since the previous snapshot"


@dataclass
class Stats:
    """What happened, in enough detail for the reconciliation report."""

    rows_read: dict[str, int] = field(default_factory=dict)        # event tables
    master_rows: dict[str, int] = field(default_factory=dict)      # entity_master tables
    events_by_table: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, dict[str, int]] = field(default_factory=dict)
    events_written: int = 0
    profiles_written: int = 0
    customers: int = 0
    partitions: int = 0
    ranges: int = 0
    gpu_steps: int = 0
    cpu_steps: int = 0
    peak_rss: int = 0
    seconds: float = 0.0
    reconciliation: dict[str, dict] = field(default_factory=dict)

    def drop(self, table: str, reason: str, n: int) -> None:
        if n:
            per = self.dropped.setdefault(table, {})
            per[reason] = per.get(reason, 0) + n


# ---------------------------------------------------------------- planning

def _enrichments(mapping: Mapping, tables: dict[str, TableFiles], target: str,
                 schema: dict) -> list[tuple]:
    """Dimension tables that add columns to ``target``, and which columns."""
    out = []
    for dim in mapping.by_shape("dimension"):
        if dim.action != "enrich" or dim.enrich_into != target:
            continue
        dim_schema = tables[dim.name].schema
        wanted = dim.fields.resolve(list(dim_schema), [dim.enrich_on])
        # the fact table's own column wins on a name clash
        added = [c for c in wanted if c not in schema]
        out.append((dim, added))
    return out


def plan_tables(mapping: Mapping, tables: dict[str, TableFiles]) -> tuple[
        list[TablePlan], list[TablePlan], list[str]]:
    """(event plans, master plans, key names in id order)."""
    event_plans, master_plans = [], []

    for tmap in mapping.active():
        if tmap.shape == "dimension":
            continue
        table = tables[tmap.name]
        dtypes = dict(table.schema)
        enrich = _enrichments(mapping, tables, tmap.name, dtypes)
        for dim, added in enrich:
            for c in added:
                dtypes[c] = tables[dim.name].schema[c]

        cols, skipped = field_columns(tmap, dtypes)
        keys = output_keys(tmap, cols)
        clash = sorted({k for k in keys if keys.count(k) > 1})
        if clash:
            raise MappingError(f"{tmap.name}: two columns would both become field "
                               f"{clash[0]!r}; check the rename map")

        if tmap.shape == "entity_master":
            lifelong = [c for c in tmap.lifelong if c in dtypes]
            master_plans.append(TablePlan(tmap=tmap, table=table, kind="master",
                                          ordinal=len(master_plans), fields=cols, keys=keys,
                                          dtypes=dtypes, lifelong=lifelong, enrich=enrich,
                                          skipped=skipped))
        else:
            event_plans.append(TablePlan(tmap=tmap, table=table, kind="event",
                                         ordinal=len(event_plans) + 1, fields=cols, keys=keys,
                                         dtypes=dtypes, enrich=enrich, skipped=skipped))

    # one dictionary of field names across every event table. Ids follow name
    # order, so sorting by id puts an event's fields in name order.
    names = set()
    for p in event_plans:
        names.update(p.keys if p.keys else [MARKER_KEY])
    key_names = sorted(names)
    index = {k: i for i, k in enumerate(key_names)}
    for p in event_plans:
        p.key_ids = [index[k] for k in (p.keys if p.keys else [MARKER_KEY])]
    return event_plans, master_plans, key_names


def estimate_cost(plans: list[TablePlan]) -> float:
    """Rows plus non-null field values, from footer statistics where we have them.

    Only used to choose the partition count before pass 1; the exact counts
    pass 1 produces drive everything after it.
    """
    from .tables import footer_stats
    total = 0.0
    for p in plans:
        stats = footer_stats(p.table)
        values = 0.0
        for c in p.fields + p.lifelong:
            nulls = stats.get(c).nulls if c in stats and stats.get(c) else None
            filled = (p.table.rows - nulls) if nulls is not None else p.table.rows * 0.5
            values += max(filled, 0)
        total += p.table.rows + values
    return total


# ---------------------------------------------------------------- the work folder

def fingerprint(mapping: Mapping, tables: dict[str, TableFiles], now: datetime) -> str:
    """Everything that decides the staged bytes. Same fingerprint, same staging.

    Every mapped table's files count, dimensions included: a changed lookup
    table changes the enriched values even though it is never staged itself.
    """
    doc = {
        "format": STAGE_FORMAT,
        "mapping": dataclasses.asdict(mapping),
        "now": now.isoformat(),
        "files": sorted([str(f), f.stat().st_size, f.stat().st_mtime_ns]
                        for t in tables.values() for f in t.files),
    }
    blob = json.dumps(doc, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _is_ours(work: Path) -> bool:
    """True only for a folder carrying intake's own marker file."""
    marker = work / WORK_MARKER
    return marker.exists() and marker.read_text(encoding="utf-8").strip() == WORK_MARKER_TEXT


def _check_work_location(work: Path, out_dir: Path, source_dir: Path) -> None:
    """The work folder is deleted after a run, so it must not hold anything else.

    It may not be the output folder or contain it, and it may not overlap the
    source data in either direction.
    """
    w, o, s = work.resolve(), out_dir.resolve(), source_dir.resolve()
    if w == o or o.is_relative_to(w):
        raise SystemExit(f"--work-dir {work} would hold the output folder {out_dir}, and the "
                         f"work folder is deleted after the run; choose a separate folder")
    if w.is_relative_to(s) or s.is_relative_to(w):
        raise SystemExit(f"--work-dir {work} overlaps the source data {source_dir}; "
                         f"choose a separate folder")


def _open_work(work: Path, resume: bool) -> dict | None:
    """The previous run's manifest when resuming; otherwise a clean folder.

    Only a folder carrying intake's own marker file is ever deleted. A common
    file name such as manifest.json is not proof enough: a mistyped --work-dir
    must not be able to wipe someone's project.
    """
    manifest = work / "manifest.json"
    if resume and manifest.exists() and _is_ours(work):
        return json.loads(manifest.read_text(encoding="utf-8"))
    if resume:
        print("  nothing to resume in %s; starting fresh" % work)

    if work.exists() and any(work.iterdir()) and not _is_ours(work):
        raise SystemExit(f"work folder {work} is not empty and was not made by intake; "
                         f"choose an empty or new folder for --work-dir")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    (work / WORK_MARKER).write_text(WORK_MARKER_TEXT + "\n", encoding="utf-8")
    return None


def _check_disk(work: Path, out_dir: Path, plans: list[TablePlan]) -> None:
    """Refuse to start a run that cannot fit on disk; warn when it is tight.

    Staging holds every table once in text form - measured at about 3.4x the
    parquet source on the sample - and the output is about 1x again, so plan
    on 5x the source size free.
    """
    source = sum(f.stat().st_size for p in plans for f in p.table.files)
    free = min(disk_free(work), disk_free(out_dir))
    print("  disk          %.1f GB free, source tables %.1f GB on disk"
          % (free / GIB, source / GIB))
    if free < DISK_REFUSE * source:
        raise SystemExit(f"not enough disk: {free / GIB:.1f} GB free, and staging plus output "
                         f"need about {DISK_PLAN * source / GIB:.1f} GB. Point --work-dir at a "
                         f"bigger volume.")
    if free < DISK_PLAN * source:
        print("  warning: under %dx the source size is free (%.1f GB wanted); staging and "
              "output may not fit" % (DISK_PLAN, DISK_PLAN * source / GIB))


# ---------------------------------------------------------------- the pipeline

def run(mapping: Mapping, source_dir: str | Path, out_dir: str | Path,
        engine: Engine, now: datetime | None = None, budget: Budget | None = None,
        work_dir: str | Path | None = None, resume: bool = False,
        keep_work: bool = False) -> Stats:
    """Turn one client's tables into events.parquet and profiles.parquet.

    ``now`` fixes the "future event" cutoff and the default profile as_of, so a
    run is reproducible rather than depending on the wall clock. ``budget``
    caps RAM and GPU memory; by default 85% of what is free. ``work_dir`` holds
    the staged partitions (default: a _work folder inside ``out_dir``).
    """
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir) if work_dir else out_dir / "_work"
    stage_dir = work / "stage"
    t_start = time.time()
    stats = Stats()

    if budget is None:
        budget = make_budget(measure())
    file_limit = raise_open_file_limit()

    # ---- what is there, and what each table contributes
    names = [t.name for t in mapping.active()]
    tables = {name: open_table(source_dir, name) for name in names}
    event_plans, master_plans, key_names = plan_tables(mapping, tables)
    plans = master_plans + event_plans
    for p in plans:
        if p.skipped:
            print("  %s: skipping %s - nested or binary columns cannot be a text value"
                  % (p.name, ", ".join(p.skipped)))

    # ---- the work folder: resume it, or start it clean
    _check_work_location(work, out_dir, source_dir)
    previous = _open_work(work, resume)
    if previous is not None and now is None:
        now = datetime.fromisoformat(previous["now"])
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    # everything in the data is naive UTC, so the clock must be too
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    print_fp = fingerprint(mapping, tables, now)
    if previous is not None and previous.get("fingerprint") != print_fp:
        print("  the data, mapping or clock changed since the saved run; starting fresh")
        previous = _open_work(work, resume=False)
    _check_disk(work, out_dir, plans)

    # ---- number the customers
    if previous is not None and (work / "customers.parquet").exists():
        customers = Customers(ids=pl.read_parquet(work / "customers.parquet").get_column("_id"))
    else:
        t0 = time.time()
        customers = build_customers(tables, mapping, SMALL_READ_ROWS)
        customers.ids.to_frame().write_parquet(work / "customers.parquet")
        print("  customers     %s  (%.1fs)" % (format(customers.count, ","), time.time() - t0))
    if customers.count == 0:
        raise SystemExit("no customer ids found in any mapped table; nothing to build")

    # ---- how finely to partition: about four partitions per pass-2 range
    if previous is not None:
        n_partitions = int(previous["partitions"])
    else:
        target = budget.room() * _build.RAM_SHARE / _build.START_RAM_BYTES
        n_partitions = partitions_for(estimate_cost(plans), target, customers.count, file_limit)
    stats.partitions = n_partitions
    manifest = {"fingerprint": print_fp, "now": now.isoformat(), "partitions": n_partitions,
                "format": STAGE_FORMAT}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("  partitions    %d" % n_partitions)
    print("")

    # ---- pass 1: stage every table
    print("  pass 1: staging tables into customer partitions")
    stage_dir.mkdir(parents=True, exist_ok=True)
    cost_by_code = np.zeros(customers.count, dtype=np.int64)
    staged: dict[str, TableStage] = {}
    for plan in plans:
        folder = stage_dir / plan.name
        done = load_table(folder) if previous is not None else None
        if done is not None:
            stage, rows_by_code, pairs_by_code = done
            print("    %-28s reused from the saved run" % plan.name[:28])
        else:
            stage, rows_by_code, pairs_by_code = stage_into(
                plan, folder, mapping=mapping, tables=tables, customers=customers,
                n_partitions=n_partitions, now=now, budget=budget)
            dropped = sum(stage.dropped.values())
            print("    %-28s %12s rows  %8.1fs  peak %5.1f GB  dropped %s"
                  % (plan.name[:28], format(stage.rows_read, ","), stage.seconds,
                     stage.peak_rss / GIB, format(dropped, ",")))
        staged[plan.name] = stage
        cost_by_code += rows_by_code + pairs_by_code
        stats.peak_rss = max(stats.peak_rss, stage.peak_rss)
        for reason, n in stage.dropped.items():
            stats.drop(plan.name, reason, n)
        if plan.kind == "event":
            stats.rows_read[plan.name] = stage.rows_read
        else:
            stats.master_rows[plan.name] = stage.rows_read

    # ---- pass 2: build the two files, one customer range at a time
    print("")
    print("  pass 2: building events and profiles")
    source_by_ordinal = [""] * (len(event_plans) + 1)
    for p in event_plans:
        source_by_ordinal[p.ordinal] = p.tmap.source_name
    tally = FieldTally(key_names, source_by_ordinal)
    key_dtype = pl.UInt16 if len(key_names) <= 65_535 else pl.UInt32

    events_tmp = out_dir / "events.parquet.tmp"
    profiles_tmp = out_dir / "profiles.parquet.tmp"
    built = _build.build(event_plans, master_plans, stage_dir, customers, n_partitions,
                         cost_by_code, key_names, key_dtype, now, engine, budget,
                         events_tmp, profiles_tmp, tally)

    # only a finished run replaces the previous output
    move_into_place(events_tmp, out_dir / "events.parquet")
    move_into_place(profiles_tmp, out_dir / "profiles.parquet")

    stats.events_written = built.events_written
    stats.profiles_written = built.profiles_written
    stats.customers = built.profiles_written
    stats.ranges = built.ranges
    stats.gpu_steps = engine.runs["gpu"]
    stats.cpu_steps = engine.runs["cpu"]
    stats.peak_rss = max(stats.peak_rss, built.peak_rss)

    # ---- account for every row. Two checks, one per pass:
    #   pass 1: rows in  = dropped while staging + staged
    #   pass 2: staged   = all-empty rows + unchanged snapshot rows + events
    # anything left over is a row we lost, and it is reported loudly
    by_source: dict[str, int] = {}
    for p in event_plans:
        stage = staged[p.name]
        events = built.events_by_table.get(p.ordinal, 0)
        stats.events_by_table[p.name] = events
        by_source[p.tmap.source_name] = by_source.get(p.tmap.source_name, 0) + events

        lost_staging = stage.rows_read - sum(stage.dropped.values()) - stage.staged
        empty = built.no_fields_by_table.get(p.ordinal, 0)
        rest = stage.staged - empty - events
        stats.drop(p.name, EMPTY_FIELDS, empty)
        if p.is_snapshot and p.tmap.emit == "changes_only":
            # only a snapshot may lose rows here, and never a negative number
            stats.drop(p.name, UNCHANGED, max(rest, 0))
            lost_building = min(rest, 0)
        else:
            lost_building = rest

        stats.reconciliation[p.name] = {"rows_in": stage.rows_read, "events": events,
                                        "dropped": dict(stats.dropped.get(p.name, {})),
                                        "unexplained": lost_staging + lost_building}
    for p in master_plans:
        stage = staged[p.name]
        lost_staging = stage.rows_read - sum(stage.dropped.values()) - stage.staged
        stats.reconciliation[p.name] = {"rows_in": stage.rows_read, "events": 0,
                                        "profile_rows": stage.staged,
                                        "dropped": dict(stats.dropped.get(p.name, {})),
                                        "unexplained": lost_staging}

    bad = {n: r["unexplained"] for n, r in stats.reconciliation.items() if r["unexplained"]}
    if bad:
        print("  RECONCILIATION MISMATCH - rows not accounted for: %s" % bad)

    stats.seconds = time.time() - t_start
    numeric = {p.tmap.fields.rename.get(c, c) for p in event_plans for c in p.tmap.fields.numeric}
    resources = {"ram_cap_gb": round(budget.ram_cap / GIB, 2),
                 "gpu_cap_gb": round(budget.gpu_cap / GIB, 2),
                 "peak_rss_gb": round(stats.peak_rss / GIB, 2),
                 "partitions": n_partitions, "ranges": built.ranges,
                 "gpu_steps": stats.gpu_steps, "cpu_steps": stats.cpu_steps,
                 "ranges_over_cap": built.over_cap,
                 "gpu_off_for": dict(engine.disabled) or None,
                 "seconds": round(stats.seconds, 1)}
    rep = build_report(tally, rows_in=sum(stats.rows_read.values()),
                       events_out=stats.events_written, profiles_out=stats.profiles_written,
                       by_source=by_source, numeric_keys=numeric,
                       reconciliation=stats.reconciliation, resources=resources)
    write_report(rep, out_dir)

    if not keep_work and _is_ours(work):
        shutil.rmtree(work, ignore_errors=True)
    return stats
