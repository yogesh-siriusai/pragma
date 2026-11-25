"""Everything the CLI can do, callable from Python.

All the logic lives here; intake/cli.py only parses arguments and prints.
That is the same split fluxtra uses, so anything you can run in a terminal you
can run in a notebook and the other way round.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import discover as _discover
from . import quality, report as _report, validate as _validate
from .connect import Source, is_remote, load_sources, resolve, staged, staged_out
from .engine import Engine, print_engine, resolve_engine, self_test
from .mapping import load_mapping, write_mapping
from .resources import GIB, Budget, disk_free, make_budget, measure, print_machine
from .transform import Stats, run as _run_transform

log = logging.getLogger(__name__)


def _sources(connectors: str | Path | None) -> dict[str, Source]:
    return load_sources(connectors) if connectors else {}


def _now(fixed: str | None) -> datetime:
    """A fixed clock makes a run reproducible; otherwise use now. Always naive UTC."""
    if fixed:
        clock = datetime.fromisoformat(fixed)
        if clock.tzinfo is not None:
            clock = clock.astimezone(timezone.utc).replace(tzinfo=None)
        return clock
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _setup(device: str, memory_fraction: float | None,
           gpu_memory_fraction: float | None) -> tuple[Budget, Engine]:
    """Measure the machine, set the memory caps and pick the engine, once."""
    machine = measure()
    budget = make_budget(machine, memory_fraction, gpu_memory_fraction)
    engine = resolve_engine(device, gpu_cap=budget.gpu_cap)
    print_engine(engine)
    print_machine(machine, budget)
    return budget, engine


# ---------------------------------------------------------------- doctor

def doctor(device: str = "auto", memory_fraction: float | None = None,
           gpu_memory_fraction: float | None = None, work_dir: str | Path = ".") -> dict[str, Any]:
    """Say what this machine has and prove the GPU really runs the transform's steps.

    Run this first on a new machine. It answers "will intake use the GPU?"
    with an actual GPU query, not with what the log claims.
    """
    budget, engine = _setup(device, memory_fraction, gpu_memory_fraction)
    print("  disk          %.1f GB free at %s" % (disk_free(work_dir) / GIB, Path(work_dir).resolve()))

    import polars as pl
    import pyarrow as pa
    print("  python        %s" % sys.version.split()[0])
    print("  polars        %s" % pl.__version__)
    print("  pyarrow       %s" % pa.__version__)
    try:
        import cudf_polars
        print("  cudf-polars   %s" % getattr(cudf_polars, "__version__", "installed"))
    except Exception as exc:  # noqa: BLE001 - reporting, not failing
        print("  cudf-polars   not usable (%s)" % str(exc).splitlines()[0][:100])

    ok, detail = self_test(engine)
    print("")
    print("  gpu self-test %s - %s" % ("PASSED" if ok else "not run on gpu", detail))
    print("")
    return {"gpu_ok": ok, "detail": detail, "engine": engine, "budget": budget}


# ---------------------------------------------------------------- discover

def discover(source: str, out: str | Path | None = None, client: str = "",
             connectors: str | Path | None = None,
             device: str = "auto") -> dict[str, Any]:
    """Scan client tables and write a draft mapping for review.

    Returns the draft mapping, the per-table facts behind it, and where the
    draft was written.
    """
    src = resolve(source, _sources(connectors))
    print("")
    print("  source        %s" % src.uri)

    with staged(src) as path:
        mapping, facts = _discover.draft(path, client)
        _discover.print_draft(mapping, facts)

        written = None
        if out:
            written = write_mapping(mapping, Path(out))
            print("  draft mapping written to %s" % written)
            print("  review it before running intake - the shapes are a guess")

    return {"mapping": mapping, "facts": facts, "written": written}


# ---------------------------------------------------------------- check

def check(source: str, mapping_path: str | Path,
          connectors: str | Path | None = None, device: str = "auto",
          out: str | Path | None = None, now: str | None = None,
          deep_links: bool = True, memory_fraction: float | None = None) -> dict[str, Any]:
    """Score the data against its mapping and say whether to proceed.

    Two parts: a hard gate (does the mapping match the data at all) and a
    readiness score out of 100 with a traffic light. Both read the source in
    bounded batches, so they are safe on any size of table.
    """
    mapping = load_mapping(mapping_path)
    src = resolve(source, _sources(connectors))
    budget, _engine = _setup(device, memory_fraction, None)

    with staged(src) as path:
        gate = _validate.validate_source(mapping, path)
        print("")
        print("  mapping gate   %s" % gate.summary().splitlines()[0])
        for line in gate.summary().splitlines()[1:]:
            print("  " + line)

        if not gate.ok:
            print("\n  the mapping does not match the data; fix the errors above first\n")
            return {"gate": gate, "readiness": None, "profiles": {}}

        if deep_links:
            links = _validate.check_links_resolve(mapping, path, room=budget.room())
            for line in links.summary().splitlines()[1:]:
                print("  " + line)
            gate.errors.extend(links.errors)
            gate.warnings.extend(links.warnings)

        readiness, profiles = quality.score(mapping, path, now=_now(now), room=budget.room())
        quality.print_readiness(readiness)

        written = None
        if out:
            written = quality.write_report(readiness, profiles, out, mapping)
            print("  quality report written to %s\n" % written)

    return {"gate": gate, "readiness": readiness, "profiles": profiles,
            "written": written}


# ---------------------------------------------------------------- run

def run(source: str, mapping_path: str | Path, out: str | Path,
        connectors: str | Path | None = None, device: str = "auto",
        now: str | None = None, skip_check: bool = False,
        memory_fraction: float | None = None, gpu_memory_fraction: float | None = None,
        work_dir: str | Path | None = None, resume: bool = False,
        keep_work: bool = False) -> Stats:
    """Transform a client's tables into events.parquet and profiles.parquet.

    ``memory_fraction`` and ``gpu_memory_fraction`` cap how much of the free
    RAM and GPU memory the run may plan to use (0.85 by default). ``resume``
    continues a run that stopped, reusing the tables it had already staged.
    """
    mapping = load_mapping(mapping_path)
    src = resolve(source, _sources(connectors))
    budget, engine = _setup(device, memory_fraction, gpu_memory_fraction)
    print("  source        %s" % src.uri)
    print("  mapping       %s" % mapping_path)
    print("  out           %s" % out)

    # a remote output is built in a scratch folder that is deleted afterwards,
    # so the work folder must live elsewhere or --resume could never find it,
    # and --keep-work would upload the staging along with the results
    if work_dir is None and is_remote(str(out)):
        work_dir = Path.cwd() / f"intake_work_{mapping.client}"
        print("  work          %s  (output is remote)" % work_dir)

    with staged(src) as path, staged_out(str(out)) as out_local:
        if not skip_check:
            gate = _validate.validate_source(mapping, path)
            if not gate.ok:
                print(gate.summary())
                raise SystemExit("mapping does not match the data; run 'check' first")

        clock = _now(now) if now else None
        stats = _run_transform(mapping, path, out_local, engine, now=clock, budget=budget,
                               work_dir=work_dir, resume=resume, keep_work=keep_work)

    print_stats(stats)
    return stats


def print_stats(stats: Stats) -> None:
    """What the transform did, table by table."""
    print("")
    print("  %-28s %12s %12s  %s" % ("table", "rows in", "events", "dropped"))
    for name, rec in stats.reconciliation.items():
        dropped = ", ".join(f"{k} {v:,}" for k, v in rec.get("dropped", {}).items() if v)
        print("  %-28s %12s %12s  %s"
              % (name[:28], format(rec["rows_in"], ","), format(rec.get("events", 0), ","),
                 dropped or "-"))
    print("")
    print("  events written      %12s" % format(stats.events_written, ","))
    print("  profiles written    %12s" % format(stats.profiles_written, ","))
    print("  partitions          %12d" % stats.partitions)
    print("  customer ranges     %12d" % stats.ranges)
    print("  steps on gpu / cpu  %8d / %d" % (stats.gpu_steps, stats.cpu_steps))
    print("  peak memory         %11.1f GB" % (stats.peak_rss / GIB))
    print("  elapsed             %12.1fs" % stats.seconds)
    lost = {n: r["unexplained"] for n, r in stats.reconciliation.items() if r["unexplained"]}
    if lost:
        print("  WARNING rows not accounted for: %s" % lost)
    print("")


# ---------------------------------------------------------------- report

def output_report(out: str | Path, rows_in: int = 0, device: str = "auto",
                  mapping_path: str | Path | None = None,
                  write: bool = True) -> _report.OutputReport:
    """Show the reconciliation and the tokenizer forecast for a finished run.

    A run writes its report as it goes; this reads it back. Output without
    one is re-counted from the files in bounded batches.
    """
    rep = _report.load_report(out)
    if rep is None:
        numeric: set[str] = set()
        if mapping_path:
            m = load_mapping(mapping_path)
            for t in m.active():
                numeric |= {t.fields.rename.get(c, c) for c in t.fields.numeric}
        print("  no report from the run beside these files; counting them now")
        rep = _report.report_from_files(out, numeric)
        rep.rows_in = rows_in
        if write:
            path = _report.write_report(rep, out)
            print("  output report written to %s\n" % path)
    _report.print_report(rep)
    return rep


# ---------------------------------------------------------------- one-shot

def pipeline(source: str, mapping_path: str | Path, out: str | Path,
             connectors: str | Path | None = None, device: str = "auto",
             now: str | None = None, memory_fraction: float | None = None,
             gpu_memory_fraction: float | None = None,
             work_dir: str | Path | None = None) -> dict[str, Any]:
    """check, run and report in one call - the usual onboarding sequence."""
    checked = check(source, mapping_path, connectors, device, out=out, now=now,
                    memory_fraction=memory_fraction)
    readiness = checked["readiness"]
    if readiness is None:
        raise SystemExit("the mapping does not match the data; fix the gate errors first")
    if readiness.band == "RED":
        raise SystemExit(
            f"readiness is RED ({readiness.score:.1f}/100); fix the data or the "
            f"mapping before transforming")

    stats = run(source, mapping_path, out, connectors, device, now=now, skip_check=True,
                memory_fraction=memory_fraction, gpu_memory_fraction=gpu_memory_fraction,
                work_dir=work_dir)
    rep = output_report(out, sum(stats.rows_read.values()), device, mapping_path)
    return {"readiness": readiness, "stats": stats, "report": rep}
