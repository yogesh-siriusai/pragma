"""Is this client's data good enough to train on, and will the mapping hold?

Two things live here. A profile, which describes every table and column, and a
readiness score, which turns that description into one number and a traffic
light so the conversation with a client is specific rather than vague.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from .engine import Engine
from .mapping import Mapping
from .shapes import entity_expr
from .tables import TableFiles, footer_stats, iter_batches, open_table
from .validate import count_missing_parents

# weight out of 100 for each check. These are configuration, not physics: a
# domain that is genuinely 60% null should not be marked down once that is
# understood and written here.
WEIGHTS = {
    "entity_resolution": 20,
    "time_validity": 20,
    "referential_integrity": 10,
    "uniqueness": 10,
    "type_conformance": 10,
    "signal_density": 10,
    "temporal_sanity": 10,
    "volume_adequacy": 10,
}

GREEN, AMBER = 85, 60

# a field that is almost always empty carries little, and a table of them
# produces thin events
SPARSE_AT = 0.95

# distinct counts are exact on tables up to this many rows. Above it they come
# from a sample, except the customer column, which is always counted exactly.
EXACT_DISTINCT_ROWS = 5_000_000
SAMPLE_ROWS = 1_000_000

# past this many distinct values a column's count stops being tracked: an id
# column has one value per row, and holding them all is not worth the memory
DISTINCT_CAP = 200_000


@dataclass
class ColumnProfile:
    """One column, measured."""

    name: str
    dtype: str
    nulls: int
    null_rate: float
    distinct: int | None = None
    min_value: str | None = None
    max_value: str | None = None
    # how the distinct count was made: "exact", "sample of N rows", or
    # "over N" when the column had more values than we were willing to hold
    distinct_basis: str = "exact"


@dataclass
class TableProfile:
    """One table, measured."""

    name: str
    rows: int
    columns: list[ColumnProfile] = field(default_factory=list)
    shape: str = ""
    note: str = ""
    # the distinct customer ids, kept for the link check rather than re-read
    entity_values: pl.Series | None = field(default=None, repr=False)

    def column(self, name: str) -> ColumnProfile | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None


@dataclass
class Check:
    """One of the eight readiness checks."""

    name: str
    weight: int
    scored: float
    findings: list[str] = field(default_factory=list)

    @property
    def lost(self) -> float:
        return self.weight - self.scored


@dataclass
class Readiness:
    """The verdict: a score, a band and exactly what cost points."""

    score: float
    band: str
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return self.band in ("GREEN", "AMBER")

    def findings(self) -> list[str]:
        out = []
        for c in self.checks:
            out.extend(c.findings)
        return out


# ---------------------------------------------------------------- profiling

def _batch_rows(table: TableFiles, room: int) -> int:
    """Rows per batch so one batch stays a small slice of the free room."""
    per_row = max(table.raw_bytes_per_row * 3, 64)
    return int(min(max(room * 0.1 / per_row, 16_384), 2_000_000))


def _text(value) -> str | None:
    return None if value is None else str(value)


def profile_table(table: TableFiles, name: str, entity: str | None = None,
                  room: int = 2 << 30, deep: bool = True) -> TableProfile:
    """Measure one table: row count, and per column nulls, distincts, range.

    Row counts, null counts and min/max come from the parquet footers when
    every row group carries them, which costs no data reading at all. What
    the footers do not say is counted from bounded batches.
    """
    schema = table.schema
    rows = table.rows
    stats = footer_stats(table)
    ranged = [c for c, dt in schema.items() if dt.is_temporal() or dt.is_numeric()]

    nulls = {c: stats[c].nulls for c in schema}
    lows = {c: _text(stats[c].min_value) for c in ranged}
    highs = {c: _text(stats[c].max_value) for c in ranged}
    need_nulls = [c for c in schema if nulls[c] is None]
    need_range = [c for c in ranged if lows[c] is None or highs[c] is None]

    # distinct counts: every column on a modest table, a sample on a big one
    exact = rows <= EXACT_DISTINCT_ROWS
    distinct_cols = list(schema) if deep else []
    seen: dict[str, pl.Series | None] = {c: None for c in distinct_cols}
    over: set[str] = set()
    batch_rows = _batch_rows(table, room)

    # one bounded pass for whatever the footers could not tell us
    if need_nulls or need_range or (distinct_cols and exact):
        computed_nulls = {c: 0 for c in need_nulls}
        computed_lo: dict[str, object] = {}
        computed_hi: dict[str, object] = {}
        cols = list(dict.fromkeys(need_nulls + need_range + (distinct_cols if exact else [])))
        for df, _ in iter_batches(table, cols, batch_rows):
            for c in need_nulls:
                computed_nulls[c] += df.get_column(c).null_count()
            for c in need_range:
                col = df.get_column(c)
                lo, hi = col.min(), col.max()
                if lo is not None:
                    computed_lo[c] = lo if c not in computed_lo else min(computed_lo[c], lo)
                    computed_hi[c] = hi if c not in computed_hi else max(computed_hi[c], hi)
            if exact:
                _track_distinct(df, distinct_cols, seen, over)
        nulls.update(computed_nulls)
        for c in need_range:
            lows[c] = _text(computed_lo.get(c))
            highs[c] = _text(computed_hi.get(c))

    # a big table: distinct values from a sample of its first rows
    looked = rows
    if distinct_cols and not exact:
        looked = 0
        for df, _ in iter_batches(table, distinct_cols, min(batch_rows, SAMPLE_ROWS)):
            _track_distinct(df, distinct_cols, seen, over)
            looked += df.height
            if looked >= SAMPLE_ROWS:
                break

    # the customer column is always counted exactly: uniqueness and the link
    # check depend on it, and it only holds as many values as there are customers
    entity_values = None
    if entity and entity in schema:
        acc = None
        for df, _ in iter_batches(table, [entity], batch_rows):
            ids = df.select(entity_expr(entity)).get_column("_id").drop_nulls().unique()
            acc = ids if acc is None else pl.concat([acc, ids]).unique()
        entity_values = acc if acc is not None else pl.Series("_id", [], dtype=pl.String)

    cols = []
    for c, dt in schema.items():
        n = int(nulls[c] or 0)
        distinct, basis = None, "exact"
        if c == entity and entity_values is not None:
            distinct = entity_values.len()
        elif c in over:
            basis = f"over {DISTINCT_CAP:,}"
        elif c in seen and seen[c] is not None:
            distinct = seen[c].len()
            basis = "exact" if exact else f"sample of {looked:,} rows"
        elif c in seen:
            distinct = 0
        cols.append(ColumnProfile(name=c, dtype=str(dt), nulls=n,
                                  null_rate=(n / rows) if rows else 0.0,
                                  distinct=distinct, min_value=lows.get(c),
                                  max_value=highs.get(c), distinct_basis=basis))

    return TableProfile(name=name, rows=rows, columns=cols, entity_values=entity_values)


def _track_distinct(df: pl.DataFrame, columns: list[str], seen: dict, over: set) -> None:
    """Fold one batch into the running distinct values, giving up past the cap."""
    for c in columns:
        if c in over:
            continue
        col = df.get_column(c)
        if col.dtype.is_nested() or col.dtype == pl.Object:
            over.add(c)
            continue
        col = col.drop_nulls().unique()
        acc = seen[c]
        acc = col if acc is None else pl.concat([acc, col]).unique()
        if acc.len() > DISTINCT_CAP:
            over.add(c)
            seen[c] = None
        else:
            seen[c] = acc


def profile_source(mapping: Mapping, source_dir: str | Path, engine: Engine | None = None,
                   deep: bool = True, room: int = 2 << 30) -> dict[str, TableProfile]:
    """Measure every mapped table."""
    source_dir = Path(source_dir)
    out: dict[str, TableProfile] = {}
    for tmap in mapping.active():
        table = open_table(source_dir, tmap.name)
        p = profile_table(table, tmap.name, entity=tmap.entity, room=room, deep=deep)
        p.shape = tmap.shape
        out[tmap.name] = p
    return out


# ---------------------------------------------------------------- the checks

def _check_entity(mapping: Mapping, profiles: dict[str, TableProfile]) -> Check:
    """Can every row be attributed to a customer?"""
    w = WEIGHTS["entity_resolution"]
    bad, total = 0, 0
    findings = []
    for tmap in mapping.active():
        if tmap.shape == "dimension":
            continue
        p = profiles.get(tmap.name)
        if not p or not p.rows:
            continue
        if tmap.shape == "child_of_event":
            key = tmap.resolve_entity_via["key"]
            col = p.column(key)
        else:
            col = p.column(tmap.entity) if tmap.entity else None
        if col is None:
            findings.append(f"{tmap.name}: no entity column to resolve against")
            bad += p.rows
        else:
            bad += col.nulls
            if col.nulls:
                findings.append(
                    f"{tmap.name}.{col.name}: {col.nulls:,} rows have no customer")
        total += p.rows
    rate = (bad / total) if total else 0.0
    return Check("entity_resolution", w, round(w * (1 - rate), 2), findings)


def _check_time(mapping: Mapping, profiles: dict[str, TableProfile]) -> Check:
    """Does every event row carry a usable timestamp?"""
    w = WEIGHTS["time_validity"]
    bad, total = 0, 0
    findings = []
    for tmap in mapping.active():
        if tmap.shape in ("dimension", "entity_master"):
            continue
        p = profiles.get(tmap.name)
        if not p or not p.rows:
            continue
        col = p.column(tmap.time) if tmap.time else None
        if col is None:
            findings.append(f"{tmap.name}: time column {tmap.time!r} is missing")
            bad += p.rows
        else:
            bad += col.nulls
            if col.nulls:
                findings.append(
                    f"{tmap.name}.{col.name}: {col.nulls:,} rows have no timestamp "
                    f"({100 * col.null_rate:.1f}%)")
        total += p.rows
    rate = (bad / total) if total else 0.0
    return Check("time_validity", w, round(w * (1 - rate), 2), findings)


def _check_links(mapping: Mapping, source_dir: Path, profiles: dict[str, TableProfile],
                 room: int) -> Check:
    """Do child tables and event customers resolve to a real customer?"""
    w = WEIGHTS["referential_integrity"]
    findings = []

    # the customers the master tables know, already collected while profiling
    known_parts = [profiles[t.name].entity_values for t in mapping.by_shape("entity_master")
                   if t.name in profiles and profiles[t.name].entity_values is not None]
    known = pl.concat(known_parts).unique() if known_parts else pl.Series([], dtype=pl.String)

    penalty = 0.0
    checked = 0
    for tmap in mapping.active():
        if tmap.shape in ("dimension", "entity_master"):
            continue
        if tmap.shape == "child_of_event":
            spec = tmap.resolve_entity_via
            pmap = mapping.tables[spec["parent"]]
            child = open_table(source_dir, tmap.name)
            parent = open_table(source_dir, pmap.name)
            miss, n = count_missing_parents(child, spec["key"], parent, spec["key"], room,
                                            _batch_rows(child, room))
            if miss:
                findings.append(
                    f"{tmap.name}: {miss:,} of {n:,} rows have no parent in {pmap.name}")
                penalty += miss / max(n, 1)
            checked += 1
        elif tmap.entity:
            p = profiles.get(tmap.name)
            if p is not None and p.entity_values is not None:
                uniq = p.entity_values
            else:
                uniq = pl.Series([], dtype=pl.String)
            orphan = int((~uniq.is_in(known.implode())).sum())
            if orphan:
                findings.append(
                    f"{tmap.name}: {orphan:,} customers are not in any master table")
                penalty += orphan / max(uniq.len(), 1)
            checked += 1

    rate = (penalty / checked) if checked else 0.0
    return Check("referential_integrity", w, round(w * (1 - min(rate, 1.0)), 2), findings)


def _check_unique(mapping: Mapping, profiles: dict[str, TableProfile]) -> Check:
    """One row per customer in a master table; no repeated snapshot subjects."""
    w = WEIGHTS["uniqueness"]
    findings = []
    penalty, checked = 0.0, 0
    for tmap in mapping.by_shape("entity_master"):
        p = profiles.get(tmap.name)
        if not p or not p.rows:
            continue
        col = p.column(tmap.entity)
        if col and col.distinct is not None:
            if tmap.latest_by:
                checked += 1  # several rows per customer is the point here
                continue
            dup = p.rows - col.distinct
            if dup > 0:
                findings.append(
                    f"{tmap.name}: {dup:,} duplicate rows on {tmap.entity} "
                    f"(a master table needs one row per customer)")
                penalty += dup / p.rows
            checked += 1
    rate = (penalty / checked) if checked else 0.0
    return Check("uniqueness", w, round(w * (1 - min(rate, 1.0)), 2), findings)


def _check_types(mapping: Mapping, profiles: dict[str, TableProfile]) -> Check:
    """Are the columns the mapping names actually present and sane?"""
    w = WEIGHTS["type_conformance"]
    findings = []
    missing, wanted = 0, 0
    for tmap in mapping.active():
        p = profiles.get(tmap.name)
        if not p:
            continue
        names = {c.name for c in p.columns}
        for col in tmap.fields.include + tmap.reserved_columns():
            wanted += 1
            if col not in names:
                missing += 1
                findings.append(f"{tmap.name}: mapping names {col!r}, which is not in the data")
    rate = (missing / wanted) if wanted else 0.0
    return Check("type_conformance", w, round(w * (1 - rate), 2), findings)


def _check_density(mapping: Mapping, profiles: dict[str, TableProfile]) -> Check:
    """How much of what we are about to emit is actually populated?"""
    w = WEIGHTS["signal_density"]
    findings = []
    rates = []
    for tmap in mapping.active():
        if tmap.shape == "dimension":
            continue
        p = profiles.get(tmap.name)
        if not p or not p.rows:
            continue
        cols = tmap.fields.resolve([c.name for c in p.columns], tmap.reserved_columns())
        sparse = []
        for name in cols:
            c = p.column(name)
            if c is None:
                continue
            rates.append(1 - c.null_rate)
            if c.null_rate >= SPARSE_AT:
                sparse.append(f"{name} {100 * c.null_rate:.0f}%")
        if sparse:
            findings.append(f"{tmap.name}: near-empty fields - " + ", ".join(sparse[:6])
                            + (f" (+{len(sparse) - 6} more)" if len(sparse) > 6 else ""))
    density = (sum(rates) / len(rates)) if rates else 1.0
    return Check("signal_density", w, round(w * density, 2), findings)


def _check_temporal(mapping: Mapping, profiles: dict[str, TableProfile],
                    now: datetime) -> Check:
    """Future-dated event times, and columns that look like commitments."""
    w = WEIGHTS["temporal_sanity"]
    findings = []
    offenders = 0
    looked = 0
    stamp = now.isoformat(sep=" ")
    for tmap in mapping.active():
        p = profiles.get(tmap.name)
        if not p:
            continue
        for c in p.columns:
            if c.max_value is None or "date" not in c.dtype.lower() and "time" not in c.dtype.lower():
                continue
            looked += 1
            if str(c.max_value) > stamp:
                offenders += 1
                role = "EVENT TIME" if c.name == tmap.time else "value"
                findings.append(
                    f"{tmap.name}.{c.name}: latest is {c.max_value} which is in the "
                    f"future ({role})")
    rate = (offenders / looked) if looked else 0.0
    return Check("temporal_sanity", w, round(w * (1 - rate), 2), findings)


def _check_volume(mapping: Mapping, profiles: dict[str, TableProfile]) -> Check:
    """Enough events per customer to be worth modelling?"""
    w = WEIGHTS["volume_adequacy"]
    findings = []

    customers = 0
    for tmap in mapping.by_shape("entity_master"):
        p = profiles.get(tmap.name)
        if p:
            col = p.column(tmap.entity)
            customers = max(customers, col.distinct or p.rows)

    events = sum(p.rows for n, p in profiles.items()
                 if mapping.tables[n].shape in ("event_stream", "periodic_snapshot",
                                                "child_of_event"))
    if not customers:
        return Check("volume_adequacy", w, 0.0, ["no customers found in any master table"])

    per = events / customers
    findings.append(f"{events:,} event rows over {customers:,} customers "
                    f"({per:,.0f} each on average)")
    # below ~20 events a customer there is very little for the model to learn
    scored = w if per >= 50 else w * max(per / 50.0, 0.0)
    return Check("volume_adequacy", w, round(scored, 2), findings)


# ---------------------------------------------------------------- the verdict

def score(mapping: Mapping, source_dir: str | Path, engine: Engine | None = None,
          profiles: dict[str, TableProfile] | None = None,
          now: datetime | None = None,
          room: int = 2 << 30) -> tuple[Readiness, dict[str, TableProfile]]:
    """Run all eight checks and return the verdict plus the profile it used.

    ``room`` is how much RAM the checks may plan around, in bytes.
    """
    source_dir = Path(source_dir)
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    profiles = profiles or profile_source(mapping, source_dir, engine, room=room)

    checks = [
        _check_entity(mapping, profiles),
        _check_time(mapping, profiles),
        _check_links(mapping, source_dir, profiles, room),
        _check_unique(mapping, profiles),
        _check_types(mapping, profiles),
        _check_density(mapping, profiles),
        _check_temporal(mapping, profiles, now),
        _check_volume(mapping, profiles),
    ]
    total = round(sum(c.scored for c in checks), 1)
    band = "GREEN" if total >= GREEN else ("AMBER" if total >= AMBER else "RED")
    return Readiness(score=total, band=band, checks=checks), profiles


def print_readiness(r: Readiness) -> None:
    """The score, then exactly what cost points."""
    print("")
    print("  readiness score   %.1f / 100   %s" % (r.score, r.band))
    print("  %-26s %6s %6s" % ("check", "got", "of"))
    for c in r.checks:
        flag = "  " if c.lost < 0.5 else " <"
        print("  %-26s %6.1f %6d%s" % (c.name, c.scored, c.weight, flag))
    findings = r.findings()
    if findings:
        print("")
        print("  what cost points")
        for f in findings:
            print("    - %s" % f)
    print("")


def write_report(readiness: Readiness, profiles: dict[str, TableProfile],
                 out_dir: str | Path, mapping: Mapping) -> Path:
    """Write the machine-readable quality report beside the run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "client": mapping.client,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "readiness": {
            "score": readiness.score,
            "band": readiness.band,
            "checks": [{"name": c.name, "weight": c.weight, "scored": c.scored,
                        "findings": c.findings} for c in readiness.checks],
        },
        "tables": {
            name: {
                "rows": p.rows,
                "shape": p.shape,
                "columns": [{"name": c.name, "dtype": c.dtype, "nulls": c.nulls,
                             "null_rate": round(c.null_rate, 4),
                             "distinct": c.distinct,
                             "distinct_basis": c.distinct_basis,
                             "min": c.min_value, "max": c.max_value}
                            for c in p.columns],
            } for name, p in profiles.items()
        },
    }
    path = out_dir / "quality_report.json"
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return path
