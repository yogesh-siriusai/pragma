"""Look at a folder of client tables and guess how they fit together.

The output is a draft mapping, not an answer. It gets the boring 80% right -
which column is the customer, which is the time, what shape each table is -
so a human can spend their attention on the decisions that actually matter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .mapping import FieldPolicy, Mapping, TableMap
from .tables import distinct_values, open_table

log = logging.getLogger(__name__)

# rows per read while counting customers; small enough for any machine
DISCOVER_BATCH_ROWS = 262_144

# column names that usually mean "this is the customer"
ENTITY_PATTERNS = (r"^customer_?id$", r"^cust_?id$", r"^party_?id$", r"^client_?id$",
                   r"^user_?id$", r"^cif$", r"^cin$",
                   r"^customer$", r"^cust$", r"^party$", r"^client$", r"^user$")

# a column name that usually means "this row is a periodic observation"
SNAPSHOT_PATTERNS = (r"snapshot", r"as_?of", r"_dt$", r"report_date")

# a column name that usually means "this is when the thing happened". Telling a
# master table from an event table by row counts alone fails on small tables,
# so the name gets a vote too. Deliberately narrow: a profile's acc_cre_ts or
# relationship_open_date must NOT match, or every master becomes an event.
EVENT_TIME_PATTERNS = (r"_timestamp$", r"_at$", r"^event_time", r"occurred",
                       r"happened", r"_start$", r"record_available", r"posted",
                       r"settle")

# identifier-ish names we should never melt into fields: high cardinality and
# no behaviour in them
ID_PATTERNS = (r"_id$", r"^id$", r"_hash$", r"_masked$", r"_number$", r"_reference$")

# the four source names fluxtra's contract accepts, and the words that hint
# at each. Anything unmatched falls back to "app" with a comment for review.
SOURCE_HINTS = {
    "transaction": ("txn", "transaction", "payment", "card", "balance", "product_snapshot"),
    "app": ("session", "interaction", "event", "login", "click", "journey", "app"),
    "trading": ("trade", "trading", "order", "instrument", "portfolio"),
    "communication": ("comm", "message", "notification", "contact", "collection",
                      "fraud", "case", "alert", "complaint"),
}


@dataclass
class TableFacts:
    """What a quick scan can tell us about one table."""

    name: str
    rows: int
    columns: dict[str, str]
    entity: str | None
    times: list[str]
    snapshot_col: str | None
    distinct_entity: int | None
    id_columns: list[str]
    event_times: list[str]


def _matches(name: str, patterns) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in patterns)


def _guess_source(table: str) -> str:
    low = table.lower()
    for source, hints in SOURCE_HINTS.items():
        if any(h in low for h in hints):
            return source
    return "app"


def scan_facts(source_dir: Path, name: str) -> TableFacts:
    """Cheap pass over one table: schema, row count and a few counts.

    The row count comes from the file footers. The only data read is the
    customer column, in bounded batches, to count distinct customers.
    """
    table = open_table(source_dir, name)
    schema = table.schema

    entity = next((c for c in schema if _matches(c, ENTITY_PATTERNS)), None)
    # a time of day or a duration is temporal but is not a moment in time
    times = [c for c, dt in schema.items() if dt == pl.Date or isinstance(dt, pl.Datetime)]
    snapshot_col = next((c for c in times if _matches(c, SNAPSHOT_PATTERNS)), None)
    event_times = [c for c in times if _matches(c, EVENT_TIME_PATTERNS)]
    ids = [c for c in schema if _matches(c, ID_PATTERNS)]

    rows = table.rows
    distinct = None
    if entity:
        values, _ = distinct_values(table, entity, DISCOVER_BATCH_ROWS)
        distinct = values.len()

    return TableFacts(name=name, rows=rows,
                      columns={c: str(dt) for c, dt in schema.items()},
                      entity=entity, times=times, snapshot_col=snapshot_col,
                      distinct_entity=distinct, id_columns=ids,
                      event_times=event_times)


def classify(facts: TableFacts) -> str:
    """Pick one of the five shapes from what the scan found."""
    # no customer and no time: a lookup table
    if not facts.entity and not facts.times:
        return "dimension"

    # time but no customer: it must borrow one from a parent
    if not facts.entity and facts.times:
        return "child_of_event"

    # one row per customer and nothing periodic about it. A column that is
    # plainly an event time overrules the row count, because on a small table
    # "one row each" happens by accident.
    one_each = facts.distinct_entity and facts.distinct_entity == facts.rows
    if one_each and not facts.snapshot_col and not facts.event_times:
        return "entity_master"

    # repeats the same customer on a schedule
    if facts.snapshot_col:
        return "periodic_snapshot"

    if facts.times:
        return "event_stream"

    # a customer, no time, several rows each: treat as a master and let the
    # reviewer decide, rather than silently inventing timestamps
    return "entity_master"


def _find_parent(facts: TableFacts, others: dict[str, TableFacts]) -> dict[str, str]:
    """Which table owns the key this child table carries?"""
    for key in facts.id_columns:
        for other in others.values():
            if other.name == facts.name or not other.entity:
                continue
            # the parent is the table where this key is unique and a customer
            # sits beside it
            if key in other.columns:
                return {"parent": other.name, "key": key}
    return {}


def _choose_time(facts: TableFacts) -> str | None:
    """Pick an event time, preferring when the bank knew over when it happened."""
    if not facts.times:
        return None
    # a "record available" or "posted" column is the point-in-time-correct
    # choice; see open decision 1 in the design document
    for want in ("record_available", "available", "posted", "booking"):
        for c in facts.times:
            if want in c.lower():
                return c
    if facts.snapshot_col:
        return facts.snapshot_col
    return facts.times[0]


def draft(source_dir: str | Path, client: str = "") -> tuple[Mapping, dict[str, TableFacts]]:
    """Scan a folder of tables and propose a mapping for review."""
    source_dir = Path(source_dir)
    names = sorted(p.name for p in source_dir.iterdir()
                   if p.is_dir() and any(p.glob("*.parquet")))
    names += sorted(p.stem for p in source_dir.glob("*.parquet"))
    if not names:
        raise FileNotFoundError(f"no parquet tables under {source_dir}")

    facts = {n: scan_facts(source_dir, n) for n in names}

    tables: dict[str, TableMap] = {}
    for name, f in facts.items():
        shape = classify(f)
        drop = [c for c in f.id_columns if c != f.entity]
        tmap = TableMap(name=name, shape=shape, entity=f.entity,
                        fields=FieldPolicy(drop=drop))

        if shape in ("event_stream", "periodic_snapshot"):
            tmap.time = _choose_time(f)
            tmap.source_name = _guess_source(name)
        if shape == "periodic_snapshot":
            tmap.emit = "changes_only"
            # if the table carries another id beside the customer, the snapshot
            # is probably about that thing, not about the customer
            subject = next((c for c in f.id_columns
                            if c != f.entity and c in f.columns), None)
            tmap.snapshot_of = subject
        if shape == "child_of_event":
            tmap.time = _choose_time(f)
            tmap.source_name = _guess_source(name)
            tmap.resolve_entity_via = _find_parent(f, facts)
        if shape == "entity_master":
            tmap.lifelong = [c for c in f.times if c != f.snapshot_col][:6]
            if f.snapshot_col:
                tmap.latest_by = f.snapshot_col
                tmap.as_of = f.snapshot_col
        if shape == "dimension":
            tmap.action = "drop"

        tables[name] = tmap

    mapping = Mapping(client=client or source_dir.name, tables=tables,
                      source=str(source_dir))
    return mapping, facts


def print_draft(mapping: Mapping, facts: dict[str, TableFacts]) -> None:
    """Show what we think each table is, so a reviewer can disagree quickly."""
    print("")
    print("  %-28s %10s %-18s %-16s %s"
          % ("table", "rows", "shape", "entity", "time"))
    for name, t in mapping.tables.items():
        f = facts[name]
        print("  %-28s %10s %-18s %-16s %s"
              % (name[:28], format(f.rows, ","), t.shape,
                 (t.entity or "-")[:16], t.time or "-"))
    print("")

    unresolved = [t.name for t in mapping.active()
                  if t.shape == "child_of_event" and not t.resolve_entity_via]
    if unresolved:
        print("  could not find a parent for: %s" % ", ".join(unresolved))
        print("  set resolve_entity_via by hand for these")
        print("")
