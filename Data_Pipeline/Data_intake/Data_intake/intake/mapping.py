"""The mapping file: every decision that is specific to one client.

One YAML per client. Nothing else in this package knows a client's column
names, so onboarding a new bank is a new mapping, not a new branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# the five shapes from the design document
SHAPES = ("entity_master", "event_stream", "periodic_snapshot",
          "child_of_event", "dimension")

# every shape needs a different minimum set of answers
REQUIRED = {
    "entity_master":     ("entity",),
    "event_stream":      ("entity", "time", "source_name"),
    "periodic_snapshot": ("entity", "time", "source_name"),
    "child_of_event":    ("time", "source_name", "resolve_entity_via"),
    "dimension":         (),
}


class MappingError(ValueError):
    """A mapping that cannot be used, with the file and key that is wrong."""


@dataclass
class FieldPolicy:
    """Which columns of a table become model-visible fields, and how."""

    include: list[str] = field(default_factory=list)   # empty means "all that survive drop"
    drop: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)   # keep full precision, mark for binning
    rename: dict[str, str] = field(default_factory=dict)

    def resolve(self, available: list[str], reserved: list[str]) -> list[str]:
        """The actual column list to melt, given what the table really has."""
        if self.include:
            cols = [c for c in self.include if c in available]
        else:
            cols = [c for c in available if c not in self.drop]
        # the key and time columns are carried separately, never as fields, and
        # a leading underscore marks a column this package added for its own use
        return [c for c in cols
                if c not in reserved and c not in self.drop and not c.startswith("_")]


@dataclass
class TableMap:
    """How one client table becomes events or profile rows."""

    name: str
    shape: str
    entity: str | None = None
    time: str | None = None
    source_name: str | None = None
    fields: FieldPolicy = field(default_factory=FieldPolicy)
    skip: bool = False

    # periodic_snapshot
    emit: str = "changes_only"          # or "all"
    # what the snapshot is actually about. A customer snapshot is about the
    # customer, but a product snapshot is about one of several products they
    # hold, and comparing those against each other would invent changes that
    # never happened. Defaults to the entity column.
    snapshot_of: str | None = None

    # child_of_event
    resolve_entity_via: dict[str, str] = field(default_factory=dict)

    # entity_master
    as_of: str | None = None
    lifelong: list[str] = field(default_factory=list)
    # when a "master" table actually has one row per customer per date, take
    # the most recent row. This is how a state snapshot becomes current profile
    # context instead of being forced into an event source it does not fit.
    latest_by: str | None = None

    # dimension
    action: str = "drop"                # or "enrich"
    enrich_into: str | None = None
    enrich_on: str | None = None

    def reserved_columns(self) -> list[str]:
        """Columns carried as structure, so never melted into fields."""
        out = [c for c in (self.entity, self.time, self.as_of, self.latest_by,
                           self.snapshot_of) if c]
        out += list(self.lifelong)
        if self.resolve_entity_via:
            key = self.resolve_entity_via.get("key")
            if key:
                out.append(key)
        return out


@dataclass
class Mapping:
    """A whole client: the tables, plus the handful of run-wide settings."""

    client: str
    tables: dict[str, TableMap]
    source: str = ""
    # how a naive (zone-less) timestamp in this client's data should be read
    timezone: str = "UTC"
    # a row with no customer or no time is always dropped - the contract has
    # no place for it. A future-dated row is a choice, on by default.
    drop_future_events: bool = True

    def active(self) -> list[TableMap]:
        """Tables we will actually read, in a stable order."""
        return [t for t in self.tables.values() if not t.skip]

    def by_shape(self, shape: str) -> list[TableMap]:
        return [t for t in self.active() if t.shape == shape]


def _field_policy(raw: dict[str, Any] | None) -> FieldPolicy:
    raw = raw or {}
    return FieldPolicy(include=list(raw.get("include", [])),
                       drop=list(raw.get("drop", [])),
                       numeric=list(raw.get("numeric", [])),
                       rename=dict(raw.get("rename", {})))


def _table(name: str, raw: dict[str, Any]) -> TableMap:
    shape = raw.get("shape")
    if shape not in SHAPES:
        raise MappingError(
            f"table {name!r}: shape {shape!r} is not one of {', '.join(SHAPES)}")

    t = TableMap(name=name,
                 shape=shape,
                 entity=raw.get("entity"),
                 time=raw.get("time"),
                 source_name=raw.get("source_name"),
                 fields=_field_policy(raw.get("fields")),
                 skip=bool(raw.get("skip", False)),
                 emit=raw.get("emit", "changes_only"),
                 snapshot_of=raw.get("snapshot_of"),
                 resolve_entity_via=dict(raw.get("resolve_entity_via", {})),
                 as_of=raw.get("as_of"),
                 lifelong=list(raw.get("lifelong", [])),
                 latest_by=raw.get("latest_by"),
                 action=raw.get("action", "drop"),
                 enrich_into=raw.get("enrich_into"),
                 enrich_on=raw.get("enrich_on"))

    # check the answers this shape cannot work without
    for need in REQUIRED[shape]:
        if not getattr(t, need):
            raise MappingError(f"table {name!r} (shape {shape}): '{need}' is required")

    if t.shape == "periodic_snapshot" and t.emit not in ("all", "changes_only"):
        raise MappingError(f"table {name!r}: emit must be 'all' or 'changes_only'")

    if t.shape == "child_of_event":
        for need in ("parent", "key"):
            if need not in t.resolve_entity_via:
                raise MappingError(
                    f"table {name!r}: resolve_entity_via needs '{need}'")

    if t.shape == "dimension" and t.action == "enrich":
        if not (t.enrich_into and t.enrich_on):
            raise MappingError(
                f"table {name!r}: action 'enrich' needs enrich_into and enrich_on")

    return t


def load_mapping(path: str | Path) -> Mapping:
    """Read and check a mapping file. Raises MappingError with the bad key."""
    path = Path(path)
    if not path.exists():
        raise MappingError(f"mapping file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tables_raw = raw.get("tables") or {}
    if not tables_raw:
        raise MappingError(f"{path}: no tables declared")

    tables = {name: _table(name, spec or {}) for name, spec in tables_raw.items()}
    # older mappings carried a bucket count; partitioning is now sized from
    # the machine at run time, so that setting is accepted and ignored
    opts = raw.get("options") or {}

    m = Mapping(client=raw.get("client", path.stem),
                source=raw.get("source", ""),
                tables=tables,
                timezone=opts.get("timezone", "UTC"),
                drop_future_events=bool(opts.get("drop_future_events", True)))

    _check_cross_table(m, path)
    return m


def _check_cross_table(m: Mapping, path: Path) -> None:
    """Links and enrichments must point at tables that exist and are not skipped."""
    for t in m.active():
        if t.shape == "child_of_event":
            parent = t.resolve_entity_via["parent"]
            if parent not in m.tables:
                raise MappingError(
                    f"{path}: table {t.name!r} links to {parent!r}, which is not mapped")
            if not m.tables[parent].entity:
                raise MappingError(
                    f"{path}: parent {parent!r} has no entity column to borrow")
            if m.tables[parent].skip:
                raise MappingError(
                    f"{path}: table {t.name!r} borrows its customer from {parent!r}, "
                    f"which is marked skip; un-skip the parent or skip {t.name!r} too")
        if t.shape == "dimension" and t.action == "enrich":
            if t.enrich_into not in m.tables:
                raise MappingError(
                    f"{path}: {t.name!r} enriches {t.enrich_into!r}, which is not mapped")
            if m.tables[t.enrich_into].skip:
                raise MappingError(
                    f"{path}: {t.name!r} enriches {t.enrich_into!r}, which is marked skip")

    # at least one entity_master, or we do not know where profiles come from.
    # several are allowed: a base master plus state tables merged by latest row.
    masters = m.by_shape("entity_master")
    if not masters:
        raise MappingError(f"{path}: no entity_master table, so no profiles can be built")

    # a source name the contract does not know will fail fluxtra validate later,
    # so catch it here where the error can name the table and the fix
    allowed = _contract_sources()
    if allowed:
        for t in m.active():
            if t.source_name and t.source_name not in allowed:
                raise MappingError(
                    f"{path}: table {t.name!r} uses source {t.source_name!r}, which is not in "
                    f"the fluxtra contract {sorted(allowed)}. Either map it onto one of those, "
                    f"or add the name to SOURCES in fluxtra/core/schema.py.")


def _contract_sources() -> set[str]:
    """The event source names fluxtra will accept, if fluxtra is importable."""
    try:
        from fluxtra.core.schema import SOURCES
    except ImportError:
        return set()
    return set(SOURCES)


def write_mapping(m: Mapping, path: str | Path) -> Path:
    """Write a mapping back out as YAML, for the draft that discover() produces."""
    path = Path(path)
    out: dict[str, Any] = {"client": m.client, "source": m.source,
                           "options": {"timezone": m.timezone,
                                       "drop_future_events": m.drop_future_events},
                           "tables": {}}
    for name, t in m.tables.items():
        spec: dict[str, Any] = {"shape": t.shape}
        for key, val in (("entity", t.entity), ("time", t.time),
                         ("source_name", t.source_name), ("as_of", t.as_of),
                         ("latest_by", t.latest_by)):
            if val:
                spec[key] = val
        if t.skip:
            spec["skip"] = True
        if t.shape == "periodic_snapshot":
            spec["emit"] = t.emit
            if t.snapshot_of:
                spec["snapshot_of"] = t.snapshot_of
        if t.resolve_entity_via:
            spec["resolve_entity_via"] = t.resolve_entity_via
        if t.lifelong:
            spec["lifelong"] = t.lifelong
        if t.shape == "dimension":
            spec["action"] = t.action
        policy = {k: v for k, v in (("include", t.fields.include),
                                    ("drop", t.fields.drop),
                                    ("numeric", t.fields.numeric)) if v}
        if policy:
            spec["fields"] = policy
        out["tables"][name] = spec

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(out, sort_keys=False, width=100), encoding="utf-8")
    return path
