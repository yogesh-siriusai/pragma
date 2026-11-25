"""The hard gate: does this data match what the mapping claims about it?

Separate from quality.py on purpose. Quality scores how good the data is and
is advisory. This answers one yes/no question - will the transform run - and
it answers it before any GPU time is spent. Every message names the table, the
column and the fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from .customers import in_slice, key_expr, lookup_slices, parent_pairs
from .mapping import Mapping, MappingError
from .shapes import field_columns, output_keys
from .tables import TableFiles, iter_batches, open_table, scan_table


@dataclass
class Report:
    """Collected problems. ``ok`` is True only when there are no errors."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        head = "%s: %d error(s), %d warning(s)" % (
            "OK" if self.ok else "FAILED", len(self.errors), len(self.warnings))
        lines = [head]
        lines += ["  ERROR: " + e for e in self.errors]
        lines += ["  warn:  " + w for w in self.warnings]
        return "\n".join(lines)


def _column(schema: dict, table: str, col: str, role: str, rep: Report):
    """Fetch a column's dtype, recording an error when it is not there."""
    if col not in schema:
        near = [c for c in schema if col.lower() in c.lower() or c.lower() in col.lower()]
        hint = f" Did you mean {near[0]!r}?" if near else ""
        rep.error(f"{table}: {role} column {col!r} is not in the table.{hint}")
        return None
    return schema[col]


def _is_point_in_time(dtype) -> bool:
    """A date or a timestamp. A time of day or a duration is not a moment."""
    return dtype == pl.Date or isinstance(dtype, pl.Datetime)


def validate_source(mapping: Mapping, source_dir: str | Path) -> Report:
    """Check a client folder against its mapping. Only file footers are read."""
    source_dir = Path(source_dir)
    rep = Report()

    if not source_dir.exists():
        rep.error(f"source directory does not exist: {source_dir}")
        return rep

    schemas: dict[str, dict] = {}

    # ---- every mapped table must be present and readable
    for tmap in mapping.active():
        try:
            lf = scan_table(source_dir, tmap.name)
            schemas[tmap.name] = dict(lf.collect_schema())
        except FileNotFoundError:
            rep.error(f"{tmap.name}: no parquet found under {source_dir}")
        except Exception as exc:  # noqa: BLE001 - a bad file is a user problem
            rep.error(f"{tmap.name}: cannot read ({type(exc).__name__}: {exc})")

    # ---- the columns the mapping leans on must exist and have sane types
    for tmap in mapping.active():
        schema = schemas.get(tmap.name)
        if schema is None:
            continue

        if tmap.entity:
            _column(schema, tmap.name, tmap.entity, "entity", rep)

        if tmap.time:
            dt = _column(schema, tmap.name, tmap.time, "time", rep)
            if dt is not None and not _is_point_in_time(dt):
                rep.error(f"{tmap.name}: time column {tmap.time!r} is {dt}, "
                          f"which is not a date or timestamp")

        if tmap.latest_by:
            dt = _column(schema, tmap.name, tmap.latest_by, "latest_by", rep)
            if dt is not None and not _is_point_in_time(dt):
                rep.error(f"{tmap.name}: latest_by column {tmap.latest_by!r} is {dt}, "
                          f"which is not a date or timestamp")
        if tmap.as_of:
            dt = _column(schema, tmap.name, tmap.as_of, "as_of", rep)
            if dt is not None and not _is_point_in_time(dt):
                rep.error(f"{tmap.name}: as_of column {tmap.as_of!r} is {dt}, "
                          f"which is not a date or timestamp")
        if tmap.snapshot_of:
            _column(schema, tmap.name, tmap.snapshot_of, "snapshot_of", rep)

        for col in tmap.lifelong:
            dt = _column(schema, tmap.name, col, "lifelong", rep)
            if dt is not None and not _is_point_in_time(dt):
                rep.error(f"{tmap.name}: lifelong column {col!r} is {dt}; a milestone "
                          f"needs a date so the model knows when it happened")

        # named-but-absent fields are a warning: the mapping may be shared
        # across clients whose tables differ slightly
        for col in tmap.fields.include:
            if col not in schema:
                rep.warn(f"{tmap.name}: field {col!r} is named in the mapping but not "
                         f"in the data; it will be skipped")

        if tmap.shape == "child_of_event":
            key = tmap.resolve_entity_via.get("key")
            if key:
                _column(schema, tmap.name, key, "link", rep)
            parent = mapping.tables.get(tmap.resolve_entity_via.get("parent", ""))
            pschema = schemas.get(parent.name) if parent else None
            if parent and pschema is not None:
                if key and key not in pschema:
                    rep.error(f"{parent.name}: link key {key!r} is not in the parent, "
                              f"so {tmap.name} cannot find a customer")

        if tmap.shape == "dimension" and tmap.action == "enrich":
            _column(schema, tmap.name, tmap.enrich_on, "enrich_on", rep)
            target = schemas.get(tmap.enrich_into)
            if target is not None and tmap.enrich_on not in target:
                rep.error(f"{tmap.enrich_into}: enrich key {tmap.enrich_on!r} is not in the "
                          f"table {tmap.name} is meant to enrich")

    # ---- what each table's fields will look like
    for tmap in mapping.active():
        schema = schemas.get(tmap.name)
        if schema is None or tmap.shape == "dimension":
            continue
        cols, skipped = field_columns(tmap, schema)
        if skipped:
            rep.warn(f"{tmap.name}: {', '.join(skipped)} are nested or binary and cannot be "
                     f"a text value; they will be skipped")
        if not cols and tmap.shape != "entity_master":
            rep.warn(f"{tmap.name}: no columns left after include/drop; its events "
                     f"will carry only a marker field")
        keys = output_keys(tmap, cols)
        clash = sorted({k for k in keys if keys.count(k) > 1})
        if clash:
            rep.error(f"{tmap.name}: two columns would both become field {clash[0]!r}; "
                      f"check the rename map")

    return rep


# ---------------------------------------------------------------- reading the links

def count_missing_parents(child: TableFiles, child_key: str, parent: TableFiles,
                          parent_key: str, room: int, batch_rows: int) -> tuple[int, int]:
    """(child rows with no parent, child rows) - in bounded memory.

    Holds the parent's distinct keys, one hash slice at a time when they are
    many, and streams the child's key column past them.
    """
    n_slices = lookup_slices(parent, room)
    missing, total = 0, 0
    for index in range(n_slices):
        keys = []
        for df, _ in iter_batches(parent, [parent_key], batch_rows):
            keys.append(df.select(key_expr(parent_key)).filter(
                pl.col("_key").is_not_null() & in_slice(n_slices, index)).unique())
        have = pl.concat(keys).unique().get_column("_key") if keys else \
            pl.Series("_key", [], dtype=pl.String)

        for df, _ in iter_batches(child, [child_key], batch_rows):
            k = df.select(key_expr(child_key))
            if index == 0:
                total += k.height
                # a row with no key at all has no parent, and is counted once
                missing += k.get_column("_key").null_count()
            k = k.filter(pl.col("_key").is_not_null() & in_slice(n_slices, index))
            missing += int((~k.get_column("_key").is_in(have.implode())).sum())
    return missing, total


def check_links_resolve(mapping: Mapping, source_dir: str | Path, room: int = 2 << 30,
                        batch_rows: int = 262_144) -> Report:
    """Actually resolve every child row, which the schema check cannot do.

    Also proves each link key names one customer, and each dimension key one
    row. Reads only the key columns, in bounded batches, so it stays cheap and
    safe on any size of table.
    """
    source_dir = Path(source_dir)
    rep = Report()

    for tmap in mapping.by_shape("child_of_event"):
        spec = tmap.resolve_entity_via
        pmap = mapping.tables[spec["parent"]]
        key = spec["key"]
        child = open_table(source_dir, tmap.name)
        parent = open_table(source_dir, pmap.name)

        missing, n_child = count_missing_parents(child, key, parent, key, room, batch_rows)
        if missing:
            rate = 100 * missing / max(n_child, 1)
            msg = (f"{tmap.name}: {missing:,} of {n_child:,} rows ({rate:.1f}%) have no "
                   f"match in {pmap.name}.{key}, so they would lose their customer")
            if rate > 5:
                rep.error(msg)
            else:
                rep.warn(msg)

        # a key shared by two customers would make the child's customer a guess
        try:
            n_slices = lookup_slices(parent, room)
            for index in range(n_slices):
                parent_pairs(parent, pmap, key, batch_rows, n_slices, index)
        except MappingError as exc:
            rep.error(str(exc))

    for tmap in mapping.by_shape("dimension"):
        if tmap.action != "enrich":
            continue
        dim = open_table(source_dir, tmap.name)
        seen = []
        for df, _ in iter_batches(dim, [tmap.enrich_on], batch_rows):
            seen.append(df.get_column(tmap.enrich_on).drop_nulls())
        keys = pl.concat(seen) if seen else pl.Series([], dtype=pl.String)
        dupes = int(keys.is_duplicated().sum())
        if dupes:
            rep.error(f"{tmap.name}: enrich key {tmap.enrich_on!r} repeats on {dupes:,} rows; "
                      f"each fact row would turn into several events")

    return rep
