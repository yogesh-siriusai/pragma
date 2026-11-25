"""The rules that turn one table's rows into model-ready values.

Two halves, matching the two passes of the transform:

  pass 1 (staging)  clean one batch of a source table: which columns become
                    fields, how a value is written as text, how a timestamp
                    becomes UTC. Done once, on the CPU, so a value is written
                    the same way whichever engine runs pass 2.

  pass 2 (events)   take the staged rows of one customer range and melt them
                    into (customer, time, event, key, value) rows, apply the
                    snapshot "changes only" rule, and build profile pieces.
"""

from __future__ import annotations

import polars as pl

from .mapping import TableMap

# event_uid is (table number << SHIFT) + source row number. Ids never collide
# across tables, and since the row number comes from the source file, an id
# does not change with batch sizes, partition counts or the machine.
UID_SHIFT = 44
MAX_ROWS_PER_TABLE = (1 << UID_SHIFT) - 1

# the key a table with no usable fields emits, so its events keep their timing
MARKER_KEY = "_event"
MARKER_VALUE = "1"

# column types that cannot be written as one clean text value
_UNSUPPORTED = (pl.List, pl.Array, pl.Struct, pl.Object, pl.Binary)


# ---------------------------------------------------------------- choosing fields

def field_columns(tmap: TableMap, schema: dict) -> tuple[list[str], list[str]]:
    """(columns that become fields, columns skipped because of their type)."""
    wanted = tmap.fields.resolve(list(schema), tmap.reserved_columns())
    keep, skipped = [], []
    for c in wanted:
        if isinstance(schema[c], _UNSUPPORTED) or schema[c] in _UNSUPPORTED:
            skipped.append(c)
        else:
            keep.append(c)
    return keep, skipped


def output_keys(tmap: TableMap, columns: list[str]) -> list[str]:
    """The names the model will see for these columns."""
    return [tmap.fields.rename.get(c, c) for c in columns]


# ---------------------------------------------------------------- pass 1 expressions

def value_expr(col: str, dtype, alias: str) -> pl.Expr:
    """One column as a clean string.

    A full timestamp used as a *value* is unique per row, which would flood the
    tokenizer's text vocabulary, so datetimes are coarsened to the day. Dates,
    decimals, ints and bools all stringify cleanly as they are.
    """
    e = pl.col(col)
    if isinstance(dtype, pl.Datetime):
        e = e.dt.date()
    return e.cast(pl.String).alias(alias)


def time_expr(col: str, dtype, timezone: str, alias: str) -> pl.Expr:
    """An event or milestone time as a naive UTC timestamp in microseconds.

    A zone-aware column is converted to UTC. A naive one is read as the
    mapping's timezone first. A local time that does not exist (the hour a
    clock skips in spring) becomes null and is dropped as "no timestamp",
    rather than being shifted to a time that never happened.
    """
    e = pl.col(col)
    if dtype == pl.Date:
        return e.cast(pl.Datetime("us")).alias(alias)
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            e = e.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        elif timezone and timezone.upper() != "UTC":
            e = (e.dt.replace_time_zone(timezone, ambiguous="earliest", non_existent="null")
                  .dt.convert_time_zone("UTC").dt.replace_time_zone(None))
        return e.cast(pl.Datetime("us")).alias(alias)
    return e.cast(pl.Datetime("us")).alias(alias)


def entity_expr(col: str, alias: str = "_id") -> pl.Expr:
    """A customer id as text, with an empty string treated as missing."""
    s = pl.col(col).cast(pl.String)
    return pl.when(s == "").then(None).otherwise(s).alias(alias)


# ---------------------------------------------------------------- pass 2: events

def melt_table(lf: pl.LazyFrame, staged: list[str], key_ids: list[int],
               key_dtype, subject: bool) -> pl.LazyFrame:
    """Staged wide rows to long (code, ts, uid, key_id, value) rows.

    Written as one select per field, appended, rather than unpivot: on the GPU
    an unpivot is forced into a single partition, while appended selects run
    as ordinary column work. The key travels as a small integer so a billion
    rows do not each carry the key's name as text.
    """
    index = ["code", "ts", "uid"] + (["_subject"] if subject else [])

    if not staged:
        # a table with no usable fields still contributes its timing, so we
        # keep one marker row per event rather than losing the event entirely
        return lf.select(index + [pl.lit(key_ids[0], dtype=key_dtype).alias("key_id"),
                                  pl.lit(MARKER_VALUE, dtype=pl.String).alias("value")])

    parts = []
    for col, kid in zip(staged, key_ids):
        parts.append(lf.select(index + [pl.lit(kid, dtype=key_dtype).alias("key_id"),
                                        pl.col(col).alias("value")])
                       .filter(pl.col("value").is_not_null()))
    return pl.concat(parts, how="vertical")


def changes_only(long: pl.LazyFrame) -> pl.LazyFrame:
    """Keep a snapshot field only when it differs from its previous value.

    "Previous" means the same customer, the same subject and the same field,
    earlier in time - nulls were already removed, so it is the previous
    non-null value. The first value is always kept, so the opening state
    survives. The customer is part of the subject on purpose: an account that
    moves to a new customer is new to that customer, and they see its state.

    Ties on time are broken by the source row, so the result is the same on
    every run and every engine.
    """
    ordered = long.sort(["code", "_subject", "key_id", "ts", "uid"])

    # same group as the row above? Only plain column comparisons and a shift,
    # so the GPU can run it. The very first row compares against nothing; a
    # null never equals anything, so it can never count as "same" and is kept.
    subject = pl.col("_subject")
    same_subject = ((subject == subject.shift(1)).fill_null(False)
                    | (subject.is_null() & subject.shift(1).is_null()))
    same_group = ((pl.col("code") == pl.col("code").shift(1)).fill_null(False)
                  & same_subject
                  & (pl.col("key_id") == pl.col("key_id").shift(1)).fill_null(False))
    same_value = (pl.col("value") == pl.col("value").shift(1)).fill_null(False)

    return ordered.filter(~same_group | ~same_value).drop("_subject")


def sort_events(long: pl.LazyFrame) -> pl.LazyFrame:
    """The order the tokenizer needs.

    Customers in id order (codes are assigned in sorted id order), each
    customer's events in time order, ties broken by event id, and an event's
    fields in key order. This is a total order, so the output never depends
    on how a sort treats ties.
    """
    return long.sort(["code", "ts", "uid", "key_id"])


# ---------------------------------------------------------------- pass 2: profiles

def master_latest(lf: pl.LazyFrame, tmap: TableMap) -> pl.LazyFrame:
    """For a master with one row per customer per date, keep the newest row.

    Two rows sharing the newest date are decided by source row - the later
    row in the file wins - so the choice is the same on every run.
    """
    if not tmap.latest_by:
        return lf
    return (lf.sort(["code", "_latest", "_row"], nulls_last=False)
              .unique(subset=["code"], keep="last", maintain_order=True))


def master_attributes(lf: pl.LazyFrame, staged: list[str], keys: list[str],
                      master_ord: int) -> pl.LazyFrame:
    """Wide master rows to (code, master, row, key, value), nulls removed."""
    if not staged:
        return pl.LazyFrame(schema={"code": pl.UInt32, "_master": pl.Int32, "_row": pl.Int64,
                                    "key": pl.String, "value": pl.String})
    parts = []
    for col, key in zip(staged, keys):
        parts.append(lf.select(pl.col("code"),
                               pl.lit(master_ord, dtype=pl.Int32).alias("_master"),
                               pl.col("_row"),
                               pl.lit(key, dtype=pl.String).alias("key"),
                               pl.col(col).alias("value"))
                       .filter(pl.col("value").is_not_null()))
    return pl.concat(parts, how="vertical")


def master_lifelong(lf: pl.LazyFrame, staged: list[str], keys: list[str]) -> pl.LazyFrame:
    """Milestone columns to (code, key, ts): the column name is the key."""
    if not staged:
        return pl.LazyFrame(schema={"code": pl.UInt32, "key": pl.String,
                                    "ts": pl.Datetime("us")})
    parts = [lf.select(pl.col("code"), pl.lit(key, dtype=pl.String).alias("key"),
                       pl.col(col).alias("ts"))
             for col, key in zip(staged, keys)]
    return pl.concat(parts, how="vertical").filter(pl.col("ts").is_not_null())
