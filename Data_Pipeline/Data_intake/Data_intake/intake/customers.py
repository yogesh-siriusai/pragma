"""Who the customers are, and how a row finds its customer.

Every customer id seen anywhere is collected once, sorted, and numbered
0..N-1. That number - the customer code - is what every later step carries
instead of the id itself:

  * it decides which partition a customer lands in, the same way on any
    engine and any machine (a hash would not: the CPU and GPU hash
    differently, and one customer could end up in two places);
  * customers come out in id order, because codes follow id order;
  * a 4-byte integer is far cheaper to sort than a text id.

Child tables have no customer column. They borrow one from a parent table
through a link key, and that lookup is built here too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from .mapping import Mapping, MappingError, TableMap
from .shapes import entity_expr
from .tables import TableFiles, iter_batches

# a lookup we hold in memory should take at most this share of the free room;
# beyond it the child table is processed in hash slices of the parent
LOOKUP_SHARE = 0.25

# rough in-memory bytes per lookup row (a short text key plus a code); only
# used to decide how many slices to cut, never for correctness
LOOKUP_ROW_BYTES = 64


@dataclass
class Customers:
    """The sorted customer ids; a customer's code is its position here."""

    ids: pl.Series              # sorted, unique, no nulls, String

    @property
    def count(self) -> int:
        return self.ids.len()

    def frame(self) -> pl.DataFrame:
        """(_id, code) for joining a batch onto its customer codes."""
        return pl.DataFrame({"_id": self.ids,
                             "code": pl.int_range(self.count, dtype=pl.UInt32, eager=True)})


def entity_tables(mapping: Mapping) -> list[TableMap]:
    """Mapped tables that name their customer directly."""
    return [t for t in mapping.active()
            if t.shape in ("entity_master", "event_stream", "periodic_snapshot") and t.entity]


def build_customers(tables: dict[str, TableFiles], mapping: Mapping,
                    batch_rows: int) -> Customers:
    """Collect, sort and number every customer id across the client's tables.

    Reads one column per table in bounded batches; what we hold is the set of
    distinct ids, which is the size of the customer base, not of the data.
    """
    acc: pl.Series | None = None
    for tmap in entity_tables(mapping):
        table = tables[tmap.name]
        for df, _ in iter_batches(table, [tmap.entity], batch_rows):
            ids = df.select(entity_expr(tmap.entity)).get_column("_id").drop_nulls().unique()
            acc = ids if acc is None else pl.concat([acc, ids]).unique()

    if acc is None:
        acc = pl.Series("_id", [], dtype=pl.String)
    ids = acc.sort().rename("_id")
    if ids.len() >= (1 << 32) - 1:
        raise MappingError(f"{ids.len():,} customers is more than a 4-byte code can number")
    return Customers(ids=ids)


# ---------------------------------------------------------------- parent lookups

def lookup_slices(parent: TableFiles, room: int) -> int:
    """How many hash slices a parent lookup needs to fit in memory."""
    estimate = parent.rows * LOOKUP_ROW_BYTES
    allowed = max(int(room * LOOKUP_SHARE), 1)
    return max(1, math.ceil(estimate / allowed))


def key_expr(col: str, alias: str = "_key") -> pl.Expr:
    """A link key as text, so an integer key and a text key still match."""
    return pl.col(col).cast(pl.String).alias(alias)


def in_slice(n_slices: int, index: int, column: str = "_key") -> pl.Expr:
    """Rows whose key falls in hash slice ``index`` of ``n_slices``.

    Both sides of a lookup are cut with this same expression in the same
    process on the CPU, so a key always lands in the same slice on both sides.
    """
    if n_slices == 1:
        return pl.lit(True)
    return (pl.col(column).hash(seed=0) % n_slices) == index


def parent_pairs(parent: TableFiles, parent_map: TableMap, key: str, batch_rows: int,
                 n_slices: int = 1, index: int = 0) -> pl.DataFrame:
    """Distinct (_key, _id) pairs for one slice of a parent table.

    Raises when a key points at two different customers. That is an error, not
    a coin toss: which customer a child row belongs to would otherwise change
    from run to run. The same key repeated for the same customer is fine.
    """
    parts = []
    for df, _ in iter_batches(parent, [key, parent_map.entity], batch_rows):
        part = (df.select(key_expr(key), entity_expr(parent_map.entity))
                  .filter(pl.col("_key").is_not_null() & pl.col("_id").is_not_null())
                  .filter(in_slice(n_slices, index))
                  .unique())
        parts.append(part)
    if not parts:
        return pl.DataFrame(schema={"_key": pl.String, "_id": pl.String})

    pairs = pl.concat(parts).unique()
    clash = pairs.group_by("_key").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if clash.height:
        examples = ", ".join(clash.sort("_key").get_column("_key").head(5).to_list())
        raise MappingError(
            f"{parent_map.name}.{key}: {clash.height:,} key(s) belong to more than one "
            f"customer (for example {examples}), so a child row cannot be given a "
            f"single customer. Fix the data or link the child table on another key.")
    return pairs


def parent_lookup(parent: TableFiles, parent_map: TableMap, key: str,
                  customers: Customers, batch_rows: int,
                  n_slices: int = 1, index: int = 0) -> pl.DataFrame:
    """(_key, code) for one slice of a parent table."""
    pairs = parent_pairs(parent, parent_map, key, batch_rows, n_slices, index)
    return pairs.join(customers.frame(), on="_id", how="inner").select("_key", "code")


def dimension_frame(dim: TableFiles, tmap: TableMap, columns: list[str],
                    batch_rows: int, room: int) -> pl.DataFrame:
    """A whole dimension table, for enriching a fact table.

    Dimensions are lookups - merchants, product codes - and are expected to be
    small. Each key must appear once, or one fact row would turn into several
    events sharing one event id. One far too big to hold is refused rather
    than allowed to run the machine out of memory.
    """
    on = tmap.enrich_on
    estimate = dim.rows * max(dim.raw_bytes_per_row * 10, LOOKUP_ROW_BYTES)
    if estimate > room * LOOKUP_SHARE:
        raise MappingError(
            f"{tmap.name}: about {estimate / 2**30:.1f} GB to hold in memory for enrichment, "
            f"more than this machine can spare. Drop it, or join it into the fact table "
            f"before intake.")
    parts = [df for df, _ in iter_batches(dim, [on] + columns, batch_rows)]
    frame = pl.concat(parts) if parts else pl.DataFrame(schema={c: dim.schema[c]
                                                                for c in [on] + columns})
    dupes = frame.filter(pl.col(on).is_not_null()).get_column(on).is_duplicated().sum()
    if dupes:
        raise MappingError(f"{tmap.name}: enrich key {on!r} is not unique "
                           f"({dupes:,} repeated rows); enriching would duplicate events")
    return frame.filter(pl.col(on).is_not_null())
