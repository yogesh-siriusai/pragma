"""Reading client tables without ever holding a whole one in memory.

A table is a folder of parquet parts (or a single file). Everything here works
from the file footers where it can - row counts, sizes, null counts and
min/max come for free - and reads data only in bounded batches when it must.

Why not let Polars stream the scan: measured on Polars 1.44, a streaming sink
over several large parquet files reads ahead faster than it writes, and its
memory grew with the number of files (1.3 GB for one file, 6 GB for six).
Reading row groups ourselves keeps memory flat no matter how big the table is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

# read column chunks through a small buffer instead of loading each row group
# whole. On a 1M-row row group this cut the reader's peak from 329 MB to 86 MB.
READ_BUFFER = 8 << 20

# rows per read from a file. Batches are gathered from pieces of this size,
# so it only has to be small next to any batch we would ask for.
READ_ROWS = 65_536


@dataclass
class TableFiles:
    """One client table on disk, with what its footers tell us."""

    name: str
    files: list[Path]
    rows: int
    file_bytes: int            # compressed size on disk
    raw_bytes: int             # uncompressed size, from the footers
    schema: dict               # column -> polars dtype

    @property
    def raw_bytes_per_row(self) -> float:
        return self.raw_bytes / self.rows if self.rows else 0.0


def table_paths(source_dir: str | Path, name: str) -> list[Path]:
    """Every parquet part of one table, in a fixed order."""
    source_dir = Path(source_dir)
    folder = source_dir / name
    if folder.is_dir():
        parts = sorted(folder.glob("*.parquet"))
    else:
        parts = sorted(source_dir.glob(f"{name}.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parquet found for table {name!r} under {source_dir}")
    return parts


def scan_table(source_dir: str | Path, name: str) -> pl.LazyFrame:
    """Lazily scan every parquet part of one client table.

    Use this for schema questions and small tables only. Big reads go through
    iter_batches(), which keeps memory bounded.
    """
    return pl.scan_parquet([str(p) for p in table_paths(source_dir, name)])


def open_table(source_dir: str | Path, name: str) -> TableFiles:
    """Footer-only facts about a table: no data is read."""
    files = table_paths(source_dir, name)
    rows, raw = 0, 0
    for path in files:
        md = pq.ParquetFile(path).metadata
        rows += md.num_rows
        for i in range(md.num_row_groups):
            raw += md.row_group(i).total_byte_size
    schema = dict(pl.scan_parquet([str(p) for p in files]).collect_schema())
    return TableFiles(name=name, files=files, rows=rows,
                      file_bytes=sum(p.stat().st_size for p in files),
                      raw_bytes=raw, schema=schema)


# ---------------------------------------------------------------- bounded reading

def iter_batches(table: TableFiles, columns: list[str],
                 batch_rows: int | Callable[[], int]) -> Iterator[tuple[pl.DataFrame, int]]:
    """Yield (batch, row offset) over the whole table, a bounded batch at a time.

    The offset is the global row number of the batch's first row, counting
    across all parts in file order. It is how every row gets a stable id that
    does not depend on how the table was chunked.

    ``batch_rows`` is a number, or a function asked before every batch, so a
    caller can resize the next batch after measuring the last one. Files are
    read in small fixed pieces and gathered up to the size asked for, so the
    batch size can change freely without re-reading anything.
    """
    next_size = batch_rows if callable(batch_rows) else (lambda: batch_rows)
    want = max(int(next_size()), 1)
    offset = 0
    pending: list[pl.DataFrame] = []
    pending_rows = 0

    for path in table.files:
        pf = pq.ParquetFile(path, pre_buffer=False, buffer_size=READ_BUFFER)
        if pf.metadata.num_rows == 0:
            continue
        # a part may lack a column another part has; read what is there and
        # let the missing ones come back as nulls of the right type
        present = [c for c in columns if c in pf.schema_arrow.names]

        # never read a piece bigger than the batch being asked for
        for piece in _pieces(pf, present, min(READ_ROWS, want)):
            pending.append(_frame(piece, table, columns))
            pending_rows += piece.height if isinstance(piece, pl.DataFrame) else piece.num_rows

            # hand out full batches as soon as we have enough rows for one
            while pending_rows >= want:
                gathered = pl.concat(pending, how="vertical_relaxed", rechunk=False)
                head = gathered.slice(0, want)
                rest = gathered.slice(want)
                yield head, offset
                offset += head.height
                pending = [rest] if rest.height else []
                pending_rows = rest.height
                want = max(int(next_size()), 1)

    # whatever is left is the last, shorter batch
    if pending_rows:
        yield pl.concat(pending, how="vertical_relaxed", rechunk=False), offset


def _pieces(pf: pq.ParquetFile, present: list[str], piece_rows: int):
    """Small pieces of one file, holding only the columns it has.

    A file holding none of the wanted columns still has rows, and they still
    count: pyarrow would hand back empty batches and those rows would vanish,
    shifting every row number after them. So we make null rows of the right
    count from the footer instead.
    """
    if present:
        yield from pf.iter_batches(batch_size=piece_rows, columns=present, use_threads=True)
        return
    left = pf.metadata.num_rows
    while left > 0:
        n = min(left, piece_rows)
        # one placeholder column carries the row count; _frame drops it
        yield pl.select(pl.repeat(None, n, dtype=pl.Null).alias("__rows"))
        left -= n


def _frame(piece, table: TableFiles, columns: list[str]) -> pl.DataFrame:
    """One piece read from disk, with every asked-for column present."""
    df = piece if isinstance(piece, pl.DataFrame) else pl.from_arrow(piece)
    if not isinstance(df, pl.DataFrame):
        df = df.to_frame()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None, dtype=table.schema[c]).alias(c) for c in missing])
    return df.select(columns)


# ---------------------------------------------------------------- footer statistics

@dataclass
class ColumnStats:
    """What the footers say about one column; None where they do not say."""

    nulls: int | None
    min_value: object = None
    max_value: object = None


def footer_stats(table: TableFiles) -> dict[str, ColumnStats]:
    """Null counts and min/max per column, read from the footers alone.

    A statistic is only reported when every row group carries it, so a
    partial answer never passes for a whole one.
    """
    nulls: dict[str, int | None] = {}
    lows: dict[str, object] = {}
    highs: dict[str, object] = {}
    seen_rows: dict[str, int] = {}

    for path in table.files:
        md = pq.ParquetFile(path).metadata
        names = [md.schema.column(i).path for i in range(md.num_columns)]
        for rg in range(md.num_row_groups):
            group = md.row_group(rg)
            for i, name in enumerate(names):
                if "." in name:      # a nested leaf; we only profile flat columns
                    continue
                stats = group.column(i).statistics
                seen_rows[name] = seen_rows.get(name, 0) + group.num_rows
                if stats is None or not stats.has_null_count:
                    nulls[name] = None
                elif nulls.get(name, 0) is not None:
                    nulls[name] = nulls.get(name, 0) + stats.null_count
                if stats is None or not stats.has_min_max:
                    lows[name] = highs[name] = _UNKNOWN
                    continue
                if lows.get(name) is _UNKNOWN:
                    continue
                lo, hi = stats.min, stats.max
                lows[name] = lo if name not in lows else _smaller(lows[name], lo)
                highs[name] = hi if name not in highs else _larger(highs[name], hi)

    out = {}
    for name in table.schema:
        # a column absent from some part has stats for fewer rows than the
        # table holds; its nulls then cannot be trusted from footers alone
        complete = seen_rows.get(name) == table.rows
        n = nulls.get(name) if complete else None
        lo, hi = lows.get(name, _UNKNOWN), highs.get(name, _UNKNOWN)
        if lo is _UNKNOWN or hi is _UNKNOWN or not complete:
            lo = hi = None
        out[name] = ColumnStats(nulls=n, min_value=lo, max_value=hi)
    return out


_UNKNOWN = object()


def _smaller(a, b):
    try:
        return b if b < a else a
    except TypeError:
        return _UNKNOWN


def _larger(a, b):
    try:
        return b if b > a else a
    except TypeError:
        return _UNKNOWN


# ---------------------------------------------------------------- distinct values

def distinct_values(table: TableFiles, column: str, batch_rows: int,
                    cap: int | None = None, cast: pl.DataType | None = pl.String,
                    sample_rows: int | None = None) -> tuple[pl.Series | None, int]:
    """The distinct non-null values of one column, read in bounded batches.

    Returns (values, rows looked at). With ``cap`` set, gives up and returns
    None once more than ``cap`` distinct values have been seen - an id column
    has as many values as rows, and holding all of them is exactly the kind
    of memory blow-up this module exists to avoid. ``sample_rows`` stops
    after that many rows, for an estimate on a very large table.
    """
    acc: pl.Series | None = None
    looked = 0
    for df, _ in iter_batches(table, [column], batch_rows):
        s = df.get_column(column)
        if cast is not None:
            s = s.cast(cast)
        s = s.drop_nulls().unique()
        acc = s if acc is None else pl.concat([acc, s]).unique()
        looked += df.height
        if cap is not None and acc.len() > cap:
            return None, looked
        if sample_rows is not None and looked >= sample_rows:
            break
    if acc is None:
        acc = pl.Series(column, [], dtype=cast or table.schema[column])
    return acc, looked
