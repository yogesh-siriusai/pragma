"""What came out, and did we lose anything we should not have.

Two jobs. Reconcile the output against the input so no row vanishes silently,
and predict what the tokenizer will do with each field - because the failure
that costs the most is the one that does not raise.

Everything here is counted while the transform runs, range by range, so the
report costs nothing extra at the end. Re-reading a finished events file to
count it would mean expanding every field of every event again - about 89 GB
at 100k customers - which is exactly the kind of step that kills a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

# the tokenizer's own thresholds (fluxtra/data/tokenizer.py, TokenizerConfig).
# A field above the categorical threshold goes to BPE, which is where a
# vocabulary fills up with strings that can never repeat.
CATEGORICAL_THRESHOLD = 1000
NUMERIC_BUCKETS = 64

# we stop collecting a field's distinct values past this many: the advice
# below only needs to know "over 1,000" and "over 10,000"
DISTINCT_CAP = 10 * CATEGORICAL_THRESHOLD

# a field present on nearly every event of its source is structural; one on a
# handful is an exception flag. Both are fine, but worth telling apart.
RARE_AT = 0.01


@dataclass
class FieldPressure:
    """What the tokenizer will probably make of one field."""

    key: str
    source: str
    events: int
    distinct: int
    treatment: str
    advice: str = ""
    distinct_capped: bool = False


@dataclass
class OutputReport:
    """The reconciliation plus the vocabulary forecast."""

    rows_in: int
    events_out: int
    customers: int
    profiles_out: int
    events_per_customer: dict[str, float] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    pressure: list[FieldPressure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reconciliation: dict[str, dict] = field(default_factory=dict)
    resources: dict = field(default_factory=dict)

    @property
    def vocab_estimate(self) -> int:
        """Rough distinct-value count the tokenizer would have to carry."""
        return sum(p.distinct for p in self.pressure)


# ---------------------------------------------------------------- counting as we go

class FieldTally:
    """Per (field, source): how many events carry it and how many values it has.

    Distinct values are kept as real sets up to DISTINCT_CAP and then dropped,
    so memory stays bounded by fields x cap whatever the data size.
    """

    def __init__(self, key_names: list[str], source_by_ordinal: list[str]):
        self.key_names = key_names
        self.source_by_ordinal = source_by_ordinal
        self.events: dict[tuple[int, int], int] = {}
        self.values: dict[tuple[int, int], set] = {}
        self.capped: set[tuple[int, int]] = set()
        self.events_per_customer: np.ndarray | None = None

    def add(self, long: pl.DataFrame, ordinals_per_pair: np.ndarray) -> None:
        """Count one range's key/value rows (one row is one field of one event)."""
        if long.height == 0:
            return
        frame = pl.DataFrame({"key_id": long.get_column("key_id"),
                              "ord": pl.Series(ordinals_per_pair, dtype=pl.Int64),
                              "value": long.get_column("value")})

        # every row is one field on one event, so rows per (key, source) = events
        counts = frame.group_by(["key_id", "ord"]).agg(pl.len().alias("n"))
        for kid, o, n in counts.iter_rows():
            k = (int(kid), int(o))
            self.events[k] = self.events.get(k, 0) + int(n)

        # distinct values, only for fields still under the cap
        open_keys = [k for k in self.events if k not in self.capped]
        if not open_keys:
            return
        keep = pl.DataFrame({"key_id": [k[0] for k in open_keys],
                             "ord": [k[1] for k in open_keys]},
                            schema={"key_id": frame.schema["key_id"], "ord": pl.Int64})
        uniq = frame.join(keep, on=["key_id", "ord"], how="semi").unique()
        for (kid, o), part in uniq.partition_by(["key_id", "ord"], as_dict=True).items():
            k = (int(kid), int(o))
            # this range alone has more distinct values than the cap: stop
            # here, before turning millions of them into Python strings
            if part.height > DISTINCT_CAP:
                self.capped.add(k)
                self.values[k] = set()
                continue
            seen = self.values.setdefault(k, set())
            seen.update(part.get_column("value").to_list())
            if len(seen) > DISTINCT_CAP:
                self.capped.add(k)
                self.values[k] = set()

    def pressure(self, numeric_keys: set[str]) -> list[FieldPressure]:
        out = []
        for (kid, o), n in self.events.items():
            key = self.key_names[kid]
            capped = (kid, o) in self.capped
            distinct = DISTINCT_CAP + 1 if capped else len(self.values.get((kid, o), ()))
            treat, advice = _treatment(key, distinct, numeric_keys)
            out.append(FieldPressure(key=key, source=self.source_by_ordinal[o], events=n,
                                     distinct=distinct, treatment=treat, advice=advice,
                                     distinct_capped=capped))
        out.sort(key=lambda p: (-p.distinct, p.key, p.source))
        return out


def _treatment(key: str, distinct: int, numeric_keys: set[str]) -> tuple[str, str]:
    """Predict the tokenizer's choice, and say what to do about it."""
    if key in numeric_keys:
        return f"numeric, {NUMERIC_BUCKETS} percentile buckets", ""
    if distinct >= CATEGORICAL_THRESHOLD:
        advice = ("drop - this looks like an identifier, not behaviour"
                  if distinct > DISTINCT_CAP
                  else "consider dropping or bucketing")
        return "text, BPE", advice
    return f"categorical, {distinct} ids", ""


def _quantiles(per_customer: np.ndarray) -> dict[str, float]:
    s = per_customer[per_customer > 0]
    if s.size == 0:
        return {}
    return {"min": float(s.min()), "p50": float(np.quantile(s, 0.5)),
            "p90": float(np.quantile(s, 0.9)), "p99": float(np.quantile(s, 0.99)),
            "max": float(s.max()), "mean": round(float(s.mean()), 1)}


def build_report(tally: FieldTally, rows_in: int, events_out: int, profiles_out: int,
                 by_source: dict[str, int], numeric_keys: set[str],
                 reconciliation: dict[str, dict], resources: dict) -> OutputReport:
    """Assemble the report from what the run counted."""
    per = tally.events_per_customer if tally.events_per_customer is not None \
        else np.zeros(0, np.int64)
    customers = int(np.count_nonzero(per))
    pressure = tally.pressure(numeric_keys)
    rep = OutputReport(rows_in=rows_in, events_out=events_out, customers=customers,
                       profiles_out=profiles_out, events_per_customer=_quantiles(per),
                       by_source=dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
                       pressure=pressure, reconciliation=reconciliation,
                       resources=resources)

    # things worth saying out loud
    for p in pressure:
        if p.advice.startswith("drop"):
            shown = f"over {DISTINCT_CAP:,}" if p.distinct_capped else f"{p.distinct:,}"
            rep.warnings.append(f"{p.key} has {shown} distinct values - {p.advice}")
    if profiles_out < customers:
        rep.warnings.append(f"{customers - profiles_out:,} customers have events but no profile row")
    rare = [p.key for p in pressure if p.events < max(events_out * RARE_AT, 1)]
    if rare:
        rep.warnings.append(f"{len(rare)} fields appear on under 1% of events")
    for table, rec in reconciliation.items():
        if rec.get("unexplained"):
            rep.warnings.append(f"{table}: {rec['unexplained']:,} rows are not accounted for")
    return rep


# ---------------------------------------------------------------- showing it

def print_report(rep: OutputReport) -> None:
    """The reconciliation, then the fields most likely to cause trouble."""
    print("")
    print("  source rows in       %12s" % format(rep.rows_in, ","))
    print("  events out           %12s" % format(rep.events_out, ","))
    print("  customers            %12s" % format(rep.customers, ","))
    print("  profiles out         %12s" % format(rep.profiles_out, ","))
    if rep.events_per_customer:
        q = rep.events_per_customer
        print("  events per customer  min %s  p50 %s  p90 %s  max %s"
              % (format(int(q["min"]), ","), format(int(q["p50"]), ","),
                 format(int(q["p90"]), ","), format(int(q["max"]), ",")))

    if rep.reconciliation:
        print("")
        print("  where every source row went")
        print("    %-26s %12s %12s  %s" % ("table", "rows in", "events", "dropped"))
        for name, rec in rep.reconciliation.items():
            dropped = ", ".join(f"{k} {v:,}" for k, v in rec.get("dropped", {}).items() if v)
            print("    %-26s %12s %12s  %s"
                  % (name[:26], format(rec["rows_in"], ","),
                     format(rec.get("events", 0), ","), dropped or "-"))

    print("")
    print("  events by source")
    for src, n in rep.by_source.items():
        print("    %-16s %12s" % (src, format(n, ",")))

    print("")
    print("  highest cardinality fields (what the tokenizer will see)")
    print("    %-26s %-14s %10s  %s" % ("field", "source", "distinct", "treatment"))
    for p in rep.pressure[:12]:
        shown = f">{DISTINCT_CAP:,}" if p.distinct_capped else format(p.distinct, ",")
        print("    %-26s %-14s %10s  %s%s"
              % (p.key[:26], p.source[:14], shown, p.treatment,
                 "  <- " + p.advice if p.advice else ""))

    if rep.warnings:
        print("")
        print("  warnings")
        for w in rep.warnings:
            print("    - %s" % w)
    print("")


def write_report(rep: OutputReport, out_dir: str | Path) -> Path:
    """Persist the reconciliation next to the files it describes."""
    out_dir = Path(out_dir)
    doc = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_in": rep.rows_in,
        "events_out": rep.events_out,
        "customers": rep.customers,
        "profiles_out": rep.profiles_out,
        "events_per_customer": rep.events_per_customer,
        "by_source": rep.by_source,
        "vocabulary_estimate": rep.vocab_estimate,
        "reconciliation": rep.reconciliation,
        "resources": rep.resources,
        "fields": [{"key": p.key, "source": p.source, "events": p.events,
                    "distinct": p.distinct, "distinct_capped": p.distinct_capped,
                    "treatment": p.treatment, "advice": p.advice} for p in rep.pressure],
        "warnings": rep.warnings,
    }
    path = out_dir / "output_report.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def load_report(out_dir: str | Path) -> OutputReport | None:
    """The report a run left behind, if it still matches the files beside it."""
    out_dir = Path(out_dir)
    path = out_dir / "output_report.json"
    events = out_dir / "events.parquet"
    if not path.exists() or not events.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    # a report from an older run over the same folder must not describe new files
    if pq.ParquetFile(events).metadata.num_rows != doc.get("events_out"):
        return None
    pressure = [FieldPressure(key=f["key"], source=f["source"], events=f["events"],
                              distinct=f["distinct"], treatment=f["treatment"],
                              advice=f.get("advice", ""),
                              distinct_capped=f.get("distinct_capped", False))
                for f in doc.get("fields", [])]
    return OutputReport(rows_in=doc["rows_in"], events_out=doc["events_out"],
                        customers=doc["customers"], profiles_out=doc["profiles_out"],
                        events_per_customer=doc.get("events_per_customer", {}),
                        by_source=doc.get("by_source", {}), pressure=pressure,
                        warnings=doc.get("warnings", []),
                        reconciliation=doc.get("reconciliation", {}),
                        resources=doc.get("resources", {}))


def report_from_files(out_dir: str | Path, numeric_keys: set[str],
                      batch_rows: int = 65_536) -> OutputReport:
    """Rebuild the report by reading finished files, a bounded batch at a time.

    Only needed for output written without a report beside it. Memory stays
    bounded: per-customer counts are one number per customer, and field
    values are tallied with the same cap as during a run.
    """
    out_dir = Path(out_dir)
    pf = pq.ParquetFile(out_dir / "events.parquet")
    n_profiles = pq.ParquetFile(out_dir / "profiles.parquet").metadata.num_rows

    key_index: dict[str, int] = {}
    source_index: dict[str, int] = {}
    per_customer: dict[str, int] = {}
    by_source: dict[str, int] = {}
    tally = FieldTally([], [])

    for batch in pf.iter_batches(batch_size=batch_rows):
        df = pl.from_arrow(batch)
        for uid, n in df.group_by("user_id").agg(pl.len()).iter_rows():
            per_customer[uid] = per_customer.get(uid, 0) + n
        for src, n in df.group_by("source").agg(pl.len()).iter_rows():
            by_source[src] = by_source.get(src, 0) + n
            source_index.setdefault(src, len(source_index))

        pairs = df.select("source", "fields").explode("fields", empty_as_null=True).select(
            pl.col("source"),
            pl.col("fields").struct.field("key").alias("key"),
            pl.col("fields").struct.field("value").alias("value")).drop_nulls("key")
        for k in pairs.get_column("key").unique().to_list():
            key_index.setdefault(k, len(key_index))
        long = pairs.select(
            pl.col("key").replace_strict(key_index, return_dtype=pl.Int64).alias("key_id"),
            pl.col("value"))
        ords = pairs.get_column("source").replace_strict(source_index,
                                                         return_dtype=pl.Int64).to_numpy()
        tally.add(long, ords)

    tally.key_names = [k for k, _ in sorted(key_index.items(), key=lambda kv: kv[1])]
    tally.source_by_ordinal = [s for s, _ in sorted(source_index.items(), key=lambda kv: kv[1])]
    tally.events_per_customer = np.array(list(per_customer.values()), dtype=np.int64)
    return build_report(tally, rows_in=0, events_out=pf.metadata.num_rows,
                        profiles_out=n_profiles, by_source=by_source,
                        numeric_keys=numeric_keys, reconciliation={}, resources={})
