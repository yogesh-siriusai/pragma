"""Train a pragmatiq (PRAGMA) model on the Data_intake output.

Pipeline: validate -> tokenize -> pretrain -> embed.

The intake tool already writes pragmatiq's data contract
(events.parquet + profiles.parquet), so no conversion step is needed.

Run with the pragmatiq venv:
    ..\\pragmatiq\\.venv\\Scripts\\python.exe train_pragma.py                 # full run
    ..\\pragmatiq\\.venv\\Scripts\\python.exe train_pragma.py --max-users 500 --max-steps 50   # smoke test
    ..\\pragmatiq\\.venv\\Scripts\\python.exe train_pragma.py --resume         # continue a stopped run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pragmatiq import api

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "Data_intake" / "Data_intake" / "out" / "demo10k"
DEFAULT_WORK = HERE / "pragma_runs"

# Continuous money/score fields from output_report.json. The intake report
# suggests BPE/drop for these, but pragmatiq bins numbers into percentile
# buckets, which is what we want; pinning them avoids the cardinality heuristic.
NUMERIC_KEYS = (
    "amount", "amt_txn", "balance_before", "current_balance", "fx_rate",
    "minimum_due", "total_aum", "overdue_amount", "overdue_principal_amt",
    "credit_score_internal",
)

TOKENIZER_CONFIG = {
    "target_vocab": 28_000,
    "n_buckets": 64,
    "categorical_threshold": 1000,
    "force_numeric": list(NUMERIC_KEYS),
    "max_events_per_user": 6500,   # p99 in your data is ~5.4k events/user
    "max_event_tokens": 24,
    "calendar_tz": "UTC",
    "seed": 0,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dir with events.parquet + profiles.parquet")
    p.add_argument("--work", type=Path, default=DEFAULT_WORK, help="output dir for shards, runs, embeddings")
    p.add_argument("--run-name", default="pragma_demo10k")
    p.add_argument("--model-size", default="nano", choices=["nano", "small", "medium", "large"],
                   help="nano is the only practical size on CPU")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--token-budget", type=int, default=8192, help="tokens per batch")
    p.add_argument("--max-users", type=int, default=None, help="cap users (for a quick smoke test)")
    p.add_argument("--n-workers", type=int, default=4, help="tokenizer worker processes")
    p.add_argument("--retokenize", action="store_true", help="rebuild shards even if they exist")
    p.add_argument("--resume", action="store_true", help="resume runs/<run-name>/checkpoints/last.pt")
    p.add_argument("--auto-config", action="store_true",
                   help="let pragmatiq size batch/schedule from data+device instead of the flags above")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    shard_dir = args.work / "tok"
    runs_root = args.work / "runs"
    args.work.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] validating {args.data}")
    report = api.validate(args.data)
    for w in report["warnings"]:
        print("  warning:", w)
    if not report["ok"]:
        for e in report["errors"]:
            print("  ERROR:", e)
        raise SystemExit("dataset failed validation; fix the errors above")

    if args.retokenize or not (shard_dir / "shard_manifest.json").exists():
        print(f"[2/4] tokenizing -> {shard_dir}")
        manifest = api.tokenize(args.data, shard_dir, config=TOKENIZER_CONFIG,
                                max_users=args.max_users, n_workers=args.n_workers)
        print(f"  vocab={manifest.get('vocab_size')} users={manifest.get('n_users')}")
    else:
        print(f"[2/4] reusing shards in {shard_dir} (pass --retokenize to rebuild)")

    print(f"[3/4] pretraining '{args.run_name}' ({args.model_size})")
    if args.auto_config:
        config: str | dict = "auto"
        overrides = {"max_steps": args.max_steps}
    else:
        config = {
            "max_steps": args.max_steps,
            "token_budget": args.token_budget,
            "warmup_steps": max(10, args.max_steps // 10),
            "checkpoint_every_min": 10.0,
            "log_every": 20,
            "devices": 1,
            "seed": 0,
        }
        overrides = {}
    plan = api.pretrain_plan(shard_dir, model_size=args.model_size, config=config,
                             run_name=args.run_name, runs_root=runs_root,
                             resume="auto" if args.resume else None, **overrides)
    print("  device:", plan["device"], "| dim:", plan["dim"], "| steps:", plan["max_steps"],
          "| token_budget:", plan["token_budget"])
    summary = api.pretrain(shard_dir, args.run_name, model_size=args.model_size, config=config,
                           runs_root=runs_root, resume="auto" if args.resume else None, **overrides)
    print("  done:", json.dumps(summary, default=str))

    emb_path = args.work / f"{args.run_name}_embeddings.parquet"
    print(f"[4/4] embedding users -> {emb_path}")
    res = api.embed(shard_dir, summary["run_dir"], out=emb_path, token_budget=args.token_budget)
    print(f"  embedded {res['n_users']} users, dim={res['dim']}")
    print(f"\nModel: {summary['run_dir']}\nEmbeddings: {emb_path}")


if __name__ == "__main__":  # required on Windows for the tokenizer's worker processes
    main()
