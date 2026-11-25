"""Command line for fluxtra-intake. Parses arguments, calls api, prints.

No logic lives here. Run it as:

    python -m intake.cli doctor
    python -m intake.cli discover ../data --out work/mapping.yaml
    python -m intake.cli check    ../data --mapping work/mapping.yaml
    python -m intake.cli run      ../data --mapping work/mapping.yaml --out out/client
    python -m intake.cli report   out/client
    python -m intake.cli pipeline ../data --mapping work/mapping.yaml --out out/client
"""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    """Every command and flag the tool accepts."""
    parser = argparse.ArgumentParser(
        prog="fluxtra-intake",
        description="Turn client tables into the two files fluxtra reads.")
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='where to run; auto honours FLUXTRA_DEVICE')
    parser.add_argument('--memory-fraction', type=float, default=None,
                        help='share of free RAM the run may use (default 0.85)')
    parser.add_argument('--gpu-memory-fraction', type=float, default=None,
                        help='share of free GPU memory the run may use (default 0.85)')
    parser.add_argument('--connectors', type=str, default=None,
                        help='connectors.yaml defining named sources')
    parser.add_argument('--verbose', action='store_true', help='debug logging')

    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('doctor', help='show the machine and prove the GPU works')
    p.add_argument('--work-dir', type=str, default='.',
                   help='where staging would go, to report its free disk')

    p = sub.add_parser('discover', help='scan tables and draft a mapping')
    p.add_argument('source', type=str, help='folder or named source')
    p.add_argument('--out', type=str, default=None, help='where to write the draft')
    p.add_argument('--client', type=str, default='', help='client name for the draft')

    p = sub.add_parser('check', help='score readiness against a mapping')
    p.add_argument('source', type=str)
    p.add_argument('--mapping', type=str, required=True)
    p.add_argument('--out', type=str, default=None, help='where to write the report')
    p.add_argument('--now', type=str, default=None,
                   help='fix the clock, e.g. 2026-10-03, for a reproducible run')
    p.add_argument('--no-deep-links', action='store_true',
                   help='skip actually resolving child rows to their parents')

    p = sub.add_parser('run', help='transform into events/profiles parquet')
    p.add_argument('source', type=str)
    p.add_argument('--mapping', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--now', type=str, default=None)
    p.add_argument('--skip-check', action='store_true',
                   help='do not re-run the mapping gate first')
    p.add_argument('--work-dir', type=str, default=None,
                   help='where staged partitions go (default: <out>/_work)')
    p.add_argument('--resume', action='store_true',
                   help='continue a stopped run, reusing the tables it already staged')
    p.add_argument('--keep-work', action='store_true',
                   help='keep the staged partitions after a successful run')

    p = sub.add_parser('report', help='reconcile output and forecast the vocabulary')
    p.add_argument('out', type=str, help='the directory run wrote to')
    p.add_argument('--mapping', type=str, default=None,
                   help='so numeric fields are reported as numeric')
    p.add_argument('--rows-in', type=int, default=0,
                   help='source row count, for the reconciliation line')

    p = sub.add_parser('pipeline', help='check, run and report in one go')
    p.add_argument('source', type=str)
    p.add_argument('--mapping', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--now', type=str, default=None)
    p.add_argument('--work-dir', type=str, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a shell exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format='%(levelname)s %(name)s: %(message)s',
                        stream=sys.stderr)

    # imported here so --help works even where the heavy packages are missing
    from . import api

    if args.command == 'doctor':
        result = api.doctor(device=args.device, memory_fraction=args.memory_fraction,
                            gpu_memory_fraction=args.gpu_memory_fraction,
                            work_dir=args.work_dir)
        return 0 if result['gpu_ok'] or args.device == 'cpu' else 3

    if args.command == 'discover':
        api.discover(args.source, out=args.out, client=args.client,
                     connectors=args.connectors, device=args.device)
        return 0

    if args.command == 'check':
        result = api.check(args.source, args.mapping, connectors=args.connectors,
                           device=args.device, out=args.out, now=args.now,
                           deep_links=not args.no_deep_links,
                           memory_fraction=args.memory_fraction)
        gate, readiness = result['gate'], result['readiness']
        if not gate.ok:
            return 2
        return 0 if readiness.band != 'RED' else 1

    if args.command == 'run':
        api.run(args.source, args.mapping, args.out, connectors=args.connectors,
                device=args.device, now=args.now, skip_check=args.skip_check,
                memory_fraction=args.memory_fraction,
                gpu_memory_fraction=args.gpu_memory_fraction,
                work_dir=args.work_dir, resume=args.resume, keep_work=args.keep_work)
        return 0

    if args.command == 'report':
        api.output_report(args.out, rows_in=args.rows_in, device=args.device,
                          mapping_path=args.mapping)
        return 0

    if args.command == 'pipeline':
        api.pipeline(args.source, args.mapping, args.out,
                     connectors=args.connectors, device=args.device, now=args.now,
                     memory_fraction=args.memory_fraction,
                     gpu_memory_fraction=args.gpu_memory_fraction,
                     work_dir=args.work_dir)
        return 0

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
