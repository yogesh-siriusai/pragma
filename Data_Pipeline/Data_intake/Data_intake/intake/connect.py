"""Getting the data in, from wherever the client keeps it.

Fluxtra already resolves any path or URL through one fsspec-backed helper and
stages remote inputs to local scratch. We reuse that rather than inventing a
second way to read a bucket. Without fluxtra installed, local paths still work
and a remote URL says exactly which package is missing.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

# schemes fluxtra's storage layer understands, each behind an optional extra
REMOTE_SCHEMES = ("s3://", "gs://", "gcs://", "az://", "abfs://", "abfss://",
                  "adl://", "memory://", "http://", "https://")


class ConnectError(RuntimeError):
    """A source we cannot reach, with the reason and the fix."""


@dataclass
class Source:
    """One named place client data lives."""

    name: str
    uri: str
    description: str = ""
    # env vars that must be present before we try to read a remote source, so
    # a missing credential fails here instead of halfway through a scan
    requires_env: tuple[str, ...] = ()

    @property
    def is_remote(self) -> bool:
        return is_remote(self.uri)


def is_remote(uri: str) -> bool:
    """True when this needs a cloud filesystem rather than plain disk."""
    return str(uri).startswith(REMOTE_SCHEMES)


def load_sources(path: str | Path) -> dict[str, Source]:
    """Read connectors.yaml: named sources an operator can refer to by name."""
    path = Path(path)
    if not path.exists():
        raise ConnectError(f"connector file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, Source] = {}
    for name, spec in (raw.get("sources") or {}).items():
        spec = spec or {}
        uri = spec.get("uri")
        if not uri:
            raise ConnectError(f"source {name!r} has no uri")
        out[name] = Source(name=name, uri=str(uri),
                           description=spec.get("description", ""),
                           requires_env=tuple(spec.get("requires_env", [])))
    if not out:
        raise ConnectError(f"{path}: no sources declared")
    return out


def resolve(target: str, sources: dict[str, Source] | None = None) -> Source:
    """Turn a CLI argument into a Source: either a known name or a raw path."""
    sources = sources or {}
    if target in sources:
        return sources[target]
    return Source(name=target, uri=target)


def _check_credentials(src: Source) -> None:
    missing = [v for v in src.requires_env if not os.environ.get(v)]
    if missing:
        raise ConnectError(
            f"source {src.name!r} needs these environment variables and they are "
            f"not set: {', '.join(missing)}")


@contextmanager
def staged(src: Source, keep: bool = False):
    """Yield a local directory holding the source data.

    Local paths are handed back untouched - no copy, no temp directory. A
    remote URI is materialised to scratch and removed afterwards, which is the
    same stage-in behaviour the rest of fluxtra uses at its api boundary.
    """
    if not src.is_remote:
        path = Path(src.uri)
        if not path.exists():
            raise ConnectError(f"source path does not exist: {path}")
        yield path
        return

    _check_credentials(src)

    try:
        from fluxtra.storage.cache import materialize_dir
    except ImportError as exc:
        raise ConnectError(
            f"reading {src.uri} needs fluxtra's storage layer "
            f"(pip install fluxtra[aws] / [gcp] / [azure]); {exc}") from exc

    scratch = Path(tempfile.mkdtemp(prefix="fluxtra-intake-src-"))
    try:
        log.info("staging %s -> %s", src.uri, scratch)
        print("  staging %s" % src.uri)
        materialize_dir(src.uri, scratch)
        yield scratch
    finally:
        if not keep:
            shutil.rmtree(scratch, ignore_errors=True)


@contextmanager
def staged_out(dest: str, keep: bool = False):
    """Yield a local directory to write into, then put it where it belongs.

    A local destination is written straight to. A remote one is written to
    scratch and uploaded only when the block finishes without raising, so a
    run that dies halfway never leaves half a dataset in a client's bucket.
    """
    if not is_remote(dest):
        path = Path(dest)
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return

    try:
        from fluxtra.storage.cache import put_dir
    except ImportError as exc:
        raise ConnectError(
            f"writing to {dest} needs fluxtra's storage layer "
            f"(pip install fluxtra[aws] / [gcp] / [azure]); {exc}") from exc

    scratch = Path(tempfile.mkdtemp(prefix="fluxtra-intake-out-"))
    try:
        yield scratch
        # only reached when the caller did not raise
        print("  uploading results to %s" % dest)
        put_dir(scratch, dest)
    finally:
        if not keep:
            shutil.rmtree(scratch, ignore_errors=True)


def describe(src: Source) -> str:
    """One line for the run log."""
    kind = "remote" if src.is_remote else "local"
    return f"{src.name}  ({kind})  {src.uri}"
