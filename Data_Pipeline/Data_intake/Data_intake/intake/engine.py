"""Where a query runs: GPU when we have one, CPU when we do not.

Every heavy step is written once as a Polars lazy plan, and this module is the
only place that decides which engine executes it. Two rules keep it honest:

  * the GPU engine is configured explicitly - its own memory pool, capped,
    with no spill into host RAM - instead of left on defaults that page into
    RAM and get the process killed on a small machine;
  * a plan the GPU cannot run raises instead of quietly running somewhere
    else, so every step reports where it really ran.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import polars as pl

from .resources import gpu_memory

log = logging.getLogger(__name__)

# same env var the rest of fluxtra uses, so one setting controls the whole tool
DEVICE_ENV = "FLUXTRA_DEVICE"

# cudf-polars defaults to CUDA managed memory, which lets the driver move GPU
# pages into host RAM when the card fills up. With little RAM and no swap that
# turns a catchable GPU out-of-memory into the kernel killing the process.
# https://developer.nvidia.com/blog/how-to-work-with-data-exceeding-vram-in-the-polars-gpu-engine
os.environ.setdefault("POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY", "0")


@dataclass
class Engine:
    """The resolved execution target, plus what happened on it so far."""

    device: str          # "cuda" or "cpu"
    backend: str         # "cudf-polars" or "polars"
    reason: str          # why we landed here, printed once at startup
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_cap: int = 0     # bytes the GPU pool may hand out, 0 = no limit set

    # filled in as the run goes
    runs: dict[str, int] = field(default_factory=lambda: {"gpu": 0, "cpu": 0})
    # step -> why the GPU cannot run it. A step the GPU cannot do moves to the
    # CPU on its own; every other step keeps the GPU. "*" means no GPU at all.
    disabled: dict[str, str] = field(default_factory=dict)
    _gpu: object = None

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda" and ALL_STEPS not in self.disabled

    def uses_gpu(self, step: str) -> bool:
        return self.is_gpu and step not in self.disabled


ALL_STEPS = "*"


class GpuOutOfMemory(RuntimeError):
    """The step did not fit on the card. The caller should shrink the work."""


def _probe_gpu():
    """Return (ok, name, memory_gb, why_not). Never raises."""
    name, total, _free = gpu_memory()
    if not total:
        return False, "", 0.0, "no CUDA device visible"

    # the GPU engine itself is a separate package; a CUDA card without it is
    # still a CPU run, and saying so beats a confusing failure later
    try:
        import cudf_polars  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - ImportError or a broken CUDA install
        return False, name, total / 1e9, f"found {name} but cudf-polars cannot load ({exc})"

    return True, name, total / 1e9, ""


def resolve_engine(device: str = "auto", gpu_cap: int = 0) -> Engine:
    """Pick the execution engine for this run.

    ``device`` is "auto", "cuda" or "cpu". FLUXTRA_DEVICE overrides "auto" so a
    scheduler can force a device without changing the command. ``gpu_cap`` is
    the most the GPU memory pool may allocate, in bytes.
    """
    if device == "auto":
        device = os.environ.get(DEVICE_ENV, "auto")

    if device == "cpu":
        return Engine(device="cpu", backend="polars", reason="asked for cpu")

    ok, name, mem, why_not = _probe_gpu()

    if device == "cuda" and not ok:
        # an explicit request we cannot honour is worth a loud warning, but it
        # must not stop the run - CPU produces the same answer
        log.warning("cuda was requested but %s; falling back to cpu", why_not)
        return Engine(device="cpu", backend="polars",
                      reason=f"cuda requested but {why_not}")

    if ok:
        return Engine(device="cuda", backend="cudf-polars",
                      reason="CUDA device and cudf-polars both present",
                      gpu_name=name, gpu_memory_gb=mem, gpu_cap=gpu_cap)

    return Engine(device="cpu", backend="polars", reason=why_not)


def print_engine(engine: Engine) -> None:
    """Print the device decision once, at the top of a run."""
    print("")
    print("  engine        %s  (%s)" % (engine.device, engine.backend))
    if engine.device == "cuda":
        print("  gpu           %s" % engine.gpu_name)
    print("  reason        %s" % engine.reason)


# ---------------------------------------------------------------- the gpu engine

def _build_gpu_engine(engine: Engine):
    """A GPU engine with its own capped memory pool and no silent fallback.

    The in-memory executor is the right one here: every plan we hand it is a
    single batch already sized to fit, so there is nothing to stream.
    """
    mr = None
    try:
        import rmm
        upstream = rmm.mr.CudaAsyncMemoryResource()
        if engine.gpu_cap:
            # allocations past the cap raise MemoryError instead of growing
            mr = rmm.mr.LimitingResourceAdaptor(upstream, allocation_limit=engine.gpu_cap)
        else:
            mr = upstream
    except Exception as exc:  # noqa: BLE001 - fall back to the engine's own pool
        log.warning("could not set up a capped GPU memory pool (%s); using the default", exc)

    # GPUEngine accepts any option here and only checks it when a query runs,
    # so a bad option shows up in probe_steps(), before any real work
    return pl.GPUEngine(memory_resource=mr, raise_on_fail=True, executor="in-memory")


def _gpu_engine(engine: Engine):
    if engine._gpu is None:
        engine._gpu = _build_gpu_engine(engine)
    return engine._gpu


def _is_out_of_memory(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("out of memory", "out_of_memory", "bad_alloc",
                                   "outofmemory", "cudaerrormemoryallocation",
                                   "maximum pool size exceeded", "allocation limit"))


# ---------------------------------------------------------------- running a plan

def collect(lf: pl.LazyFrame, engine: Engine, what: str = "",
            step: str | None = None) -> pl.DataFrame:
    """Run a plan on the GPU when we can, otherwise on the CPU.

    Out of GPU memory raises GpuOutOfMemory so the caller can split the work
    and try again. Any other GPU failure - usually an operation cudf-polars
    does not support - is logged once, that ``step`` moves to the CPU for the
    rest of the run, and the identical plan runs on the CPU. Other steps keep
    the GPU.
    """
    df, _where = collect_where(lf, engine, what, step)
    return df


def collect_where(lf: pl.LazyFrame, engine: Engine, what: str = "",
                  step: str | None = None) -> tuple[pl.DataFrame, str]:
    """Like collect(), but also say where the plan actually ran."""
    t0 = time.time()
    step = step or what or "step"

    if engine.uses_gpu(step):
        try:
            gpu = _gpu_engine(engine)
        except Exception as exc:  # noqa: BLE001 - no usable GPU engine at all
            engine.disabled[ALL_STEPS] = _why(exc)
            print("  the gpu engine could not start (%s); this run uses the cpu"
                  % engine.disabled[ALL_STEPS])
            gpu = None

        if gpu is not None:
            try:
                df = lf.collect(engine=gpu)
                engine.runs["gpu"] += 1
                _say_done(what, df, t0, "gpu")
                return df, "gpu"
            except Exception as exc:  # noqa: BLE001 - classified just below
                if _is_out_of_memory(exc):
                    raise GpuOutOfMemory(f"{what or step}: {exc}") from exc
                engine.disabled[step] = _why(exc)
                print("  gpu could not run the %s step (%s); that step uses the cpu from now on"
                      % (step, engine.disabled[step]))

    df = lf.collect()
    engine.runs["cpu"] += 1
    _say_done(what, df, t0, "cpu")
    return df, "cpu"


def _why(exc: BaseException) -> str:
    """One short line naming what went wrong."""
    text = str(exc).strip()
    first = text.splitlines()[0][:200] if text else ""
    return f"{type(exc).__name__}: {first}"


def _say_done(what: str, df: pl.DataFrame, t0: float, where: str) -> None:
    if not what:
        return
    print("    %-34s %10s rows   %6.2fs  [%s]"
          % (what, format(df.height, ","), time.time() - t0, where))


def _sample_steps():
    """The transform's own two GPU steps, on a tiny sample."""
    # imported here: shapes does not need the engine, and this avoids a cycle
    from .shapes import changes_only, melt_table, sort_events

    wide = pl.LazyFrame({
        "code": pl.Series([2, 1, 1, 2, 1], dtype=pl.UInt32),
        "ts": pl.Series([3, 1, 2, 4, 5]).cast(pl.Datetime("us")),
        "uid": pl.Series([10, 11, 12, 13, 14], dtype=pl.Int64),
        "_subject": ["k", "k", "k", None, "k"],
        "f0": ["a", None, "b", "c", "b"],
        "f1": ["x", "y", "y", None, "y"],
    })
    return {
        "events": sort_events(melt_table(wide.drop("_subject"), ["f0", "f1"], [0, 1],
                                         pl.UInt16, subject=False)),
        "snapshots": changes_only(melt_table(wide, ["f0", "f1"], [0, 1],
                                             pl.UInt16, subject=True)),
    }


def probe_steps(engine: Engine) -> dict[str, tuple[bool, str]]:
    """Try each GPU step once on a tiny sample, against the CPU answer.

    A step that fails or disagrees is switched to the CPU for the whole run,
    before any real data is spent on it. Used at the start of pass 2 and by
    `doctor`, so both report the same thing.
    """
    results: dict[str, tuple[bool, str]] = {}
    if not engine.is_gpu:
        return {step: (False, engine.reason) for step in ("events", "snapshots")}

    try:
        gpu = _gpu_engine(engine)
    except Exception as exc:  # noqa: BLE001 - no usable GPU engine at all
        engine.disabled[ALL_STEPS] = _why(exc)
        return {step: (False, engine.disabled[ALL_STEPS]) for step in ("events", "snapshots")}

    for step, plan in _sample_steps().items():
        try:
            got = plan.collect(engine=gpu)
        except Exception as exc:  # noqa: BLE001 - this is the answer we want to show
            engine.disabled[step] = _why(exc)
            results[step] = (False, engine.disabled[step])
            continue
        if not got.equals(plan.collect()):
            engine.disabled[step] = "the GPU answer differs from the CPU"
            results[step] = (False, engine.disabled[step])
        else:
            results[step] = (True, "ran on the GPU and matched the CPU")
    return results


def self_test(engine: Engine) -> tuple[bool, str]:
    """Prove the GPU really runs the transform's steps, for `doctor`."""
    if not engine.is_gpu:
        return False, engine.reason
    results = probe_steps(engine)
    detail = "; ".join(f"{step} step: {'gpu' if ok else 'cpu (' + why + ')'}"
                       for step, (ok, why) in results.items())
    return all(ok for ok, _ in results.values()), detail
