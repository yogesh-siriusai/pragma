"""What this machine has, and how much of it intake is allowed to use.

Two different memories matter and they are easy to confuse:

  RAM (host memory)   where Python, Polars and Arrow keep data. Every byte
                      read from disk lands here first. Running out of it gets
                      the process killed by the operating system - no error,
                      no traceback, just "Killed".

  GPU memory (VRAM)   on the graphics card. Data only goes there when a step
                      runs on the GPU, and it has to pass through RAM to get
                      there. Running out of it raises an error we can catch.

Intake caps each one separately at a fraction of what is free when the run
starts (85% by default), and sizes every piece of work to fit under the cap.
The same pattern fluxtra uses for its training batch size, applied to data.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# the share of free memory a run may plan to use. The rest is headroom for
# the interpreter, the allocator and anything we measured slightly wrong.
DEFAULT_MEMORY_FRACTION = 0.85
DEFAULT_GPU_MEMORY_FRACTION = 0.85

MEMORY_FRACTION_ENV = "FLUXTRA_INTAKE_MEMORY_FRACTION"
GPU_MEMORY_FRACTION_ENV = "FLUXTRA_INTAKE_GPU_MEMORY_FRACTION"

# when nothing can tell us the RAM size, assume a small machine rather than a
# large one: too-small batches are slow, too-large batches get us killed
FALLBACK_RAM_BYTES = 4 << 30

GIB = float(1 << 30)


@dataclass
class Machine:
    """What we measured when the run started."""

    ram_total: int
    ram_available: int
    rss_at_start: int
    cpu_count: int
    gpu_name: str = ""
    gpu_total: int = 0
    gpu_free: int = 0
    ram_source: str = ""        # where the RAM numbers came from


@dataclass
class Budget:
    """How much of the machine this run may use, in bytes."""

    ram_cap: int                # the process must stay under this RSS
    gpu_cap: int                # allocations on the card stop here
    memory_fraction: float
    gpu_memory_fraction: float

    def room(self) -> int:
        """RAM still free under the cap right now."""
        return max(self.ram_cap - process_rss(), 0)


# ---------------------------------------------------------------- host memory

def _from_psutil() -> tuple[int, int] | None:
    try:
        import psutil
    except ImportError:
        return None
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


def _from_proc_meminfo() -> tuple[int, int] | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    values = {}
    for line in path.read_text().splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            values[name] = int(parts[0]) * 1024
    if "MemTotal" not in values:
        return None
    # MemAvailable is the kernel's own estimate of what a new program can use
    # without swapping. MemFree alone ignores reclaimable page cache.
    available = values.get("MemAvailable", values.get("MemFree", 0))
    return values["MemTotal"], available


def _cgroup_limit() -> tuple[int, int] | None:
    """A container limit, which can be far below the host's RAM.

    Inside docker or kubernetes /proc/meminfo still shows the whole host, so a
    pod with a 16 GB limit on a 512 GB node would plan for 512 GB and die.
    """
    candidates = [
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
         Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
    ]
    for limit_path, usage_path in candidates:
        if not limit_path.exists():
            continue
        raw = limit_path.read_text().strip()
        if raw == "max":
            return None
        try:
            limit = int(raw)
            usage = int(usage_path.read_text().strip()) if usage_path.exists() else 0
        except ValueError:
            return None
        # cgroup v1 reports "no limit" as a huge number near 2**63
        if limit >= (1 << 60):
            return None
        return limit, usage
    return None


def host_memory() -> tuple[int, int, str]:
    """(total, available, where it came from) for this process's RAM."""
    got = _from_proc_meminfo()
    source = "/proc/meminfo"
    if got is None:
        got = _from_psutil()
        source = "psutil"
    if got is None:
        log.warning("cannot read RAM size; assuming %.0f GB", FALLBACK_RAM_BYTES / GIB)
        return FALLBACK_RAM_BYTES, FALLBACK_RAM_BYTES, "assumed"

    total, available = got
    limit = _cgroup_limit()
    if limit is not None:
        cg_limit, cg_usage = limit
        if cg_limit < total:
            total = cg_limit
            available = min(available, max(cg_limit - cg_usage, 0))
            source = "cgroup limit"
    return total, available, source


def process_rss() -> int:
    """Resident memory of this process, in bytes."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        pages = int(statm.read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return 0


def cpu_count() -> int:
    """Cores this process may actually run on, not just cores in the box."""
    if hasattr(os, "sched_getaffinity"):
        return max(len(os.sched_getaffinity(0)), 1)
    return max(os.cpu_count() or 1, 1)


# ---------------------------------------------------------------- gpu memory

def gpu_memory() -> tuple[str, int, int]:
    """(name, total, free) of GPU 0 through NVML. Zeros when there is none.

    NVML reads the driver directly, so it does not create a CUDA context or
    reserve any GPU memory - unlike asking torch, which would.
    """
    try:
        import pynvml
    except ImportError:
        return "", 0, 0
    try:
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() < 1:
                return "", 0, 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return str(name), int(info.total), int(info.free)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 - no driver is a normal CPU box
        log.debug("nvml unavailable: %s", exc)
        return "", 0, 0


# ---------------------------------------------------------------- the budget

def measure() -> Machine:
    """Take one snapshot of the machine at the start of a run."""
    total, available, source = host_memory()
    name, gpu_total, gpu_free = gpu_memory()
    return Machine(ram_total=total, ram_available=available,
                   rss_at_start=process_rss(), cpu_count=cpu_count(),
                   gpu_name=name, gpu_total=gpu_total, gpu_free=gpu_free,
                   ram_source=source)


def _fraction(value: float | None, env: str, default: float) -> float:
    if value is None:
        raw = os.environ.get(env)
        value = float(raw) if raw else default
    if not 0.05 <= value <= 0.98:
        raise ValueError(f"memory fraction must be between 0.05 and 0.98, got {value}")
    return value


def make_budget(machine: Machine, memory_fraction: float | None = None,
                gpu_memory_fraction: float | None = None) -> Budget:
    """Turn what we measured into hard numbers for this run.

    The RAM cap counts what this process already holds, so the cap is "our
    share of what was free, plus what we were already using".
    """
    ram_frac = _fraction(memory_fraction, MEMORY_FRACTION_ENV, DEFAULT_MEMORY_FRACTION)
    gpu_frac = _fraction(gpu_memory_fraction, GPU_MEMORY_FRACTION_ENV,
                         DEFAULT_GPU_MEMORY_FRACTION)
    ram_cap = machine.rss_at_start + int(machine.ram_available * ram_frac)
    gpu_cap = int(machine.gpu_free * gpu_frac)
    return Budget(ram_cap=ram_cap, gpu_cap=gpu_cap,
                  memory_fraction=ram_frac, gpu_memory_fraction=gpu_frac)


def print_machine(machine: Machine, budget: Budget) -> None:
    """The numbers every later decision is based on, printed once."""
    print("  ram           %.1f GB total, %.1f GB free  (%s)"
          % (machine.ram_total / GIB, machine.ram_available / GIB, machine.ram_source))
    print("  ram cap       %.1f GB  (%.0f%% of free)"
          % (budget.ram_cap / GIB, 100 * budget.memory_fraction))
    if machine.gpu_total:
        print("  gpu memory    %.1f GB total, %.1f GB free"
              % (machine.gpu_total / GIB, machine.gpu_free / GIB))
        print("  gpu cap       %.1f GB  (%.0f%% of free)"
              % (budget.gpu_cap / GIB, 100 * budget.gpu_memory_fraction))
    print("  cpu cores     %d" % machine.cpu_count)


# ---------------------------------------------------------------- disk and files

def disk_free(path: str | Path) -> int:
    """Free bytes on the volume that holds ``path`` (or its nearest parent)."""
    p = Path(path).resolve()
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def raise_open_file_limit() -> int:
    """Lift the soft open-file limit to the hard one; return what we can use.

    Staging keeps one file open per customer partition, and the common Linux
    default of 1024 would cap us long before memory does.
    """
    if sys.platform.startswith("win"):
        return 4096
    try:
        import resource
    except ImportError:
        return 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY:
        hard = 1 << 20
    if soft < hard:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            soft = hard
        except (ValueError, OSError):
            pass
    return int(soft)


# ---------------------------------------------------------------- watching memory

class PeakMemory:
    """Track the highest RSS seen while a block runs.

    A background thread samples every few milliseconds, because the peak of a
    sort or a join lasts a fraction of a second and a single reading after the
    fact would miss it entirely.
    """

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.start = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakMemory":
        self.start = process_rss()
        self.peak = self.start
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def _watch(self) -> None:
        while not self._stop.is_set():
            rss = process_rss()
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, process_rss())

    @property
    def grew(self) -> int:
        """How far above the starting point the peak went."""
        return max(self.peak - self.start, 0)
