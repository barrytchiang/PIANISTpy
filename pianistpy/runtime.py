"""Runtime options of the session: verbosity, timing and the number of cores.

None of these changes a number, and none is part of the identity of a
result. Each is set once for the session, by a call or by an environment
variable, and no function of the package takes it as an argument.
"""

__all__ = [
    "Timing",
    "allowed_cores",
    "card_memory",
    "configure_jax",
    "default_compilation_cache",
    "get_workers",
    "log",
    "log_to_file",
    "peak_memory",
    "recent_messages",
    "platform",
    "set_compilation_cache",
    "set_platform",
    "set_timing",
    "set_verbosity",
    "set_workers",
    "timed",
    "timing_enabled",
    "timing_record",
    "workers_explicit",
]

import contextlib
import contextvars
import logging
import os
import sys
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from .exceptions import InputError

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    import resource

VERBOSITY_LEVELS = {
    "quiet": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
VERBOSITY_VARIABLE = "PIANISTPY_VERBOSITY"
TIMING_VARIABLE = "PIANISTPY_TIMING"
WORKERS_VARIABLE = "PIANISTPY_WORKERS"
PLATFORM_VARIABLE = "PIANISTPY_PLATFORM"
# The platforms the switch names: "auto" leaves JAX its own order (a GPU
# where one is usable, the CPU otherwise); "cpu" is the CPU-only mode, the
# same numbers on every machine; "gpu" asks for the CUDA backend with the
# CPU behind it.
PLATFORMS = {"auto": None, "cpu": "cpu", "gpu": "cuda,cpu"}
# Number of the most recent messages kept, whatever the verbosity, to be
# shown when the package raises one of its own errors.
RECENT_MESSAGES_MAX = 50
# Environment variables a batch scheduler sets inside a job.
SCHEDULER_JOB_VARIABLES = (
    "PBS_JOBID",
    "SLURM_JOB_ID",
    "LSB_JOBID",
    "JOB_ID",
    "COBALT_JOBID",
)
# Environment variables in which a scheduler states the cores of the job.
SCHEDULER_CORE_VARIABLES = ("SLURM_CPUS_PER_TASK", "PBS_NP", "NCPUS", "NSLOTS")

log = logging.getLogger("pianistpy")
log.setLevel(logging.DEBUG)
log.propagate = False


class _RecentHandler(logging.Handler):
    """Keep the most recent messages of every level."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: deque[str] = deque(maxlen=RECENT_MESSAGES_MAX)

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


_FORMAT = logging.Formatter("%(name)s %(levelname)s: %(message)s")
_recent = _RecentHandler()
_recent.setFormatter(_FORMAT)
_screen = logging.StreamHandler()
_screen.setFormatter(_FORMAT)
log.addHandler(_recent)
log.addHandler(_screen)


def set_verbosity(level: str) -> None:
    """Set what the package prints for the rest of the session.

    Parameters
    ----------
    level : {"quiet", "info", "debug"}
        ``"quiet"`` shows warnings and errors only and is the default;
        ``"info"`` adds one line per stage; ``"debug"`` adds the detail of
        each stage.

    Notes
    -----
    The verbosity never changes what is computed. The environment variable
    ``PIANISTPY_VERBOSITY`` sets the level at import.
    """
    if level not in VERBOSITY_LEVELS:
        raise InputError(
            f"The verbosity is one of {list(VERBOSITY_LEVELS)}; got {level!r}."
        )
    _screen.setLevel(VERBOSITY_LEVELS[level])


def recent_messages() -> tuple[str, ...]:
    """Return the most recent messages of the package, oldest first.

    The messages are kept whatever the verbosity, so that a quiet run that
    fails can still say what it was doing.
    """
    return tuple(_recent.messages)


@contextlib.contextmanager
def log_to_file(path: str | os.PathLike, level: str = "debug") -> Iterator[None]:
    """Write the messages of the enclosed block to a file as well.

    Parameters
    ----------
    path : path
        File to append to.
    level : {"quiet", "info", "debug"}, optional
        Lowest level written to the file.
    """
    if level not in VERBOSITY_LEVELS:
        raise InputError(
            f"The verbosity is one of {list(VERBOSITY_LEVELS)}; got {level!r}."
        )
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(VERBOSITY_LEVELS[level])
    handler.setFormatter(_FORMAT)
    log.addHandler(handler)
    try:
        yield
    finally:
        log.removeHandler(handler)
        handler.close()


if sys.platform == "win32":

    class _MemoryCounters(ctypes.Structure):
        """The process memory counters of the Windows API."""

        _fields_ = (
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        )


def peak_memory() -> int | None:
    """Return the largest resident memory this process has held, in bytes.

    This is the memory of the host process. Memory held on a GPU is not
    part of it.

    Returns
    -------
    peak : int or None
        The peak resident memory since the process started, or None where
        the operating system does not report it.
    """
    if sys.platform == "win32":
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(_MemoryCounters),
            wintypes.DWORD,
        )
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes.
        return int(peak) if sys.platform == "darwin" else int(peak) * 1024


@dataclass
class Timing:
    """Wall times of the stages of one computation, and the memory they reached.

    Parameters
    ----------
    stages : dict of str to float
        Seconds spent in each stage, in the order the stages first ran. A
        stage that runs more than once accumulates.
    memory : dict of str to int
        Peak resident memory of the process, in bytes, as it stood when each
        stage last ended. The value never decreases from one stage to the
        next: a stage raised the peak if its value exceeds the one before.
    """

    stages: dict[str, float] = field(default_factory=dict)
    memory: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Sum of the stage times, in seconds."""
        return sum(self.stages.values())

    def to_dict(self) -> dict[str, float]:
        """Return the stage times as a plain dictionary."""
        return dict(self.stages)


_timing_on = False
_current_timing: contextvars.ContextVar[Timing | None] = contextvars.ContextVar(
    "pianistpy_timing", default=None
)


def set_timing(enabled: bool) -> None:
    """Switch the printing of stage times on or off.

    The times are always measured and kept with the result. This switch
    controls printing only and is independent of the verbosity. The
    environment variable ``PIANISTPY_TIMING`` sets it at import.
    """
    global _timing_on  # noqa: PLW0603
    _timing_on = bool(enabled)


def timing_enabled() -> bool:
    """Return whether stage times are printed."""
    return _timing_on


@contextlib.contextmanager
def timing_record() -> Iterator[Timing]:
    """Collect the stage times of the enclosed computation.

    Yields
    ------
    timing : Timing
        The record that every `timed` stage of the block adds to.
    """
    record = Timing()
    token = _current_timing.set(record)
    try:
        yield record
    finally:
        _current_timing.reset(token)
        if _timing_on:
            for name, seconds in record.stages.items():
                peak = record.memory.get(name)
                memory = "" if peak is None else f" {peak / 2**20:10.1f} MiB peak"
                print(f"pianistpy timing: {name:<28s} {seconds:10.4f} s{memory}")
            print(f"pianistpy timing: {'total':<28s} {record.total:10.4f} s")


@contextlib.contextmanager
def timed(stage: str) -> Iterator[None]:
    """Time one stage and announce it.

    Parameters
    ----------
    stage : str
        Name of the stage, from the fixed list of its computation.
    """
    log.info("Stage %s started.", stage)
    start = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - start
        record = _current_timing.get()
        if record is not None:
            record.stages[stage] = record.stages.get(stage, 0.0) + seconds
            peak = peak_memory()
            if peak is not None:
                record.memory[stage] = peak
        log.debug("Stage %s took %.4f s.", stage, seconds)


def allowed_cores() -> int:
    """Return the number of cores this process is allowed to use.

    Inside a batch job the scheduler's own statement of the job's cores is
    taken where it gives one; otherwise the cores the operating system lets
    the process run on; otherwise the cores of the machine.
    """
    for name in SCHEDULER_CORE_VARIABLES:
        value = os.environ.get(name, "")
        if value.isdigit() and int(value) > 0:
            return int(value)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _in_scheduler_job() -> bool:
    return any(name in os.environ for name in SCHEDULER_JOB_VARIABLES)


_workers: str | int = "auto"


def set_workers(value: str | int) -> None:
    """Set how many cores the package uses for parallel work.

    Parameters
    ----------
    value : {"auto", "full", "half"} or int
        ``"auto"``, the default, uses every allowed core inside a batch job
        and half of them anywhere else; ``"full"`` every allowed core;
        ``"half"`` half of them; an integer exactly that many.

    Notes
    -----
    The number of cores never changes a number: a result is the same, to
    the bit, for any value. The environment variable ``PIANISTPY_WORKERS``
    sets the value at import.
    """
    global _workers  # noqa: PLW0603
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not (
        value in ("auto", "full", "half") or (isinstance(value, int) and value >= 1)
    ):
        raise InputError(
            "The number of cores is 'auto', 'full', 'half' or a positive integer; "
            f"got {value!r}."
        )
    _workers = value


def workers_explicit() -> bool:
    """Whether the number of cores was set exactly (`set_workers`)."""
    return isinstance(_workers, int)


def card_memory() -> tuple[int, int] | None:
    """Return the free and the total memory of the GPU JAX runs on, in
    bytes, as the CUDA driver states them; None off a CUDA GPU.

    The driver is asked in the device's primary context, the one JAX uses,
    so asking makes no context of its own. Its free memory counts what
    every process holds on the GPU, a CUDA context included, which the
    allocator's own statistics do not.
    """
    import ctypes

    if platform() != "gpu":
        return None
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    if cu.cuInit(0) or cu.cuDeviceGet(ctypes.byref(dev), 0):
        return None
    if cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev):
        return None
    try:
        if cu.cuCtxPushCurrent_v2(ctx):
            return None
        status = cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
    finally:
        cu.cuDevicePrimaryCtxRelease_v2(dev)
    return None if status else (free.value, total.value)


def get_workers() -> int:
    """Return the number of cores the current setting resolves to."""
    if isinstance(_workers, int):
        return _workers
    cores = allowed_cores()
    if _workers == "full" or (_workers == "auto" and _in_scheduler_job()):
        return cores
    return max(1, cores // 2)


# A worker process of a pool is the parallelism, so its BLAS and OpenMP run
# one thread (with every BLAS thread, each worker commits gigabytes of memory
# at import). A worker runs on the CPU whatever the main process's platform:
# the pools serve the observables, where a GPU is faster nowhere (the device
# table on the install page), and a GPU in Exclusive_Process mode takes a
# second process only through an MPS server. The variables are read when a
# worker imports numpy and the package, so they are set in the environment
# for the spawn and restored after it.
WORKER_VARIABLES = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JAX_PLATFORMS": "cpu",
    PLATFORM_VARIABLE: "cpu",
}


# XLA's CPU threads in a worker: its flags are read once, when the worker's
# JAX starts, so they are set for the spawn too; added to the flags the
# process already has, which they do not otherwise change
WORKER_XLA_FLAGS = (
    "--xla_cpu_multi_thread_eigen=false",
    "intra_op_parallelism_threads=1",
)


def _worker_xla_flags(flags: str | None) -> str:
    """The XLA flags of a worker: `flags` with the one-thread flags added."""
    present = (flags or "").split()
    return " ".join(present + [f for f in WORKER_XLA_FLAGS if f not in present])


@contextlib.contextmanager
def worker_environment() -> Iterator[None]:
    """Hold the workers' variables (one thread, the CPU, and one XLA thread)
    while a pool's workers are spawned, then restore them."""
    saved = {k: os.environ.get(k) for k in (*WORKER_VARIABLES, "XLA_FLAGS")}
    os.environ.update(WORKER_VARIABLES)
    os.environ["XLA_FLAGS"] = _worker_xla_flags(saved["XLA_FLAGS"])
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# The words a switch in the environment reads, in any case; unset is off.
_ON = ("1", "true", "yes", "on")
_OFF = ("0", "false", "no", "off")


def _env_flag(name: str) -> bool:
    """Read the switch the environment variable `name` sets."""
    value = os.environ.get(name, "").strip().lower()
    if value in _ON:
        return True
    if value in ("", *_OFF):
        return False
    raise InputError(
        f"{name} is one of {list(_ON + _OFF)}, in any case; got {os.environ[name]!r}."
    )


def _env_word(name: str, default: str) -> str:
    """Read the value the environment variable `name` names, in lower case."""
    return os.environ.get(name, "").strip().lower() or default


set_verbosity(_env_word(VERBOSITY_VARIABLE, "quiet"))
set_timing(_env_flag(TIMING_VARIABLE))
set_workers(_env_word(WORKERS_VARIABLE, "auto"))


COMPILATION_CACHE_VARIABLE = "PIANISTPY_COMPILATION_CACHE"


def _cache_off() -> bool:
    """Whether ``PIANISTPY_COMPILATION_CACHE`` turns the cache off."""
    return os.environ.get(COMPILATION_CACHE_VARIABLE, "").strip().lower() in _OFF


def _package_cache_folder() -> str:
    """The package's own folder of the compilation cache."""
    return os.path.join(os.path.expanduser("~"), ".cache", "pianistpy", "jax")


def default_compilation_cache() -> str | None:
    """Return the default folder of the compilation cache.

    The environment variable ``PIANISTPY_COMPILATION_CACHE`` if it is set,
    None if it is set to ``off`` (the cache is off), otherwise
    ``~/.cache/pianistpy/jax``.
    """
    if _cache_off():
        return None
    given = os.environ.get(COMPILATION_CACHE_VARIABLE)
    if given:
        return given
    return _package_cache_folder()


def set_compilation_cache(
    cache: bool | str | os.PathLike = True,
    max_size: int | None = None,
) -> str | None:
    """Turn the persistent compilation cache of JAX on or off.

    With the cache on, the code compiled in one run is reused by the next,
    which shortens every run after the first. It changes no number. The
    cache is on by default, its folder unbounded in size, as JAX's own is.

    A bound costs time: JAX then takes a lock on the folder at every read
    and write, and scans every entry at every write. Measured on a folder of
    22,000 entries on a network file system, a test that takes 5.6 minutes
    with no bound (4.3 to 5.2 minutes with eight runs at once on the folder,
    as with the cache off) took 2 hours 8 minutes alone with a bound of
    512 MiB.

    Parameters
    ----------
    cache : bool or path, optional
        True for the default folder (`default_compilation_cache`, or the
        package's own folder where the environment variable turns the cache
        off), a path for another folder, False to turn the cache off.
    max_size : int or None, optional
        Bound on the folder in bytes, above which JAX evicts the oldest
        entries; None, the default, leaves the folder unbounded.

    Returns
    -------
    folder : str or None
        The folder in use, or None if the cache is off.
    """
    import jax

    if cache is False:
        folder = None
    elif cache is True:
        folder = default_compilation_cache() or _package_cache_folder()
    else:
        folder = os.fspath(cache)
    jax.config.update("jax_compilation_cache_dir", folder)
    if folder is not None:
        # every compiled function is kept, however short its compilation
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update(
            "jax_compilation_cache_max_size", -1 if max_size is None else int(max_size)
        )
    log.debug("Compilation cache: %s.", folder)
    return folder


def _cache_file(kind: str, name: str) -> str | None:
    """The file of one cached text: in the folder ``kind`` beside the
    compilation cache's, or None where that cache is off."""
    folder = default_compilation_cache()
    if folder is None:
        return None
    root = os.path.dirname(folder)
    return os.path.join(root, kind, name)


def read_cache(kind: str, name: str) -> str | None:
    """Return a text kept by `write_cache`, or None where there is none.

    Parameters
    ----------
    kind : str
        The folder of the cache, beside the compilation cache's.
    name : str
        The file's name.
    """
    path = _cache_file(kind, name)
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as file:
            return file.read()
    except OSError:
        return None


def write_cache(kind: str, name: str, text: str) -> None:
    """Keep a text for a later run, written whole or not at all; a folder
    that cannot be written, or the compilation cache off, keeps nothing.

    Parameters
    ----------
    kind : str
        The folder of the cache, beside the compilation cache's.
    name : str
        The file's name.
    text : str
        What is kept.
    """
    import tempfile

    path = _cache_file(kind, name)
    if path is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=os.path.dirname(path),
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as file:
            file.write(text)
        os.replace(file.name, path)
    except OSError:
        pass


def set_platform(name: str) -> None:
    """Choose the hardware JAX runs on, before the first computation.

    The CPU-only mode (``"cpu"``) is first class: every function of the
    package runs on it and gives the same numbers on every machine. It
    changes no number on a machine without a GPU; on one with a GPU the
    CPU and GPU results differ at rounding, as the provenance of each
    file records (`pianistpy.io.backend_provenance`).

    Parameters
    ----------
    name : {"auto", "cpu", "gpu"}
        ``"auto"`` leaves JAX its own order (a GPU where one is usable);
        ``"cpu"`` is the CPU-only mode; ``"gpu"`` asks for the CUDA backend
        with the CPU behind it. Any other string is handed to JAX as its
        own platform list (for example ``"rocm,cpu"``).

    Notes
    -----
    JAX picks its backend at the first computation of the process and
    keeps it: call this before any result is computed, or set the
    environment variable ``PIANISTPY_PLATFORM`` before the package is
    imported. A later call has no effect and is reported at ``info``.
    """
    import jax

    if not isinstance(name, str) or not name:
        raise InputError(f"platform must be a name; got {name!r}.")
    platforms = PLATFORMS.get(name.lower(), name)
    before = getattr(jax.config, "jax_platforms", "") or None
    jax.config.update("jax_platforms", platforms or "")
    log.info("Platform: %s (JAX platforms %r, before %r).", name, platforms, before)


def platform() -> str:
    """Return the backend JAX runs on: ``"cpu"``, ``"gpu"`` or another.

    Asking initializes the backend if it is not yet.
    """
    import jax

    return str(jax.default_backend())


def configure_jax() -> None:
    """Set JAX as the package needs it: 64 bits, the platform and the
    compilation cache.

    Called once when the package is imported. The environment variable
    ``PIANISTPY_PLATFORM`` names the platform (`set_platform`), and
    ``PIANISTPY_COMPILATION_CACHE`` set to ``off`` leaves the cache off.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    wanted = _env_word(PLATFORM_VARIABLE, "")
    if wanted:
        set_platform(wanted)
    set_compilation_cache(not _cache_off())
