"""Options of the Python session: the messages printed, the stage times, the
number of cores, the hardware JAX runs on, and JAX's compilation cache.

A user sets each option once for the session, by a call such as
``pp.set_verbosity("info")`` or ``pp.set_cores(8)``, or by an environment
variable read when the package is imported: ``PIANISTPY_VERBOSITY``,
``PIANISTPY_TIMING``, ``PIANISTPY_CORES``, ``PIANISTPY_PLATFORM`` and
``PIANISTPY_COMPILATION_CACHE``. No option changes a computed number, and no
option is part of a result's identity. No computation of the package takes
a runtime option as an argument.
"""

__all__ = [
    "Timing",
    "allowed_cores",
    "card_memory",
    "configure_jax",
    "cores_explicit",
    "default_compilation_cache",
    "get_cores",
    "log",
    "log_to_file",
    "peak_memory",
    "recent_messages",
    "platform",
    "set_compilation_cache",
    "set_cores",
    "set_platform",
    "set_timing",
    "set_verbosity",
    "timed",
    "timing_enabled",
    "timing_record",
]

import contextlib
import contextvars
import logging
import os
import pickle
import shutil
import sys
import tempfile
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from multiprocessing.reduction import ForkingPickler
from typing import Any

from .exceptions import InputError

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    import resource

# Each verbosity level mapped to the lowest logging level printed on the
# screen at that verbosity.
VERBOSITY_LEVELS = {
    "quiet": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
VERBOSITY_VARIABLE = "PIANISTPY_VERBOSITY"
TIMING_VARIABLE = "PIANISTPY_TIMING"
CORES_VARIABLE = "PIANISTPY_CORES"
PLATFORM_VARIABLE = "PIANISTPY_PLATFORM"
# Each platform name of set_platform mapped to JAX's platform list: "auto"
# maps to None, which leaves JAX's own order (a usable GPU first, the CPU
# otherwise); "cpu" maps to the CPU alone, which gives the same numbers on
# every machine; "gpu" maps to the CUDA backend first and the CPU second.
PLATFORMS = {"auto": None, "cpu": "cpu", "gpu": "cuda,cpu"}
# The number of most recent messages the package keeps at every verbosity,
# so that recent_messages can return the messages that led to an error.
RECENT_MESSAGES_MAX = 50
# Environment variables that batch schedulers set inside a job, one per
# scheduler the package knows; any one of them set means the process runs in
# a batch job.
SCHEDULER_JOB_VARIABLES = (
    "PBS_JOBID",
    "SLURM_JOB_ID",
    "LSB_JOBID",
    "JOB_ID",
    "COBALT_JOBID",
)
# Environment variables in which a batch scheduler states the number of cores
# of the job.
SCHEDULER_CORE_VARIABLES = ("SLURM_CPUS_PER_TASK", "PBS_NP", "NCPUS", "NSLOTS")

# The package's logger. Every message goes to the recent-message handler and
# to the screen handler; the logger does not pass messages on to the root
# logger, so that the package's verbosity alone decides what is printed.
log = logging.getLogger("pianistpy")
log.setLevel(logging.DEBUG)
log.propagate = False


class _RecentHandler(logging.Handler):
    """A logging handler that keeps the RECENT_MESSAGES_MAX most recent
    messages of every level."""

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
    """Set which of the package's messages are printed on the screen.

    The verbosity holds for the rest of the session. The verbosity changes
    no computed number. The environment variable ``PIANISTPY_VERBOSITY``
    sets the verbosity when the package is imported; with the variable
    unset, the verbosity is ``"quiet"``.

    Parameters
    ----------
    level : {"quiet", "info", "debug"}
        ``"quiet"`` prints warnings and errors only. ``"info"`` also prints
        one line when each stage of a computation starts, and the messages
        of results with relative error above :math:`10^{-6}` and at most
        :math:`10^{-4}`. ``"debug"`` also prints the details of each stage.

    Raises
    ------
    InputError
        If `level` is not ``"quiet"``, ``"info"`` or ``"debug"``.

    See Also
    --------
    log_to_file : Write the messages of a block to a file.
    recent_messages : The most recent messages at every verbosity.

    Examples
    --------
    >>> set_verbosity("info")
    >>> set_verbosity("quiet")
    """
    if level not in VERBOSITY_LEVELS:
        raise InputError(
            f"The verbosity must be one of {list(VERBOSITY_LEVELS)}; the value "
            f"given is {level!r}."
        )
    _screen.setLevel(VERBOSITY_LEVELS[level])


def recent_messages() -> tuple[str, ...]:
    """Return the package's most recent messages, oldest first.

    The package keeps its 50 most recent messages at every verbosity, so
    that the messages that led to an error can be read after a run at the
    verbosity ``"quiet"``.

    Returns
    -------
    messages : tuple of str
        Each message as printed: the logger's name, the message's level and
        the message text.

    See Also
    --------
    set_verbosity : Choose the messages printed on the screen.

    Examples
    --------
    >>> log.debug("A message kept at every verbosity.")
    >>> recent_messages()[-1]
    'pianistpy DEBUG: A message kept at every verbosity.'
    """
    return tuple(_recent.messages)


@contextlib.contextmanager
def log_to_file(path: str | os.PathLike, level: str = "debug") -> Iterator[None]:
    """Write the package's messages during a ``with`` block to a file.

    The file receives the messages in addition to the screen; the screen
    keeps the session's verbosity.

    Parameters
    ----------
    path : str or path-like
        The file the messages are appended to. The file is created if
        `path` does not exist.
    level : {"quiet", "info", "debug"}, optional
        The verbosity of the file, with the meanings of `set_verbosity`.
        Default: ``"debug"``, which means every message is written.

    Raises
    ------
    InputError
        If `level` is not ``"quiet"``, ``"info"`` or ``"debug"``.

    See Also
    --------
    set_verbosity : Choose the messages printed on the screen.

    Examples
    --------
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), "run.log")
    >>> with log_to_file(path):
    ...     log.debug("A message for the file.")
    >>> open(path).read()
    'pianistpy DEBUG: A message for the file.\\n'
    """
    if level not in VERBOSITY_LEVELS:
        raise InputError(
            f"The verbosity must be one of {list(VERBOSITY_LEVELS)}; the value "
            f"given is {level!r}."
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
        """The process memory counters of the Windows API
        (PROCESS_MEMORY_COUNTERS)."""

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
    """Return the largest resident memory of the Python process, in bytes.

    The value counts the memory of the Python process on the host. Memory
    held on a GPU is not counted.

    Returns
    -------
    peak : int or None
        The largest resident memory since the process started, in bytes;
        None if the operating system does not report the value.

    See Also
    --------
    Timing : The stage times and memory of one computation.

    Examples
    --------
    >>> peak = peak_memory()
    >>> peak is None or peak > 0
    True
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
        # Linux reports ru_maxrss in kilobytes and macOS in bytes.
        return int(peak) if sys.platform == "darwin" else int(peak) * 1024


@dataclass
class Timing:
    """The elapsed time of each stage of one computation, and the largest
    memory of the process at the end of each stage.

    Parameters
    ----------
    stages : dict of str to float, optional
        Each stage's name mapped to the seconds spent in the stage, in the
        order in which the stages first ran. A stage that runs more than
        once has the sum of its times. Default: an empty dictionary.
    memory : dict of str to int, optional
        Each stage's name mapped to the largest resident memory of the
        process, in bytes, at the last end of the stage (`peak_memory`).
        The largest memory never decreases from one stage to the next, so
        a stage raised the largest memory if the stage's value exceeds the
        value of the stage before. Default: an empty dictionary.

    Attributes
    ----------
    total : float
        The sum of the stage times, in seconds.

    See Also
    --------
    timing_record : Collect a `Timing` for a block of code.
    set_timing : Print each result's stage times.

    Examples
    --------
    >>> timing = Timing(stages={"inversion": 1.5, "truncation": 0.25})
    >>> timing.total
    1.75
    """

    stages: dict[str, float] = field(default_factory=dict)
    memory: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """The sum of the stage times, in seconds."""
        return sum(self.stages.values())

    def to_dict(self) -> dict[str, float]:
        """Return the stage times as a plain dictionary.

        Returns
        -------
        stages : dict of str to float
            Each stage's name mapped to the seconds spent in the stage.

        Examples
        --------
        >>> Timing(stages={"inversion": 1.5}).to_dict()
        {'inversion': 1.5}
        """
        return dict(self.stages)


_timing_on = False
_current_timing: contextvars.ContextVar[Timing | None] = contextvars.ContextVar(
    "pianistpy_timing", default=None
)


def set_timing(enabled: bool) -> None:
    """Turn the printing of each computation's stage times on or off.

    The package measures the stage times of every computation and keeps
    the times with the result, whether the printing is on or off. The
    printing does not depend on the verbosity. The environment variable
    ``PIANISTPY_TIMING`` turns the printing on when the package is imported
    if the variable is ``1``, ``true``, ``yes`` or ``on``.

    Parameters
    ----------
    enabled : bool
        Whether the stage times are printed at the end of each computation.

    See Also
    --------
    timing_enabled : Whether the stage times are printed.
    Timing : The stage times of one computation.

    Examples
    --------
    >>> set_timing(True)
    >>> timing_enabled()
    True
    >>> set_timing(False)
    """
    global _timing_on  # noqa: PLW0603
    _timing_on = bool(enabled)


def timing_enabled() -> bool:
    """Return whether the stage times are printed at the end of each
    computation.

    Returns
    -------
    enabled : bool
        The value last given to `set_timing`.

    See Also
    --------
    set_timing : Turn the printing of stage times on or off.

    Examples
    --------
    >>> isinstance(timing_enabled(), bool)
    True
    """
    return _timing_on


@contextlib.contextmanager
def timing_record() -> Iterator[Timing]:
    """Collect the stage times of the computation in a ``with`` block.

    With the printing of stage times on (`set_timing`), the times are
    printed when the block ends.

    Yields
    ------
    timing : Timing
        The record to which each `timed` stage inside the block adds its
        time and memory.

    See Also
    --------
    timed : Time one stage.
    Timing : The stage times of one computation.

    Examples
    --------
    >>> with timing_record() as record:
    ...     with timed("example"):
    ...         pass
    >>> list(record.stages)
    ['example']
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
    """Time one stage of a computation in a ``with`` block.

    The stage's start is written to the log at level INFO and the stage's
    time at level DEBUG. Inside a `timing_record` block, the stage's time
    and the process's largest memory are added to the record.

    Parameters
    ----------
    stage : str
        The stage's name.

    See Also
    --------
    timing_record : Collect the stage times of a block.

    Examples
    --------
    >>> with timed("example"):
    ...     pass
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
    """Return the number of cores the Python process may use.

    The number is the first one available of three: the job's number of
    cores as a batch scheduler states the number in its environment
    variable; the number of cores the operating system lets the process run
    on; the number of cores of the machine.

    Returns
    -------
    cores : int
        The number of cores, at least 1.

    See Also
    --------
    set_cores : Choose the number of cores the package uses.
    get_cores : The number of cores the package uses.

    Examples
    --------
    >>> allowed_cores() >= 1
    True
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


_cores: str | int = "auto"


def set_cores(value: str | int) -> None:
    """Set the number of cores the package uses for parallel computations.

    The number of cores changes no computed number: a result is the same,
    bit for bit, for every value. Each parallel process of the package runs
    one thread, so the number of cores is also the largest number of
    parallel processes. The environment variable ``PIANISTPY_CORES`` sets
    the value when the package is imported; with the variable unset, the
    value is ``"auto"``.

    Parameters
    ----------
    value : {"auto", "full", "half"} or int
        ``"auto"`` uses every core `allowed_cores` counts inside a batch job
        and half of those cores, at least 1, elsewhere. ``"full"`` uses
        every core `allowed_cores` counts. ``"half"`` uses half of those
        cores, at least 1. A positive integer, or a string of digits, uses
        that number of cores.

    Raises
    ------
    InputError
        If `value` is none of ``"auto"``, ``"full"``, ``"half"`` and a
        positive integer.

    See Also
    --------
    get_cores : The number of cores the package uses.
    allowed_cores : The number of cores the process may use.

    Examples
    --------
    >>> set_cores(2)
    >>> get_cores()
    2
    >>> set_cores("auto")
    """
    global _cores  # noqa: PLW0603
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not (
        value in ("auto", "full", "half") or (isinstance(value, int) and value >= 1)
    ):
        raise InputError(
            "The number of cores must be 'auto', 'full', 'half' or a positive "
            f"integer; the value given is {value!r}."
        )
    _cores = value


def cores_explicit() -> bool:
    """Return whether the number of cores was set as an integer.

    Returns
    -------
    explicit : bool
        True if the last call of `set_cores`, or ``PIANISTPY_CORES``, gave
        an integer; False if the value is ``"auto"``, ``"full"`` or
        ``"half"``.

    See Also
    --------
    set_cores : Choose the number of cores the package uses.

    Examples
    --------
    >>> set_cores(2)
    >>> cores_explicit()
    True
    >>> set_cores("auto")
    >>> cores_explicit()
    False
    """
    return isinstance(_cores, int)


def card_memory() -> tuple[int, int] | None:
    """Return the free memory and the total memory of the GPU that JAX
    runs on, in bytes, as the CUDA driver states them.

    The free memory counts the memory that every process holds on the GPU,
    including each process's CUDA context; JAX's allocator statistics do
    not count the contexts. The call uses the GPU's primary context, which
    JAX also uses, so the call creates no CUDA context.

    Returns
    -------
    memory : tuple of (int, int) or None
        The free memory and the total memory, in bytes; None if JAX does
        not run on a GPU, if the CUDA driver library cannot be loaded, or
        if the driver reports an error.

    See Also
    --------
    platform : The hardware JAX runs on.
    peak_memory : The largest memory of the process on the host.

    Examples
    --------
    >>> memory = card_memory()
    >>> memory is None or memory[0] <= memory[1]
    True
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


def get_cores() -> int:
    """Return the number of cores the package uses.

    Returns
    -------
    cores : int
        The integer last given to `set_cores`; for ``"full"``, the count of
        `allowed_cores`; for ``"half"``, half of that count, at least 1;
        for ``"auto"``, the count of `allowed_cores` inside a batch job and
        half of that count, at least 1, elsewhere.

    See Also
    --------
    set_cores : Choose the number of cores the package uses.

    Examples
    --------
    >>> set_cores(3)
    >>> get_cores()
    3
    >>> set_cores("auto")
    """
    if isinstance(_cores, int):
        return _cores
    cores = allowed_cores()
    if _cores == "full" or (_cores == "auto" and _in_scheduler_job()):
        return cores
    return max(1, cores // 2)


# The environment of each parallel process the package starts. The parallel
# processes are the parallelism, so BLAS and OpenMP in each process run one
# thread: with one BLAS thread per core, each process would reserve gigabytes
# of memory when numpy is imported. Each parallel process runs on the CPU
# whatever the main process's platform, because the parallel processes compute
# the observables, which run no faster on a GPU, and because a GPU in
# Exclusive_Process mode accepts a second process only through an MPS server.
# A parallel process reads the variables when the process imports numpy and
# the package, so worker_environment sets the variables before the processes
# start and restores them afterwards.
WORKER_VARIABLES = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JAX_PLATFORMS": "cpu",
    PLATFORM_VARIABLE: "cpu",
}


# The XLA flags that limit XLA's CPU computations in a parallel process to
# one thread. XLA reads its flags once, when JAX starts in the process, so the
# flags are set before the parallel processes start; the flags are added to
# the flags already in XLA_FLAGS, which keep their values.
WORKER_XLA_FLAGS = (
    "--xla_cpu_multi_thread_eigen=false",
    "intra_op_parallelism_threads=1",
)


def _worker_xla_flags(flags: str | None) -> str:
    """Return the XLA flags of a parallel process: the flags in `flags`
    followed by each flag of WORKER_XLA_FLAGS that `flags` does not hold."""
    present = (flags or "").split()
    return " ".join(present + [f for f in WORKER_XLA_FLAGS if f not in present])


@contextlib.contextmanager
def worker_environment() -> Iterator[None]:
    """Set the environment variables of WORKER_VARIABLES and the one-thread
    XLA flags while a ``with`` block starts parallel processes, and restore
    every variable's former value when the block ends."""
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


def spill(obj: Any, name: str) -> tuple[str, str]:
    """Write `obj` once to a file that parallel processes read.

    The object is pickled as the start of a parallel process pickles its
    arguments (`ForkingPickler`), into the file ``<name>.pickle`` in a new
    folder under the temporary directory that `tempfile` chooses (the
    directory TMPDIR gives, if set). A parallel process reads the object
    with `unspill`, and `drop_spill` removes the folder.

    Returns
    -------
    path : str
        The path of the file.
    folder : str
        The path of the new folder that holds the file.
    """
    folder = tempfile.mkdtemp(prefix=f"pianistpy-{name}-")
    path = os.path.join(folder, f"{name}.pickle")
    try:
        with open(path, "wb") as f:
            f.write(ForkingPickler.dumps(obj))
    except BaseException:
        drop_spill(folder)
        raise
    return path, folder


def unspill(path: str) -> Any:
    """Return the object that `spill` wrote to the file at `path`."""
    with open(path, "rb") as f:
        return pickle.load(f)


def drop_spill(folder: str) -> None:
    """Remove a folder that `spill` made, with the folder's files; a folder
    that no longer exists is not an error."""
    shutil.rmtree(folder, ignore_errors=True)


# The values, in any letter case, that turn a switch set by an environment
# variable on (_ON) or off (_OFF); an unset variable leaves the switch off.
_ON = ("1", "true", "yes", "on")
_OFF = ("0", "false", "no", "off")


def _env_flag(name: str) -> bool:
    """Return whether the environment variable `name` turns its switch on.

    Raises
    ------
    InputError
        If the variable is set to a value in neither _ON nor _OFF.
    """
    value = os.environ.get(name, "").strip().lower()
    if value in _ON:
        return True
    if value in ("", *_OFF):
        return False
    raise InputError(
        f"The environment variable {name} must be one of {list(_ON + _OFF)}, in "
        f"any letter case; the value set is {os.environ[name]!r}."
    )


def _env_word(name: str, default: str) -> str:
    """Return the value of the environment variable `name` in lower case,
    without surrounding spaces, or `default` if the variable is unset or
    empty."""
    return os.environ.get(name, "").strip().lower() or default


# The runtime options that the environment variables set when the package is
# imported.
set_verbosity(_env_word(VERBOSITY_VARIABLE, "quiet"))
set_timing(_env_flag(TIMING_VARIABLE))
if os.environ.get(CORES_VARIABLE, "").strip():
    set_cores(_env_word(CORES_VARIABLE, "auto"))
else:
    set_cores("auto")


# The environment variable that gives the compilation cache's folder, or
# turns the cache off with a value of _OFF.
COMPILATION_CACHE_VARIABLE = "PIANISTPY_COMPILATION_CACHE"


def _cache_off() -> bool:
    """Return whether ``PIANISTPY_COMPILATION_CACHE`` turns the
    compilation cache off."""
    return os.environ.get(COMPILATION_CACHE_VARIABLE, "").strip().lower() in _OFF


def _package_cache_folder() -> str:
    """Return the package's folder of the compilation cache,
    ``~/.cache/pianistpy/jax``."""
    return os.path.join(os.path.expanduser("~"), ".cache", "pianistpy", "jax")


def default_compilation_cache() -> str | None:
    """Return the folder that the compilation cache uses by default.

    Returns
    -------
    folder : str or None
        None if the environment variable ``PIANISTPY_COMPILATION_CACHE`` is
        ``0``, ``false``, ``no`` or ``off``, which turns the cache off; the
        variable's value, a folder's path, if the variable has any other
        value;
        ``~/.cache/pianistpy/jax`` if the variable is unset.

    See Also
    --------
    set_compilation_cache : Turn the compilation cache on or off.

    Examples
    --------
    >>> folder = default_compilation_cache()
    >>> folder is None or isinstance(folder, str)
    True
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
    """Turn JAX's persistent compilation cache on or off.

    With the cache on, a later Python session reuses the code that an
    earlier session compiled, which shortens every session after the first.
    The cache changes no computed number. When the package is imported,
    the cache is on, in the folder of `default_compilation_cache`, with no
    bound on the folder's size, unless ``PIANISTPY_COMPILATION_CACHE``
    turns the cache off.

    A bound on the folder's size slows every compilation: with a bound,
    JAX locks the folder at every read and every write, and scans every
    entry of the folder at every write. On a shared or network file
    system, keep the folder unbounded.

    Parameters
    ----------
    cache : bool or str or path-like, optional
        True for the folder of `default_compilation_cache`, or for
        ``~/.cache/pianistpy/jax`` if the environment variable turns the
        cache off; a path for another folder; False to turn the cache off.
        Default: ``True``.
    max_size : int or None, optional
        The largest size of the folder, in bytes; above `max_size`, JAX
        removes the oldest entries. Default: ``None``, which means the
        folder has no bound.

    Returns
    -------
    folder : str or None
        The folder the cache uses; None if the cache is off.

    See Also
    --------
    default_compilation_cache : The folder the cache uses by default.

    Examples
    --------
    >>> on = default_compilation_cache() is not None
    >>> set_compilation_cache(on) == default_compilation_cache()
    True
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
        # Every compiled function is kept in the cache, however short the
        # function's compilation and however small the function's entry.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update(
            "jax_compilation_cache_max_size", -1 if max_size is None else int(max_size)
        )
    log.debug("Compilation cache folder: %s.", folder)
    return folder


def _cache_file(kind: str, name: str) -> str | None:
    """Return the path of one cached text: the file `name` in the folder
    `kind` beside the compilation cache's folder; None if the compilation
    cache is off."""
    folder = default_compilation_cache()
    if folder is None:
        return None
    root = os.path.dirname(folder)
    return os.path.join(root, kind, name)


def read_cache(kind: str, name: str) -> str | None:
    """Return a text that `write_cache` kept for a later session.

    Parameters
    ----------
    kind : str
        The name of the text's folder, beside the compilation cache's
        folder.
    name : str
        The name of the text's file.

    Returns
    -------
    text : str or None
        The text; None if the compilation cache is off or the file cannot
        be read.
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
    """Keep a text for a later session, beside the compilation cache.

    The text is written to a temporary file that then replaces the file
    `name`, so a reader sees either the whole text or the former file.
    Nothing is written if the compilation cache is off or the folder
    cannot be written.

    Parameters
    ----------
    kind : str
        The name of the text's folder, beside the compilation cache's
        folder.
    name : str
        The name of the text's file.
    text : str
        The text to keep.
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
    """Choose the hardware JAX runs on, before the session's first
    computation.

    Every function of the package runs on the CPU alone (``"cpu"``), and
    the CPU alone gives the same numbers on every machine. On a machine
    with a GPU, results computed on the GPU and results computed on the CPU
    differ at the level of rounding; each result file records the hardware
    that computed the result.

    JAX chooses its hardware at the first computation of the Python process
    and keeps the choice for the rest of the process. A call of
    `set_platform` after the first computation does not change the
    hardware. To choose the hardware before the package is imported, set
    the environment variable ``PIANISTPY_PLATFORM``.

    Parameters
    ----------
    name : str
        ``"auto"`` leaves JAX's own order: a usable GPU first, the CPU
        otherwise. ``"cpu"`` runs on the CPU alone. ``"gpu"`` runs on a
        CUDA GPU first and on the CPU second. Any other string is passed to
        JAX as JAX's platform list, for example ``"rocm,cpu"``.

    Raises
    ------
    InputError
        If `name` is not a non-empty string.

    See Also
    --------
    platform : The hardware JAX runs on.

    Examples
    --------
    >>> set_platform("")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    pianistpy.exceptions.InputError: ...
    """
    import jax

    if not isinstance(name, str) or not name:
        raise InputError(
            "set_platform takes a platform name, a non-empty string; the value "
            f"given is {name!r}."
        )
    platforms = PLATFORMS.get(name.lower(), name)
    before = getattr(jax.config, "jax_platforms", "") or None
    jax.config.update("jax_platforms", platforms or "")
    log.info(
        "Platform %s: JAX's platform list is %r; before the call, the list was %r.",
        name,
        platforms,
        before,
    )


def platform() -> str:
    """Return the hardware JAX runs on.

    A call of `platform` makes JAX choose its hardware if no computation
    has chosen the hardware yet.

    Returns
    -------
    name : str
        JAX's name of the hardware: ``"cpu"``, ``"gpu"``, or another name
        that JAX reports.

    See Also
    --------
    set_platform : Choose the hardware JAX runs on.

    Examples
    --------
    >>> isinstance(platform(), str)
    True
    """
    import jax

    return str(jax.default_backend())


def configure_jax() -> None:
    """Configure JAX for the package: 64-bit floating point, the hardware,
    and the compilation cache.

    The package calls `configure_jax` once, when the package is imported.
    The environment variable ``PIANISTPY_PLATFORM``, if set, chooses the
    hardware (`set_platform`). The compilation cache is turned on in the
    folder of `default_compilation_cache`, unless
    ``PIANISTPY_COMPILATION_CACHE`` turns the cache off.

    See Also
    --------
    set_platform : Choose the hardware JAX runs on.
    set_compilation_cache : Turn the compilation cache on or off.

    Examples
    --------
    >>> import jax
    >>> jax.config.jax_enable_x64
    True
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    wanted = _env_word(PLATFORM_VARIABLE, "")
    if wanted:
        set_platform(wanted)
    set_compilation_cache(not _cache_off())
