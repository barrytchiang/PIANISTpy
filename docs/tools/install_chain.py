"""The install page's representative chain, one leg: each stage's wall and peak memory.

The rows, in the install page's order (the paper's own chain): 1. an isotropic NFW
halo, inverted (compute_df); 2. truncated at K_t = 0.05 (Method I); 3. a child with a
constant anisotropy beta_c = 0.2 on its truncated profile (Method II: compute_df and
truncate); 4. the child's Sampler tables; 5. 1e7 particles; 6. the child's LosTable at
the default settings (its row count stated); 7. the observables by route: on the model
(the second-moment route, no table) sigma_r, sigma_t and sigma_los at 50 radii
log-spaced over 0.01 to 10 and sigma_los_aperture on a circle of radius 1; from the
table the chain builds, losvd at R = 1 with 201 velocities over -1.5 to 1.5,
aperture_losvd on the circle at those velocities, sigma_los_aperture again (the two
routes side by side),
the Gauss-Hermite moments at 10 radii log-spaced over 0.05 to 5, pdf_v at the 10 radii
at V = 0.3, the velocity distribution at r = 1, and Differentiable's sigma_los
gradient at R = 1. No row times a table build. Lengths and velocities are in the
model's internal
units (the concentration sets only the physical units, so it does not change a wall
time). Two of the rows use internal functions, which the package does not export: the
velocity distribution (`pianistpy.observables.stages.velocity_distribution`) and the
Gauss-Hermite moments (`gauss_hermite_table`).

Each stage is called once to warm (its compilation and caches), then three times, and
the median of the three is its wall time (every wall is kept, the warm call's first);
the LosTable is timed on its first build only. The leg's peak memory is that of this
process and of its largest worker process, each its own figure (VmHWM on Linux,
psutil's peak_wset on Windows), the reading named beside it. A column is labeled by its
hardware: the CPU's model, its threads, the memory, the operating system, the GPU or
"CPU only", and the cores.

Usage (one leg a run; two legs on one machine, one after the other):

    python install_chain.py --platform cpu --cores 28 --out cpu.json
    python install_chain.py --platform gpu --cores 28 --out gpu.json
    python install_chain.py --platform cpu --cores 12 --out third.json
    python install_chain.py --table cpu.json gpu.json third.json

The last form prints the legs as one table. --profile adds one more call of rows 7a to
7d under cProfile, after their timing (<out>.profile_7a.txt and so on). --repeats sets
the timed calls of rows 7a to 7d (default 3), whose serial moments solve runs on one
core; on Windows each of their calls records the logical processor it started and
ended on and that processor's efficiency class (two samples a call: a thread can move
between cores within a call). On Windows the
memory readings need psutil. Set the package's caches outside the home directory
first, for example PIANISTPY_COMPILATION_CACHE and XDG_CACHE_HOME. Each stage's line is
flushed as it is printed; where a shell still holds redirected output until the run
ends, run Python unbuffered (python -u).
"""

import argparse
import cProfile
import glob
import json
import os
import platform as host_platform
import pstats
import statistics
import sys
import threading
import time

# the stages a GPU leg does not velocity up, flagged "no GPU acceleration" in its column
# beside their number: rows 2 and 3 run their kernels on the GPU with no gain, and row
# 6's rows run on the CPU by the package's rule
NO_GPU_VELOCITYUP = (
    "2. Method I truncation (K_t 0.05)",
    "3. constant beta_c 0.2 child, Method II (compute_df, truncate)",
    "6. LosTable(child), first build",
)


def _own_peak_gib(pid: int | None = None) -> tuple[float, str] | None:
    """The process's own peak memory in GiB and what was read.

    VmHWM of /proc/<pid>/status on Linux, psutil's peak_wset on Windows; on another
    POSIX host the process's own ru_maxrss (pid None only), which on a spawned worker
    can carry its parent's peak at the fork; None if none can be read. Not ru_maxrss on
    Linux: there a spawned worker's ru_maxrss keeps its parent's peak across the exec,
    so every worker would read at least the main process's figure.
    """
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{'self' if pid is None else pid}/status") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 1024.0**2, "VmHWM"
        except OSError:
            return None
        return None
    if sys.platform == "win32":
        import psutil

        try:
            return psutil.Process(pid).memory_info().peak_wset / 1024.0**3, "peak_wset"
        except psutil.Error:
            return None
    if pid is None:
        import resource

        return (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**3,
            "ru_maxrss (a worker's can carry its parent's peak at the fork)",
        )
    return None


def _worker_peak(_: int) -> tuple[int, float, str]:
    """Run in a worker: its pid and its own peak, sent back to the main process."""
    time.sleep(0.2)  # so that each worker takes some of the calls
    peak, source = _own_peak_gib() or (float("nan"), "none")
    return os.getpid(), peak, source


def _children() -> list[int]:
    """The live children: from /proc on Linux, psutil on Windows, none elsewhere."""
    if sys.platform.startswith("linux"):
        pids = []
        for children in glob.glob(f"/proc/{os.getpid()}/task/*/children"):
            try:
                pids += [int(p) for p in open(children).read().split()]
            except OSError:
                continue
        return pids
    if sys.platform == "win32":
        import psutil

        return [child.pid for child in psutil.Process().children(recursive=True)]
    return []


class _ChildPeaks(threading.Thread):
    """Reads each live child's own peak every 0.25 s while the leg runs, for the pools
    the package closes itself (the LosTable build's); Linux and Windows."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peaks: dict[int, float] = {}
        self.sources: set[str] = set()
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.wait(0.25):
            for pid in _children():
                read = _own_peak_gib(pid)
                if read is not None:
                    self.peaks[pid] = max(read[0], self.peaks.get(pid, 0.0))
                    self.sources.add(read[1])


def _hardware(platform_name: str, devices: list) -> dict:
    """The leg's hardware, which labels its column (never the host's name): the CPU's
    model, its hardware threads, the memory, the operating system and, on a GPU leg,
    the GPU."""
    cpu = host_platform.processor() or host_platform.machine()
    memory = None
    if sys.platform.startswith("linux"):
        with open("/proc/cpuinfo") as handle:
            names = (
                line.split(":", 1)[1].strip()
                for line in handle
                if line.startswith("model name")
            )
            cpu = next(names, cpu)
        with open("/proc/meminfo") as handle:
            total = next(line for line in handle if line.startswith("MemTotal:"))
            memory = int(total.split()[1]) * 1024
    elif sys.platform == "win32":
        import winreg

        import psutil

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        memory = psutil.virtual_memory().total
    return {
        "cpu": cpu,
        "threads": os.cpu_count(),
        "memory_gb": None if memory is None else round(memory / 1024.0**3),
        "os": f"{host_platform.system()} {host_platform.release()}",
        "gpu": devices[0].device_kind if platform_name == "gpu" else None,
    }


def _label(leg: dict) -> str:
    """A leg's column header, from its hardware."""
    hw = leg.get("hardware")
    if hw is None:
        return f"{leg['platform']} ({leg['workers']} cores)"
    parts = [hw["cpu"], f"{hw['threads']} threads"]
    if hw["memory_gb"] is not None:
        parts.append(f"{hw['memory_gb']} GB")
    parts.append(hw["os"])
    parts.append(hw["gpu"] if hw["gpu"] else "CPU only")
    parts.append(f"{leg['workers']} cores")
    return ", ".join(parts)


def _efficiency_classes() -> dict[int, int] | None:
    """Windows: each logical processor's efficiency class (GetSystemCpuSetInformation;
    a higher class is a faster core); None elsewhere or if it cannot be read."""
    if sys.platform != "win32":
        return None
    import ctypes

    try:
        kernel32 = ctypes.WinDLL("kernel32")
        size = ctypes.c_ulong(0)
        kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
        buffer = ctypes.create_string_buffer(size.value)
        if not kernel32.GetSystemCpuSetInformation(
            buffer, size, ctypes.byref(size), None, 0
        ):
            return None
        raw, classes, at = buffer.raw, {}, 0
        while at < size.value:
            entry = int.from_bytes(raw[at : at + 4], "little")
            # SYSTEM_CPU_SET_INFORMATION: LogicalProcessorIndex at byte 14,
            # EfficiencyClass at byte 18
            classes[raw[at + 14]] = raw[at + 18]
            at += entry
        return classes
    except (AttributeError, OSError):
        return None


def _core(classes: dict[int, int] | None) -> list[int] | None:
    """Windows: the logical processor this thread is on and its efficiency class."""
    if classes is None:
        return None
    import ctypes

    cpu = ctypes.WinDLL("kernel32").GetCurrentProcessorNumber()
    return [cpu, classes.get(cpu, -1)]


def _timed(
    stages: dict,
    walls: dict,
    name: str,
    call,
    repeats: int = 3,
    cores: dict | None = None,
    classes: dict[int, int] | None = None,
) -> object:
    """Warm `call` once, then time it `repeats` times; keep the median, and every wall
    (the warm call's first). With `cores`, each call's processor and efficiency class
    at its start and its end."""
    if cores is not None:
        cores[name] = []
    for i in range(repeats + 1):
        first = _core(classes) if cores is not None else None
        start = time.perf_counter()
        out = call()
        wall = time.perf_counter() - start
        if cores is not None:
            cores[name].append([first, _core(classes)])
        if i == 0:
            walls[name] = [wall]
        else:
            walls[name].append(wall)
    stages[name] = statistics.median(walls[name][1:])
    every = ", ".join(f"{w:.4f}" for w in walls[name])
    print(
        f"{name}: {stages[name]:.4f} s (median of {repeats} after one warm call;"
        f" walls, warm first: {every})",
        flush=True,
    )
    if cores and classes is not None:
        print(f"{name}: [processor, class] at start and end: {cores[name]}", flush=True)
    return out


def run(
    platform_name: str,
    workers: int,
    out: str,
    profile: bool = False,
    repeats: int = 3,
) -> None:
    import jax
    import numpy as np

    import pianistpy as pp
    from pianistpy.observables.stages import gauss_hermite_table, velocity_distribution

    pp.set_platform(platform_name)
    pp.set_cores(workers)
    stages: dict[str, float] = {}
    walls: dict[str, list[float]] = {}
    cores: dict[str, list] = {}
    classes = _efficiency_classes()
    serial = {"repeats": repeats, "cores": cores, "classes": classes}
    sampled = _ChildPeaks()
    sampled.start()

    nfw = pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy())
    r50 = np.geomspace(0.01, 10.0, 50)
    r10 = np.geomspace(0.05, 5.0, 10)
    v201 = np.linspace(-1.5, 1.5, 201)
    circle = pp.Aperture.circle(1.0)
    # rows 1 to 7, in the install page's order
    df = _timed(
        stages,
        walls,
        "1. parent DF: compute_df (NFW, isotropic)",
        lambda: pp.compute_df(nfw),
    )
    parent = _timed(
        stages,
        walls,
        "2. Method I truncation (K_t 0.05)",
        lambda: pp.truncate(df, pp.Truncation(k_t=0.05)),
    )
    member = pp.Component(parent.profile, pp.Anisotropy(beta_c=0.2))
    child = _timed(
        stages,
        walls,
        "3. constant beta_c 0.2 child, Method II (compute_df, truncate)",
        lambda: pp.truncate(pp.compute_df(member)),
    )
    sampler = _timed(
        stages,
        walls,
        "4. Sampler(child): the velocity table",
        lambda: pp.Sampler(child),
    )
    _timed(stages, walls, "5. sample(1e7)", lambda: sampler.sample(10_000_000, seed=1))
    build = "6. LosTable(child), first build"
    start = time.perf_counter()
    table = pp.LosTable(child)
    stages[build] = time.perf_counter() - start
    n_rows = len(table.table.rows)
    print(f"{build}: {stages[build]:.4f} s; {n_rows} rows", flush=True)
    # row 7 by route: on the model, the second-moment route (no table)
    _timed(
        stages,
        walls,
        "7a. sigma_r on the model, 50 radii",
        lambda: pp.sigma_r(child, r50),
        **serial,
    )
    _timed(
        stages,
        walls,
        "7b. sigma_t on the model, 50 radii",
        lambda: pp.sigma_t(child, r50),
        **serial,
    )
    _timed(
        stages,
        walls,
        "7c. sigma_los on the model, 50 radii",
        lambda: pp.sigma_los(child, r50),
        **serial,
    )
    _timed(
        stages,
        walls,
        "7d. sigma_los_aperture on the model, circle 1",
        lambda: pp.sigma_los_aperture(child, circle),
        **serial,
    )
    # from the table the chain built: no row times a table build
    _timed(
        stages,
        walls,
        "7e. losvd from the table, R 1, 201 velocities",
        lambda: pp.losvd(table, 1.0, v201),
    )
    _timed(
        stages,
        walls,
        "7f. aperture_losvd from the table, circle 1, 201 velocities",
        lambda: pp.aperture_losvd(table, circle, v201),
    )
    _timed(
        stages,
        walls,
        "7g. sigma_los_aperture from the table, circle 1",
        lambda: pp.sigma_los_aperture(table, circle),
    )
    print(
        f"the table's aperture pools after 7f and 7g: {len(table.table.pools)}",
        flush=True,
    )
    _timed(
        stages,
        walls,
        "7h. Gauss-Hermite moments from the table, 10 radii (stage function)",
        lambda: gauss_hermite_table(table.table).read(r10),
    )
    _timed(
        stages,
        walls,
        "7i. pdf_v, 10 radii, V 0.3",
        lambda: pp.pdf_v(child, r10, np.full(10, 0.3)),
    )
    _timed(
        stages,
        walls,
        "7j. velocity distribution, r 1 (stage function)",
        lambda: velocity_distribution(child, 1.0, child.settings),
    )
    d = pp.Differentiable(child)
    grad = jax.grad(lambda t: d.sigma_los(t, 1.0))
    _timed(
        stages,
        walls,
        "7k. Differentiable: jax.grad of sigma_los, R 1",
        lambda: np.asarray(grad(d.theta)),
    )

    if profile:
        # one more call of each model-path row under cProfile, after its timing: which
        # stages each call runs
        for name, call in (
            ("7a", lambda: pp.sigma_r(child, r50)),
            ("7b", lambda: pp.sigma_t(child, r50)),
            ("7c", lambda: pp.sigma_los(child, r50)),
            ("7d", lambda: pp.sigma_los_aperture(child, circle)),
        ):
            profiler = cProfile.Profile()
            profiler.runcall(call)
            with open(f"{out}.profile_{name}.txt", "w") as handle:
                stats = pstats.Stats(profiler, stream=handle)
                stats.sort_stats("cumulative").print_stats(40)
            print(f"{name}: profile in {out}.profile_{name}.txt", flush=True)

    # the largest worker's own peak: each live worker of the table's aperture pool
    # reports it (sent back from the worker); the LosTable build's pool, which the
    # package closes at the build's end, is read while it lives (Linux and Windows)
    reported: dict[int, float] = {}
    how = set()
    for pool in table.table.pools.values():
        for pid, peak, source in pool.map(
            _worker_peak, range(8 * workers), chunksize=1
        ):
            reported[pid] = max(peak, reported.get(pid, 0.0))
            how.add(source)
    sampled.stop.set()
    sampled.join()
    own, own_source = _own_peak_gib() or (float("nan"), "none")
    nan = float("nan")
    largest = max([*reported.values(), *sampled.peaks.values()], default=nan)
    print(
        f"main: {own:.3f} GiB ({own_source}); processes reporting: {len(reported)}"
        f" ({', '.join(sorted(how))}), largest"
        f" {max(reported.values(), default=nan):.3f} GiB; children read while alive:"
        f" {len(sampled.peaks)} ({', '.join(sorted(sampled.sources))}), largest"
        f" {max(sampled.peaks.values(), default=nan):.3f} GiB",
        flush=True,
    )
    record = {
        "platform": platform_name,
        "workers": workers,
        "devices": [str(x) for x in jax.devices()],
        "pianistpy": pp.__version__,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "python": host_platform.python_version(),
        "hardware": _hardware(platform_name, jax.devices()),
        "machine": host_platform.processor() or host_platform.machine(),
        "rows": n_rows,
        "tier": child.validation.tier,
        "settings": "default",
        "peak_gib": {"main": own, "largest worker": largest},
        "peak_read": sorted({own_source, *how, *sampled.sources}),
        "stages": stages,
        "walls": walls,
        "repeats_7a_to_7d": repeats,
        "cores_7a_to_7d": {
            "read": "Windows only: [processor, efficiency class] at each call's start"
            " and end, the warm call first; a higher class is a faster core",
            "calls": cores,
        }
        if classes is not None
        else None,
    }
    with open(out, "w") as handle:
        json.dump(record, handle, indent=1)
    print(
        json.dumps({k: v for k, v in record.items() if k not in ("stages", "walls")}),
        flush=True,
    )


# The rows' labels on the computing-time page (Barry's wording, 2026-10-20), keyed by
# the stage names the legs' JSON records carry, which stay as they are so that legs of
# different versions compare. R is a 3D radius and R_2D a projected one, as in the
# paper; the page sets the symbols and the units in math.
LABELS = {
    "1. parent DF: compute_df (NFW, isotropic)": (
        "1. Untruncated DF: compute_df (NFW, isotropic)"
    ),
    "2. Method I truncation (K_t 0.05)": "2. Method I truncation (K_t = 0.05)",
    "3. constant beta_c 0.2 child, Method II (compute_df, truncate)": (
        "3. Method II model at constant beta_c = 0.2 (compute_df, truncate)"
    ),
    "4. Sampler(child): the velocity table": "4. Sampler: the velocity table",
    "5. sample(1e7)": "5. Drawing 10^7 particles",
    "6. LosTable(child), first build": "6. LosTable: the first build",
    "7a. sigma_r on the model, 50 radii": "sigma_r evaluated at 50 3D radii",
    "7b. sigma_t on the model, 50 radii": "sigma_t evaluated at 50 3D radii",
    "7c. sigma_los on the model, 50 radii": (
        "sigma_los evaluated at 50 projected radii"
    ),
    "7d. sigma_los_aperture on the model, circle 1": (
        "sigma_los_aperture over a circle of radius 1"
    ),
    "7e. losvd from the table, R 1, 201 velocities": (
        "losvd at R_2D = 1, evaluated at 201 line-of-sight velocities"
    ),
    "7f. aperture_losvd from the table, circle 1, 201 velocities": (
        "aperture_losvd over a circle of radius 1, evaluated at 201 "
        "line-of-sight velocities"
    ),
    "7g. sigma_los_aperture from the table, circle 1": (
        "sigma_los_aperture over a circle of radius 1"
    ),
    "7h. Gauss-Hermite moments from the table, 10 radii (stage function)": (
        "Gauss-Hermite moments evaluated at 10 projected radii"
    ),
    "7i. pdf_v, 10 radii, V 0.3": (
        "pdf_v evaluated at 10 3D radii, at the velocity V = 0.3"
    ),
    "7k. Differentiable: jax.grad of sigma_los, R 1": (
        "jax.grad of sigma_los at R_2D = 1"
    ),
}


def _time(seconds: float) -> str:
    """A time as the page writes it: under 0.1 s in ms, 10 min and above in whole
    minutes."""
    if seconds < 0.1:
        return f"{seconds * 1e3:.3g} ms"
    if seconds >= 600.0:
        return f"{round(seconds / 60.0):d} min"
    return f"{seconds:.3g} s"


FIRST_APERTURE = "7f. aperture_losvd from the table, circle 1, 201 velocities"


# The page's three tables (one per group of rows, each with its own heading; no row
# is referred to by its number or position): building a model, the observables that
# read the model, and the observables that read the line-of-sight table. The stage
# 7j (an internal stage function a reader cannot call) is not on the page.
GROUPS = (
    (
        "Building a model",
        (
            "1. parent DF: compute_df (NFW, isotropic)",
            "2. Method I truncation (K_t 0.05)",
            "3. constant beta_c 0.2 child, Method II (compute_df, truncate)",
            "4. Sampler(child): the velocity table",
            "5. sample(1e7)",
            "6. LosTable(child), first build",
        ),
    ),
    (
        "Observables from the distribution function",
        (
            "7a. sigma_r on the model, 50 radii",
            "7b. sigma_t on the model, 50 radii",
            "7c. sigma_los on the model, 50 radii",
            "7d. sigma_los_aperture on the model, circle 1",
            "7i. pdf_v, 10 radii, V 0.3",
            "7k. Differentiable: jax.grad of sigma_los, R 1",
        ),
    ),
    (
        "Observables from the line-of-sight table",
        (
            "7e. losvd from the table, R 1, 201 velocities",
            "7f. aperture_losvd from the table, circle 1, 201 velocities",
            "7g. sigma_los_aperture from the table, circle 1",
            "7h. Gauss-Hermite moments from the table, 10 radii (stage function)",
        ),
    ),
)


def table(paths: list[str]) -> None:
    legs = [json.load(open(p)) for p in paths]
    head = ["stage"] + [_label(leg) for leg in legs]
    for heading, names in GROUPS:
        print(f"{heading}")
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for name in names:
            cells = []
            for leg in legs:
                value = leg["stages"].get(name)
                cell = "" if value is None else _time(value)
                if leg["platform"] == "gpu" and name in NO_GPU_VELOCITYUP:
                    cell = f"{cell} (no GPU acceleration)"
                cells.append(cell)
            print(f"| {LABELS.get(name, name)} | " + " | ".join(cells) + " |")
        if heading == "Building a model":
            memory = []
            for leg in legs:
                peak = leg["peak_gib"]
                read = ", ".join(leg.get("peak_read", ["not recorded"]))
                memory.append(
                    f"{peak['main']:.2f} GiB, {peak['largest worker']:.2f} GiB ({read})"
                )
            print(
                "| Peak memory (main process, largest process) | "
                + " | ".join(memory)
                + " |"
            )
        print()
    print("LosTable rows: " + ", ".join(str(leg["rows"]) for leg in legs))
    print(
        "The Method II model's report band under the default settings: "
        + ", ".join(str(leg["tier"]) for leg in legs)
    )
    for leg in legs:
        print(
            f"{leg['platform']}: pianistpy {leg['pianistpy']}, jax {leg['jax']},"
            f" numpy {leg['numpy']}, python {leg['python']}, {leg['machine']},"
            f" devices {leg['devices']}"
        )
    # for the page's intro sentence, not a row of its table: the first aperture
    # call on a table (the aperture stage's warm call), per leg
    for leg in legs:
        walls = leg.get("walls", {}).get(FIRST_APERTURE)
        if walls:
            print(
                f"{leg['platform']}: the first aperture call on the table"
                f" {walls[0]:.1f} s"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--platform", choices=("cpu", "gpu"))
    parser.add_argument("--cores", type=int, default=28, dest="workers")
    parser.add_argument("--out")
    parser.add_argument("--table", nargs="+")
    parser.add_argument(
        "--profile", action="store_true", help="profile one more call of rows 7a to 7d"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="timed calls of rows 7a to 7d"
    )
    args = parser.parse_args()
    if args.table:
        table(args.table)
    else:
        run(args.platform, args.workers, args.out, args.profile, args.repeats)
