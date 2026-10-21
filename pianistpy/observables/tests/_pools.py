"""Helpers for the tests of a process pool whose process ends during a call.

The pool tests in test_losvd_stages.py and test_table_vjp.py import two
helpers from this module. `bounded` fails a test whose block runs longer than
BOUND = 120 s, instead of letting the test hang. `end_first_worker` makes every
pool that a module starts lose its first process before the module uses the
pool, as the kernel ends a process when the machine runs out of memory. The
helpers send no signal and catch none, so the tests run on every platform.
"""

import _thread
import contextlib
import multiprocessing as mp
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

# BOUND is the longest wait, in seconds, of a test whose pool has lost a
# process. A pool call raises WorkerError at the pool's first check after the
# process ended, and the pool checks its processes every WORKER_POLL = 0.1 s
# while the call waits. BOUND = 120 s is set to exceed, many times over, the
# time to start the pool's processes and to run the computation before the
# pool's first call.
BOUND = 120


@contextlib.contextmanager
def bounded(seconds: int = BOUND) -> Iterator[None]:
    """Fail the test where the block runs longer than `seconds`, never hang it.

    When `seconds` have passed, a timer kills every live child process and
    interrupts the main thread. Killing the child processes lets a pool call
    that checks for an ended process return. The interrupt reaches the main
    thread at its next Python bytecode; a wait inside a single C call is not
    cut short. When the timer has fired, the context manager fails the test,
    whatever exception the block raised.
    """
    fired = threading.Event()

    def stop() -> None:
        fired.set()
        for process in mp.active_children():
            process.kill()
        _thread.interrupt_main()

    timer = threading.Timer(seconds, stop)
    timer.daemon = True
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        if not fired.is_set():
            raise
    finally:
        timer.cancel()
    if fired.is_set():
        pytest.fail(f"The call did not return within {seconds} s.")


def end_first_worker(module: Any, monkeypatch: Any) -> list[Any]:
    """Patch `module.new_pool` so that every pool it starts loses its first process.

    The patched `new_pool` kills the pool's first process (`Process.kill`),
    as the kernel ends a process when the machine runs out of memory. The
    patched `new_pool` then waits, for at most BOUND seconds, until that
    process has an exit code, and returns the pool. The function returns
    the list to which the patched `new_pool` appends every pool it starts.
    """
    start = module.new_pool
    started: list[Any] = []

    def new_pool(*args: Any, **kwargs: Any) -> Any:
        pool = start(*args, **kwargs)
        started.append(pool)
        worker = pool.workers_started[0]
        worker.kill()
        deadline = time.monotonic() + BOUND
        while worker.exitcode is None and time.monotonic() < deadline:
            time.sleep(0.01)
        return pool

    monkeypatch.setattr(module, "new_pool", new_pool)
    return started
