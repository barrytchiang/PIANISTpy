"""Tests of the switches of the session."""

import logging

import pytest

from pianistpy import InputError, runtime


@pytest.fixture(autouse=True)
def _restore_switches():
    yield
    runtime.set_verbosity("quiet")
    runtime.set_timing(False)
    runtime.set_workers("auto")


def test_quiet_by_default_and_messages_are_still_remembered(capsys):
    runtime.log.debug("A detail with the value %s.", 3)
    runtime.log.info("A stage message.")
    assert capsys.readouterr().err == ""
    recent = runtime.recent_messages()
    assert recent[-2].endswith("A detail with the value 3.")
    assert recent[-1].endswith("A stage message.")
    assert len(recent) <= runtime.RECENT_MESSAGES_MAX


def test_verbosity_levels(capsys):
    runtime.set_verbosity("info")
    assert runtime._screen.level == logging.INFO
    runtime.set_verbosity("debug")
    assert runtime._screen.level == logging.DEBUG
    with pytest.raises(InputError):
        runtime.set_verbosity("loud")


def test_log_to_file(tmp_path):
    path = tmp_path / "run.log"
    with runtime.log_to_file(path):
        runtime.log.debug("Written to the file.")
    runtime.log.debug("Not written to the file.")
    text = path.read_text(encoding="utf-8")
    assert "Written to the file." in text
    assert "Not written" not in text


def test_timing_enabled_reports_the_switch():
    assert runtime.timing_enabled() is False
    runtime.set_timing(True)
    assert runtime.timing_enabled() is True
    runtime.set_timing(0)
    assert runtime.timing_enabled() is False


def test_stages_are_timed_whether_or_not_timing_is_printed(capsys):
    with runtime.timing_record() as timing:
        with runtime.timed("energy_grid"):
            pass
        with runtime.timed("invert_rows"):
            pass
        with runtime.timed("energy_grid"):
            pass
    assert list(timing.stages) == ["energy_grid", "invert_rows"]
    assert all(seconds >= 0.0 for seconds in timing.stages.values())
    assert timing.total == sum(timing.to_dict().values())
    assert capsys.readouterr().out == ""
    runtime.set_timing(True)
    with runtime.timing_record(), runtime.timed("solve_poisson"):
        pass
    printed = capsys.readouterr().out
    assert "solve_poisson" in printed
    assert "total" in printed


def test_peak_memory_is_recorded_with_each_stage(monkeypatch):
    # The peak is the largest memory of the whole process, so what an
    # allocation here adds to it depends on the tests that ran before: the
    # reading is checked on its own and the recording with stated readings.
    before = runtime.peak_memory()
    assert before is not None and before > 2**20
    readings = iter([before + 64 * 2**20, before + 96 * 2**20])
    monkeypatch.setattr(runtime, "peak_memory", lambda: next(readings))
    with runtime.timing_record() as timing:
        with runtime.timed("first"):
            pass
        with runtime.timed("second"):
            pass
    assert timing.memory == {
        "first": before + 64 * 2**20,
        "second": before + 96 * 2**20,
    }
    assert list(timing.memory) == ["first", "second"]
    assert timing.to_dict() == timing.stages


def test_peak_memory_never_decreases():
    first = runtime.peak_memory()
    block = bytearray(8 * 2**20)
    block[::4096] = b"\x01" * len(block[::4096])
    assert runtime.peak_memory() >= first
    del block


def test_a_stage_outside_a_record_is_still_announced():
    with runtime.timed("lonely_stage"):
        pass
    assert any("lonely_stage" in message for message in runtime.recent_messages())


def test_workers(monkeypatch):
    for name in runtime.SCHEDULER_JOB_VARIABLES + runtime.SCHEDULER_CORE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "allowed_cores", lambda: 16)
    runtime.set_workers("auto")
    assert runtime.get_workers() == 8
    runtime.set_workers("full")
    assert runtime.get_workers() == 16
    runtime.set_workers("half")
    assert runtime.get_workers() == 8
    runtime.set_workers(3)
    assert runtime.get_workers() == 3
    runtime.set_workers("5")
    assert runtime.get_workers() == 5
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    runtime.set_workers("auto")
    assert runtime.get_workers() == 16
    monkeypatch.setattr(runtime, "allowed_cores", lambda: 1)
    runtime.set_workers("half")
    assert runtime.get_workers() == 1
    for bad in (0, -2, "many", True, 1.5):
        with pytest.raises(InputError):
            runtime.set_workers(bad)


def test_allowed_cores_follows_the_scheduler(monkeypatch):
    for name in runtime.SCHEDULER_CORE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert runtime.allowed_cores() >= 1
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert runtime.allowed_cores() == 6


def test_an_error_of_the_package_carries_the_recent_messages():
    runtime.log.debug("The last thing done before the failure.")
    error = InputError("A parameter is out of range.")
    assert error.recent[-1].endswith("The last thing done before the failure.")
    assert str(error) == "A parameter is out of range."


def test_the_card_memory_is_none_off_a_gpu(monkeypatch):
    monkeypatch.setattr(runtime, "platform", lambda: "cpu")
    assert runtime.card_memory() is None


def test_the_workers_are_explicit_only_when_set_to_a_number():
    assert not runtime.workers_explicit()
    runtime.set_workers("3")
    assert runtime.workers_explicit()
    runtime.set_workers("half")
    assert not runtime.workers_explicit()


def test_the_environment_switches_read_one_set_of_words_in_any_case(monkeypatch):
    for value, expected in (
        ("", False),
        ("false", False),
        ("OFF", False),
        (" 0 ", False),
        ("1", True),
        ("True", True),
        ("yes", True),
    ):
        monkeypatch.setenv(runtime.TIMING_VARIABLE, value)
        assert runtime._env_flag(runtime.TIMING_VARIABLE) is expected, value
    monkeypatch.setenv(runtime.TIMING_VARIABLE, "sometimes")
    with pytest.raises(InputError, match=runtime.TIMING_VARIABLE):
        runtime._env_flag(runtime.TIMING_VARIABLE)
    monkeypatch.setenv(runtime.VERBOSITY_VARIABLE, " INFO ")
    runtime.set_verbosity(runtime._env_word(runtime.VERBOSITY_VARIABLE, "quiet"))
    monkeypatch.delenv(runtime.VERBOSITY_VARIABLE)
    assert runtime._env_word(runtime.VERBOSITY_VARIABLE, "quiet") == "quiet"


def test_the_workers_run_xla_on_one_thread(monkeypatch):
    # a worker's XLA flags carry the one-thread flags beside
    # the process's own, for the spawn only
    import os

    from pianistpy import runtime

    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/nowhere")
    with runtime.worker_environment():
        flags = os.environ["XLA_FLAGS"].split()
        assert flags[0] == "--xla_dump_to=/nowhere"
        assert set(runtime.WORKER_XLA_FLAGS) <= set(flags)
    assert os.environ["XLA_FLAGS"] == "--xla_dump_to=/nowhere"
    monkeypatch.delenv("XLA_FLAGS")
    with runtime.worker_environment():
        assert os.environ["XLA_FLAGS"].split() == list(runtime.WORKER_XLA_FLAGS)
    assert "XLA_FLAGS" not in os.environ
