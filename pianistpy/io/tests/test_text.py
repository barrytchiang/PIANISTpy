"""Tests of the fixed-width text tables."""

import numpy as np
import pytest

from pianistpy import InputError, ResultError
from pianistpy.io import write_table
from pianistpy.io.text import TEXT_FIELD


def test_a_table_reads_back_to_the_digits_written(tmp_path):
    rng = np.random.default_rng(5)
    r = np.logspace(-3, 3, 7)
    rho = rng.uniform(1e-9, 1e3, 7)
    path = tmp_path / "profile.txt"
    write_table(
        path, {"r": r, "density": rho}, header=["PIANISTpy test", "units: internal"]
    )
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[:3] == [
        "# PIANISTpy test",
        "# units: internal",
        "# columns: r density",
    ]
    assert len(lines) == 3 + 7
    back = np.loadtxt(path)
    expected = np.column_stack(
        [[float(TEXT_FIELD.format(v).strip()) for v in c] for c in (r, rho)]
    )
    assert np.array_equal(back, expected)
    assert np.max(np.abs(back / np.column_stack([r, rho]) - 1.0)) < 1e-14


def test_integer_columns_are_written_as_integers(tmp_path):
    path = tmp_path / "t.txt"
    write_table(
        path, {"k": np.linspace(0.1, 0.9, 4), "component": np.array([0, 0, 1, 1])}
    )
    rows = path.read_text(encoding="ascii").splitlines()[1:]
    assert all(row.split()[1] in ("0", "1") for row in rows)
    assert np.array_equal(np.loadtxt(path)[:, 1], [0, 0, 1, 1])


def test_non_finite_values_survive_the_round_trip(tmp_path):
    path = tmp_path / "t.txt"
    column = np.array([1.0, -np.inf, np.nan, np.inf])
    write_table(path, {"log_df": column})
    back = np.loadtxt(path, ndmin=2)[:, 0]
    assert np.array_equal(back, column, equal_nan=True)


def test_an_empty_table_is_a_header_alone(tmp_path):
    path = tmp_path / "t.txt"
    write_table(path, {"r": np.zeros(0)})
    assert path.read_text(encoding="ascii") == "# columns: r\n"


@pytest.mark.parametrize(
    "columns",
    [
        {},
        {"r": np.zeros((2, 2))},
        {"r": np.zeros(3), "m": np.zeros(2)},
        {"a b": np.zeros(2)},
        {"name": np.array(["x", "y"])},
    ],
)
def test_bad_columns_are_refused(tmp_path, columns):
    with pytest.raises(InputError):
        write_table(tmp_path / "t.txt", columns)


def test_an_existing_file_needs_overwrite(tmp_path):
    path = tmp_path / "t.txt"
    write_table(path, {"r": np.ones(2)})
    with pytest.raises(InputError):
        write_table(path, {"r": np.ones(2)})
    write_table(path, {"r": np.full(2, 2.0)}, overwrite=True)
    assert np.array_equal(np.loadtxt(path), [2.0, 2.0])


def test_a_file_that_does_not_read_back_is_refused(tmp_path, monkeypatch):
    import pianistpy.io.text as text

    def broken(path, ndmin=2):
        return np.zeros((1, 1))

    monkeypatch.setattr(text.np, "loadtxt", broken)
    with pytest.raises(ResultError):
        write_table(tmp_path / "t.txt", {"r": np.ones(3)})
