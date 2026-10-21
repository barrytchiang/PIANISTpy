"""Tests of the HDF5 result files.

The tests cover `write_result`, `read_result`, `load_result` and
`data_digest`.
"""

import h5py
import numpy as np
import pytest

from pianistpy import InputError, ResultError
from pianistpy.identity import identity_string
from pianistpy.io import data_digest, load_result, read_result, write_result

RECORD = {"inputs": {"beta_c": -0.25, "r_a": None}, "settings": {"level": "default"}}
IDENTITY = identity_string(RECORD["inputs"], RECORD["settings"], "1.0")


def _arrays():
    rng = np.random.default_rng(0)
    return {
        "k": np.linspace(0.0, 1.0, 11),
        "df/log_df": rng.standard_normal(11),
        "count": np.arange(5, dtype=np.int64),
    }


def test_round_trip_returns_the_same_result(tmp_path):
    path = tmp_path / "result.h5"
    digest = write_result(path, "model", IDENTITY, RECORD, _arrays(), parents=("abc",))
    stored = read_result(path)
    assert stored.kind == "model"
    assert stored.identity == IDENTITY
    assert stored.data_sha256 == digest
    assert stored.record == RECORD
    assert stored.parents == ("abc",)
    assert set(stored.arrays) == set(_arrays())
    for name, values in _arrays().items():
        assert np.array_equal(stored.arrays[name], values)
        assert stored.arrays[name].dtype == values.dtype
    assert stored.provenance["numpy_version"] == np.__version__


def test_a_live_result_and_a_path_load_to_the_same_thing(tmp_path):
    path = tmp_path / "result.h5"
    write_result(path, "model", IDENTITY, RECORD, _arrays())
    stored = read_result(path)
    assert load_result(stored) is stored
    assert load_result(path).data_sha256 == stored.data_sha256


def test_an_existing_file_is_not_replaced_silently(tmp_path):
    path = tmp_path / "result.h5"
    write_result(path, "model", IDENTITY, RECORD, _arrays())
    with pytest.raises(InputError, match="exists"):
        write_result(path, "model", IDENTITY, RECORD, _arrays())
    write_result(path, "model", IDENTITY, RECORD, _arrays(), overwrite=True)


def test_the_digest_ignores_attributes_and_sees_the_arrays(tmp_path):
    path = tmp_path / "result.h5"
    digest = write_result(path, "model", IDENTITY, RECORD, _arrays())
    with h5py.File(path, "a") as file:
        file.attrs["note"] = "metadata only"
    assert data_digest(path) == digest
    read_result(path)
    with h5py.File(path, "a") as file:
        file["data/k"][3] += 1e-15
    with pytest.raises(ResultError):
        read_result(path)


def test_a_file_of_another_format_is_refused(tmp_path):
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("x", data=np.zeros(3))
    with pytest.raises(ResultError):
        read_result(path)
    write_result(path, "model", IDENTITY, RECORD, _arrays(), overwrite=True)
    with h5py.File(path, "a") as file:
        file.attrs["format_version"] = "1.2.0"
    with pytest.raises(ResultError):
        read_result(path)


def test_a_file_that_does_not_read_back_as_written_is_refused(tmp_path, monkeypatch):
    create = h5py.Group.create_dataset

    def perturbed(group, name, data=None, **kwargs):
        if name == "k":
            data = np.asarray(data) + 1e-15
        return create(group, name, data=data, **kwargs)

    monkeypatch.setattr(h5py.Group, "create_dataset", perturbed)
    with pytest.raises(ResultError, match="does not read back as written"):
        write_result(tmp_path / "result.h5", "model", IDENTITY, RECORD, _arrays())


def test_each_dataset_is_read_once_to_write_and_once_to_read(tmp_path, monkeypatch):
    read = h5py.Dataset.__getitem__
    count = {"n": 0}

    def counted(dataset, key):
        count["n"] += 1
        return read(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", counted)
    path = tmp_path / "result.h5"
    digest = write_result(path, "model", IDENTITY, RECORD, _arrays())
    assert count["n"] == len(_arrays())
    count["n"] = 0
    stored = read_result(path)
    assert count["n"] == len(_arrays())
    assert stored.data_sha256 == digest == data_digest(path)
