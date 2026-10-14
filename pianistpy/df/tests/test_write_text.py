"""Tests of the text exports of a model and of a distribution function."""

import numpy as np
import pytest

from pianistpy import InputError
from pianistpy.df.analytic import analytic_model
from pianistpy.io.text import TEXT_FIELD


def _rounded(values):
    return np.array([float(TEXT_FIELD.format(v).strip()) for v in np.asarray(values)])


@pytest.fixture(scope="module")
def model():
    return analytic_model("hernquist", n_nodes=11)


def test_the_profile_is_the_radial_profiles_in_the_model_units(tmp_path, model):
    path = tmp_path / "profile.txt"
    model.write_text(path)
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[0] == "# PIANISTpy radial profile"
    assert f"# identity: {model.identity}" in lines
    assert "# unit_system: internal" in lines
    assert "# columns: r potential density mass" in lines
    profiles = model.radial_profiles()
    back = np.loadtxt(path)
    assert back.shape == (len(profiles["r"]), 4)
    assert np.array_equal(back[:, 0], _rounded(profiles["r"]))
    assert np.array_equal(back[:, 1], _rounded(profiles["potential"]))
    assert np.array_equal(back[:, 2], _rounded(np.asarray(profiles["density"])[0]))
    assert np.array_equal(back[:, 3], _rounded(profiles["mass"]))


def test_the_profile_follows_the_unit_system(tmp_path, model):
    scaled = model.to("physical", concentration=10.0, mass_0=1e10, r_0=1000.0)
    path = tmp_path / "profile.txt"
    scaled.write_text(path)
    lines = path.read_text(encoding="ascii").splitlines()
    assert "# unit_system: physical" in lines
    units = next(line for line in lines if line.startswith("# units: "))
    assert "r: pc" in units and "mass: solMass" in units
    back = np.loadtxt(path)
    r = scaled.radial_profiles()["r"]
    assert np.array_equal(back[:, 0], _rounded(r.value))
    assert back[0, 0] > 0 and back[-1, 0] / back[0, 0] == pytest.approx(
        float(model.radial_profiles()["r"][-1] / model.radial_profiles()["r"][0])
    )


def test_the_df_table_is_the_support_of_the_table(tmp_path, model):
    path = tmp_path / "df.txt"
    model.write_text(path, table="df")
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[0] == "# PIANISTpy distribution function table"
    assert "# columns: k log10_df dlog10_df_dlnk" in lines
    table = model.table
    support = slice(int(table.i_first[0]), int(table.i_last[0]) + 1)
    back = np.loadtxt(path, ndmin=2)
    assert back.shape[0] == support.stop - support.start
    assert np.array_equal(back[:, 0], _rounded(table.k[0, support]))
    assert np.array_equal(
        back[:, 1], _rounded(table.log_df[0, support]), equal_nan=True
    )
    assert np.array_equal(
        back[:, 2], _rounded(table.log_df_slope[0, support]), equal_nan=True
    )


def test_the_table_name_is_checked(tmp_path, model):
    with pytest.raises(InputError, match="profile"):
        model.write_text(tmp_path / "x.txt", table="grid")
    model.write_text(tmp_path / "p.txt")
    with pytest.raises(InputError, match="overwrite"):
        model.write_text(tmp_path / "p.txt")
