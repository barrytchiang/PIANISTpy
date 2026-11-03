"""Tests of the names ver. 1.0.0 used and ver. 1.0.1 renamed.

Each renamed name keeps a bridge for ver. 1.0.0 users: the class alias
``EnergyGrid`` resolves to `KGrid` with a `DeprecationWarning`, a
result file that names the rule ``"edge_law"`` reads back with the rule
``"boundary_form"``, and the losvd fields given under their former names
``n_energy_nodes`` and ``energy_layer_ratio`` are recorded as ``n_k_nodes``
and ``k_layer_ratio``.
"""

import warnings

import pytest

import pianistpy.df.stages as stages
from pianistpy.df.results import _FORMER_RULES
from pianistpy.settings import FORMER_FIELDS, Settings


def test_energy_grid_is_a_deprecated_alias_of_kgrid():
    with pytest.warns(DeprecationWarning, match="EnergyGrid is KGrid since ver. 1.0.1"):
        alias = stages.EnergyGrid
    assert alias is stages.KGrid
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The new name itself warns about nothing.
        assert stages.KGrid is not None
    with pytest.raises(AttributeError, match="has no attribute 'NoSuchName'"):
        stages.__getattr__("NoSuchName")


def test_the_former_rule_name_maps_to_the_boundary_form():
    assert _FORMER_RULES == {"edge_law": "boundary_form"}


def test_the_former_losvd_field_names_are_recorded_under_the_current_names():
    assert FORMER_FIELDS == {
        "losvd": {"n_energy_nodes": "n_k_nodes", "energy_layer_ratio": "k_layer_ratio"}
    }
    former = Settings(losvd={"n_energy_nodes": 6, "energy_layer_ratio": 0.3})
    current = Settings(losvd={"n_k_nodes": 6, "k_layer_ratio": 0.3})
    assert former == current
    assert dict(former.overrides["losvd"]) == {"n_k_nodes": 6, "k_layer_ratio": 0.3}
    # A settings record of ver. 1.0.0 reads through the same bridge.
    record = {"level": "default", "overrides": {"losvd": {"n_energy_nodes": 6}}}
    assert Settings.from_dict(record).losvd["n_k_nodes"] == 6
