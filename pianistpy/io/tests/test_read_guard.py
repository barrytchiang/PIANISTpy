"""The read of a result recorded as unphysical never returns a Model.

A product of the pipeline (its record states ``physical`` False) is not a
result of this package, and the read refuses it by name; a distribution
function whose inversion was refused (signed, ``is_physical`` False) reads
back as a DistributionFunction, never a Model.
"""

import h5py
import numpy as np
import pytest

from pianistpy import ResultError, read
from pianistpy.df import DistributionFunction, Model
from pianistpy.superposition.tests.test_models_only import signed  # noqa: F401


def test_read_refuses_an_unphysical_product(tmp_path):
    path = tmp_path / "product.h5"
    with h5py.File(path, "w") as file:
        file.attrs.update(
            {
                "format_version": "1.0",
                "data_sha256": "0" * 64,
                "product_type": "df_model",
                "physical": False,
                "validity_gates_fired": "df_sign",
            }
        )
        file.create_dataset("df/F", data=np.linspace(-1.0, 1.0, 5))
    with pytest.raises(ResultError, match="lacks the attribute"):
        read(path)


def test_read_of_a_signed_df_is_never_a_model(tmp_path, signed):  # noqa: F811
    path = tmp_path / "df.h5"
    signed.write(path)
    back = read(path)
    assert isinstance(back, DistributionFunction)
    assert not isinstance(back, Model)
    assert not back.is_physical
