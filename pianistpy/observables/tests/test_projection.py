"""Tests of the projection and of the readers of a projected table.

The uniform test sphere has unit density out to the radius R_T = 2 and the
constant squared radial dispersion SIGMA_R2 = 0.3. The projection of the uniform
sphere is checked against closed forms. The readers of a projected table are
checked for their values between rows, for the InputError they raise outside
the rows, and for the AccuracyError they raise at a negative variance or a
divergent inner disk.
"""

import numpy as np

from pianistpy import Anisotropy
from pianistpy.observables._projection import (
    certify_projection,
    dispersion_table_reader,
    edge_variable,
    edge_variable_inverse,
    project,
    surface_density_reader,
)
from pianistpy.validation import TIER_BOUNDS

R_T = 2.0
SIGMA_R2 = 0.3


def _density(r):
    return np.ones_like(np.asarray(r, dtype=np.float64))


def _sigma_r2(r):
    return np.full_like(np.asarray(r, dtype=np.float64), SIGMA_R2)


def test_projection_of_a_uniform_isotropic_sphere():
    """Assert that `project` gives the closed forms of the uniform sphere.

    The surface density of the uniform sphere is ``2 (R_T^2 - R_2D^2)^(1/2)``.
    For beta_sigma = 0 and a constant SIGMA_R2, the squared line-of-sight
    dispersion equals SIGMA_R2 at every projected radius (Chiang+(2026d),
    Eq. (B46), with beta_sigma = 0). Both closed forms hold to ``rtol=1e-12``.
    """
    r_2d = np.array([1e-3, 0.1, 0.5, 1.0, 1.9])
    surface_density, sigma_los2 = project(
        r_2d, _density, _sigma_r2, R_T, Anisotropy(), 200
    )
    np.testing.assert_allclose(
        surface_density, 2.0 * np.sqrt(R_T**2 - r_2d**2), rtol=1e-12
    )
    np.testing.assert_allclose(sigma_los2, SIGMA_R2, rtol=1e-12)


def test_edge_variable_round_trip():
    """Assert that `edge_variable_inverse` inverts `edge_variable` at R_T to
    ``rtol=1e-13``."""
    r = np.array([1e-3, 0.5, 1.9, 1.999999])
    np.testing.assert_allclose(
        edge_variable_inverse(edge_variable(r, R_T), R_T), r, rtol=1e-13
    )


def test_readers_and_certificate():
    """Assert that a table checked by `certify_projection` reads the closed forms.

    The rows start at 60 logarithmic projected radii from ``1e-3`` to
    ``0.9999 * R_T``. `certify_projection` adds rows until the reading at every
    interval's midpoint is within a relative ``0.01 * TIER_BOUNDS[0]`` of
    `project`. The
    test asserts that the check ends with ``record["end"] == "tolerance"``. At the
    geometric midpoints of the final rows, the surface density reader and the
    dispersion reader match the closed forms of the uniform sphere to
    ``rtol=1e-8``.
    """
    r_2d = np.logspace(-3, np.log10(0.9999 * R_T), 60)

    def rule(radii):
        return project(radii, _density, _sigma_r2, R_T, Anisotropy(), 200)

    surface_density, sigma_los2 = rule(r_2d)
    rows = certify_projection(
        r_2d, surface_density, sigma_los2, R_T, True, rule, 0.01 * TIER_BOUNDS[0]
    )
    r_rows, sigma_rows, los_rows, record = rows
    assert record["end"] == "tolerance"
    read_sigma = surface_density_reader(r_rows, sigma_rows, R_T, True)
    read_los = dispersion_table_reader(r_rows, los_rows)
    r_mid = np.sqrt(r_rows[:-1] * r_rows[1:])
    np.testing.assert_allclose(
        read_sigma(r_mid), 2.0 * np.sqrt(R_T**2 - r_mid**2), rtol=1e-8
    )
    np.testing.assert_allclose(read_los(r_mid), SIGMA_R2, rtol=1e-8)


def test_a_read_outside_the_rows_is_refused_with_cause():
    """Assert that both readers raise InputError outside the rows, with the
    note set on the reader.

    The rows are 60 logarithmic projected radii from ``1e-3`` to
    ``0.9999 * R_T``. A radius 1.001 times the last row raises InputError whose
    message gives the outermost radius and the outer note. A radius of
    ``0.999e-3`` raises InputError whose message gives the innermost radius and
    the inner note.
    """
    import pytest

    from pianistpy import InputError

    r_rows = np.logspace(-3, np.log10(0.9999 * R_T), 60)
    surface = 2.0 * np.sqrt(R_T**2 - r_rows**2)
    for reader in (
        surface_density_reader(r_rows, surface, R_T, True),
        dispersion_table_reader(r_rows, np.full(60, SIGMA_R2)),
    ):
        reader.inner_note, reader.outer_note = "Inner note.", "Outer note."
        # An end row times 1 + 1e-13 lies within END_TOLERANCE (1e-12) of the
        # end row, and the reader returns a finite value there.
        assert np.isfinite(reader(r_rows[[0, -1]] * (1.0 + 1e-13))).all()
        with pytest.raises(InputError, match="beyond the outermost.* Outer note"):
            reader(np.array([1.0, 1.001 * r_rows[-1]]))
        with pytest.raises(InputError, match="inside the innermost.* Inner note"):
            reader(np.array([0.999e-3, 1.0]))


def test_a_negative_variance_is_refused_not_clipped():
    """Assert that the dispersion reader raises AccuracyError at a negative value
    and does not clip the value.

    The 40 rows are the cosines of 40 angles from 0 to pi, and the last 20 rows
    are negative. The test asserts five properties of the reader: `raw` returns
    negative values; the reader has no ``n_clipped`` attribute; at the first 10
    rows, which are positive, the call returns the values of `raw`; at all 40
    rows, the call raises AccuracyError; `traced` returns NaN at the 20 negative
    rows.
    """
    import pytest

    from pianistpy.exceptions import AccuracyError

    r_rows = np.logspace(-3, 0, 40)
    # The rows take both signs, as the rows of a tangent table can.
    rows = np.cos(np.linspace(0.0, np.pi, 40))
    read = dispersion_table_reader(r_rows, rows)
    assert (read.raw(r_rows) < 0.0).any()
    assert not hasattr(read, "n_clipped")
    np.testing.assert_array_equal(read(r_rows[:10]), read.raw(r_rows[:10]))
    with pytest.raises(AccuracyError, match="a negative variance is not returned"):
        read(r_rows)
    traced = read.traced(r_rows)
    assert np.isnan(traced[rows < 0.0]).all()


def test_the_inner_disk_is_the_slopes_own_or_refused():
    """Assert the inner disk of a power-law surface density, or AccuracyError.

    For the surface density ``R**-s`` with s in (-0.5, 0, 1.5, 1.995), the
    integral of Sigma R dR from zero to the first row at ``1e-3`` is
    ``1e-3 ** (2 - s) / (2 - s)``, and `inner_disk` matches the integral to
    ``rel=1e-9``. For s = 2.5, the reader returns finite values at the first
    three rows, and `inner_disk` raises AccuracyError.
    """
    import pytest

    from pianistpy.exceptions import AccuracyError

    r_rows = np.logspace(-3, 0, 40)
    for s in (-0.5, 0.0, 1.5, 1.995):
        read = surface_density_reader(r_rows, r_rows**-s, 10.0, False)
        assert read.inner_disk == pytest.approx(1e-3 ** (2.0 - s) / (2.0 - s), rel=1e-9)
    steep = surface_density_reader(r_rows, r_rows**-2.5, 10.0, False)
    # The reader itself raises no error for s = 2.5.
    assert np.isfinite(steep(r_rows[:3])).all()
    with pytest.raises(AccuracyError, match="is not below 2"):
        _ = steep.inner_disk
