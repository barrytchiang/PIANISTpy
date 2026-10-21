"""Tests of the error handling of the truncation-energy search.

The search for the truncation energy that meets a target radius or mass
(`pianistpy.df.target.solve_target`) treats the package's own exceptions from
a trial truncation as an inadmissible energy. Any other exception from a trial
truncation is raised to the caller.
"""

import pytest

from pianistpy import Anisotropy, Component, DoublePowerLaw
from pianistpy.df import compute_df, target


def test_an_error_that_is_not_the_solves_is_raised(monkeypatch):
    # A TypeError is not a package exception, so solve_target must raise the
    # TypeError instead of treating the trial energy as inadmissible.
    df = compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy()))

    def broken(df, z):
        raise TypeError("a fault, not an inadmissible K_t")

    monkeypatch.setattr(target, "truncation_chain", broken)
    with pytest.raises(TypeError, match="a fault"):
        target.solve_target(df, "radius", 10.0)


# A Type II model: an NFW profile with Osipkov-Merritt Type II anisotropy of
# anisotropy radius R_a = 3. Its DF is defined above P(R_a) = ln(4)/3, and the
# DF's rows below P(R_a) belong to the continuation.
R_A = 3.0


@pytest.fixture(scope="module")
def type_ii_df():
    return compute_df(
        Component(DoublePowerLaw.nfw(), Anisotropy(r_a=R_A, kind="II"), name="om2")
    )


def _fake_chain(radius, mass):
    """Return a stand-in for `truncation_chain` whose models are finite.

    The stand-in records the truncation energies it is called at and returns a
    finite model of truncation radius ``radius(z)`` and enclosed mass
    ``mass(z)`` at truncation energy z.
    """
    from types import SimpleNamespace

    import numpy as np

    calls = []

    def chain(df, z):
        calls.append(float(z))
        solve = SimpleNamespace(event_hit=True, r_event=radius(z))
        rows = SimpleNamespace(mass=np.array([mass(z)]))
        return {"solve": solve, "rows": rows}

    return chain, calls


@pytest.mark.parametrize("kind", ["radius", "mass"])
def test_a_type_ii_search_starts_at_the_domain_edge(type_ii_df, monkeypatch, kind):
    # The search's shallowest truncation energy lies above P(R_a), the edge of
    # the anisotropy's domain, so that no trial truncation reads the
    # continuation's rows. A target that the models at that shallowest energy
    # do not reach is refused with the domain's gate stated as the binding
    # limit, at the first truncation energy above P(R_a) (the next double,
    # which the message prints to 8 significant digits).
    import math

    from pianistpy.exceptions import UnphysicalModelError

    p_edge = math.log(1.0 + R_A) / R_A
    chain, calls = _fake_chain(lambda z: 0.5 / z, lambda z: 1.0 / z)
    monkeypatch.setattr(target, "truncation_chain", chain)
    goal = 2.0 if kind == "radius" else 1e6
    with pytest.raises(UnphysicalModelError) as caught:
        target.solve_target(
            type_ii_df,
            kind,
            goal,
            concentration=10.0 if kind == "mass" else None,
            probe=0.01,
        )
    assert calls and min(calls) > p_edge
    message = str(caught.value)
    assert "the gate Z_t_min_allowed binds" in message
    z_stated = float(message.split("Z_t = ")[1].split(",")[0])
    assert z_stated == pytest.approx(p_edge, rel=1e-7)
