r"""The superposition of models that share one truncated potential-density pair.

A user calls `superpose` to combine models into one model, `moments` to obtain
the second velocity moments of the combined model, and `mixture` to combine
any per-component quantity by the weights. Several distribution functions
inverted from one truncated potential-density pair, for example Method II
children of one truncated parent, combine into an equilibrium of that pair
for any positive weights :math:`w_k` that sum to 1:

.. math::

    f = \sum_k w_k f_k, \qquad \rho = \sum_k w_k \rho_k,

with :math:`f_k` the distribution function of member :math:`k` and
:math:`\rho_k` the density of member :math:`k`, every :math:`\rho_k` equal to
the shared density :math:`\rho`. `superpose` returns the equilibrium as a
`Model` whose components are the members' components; each component keeps
the member's DF table on the member's own energy grid, and the model's
density is the shared density split among the components by weight. The
weights are normalized to sum to 1, with a logged warning where the weights'
sum differs from 1 by more than :math:`10^{-10}`. The construction follows
Chiang+(2026d), Appendix B5, and Notes Section T:4.
"""

__all__ = ["mixture", "moments", "superpose"]

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from ..df import Model
from ..df.core import model_identity
from ..df.stages import AsymptoticForm, DfTable, InversionRecord
from ..exceptions import InputError
from ..observables._dispersion import beta_sigma, radial_dispersion
from ..profiles import TruncatedProfile
from ..runtime import log
from ..validation import Check, ValidationReport

# The weights are normalized to sum to 1 in every case. A weights' sum that
# differs from 1 by more than 1e-10 is logged as a warning: a difference of
# 1e-10 is far above the rounding error of a sum of a few float64 weights
# (about 1e-16), so a larger difference signals weights not meant to sum to 1.
WEIGHT_SUM_TOLERANCE = 1e-10


def mixture(weights: Any, values: Any, xp: Any = np) -> Any:
    r"""Return the weighted sum of per-component values, with the weights normalized.

    A superposed model combines a density, a velocity moment or any other
    quantity computed per component by the weighted sum `mixture` returns.
    The sum is linear in the values and, through the normalization, a smooth
    function of the
    weights. With ``xp=jax.numpy`` the sum is differentiable in the weights
    by JAX.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        The weights :math:`w_k`, positive. The call divides the weights by
        the weights' sum and does not check the weights.
    values : array-like, shape (n_components, ...)
        The value :math:`v_k` of each component, along the first axis.
    xp : module, optional
        The array module, ``numpy`` or ``jax.numpy``. Default: ``np``, which
        means NumPy arrays in and out.

    Returns
    -------
    value : array, shape (...)
        The sum :math:`\sum_k w_k v_k / \sum_k w_k`, with the shape of one
        component's value, as an array of `xp`.

    See Also
    --------
    superpose : The model that combines components by the weights.
    moments : The velocity moments of a superposed model.

    Examples
    --------
    >>> from pianistpy.superposition import mixture
    >>> mixture([1.0, 3.0], [[2.0, 4.0], [6.0, 8.0]])
    array([5., 7.])
    """
    weights = xp.asarray(weights)
    values = xp.asarray(values)
    normalized = weights / xp.sum(weights)
    return xp.tensordot(normalized, values, axes=1)


def moments(model: Model, r: Any, n_nodes: int | None = None) -> dict[str, Any]:
    r"""Return the second velocity moments of a superposed model at 3D radii.

    Parameters
    ----------
    model : Model
        A superposed model (`superpose`) whose components share one spatially
        finite density.
    r : array-like, shape (n,)
        User-specified 3D radii, in internal units whatever the model's unit
        system.
    n_nodes : int, optional
        The number of Gauss-Legendre nodes of each component's Jeans
        integral. Default: ``None``, which means the value of the model's
        setting ``observables.n_jeans_nodes``.

    Returns
    -------
    moments : dict
        Arrays in internal units, whatever the model's unit system:
        ``sigma_r2`` and ``sigma_t2``, the squared radial and tangential
        dispersions of the superposed model, and ``beta_sigma``, the
        superposed model's velocity anisotropy, each with the shape of `r`;
        ``sigma_r2_components`` and ``beta_components``, the squared radial
        dispersion and the anisotropy of each component, one row per
        component along the first axis, each row with the shape of `r`.
        ``beta_sigma`` is NaN at a radius where ``sigma_r2`` is not positive.

    Raises
    ------
    InputError
        If `model` is not a `Model`.
    InputError
        If the density of `model` is not spatially finite.

    See Also
    --------
    superpose : The superposed model.
    mixture : The weighted sum the moments are formed by.

    Notes
    -----
    The squared radial dispersion :math:`\sigma_{r,k}^2` of component
    :math:`k` is the Jeans integral of the component's augmented density on
    the shared profile, evaluated by a Gauss-Legendre rule in the variable
    :math:`\ln(1 - \mathcal{K}/P_0)`, with :math:`\mathcal{K}` the binding
    energy and :math:`P_0` the central potential. Because the components
    share one density, the moments of the superposed model are the weighted
    sums

    .. math::

        \sigma_r^2 = \sum_k w_k \sigma_{r,k}^2, \qquad
        \sigma_t^2 = 2 \sum_k w_k (1 - \beta_k) \sigma_{r,k}^2, \qquad
        \beta_\sigma = 1 - \frac{\sigma_t^2}{2 \sigma_r^2},

    with :math:`w_k` the normalized weight and :math:`\beta_k` the
    velocity anisotropy of component :math:`k`.

    References
    ----------
    Chiang+(2026d), Eq. (B43), Chiang+(2026d), Eq. (B44) and
    Chiang+(2026d), Eq. (B74).

    Examples
    --------
    >>> import pianistpy as pp
    >>> settings = pp.Settings(quick=True)
    >>> profile = pp.DoublePowerLaw.nfw()
    >>> parent = pp.solve(pp.Component(profile), pp.Truncation(k_t=0.05), settings)
    >>> children = [
    ...     pp.solve(
    ...         pp.Component(parent.profile, pp.Anisotropy(beta_c=b)), None, settings
    ...     )
    ...     for b in (0.0, 0.2)
    ... ]
    >>> model = pp.superpose({children[0]: 0.5, children[1]: 0.5})
    >>> result = pp.superposition.moments(model, [0.5, 1.0])
    >>> result["sigma_r2"].shape, result["sigma_r2_components"].shape
    ((2,), (2, 2))
    """
    if not isinstance(model, Model):
        raise InputError(
            "moments takes a Model, such as the Model that superpose returns."
        )
    r = np.asarray(r, dtype=np.float64)
    profile = model.profile
    if n_nodes is None:
        n_nodes = int(model.settings.observables["n_jeans_nodes"])
    p_outer = float(profile.p_trunc[-1])
    if not model.spatially_finite:
        raise InputError(
            "moments takes a Model built on a spatially finite potential-density "
            "pair; the Model given is built on a pair that is not spatially finite."
        )
    rows_sigma, rows_beta = [], []
    for index, component in enumerate(model.components):

        def density(radii: Any, index: int = index) -> Any:
            return profile.density(radii)[index]

        s2, _ = radial_dispersion(
            r,
            profile.potential,
            density,
            profile.radius,
            component.anisotropy,
            p_outer,
            0.0,
            n_nodes,
        )
        rows_sigma.append(s2)
        rows_beta.append(beta_sigma(r, component.anisotropy))
    weights = np.array([c.mass for c in model.components])
    sigma_r2, beta = np.array(rows_sigma), np.array(rows_beta)
    total = mixture(weights, sigma_r2)
    tangential = 2.0 * mixture(weights, (1.0 - beta) * sigma_r2)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta_mix = np.where(total > 0.0, 1.0 - tangential / (2.0 * total), np.nan)
    return dict(
        sigma_r2=total,
        sigma_t2=tangential,
        beta_sigma=beta_mix,
        sigma_r2_components=sigma_r2,
        beta_components=beta,
    )


def superpose(members: Mapping[Model, float]) -> Model:
    r"""Return the superposition of models built on one truncated profile.

    The returned model is an equilibrium of the shared pair, because every
    member's distribution function was inverted from the shared density.

    Parameters
    ----------
    members : mapping of Model to float
        Two or more models, each of one component and each built on the same
        truncated profile (for example, Method II children of one truncated
        parent), mapped to the models' weights. Each weight is positive and
        finite. The weights are normalized to sum to 1, with a logged warning
        where the weights' sum differs from 1 by more than :math:`10^{-10}`.

    Returns
    -------
    model : Model
        A model with one component per member, in the order of `members`.
        Each component is the member's component, with a mass equal to the
        member's weight divided by the first member's weight, and with the
        member's DF table on the member's own energy grid. The model's
        profile is the shared truncated profile with the density split among
        the components by weight. Each check of the model's
        `ValidationReport` holds the largest error of the check over the
        members.

    Raises
    ------
    InputError
        If `members` is not a mapping.
    InputError
        If fewer than two members are given.
    InputError
        If a member is not a `Model`, or has more than one component.
    InputError
        If a member's profile is not a truncated profile with
        :math:`\mathcal{K}_{\rm t} > 0`.
    InputError
        If a weight is not positive and finite.
    InputError
        If the members are built on different truncated profiles.
    InputError
        If the members were computed with different settings, or are in
        different units.
    InputError
        If a member's DF is signed.
    InputError
        If the members' central values of the potential differ.
    InputError
        If some members' distribution functions come from closed forms
        (directly or through a closed-form parent) and the other members'
        distribution functions were computed by inversion. Build every member
        the same way, from closed forms or by inversion, to superpose the
        members.
    InputError
        If the members' validation reports hold different sets of checks.

    See Also
    --------
    moments : The velocity moments of the superposed model.
    mixture : The weighted sum of per-component quantities.

    References
    ----------
    Chiang+(2026d), Appendix B5; Notes Section T:4.

    Examples
    --------
    >>> import pianistpy as pp
    >>> settings = pp.Settings(quick=True)
    >>> profile = pp.DoublePowerLaw.nfw()
    >>> parent = pp.solve(pp.Component(profile), pp.Truncation(k_t=0.05), settings)
    >>> children = [
    ...     pp.solve(
    ...         pp.Component(parent.profile, pp.Anisotropy(beta_c=b)), None, settings
    ...     )
    ...     for b in (0.0, 0.2)
    ... ]
    >>> model = pp.superpose({children[0]: 0.5, children[1]: 0.5})
    >>> len(model.components), [c.mass for c in model.components]
    (2, [1.0, 1.0])
    """
    models, weights = _checked(members)
    first = models[0]
    components = tuple(
        dataclasses.replace(model.components[0], mass=float(w / weights[0]))
        for model, w in zip(models, weights, strict=True)
    )
    identity = model_identity(components, None, first.settings)
    return Model(
        components=components,
        truncation=None,
        settings=first.settings,
        identity=identity,
        parent=identity,
        table=_stack_tables([m.table for m in models]),
        form=_stack_tuples(AsymptoticForm, [m.form for m in models]),
        profile=_shared_profile(models, weights, identity),
        validation=_worst_report([m.validation for m in models]),
        timing={},
        r_trust={},
        inversion=_stack_inversion([m.inversion for m in models]),
        units=first.units,
    )


def _checked(members: Mapping[Model, float]) -> tuple[list[Model], np.ndarray]:
    """Return the members in order and the members' normalized weights.

    A member that fails a check raises `InputError` with a message that gives
    the member's position in `members`.
    """
    if not isinstance(members, Mapping):
        raise InputError(
            "superpose takes a mapping from each Model to the Model's weight."
        )
    models, raw = list(members), np.array([float(w) for w in members.values()])
    if len(models) < 2:
        raise InputError(
            "superpose needs at least two models; the mapping given holds "
            f"{len(models)}."
        )
    for index, model in enumerate(models):
        if not isinstance(model, Model):
            raise InputError(f"Member {index} of the mapping is not a Model.")
        if len(model.components) != 1:
            raise InputError(
                f"Member {index} has {len(model.components)} components; every member "
                "of a superposition must be a model of one component."
            )
        profile = model.components[0].profile
        if not isinstance(profile, TruncatedProfile) or not profile.k_t > 0.0:
            raise InputError(
                f"Member {index} is not built on a truncated profile with K_t > 0; "
                "superpose takes Method II models of one truncated parent."
            )
    if not np.all(np.isfinite(raw) & (raw > 0.0)):
        raise InputError(
            f"Every weight must be positive and finite; the weights given are "
            f"{raw.tolist()}."
        )
    # The loop over the members has checked that every member's profile is a
    # TruncatedProfile, so the casts to TruncatedProfile hold.
    shared = cast(TruncatedProfile, models[0].components[0].profile)
    for index, model in enumerate(models[1:], start=1):
        profile = cast(TruncatedProfile, model.components[0].profile)
        if profile != shared:
            raise InputError(
                f"Member {index} is built on the truncated profile of model "
                f"{profile.model}, and member 0 on the truncated profile of model "
                f"{shared.model}; the members must share one truncated pair."
            )
        if model.settings != models[0].settings:
            raise InputError(
                f"Member {index} was computed with other settings than the settings "
                "of member 0."
            )
        if model.units != models[0].units:
            raise InputError(
                f"Member {index} is in other units than the units of member 0."
            )
    total = float(raw.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_TOLERANCE):
        log.warning(
            "The weights given sum to %.12g, not to 1; superpose normalized the "
            "weights to sum to 1.",
            total,
        )
    return models, raw / total


def _pad(rows: list[np.ndarray], fill: str) -> np.ndarray:
    """Return the arrays of `rows` stacked along the first axis, each padded
    at the end to the length of the longest array.

    The padding repeats an array's last value when `fill` is ``"last"`` and
    is NaN otherwise."""
    width = max(row.shape[-1] for row in rows)
    out = []
    for row in rows:
        pad = width - row.shape[-1]
        if pad and fill == "last":
            tail = np.repeat(row[..., -1:], pad, axis=-1)
        else:
            tail = np.full(row.shape[:-1] + (pad,), np.nan)
        out.append(np.concatenate([row, tail], axis=-1))
    return np.concatenate(out, axis=0)


def _stack_tables(tables: list[DfTable]) -> DfTable:
    """Return the members' DF tables stacked along the component axis.

    Each member's table keeps the member's own energy grid. Beyond a member's
    last energy, the member's grid repeats the last energy and the member's
    values are NaN, the rule a `DfTable` follows for grids of different
    lengths. Below the table, each member keeps the member's own number of
    nodes. A signed member table raises `InputError`."""
    if any(t.sign.size for t in tables):
        raise InputError(
            "A member's DF is signed; superpose takes only members whose DF is "
            "non-negative."
        )
    has_below = [t.k_below.size > 0 for t in tables]
    if any(has_below):
        k_below = _pad([t.k_below for t in tables], "last")
        log_below = _pad([t.log_df_below for t in tables], "nan")
        slope_below = _pad([t.log_df_slope_below for t in tables], "nan")
        n_below = tuple(n for t in tables for n in _nodes_below(t))
    else:
        k_below = log_below = slope_below = np.zeros((len(tables), 0))
        n_below = tuple(0 for t in tables for _ in t.i_first)
    return DfTable(
        k=_pad([t.k for t in tables], "last"),
        log_df=_pad([t.log_df for t in tables], "nan"),
        log_df_slope=_pad([t.log_df_slope for t in tables], "nan"),
        i_first=tuple(i for t in tables for i in t.i_first),
        i_last=tuple(i for t in tables for i in t.i_last),
        slopes_exact=tuple(s for t in tables for s in t.slopes_exact),
        k_below=k_below,
        log_df_below=log_below,
        log_df_slope_below=slope_below,
        n_below=n_below,
    )


def _nodes_below(table: DfTable) -> tuple[int, ...]:
    """Return the number of nodes below the table of each component of `table`."""
    return tuple(table.nodes_below(c) for c in range(len(table.i_first)))


def _stack_tuples(cls: type, items: list[Any]) -> Any:
    """Return one record of class `cls` that combines the members' records.

    A field that holds a tuple, one entry per component, is the
    concatenation of the members' tuples. A field that holds one value for
    the whole model, such as `AsymptoticForm.p_0`, must hold the same value
    in every member, and a field whose values differ raises `InputError`."""
    values = {}
    for f in dataclasses.fields(cls):
        got = [getattr(item, f.name) for item in items]
        if isinstance(got[0], tuple):
            values[f.name] = tuple(x for g in got for x in g)
        elif all(g == got[0] for g in got):
            values[f.name] = got[0]
        else:
            raise InputError(
                f"The members differ in {cls.__name__}.{f.name} ({got}), so the "
                "members do not share one truncated pair."
            )
    return cls(**values)


def _stack_inversion(records: list[InversionRecord | None]) -> InversionRecord | None:
    """Return the members' inversion records combined, or None where no member has one.

    A member whose DF comes from a closed form has no inversion record. A
    mixture of members with and without an inversion record raises
    `InputError`."""
    if all(r is None for r in records):
        return None
    if any(r is None for r in records):
        raise InputError(
            "Some members' DFs come from closed forms and the other members' DFs "
            "were computed by inversion; superpose takes members built the same "
            "way, all from closed forms or all by inversion."
        )
    return _stack_tuples(InversionRecord, records)


def _shared_profile(
    models: list[Model], weights: np.ndarray, identity: str
) -> TruncatedProfile:
    """Return the members' shared profile with one density row per member.

    Each density row is the shared density times the member's weight, so the
    rows sum to the shared density. A member whose profile differs from the
    first member's profile raises `InputError`."""
    first = models[0].profile
    for index, model in enumerate(models[1:], start=1):
        p = model.profile
        same = all(
            np.array_equal(getattr(p, name), getattr(first, name))
            for name in ("r", "p_trunc", "rho_trunc", "mass_trunc", "u_trunc")
        ) and (p.k_t, p.r_t, p.mass_t, p.spatially_finite) == (
            first.k_t,
            first.r_t,
            first.mass_t,
            first.spatially_finite,
        )
        if not same:
            raise InputError(
                f"The tabulated profile of member {index} differs from the tabulated "
                "profile of member 0, so the members do not share one truncated pair."
            )
    rho = np.concatenate(
        [w * m.profile.rho_trunc for m, w in zip(models, weights, strict=True)]
    )
    # Every component's density is the shared density times the component's
    # weight, so every component takes the shared density's edge exponent, as
    # the parent's truncated profile records the edge exponent.
    parent = cast(TruncatedProfile, models[0].components[0].profile)
    edge = parent.edge_exponent[:1] or first.edge_exponent[:1]
    return dataclasses.replace(
        first,
        rho_trunc=rho,
        model=identity,
        source_anisotropy=models[0].components[0].anisotropy,
        edge_exponent=edge * len(models),
        # The central deficit is recorded per component when every member
        # records exactly one central deficit, and is not recorded otherwise.
        central_deficit=(
            tuple(d for m in models for d in m.profile.central_deficit)
            if all(len(m.profile.central_deficit) == 1 for m in models)
            else ()
        ),
    )


def _worst_report(reports: list[ValidationReport]) -> ValidationReport:
    """Return a validation report whose checks hold the members' largest errors.

    Each check of the returned report holds the largest error of the check
    over the members. The report's parents' level is the highest level over
    the members' own levels and the members' parents' levels. Members whose
    reports hold different sets of checks raise `InputError`."""
    keys = reports[0].keys
    if any(r.keys != keys for r in reports):
        raise InputError(
            "The members' validation reports hold different sets of checks."
        )
    checks: dict[str, Check] = {}
    for key in keys:
        present = [r.checks[key] for r in reports if key in r.checks]
        if present:
            checks[key] = max(present, key=lambda c: c.error)
    return ValidationReport(
        keys=keys,
        checks=checks,
        parents_level=max(max(r.level, r.parents_level) for r in reports),
    )
