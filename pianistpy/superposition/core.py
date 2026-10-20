"""The superposition of models that share one truncated potential-density pair.

Several distribution functions inverted from one truncated pair (Method II
children of one parent) are an equilibrium in that pair for any positive
weights that sum to one (Chiang+(2026d), Appendix B5). `superpose` returns that
equilibrium as a `Model`: its components are the members' components, each
with its own DF table on its own energy grid, and its density is the shared
one, split among the components by weight.
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

# The weights are normalized to sum to one; a sum this far from one says so.
WEIGHT_SUM_TOLERANCE = 1e-10


def mixture(weights: Any, values: Any, xp: Any = np) -> Any:
    """Return the weighted sum of per-component values, the weights normalized.

    The one way a superposed model combines its components: a density, a
    moment or a table read per component. Linear in the values, and in the
    weights up to their normalization; with ``xp=jax.numpy`` the weights may
    be traced, so a quantity of the mixture is differentiable in them.

    Parameters
    ----------
    weights : array-like, shape (n_components,)
        Positive weights; normalized here.
    values : array-like, shape (n_components, ...)
        The value of each component.
    xp : module, optional
        ``numpy``, or ``jax.numpy`` for traced weights.

    Returns
    -------
    value : array, shape (...)
    """
    weights = xp.asarray(weights)
    values = xp.asarray(values)
    normalized = weights / xp.sum(weights)
    return xp.tensordot(normalized, values, axes=1)


def moments(model: Model, r: Any, n_nodes: int | None = None) -> dict[str, Any]:
    r"""Return the second moments of a superposed model at radii `r`.

    Parameters
    ----------
    model : Model
        A superposed model (`superpose`), its components sharing one density.
    r : array-like, shape (n,)
        Radii, in internal units.
    n_nodes : int, optional
        Gauss-Legendre nodes of each component's Jeans integral; the model's
        ``observables.n_jeans_nodes`` if omitted.

    Returns
    -------
    moments : dict
        ``sigma_r2`` and ``sigma_t2`` of the mixture, ``beta_sigma`` its
        anisotropy, and ``sigma_r2_components`` and ``beta_components`` (one
        row per component), with the shape of `r`.

    Notes
    -----
    Each component's radial dispersion is the Jeans integral of its
    auxiliary density on the shared profile (`radial_dispersion`, the
    Gauss-Legendre rule in ``ln(1 - K / P_0)``); its components share the density,
    so the moments of the mixture are the weighted sums
    :math:`\sigma_r^2 = \sum_k w_k\sigma_{r,k}^2`,
    :math:`\sigma_t^2 = 2\sum_k w_k(1-\beta_k)\sigma_{r,k}^2` and
    :math:`\beta_\sigma = 1 - \sigma_t^2 / (2\sigma_r^2)`.

    References
    ----------
    Chiang+(2026d), Eqs. (B42), (B43) [eqn:linearity_2, eqn:linearity_beta_1] and
    eqn:beta_sigma_sup.
    """
    if not isinstance(model, Model):
        raise InputError("moments takes a Model (`superpose`).")
    r = np.asarray(r, dtype=np.float64)
    profile = model.profile
    if n_nodes is None:
        n_nodes = int(model.settings.observables["n_jeans_nodes"])
    p_outer = float(profile.p_trunc[-1])
    if not model.spatially_finite:
        raise InputError(
            "The moments of a superposition are formed on a spatially finite pair; "
            "this one is not."
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
    """Return the superposition of models that share one truncated pair.

    Parameters
    ----------
    members : mapping of Model to float
        Two or more models of one component each, every one built on the same
        truncated profile (Method II children of one truncated parent), with
        their weights. The weights must be positive; they are normalized to
        sum to one, with a warning where they did not.

    Returns
    -------
    model : Model
        One component per member, in the order given: the member's component
        with its mass the weight relative to the first member's; its DF table
        on its own energy grid (`DfTable` carries one grid per component); the
        shared profile with the density split by weight; the members' checks,
        each at its worst over the members.

    Raises
    ------
    InputError
        If fewer than two members are given, a weight is not positive and
        finite, a member is not a single Method II component on a truncated
        profile, the members are built on different profiles, or their
        settings or units differ.
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
    """The members in order and their normalized weights, or a refusal by name."""
    if not isinstance(members, Mapping):
        raise InputError("superpose takes a mapping of Model to weight.")
    models, raw = list(members), np.array([float(w) for w in members.values()])
    if len(models) < 2:
        raise InputError(
            f"A superposition needs at least two models; got {len(models)}."
        )
    for index, model in enumerate(models):
        if not isinstance(model, Model):
            raise InputError(f"Member {index} is not a Model.")
        if len(model.components) != 1:
            raise InputError(
                f"Member {index} has {len(model.components)} components; each member "
                "is a model of one component."
            )
        profile = model.components[0].profile
        if not isinstance(profile, TruncatedProfile) or not profile.k_t > 0.0:
            raise InputError(
                f"Member {index} is not built on a truncated profile: a superposition "
                "takes Method II models of one truncated parent (K_t > 0)."
            )
    if not np.all(np.isfinite(raw) & (raw > 0.0)):
        raise InputError(
            f"The weights must be positive and finite; got {raw.tolist()}."
        )
    # every member's profile is a TruncatedProfile (checked above)
    shared = cast(TruncatedProfile, models[0].components[0].profile)
    for index, model in enumerate(models[1:], start=1):
        profile = cast(TruncatedProfile, model.components[0].profile)
        if profile != shared:
            raise InputError(
                f"Member {index} is built on the profile of model {profile.model}, "
                f"member 0 on that of {shared.model}: the members must share one "
                "truncated pair."
            )
        if model.settings != models[0].settings:
            raise InputError(
                f"Member {index} was computed with other settings than member 0."
            )
        if model.units != models[0].units:
            raise InputError(f"Member {index} is in other units than member 0.")
    total = float(raw.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_TOLERANCE):
        log.warning("The weights sum to %.12g, not 1: normalized.", total)
    return models, raw / total


def _pad(rows: list[np.ndarray], fill: str) -> np.ndarray:
    """Rows of one axis stacked, the shorter ones padded at the end by their
    last value (``fill="last"``) or by NaN."""
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
    """The members' tables on the component axis, each on its own grid: beyond a
    member's last energy its grid repeats that energy and its values are NaN
    (`DfTable`'s rule); below the table, its own count of nodes."""
    if any(t.sign.size for t in tables):
        raise InputError("A member's DF is signed: a model's DF is non-negative.")
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
    return tuple(table.nodes_below(c) for c in range(len(table.i_first)))


def _stack_tuples(cls: type, items: list[Any]) -> Any:
    """A per-component record: its tuples, the members' concatenated; a field
    of the model as a whole (`AsymptoticForm.p_0`) the same in every member."""
    values = {}
    for f in dataclasses.fields(cls):
        got = [getattr(item, f.name) for item in items]
        if isinstance(got[0], tuple):
            values[f.name] = tuple(x for g in got for x in g)
        elif all(g == got[0] for g in got):
            values[f.name] = got[0]
        else:
            raise InputError(
                f"The members' {cls.__name__}.{f.name} differ ({got}): the members "
                "do not share one truncated pair."
            )
    return cls(**values)


def _stack_inversion(records: list[InversionRecord | None]) -> InversionRecord | None:
    if all(r is None for r in records):
        return None
    if any(r is None for r in records):
        raise InputError(
            "Some members state how their table was made and some do not (a "
            "closed form among inverted tables)."
        )
    return _stack_tuples(InversionRecord, records)


def _shared_profile(
    models: list[Model], weights: np.ndarray, identity: str
) -> TruncatedProfile:
    """The members' common profile with one density row per member, the shared
    density times the member's weight, so that the rows sum to it."""
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
                f"The profile of member {index} differs from member 0's in its rows: "
                "the members do not share one truncated pair."
            )
    rho = np.concatenate(
        [w * m.profile.rho_trunc for m, w in zip(models, weights, strict=True)]
    )
    # every component's density is the shared one times its weight, so each
    # takes the shared density's edge exponent, as the parent's profile states it
    parent = cast(TruncatedProfile, models[0].components[0].profile)
    edge = parent.edge_exponent[:1] or first.edge_exponent[:1]
    return dataclasses.replace(
        first,
        rho_trunc=rho,
        model=identity,
        source_anisotropy=models[0].components[0].anisotropy,
        edge_exponent=edge * len(models),
        # per component where every member records it, else not recorded
        central_deficit=(
            tuple(d for m in models for d in m.profile.central_deficit)
            if all(len(m.profile.central_deficit) == 1 for m in models)
            else ()
        ),
    )


def _worst_report(reports: list[ValidationReport]) -> ValidationReport:
    """Each check at its worst over the members, the members' levels as parents'."""
    keys = reports[0].keys
    if any(r.keys != keys for r in reports):
        raise InputError("The members' reports hold different checks.")
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
