"""The calls that compute distribution functions and equilibrium models.

`compute_df` returns the untruncated distribution function (DF) of a
component, whatever the DF's sign. `truncate` builds an equilibrium model
from that DF. `solve` calls `compute_df` and then `truncate`.
`model_record` and `model_identity` state what determines a result and the
identity string derived from it.
"""

__all__ = ["compute_df", "model_identity", "model_record", "solve", "truncate"]

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .._version import __version__
from ..exceptions import InputError
from ..identity import identity_string
from ..profiles import TruncatedProfile
from ..runtime import log, timing_record
from ..settings import Settings
from .inputs import Component, Truncation
from .results import DistributionFunction, Model

# The largest number of components that `compute_df`, `solve` and
# `DistributionFunction.from_function` accept. The result classes and the
# file format hold several components.
N_COMPONENTS_MAX = 1


def model_record(
    components: Sequence[Component],
    truncation: Truncation | None,
    settings: Settings,
) -> dict[str, Any]:
    """Return the dictionary of the inputs and settings that determine a result.

    Parameters
    ----------
    components : sequence of Component
        Components of the model.
    truncation : Truncation or None
        Truncation of the model. None for an untruncated model and for a
        distribution function.
    settings : Settings
        Numerical settings of the computation.

    Returns
    -------
    record : dict
        Dictionary with the key ``"inputs"`` (the components and the
        truncation as dictionaries) and the key ``"settings"`` (the output
        of `Settings.numerical`). The identity string is a digest of the
        record dictionary. A result's file records the full settings
        (`Settings.to_dict`), but ``"settings"`` holds only the settings
        that change the numbers. A result computed with
        ``write_anyway=True`` therefore has the identity string of a result
        computed from equal inputs and settings without ``write_anyway``.
    """
    return {
        "inputs": {
            "components": [component.to_dict() for component in components],
            "truncation": None if truncation is None else truncation.to_dict(),
        },
        "settings": settings.numerical(),
    }


def model_identity(
    components: Sequence[Component],
    truncation: Truncation | None,
    settings: Settings,
) -> str:
    """Return the identity string of a model or of a distribution function.

    Parameters
    ----------
    components : sequence of Component
        Components of the model.
    truncation : Truncation or None
        Truncation of the model. None for an untruncated model and for a
        distribution function.
    settings : Settings
        Numerical settings of the computation.

    Returns
    -------
    identity : str
        Digest of the inputs, of the numerical settings (`model_record`),
        of the package version and of the version of the file format.
    """
    record = model_record(components, truncation, settings)
    return identity_string(record["inputs"], record["settings"], __version__)


def _as_components(
    components: Component | Sequence[Component],
) -> tuple[Component, ...]:
    """Return one component, or a sequence of components, as a tuple.

    Raises
    ------
    InputError
        If the sequence is empty, holds an item that is not a `Component`,
        or holds more than `N_COMPONENTS_MAX` components.
    """
    if isinstance(components, Component):
        return (components,)
    result = tuple(components)
    if not result or not all(isinstance(item, Component) for item in result):
        raise InputError("A model needs at least one Component.")
    if len(result) > N_COMPONENTS_MAX:
        raise InputError(
            f"PIANISTpy computes models of one component; got {len(result)}."
        )
    return result


def compute_df(
    components: Component | Sequence[Component],
    settings: Settings | None = None,
) -> DistributionFunction:
    r"""Return the untruncated distribution function of a component.

    The distribution function (DF) is returned whatever its sign. The
    result states where the DF is negative, and the lowest truncation
    energy :math:`\mathcal{K}_{\rm t}` that gives a valid equilibrium. Call
    `solve` for the model itself. Call `compute_df` to inspect the DF before
    choosing a truncation, or to truncate one DF in several ways.

    Parameters
    ----------
    components : Component or sequence of Component
        The component of the model, or a sequence that holds exactly one
        `Component`.
    settings : Settings, optional
        Numerical settings. Default: ``None``, which means ``Settings()``,
        the default settings.

    Returns
    -------
    df : DistributionFunction
        The DF of the component, tabulated on an energy grid, with its
        analytic forms below and above the energies of the table, the
        lowest admissible truncation energy ``k_t_min`` in internal units,
        and the validation report of the DF.

    Raises
    ------
    InputError
        If `components` is an empty sequence, a sequence with an item that
        is not a `Component`, or a sequence of more than one component.
    InputError
        If the component is built on a `TruncatedProfile` whose model has an
        anisotropy other than isotropic or Osipkov-Merritt Type I, or whose
        model carries no DF.
    InputError
        If the component is built on a `TruncatedProfile`, has a Type II
        anisotropy, and the edge of the `TruncatedProfile` does not lie
        inside the component's anisotropy radius :math:`R_{\rm a}`.
    InputError
        If the component is built on a `TruncatedProfile` and the
        component's central anisotropy :math:`\beta_{\rm c}` is below
        :math:`-1` and is not :math:`-3/2`.
    UnphysicalModelError
        If the component has a Cuddeford-Osipkov-Merritt (COM) Type II
        anisotropy and the integrand of the DF's inversion is not
        integrable at the anisotropy radius :math:`R_{\rm a}`.
    UnphysicalModelError
        If the component is built on a `TruncatedProfile` and the central
        density slope :math:`\gamma` is below :math:`2\beta_{\rm c}`.
    UnphysicalModelError
        If the component is built on the `TruncatedProfile` of a model that
        was truncated at :math:`\mathcal{K}_{\rm t} > 0` but has no edge.
    AccuracyError
        If the inversion cannot resolve the DF at the resolution of
        `settings`. ``Settings(write_anyway=True)`` does not turn this error
        into a warning.

    Warns
    -----
    AccuracyWarning
        If the DF is non-negative on its table and a check of the DF's
        validation report lies above :math:`10^{-4}`.

    See Also
    --------
    solve : The equilibrium model of a component in one call.
    truncate : The equilibrium model built from a distribution function.
    DistributionFunction.from_function : A DF given as a function of the
        energy.

    Notes
    -----
    The DF has the separable form
    :math:`F(\mathcal{K}, \mathcal{L}) = F_{\mathcal{K}}(\mathcal{K})\,
    \mathcal{L}^{-2\beta_{\rm c}}`, with :math:`\mathcal{K}` the
    energy-like variable and :math:`\mathcal{L}` the angular momentum, both
    in internal units. The result tabulates
    :math:`F_{\mathcal{K}}(\mathcal{K})`.

    References
    ----------
    Chiang+(2026d), Eq. (8). Notes Section T:2.1.

    Examples
    --------
    >>> import pianistpy as pp
    >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> df.is_physical, df.k_t_min
    (True, 0.0)
    """
    found = _as_components(components)
    from .invert import inversion_df

    settings = Settings() if settings is None else settings
    # The stages run inside the timing record, so that the DF carries the
    # elapsed time of each stage.
    with timing_record() as times:
        stages: tuple[Any, ...]
        if isinstance(found[0].profile, TruncatedProfile):
            from .tabulated import tabulated_df

            stages = tabulated_df(found[0], settings)
        else:
            stages = inversion_df(found[0], settings)
    return _df_result(found, settings, stages, times.stages)


def df_from_function(
    component: Component,
    function: Callable[[np.ndarray, np.ndarray], Any],
    settings: Settings | None = None,
) -> DistributionFunction:
    """Return the untruncated DF that a user-specified function of K gives.

    `DistributionFunction.from_function` documents the parameters, the
    result and the errors.
    """
    from .invert import inversion_df

    found = _as_components(component)
    if isinstance(found[0].profile, TruncatedProfile):
        raise InputError(
            "A DF from a function is tabulated on the grid of an analytic "
            "profile; the component is built on a tabulated profile."
        )
    if not callable(function):
        raise InputError("The DF must be a callable of K; got a non-callable.")
    settings = Settings() if settings is None else settings
    with timing_record() as times:
        stages = inversion_df(found[0], settings, function=function)
    name = str(getattr(function, "__name__", type(function).__name__))
    # The result stores the table of the function's values and not the
    # function itself. The identity therefore digests the table's values on
    # the grid and the values on the nodes below the written floor.
    log_df = np.ascontiguousarray(stages[1].log_df, dtype=np.float64)
    values = hashlib.sha256(log_df.tobytes())
    values.update(np.ascontiguousarray(stages[1].log_df_below, np.float64).tobytes())
    record = model_record(found, None, settings)
    record["inputs"]["function"] = {"name": name, "values": values.hexdigest()}
    identity = identity_string(record["inputs"], record["settings"], __version__)
    return _df_result(found, settings, stages, times.stages, identity, name)


def _df_result(
    found: tuple[Component, ...],
    settings: Settings,
    stages: tuple[Any, ...],
    times: dict[str, float],
    identity: str | None = None,
    function_name: str | None = None,
) -> DistributionFunction:
    """Return the DF built from the stage results, after announcing its report.

    The report is announced only for a DF whose table has no sign column,
    that is, a DF that is non-negative on its table.
    """
    from ..validation import announce

    grid, table, form, k_t_min, report, inversion, rows = stages
    # A table with a sign column holds a DF that is negative somewhere. The
    # checks of such a DF are not measured against a valid equilibrium, so
    # its report is not announced.
    if table.sign.size == 0:
        detail = "; ".join(
            f"{key} {check.error:.2e}"
            + ("" if check.where is None else f" at {check.where:.6g}")
            for key, check in report.checks.items()
            if check.level > 0
        )
        announce(f"The distribution function of {found[0].name!r}", report, detail)
    return DistributionFunction(
        components=found,
        settings=settings,
        identity=model_identity(found, None, settings)
        if identity is None
        else identity,
        grid=grid,
        table=table,
        form=form,
        k_t_min=k_t_min,
        validation=report,
        timing=dict(times),
        inversion=inversion,
        rows=rows,
        source="inversion" if function_name is None else "function",
        function_name=function_name,
    )


def truncate(
    df: DistributionFunction,
    truncation: Truncation | None = None,
) -> Model:
    r"""Return the equilibrium model built from a distribution function.

    With a `Truncation`, the model's DF is the untruncated DF shifted in
    energy by the truncation energy :math:`\mathcal{K}_{\rm t}` and lowered
    by the untruncated DF's value at :math:`\mathcal{K}_{\rm t}`, which
    gives a model of finite mass and finite radius (Method I of
    Chiang+(2026d)). Without a truncation, the untruncated DF is the
    model's DF. An untruncated DF is a valid equilibrium only if
    `DistributionFunction.is_physical` is true.

    Parameters
    ----------
    df : DistributionFunction
        The untruncated DF, as `compute_df` returns it.
    truncation : Truncation, optional
        The truncation. Default: ``None``, which means the untruncated
        model. For a component built on a `TruncatedProfile`, ``None``
        means the model that ends at the edge of that `TruncatedProfile`.

    Returns
    -------
    model : Model
        The equilibrium model: its non-negative DF, its self-consistent
        density and potential (a `TruncatedProfile`), and its validation
        report.

    Raises
    ------
    InputError
        If `df` was built by `DistributionFunction.from_function`. Such a DF
        holds the function's values only, and a truncation needs the values
        of the inversion.
    UnphysicalModelError
        If the truncation energy :math:`\mathcal{K}_{\rm t}` is below
        ``df.k_t_min``, or no admissible truncation energy reaches the
        target mass or the target radius of `truncation`.
    UnphysicalModelError
        If the central density slope :math:`\gamma` is below
        :math:`2\beta_{\rm c}(0)`, with :math:`\beta_{\rm c}(0)` the
        central anisotropy (0 for the isotropic and Osipkov-Merritt
        models), or if the DF is negative or tends to minus infinity at the
        center.
    UnphysicalModelError
        If :math:`\gamma = 2\beta_{\rm c}(0)` and the DF has no closed form
        at the center to join the table of the DF.
    UnphysicalModelError
        If the truncated model has no edge, or the edge of a Type II model
        lies at or beyond the anisotropy radius :math:`R_{\rm a}`.
    UnphysicalModelError
        If `truncation` is ``None`` and the untruncated DF is not a valid
        equilibrium (``df.is_physical`` is false), or the untruncated mass
        diverges.
    UnphysicalModelError
        If the component is built on a `TruncatedProfile` and the DF is
        negative at an energy inside the DF's support.
    AccuracyError
        If a check of the model's validation report lies above
        :math:`10^{-2}` (the band "refused"), the round trip of the model's
        DF against the model's density included. With
        ``Settings(write_anyway=True)`` in the settings of `df`, the model is
        returned with an `AccuracyWarning` instead.
    AccuracyError
        If the truncated DF, or the potential solved from the truncated DF,
        is not finite at the resolution of the settings.
        ``write_anyway=True`` does not turn this error into a warning.

    Warns
    -----
    AccuracyWarning
        If a check of the model's validation report lies above
        :math:`10^{-4}`.

    See Also
    --------
    compute_df : The untruncated distribution function.
    solve : `compute_df` and `truncate` in one call.
    Truncation : The truncation energy, target mass or target radius.

    Notes
    -----
    An `UnphysicalModelError` raised by `truncate` holds a result in its
    ``result`` attribute. The ``result`` attribute holds the model without
    an edge where the truncation gives no edge, and `df` otherwise.

    The truncated DF is

    .. math::

        \hat{F}(\hat{\mathcal{K}}, \mathcal{L}) =
        F(\hat{\mathcal{K}} + \mathcal{K}_{\rm t}, \mathcal{L})
        - F(\mathcal{K}_{\rm t}, \mathcal{L})

    for :math:`\hat{\mathcal{K}} = \mathcal{K} - \mathcal{K}_{\rm t} \ge 0`,
    and :math:`\hat{F} = 0` for :math:`\hat{\mathcal{K}} < 0`. Here
    :math:`F` is the untruncated DF, :math:`\mathcal{K}` the energy-like
    variable and :math:`\mathcal{L}` the angular momentum, all in internal
    units. The truncated relative potential :math:`\hat{P}(R)` solves
    Poisson's equation with the density of :math:`\hat{F}`, from
    :math:`\hat{P}(0) = 1 - \mathcal{K}_{\rm t}` to the truncation radius
    :math:`R_{\rm t}`, where :math:`\hat{P}(R_{\rm t}) = 0`.

    References
    ----------
    Chiang+(2026d), Eqs. (9), (10) and (11). Notes Section T:3.

    Examples
    --------
    >>> import pianistpy as pp
    >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> model = pp.truncate(df, pp.Truncation(k_t=0.05))
    >>> model.spatially_finite
    True
    """
    if df.source == "function":
        raise InputError(
            f"The distribution function of {df.components[0].name!r} comes from a "
            f"function ({df.function_name}): it carries no inversion rows, and "
            "truncation needs them, so it is not truncated."
        )
    if truncation is None:
        if isinstance(df.components[0].profile, TruncatedProfile):
            # A DF on a TruncatedProfile (Method II) gives the model that ends
            # at the edge of that profile.
            from .tabulated import child_model

            model = child_model(df)
            _warn_of_divergence(model)
            return model
        from .truncated import untruncated_model

        model = untruncated_model(df)
        _warn_of_divergence(model)
        return model
    from .truncated import truncate_model

    model = truncate_model(df, truncation)
    _warn_of_divergence(model)
    return model


def _warn_of_divergence(model: Model) -> None:
    """Log a warning for each component whose DF diverges as K goes to zero."""
    for index, diverges in enumerate(model.diverges_at_zero):
        if diverges:
            exponent = model.form.exponent_at_zero(index)
            log.warning(
                "The DF of %r goes as K^%.4g as K -> 0: it diverges there. The model "
                "is a valid equilibrium with a finite density, but its DF is not "
                "monotonic, and its dynamical stability is not established.",
                model.components[index].name,
                exponent,
            )


def solve(
    components: Component | Sequence[Component],
    truncation: Truncation | None = None,
    settings: Settings | None = None,
) -> Model:
    r"""Return the equilibrium model of a component.

    `solve` calls `compute_df` and then `truncate`. Call `compute_df` and
    `truncate` separately to inspect the distribution function (DF) before
    the truncation, or to truncate one DF in several ways.

    Parameters
    ----------
    components : Component or sequence of Component
        The component of the model, or a sequence that holds exactly one
        `Component`.
    truncation : Truncation, optional
        The truncation. Default: ``None``, which means the untruncated
        model, as in `truncate`.
    settings : Settings, optional
        Numerical settings. Default: ``None``, which means ``Settings()``,
        the default settings.

    Returns
    -------
    model : Model
        The equilibrium model, as `truncate` returns it.

    Raises
    ------
    InputError
        If `compute_df` or `truncate` raises `InputError` for the inputs.
    UnphysicalModelError
        If no valid equilibrium exists for the inputs. The Raises sections
        of `compute_df` and `truncate` list the conditions.
    AccuracyError
        If the DF or the model misses the accuracy that `compute_df` and
        `truncate` require.

    Warns
    -----
    AccuracyWarning
        If a check of the DF's or of the model's validation report lies
        above :math:`10^{-4}`.

    See Also
    --------
    compute_df : The untruncated distribution function.
    truncate : The equilibrium model built from a distribution function.
    Component : A density profile with its velocity anisotropy.

    References
    ----------
    Chiang+(2026d), Section 3.1.

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> model.truncation is None, model.spatially_finite
    (True, False)
    """
    return truncate(compute_df(components, settings), truncation)
