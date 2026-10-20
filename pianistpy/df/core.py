"""The public functions of the distribution-function computation.

The computation has two steps. `compute_df` returns the untruncated distribution
function (DF) of each component, whatever its sign. `truncate` builds a
model, a valid equilibrium, from it. `solve` runs both.
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

# The number of components the public calls accept. The results, the stage
# contracts and the files hold several; the calls open to them in a later
# release.
N_COMPONENTS_MAX = 1


def model_record(
    components: Sequence[Component],
    truncation: Truncation | None,
    settings: Settings,
) -> dict[str, Any]:
    """Return the dictionary form of everything that determines a model.

    Parameters
    ----------
    components : sequence of Component
        The components of the model.
    truncation : Truncation or None
        The truncation, or None for an untruncated model or for the
        untruncated distribution function.
    settings : Settings
        The numerical settings.

    Returns
    -------
    record : dict
        The inputs and the numerical settings, as hashed into the identity
        string. A result's file records the settings in full
        (`Settings.to_dict`); the identity takes only what determines the
        numbers, so a result asked for with ``write_anyway`` has the identity
        of the same result without it.
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
    """Return the identity string of a model or of its distribution function.

    Parameters
    ----------
    components : sequence of Component
        The components of the model.
    truncation : Truncation or None
        The truncation, or None.
    settings : Settings
        The numerical settings.

    Returns
    -------
    identity : str
        Digest of the inputs, the settings, the version of the code and the
        version of the file format.
    """
    record = model_record(components, truncation, settings)
    return identity_string(record["inputs"], record["settings"], __version__)


def _as_components(
    components: Component | Sequence[Component],
) -> tuple[Component, ...]:
    """Return one component or a sequence of them as a tuple."""
    if isinstance(components, Component):
        return (components,)
    result = tuple(components)
    if not result or not all(isinstance(item, Component) for item in result):
        raise InputError("A model needs at least one Component.")
    if len(result) > N_COMPONENTS_MAX:
        raise InputError(
            f"This release computes models of one component; got {len(result)}. "
            "Several components in one potential are not available yet."
        )
    return result


def compute_df(
    components: Component | Sequence[Component],
    settings: Settings | None = None,
) -> DistributionFunction:
    """Return the untruncated distribution function of each component.

    The DF is returned whatever its sign, with the information needed to
    truncate it: where it is negative, its minimum, and the lowest admissible
    truncation energy.

    Parameters
    ----------
    components : Component or sequence of Component
        The component of the model. This release takes one; a sequence of
        several, in one total potential, is refused.
    settings : Settings, optional
        Numerical settings. The default tier if omitted.

    Returns
    -------
    df : DistributionFunction

    Raises
    ------
    NotImplementedError
        If the component's anisotropy family is not served yet (served:
        isotropic, constant beta and Osipkov-Merritt Type I).
    """
    found = _as_components(components)
    from .invert import inversion_df

    settings = Settings() if settings is None else settings
    # the stages run under the timing record, so the DF carries their times
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
    """Return the untruncated distribution function a function of K gives.

    See `DistributionFunction.from_function`.
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
    # the identity takes the function by its values on the grid and below the
    # written floor: the table is stored and the function is not
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
    """The DF from its stages: the report announced, the result built."""
    from ..validation import announce

    grid, table, form, k_t_min, report, inversion, rows = stages
    if table.sign.size == 0:  # a signed DF's flagged report is not measured
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
    """Return the equilibrium model built from a distribution function.

    Parameters
    ----------
    df : DistributionFunction
        Result of `compute_df`.
    truncation : Truncation, optional
        The truncation. If omitted, the untruncated model is returned where
        it is a valid equilibrium.

    Returns
    -------
    model : Model

    Raises
    ------
    UnphysicalModelError
        If no valid equilibrium exists for the request. The exception carries
        `df` as its ``result``.
    InputError
        If the DF comes from a function (`DistributionFunction.from_function`):
        it carries no inversion rows, which truncation needs.
    """
    if df.source == "function":
        raise InputError(
            f"The distribution function of {df.components[0].name!r} comes from a "
            f"function ({df.function_name}): it carries no inversion rows, and "
            "truncation needs them, so it is not truncated."
        )
    if truncation is None:
        if isinstance(df.components[0].profile, TruncatedProfile):
            # a DF on a tabulated pair: the model at the edge of the profile
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
    """Say so where a model's DF diverges as the energy goes to zero."""
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
    """Return the equilibrium model of the given components.

    This runs `compute_df` and then `truncate`.

    Parameters
    ----------
    components : Component or sequence of Component
        The component of the model. This release takes one; a sequence of
        several, in one total potential, is refused.
    truncation : Truncation, optional
        The truncation. If omitted, the model is untruncated.
    settings : Settings, optional
        Numerical settings. The default tier if omitted.

    Returns
    -------
    model : Model
    """
    return truncate(compute_df(components, settings), truncation)
