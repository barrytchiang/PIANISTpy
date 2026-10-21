"""The results of a distribution-function computation, and their files.

`DistributionFunction` holds the untruncated distribution function (DF) of
the components of a model, and `Model` holds an equilibrium model. Each
result is a frozen dataclass that holds the stage results from which the
result was built. A result written to a file and read back holds the
identity string and the arrays of the result that was written.
"""

__all__ = ["DistributionFunction", "Model"]

import dataclasses
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..anisotropy import Anisotropy
from ..exceptions import InputError, ResultError
from ..io import StoredResult, load_result, register_reader, write_result, write_table
from ..numerics.pytrees import register_result
from ..profiles import DoublePowerLaw, TruncatedProfile
from ..settings import Settings
from ..typing import Array, ArrayLike
from ..units import Units
from ..validation import ValidationReport
from .inputs import Component, Truncation
from .stages import (
    AsymptoticForm,
    DfTable,
    EnergyGrid,
    InversionRecord,
    InversionRows,
)

DF_KIND = "distribution_function"
MODEL_KIND = "model"
# Some files key r_trust by "goal" and "ceiling". The reader maps those keys
# to the tier names "silent" and "recorded" that Model.r_trust uses.
_FORMER_TRUST_KEYS = {"goal": "silent", "ceiling": "recorded"}


@dataclasses.dataclass(frozen=True, eq=False)
class DistributionFunction:
    r"""Untruncated distribution functions of the components of a model.

    `compute_df` and `DistributionFunction.from_function` return a
    `DistributionFunction`. An untruncated distribution function (DF) may
    be negative or non-monotonic, and the `DistributionFunction` states
    where. A `DistributionFunction` is not sampled or projected: `truncate`
    builds the `Model` that is.

    Parameters
    ----------
    components : tuple of Component
        Components to which the DFs belong.
    settings : Settings
        Numerical settings with which the DFs were computed.
    identity : str
        Identity string of the result.
    grid : EnergyGrid
        Energies of the DF table and the radii at which the untruncated
        relative potential equals each energy.
    table : DfTable
        The DF of each component at the energies of `grid`.
    form : AsymptoticForm
        Analytic form of each DF below and above the energies of the DF
        table.
    k_t_min : float
        Lowest admissible truncation energy :math:`\mathcal{K}_{\rm t}`, in
        internal units. ``k_t_min`` is zero where every untruncated DF is a
        valid equilibrium as it stands (`is_physical`). Otherwise
        ``k_t_min`` is the largest of the highest energy at which a DF is
        negative, the energy of a DF's interior minimum and, for a Type II
        model, the relative potential :math:`P(R_{\rm a})` at the anisotropy
        radius :math:`R_{\rm a}`. Where :math:`P(R_{\rm a})` is the largest
        of these energies, ``k_t_min`` is the next double above
        :math:`P(R_{\rm a})`. `truncate` also checks
        that the edge of a truncated Type II model lies inside
        :math:`R_{\rm a}`.
    validation : ValidationReport
        Validation report of the DFs.
    timing : dict of str to float
        Elapsed time of each step of the computation, in seconds.
    inversion : InversionRecord or None
        Record of how the inversion made the DF table. None where the DF
        table has a closed form. Every `DistributionFunction` from
        `compute_df` holds an `InversionRecord`.
    units : Units, optional
        Unit system in which the result is presented (`to`). Default:
        ``Units()``, which means internal units.
    rows : InversionRows or None, optional
        Values of the inversion at every energy of the inversion's grid,
        which `truncate` reads. Default: ``None``, which means that the DF
        table was not made by an inversion.
    source : {"inversion", "function"}, optional
        Origin of the DF table's values: ``"inversion"`` (`compute_df`) or
        ``"function"`` (`DistributionFunction.from_function`). Default:
        ``"inversion"``.
    function_name : str or None, optional
        The ``__name__`` of the function from which the DF table was
        tabulated. Default: ``None``, which means that the DF table comes
        from an inversion.

    See Also
    --------
    compute_df : The untruncated DF of a component.
    truncate : The equilibrium model built from a `DistributionFunction`.
    Model : An equilibrium model.

    References
    ----------
    Chiang+(2026d), Eq. (8). Notes Section T:2.

    Examples
    --------
    >>> import pianistpy as pp
    >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
    >>> df.source, df.function_name, len(df.components)
    ('inversion', None, 1)
    """

    components: tuple[Component, ...]
    settings: Settings
    identity: str
    grid: EnergyGrid
    table: DfTable
    form: AsymptoticForm
    k_t_min: float
    validation: ValidationReport
    timing: dict[str, float]
    inversion: InversionRecord | None
    units: Units = Units()
    rows: InversionRows | None = None
    source: str = "inversion"
    function_name: str | None = None

    @classmethod
    def from_function(
        cls,
        component: Component,
        f: Callable[[Array, Array], ArrayLike],
        settings: Settings | None = None,
    ) -> "DistributionFunction":
        r"""Return the distribution function that a user-specified function gives.

        The user-specified function is a distribution function (DF) of the
        energy, such as a fitting function. The package tabulates the
        function on the energy grid that `compute_df` uses for the given
        component and settings, and validates the tabulated DF against the
        component's density. Like a `DistributionFunction` from
        `compute_df`, the result holds analytic forms below and above the
        DF table, a validation report and an identity string, and
        `DistributionFunction.write` writes the result to an HDF5 file. The
        result stores the DF table and does not store the function.

        Parameters
        ----------
        component : Component
            The component to which the DF belongs. The component's profile
            must be an analytic profile (a `DoublePowerLaw`).
        f : callable
            Function ``f(k, depth)`` of two arrays of equal shape: the
            energies :math:`\mathcal{K}` of the DF table in internal units,
            and the depth :math:`P_0 - \mathcal{K}` at each energy, with
            :math:`P_0 = 1` the central relative potential. ``f`` returns
            the DF :math:`F` (not its logarithm) at each energy, as an array
            of the shape of ``k``. The package passes ``depth`` as
            ``1 - k`` in double precision, which is exact for every energy
            :math:`\mathcal{K} \ge 1/2`. A function of :math:`\mathcal{K}`
            alone may ignore ``depth``.
        settings : Settings, optional
            Numerical settings. Default: ``None``, which means
            ``Settings()``, the default settings.

        Returns
        -------
        df : DistributionFunction
            The tabulated DF. The attribute `source` is ``"function"`` and
            the attribute `function_name` is the ``__name__`` of `f`. The DF
            table has no column of the derivative of the DF, and the check
            ``slope_column`` of the validation report states that the
            derivative is not judged. The result holds no inversion record
            and no inversion values, so `truncate` does not accept it.

        Raises
        ------
        InputError
            If `f` is not callable, returns an array whose shape differs
            from the shape of ``k``, or returns a value that is not finite
            at an energy of the grid. For a value that is not finite, the
            message states the first energy with such a value.
        InputError
            If the component is built on a `TruncatedProfile`.

        Warns
        -----
        AccuracyWarning
            If the DF is non-negative on its table and a check of the DF's
            validation report lies above :math:`10^{-4}`.

        See Also
        --------
        compute_df : The DF computed by inversion of the density.
        DistributionFunction.is_physical : Whether the DF is a valid
            equilibrium as it stands.

        References
        ----------
        Notes Section T:2.7.

        Examples
        --------
        The isotropic Plummer DF in internal units is
        :math:`F = A\,\mathcal{K}^{7/2}` with :math:`A \approx 1.965117`:

        >>> import pianistpy as pp
        >>> def plummer(k, depth):
        ...     return 1.965117148288 * k**3.5
        >>> component = pp.Component(pp.DoublePowerLaw.plummer())
        >>> df = pp.DistributionFunction.from_function(component, plummer)
        >>> df.source, df.function_name
        ('function', 'plummer')
        """
        from .core import df_from_function

        return df_from_function(component, f, settings)

    @property
    def is_physical(self) -> bool:
        """Whether the untruncated DFs are a valid equilibrium without a truncation.

        `is_physical` is true if `k_t_min` is zero.
        """
        return self.k_t_min == 0.0

    @property
    def diverges_at_zero(self) -> tuple[bool, ...]:
        r"""Whether the DF of each component diverges as the energy goes to zero.

        A DF that goes as :math:`\mathcal{K}^p` with :math:`-1 < p < 0` as
        the energy :math:`\mathcal{K}` goes to zero is integrable and gives a
        finite density, but such a DF is not monotonic.
        """
        return self.form.diverges_at_zero

    def to(
        self,
        system: str,
        concentration: float | None = None,
        mass_0: Any = None,
        r_0: Any = None,
        r_vir: Any = None,
    ) -> "DistributionFunction":
        r"""Return the distribution function presented in another unit system.

        The values of the result are not recomputed and stay in internal
        units. A quantity computed from the returned result is presented in
        the returned result's unit system.

        Parameters
        ----------
        system : {"internal", "model", "physical"}
            Target unit system: ``"internal"`` (:math:`G = \rho_0/C = r_0 =
            1`), ``"model"`` (:math:`G = M_0 = r_0 = 1`) or ``"physical"``
            (Astropy quantities).
        concentration : float, optional
            Concentration :math:`c`, positive. The mass unit :math:`M_0` is
            the mass of the untruncated profile inside :math:`c\,r_0`. The
            model and the physical systems need `concentration`. Default:
            ``None``, which means that no concentration is given.
        mass_0 : float or Quantity, optional
            Mass unit :math:`M_0`, positive, for the physical system. A float
            is in solar masses, and a Quantity of any mass unit is accepted.
            Default: ``None``, which means that no mass unit is given.
        r_0 : float or Quantity, optional
            Scale radius :math:`r_0`, positive, for the physical system. A
            float is in parsecs, and a Quantity of any length unit is
            accepted. Default: ``None``, which means that `r_vir` gives the
            length scale.
        r_vir : float or Quantity, optional
            Concentration radius :math:`c\,r_0`, positive, given in place of
            `r_0` for the physical system. A float is in parsecs, and a
            Quantity of any length unit is accepted. Default: ``None``,
            which means that `r_0` gives the length scale.

        Returns
        -------
        df : DistributionFunction
            The DF with unchanged values, with the attribute `units` set to
            the target unit system.

        Raises
        ------
        InputError
            If `system` is not ``"internal"``, ``"model"`` or
            ``"physical"``, if a scale that `system` needs is missing, if a
            scale is not positive, or if both `r_0` and `r_vir` are given.

        See Also
        --------
        Model.to : A model presented in another unit system.

        Examples
        --------
        >>> import pianistpy as pp
        >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
        >>> df.to("model", concentration=10.0).units.system
        'model'
        """
        units = Units.build(system, concentration, mass_0, r_0, r_vir)
        return dataclasses.replace(self, units=units)

    def write(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the distribution function to an HDF5 file.

        `pianistpy.read` reads the file back into a `DistributionFunction`
        with the same identity and the same arrays.

        Parameters
        ----------
        path : str or path-like
            Path of the HDF5 file to write.
        overwrite : bool, optional
            Whether to replace an existing file at `path`. Default:
            ``False``, which means that an existing file is kept and
            `InputError` is raised.

        Raises
        ------
        InputError
            If a file exists at `path` and `overwrite` is false.
        ResultError
            If the written file does not read back as written.

        See Also
        --------
        DistributionFunction.write_text : The DF table as a text file.

        Examples
        --------
        >>> import os, tempfile
        >>> import pianistpy as pp
        >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "hernquist_df.h5")
        ...     df.write(path)
        ...     same = pp.read(path).identity == df.identity
        >>> same
        True
        """
        _write_df(self, path, overwrite)

    def write_text(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        r"""Write the DF table to a fixed-width text file.

        The text file is an export for reading outside the package, and
        `pianistpy.read` does not read it. The file holds one line per
        energy of each component's support, with the columns ``k`` (the
        energy :math:`\mathcal{K}` in internal units, in which the central
        relative potential of the untruncated profile is 1), ``log10_df``
        (the base-10 logarithm of the DF) and ``dlog10_df_dlnk`` (the slope
        :math:`{\rm d}\log_{10} F / {\rm d}\ln \mathcal{K}`). A model of more
        than one component has a fourth column, ``component``, with the
        index of the component. The header states the result, the result's
        components and the units. The written file is read back and
        compared with the written values, to the digits written.

        Parameters
        ----------
        path : str or path-like
            Path of the text file to write.
        overwrite : bool, optional
            Whether to replace an existing file at `path`. Default:
            ``False``, which means that an existing file is kept and
            `InputError` is raised.

        Raises
        ------
        InputError
            If a file exists at `path` and `overwrite` is false.
        ResultError
            If the written file does not read back as written.

        See Also
        --------
        DistributionFunction.write : The full result as an HDF5 file.

        Examples
        --------
        >>> import os, tempfile
        >>> import pianistpy as pp
        >>> df = pp.compute_df(pp.Component(pp.DoublePowerLaw.hernquist()))
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "hernquist_df.txt")
        ...     df.write_text(path)
        ...     with open(path) as file:
        ...         first = file.readline().strip()
        >>> first
        '# PIANISTpy distribution function table'
        """
        _write_df_table_text(self, path, overwrite)


@dataclasses.dataclass(frozen=True, eq=False)
class Model:
    r"""An equilibrium model of one or more components, truncated or untruncated.

    `truncate` and `solve` return a `Model`. The distribution function (DF)
    of every component of a model is non-negative. A DF that diverges as
    the energy goes to zero is not monotonic (`diverges_at_zero`). Whether
    the model's density reaches zero at a finite radius is the property
    `spatially_finite`.

    Parameters
    ----------
    components : tuple of Component
        Components of the model.
    truncation : Truncation or None
        Truncation of the model. None for an untruncated model and for a
        model whose component is built on a `TruncatedProfile`.
    settings : Settings
        Numerical settings with which the model was computed.
    identity : str
        Identity string of the result.
    parent : str
        Identity string of the `DistributionFunction` from which the model
        was built.
    table : DfTable
        The DF of each component of the model.
    form : AsymptoticForm
        Analytic form of each DF below and above the energies of the DF
        table.
    profile : TruncatedProfile
        Self-consistent density, mass and relative potential of the model.
    validation : ValidationReport
        Validation report of the model.
    timing : dict of str to float
        Elapsed time of each step of the computation, in seconds.
    r_trust : mapping of str to float
        Radius, in internal units, inside which the model's values stay
        within the bound of a tier of the validation report, keyed by the
        tier's name. The key ``"silent"`` holds the radius for a relative
        error of at most :math:`10^{-6}`, and the key ``"recorded"`` the
        radius for a relative error of at most :math:`10^{-4}`
        (:doc:`/reference/accuracy`). Each radius states how far out the
        model meets the tier's bound, and is not itself an error of the
        model. On an untruncated model of an
        analytic profile, both radii are infinite. On every other model
        that `truncate` and `solve` return, the mapping is empty. A model
        read from a file holds the radii that the file records.
    inversion : InversionRecord or None
        Record of how the inversion made the model's DF table. None where
        the DF table has a closed form. Every `Model` from `solve` holds an
        `InversionRecord`.
    units : Units, optional
        Unit system in which the model is presented (`to`). Default:
        ``Units()``, which means internal units.

    See Also
    --------
    solve : The equilibrium model of a component.
    truncate : The equilibrium model built from a `DistributionFunction`.
    DistributionFunction : The untruncated DF of a component.

    References
    ----------
    Chiang+(2026d), Section 3.1. Notes Section T:3.

    Examples
    --------
    >>> import pianistpy as pp
    >>> model = pp.solve(
    ...     pp.Component(pp.DoublePowerLaw.hernquist()), pp.Truncation(k_t=0.05)
    ... )
    >>> model.spatially_finite, model.truncation.k_t
    (True, 0.05)
    """

    components: tuple[Component, ...]
    truncation: Truncation | None
    settings: Settings
    identity: str
    parent: str
    table: DfTable
    form: AsymptoticForm
    profile: TruncatedProfile
    validation: ValidationReport
    timing: dict[str, float]
    r_trust: Mapping[str, float]
    inversion: InversionRecord | None
    units: Units = Units()

    @property
    def spatially_finite(self) -> bool:
        """Whether the model's density reaches zero at a finite radius."""
        return self.profile.spatially_finite

    @property
    def diverges_at_zero(self) -> tuple[bool, ...]:
        r"""Whether the DF of each component diverges as the energy goes to zero.

        A DF that goes as :math:`\mathcal{K}^p` with :math:`-1 < p < 0` as
        the energy :math:`\mathcal{K}` goes to zero is integrable. A model
        with such a DF is a valid equilibrium with a finite density, but the
        model's DF is not monotonic.
        """
        return self.form.diverges_at_zero

    def to(
        self,
        system: str,
        concentration: float | None = None,
        mass_0: Any = None,
        r_0: Any = None,
        r_vir: Any = None,
    ) -> "Model":
        r"""Return the model presented in another unit system.

        The values of the model are not recomputed and stay in internal
        units. A quantity computed from the returned model is presented in
        the returned model's unit system.

        Parameters
        ----------
        system : {"internal", "model", "physical"}
            Target unit system: ``"internal"`` (:math:`G = \rho_0/C = r_0 =
            1`), ``"model"`` (:math:`G = M_0 = r_0 = 1`) or ``"physical"``
            (Astropy quantities).
        concentration : float, optional
            Concentration :math:`c`, positive. The mass unit :math:`M_0` is
            the mass of the untruncated profile inside :math:`c\,r_0`. The
            model and the physical systems need a concentration. Default:
            ``None``, which means the concentration of the model's
            `Truncation` where the `Truncation` gives one, and no
            concentration otherwise.
        mass_0 : float or Quantity, optional
            Mass unit :math:`M_0`, positive, for the physical system. A float
            is in solar masses, and a Quantity of any mass unit is accepted.
            Default: ``None``, which means that no mass unit is given.
        r_0 : float or Quantity, optional
            Scale radius :math:`r_0`, positive, for the physical system. A
            float is in parsecs, and a Quantity of any length unit is
            accepted. Default: ``None``, which means that `r_vir` gives the
            length scale.
        r_vir : float or Quantity, optional
            Concentration radius :math:`c\,r_0`, positive, given in place of
            `r_0` for the physical system. A float is in parsecs, and a
            Quantity of any length unit is accepted. Default: ``None``,
            which means that `r_0` gives the length scale.

        Returns
        -------
        model : Model
            The model with unchanged values, with the attribute `units` set
            to the target unit system.

        Raises
        ------
        InputError
            If `concentration` differs from the concentration of the
            model's `Truncation`.
        InputError
            If `system` is not ``"internal"``, ``"model"`` or
            ``"physical"``, if a scale that `system` needs is missing, if a
            scale is not positive, or if both `r_0` and `r_vir` are given.

        See Also
        --------
        Model.radial_profiles : The radial profiles in the model's unit
            system.
        DistributionFunction.to : A DF presented in another unit system.

        Examples
        --------
        >>> import pianistpy as pp
        >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
        >>> model.to("model", concentration=10.0).units.system
        'model'
        """
        truncation = self.truncation
        given = None if truncation is None else truncation.concentration
        if concentration is None:
            concentration = given
        elif given is not None and concentration != given:
            raise InputError(
                f"The model was truncated with concentration {given}; its units "
                f"cannot use {concentration}."
            )
        units = Units.build(system, concentration, mass_0, r_0, r_vir)
        return dataclasses.replace(self, units=units)

    def radial_profiles(self) -> dict[str, Any]:
        """Return the radial profiles of the model in the model's unit system.

        Returns
        -------
        profiles : dict
            Dictionary of four arrays tabulated at the radii of ``"r"``:
            ``"r"`` (the radii, shape ``(n,)``), ``"potential"`` (the
            relative potential, shape ``(n,)``), ``"density"`` (the density
            of each component, shape ``(n_components, n)``) and ``"mass"``
            (the total enclosed mass, shape ``(n,)``). The arrays are plain
            arrays in the internal and the model systems, and Astropy
            quantities in the physical system (`Model.to`).

        See Also
        --------
        Model.to : The model presented in another unit system.
        Model.write_text : The radial profiles as a text file.

        Examples
        --------
        >>> import pianistpy as pp
        >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
        >>> profiles = model.radial_profiles()
        >>> sorted(profiles), profiles["density"].shape[0]
        (['density', 'mass', 'potential', 'r'], 1)
        """
        profile = self.components[0].unit_profile
        units = self.units
        return {
            "r": units.convert("length", self.profile.r, profile),
            "potential": units.convert("potential", self.profile.p_trunc, profile),
            "density": units.convert("density", self.profile.rho_trunc, profile),
            "mass": units.convert("mass", self.profile.mass_trunc, profile),
        }

    @property
    def mass_t(self) -> Array:
        """Total mass of the model, in the model's unit system."""
        profile = self.components[0].unit_profile
        return self.units.convert("mass", self.profile.mass_t, profile)

    @property
    def r_t(self) -> Array:
        """Outer radius of the model, in the model's unit system.

        Infinite for a model whose density does not reach zero at a finite
        radius.
        """
        profile = self.components[0].unit_profile
        return self.units.convert("length", self.profile.r_t, profile)

    def write(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        """Write the model to an HDF5 file.

        `pianistpy.read` reads the file back into a `Model` with the same
        identity and the same arrays.

        Parameters
        ----------
        path : str or path-like
            Path of the HDF5 file to write.
        overwrite : bool, optional
            Whether to replace an existing file at `path`. Default:
            ``False``, which means that an existing file is kept and
            `InputError` is raised.

        Raises
        ------
        InputError
            If a file exists at `path` and `overwrite` is false.
        ResultError
            If the written file does not read back as written.

        See Also
        --------
        Model.write_text : A table of the model as a text file.

        Examples
        --------
        >>> import os, tempfile
        >>> import pianistpy as pp
        >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "hernquist.h5")
        ...     model.write(path)
        ...     same = pp.read(path).identity == model.identity
        >>> same
        True
        """
        _write_model(self, path, overwrite)

    def write_text(
        self, path: str | os.PathLike, table: str = "profile", overwrite: bool = False
    ) -> None:
        """Write a table of the model to a fixed-width text file.

        The text file is an export for reading outside the package, and
        `pianistpy.read` does not read it. The written file is read back and
        compared with the written values, to the digits written.

        Parameters
        ----------
        path : str or path-like
            Path of the text file to write.
        table : {"profile", "df"}, optional
            Table to write. ``"profile"`` writes the radial profiles in the
            model's unit system (`Model.radial_profiles`), with the columns
            ``r``, ``potential``, ``density`` and ``mass`` (the total
            enclosed mass). A model of more than one component has one
            density column per component, called ``density_<name>`` after
            the component's name. The header states the unit of each
            column. ``"df"`` writes the DF table in internal units, with the
            columns that `DistributionFunction.write_text` writes. Default:
            ``"profile"``, which means the radial profiles.
        overwrite : bool, optional
            Whether to replace an existing file at `path`. Default:
            ``False``, which means that an existing file is kept and
            `InputError` is raised.

        Raises
        ------
        InputError
            If `table` is neither ``"profile"`` nor ``"df"``.
        InputError
            If a file exists at `path` and `overwrite` is false.
        ResultError
            If the written file does not read back as written.

        See Also
        --------
        Model.write : The full model as an HDF5 file.
        Model.radial_profiles : The radial profiles as arrays.

        Examples
        --------
        >>> import os, tempfile
        >>> import pianistpy as pp
        >>> model = pp.solve(pp.Component(pp.DoublePowerLaw.hernquist()))
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = os.path.join(folder, "hernquist_profile.txt")
        ...     model.write_text(path)
        ...     with open(path) as file:
        ...         first = file.readline().strip()
        >>> first
        '# PIANISTpy radial profile'
        """
        if table == "profile":
            _write_profile_text(self, path, overwrite)
        elif table == "df":
            _write_df_table_text(self, path, overwrite)
        else:
            raise InputError(f"table must be 'profile' or 'df'; got {table!r}.")


def _pack(stage: Any, prefix: str, arrays: dict, record: dict) -> None:
    """Add a stage result's fields to the content of a file.

    Each NumPy array field goes into `arrays` under the key
    ``"<prefix>/<field>"``. A `DoublePowerLaw` or `Anisotropy` field goes
    into ``record[prefix]`` as its dictionary form, and every other field
    goes into ``record[prefix]`` as it is.
    """
    fields: dict[str, Any] = {}
    for item in dataclasses.fields(stage):
        value = getattr(stage, item.name)
        if isinstance(value, np.ndarray):
            arrays[f"{prefix}/{item.name}"] = value
        elif isinstance(value, (DoublePowerLaw, Anisotropy)):
            fields[item.name] = value.to_dict()
        else:
            fields[item.name] = value
    record[prefix] = fields


def _tuples(value: Any) -> Any:
    """Return a value read from a file with every nested list turned into a tuple.

    The file's record stores tuples as lists, and the stage results hold
    tuples.
    """
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def _unpack(cls: type, prefix: str, stored: StoredResult) -> Any:
    """Return the stage result of class `cls` rebuilt from a stored file.

    Raises
    ------
    ResultError
        If `cls` is `DfTable` or `EnergyGrid` and the file holds the
        energies ``k`` as a one-dimensional array, without the component
        axis that the two classes need.
    """
    values = {key: _tuples(value) for key, value in stored.record[prefix].items()}
    for item in dataclasses.fields(cls):
        key = f"{prefix}/{item.name}"
        # A field that the file holds neither in the record nor among the
        # arrays is left out of `values`, so the field keeps the default of
        # its dataclass.
        if item.name not in values and key in stored.arrays:
            values[item.name] = stored.arrays[key]
    if cls in (DfTable, EnergyGrid) and np.ndim(values.get("k", 0.0)) == 1:
        # DfTable and EnergyGrid hold the energies with a component axis,
        # shape (n_components, n); a file with one-dimensional energies
        # cannot be read into them.
        raise ResultError(
            f"The file's {prefix} holds its energies without the component axis: "
            "a file of the previous layout; compute the result again."
        )
    if cls is TruncatedProfile:
        values["source"] = DoublePowerLaw.from_dict(values["source"])
        values["source_anisotropy"] = Anisotropy.from_dict(values["source_anisotropy"])
    return cls(**values)


def _component_prefix(index: int) -> str:
    """Return the prefix under which a file holds a component's TruncatedProfile."""
    return f"component_profile_{index}"


def _pack_components(
    components: tuple[Component, ...], arrays: dict, record: dict
) -> None:
    """Add the TruncatedProfile of each Method II component to a file's content."""
    for index, component in enumerate(components):
        if isinstance(component.profile, TruncatedProfile):
            _pack(component.profile, _component_prefix(index), arrays, record)


def _unpack_components(stored: StoredResult) -> tuple[Component, ...]:
    """Return the components of a stored result, each with its TruncatedProfile.

    A component whose prefix the file holds is rebuilt on the stored
    TruncatedProfile; every other component is rebuilt on its analytic
    profile.
    """
    components = []
    for index, item in enumerate(stored.record["components"]):
        profile = None
        if _component_prefix(index) in stored.record:
            profile = _unpack(TruncatedProfile, _component_prefix(index), stored)
        components.append(Component.from_dict(item, profile=profile))
    return tuple(components)


def _unpack_inversion(stored: StoredResult) -> InversionRecord | None:
    """Return the inversion record of a stored result, or None for a closed form."""
    if stored.record["inversion"] is None:
        return None
    return _unpack(InversionRecord, "inversion", stored)


def _pack_inversion(
    result: DistributionFunction | Model, arrays: dict, record: dict
) -> None:
    """Add a result's inversion record to a file's content (None: closed form)."""
    if result.inversion is None:
        record["inversion"] = None
    else:
        _pack(result.inversion, "inversion", arrays, record)


def _common_record(result: DistributionFunction | Model) -> dict[str, Any]:
    """Return the record fields that a DF file and a model file both hold."""
    return {
        "components": [component.to_dict() for component in result.components],
        "settings": result.settings.to_dict(),
        "validation": result.validation.to_dict(),
        "timing": dict(result.timing),
        "units": result.units.to_dict(),
    }


def _write_df(
    df: DistributionFunction, path: str | os.PathLike, overwrite: bool = False
) -> None:
    """Write a distribution function to an HDF5 file.

    Parameters
    ----------
    df : DistributionFunction
        The result to write.
    path : str or path-like
        Path of the HDF5 file to write.
    overwrite : bool, optional
        Whether to replace an existing file at `path`. Default: ``False``,
        which means that an existing file is kept and `InputError` is
        raised.
    """
    arrays: dict[str, np.ndarray] = {}
    record = _common_record(df)
    record["k_t_min"] = df.k_t_min
    record["source"] = df.source
    record["function_name"] = df.function_name
    _pack_components(df.components, arrays, record)
    _pack(df.grid, "grid", arrays, record)
    _pack(df.table, "table", arrays, record)
    _pack(df.form, "form", arrays, record)
    _pack_inversion(df, arrays, record)
    if df.rows is None:
        record["rows"] = None
    else:
        _pack(df.rows, "rows", arrays, record)
    write_result(path, DF_KIND, df.identity, record, arrays, overwrite=overwrite)


def _write_model(
    model: Model, path: str | os.PathLike, overwrite: bool = False
) -> None:
    """Write a model to an HDF5 file.

    The file records the identity of the model's `DistributionFunction` as
    the model's parent.

    Parameters
    ----------
    model : Model
        The result to write.
    path : str or path-like
        Path of the HDF5 file to write.
    overwrite : bool, optional
        Whether to replace an existing file at `path`. Default: ``False``,
        which means that an existing file is kept and `InputError` is
        raised.
    """
    arrays: dict[str, np.ndarray] = {}
    record = _common_record(model)
    truncation = model.truncation
    record["truncation"] = None if truncation is None else truncation.to_dict()
    record["r_trust"] = {band: float(r) for band, r in model.r_trust.items()}
    _pack_components(model.components, arrays, record)
    _pack(model.table, "table", arrays, record)
    _pack(model.form, "form", arrays, record)
    _pack(model.profile, "profile", arrays, record)
    _pack_inversion(model, arrays, record)
    write_result(
        path,
        MODEL_KIND,
        model.identity,
        record,
        arrays,
        parents=(model.parent,),
        overwrite=overwrite,
    )


def _plain(value: Any) -> np.ndarray:
    """Return the values of a converted quantity as a float64 array.

    An Astropy quantity gives its ``value``; a plain array is returned as
    float64.
    """
    return np.asarray(getattr(value, "value", value), dtype=np.float64)


def _unit_label(value: Any, units: Units) -> str:
    """Return the unit label of a converted quantity.

    An Astropy quantity gives its unit; a plain array gives the name of the
    unit system, as the particle sampler labels its columns.
    """
    return str(getattr(value, "unit", units.system))


def _text_head(result: DistributionFunction | Model, what: str) -> list[str]:
    """Return the header lines of a text export: title, identities, components."""
    head = [f"PIANISTpy {what}", f"identity: {result.identity}"]
    parent = getattr(result, "parent", None)
    if parent is not None:
        head.append(f"parent: {parent}")
    head.append("components: " + ", ".join(c.name for c in result.components))
    return head


def _write_df_table_text(
    result: DistributionFunction | Model, path: str | os.PathLike, overwrite: bool
) -> None:
    """Write the DF table of a result as text, on each component's support."""
    table = result.table
    multi = len(result.components) > 1
    parts: dict[str, list[np.ndarray]] = {
        "k": [],
        "log10_df": [],
        "dlog10_df_dlnk": [],
        "component": [],
    }
    for i in range(len(result.components)):
        support = slice(int(table.i_first[i]), int(table.i_last[i]) + 1)
        parts["k"].append(np.asarray(table.k[i, support], dtype=np.float64))
        parts["log10_df"].append(np.asarray(table.log_df[i, support], dtype=np.float64))
        parts["dlog10_df_dlnk"].append(
            np.asarray(table.log_df_slope[i, support], dtype=np.float64)
        )
        parts["component"].append(np.full(support.stop - support.start, i))
    columns = {name: np.concatenate(parts[name]) for name in parts}
    if not multi:
        del columns["component"]
    head = _text_head(result, "distribution function table") + [
        "units: internal; k is the energy-like variable in units of the central "
        "potential, log10_df the base-10 logarithm of the DF, dlog10_df_dlnk its "
        "slope d log10 F / d ln k; minus infinity where the DF is zero"
    ]
    if multi:
        head.append("component: index into the components above")
    write_table(path, columns, head, overwrite=overwrite)


def _write_profile_text(model: Model, path: str | os.PathLike, overwrite: bool) -> None:
    """Write the radial profiles of a model as text, in the model's unit system."""
    profiles = model.radial_profiles()
    units = model.units
    density = profiles["density"]
    columns: dict[str, np.ndarray] = {
        "r": _plain(profiles["r"]),
        "potential": _plain(profiles["potential"]),
    }
    labels = {
        "r": _unit_label(profiles["r"], units),
        "potential": _unit_label(profiles["potential"], units),
    }
    rows = _plain(density)
    rows = rows.reshape(1, -1) if rows.ndim == 1 else rows
    if len(model.components) == 1:
        columns["density"] = rows[0]
        labels["density"] = _unit_label(density, units)
    else:
        for component, row in zip(model.components, rows, strict=True):
            columns[f"density_{component.name}"] = row
            labels[f"density_{component.name}"] = _unit_label(density, units)
    columns["mass"] = _plain(profiles["mass"])
    labels["mass"] = _unit_label(profiles["mass"], units)
    head = _text_head(model, "radial profile") + [
        f"unit_system: {units.system}",
        "units: " + " | ".join(f"{name}: {labels[name]}" for name in columns),
    ]
    write_table(path, columns, head, overwrite=overwrite)


def _check_kind(stored: StoredResult, kind: str) -> None:
    """Raise ResultError if a stored result is not of the given kind."""
    if stored.kind != kind:
        raise ResultError(f"The result is of kind {stored.kind!r}, not {kind!r}.")


# Key sets of validation reports made outside the package's stages. The code
# that assembles such results registers each key set here, so that a file
# holding such a report can be read back.
_REGISTERED_KEY_SETS: dict[str, list[tuple[str, ...]]] = {"df": [], "model": []}


def register_report_keys(kind: str, keys: Sequence[str]) -> None:
    """Admit one more key set for the validation reports of files read back.

    Parameters
    ----------
    kind : {"df", "model"}
        Kind of result whose reports may hold the key set.
    keys : sequence of str
        Names of the report's checks, in order.

    Raises
    ------
    InputError
        If `kind` is neither ``"df"`` nor ``"model"``.
    """
    if kind not in _REGISTERED_KEY_SETS:
        raise InputError(f"kind is 'df' or 'model'; got {kind!r}.")
    if tuple(keys) not in _REGISTERED_KEY_SETS[kind]:
        _REGISTERED_KEY_SETS[kind].append(tuple(keys))


def _df_key_sets() -> tuple[tuple[str, ...], ...]:
    """Return the admissible key sets of a DF's validation report.

    The import of the module `report` is deferred to the call, so that
    importing the module `results` does not import the module `report`.
    """
    from .report import DF_REPORT_KEYS

    return (*DF_REPORT_KEYS.values(), *_REGISTERED_KEY_SETS["df"])


def _model_key_sets() -> tuple[tuple[str, ...], ...]:
    """Return the admissible key sets of a model's validation report.

    The key sets are those of a Method I model, of an untruncated model, of
    a Method II model, the empty key set of a closed-form model, and the
    registered key sets. The imports are inside the function because the
    module `truncated` imports the module `results` at import time.
    """
    from .tabulated import _MODEL_KEYS as method_ii
    from .truncated import _MODEL_KEYS as method_i
    from .truncated import _UNTRUNCATED_KEYS as untruncated

    return (method_i, untruncated, method_ii, (), *_REGISTERED_KEY_SETS["model"])


def _read_df(source: StoredResult | str | os.PathLike) -> DistributionFunction:
    """Return a distribution function from a file or a stored result.

    Parameters
    ----------
    source : StoredResult, str or path-like
        A stored result, or the path of a file written by `_write_df`.

    Returns
    -------
    df : DistributionFunction
        The distribution function that the file holds.

    Raises
    ------
    ResultError
        If the file holds a result of a kind other than a distribution
        function, or holds energies without the component axis.
    """
    stored = load_result(source)
    _check_kind(stored, DF_KIND)
    record = stored.record
    return DistributionFunction(
        components=_unpack_components(stored),
        settings=Settings.from_dict(record["settings"]),
        identity=stored.identity,
        grid=_unpack(EnergyGrid, "grid", stored),
        table=_unpack(DfTable, "table", stored),
        form=_unpack(AsymptoticForm, "form", stored),
        k_t_min=float(record["k_t_min"]),
        validation=ValidationReport.from_dict(record["validation"], _df_key_sets()),
        timing=dict(record["timing"]),
        units=Units.from_dict(record.get("units")),
        inversion=_unpack_inversion(stored),
        rows=(
            None
            if record.get("rows") is None
            else _unpack(InversionRows, "rows", stored)
        ),
        # A file that records no source holds a DF made by inversion.
        source=str(record.get("source", "inversion")),
        function_name=record.get("function_name"),
    )


def _read_model(source: StoredResult | str | os.PathLike) -> Model:
    """Return a model from a file or a stored result.

    Parameters
    ----------
    source : StoredResult, str or path-like
        A stored result, or the path of a file written by `_write_model`.

    Returns
    -------
    model : Model
        The model that the file holds. The keys ``"goal"`` and
        ``"ceiling"`` of a stored `Model.r_trust` are read as ``"silent"``
        and ``"recorded"``.

    Raises
    ------
    ResultError
        If the file holds a result of a kind other than a model, or holds
        energies without the component axis.
    """
    stored = load_result(source)
    _check_kind(stored, MODEL_KIND)
    record = stored.record
    truncation = record["truncation"]
    return Model(
        components=_unpack_components(stored),
        truncation=None if truncation is None else Truncation.from_dict(truncation),
        settings=Settings.from_dict(record["settings"]),
        identity=stored.identity,
        parent=stored.parents[0],
        table=_unpack(DfTable, "table", stored),
        form=_unpack(AsymptoticForm, "form", stored),
        profile=_unpack(TruncatedProfile, "profile", stored),
        validation=ValidationReport.from_dict(record["validation"], _model_key_sets()),
        timing=dict(record["timing"]),
        units=Units.from_dict(record.get("units")),
        r_trust={
            _FORMER_TRUST_KEYS.get(band, band): r
            for band, r in record["r_trust"].items()
        },
        inversion=_unpack_inversion(stored),
    )


register_reader(DF_KIND, _read_df)
register_reader(MODEL_KIND, _read_model)


# Both result classes are registered as JAX pytrees. The listed fields (the
# inputs and the arrays) are the pytree's children, so that JAX can trace,
# map and differentiate a function of a result in them. Every other field
# (the settings, the identities, the reports and the records) is static
# auxiliary data.
register_result(
    DistributionFunction, ("components", "grid", "table", "form", "k_t_min", "rows")
)
register_result(Model, ("components", "truncation", "table", "form", "profile"))
