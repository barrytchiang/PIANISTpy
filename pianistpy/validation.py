r"""The accuracy bands of a result and the validation report that states them.

Every result of the package (a model, a distribution function, a sampler, a
projected table) carries a `ValidationReport` as its ``validation``
attribute. A user reads the band of a result from ``result.validation.tier``
and the measured relative errors from ``result.validation.checks``.
`tier_level` maps one relative error :math:`\epsilon` to its band:

.. math::

    \text{band}(\epsilon) =
    \begin{cases}
    \text{silent}, & \epsilon \le 10^{-6},\\
    \text{recorded}, & 10^{-6} < \epsilon \le 10^{-4},\\
    \text{warned}, & 10^{-4} < \epsilon \le 10^{-2},\\
    \text{refused}, & \epsilon > 10^{-2}.
    \end{cases}

The technical notes, Section T:0.5, state how each band is reported.
"""

__all__ = [
    "TIER_BOUNDS",
    "TIER_NAMES",
    "Check",
    "ValidationReport",
    "announce",
    "refuse_unless_asked",
    "tier_level",
]

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .exceptions import AccuracyError, AccuracyWarning, InputError, ResultError

_log = logging.getLogger("pianistpy")

# The three relative errors that separate the four bands of a report's tier:
# a result with relative error at most 1e-6 is silent, a result above 1e-6 and
# at most 1e-4 is recorded in the log, a result above 1e-4 and at most 1e-2 is
# returned with a warning, and a result above 1e-2 is refused.
TIER_BOUNDS = (1e-6, 1e-4, 1e-2)

TIER_NAMES = ("silent", "recorded", "warned", "refused")
# Each bound of TIER_BOUNDS written as a decimal for messages: entry k is the
# lower bound of the band TIER_NAMES[k + 1].
_BOUND_TEXT = ("0.000001", "0.0001", "0.01")
# The former names of the three lower bands, which result files written with
# those names carry in their "parents_tier" entry.
_FORMER_NAMES = {"goal": "silent", "ceiling": "recorded", "warning": "warned"}
_NAN = float("nan")
# The reason given to a check that a report's key set requires and that the
# file being read does not hold.
NOT_IN_FILE = (
    "not in the file: the package version that wrote the file did not make the check"
)


def tier_level(error: float) -> int:
    r"""Return the band of a relative error as a level from 0 to 3.

    The level indexes `TIER_NAMES`: level 0 is ``"silent"``, level 1 is
    ``"recorded"``, level 2 is ``"warned"`` and level 3 is ``"refused"``.

    Parameters
    ----------
    error : float
        Relative error :math:`\epsilon`, a non-negative number. A NaN error
        has level 3, the level of an error above :math:`10^{-2}`.

    Returns
    -------
    level : int
        0 if `error` is at most :math:`10^{-6}`; 1 if `error` is above
        :math:`10^{-6}` and at most :math:`10^{-4}`; 2 if `error` is above
        :math:`10^{-4}` and at most :math:`10^{-2}`; 3 if `error` is above
        :math:`10^{-2}` or is NaN.

    See Also
    --------
    TIER_NAMES : The name of each level.
    TIER_BOUNDS : The three relative errors that separate the levels.
    Check : One measured relative error and its level.

    Examples
    --------
    >>> [tier_level(e) for e in (1e-7, 1e-5, 1e-3, 1e-1, float("nan"))]
    [0, 1, 2, 3, 3]
    >>> TIER_NAMES[tier_level(3e-5)]
    'recorded'
    """
    if not error <= TIER_BOUNDS[2]:
        return 3
    if error <= TIER_BOUNDS[0]:
        return 0
    return 1 if error <= TIER_BOUNDS[1] else 2


@dataclass(frozen=True)
class Check:
    r"""One check of a result: the relative error measured and where.

    A check is either judged or not judged. A judged check has a measured
    relative error and the band of that error. A check that is not judged
    has the error NaN, a stated reason, and the name ``"not judged"`` in
    place of a band; a check that is not judged does not change the band of
    the result.

    Parameters
    ----------
    error : float
        The relative error the check measured. NaN for a check that is not
        judged.
    where : float, optional
        The position of the largest relative error, in the variable the
        check runs over (a radius or an energy, in internal units).
        Default: ``None``, which means the check has no single position of
        its largest error.
    reason : str, optional
        The reason the check is not judged. Default: ``None``, which means
        the check is judged.
    source : str, optional
        For a judged check that read its input from a source other than the
        result's own data: the source read, and the reason. The error and
        the band are those measured on that source. Default: ``None``,
        which means the check read the result's own data.
    excluded : str, optional
        For a judged check that leaves part of its input unjudged: the part
        left out, and the reason. The error and the band are those of the
        remaining input. Default: ``None``, which means the check judged
        its whole input.

    Attributes
    ----------
    level : int
        The band of the check's error as returned by `tier_level`; 0 for a
        check that is not judged.
    tier : str
        The name of the check's band, one of `TIER_NAMES`, or
        ``"not judged"``.

    See Also
    --------
    ValidationReport : The checks of one result together.
    tier_level : The band of a relative error.

    Examples
    --------
    >>> check = Check(error=3e-5, where=12.5)
    >>> check.tier
    'recorded'
    >>> Check(error=float("nan"), reason="the table has no outer rows").tier
    'not judged'
    """

    error: float
    where: float | None = None
    reason: str | None = None
    source: str | None = None
    excluded: str | None = None

    @property
    def level(self) -> int:
        """The band of the check's error as a level from 0 to 3; 0 for a check
        that is not judged."""
        return 0 if self.reason is not None else tier_level(self.error)

    @property
    def tier(self) -> str:
        """The name of the check's band, or ``"not judged"``."""
        return "not judged" if self.reason is not None else TIER_NAMES[self.level]


@dataclass(frozen=True)
class ValidationReport:
    r"""The validation report of one result: its checks and its band.

    Each kind of result reports a fixed set of keys, one `Check` per key.
    The band of the result is the worst band among the result's own checks
    and the worst band among the results the result was built from (its
    parents). A report holds only the result's own checks: a derived result
    carries the worst level of its parents as one number, not the parents'
    reports.

    Parameters
    ----------
    keys : tuple of str
        The keys the kind of result reports, in the order the report lists
        them.
    checks : mapping of str to Check
        One `Check` for each entry of `keys`.
    parents_level : int, optional
        The worst level, from 0 to 3, among the parents of the result.
        Default: ``0``, which means the result has no parent above
        :math:`10^{-6}`.

    Attributes
    ----------
    level : int
        The worst level among the result's own checks and `parents_level`.
    tier : str
        The name of the worst band, ``TIER_NAMES[level]``.

    Raises
    ------
    InputError
        If the keys of `checks` are not the entries of `keys`.

    See Also
    --------
    Check : One check of the report.
    tier_level : The band of a relative error.

    Examples
    --------
    >>> report = ValidationReport(
    ...     keys=("round_trip", "top_form"),
    ...     checks={"round_trip": Check(2e-7), "top_form": Check(3e-5)},
    ... )
    >>> report.tier
    'recorded'
    >>> ValidationReport(report.keys, report.checks, parents_level=2).tier
    'warned'
    """

    keys: tuple[str, ...]
    checks: Mapping[str, Check]
    parents_level: int = 0

    def __post_init__(self) -> None:
        if set(self.checks) != set(self.keys):
            raise InputError(
                "The checks of a validation report must have the report's keys "
                f"{list(self.keys)}; the checks given have the keys "
                f"{sorted(self.checks)}."
            )

    @property
    def level(self) -> int:
        """The worst level among the result's own checks and its parents."""
        own = max((check.level for check in self.checks.values()), default=0)
        return max(own, self.parents_level)

    @property
    def tier(self) -> str:
        """The name of the worst band of the result."""
        return TIER_NAMES[self.level]

    @classmethod
    def from_dict(
        cls,
        record: Mapping[str, Any],
        key_sets: Sequence[tuple[str, ...]] | None = None,
    ) -> "ValidationReport":
        """Build a report from the dictionary that `to_dict` writes.

        Parameters
        ----------
        record : mapping
            The report in the form `to_dict` returns: the entries
            ``"parents_tier"`` and ``"checks"``.
        key_sets : sequence of tuple of str, optional
            Every key set that a report of the record's kind of result can
            have. With `key_sets` given, the report takes the smallest key
            set that holds every check of the record, and a key of that set
            that the record lacks becomes a check that is not judged, with
            the reason that the file does not hold the check. Default:
            ``None``, which means the report takes the record's own checks
            as its keys.

        Returns
        -------
        report : ValidationReport
            The report the record describes.

        Raises
        ------
        ResultError
            If `key_sets` is given and a check of the record belongs to no
            key set of `key_sets`.
        ResultError
            If `key_sets` is given and no single key set of `key_sets` holds
            every check of the record.

        See Also
        --------
        to_dict : The dictionary form of a report.

        Examples
        --------
        >>> report = ValidationReport(("a",), {"a": Check(3e-5)})
        >>> record = report.to_dict()
        >>> ValidationReport.from_dict(record) == report
        True
        >>> wider = ValidationReport.from_dict(record, [("a", "b")])
        >>> wider.checks["b"].tier
        'not judged'
        """
        found = tuple(record["checks"])
        keys = found
        if key_sets is not None:
            known = {key for key_set in key_sets for key in key_set}
            unknown = [key for key in found if key not in known]
            if unknown:
                raise ResultError(
                    f"The validation report read holds the check(s) {unknown}, "
                    "which no validation report of the result's kind holds."
                )
            holding = [s for s in key_sets if set(found) <= set(s)]
            if not holding:
                raise ResultError(
                    "No validation report of the result's kind holds all the checks "
                    f"{list(found)} of the validation report read."
                )
            keys = min(holding, key=len)
        checks = {
            key: Check(
                error=float(item["error"]),
                where=item["where"],
                reason=item.get("reason"),
                source=item.get("source"),
                excluded=item.get("excluded"),
            )
            for key, item in record["checks"].items()
        }
        for key in keys:
            if key not in checks:
                checks[key] = Check(_NAN, reason=NOT_IN_FILE)
        return cls(
            keys=keys,
            checks=checks,
            parents_level=TIER_NAMES.index(
                _FORMER_NAMES.get(record["parents_tier"], record["parents_tier"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a plain dictionary.

        Returns
        -------
        record : dict
            The entries ``"tier"`` (the report's band), ``"parents_tier"``
            (the name of the parents' worst band) and ``"checks"`` (one
            dictionary for each key, in the order of `keys`, with the
            entries ``"error"``, ``"where"`` and ``"tier"``, and
            ``"reason"``, ``"source"`` or ``"excluded"`` where the check
            states one).

        See Also
        --------
        from_dict : The report built from the dictionary.

        Examples
        --------
        >>> report = ValidationReport(("a",), {"a": Check(3e-5, where=1.0)})
        >>> record = report.to_dict()
        >>> record["tier"], record["parents_tier"]
        ('recorded', 'silent')
        >>> record["checks"]["a"]
        {'error': 3e-05, 'where': 1.0, 'tier': 'recorded'}
        """
        return {
            "tier": self.tier,
            "parents_tier": TIER_NAMES[self.parents_level],
            "checks": {key: _check_record(self.checks[key]) for key in self.keys},
        }


def _check_record(check: Check) -> dict[str, Any]:
    """Return a check's dictionary form: the error, the position and the band,
    and the reason, the source and the excluded part only where the check
    states one, so that the dictionary of a check stating none of the three
    has three entries."""
    record: dict[str, Any] = {
        "error": check.error,
        "where": check.where,
        "tier": check.tier,
    }
    if check.reason is not None:
        record["reason"] = check.reason
    if check.source is not None:
        record["source"] = check.source
    if check.excluded is not None:
        record["excluded"] = check.excluded
    return record


def refuse_unless_asked(what: str, report: ValidationReport, settings: Any) -> None:
    r"""Raise if an input result is in the band ``"refused"`` and the settings
    do not ask for the result anyway.

    A result in the band ``"refused"`` (relative error above
    :math:`10^{-2}`) exists only if the settings that built the result set
    ``write_anyway=True``, or if the result was read from a file that holds
    such a result. A computation that takes such a result as input proceeds
    only if its own settings also set ``write_anyway=True``; the new result
    then carries the band ``"refused"``.

    Parameters
    ----------
    what : str
        The input result's name as messages write it, for example
        ``"The model 'nfw'"``.
    report : ValidationReport
        The validation report of the input result.
    settings : Settings
        The settings of the computation that takes the input result.

    Raises
    ------
    AccuracyError
        If the band of `report` is ``"refused"`` and `settings` does not set
        ``write_anyway=True``.

    See Also
    --------
    announce : The message for a result above :math:`10^{-6}`.
    Settings : The setting ``write_anyway``.

    Examples
    --------
    >>> from pianistpy import Settings
    >>> report = ValidationReport(("a",), {"a": Check(0.5)})
    >>> report.tier
    'refused'
    >>> refuse_unless_asked("The model 'x'", report, Settings(write_anyway=True))
    >>> refuse_unless_asked("The model 'x'", report, Settings())  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    pianistpy.exceptions.AccuracyError: The model 'x' is beyond 0.01 (tier ...
    """
    if report.level >= 3 and not settings.write_anyway:
        raise AccuracyError(
            f"{what} is beyond 0.01 (tier 'refused'), so a computation does not "
            "take the result as input; with Settings(write_anyway=True), the "
            "computation takes the result as input and the new result carries "
            "the tier 'refused'."
        )


def announce(what: str, report: ValidationReport, detail: str) -> None:
    r"""Report a result whose relative error is above :math:`10^{-6}`.

    The announcement depends on the band of `report`. A result in the band
    ``"silent"`` (at most :math:`10^{-6}`) is not announced. A result in the
    band ``"recorded"`` (above :math:`10^{-6}`, at most :math:`10^{-4}`) is
    written as one line to the ``pianistpy`` logger at level INFO, and no
    warning is issued. A result in the band ``"warned"`` (above
    :math:`10^{-4}`, at most :math:`10^{-2}`) is written to the
    ``pianistpy`` logger at level WARNING and issued as an
    `AccuracyWarning`, which the standard ``warnings`` filter can silence or
    turn into an error. A result in the band ``"refused"`` (above
    :math:`10^{-2}`) reaches `announce` only if its settings set
    ``write_anyway=True``; that result is announced as a result in the band
    ``"warned"`` is.

    Parameters
    ----------
    what : str
        The result's name as messages write it, for example
        ``"The sampler of 'nfw'"``.
    report : ValidationReport
        The validation report of the result.
    detail : str
        The checks above :math:`10^{-6}`, each with its relative error and
        the position of its largest error.

    Warns
    -----
    AccuracyWarning
        If the band of `report` is ``"warned"`` or ``"refused"``.

    See Also
    --------
    refuse_unless_asked : The error for an input in the band ``"refused"``.
    set_verbosity : The lowest level of the messages printed on the screen.

    Examples
    --------
    >>> silent = ValidationReport(("a",), {"a": Check(1e-8)})
    >>> announce("The sampler of 'x'", silent, "a 1.0e-08")
    """
    level = report.level
    if level == 0:
        return
    message = (
        f"{what} is above {_BOUND_TEXT[level - 1]} (tier {report.tier!r}): {detail}."
    )
    if level == 1:
        _log.info(message)
        return
    _log.warning(message)
    warnings.warn(message, AccuracyWarning, stacklevel=3)
