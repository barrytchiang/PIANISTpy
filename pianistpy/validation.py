"""Accuracy tiers and the validation report of a result."""

__all__ = [
    "CEILING",
    "GOAL",
    "HARD_LIMIT",
    "TIER_NAMES",
    "Check",
    "ValidationReport",
    "announce",
    "refuse_unless_asked",
    "tier_level",
]

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .exceptions import AccuracyError, AccuracyWarning, InputError

_log = logging.getLogger("pianistpy")

# Relative errors that separate the tiers. A result within the goal is
# silent, one within the hard limit is returned with a warning, and one
# beyond the hard limit is refused.
GOAL = 1e-6
CEILING = 1e-4
HARD_LIMIT = 1e-2

TIER_NAMES = ("goal", "ceiling", "warning", "refused")


def tier_level(error: float) -> int:
    """Return the tier of a relative error.

    Parameters
    ----------
    error : float
        Relative error, non-negative. A value that is not a number counts as
        beyond the hard limit.

    Returns
    -------
    level : int
        0 within the goal, 1 within the ceiling, 2 within the hard limit,
        3 beyond it.

    Examples
    --------
    >>> [tier_level(e) for e in (1e-7, 1e-5, 1e-3, 1e-1, float("nan"))]
    [0, 1, 2, 3, 3]
    """
    if not error <= HARD_LIMIT:
        return 3
    if error <= GOAL:
        return 0
    return 1 if error <= CEILING else 2


@dataclass(frozen=True)
class Check:
    """One check of a result: what was measured and where.

    Parameters
    ----------
    error : float
        The relative error the check measured.
    where : float, optional
        The place of the largest error, in the variable the check runs over
        (a radius or an energy). Absent where that has no meaning.
    """

    error: float
    where: float | None = None

    @property
    def level(self) -> int:
        """Tier of the error, as returned by `tier_level`."""
        return tier_level(self.error)

    @property
    def tier(self) -> str:
        """Name of the tier of the error."""
        return TIER_NAMES[self.level]


@dataclass(frozen=True)
class ValidationReport:
    """The validation report of one result.

    The report has a fixed set of keys for each kind of result and one
    `Check` per key. It is not a log: nothing is appended to it, and a
    derived result holds its own checks and the worst tier of its parents,
    never a copy of their reports.

    Parameters
    ----------
    keys : tuple of str
        The keys this kind of result reports, in order.
    checks : mapping of str to Check
        One check per key.
    parents_level : int, optional
        Worst tier among the parents of the result.
    """

    keys: tuple[str, ...]
    checks: Mapping[str, Check]
    parents_level: int = 0

    def __post_init__(self) -> None:
        if set(self.checks) != set(self.keys):
            raise InputError(
                f"A report of this kind has the keys {list(self.keys)}; "
                f"got {sorted(self.checks)}."
            )

    @property
    def level(self) -> int:
        """Worst tier of the result's own checks and of its parents."""
        own = max((check.level for check in self.checks.values()), default=0)
        return max(own, self.parents_level)

    @property
    def tier(self) -> str:
        """Name of the worst tier."""
        return TIER_NAMES[self.level]

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ValidationReport":
        """Build the report from its dictionary form."""
        checks = {
            key: Check(error=float(item["error"]), where=item["where"])
            for key, item in record["checks"].items()
        }
        return cls(
            keys=tuple(record["checks"]),
            checks=checks,
            parents_level=TIER_NAMES.index(record["parents_tier"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a plain dictionary, in the order of its keys."""
        return {
            "tier": self.tier,
            "parents_tier": TIER_NAMES[self.parents_level],
            "checks": {
                key: {
                    "error": self.checks[key].error,
                    "where": self.checks[key].where,
                    "tier": self.checks[key].tier,
                }
                for key in self.keys
            },
        }


def refuse_unless_asked(what: str, report: ValidationReport, settings: Any) -> None:
    """Refuse a result beyond the hard limit as the input of another.

    A result whose report reads ``"refused"`` exists only because it was asked
    for anyway (``Settings(write_anyway=True)``) or was read from a file that
    holds one. It is consumed only under the same request: a sampler, a table
    or a projection built on it carries its tier and says so.

    Parameters
    ----------
    what : str
        The input, named: ``"The model 'nfw'"``.
    report : ValidationReport
        Its report.
    settings : Settings
        The settings of the result being built from it.

    Raises
    ------
    AccuracyError
        If the report's tier is ``"refused"`` and the settings do not ask for
        the result anyway.
    """
    if report.level >= 3 and not settings.write_anyway:
        raise AccuracyError(
            f"{what} is beyond the hard limit {HARD_LIMIT:g} (tier 'refused') and "
            "is not consumed; Settings(write_anyway=True) builds from it regardless, "
            "with its tier carried."
        )


def announce(what: str, report: ValidationReport, detail: str) -> None:
    """Say that a result lies above the goal; silent within it.

    The one channel for every block: the package's log (the ``pianistpy``
    logger, printed by default) and Python's own warnings, as an
    `AccuracyWarning`, so that a user can silence or escalate it with the
    standard filter. Nothing is said within the goal; beyond the hard limit
    the result is refused by its stage, not announced here.

    Parameters
    ----------
    what : str
        The result, named: ``"The sampler of 'nfw'"``.
    report : ValidationReport
        Its report.
    detail : str
        The checks above the goal with their errors and where they occur.
    """
    if report.level > 0:
        message = f"{what} is above the goal (tier {report.tier!r}): {detail}."
        _log.warning(message)
        warnings.warn(message, AccuracyWarning, stacklevel=3)
