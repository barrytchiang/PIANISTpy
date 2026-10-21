"""The identity string of a result: a digest of everything that determines
the result's values.

The package computes a result's identity from the result's inputs, the
result's settings (`Settings.numerical`), the package's version and the file
format's version. Two results with the same identity were computed from
identical inputs, settings and versions. A result file stores the
identity, and a derived result stores the identity of each parent.
"""

__all__ = ["FORMAT_VERSION", "identity_string", "short_identity"]

import hashlib
import json
from collections.abc import Mapping
from typing import Any

# The version of the result file format, part of every identity.
FORMAT_VERSION = "2.0"
# The number of leading hexadecimal characters of an identity shown to the
# user.
SHORT_IDENTITY_LENGTH = 12


def identity_string(
    inputs: Mapping[str, Any],
    settings: Mapping[str, Any],
    code_version: str,
    format_version: str = FORMAT_VERSION,
) -> str:
    """Return the identity string of a result.

    The identity is the SHA-256 digest of a JSON text of four entries: the
    inputs, the settings, the package's version and the file format's
    version. The JSON text sorts every dictionary's keys, so the identity
    does not depend on the order in which a dictionary's keys were
    inserted. The date of the computation, the hardware and the computed
    arrays are not part of the identity.

    Parameters
    ----------
    inputs : mapping
        The result's inputs in their dictionary form. Every value must be
        representable in JSON.
    settings : mapping
        The result's settings in their dictionary form
        (`Settings.numerical`).
    code_version : str
        The version of the package that computes the result.
    format_version : str, optional
        The version of the result file format. Default:
        ``FORMAT_VERSION``, which means the format the package writes.

    Returns
    -------
    identity : str
        The hexadecimal SHA-256 digest, 64 characters.

    Raises
    ------
    ValueError
        If a value of `inputs` or `settings` is NaN or infinite.
    TypeError
        If a value of `inputs` or `settings` is not representable in JSON.

    See Also
    --------
    short_identity : The leading characters of an identity.

    Examples
    --------
    >>> a = identity_string({"beta_c": 0.0}, {"level": "default"}, "1.0")
    >>> b = identity_string({"beta_c": 0.0}, {"level": "default"}, "1.0")
    >>> a == b
    True
    >>> len(a)
    64
    """
    record = {
        "inputs": inputs,
        "settings": settings,
        "code_version": code_version,
        "format_version": format_version,
    }
    # Sorted keys and fixed separators make the JSON text, and so the
    # digest, independent of the order in which each dictionary's keys were
    # inserted.
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_identity(identity: str) -> str:
    """Return the short form of an identity string, as messages show it.

    Parameters
    ----------
    identity : str
        An identity string that `identity_string` returned.

    Returns
    -------
    short : str
        The first 12 characters of `identity`.

    See Also
    --------
    identity_string : The full identity of a result.

    Examples
    --------
    >>> identity = identity_string({"beta_c": 0.0}, {"level": "default"}, "1.0")
    >>> short_identity(identity) == identity[:12]
    True
    """
    return identity[:SHORT_IDENTITY_LENGTH]
