"""Identity strings of results."""

__all__ = ["FORMAT_VERSION", "identity_string", "short_identity"]

import hashlib
import json
from collections.abc import Mapping
from typing import Any

FORMAT_VERSION = "2.0"
SHORT_IDENTITY_LENGTH = 12


def identity_string(
    inputs: Mapping[str, Any],
    settings: Mapping[str, Any],
    code_version: str,
    format_version: str = FORMAT_VERSION,
) -> str:
    """Return the identity string of a result.

    The identity is a digest of everything that determines the result: the
    inputs, the settings, the version of the code and the version of the file
    format. Two results with the same identity were asked for identically.
    The date, the platform and the arrays themselves are not part of it.

    Parameters
    ----------
    inputs : mapping
        The inputs in their dictionary form.
    settings : mapping
        The settings in their dictionary form.
    code_version : str
        Version of the package that computes the result.
    format_version : str, optional
        Version of the file format.

    Returns
    -------
    identity : str
        Hexadecimal SHA-256 digest, 64 characters.

    Examples
    --------
    >>> a = identity_string({"beta_c": 0.0}, {"tier": "default"}, "1.0")
    >>> b = identity_string({"beta_c": 0.0}, {"tier": "default"}, "1.0")
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
    # Sorted keys and fixed separators make the text, and so the digest,
    # independent of the order in which the dictionaries were built.
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_identity(identity: str) -> str:
    """Return the short form of an identity string, as shown to the user.

    Parameters
    ----------
    identity : str
        Identity string returned by `identity_string`.

    Returns
    -------
    short : str
        The leading characters of the identity.
    """
    return identity[:SHORT_IDENTITY_LENGTH]
