# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Value type handlers for the configuration parser.

This module defines ``ValueTypeHandler``, a dataclass that bundles the three operations needed
to handle a single value type (type checking, parsing, and serialisation), and provides the
``VALUE_TYPE_HANDLER_REGISTRY`` that maps grammar rule names to their handlers.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _float_to_str(value: float, token: str) -> str:
    """Given a float and a Lark token, convert the float to a string using the same notation as used in the token.
    This is to handle cases where the old Fortran notation is used (e.g.: 1.0d10 or 1.0D10).

    Args:
        value (float): float to be converted
        token (Token): Lark token holding the float

    Returns:
        str: the float as a string
    """
    for c in token:
        if c in ["D", "d", "E", "e"]:
            return str(value).replace("e", c)
    else:
        return str(value)


@dataclass(frozen=True)
class ValueTypeHandler:
    """Encapsulates all operations for a single value type.

    Each handler bundles three operations related to a single value type:
    - Type checking: a function that checks if a Python value matches this type.
    - Parsing: a function that converts a Lark token string into the corresponding Python value.
    - Serialisation: a function that converts a Python value back into the string representation used in the Lark token,
      preserving any notation specifics (e.g. Fortran ``D`` exponent). This is used when updating the parse tree with
      new values, to make sure that the formatting of the original file is preserved as much as possible.

    Args:
        type_check: Predicate that returns ``True`` when a Python value matches this type.
        from_token: Converts a Lark token string into the corresponding Python value.
        to_token: Converts a Python value back into the string representation used
            in the Lark token, preserving any notation specifics (e.g. Fortran ``D`` exponent).
    """

    type_check: Callable[[Any], bool]
    from_token: Callable[[str], Any]
    to_token: Callable[[Any, str], Any]


VALUE_TYPE_HANDLER_REGISTRY: dict[str, ValueTypeHandler] = {
    "logical": ValueTypeHandler(
        type_check=lambda value: type(value) is bool,
        from_token=lambda token: str(token).lower() == ".true.",
        to_token=lambda value, token: ".true." if value else ".false.",
    ),
    "bool": ValueTypeHandler(
        type_check=lambda value: type(value) is bool,
        from_token=lambda token: str(token) == "True",
        to_token=lambda value, token: "True" if value else "False",
    ),
    "integer": ValueTypeHandler(
        type_check=lambda value: type(value) is int,
        from_token=lambda token: int(token),
        to_token=lambda value, token: value,
    ),
    "float": ValueTypeHandler(
        type_check=lambda value: type(value) is float,
        from_token=lambda token: float(token),
        to_token=lambda value, token: _float_to_str(value, token),
    ),
    "double": ValueTypeHandler(
        type_check=lambda value: type(value) is float,
        from_token=lambda token: float(token.replace("D", "E").replace("d", "e")),
        to_token=lambda value, token: _float_to_str(value, token),
    ),
    "complex": ValueTypeHandler(
        type_check=lambda value: type(value) is complex,
        from_token=lambda token: complex(*map(float, token.strip("()").split(","))),
        to_token=lambda value, token: (
            "(" + _float_to_str(value.real, token) + ", " + _float_to_str(value.imag, token) + ")"
        ),
    ),
    "double_complex": ValueTypeHandler(
        type_check=lambda value: type(value) is complex,
        from_token=lambda token: complex(*map(float, token.replace("D", "E").replace("d", "e").strip("()").split(","))),
        to_token=lambda value, token: (
            "(" + _float_to_str(value.real, token) + ", " + _float_to_str(value.imag, token) + ")"
        ),
    ),
    "identifier": ValueTypeHandler(
        type_check=lambda value: type(value) is str and value.isidentifier(),
        from_token=lambda token: str(token),
        to_token=lambda value, token: value,
    ),
    "string": ValueTypeHandler(
        type_check=lambda value: type(value) is str,
        from_token=lambda token: token[1:-1],
        to_token=lambda value, token: token[0] + value + token[-1],
    ),
    "path": ValueTypeHandler(
        type_check=lambda value: isinstance(value, Path),
        from_token=lambda token: Path(token),
        to_token=lambda value, token: str(value),
    ),
}
"""Registry mapping grammar rule names to their ``ValueTypeHandler``.  Each handler knows how to check, parse,
and serialise a single value type."""
