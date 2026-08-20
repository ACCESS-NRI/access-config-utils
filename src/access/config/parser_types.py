# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Value type handlers for the configuration parser.

This module defines ``ValueTypeHandler``, a dataclass that bundles the three operations
needed to handle a single value type (type checking, parsing, and serialisation), and
provides the ``VALUE_TYPE_HANDLER_REGISTRY`` that maps grammar rule names to their
handlers.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UnsupportedEntryError(TypeError):
    """Raised when a grammar admits no way to express a requested value or entry.

    Subclasses ``TypeError`` because it reports that a Python value has no representation in
    the target file format, which is the same category of problem as the type mismatches
    reported by ``update_node_value``.
    """


def _string_to_str(value: str, token_text: str) -> str:
    """Quote a string, keeping the quote character already in use where possible.

    Values are wrapped verbatim rather than escaped, so the quote character has to be one
    the value does not itself contain. The one already in use is preferred, so that
    rewriting a value does not gratuitously change the quoting of the line.

    Args:
        value (str): The string to quote.
        token_text (str): Text of the ``Token`` that previously held the string, whose first
            character is the quote in use.

    Returns:
        str: The quoted string.

    Raises:
        UnsupportedEntryError: If *value* contains both quote characters, and so cannot be
            written without escaping.
    """
    quote = token_text[0]
    if quote in value:
        quote = "'" if quote == '"' else '"'
        if quote in value:
            raise UnsupportedEntryError("Strings containing both quote characters cannot be represented")
    return quote + value + quote


def _float_to_str(value: float, token_text: str) -> str:
    """Convert a float to a string using the exponent notation of the original token.

    This preserves Fortran-style exponent notation (e.g. ``1.0d10`` or ``1.0D10``) when
    round-trip serialising a value back into its token text.

    Args:
        value (float): Float to be converted.
        token_text (str): Text of the ``Token`` that previously held the float (i.e.
            ``Token.value``).

    Returns:
        str: The float as a string.

    Raises:
        UnsupportedEntryError: If *value* is an infinity or a NaN. Python writes those as
            bare words that no supported format can read back, so writing one would produce
            a file that no longer parses.
    """
    if not math.isfinite(value):
        raise UnsupportedEntryError(f"{value!r} cannot be represented in a configuration file")
    for c in token_text:
        if c in ["D", "d", "E", "e"]:
            return str(value).replace("e", c)
    else:
        return str(value)


@dataclass(frozen=True)
class ValueTypeHandler:
    """Encapsulates all operations for a single value type.

    Each handler bundles three operations related to a single value type:

    - Type checking: a function that checks if a Python value matches this type.
    - Parsing: a function that converts **token text** into the corresponding Python
      value. Token text is the string matched by a grammar terminal. It is the ``str``
      value of the ``Token`` object (since ``Token`` is a subclass of ``str``, it can be
      passed directly).
    - Serialisation: a function that converts a Python value back into token text, given
      the previous token text so that notation specifics (e.g. Fortran ``D`` exponent) can
      be preserved. This is used when updating value-type rule nodes with new values.

    Args:
        type_check (Callable[[Any], bool]): Predicate that returns ``True`` when a Python
            value matches this type.
        from_token (Callable[[str], Any]): Converts token text (the ``str`` value matched
            by the terminal) into the corresponding Python value.
        to_token (Callable[[Any, str], str]): Converts a Python value back into token
            text, given the previous token text (to preserve notation, e.g. Fortran ``D``
            exponent).
    """

    type_check: Callable[[Any], bool]
    from_token: Callable[[str], Any]
    to_token: Callable[[Any, str], str]


VALUE_TYPE_HANDLER_REGISTRY: dict[str | None, ValueTypeHandler] = {
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
        to_token=lambda value, token: str(value),
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
        to_token=_string_to_str,
    ),
    "path": ValueTypeHandler(
        type_check=lambda value: isinstance(value, Path),
        from_token=lambda token: Path(token),
        to_token=lambda value, token: str(value),
    ),
}
"""Registry mapping value-type rule names to their ``ValueTypeHandler``.

Value-type rules are the grammar rules that appear as alternatives of the ``value`` rule
(e.g. ``"integer"``, ``"float"``, ``"logical"``). Each rule name maps to a handler that
knows how to check, parse, and serialise the corresponding Python type.
"""
