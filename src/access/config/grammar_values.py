# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""The value types a grammar can express, and how each maps to a Python value.

Scalar values are written in the grammar as dedicated *value-type rules* -- ``integer``,
``float``, ``logical`` and so on, all imported from ``config.lark``. This module is the
other half of that contract: ``ValueTypeHandler`` bundles the operations needed to handle
one such type (type checking, parsing, serialisation), and ``VALUE_TYPE_HANDLER_REGISTRY``
maps each rule name to its handler. A rule name absent from the registry is not a value
position, which is how the rest of the package recognises one.

Several handlers accept the same Python type, so writing a value for a key that does not
exist yet also needs a rule to be *chosen*. ``VALUE_RULE_PRIORITY`` and
``select_value_rule`` do that, and ``UnsupportedEntryError`` reports a value that no
admitted rule can express.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection, Sequence
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


def _fortran_string_to_str(value: str, token_text: str) -> str:
    """Quote a string the Fortran way, doubling any quote character it contains.

    Unlike ``_string_to_str``, no value is out of reach: Fortran escapes a quote by writing
    it twice, so a string holding both quote characters can still be written. The quote
    already in use is kept, so rewriting a value does not restyle the line.

    Args:
        value (str): The string to quote.
        token_text (str): Text of the ``Token`` that previously held the string, whose first
            character is the quote in use.

    Returns:
        str: The quoted string.
    """
    quote = token_text[0]
    return quote + value.replace(quote, quote * 2) + quote


def _is_fortran_bare_word(value: Any) -> bool:
    """Report whether a value can be written as an unquoted word in a Fortran namelist.

    Narrower than a Python identifier in two ways. The word has to be ASCII, since that is
    all the grammar's terminal matches, and it must not be one of Fortran's logical
    spellings: written bare, ``true`` is not the string "true" but ``.true.``, so a value
    that reads back as a logical has to be refused rather than silently change type.
    """
    return type(value) is str and re.fullmatch(r"(?!(?i:t|f|true|false)\Z)[A-Za-z_][A-Za-z0-9_]*", value) is not None


def _logical_to_str(value: bool, token_text: str) -> str:
    """Convert a bool to a string, keeping the notation of the original token.

    Fortran spells a logical several ways, so rewriting one should not also restyle it. The
    dots, the case and the length of the word are all kept: ``.TRUE.`` stays dotted, upper
    and spelled out, a bare ``t`` stays a bare letter, and ``true`` stays a whole word.

    Args:
        value (bool): The value to write.
        token_text (str): Text of the ``Token`` that previously held the logical.

    Returns:
        str: The logical as a string.
    """
    stripped = token_text.strip(".")
    word = ("true" if value else "false") if len(stripped) > 1 else ("t" if value else "f")
    text = token_text[: len(token_text) - len(token_text.lstrip("."))] + word
    if token_text.endswith("."):
        text += "."
    return text.upper() if token_text.isupper() else text


def _exponent_letter(token_text: str) -> str:
    """Put back the ``e`` Fortran allows a real to leave out before a signed exponent.

    ``1+0`` is 1.0 and ``5-1`` is 0.5 -- the exponent letter is optional when the exponent
    carries a sign, which Python's ``float`` does not accept. A sign at the start is not an
    exponent, so the search for one begins past it.

    Args:
        token_text (str): The token as the file wrote it.

    Returns:
        str: The same number in a spelling ``float`` accepts.
    """
    body = token_text[1:] if token_text[:1] in "+-" else token_text
    position = max(body.rfind("+"), body.rfind("-"))
    if position < 0 or any(letter in token_text.lower() for letter in "ed"):
        return token_text
    offset = position + len(token_text) - len(body)
    return f"{token_text[:offset]}e{token_text[offset:]}"


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
        seed_token (Callable[[Any], str]): Produces stand-in "previous token text" for a
            value that has none, i.e. one being written for a key that does not exist yet.
            It takes the value because the right stand-in can depend on it, as it does for
            a quoted string.
    """

    type_check: Callable[[Any], bool]
    from_token: Callable[[str], Any]
    to_token: Callable[[Any, str], str]
    seed_token: Callable[[Any], str]

    def to_new_token(self, value: Any) -> str:
        """Convert a Python value into token text for a brand-new entry.

        ``to_token`` needs the text of the token being replaced in order to preserve
        notation, but a key that does not exist yet has no such token. This supplies a seed
        instead, so that there is only one serialisation path.

        Args:
            value (Any): The value to serialise.

        Returns:
            str: The token text.
        """
        return self.to_token(value, self.seed_token(value))


VALUE_TYPE_HANDLER_REGISTRY: dict[str, ValueTypeHandler] = {
    "logical": ValueTypeHandler(
        type_check=lambda value: type(value) is bool,
        # Every Fortran spelling is an optional dot then T or F, so the letter decides.
        from_token=lambda token: str(token).lstrip(".")[:1].lower() == "t",
        to_token=_logical_to_str,
        seed_token=lambda value: ".true.",
    ),
    "bool": ValueTypeHandler(
        type_check=lambda value: type(value) is bool,
        from_token=lambda token: str(token) == "True",
        to_token=lambda value, token: "True" if value else "False",
        seed_token=lambda value: "True",
    ),
    "integer": ValueTypeHandler(
        type_check=lambda value: type(value) is int,
        from_token=lambda token: int(token),
        to_token=lambda value, token: str(value),
        seed_token=lambda value: "0",
    ),
    "float": ValueTypeHandler(
        type_check=lambda value: type(value) is float,
        from_token=lambda token: float(token),
        to_token=lambda value, token: _float_to_str(value, token),
        # No exponent character, so _float_to_str uses Python's own notation.
        seed_token=lambda value: "0.0",
    ),
    "bare_exponent": ValueTypeHandler(
        type_check=lambda value: type(value) is float,
        from_token=lambda token: float(_exponent_letter(token)),
        to_token=lambda value, token: _float_to_str(value, token),
        # Nothing writes this notation from scratch; a value written into such a node
        # takes the plain one, which every grammar admitting this rule also admits.
        seed_token=lambda value: "0.0",
    ),
    "double": ValueTypeHandler(
        type_check=lambda value: type(value) is float,
        from_token=lambda token: float(token.replace("D", "E").replace("d", "e")),
        to_token=lambda value, token: _float_to_str(value, token),
        # The "d" makes _float_to_str emit a Fortran double exponent.
        seed_token=lambda value: "0.0d0",
    ),
    "complex": ValueTypeHandler(
        type_check=lambda value: type(value) is complex,
        from_token=lambda token: complex(*map(float, token.strip("()").split(","))),
        to_token=lambda value, token: (
            "(" + _float_to_str(value.real, token) + ", " + _float_to_str(value.imag, token) + ")"
        ),
        seed_token=lambda value: "(0.0, 0.0)",
    ),
    "double_complex": ValueTypeHandler(
        type_check=lambda value: type(value) is complex,
        from_token=lambda token: complex(*map(float, token.replace("D", "E").replace("d", "e").strip("()").split(","))),
        to_token=lambda value, token: (
            "(" + _float_to_str(value.real, token) + ", " + _float_to_str(value.imag, token) + ")"
        ),
        seed_token=lambda value: "(0.0d0, 0.0d0)",
    ),
    "fortran_identifier": ValueTypeHandler(
        type_check=_is_fortran_bare_word,
        from_token=lambda token: str(token),
        to_token=lambda value, token: value,
        seed_token=lambda value: "",
    ),
    "identifier": ValueTypeHandler(
        type_check=lambda value: type(value) is str and value.isidentifier(),
        from_token=lambda token: str(token),
        to_token=lambda value, token: value,
        seed_token=lambda value: "",
    ),
    "string": ValueTypeHandler(
        type_check=lambda value: type(value) is str,
        from_token=lambda token: token[1:-1],
        to_token=_string_to_str,
        # The quote is chosen by _string_to_str; this only states the preference.
        seed_token=lambda value: '""',
    ),
    "fortran_string": ValueTypeHandler(
        type_check=lambda value: type(value) is str,
        from_token=lambda token: token[1:-1].replace(token[0] * 2, token[0]),
        to_token=_fortran_string_to_str,
        # The quote is chosen by _fortran_string_to_str; this only states the preference.
        seed_token=lambda value: '""',
    ),
    "path": ValueTypeHandler(
        type_check=lambda value: isinstance(value, Path),
        from_token=lambda token: Path(token),
        to_token=lambda value, token: str(value),
        seed_token=lambda value: "",
    ),
}
"""Registry mapping value-type rule names to their ``ValueTypeHandler``.

Value-type rules are the grammar rules that appear as alternatives of the ``value`` rule
(e.g. ``"integer"``, ``"float"``, ``"logical"``). Each rule name maps to a handler that
knows how to check, parse, and serialise the corresponding Python type.
"""


VALUE_RULE_PRIORITY: tuple[str, ...] = (
    "logical",
    "bool",
    "integer",
    "float",
    "double",
    "bare_exponent",
    "complex",
    "double_complex",
    "fortran_string",
    "string",
    "fortran_identifier",
    "identifier",
    "path",
)
"""Order in which value-type rules are tried when writing a value for a brand-new key.

Several handlers accept the same Python type, so the choice has to be made deterministically
rather than left to whichever grammar rule happens to match first:

- ``logical`` before ``bool``, so a Fortran-style grammar writes ``.true.``, not ``True``.
- ``float`` before ``double``, so plain notation is preferred over a Fortran ``D`` exponent.
- ``string`` before ``identifier``, so quoting is preferred wherever it is allowed, and bare
  words are used only by grammars that have no ``string`` rule.

Only rules the grammar actually admits are considered, so a format picks the first entry it
supports: ``True`` becomes ``.true.`` in a namelist but ``True`` in a MOM6 input file.

This expresses a preference, not a restriction. The caller tries each accepting rule in turn
and validates the result, so the order changes how output looks but never what it means. A
parser can override it through ``ConfigParser.value_rule_priority``.
"""


def select_value_rule(
    value: Any, admitted: Collection[str], priority: Sequence[str] = VALUE_RULE_PRIORITY
) -> str | None:
    """Choose the value-type rule to use when writing *value* into a new entry.

    Args:
        value (Any): The Python value to be written.
        admitted (Collection[str]): Names of the value-type rules the grammar allows in this
            position.
        priority (Sequence[str]): Order in which to consider rules.

    Returns:
        str | None: The chosen rule name, or ``None`` if no admitted rule accepts *value*.
    """
    for rule_name in priority:
        handler = VALUE_TYPE_HANDLER_REGISTRY.get(rule_name)
        if rule_name in admitted and handler is not None and handler.type_check(value):
            return rule_name
    return None
