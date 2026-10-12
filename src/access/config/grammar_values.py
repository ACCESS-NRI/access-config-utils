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
exist yet also needs a rule to be *chosen*. The order of the registry is the order of
preference and ``select_value_rule`` applies it, and ``UnsupportedEntryError`` reports a
value that no admitted rule can express.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
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
    """Quote a string the way ``fortran_string`` does, doubling any quote it contains.

    Unlike ``_string_to_str``, no value is out of reach: that rule escapes a quote by
    doubling it, so a string holding both quote characters can still be written. The quote
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
    """Report whether a value can be written as an unquoted word by ``fortran_identifier``.

    Narrower than a Python identifier in two ways. The word has to be ASCII, since that is
    all the grammar's terminal matches, and it must not be one of the logical spellings that
    rule sits beside: written bare, ``true`` is not the string "true" but ``.true.``, so a
    value that reads back as a logical has to be refused rather than silently change type.
    """
    return type(value) is str and re.fullmatch(r"(?!(?i:t|f|true|false)\Z)[A-Za-z_][A-Za-z0-9_]*", value) is not None


def _logical_to_str(value: bool, token_text: str) -> str:
    """Convert a bool to a string, keeping the notation of the original token.

    A grammar may spell a logical several ways, so rewriting one should not restyle it. The
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
    """Put back the ``e`` a real may leave out before a signed exponent.

    A grammar may admit ``1+0`` for 1.0 and ``5-1`` for 0.5, writing the exponent's sign
    in place of the letter, which Python's ``float`` does not accept. A sign at the start
    is not an exponent, so the search for one begins past it.

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

    This preserves the exponent notation of the token (e.g. ``1.0d10`` or ``1.0D10``) when
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


def _is_trimmed(value: Any, banned: str) -> bool:
    """Report whether *value* is a non-empty string, trimmed, holding none of *banned*."""
    return type(value) is str and value != "" and value == value.strip() and not any(c in value for c in banned)


def _complex_to_str(value: complex, token_text: str) -> str:
    """Convert a complex to a string, each part in the notation of the original token."""
    return "(" + _float_to_str(value.real, token_text) + ", " + _float_to_str(value.imag, token_text) + ")"


def _is_bool(value: Any) -> bool:
    """Report whether *value* is a bool, which an ``int`` is not for this purpose."""
    return type(value) is bool


def _is_str(value: Any) -> bool:
    """Report whether *value* is a plain string."""
    return type(value) is str


def _is_text_line(value: Any) -> bool:
    """Report whether *value* can be written as a line of free text.

    The text runs to the end of the line or to a comment, and the whitespace either side of
    it belongs to the line rather than to the value, so a value with a comment character, a
    line break, or whitespace at either end would not read back as itself. Nor would one
    starting with ``::``, which closes an ESMF table.
    """
    return _is_trimmed(value, "#\r\n") and not value.startswith("::")


# --- Values typed by their spelling ---
#
# The FMS field manager gives a value no declared type: it reads one from the spelling, in
# ``new_name`` of ``field_manager.F90``. A logical is ``t``, ``f``, ``.t.``, ``.f.``,
# ``true``, ``false``, ``.true.`` or ``.false.`` in any case; a number starts with a digit,
# a sign or a point and holds no letter but an exponent's, and is a real if it holds a
# point or an exponent; text between single quotes is that text; anything else is a
# string. A comma parts the elements of a list, and an equals sign a parameter from its
# value, wherever either is written, so no value can hold one.

_LOGICAL_SPELLINGS = {
    "t": True,
    "f": False,
    ".t.": True,
    ".f.": False,
    "true": True,
    "false": False,
    ".true.": True,
    ".false.": False,
}
"""The spellings of a logical, in lower case, and what each means."""

_INFERRED_SEEDS: dict[type, str] = {bool: ".true.", int: "0", float: "0.0", str: "x"}
"""A spelling of each type, standing in for the previous text of a brand-new value."""


def _single_quoted(text: str) -> bool:
    """Report whether *text* is wrapped in single quotes, which makes it a string."""
    return len(text) > 1 and text[0] == text[-1] == "'"


def _infer(text: str) -> bool | int | float | str:
    """Return the value the FMS field manager reads *text* as, typed by its spelling.

    Args:
        text (str): One value, without the whitespace around it.

    Returns:
        bool | int | float | str: The value.
    """
    if _single_quoted(text):
        return text[1:-1]
    logical = _LOGICAL_SPELLINGS.get(text.lower())
    if logical is not None:
        return logical
    if re.fullmatch(r"[+-]?[0-9]+", text):
        return int(text)
    if re.fullmatch(r"[0-9+.-][0-9+.eE-]*", text) and re.search("[.eE]", text):
        try:
            return float(text)
        except ValueError:
            pass
    return text


def _is_inferable(value: Any) -> bool:
    """Report whether *value* has a spelling the field manager reads back as exactly it.

    A logical, an integer, a finite real, or a string that is not empty, has no whitespace
    at either end, and holds no double quote, comma, equals sign or line break.
    """
    if type(value) in (bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    return _is_trimmed(value, '",=\r\n')


def _inferred_to_str(value: bool | int | float | str, previous: str, *, bare: bool = False) -> str:
    """Spell *value* so that it reads back as itself, in the notation of *previous*.

    A logical keeps the spelling it replaces, and a real its exponent letter. A string is
    written in single quotes where it would otherwise read back as something else -- ``'3'``
    for the string "3" -- or where the text it replaces was quoted.

    Args:
        value (bool | int | float | str): The value, one ``_is_inferable`` accepts.
        previous (str): The text the value replaces, without the whitespace around it.
        bare (bool): Whether the text ends a line, where a closing ``/`` would end the
            entry instead, so a string ending with one has to be quoted.

    Returns:
        str: The text.

    Raises:
        TypeError: If *previous* reads as a value of another type.
    """
    if type(value) is not type(_infer(previous)):
        raise TypeError("Trying to change value type")
    if type(value) is bool:
        return _logical_to_str(value, previous)
    if type(value) is float:
        return _float_to_str(value, previous)
    if type(value) is int:
        return str(value)
    assert isinstance(value, str)
    reads_back = identical_values(_infer(value), value) and not value.startswith("'") and not value.endswith("'")
    if _single_quoted(previous) or not reads_back or (bare and value.endswith("/")):
        return f"'{value}'"
    return value


def _rewrite_padded(value: Any, text: str, *, bare: bool = False) -> str:
    """Write *value* over *text*, keeping the whitespace around it."""
    inner = text.strip()
    start = text.index(inner) if inner else len(text)
    return text[:start] + _inferred_to_str(value, inner, bare=bare) + text[start + len(inner) :]


# --- Named assignments ---
#
# An FMS field table method can be written as three strings: the method, a name, and a
# control of comma-separated parameters, as in ``"profile_type", "fixed",
# "surface_value=3.e-6"``. The last two make one value, a record holding the name under
# ``"value"`` -- the key the FMS YAML field table gives it -- and each parameter beside it.


def _record_parts(token: str) -> tuple[str, str, str]:
    """Split a record token into its name, the text between its strings, and its control."""
    match = re.fullmatch(r'"([^"]*)"([ \t]*,[ \t]*)"([^"]*)"', token)
    if match is None:
        raise ValueError(f"Not a name and a control in double quotes: {token}")
    return match.group(1), match.group(2), match.group(3)


def _control_segments(control: str) -> list[tuple[str, list[str]]]:
    """Return each parameter of a control with the texts of its values, padding and all.

    A segment holding no equals sign continues the list of values of the parameter before
    it, as the field manager reads it, so ``a=1,2`` is ``a`` holding two values.

    Args:
        control (str): The control, without its quotes.

    Returns:
        list[tuple[str, list[str]]]: The name of each parameter as written, and its values.

    Raises:
        ValueError: If a comma comes before the first equals sign, which the field manager
            refuses, or a segment holds two equals signs.
    """
    segments: list[tuple[str, list[str]]] = []
    for segment in control.split(",") if control else []:
        if segment.count("=") > 1:
            raise ValueError(f"A parameter with two equals signs: {segment!r}")
        if "=" in segment:
            name, _, text = segment.partition("=")
            segments.append((name, [text]))
        elif not segments:
            raise ValueError(f"A comma before the first equals sign: {control!r}")
        else:
            segments[-1][1].append(segment)
    return segments


def _read_record(token: str) -> dict[str, Any]:
    """Return the record a token writes: the name under ``"value"``, then each parameter.

    Args:
        token (str): The token, ``"name", "control"``.

    Returns:
        dict[str, Any]: The record. A parameter of several values holds a list of them.

    Raises:
        ValueError: If the control is malformed, names no parameter, names one twice or
            names one ``value``, which is where the name goes.
    """
    name, _, control = _record_parts(token)
    record: dict[str, Any] = {"value": name}
    for raw, texts in _control_segments(control):
        key = raw.strip()
        if not key or key in record:
            raise ValueError(f"A parameter with no name, or named twice or 'value': {raw!r}")
        values = [_infer(text.strip()) for text in texts]
        record[key] = values[0] if len(values) == 1 else values
    return record


def _is_record(value: Any) -> bool:
    """Report whether *value* is a record a control can hold.

    A ``dict`` with a string name under ``"value"`` and at least one parameter, each a
    value ``_is_inferable`` accepts or a list of two or more of them, under a name holding
    nothing a control parts on.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if *value* can be written as a name and a control.
    """
    if type(value) is not dict or len(value) < 2:
        return False
    name = value.get("value")
    if type(name) is not str or name == "" or any(character in name for character in '"\r\n'):
        return False
    for key, item in value.items():
        if key == "value":
            continue
        if not _is_trimmed(key, '",=\r\n'):
            return False
        items = item if type(item) is list else [item]
        if (type(item) is list and len(item) < 2) or not all(_is_inferable(element) for element in items):
            return False
    return True


def _write_record(value: dict[str, Any], token: str) -> str:
    """Write a record over the token it replaces, changing only what changed.

    A parameter left as it was keeps its text, padding and all. One whose value changed
    keeps the padding and notation of each value it replaces, and one that is new is
    written after the others, parted from them as the last one was. A parameter the record
    no longer holds is left out.

    Args:
        value (dict[str, Any]): The record.
        token (str): The token it replaces.

    Returns:
        str: The token text.
    """
    _, between, control = _record_parts(token)
    parameters = _control_segments(control)
    old = {raw.strip(): (raw, texts) for raw, texts in parameters}
    # A new parameter is parted from the one before it as the file parts its last two.
    gap = next((raw[: len(raw) - len(raw.lstrip())] for raw, _ in reversed(parameters[1:])), "")
    segments: list[str] = []
    for key, item in value.items():
        if key == "value":
            continue
        raw, texts = old.get(key, (gap + key if segments else key, []))
        items = item if type(item) is list else [item]
        written = [
            _parameter_text(element, texts[index] if index < len(texts) else None)
            for index, element in enumerate(items)
        ]
        segments.append(raw + "=" + ",".join(written))
    return f'"{value["value"]}"{between}"{",".join(segments)}"'


def _parameter_text(value: Any, previous: str | None) -> str:
    """Return the text of one value of a parameter, over the text it replaces if any."""
    if previous is not None:
        if identical_values(_infer(previous.strip()), value):
            return previous
        try:
            return _rewrite_padded(value, previous)
        except TypeError:
            pass
    return _inferred_to_str(value, _INFERRED_SEEDS[type(value)])


@dataclass(frozen=True)
class ValueTypeHandler:
    """Encapsulates all operations for a single value type.

    Each handler bundles the operations related to a single value type:

    - Type checking: a function that checks if a Python value matches this type.
    - Parsing: a function that converts **token text** into the corresponding Python
      value. Token text is the string matched by a grammar terminal. It is the ``str``
      value of the ``Token`` object (since ``Token`` is a subclass of ``str``, it can be
      passed directly).
    - Serialisation: a function that converts a Python value back into token text, given
      the previous token text so that notation specifics (e.g. a ``D`` exponent) can
      be preserved. This is used when updating value-type rule nodes with new values.

    Args:
        type_check (Callable[[Any], bool]): Predicate that returns ``True`` when a Python
            value matches this type.
        from_token (Callable[[str], Any]): Converts token text (the ``str`` value matched
            by the terminal) into the corresponding Python value. The text itself by
            default.
        to_token (Callable[[Any, str], str]): Converts a Python value back into token
            text, given the previous token text (to preserve notation, e.g. a ``D``
            exponent). The value as ``str`` by default.
        seed (str | Callable[[Any], str]): The stand-in "previous token text" for a value
            that has none, i.e. one being written for a key that does not exist yet; a
            function of the value where the right stand-in depends on it, as it does for a
            value typed by its spelling. Empty by default.
        collects (bool): Whether an entry holding a value of this type is one record of
            several the file writes under the same key, rather than the key's value. An
            FMS field table writes a method that way, once per record, and the key reads
            as the list of them.
    """

    type_check: Callable[[Any], bool]
    from_token: Callable[[str], Any] = str
    to_token: Callable[[Any, str], str] = lambda value, token: str(value)  # noqa: E731
    seed: str | Callable[[Any], str] = ""
    collects: bool = False

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
        return self.to_token(value, self.seed(value) if callable(self.seed) else self.seed)

    def rewrite(self, value: Any, token_text: str) -> str:
        """Return the token text that writes *value* over a token holding *token_text*.

        A token that already holds *value* keeps its text, so writing a list leaves the
        elements that did not change written exactly as they were -- ``1.0d0`` stays
        ``1.0d0`` -- and a value that cannot be written anew, such as a NaN, can still be
        written over itself.

        Args:
            value (Any): The new value.
            token_text (str): The text of the token being replaced.

        Returns:
            str: The token text.

        Raises:
            TypeError: If *value* is not of this type.
        """
        if not self.type_check(value):
            raise TypeError("Trying to change value type")
        if identical_values(self.from_token(token_text), value):
            return token_text
        return self.to_token(value, token_text)


_FLOAT = ValueTypeHandler(
    type_check=lambda value: type(value) is float, from_token=float, to_token=_float_to_str, seed="0.0"
)
_COMPLEX = ValueTypeHandler(
    type_check=lambda value: type(value) is complex,
    from_token=lambda token: complex(*map(float, token.strip("()").split(","))),
    to_token=_complex_to_str,
    seed="(0.0, 0.0)",
)

# The order of the entries is the order in which a rule is tried when a value is written for
# a brand-new key. Several handlers accept the same Python type, so the choice has to be
# made deterministically rather than left to whichever grammar rule happens to match first:
#
# - ``logical`` before ``bool``: a grammar admitting both writes ``.true.``, not ``True``.
# - ``float`` before ``double``, so plain notation is preferred over a ``D`` exponent.
# - ``string`` before ``identifier``, so quoting is preferred wherever it is allowed, and
#   bare words are used only by grammars that have no ``string`` rule.
# - ``identifier`` before ``text_line``, so a whole line of free text is what a string
#   becomes only in a format with nothing narrower to write it as.
# - ``inferred_string`` before ``inferred_text``, so a value goes in quotes where a format
#   allows it either way.
#
# Only rules the grammar actually admits are considered, so a format picks the first entry
# it supports: ``True`` becomes ``.true.`` where a grammar has ``logical``, ``True`` where
# it has only ``bool``. This is a preference, not a restriction: every accepting rule is
# tried in turn and the result validated, so the order changes how output looks but never
# what it means.
VALUE_TYPE_HANDLER_REGISTRY: dict[str, ValueTypeHandler] = {
    "logical": ValueTypeHandler(
        type_check=_is_bool,
        # Every spelling this rule admits is a dot or nothing, then T or F: the letter
        # is what decides.
        from_token=lambda token: str(token).lstrip(".")[:1].lower() == "t",
        to_token=_logical_to_str,
        seed=".true.",
    ),
    "bool": ValueTypeHandler(
        type_check=_is_bool,
        from_token=lambda token: str(token) == "True",
        to_token=lambda value, token: "True" if value else "False",
        seed="True",
    ),
    "integer": ValueTypeHandler(type_check=lambda value: type(value) is int, from_token=int, seed="0"),
    "float": _FLOAT,
    # The "d" makes _float_to_str emit a double exponent.
    "double": replace(_FLOAT, from_token=lambda token: float(token.replace("D", "E").replace("d", "e")), seed="0.0d0"),
    # Nothing writes this notation from scratch; a value written into such a node takes the
    # plain one, which every grammar admitting this rule also admits.
    "bare_exponent": replace(_FLOAT, from_token=lambda token: float(_exponent_letter(token))),
    "complex": _COMPLEX,
    "double_complex": replace(
        _COMPLEX,
        from_token=lambda token: complex(*map(float, token.replace("D", "E").replace("d", "e").strip("()").split(","))),
        seed="(0.0d0, 0.0d0)",
    ),
    "fortran_string": ValueTypeHandler(
        type_check=_is_str,
        from_token=lambda token: token[1:-1].replace(token[0] * 2, token[0]),
        to_token=_fortran_string_to_str,
        # The quote is chosen by _fortran_string_to_str; this only states the preference.
        seed='""',
    ),
    "string": ValueTypeHandler(
        type_check=_is_str,
        from_token=lambda token: token[1:-1],
        to_token=_string_to_str,
        # The quote is chosen by _string_to_str; this only states the preference.
        seed='""',
    ),
    # A value typed by its spelling, as the FMS field manager types one, in double quotes.
    "inferred_string": ValueTypeHandler(
        type_check=_is_inferable,
        from_token=lambda token: _infer(token[1:-1].strip()),
        to_token=lambda value, token: '"' + _rewrite_padded(value, token[1:-1]) + '"',
        seed=lambda value: '"' + _INFERRED_SEEDS[type(value)] + '"',
    ),
    "fortran_identifier": ValueTypeHandler(type_check=_is_fortran_bare_word),
    "identifier": ValueTypeHandler(type_check=lambda value: type(value) is str and value.isidentifier()),
    # The same, bare, ending its line.
    "inferred_text": ValueTypeHandler(
        type_check=_is_inferable,
        from_token=lambda token: _infer(str(token)),
        to_token=lambda value, token: _inferred_to_str(value, token, bare=True),
        seed=lambda value: _INFERRED_SEEDS[type(value)],
    ),
    "text_line": ValueTypeHandler(type_check=_is_text_line),
    "path": ValueTypeHandler(type_check=lambda value: isinstance(value, Path), from_token=Path),
    "named_assignments": ValueTypeHandler(
        type_check=_is_record, from_token=_read_record, to_token=_write_record, seed='"", ""', collects=True
    ),
}
"""Registry mapping value-type rule names to their ``ValueTypeHandler``.

Value-type rules are the grammar rules that appear as alternatives of the ``value`` rule
(e.g. ``"integer"``, ``"float"``, ``"logical"``). Each rule name maps to a handler that
knows how to check, parse, and serialise the corresponding Python type.
"""


def select_value_rule(value: Any, admitted: Collection[str]) -> str | None:
    """Choose the value-type rule to use when writing *value* into a new entry.

    The first admitted rule that accepts the value, in the order of the registry, which is
    the order of preference; the comment above it says why.

    Args:
        value (Any): The Python value to be written.
        admitted (Collection[str]): Names of the value-type rules the format has.

    Returns:
        str | None: The chosen rule name, or ``None`` if no admitted rule accepts *value*.
    """
    for rule_name, handler in VALUE_TYPE_HANDLER_REGISTRY.items():
        if rule_name in admitted and handler.type_check(value):
            return rule_name
    return None


def identical_values(first: Any, second: Any) -> bool:
    """Report whether two values are the same value, of the same type.

    Equality is not enough, in either direction. ``1``, ``1.0`` and ``True`` compare equal
    though each is a different value to write, and a NaN compares unequal even to itself,
    though writing it over itself changes nothing. Comparing the type and the ``repr``
    settles both.

    Args:
        first (Any): A value.
        second (Any): Another value.

    Returns:
        bool: True if the two are indistinguishable.
    """
    return type(first) is type(second) and repr(first) == repr(second)
