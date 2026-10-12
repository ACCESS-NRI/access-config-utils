# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""The shape of one written entry, as a sequence of fragments.

A ``Fragment`` is one piece of the text of an entry: fixed text, or a placeholder for the
key, a value, some whitespace, or the contents of a new block. An ``EntryTemplate`` is a
tuple of them in the order they are written, and is the currency of entry synthesis --
``entry_style.donor_template`` copies one from a neighbouring entry, ``compile_template``
makes one of the text a parser declares, and ``entry_render`` turns one into text.

This module holds the type itself and what can be done with a template without knowing
anything else: finding what sits either side of a position, growing a list template to
any length, and compiling the text a parser declares.

Fragment kinds deliberately echo the grammar's own vocabulary, but they are a separate
namespace: a kind describes a slot in a piece of text, not a rule. Rule names always come
from ``grammar_contract``, so a bare string literal here is a fragment kind, never a rule
name.

Examples:
    A template is a flat sequence of fragments, in the order they are written::

    >>> from access.config.entry_fragments import Fragment
    >>> template = (Fragment("key"), Fragment("ws"), Fragment("literal", "="),
    ...             Fragment("ws"), Fragment("value"), Fragment("literal", "\\n"))
    >>> [fragment.kind for fragment in template]
    ['key', 'ws', 'literal', 'ws', 'value', 'literal']
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from access.config.grammar_contract import VALUE_SLOT_COUNTS

FragmentKind = Literal["literal", "ws", "key", "value", "block"]


@dataclass(frozen=True)
class Fragment:
    """One piece of the text of an entry.

    The kind names deliberately echo the grammar's own vocabulary, but they are a separate
    namespace: a fragment kind describes a slot in a piece of text, not a rule. Rule names
    always come from ``grammar_contract``, so a bare string literal in this package is a
    fragment kind and never a rule name.

    Args:
        kind (FragmentKind): ``"literal"`` for fixed text, or a placeholder: ``"ws"`` for
            optional whitespace, ``"key"`` for the key name, ``"value"`` for a value,
            ``"block"`` for the contents of a new (and therefore empty) block.
        text (str): The text, for a ``"literal"`` fragment.
        level (int): Which key of a chain this is, for a ``"key"`` fragment: 0 for the
            outermost, as in the ``a`` of ``a%b = 1``, and 1 for the ``b``.
        notation (tuple[str, str] | None): The value-type rule and the token text an
            existing entry wrote here, for a ``"value"`` fragment of a template copied from
            that entry. A new value written with the same rule keeps its notation.
    """

    kind: FragmentKind
    text: str = ""
    level: int = 0
    notation: tuple[str, str] | None = None


EntryTemplate = tuple[Fragment, ...]
"""One way of writing an entry, as an ordered sequence of fragments."""


def neighbour_fragment(template: EntryTemplate, index: int, step: int) -> Fragment | None:
    """Return the nearest fragment on one side of *index* that is not a whitespace slot.

    Args:
        template (EntryTemplate): The template to inspect.
        index (int): Index to search out from, which may be one past the end.
        step (int): ``-1`` to look backwards, ``1`` to look forwards.

    Returns:
        Fragment | None: The neighbour, or ``None`` if there is none on that side.
    """
    position = index + step
    while 0 <= position < len(template):
        if template[position].kind != "ws":
            return template[position]
        position += step
    return None


def expand_value_slots(template: EntryTemplate, count: int) -> EntryTemplate:
    """Grow a two-value list template so that it holds *count* values.

    The fragments between the two value placeholders are the element separator, so repeating
    them along with the placeholder produces a list of any length using the grammar's own
    separator.

    Args:
        template (EntryTemplate): A template with exactly two value placeholders.
        count (int): Number of values wanted, at least two.

    Returns:
        EntryTemplate: The grown template.

    Examples:
        >>> template = (Fragment("key"), Fragment("ws"), Fragment("literal", "="),
        ...             Fragment("ws"), Fragment("value"), Fragment("literal", ","),
        ...             Fragment("ws"), Fragment("value"), Fragment("literal", "\\n"))
        >>> grown = expand_value_slots(template, 3)
        >>> [f.kind for f in grown]  # doctest: +NORMALIZE_WHITESPACE
        ['key', 'ws', 'literal', 'ws', 'value', 'literal', 'ws', 'value',
         'literal', 'ws', 'value', 'literal']
    """
    values = [index for index, fragment in enumerate(template) if fragment.kind == "value"]
    first, second = values[0], values[1]
    separator = template[first + 1 : second]
    repeated = (separator + (template[second],)) * (count - 1)
    return template[: first + 1] + repeated + template[second + 1 :]


_MARKER = re.compile(r"\{key\}|\{keys:([^}]*)\}|\{value\}")


def compile_template(text: str, category: str, depth: int, indent: str | None) -> EntryTemplate | None:
    """Compile the template text a parser declares, for an entry chaining *depth* keys.

    The text is ordinary text with three markers: ``{key}``, the key of a plain entry;
    ``{keys:SEP}``, every key of a chain parted by SEP, which the Fortran ``a%b`` writes as
    ``{keys:%}``; and ``{value}``. A list template has two ``{value}`` markers, because the
    text between them is the element separator, which longer lists repeat.

    The whitespace the line of the first key starts with is the indentation the template
    declares. *indent* replaces it, so that an entry is indented like the one above it.

    Args:
        text (str): The template text.
        category (str): The entry category it is for.
        depth (int): How many keys the entry chains together; 1 for a plain entry.
        indent (str | None): The indentation the file shows, or ``None`` to keep the
            template's own.

    Returns:
        EntryTemplate | None: The compiled template, or ``None`` if it writes one key and
            *depth* is not 1.

    Raises:
        ValueError: If the markers do not match what *category* requires.

    Examples:
        >>> template = compile_template("{key} = {value}\\n", "key_value", 1, None)
        >>> [(f.kind, f.text) for f in template]
        [('key', ''), ('literal', ' = '), ('value', ''), ('literal', '\\n')]

        A chain writes each of its keys, parted as the template says, after the indent.

        >>> template = compile_template("{keys:%}=\\n", "key_null", 2, "  ")
        >>> [(f.kind, f.text) for f in template]
        [('literal', '  '), ('key', ''), ('literal', '%'), ('key', ''), ('literal', '=\\n')]

        A list template needs two value markers, because the text between them is what
        identifies the element separator.

        >>> compile_template("{key} = {value}\\n", "key_list", 1, None)
        Traceback (most recent call last):
            ...
        ValueError: A key_list template needs exactly 2 '{value}' markers
    """
    expected = VALUE_SLOT_COUNTS[category]
    if text.count("{value}") != expected:
        raise ValueError(f"A {category} template needs exactly {expected} '{{value}}' markers")
    if "{key}" not in text and "{keys:" not in text:
        raise ValueError("An entry template needs a '{key}' or '{keys:...}' marker")

    fragments: list[Fragment] = []
    position = 0
    for found in _MARKER.finditer(text):
        marked = _marker_fragments(found, depth)
        if marked is None:
            return None
        fragments += [Fragment("literal", text[position : found.start()]), *marked]
        position = found.end()
    fragments.append(Fragment("literal", text[position:]))
    return _indented(tuple(f for f in fragments if f.kind != "literal" or f.text), indent)


def _marker_fragments(marker: re.Match[str], depth: int) -> list[Fragment] | None:
    """Return what a marker writes, or ``None`` if the marker cannot write *depth* keys."""
    if marker.group() == "{value}":
        return [Fragment("value")]
    if marker.group(1) is None:
        return [Fragment("key")] if depth == 1 else None
    fragments = []
    for level in range(depth):
        if level:
            fragments.append(Fragment("literal", marker.group(1)))
        fragments.append(Fragment("key", level=level))
    return fragments


def _indented(template: EntryTemplate, indent: str | None) -> EntryTemplate:
    """Return *template* with the whitespace its first key's line starts with replaced."""
    if indent is None:
        return template
    first = next(index for index, fragment in enumerate(template) if fragment.kind == "key")
    before = template[first - 1].text if first else ""
    line_start = before.rfind("\n") + 1
    text = before[:line_start] + indent + before[line_start:].lstrip(" \t")
    return (*template[: max(first - 1, 0)], *([Fragment("literal", text)] if text else []), *template[first:])
