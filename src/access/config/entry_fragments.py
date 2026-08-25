# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""The shape of one written entry, as a sequence of fragments.

A ``Fragment`` is one piece of the text of an entry: fixed text, or a placeholder for the
key, a value, some whitespace, or the contents of a new block. An ``EntryTemplate`` is a
tuple of them in the order they are written, and is the currency of entry synthesis --
``entry_generate`` derives templates from a grammar, ``entry_render`` turns one into text.

This module holds the type itself and the questions that can be asked of a template without
knowing anything else: how many fragments of a kind it has, what sits either side of a
position, and how idiomatic its shape is. It also compiles a template written by hand, for a
format whose conventional layout is not what the grammar happens to imply.

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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from access.config.grammar_contract import KEY_BLOCK, VALUE_SLOT_COUNTS

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
        options (frozenset[str]): Names of the value-type rules allowed here, for a
            ``"value"`` fragment.
    """

    kind: FragmentKind
    text: str = ""
    options: frozenset[str] = frozenset()


EntryTemplate = tuple[Fragment, ...]
"""One way of writing an entry, as an ordered sequence of fragments."""


# --- Template inspection ---
#
# Style-independent questions about a template, asked both while generating templates and
# while ranking them.


def count_fragments(template: EntryTemplate, kind: FragmentKind) -> int:
    """Return the number of fragments of the given kind in *template*.

    Args:
        template (EntryTemplate): The template to inspect.
        kind (FragmentKind): The fragment kind to count.

    Returns:
        int: The number of matching fragments.
    """
    return sum(1 for fragment in template if fragment.kind == kind)


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


def _block_contents_start_a_line(template: EntryTemplate) -> bool:
    """Report whether the contents of a new block would begin on their own line.

    A grammar may make the newline after a block header optional, so that a block can
    legally open and close on one line, so that an empty block is written without a body.
    Preferring the form whose body is on its own line is what leaves room for the keys about
    to be added to it.

    Args:
        template (EntryTemplate): The template to inspect.

    Returns:
        bool: True if every block placeholder is preceded by a newline, or there is none.
    """
    for index, fragment in enumerate(template):
        if fragment.kind != "block":
            continue
        before = neighbour_fragment(template, index, -1)
        if before is None or before.kind != "literal" or not before.text.endswith("\n"):
            return False
    return True


def shape_key(template: EntryTemplate) -> tuple[int, ...]:
    """Return the style-independent terms of the sort key; smaller is preferred.

    In order of precedence:

    1. The entry should be a complete line. Grammars that ignore whitespace have no newlines
       at all, so this is a preference rather than a filter.
    2. A new block's contents should start on their own line.
    3. Fewest newlines *other than* the one ending the entry, so no blank lines are added.
    4. Values parted by punctuation rather than by whitespace alone. A grammar may allow
       both -- a grammar may read ``1 2 3`` as well as ``1, 2, 3`` -- and the run of
       whitespace is the weaker of the two: a stricter reader of the same format may
       not split it. This outranks the next term, which would otherwise choose the
       whitespace form for having one character less of fixed text.
    5. Least fixed text, which rejects an optional trailing separator such as ``KEY = 1,``.
    6. Most whitespace placeholders, since unwanted ones render empty anyway.
    7. Fewest repetitions of the key.

    Used both to rank finished templates and to decide which derivations survive
    ``_MAX_ALTERNATIVES``, so that the generation cap keeps what ranking would prefer.

    Args:
        template (EntryTemplate): The template to score.

    Returns:
        tuple[int, ...]: The sort key.
    """
    literals = [fragment.text for fragment in template if fragment.kind == "literal"]
    # The *last* fragment has to be the newline. Testing the last literal instead would
    # accept a template whose only newline separates two values, as a continued list does.
    last = neighbour_fragment(template, len(template), -1)
    ends_line = last is not None and last.kind == "literal" and last.text.endswith("\n")
    return (
        0 if ends_line else 1,
        0 if _block_contents_start_a_line(template) else 1,
        sum(text.count("\n") for text in literals) - (1 if ends_line else 0),
        0 if _values_parted_by_literal(template) else 1,
        sum(len(text.strip()) for text in literals),
        -count_fragments(template, "ws"),
        count_fragments(template, "key"),
    )


def _values_parted_by_literal(template: EntryTemplate) -> bool:
    """Report whether every pair of neighbouring values has punctuation between them.

    False for a template that leans on a whitespace placeholder alone to tell one value
    from the next. A template with fewer than two values is trivially True, so this term
    moves nothing for the categories that hold one value or none.

    Args:
        template (EntryTemplate): The template to inspect.

    Returns:
        bool: True if no two values are parted by whitespace alone.
    """
    positions = [index for index, fragment in enumerate(template) if fragment.kind == "value"]
    return all(
        any(template[between].kind == "literal" for between in range(first + 1, second))
        for first, second in zip(positions, positions[1:], strict=False)
    )


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


_MARKERS: Mapping[str, Fragment] = {
    "{key}": Fragment("key"),
    "{value}": Fragment("value"),
    "{block}": Fragment("block"),
    "{ws}": Fragment("ws"),
}


def compile_template(text: str, category: str, options: frozenset[str]) -> EntryTemplate:
    """Compile a hand-written entry template declared by a parser.

    The template is ordinary text with ``{key}``, ``{value}``, ``{block}`` and ``{ws}``
    markers. ``{ws}`` marks whitespace that should follow a neighbouring entry's style;
    literal spaces are left alone.

    Args:
        text (str): The template text.
        category (str): The entry category it is for.
        options (frozenset[str]): Value-type rules admitted in this position.

    Returns:
        EntryTemplate: The compiled template.

    Raises:
        ValueError: If the markers do not match what *category* requires.

    Examples:
        >>> template = compile_template("{key} = {value}\\n", "key_value",
        ...                             frozenset({"integer"}))
        >>> [(f.kind, f.text) for f in template]
        [('key', ''), ('literal', ' = '), ('value', ''), ('literal', '\\n')]

        A list template needs two value markers, because the text between them is what
        identifies the element separator.

        >>> compile_template("{key} = {value}\\n", "key_list", frozenset({"integer"}))
        Traceback (most recent call last):
            ...
        ValueError: A key_list template needs exactly 2 '{value}' markers
    """
    fragments: list[Fragment] = []
    rest = text
    while rest:
        found = [(rest.index(name), name) for name in _MARKERS if name in rest]
        if not found:
            fragments.append(Fragment("literal", rest))
            break
        position, name = min(found)
        if position:
            fragments.append(Fragment("literal", rest[:position]))
        marker = _MARKERS[name]
        fragments.append(Fragment("value", options=options) if marker.kind == "value" else marker)
        rest = rest[position + len(name) :]

    template = tuple(fragments)
    expected = VALUE_SLOT_COUNTS[category]
    if count_fragments(template, "value") != expected:
        raise ValueError(f"A {category} template needs exactly {expected} '{{value}}' markers")
    if count_fragments(template, "block") != (1 if category == KEY_BLOCK else 0):
        raise ValueError(f"A {category} template has the wrong number of '{{block}}' markers")
    if count_fragments(template, "key") < 1:
        raise ValueError("An entry template needs at least one '{key}' marker")
    return template
