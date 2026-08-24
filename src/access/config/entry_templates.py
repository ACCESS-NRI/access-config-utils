# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Every way a grammar can express one configuration entry.

This module answers "what shapes may a new entry take?" by walking the compiled grammar and
collecting each derivation of a single entry as a tuple of ``Fragment``\\ s -- fixed text
interleaved with placeholders for the key, the values, whitespace and the contents of a new
block. It is the first half of entry synthesis; ``entry_synthesis`` takes these templates
and decides which one to use and how to space it out.

Everything here depends only on the grammar and on a template's own shape, never on the file
being edited. That is the seam between the two modules: a function that needs an
``EntryStyle`` or a parse tree belongs in ``entry_synthesis``, and a function that does not
belongs here.

Templates can also be written by hand, when a format's conventional layout is not what the
grammar-derived text happens to look like -- see ``compile_template``.

Examples:
    A template is a flat sequence of fragments, in the order they are written::

    >>> from access.config.entry_templates import Fragment
    >>> template = (Fragment("key"), Fragment("ws"), Fragment("literal", "="),
    ...             Fragment("ws"), Fragment("value"), Fragment("literal", "\\n"))
    >>> [fragment.kind for fragment in template]
    ['key', 'ws', 'literal', 'ws', 'value', 'literal']
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from weakref import WeakKeyDictionary

from lark.grammar import Symbol

from access.config.grammar_contract import (
    BLOCK_RULE,
    ENTRY_CATEGORIES,
    KEY_BLOCK,
    KEY_RULE,
    VALUE_SLOT_COUNTS,
    WS_RULE,
)
from access.config.grammar_info import GrammarInfo

_MAX_ALTERNATIVES = 64
"""Cap on the derivations kept per rule, to bound generation on a large grammar.

The derivations are sorted by ``shape_key`` before the cap is applied, so what survives is
what ranking would have chosen anyway. Truncating in generation order instead would let one
prolific alternative decide the outcome before ranking sees the others.
"""

_MAX_COMBINATIONS = 4096
"""Cap on the combinations examined per rule alternative, including those pruned."""

_TEMPLATE_CACHE: WeakKeyDictionary[GrammarInfo, dict[tuple[str, str], tuple[EntryTemplate, ...]]] = WeakKeyDictionary()
"""Generated templates, keyed by the grammar they came from and then by container/category.

Held against the ``GrammarInfo`` rather than on it, so that generation stays a concern
of this module while the memo still dies with the compiled grammar it describes.
``GrammarInfo`` compares by identity, which is exactly the key wanted here.
"""


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
    legally open and close on one line (a Fortran namelist does: ``&NAME/`` is a valid if
    unusual empty group). Preferring the form whose body is on its own line is what leaves
    room for the keys about to be added to it.

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
    4. Least fixed text, which rejects an optional trailing separator such as ``KEY = 1,``.
    5. Most whitespace placeholders, since unwanted ones render empty anyway.
    6. Fewest repetitions of the key.

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
        sum(len(text.strip()) for text in literals),
        -count_fragments(template, "ws"),
        count_fragments(template, "key"),
    )


# --- Template generation ---

# A partial derivation: the fragments produced so far, and the entry categories within them.
_Derivation = tuple[EntryTemplate, tuple[str, ...]]


def _expand_symbol(info: GrammarInfo, symbol: Symbol, stack: tuple[str, ...], category: str) -> list[_Derivation]:
    """Return the derivations for a single grammar symbol."""
    name = str(symbol.name)
    if symbol.is_term:
        text = info.sample_terminal(name)
        # A terminal that cannot be reproduced prunes the whole derivation.
        return [] if text is None else [((Fragment("literal", text),), ())]
    if name == KEY_RULE:
        return [((Fragment("key"),), ())]
    if name == BLOCK_RULE:
        # A block being created is empty; its contents are added afterwards, key by key.
        return [((Fragment("block"),), ())]
    options = info.value_rules(name)
    if options:
        return [((Fragment("value", options=options),), ())]
    if name == WS_RULE:
        return [((Fragment("ws"),), ())]
    return _expand_rule(info, name, stack, category)


def _expand_rule(info: GrammarInfo, name: str, stack: tuple[str, ...], category: str) -> list[_Derivation]:
    """Return the derivations for a rule that hold at most one entry, of *category*.

    A rule already on the stack is not re-entered. Lark turns every EBNF repetition into a
    left-recursive helper rule, so cutting recursion here keeps the derivations finite and
    gives exactly the shapes wanted: ``ws*`` contributes at most one whitespace placeholder,
    a repeated value contributes exactly two value placeholders, and a container contributes
    exactly one entry.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        name (str): Name of the rule to expand.
        stack (tuple[str, ...]): Rules already being expanded, to cut recursion.
        category (str): The entry category being generated, used to prune early.

    Returns:
        list[_Derivation]: The derivations, simplest first, capped at ``_MAX_ALTERNATIVES``.
    """
    if name in stack:
        return []
    inner = (*stack, name)
    derivations: list[_Derivation] = []
    for rule in info.rules_by_origin.get(name, ()):
        seen_name = str(rule.alias) if rule.alias else name
        seen = (seen_name,) if seen_name in ENTRY_CATEGORIES else ()
        if _has_excess_entries(seen, category):
            continue
        for fragments, categories in _expand_expansion(info, rule.expansion, inner, category):
            derivations.append((fragments, seen + categories))
    # Sorted by the same terms ranking uses, so the derivations surviving the cap below are
    # the ones ranking would have chosen anyway.
    derivations.sort(key=lambda derivation: shape_key(derivation[0]))
    return derivations[:_MAX_ALTERNATIVES]


def _has_excess_entries(categories: tuple[str, ...], category: str) -> bool:
    """Report whether a partial derivation has already gone wrong, entry-wise.

    Pruning as soon as a derivation holds a foreign entry, or a second one of the wanted
    category, is what keeps generation small. Without it a rule that allows several
    assignments on one line (as a Fortran namelist does) swamps the derivation limit with
    forms that would all be discarded later, and the single-assignment form is never
    reached.

    Args:
        categories (tuple[str, ...]): Entry categories seen in the derivation so far.
        category (str): The entry category being generated.

    Returns:
        bool: True if the derivation can no longer produce exactly one *category* entry.
    """
    return any(item != category for item in categories) or categories.count(category) > 1


def _expand_expansion(
    info: GrammarInfo, expansion: Sequence[Symbol], stack: tuple[str, ...], category: str
) -> list[_Derivation]:
    """Return the derivations for one alternative, as a product over its symbols."""
    per_symbol = []
    for symbol in expansion:
        derivations = _expand_symbol(info, symbol, stack, category)
        if not derivations:
            return []
        per_symbol.append(derivations)

    combined: list[_Derivation] = []
    for combination in itertools.islice(itertools.product(*per_symbol), _MAX_COMBINATIONS):
        fragments: EntryTemplate = ()
        categories: tuple[str, ...] = ()
        for part_fragments, part_categories in combination:
            fragments += part_fragments
            categories += part_categories
        if _has_excess_entries(categories, category):
            continue
        combined.append((fragments, categories))
    return combined


def _is_usable(template: EntryTemplate, categories: tuple[str, ...], category: str) -> bool:
    """Report whether a derivation writes exactly one well-formed entry of *category*.

    Anything else about the derivation is left to the caller's validation, which reads the
    candidate back and compares it with what was asked for.

    Args:
        template (EntryTemplate): The fragments of the derivation.
        categories (tuple[str, ...]): Entry categories within it.
        category (str): The entry category wanted.

    Returns:
        bool: True if the derivation is usable.
    """
    if Counter(categories) != Counter({category: 1}):
        return False
    if count_fragments(template, "value") != VALUE_SLOT_COUNTS[category]:
        return False
    return count_fragments(template, "key") >= 1


def grammar_templates(info: GrammarInfo, container: str, category: str) -> tuple[EntryTemplate, ...]:
    """Return every way of writing one *category* entry inside *container*, unranked.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        container (str): Name of the rule whose children the entry will become.
        category (str): One of ``ENTRY_CATEGORIES``.

    Returns:
        tuple[EntryTemplate, ...]: The usable templates, empty if the grammar cannot express
            such an entry in that container.

    Examples:
        >>> from access.config.grammar_compiled import compile_grammar
        >>> from access.config import MOM6InputParser
        >>> info = compile_grammar(MOM6InputParser().grammar).info
        >>> shapes = [tuple(f.kind for f in t)
        ...           for t in grammar_templates(info, "start", "key_value")]
        >>> ('key', 'ws', 'literal', 'ws', 'value', 'ws', 'literal') in shapes
        True

        A combination the grammar cannot express yields nothing; MOM6 has no way to write
        a key with no value at all.

        >>> grammar_templates(info, "start", "key_null")
        ()
    """
    memo = _TEMPLATE_CACHE.setdefault(info, {})
    cached = memo.get((container, category))
    if cached is None:
        seen: dict[EntryTemplate, None] = {}
        for template, categories in _expand_rule(info, container, (), category):
            if _is_usable(template, categories, category):
                seen[template] = None
        cached = tuple(seen)
        memo[(container, category)] = cached
    return cached


def admitted_value_rules(info: GrammarInfo, container: str, category: str) -> frozenset[str]:
    """Return the value-type rules that can appear in a new *category* entry in *container*.

    The union over every template, so a rule admitted by only one rendering still counts.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        container (str): Container rule name.
        category (str): Entry category.

    Returns:
        frozenset[str]: Union of the admitted value-type rules over all templates.
    """
    admitted: set[str] = set()
    for template in grammar_templates(info, container, category):
        for fragment in template:
            admitted |= fragment.options
    return frozenset(admitted)


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


# --- Hand-written templates ---

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
