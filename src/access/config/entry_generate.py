# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Every way a grammar can express one configuration entry.

This module answers "what shapes may a new entry take?" by walking the compiled grammar and
collecting each derivation of a single entry as an ``EntryTemplate``. It is the first half
of entry synthesis; ``entry_render`` takes these templates and decides which one to use,
and how to space it out.

Everything here depends only on the grammar and on a template's own shape, never on the file
being edited. That is the seam between the two halves: a function that needs an
``EntryStyle`` or a parse tree belongs on the other side of it.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence
from weakref import WeakKeyDictionary

from lark.grammar import Symbol

from access.config.entry_fragments import EntryTemplate, Fragment, count_fragments, shape_key
from access.config.grammar_contract import (
    BLOCK_RULE,
    ENTRY_CATEGORIES,
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
    assignments on one line swamps the derivation limit with
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

    A container and category the grammar cannot express together yields nothing at all,
    which is how a caller learns the format has no syntax for such an entry.
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
