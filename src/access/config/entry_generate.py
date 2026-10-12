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
from collections.abc import Sequence
from weakref import WeakKeyDictionary

from lark.grammar import Symbol

from access.config.entry_fragments import EntryTemplate, Fragment, count_fragments, shape_key
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

_TEMPLATE_CACHE: WeakKeyDictionary[GrammarInfo, dict[tuple[str, str, int], tuple[EntryTemplate, ...]]] = (
    WeakKeyDictionary()
)
"""Generated templates, keyed by the grammar they came from, then by container, category and
depth.

Held against the ``GrammarInfo`` rather than on it, so that generation stays a concern
of this module while the memo still dies with the compiled grammar it describes.
``GrammarInfo`` compares by identity, which is exactly the key wanted here.
"""


# --- Template generation ---
#
# A derivation is followed through as many entries as the chain it is for holds, and no
# more: one for a plain entry, two for a component such as the Fortran ``a%b = 1``, which is
# an entry for ``a`` holding one for ``b``. The *level* passed down is the position in that
# chain of the innermost entry the derivation is inside, -1 outside them all, and every
# entry is recorded with the level it sits at, so that a derivation going astray -- an
# entry beside another, or one too deep -- is cut as soon as it does.

# A partial derivation: the fragments produced so far, and the level and category of each
# entry within them.
_Derivation = tuple[EntryTemplate, tuple[tuple[int, str], ...]]


def _expand_symbol(
    info: GrammarInfo, symbol: Symbol, stack: tuple[tuple[str, int], ...], category: str, depth: int, level: int
) -> list[_Derivation]:
    """Return the derivations for a single grammar symbol."""
    name = str(symbol.name)
    if symbol.is_term:
        text = info.sample_terminal(name)
        # A terminal that cannot be reproduced prunes the whole derivation.
        return [] if text is None else [((Fragment("literal", text),), ())]
    if name == KEY_RULE:
        # The key of the innermost entry the derivation is inside.
        return [((Fragment("key", level=level),), ())]
    if name == BLOCK_RULE:
        # A block being created is empty; its contents are added afterwards, key by key.
        return [((Fragment("block"),), ())]
    options = info.value_rules(name)
    if options:
        return [((Fragment("value", options=options),), ())]
    if name == WS_RULE:
        return [((Fragment("ws"),), ())]
    return _expand_rule(info, name, stack, category, depth, level)


def _expand_rule(
    info: GrammarInfo, name: str, stack: tuple[tuple[str, int], ...], category: str, depth: int, level: int
) -> list[_Derivation]:
    """Return the derivations for a rule that hold at most the chain of entries wanted.

    A rule already on the stack at the same level is not re-entered. Lark turns every EBNF
    repetition into a left-recursive helper rule, so cutting recursion here keeps the
    derivations finite and gives exactly the shapes wanted: ``ws*`` contributes at most one
    whitespace placeholder, a repeated value contributes exactly two value placeholders, and
    a container contributes exactly one entry. A rule may be entered again one entry deeper,
    which is how a chain is written: the Fortran ``a%b%c`` is the rule for ``a%b`` inside
    itself.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        name (str): Name of the rule to expand.
        stack (tuple[tuple[str, int], ...]): Rules already being expanded, each with the
            level it is being expanded at, to cut recursion.
        category (str): The category of the innermost entry of the chain.
        depth (int): How many entries the chain holds.
        level (int): The position in the chain of the innermost entry the rule is inside,
            -1 outside them all.

    Returns:
        list[_Derivation]: The derivations, simplest first, capped at ``_MAX_ALTERNATIVES``.
    """
    if (name, level) in stack:
        return []
    inner = (*stack, (name, level))
    derivations: list[_Derivation] = []
    for rule in info.rules_by_origin.get(name, ()):
        seen_name = str(rule.alias) if rule.alias else name
        entered = level + 1 if seen_name in ENTRY_CATEGORIES else level
        seen = ((entered, seen_name),) if entered != level else ()
        if _has_excess_entries(seen, category, depth):
            continue
        for fragments, entries in _expand_expansion(info, rule.expansion, inner, category, depth, entered):
            derivations.append((fragments, seen + entries))
    # Sorted by the same terms ranking uses, so the derivations surviving the cap below are
    # the ones ranking would have chosen anyway.
    derivations.sort(key=lambda derivation: shape_key(derivation[0]))
    return derivations[:_MAX_ALTERNATIVES]


def _has_excess_entries(entries: tuple[tuple[int, str], ...], category: str, depth: int) -> bool:
    """Report whether a partial derivation has already gone wrong, entry-wise.

    Pruning as soon as a derivation holds an entry the chain has no place for -- beside
    another at the same level, deeper than the chain goes, or of the wrong category -- is
    what keeps generation small. Without it a rule that allows several assignments on one
    line swamps the derivation limit with forms that would all be discarded later, and the
    single-assignment form is never reached.

    Args:
        entries (tuple[tuple[int, str], ...]): The level and category of each entry in the
            derivation so far.
        category (str): The category of the innermost entry of the chain.
        depth (int): How many entries the chain holds.

    Returns:
        bool: True if the derivation can no longer produce exactly the chain wanted.
    """
    levels = [level for level, _ in entries]
    if len(set(levels)) != len(levels):
        return True
    # Every entry but the innermost holds the next, so is a block.
    return any(level >= depth or seen != (category if level == depth - 1 else KEY_BLOCK) for level, seen in entries)


def _expand_expansion(
    info: GrammarInfo,
    expansion: Sequence[Symbol],
    stack: tuple[tuple[str, int], ...],
    category: str,
    depth: int,
    level: int,
) -> list[_Derivation]:
    """Return the derivations for one alternative, as a product over its symbols."""
    per_symbol = []
    for symbol in expansion:
        derivations = _expand_symbol(info, symbol, stack, category, depth, level)
        if not derivations:
            return []
        per_symbol.append(derivations)

    combined: list[_Derivation] = []
    for combination in itertools.islice(itertools.product(*per_symbol), _MAX_COMBINATIONS):
        fragments: EntryTemplate = ()
        entries: tuple[tuple[int, str], ...] = ()
        for part_fragments, part_entries in combination:
            fragments += part_fragments
            entries += part_entries
        if _has_excess_entries(entries, category, depth):
            continue
        combined.append((fragments, entries))
    return combined


def _is_usable(template: EntryTemplate, entries: tuple[tuple[int, str], ...], category: str, depth: int) -> bool:
    """Report whether a derivation writes exactly the chain of entries wanted.

    That is one entry at each level, all of them blocks but the innermost, which is of
    *category*; the values that category holds; and a key for every entry. Anything else
    about the derivation is left to the caller's validation, which reads the candidate back
    and compares it with what was asked for.

    Args:
        template (EntryTemplate): The fragments of the derivation.
        entries (tuple[tuple[int, str], ...]): The level and category of each entry in it.
        category (str): The category of the innermost entry.
        depth (int): How many entries the chain holds.

    Returns:
        bool: True if the derivation is usable.
    """
    if sorted(entries) != [*((level, KEY_BLOCK) for level in range(depth - 1)), (depth - 1, category)]:
        return False
    if count_fragments(template, "value") != VALUE_SLOT_COUNTS[category]:
        return False
    return {fragment.level for fragment in template if fragment.kind == "key"} == set(range(depth))


def grammar_templates(info: GrammarInfo, container: str, category: str, depth: int = 1) -> tuple[EntryTemplate, ...]:
    """Return every way of writing one *category* entry inside *container*, unranked.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        container (str): Name of the rule whose children the entry will become.
        category (str): One of ``ENTRY_CATEGORIES``.
        depth (int): How many keys the entry chains together: 1 for a plain entry, 2 for
            the Fortran ``a%b = 1``. All but the innermost name blocks, written one
            component per line rather than in a ``block`` of their own.

    Returns:
        tuple[EntryTemplate, ...]: The usable templates, empty if the grammar cannot express
            such an entry in that container.

    A container and category the grammar cannot express together yields nothing at all,
    which is how a caller learns the format has no syntax for such an entry.
    """
    memo = _TEMPLATE_CACHE.setdefault(info, {})
    cached = memo.get((container, category, depth))
    if cached is None:
        seen: dict[EntryTemplate, None] = {}
        for template, entries in _expand_rule(info, container, (), category, depth, -1):
            if _is_usable(template, entries, category, depth):
                seen[template] = None
        cached = tuple(seen)
        memo[(container, category, depth)] = cached
    return cached


def admitted_value_rules(info: GrammarInfo, container: str, category: str, depth: int = 1) -> frozenset[str]:
    """Return the value-type rules that can appear in a new *category* entry in *container*.

    The union over every template, so a rule admitted by only one rendering still counts.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        container (str): Container rule name.
        category (str): Entry category.
        depth (int): How many keys the entry chains together; see ``grammar_templates``.

    Returns:
        frozenset[str]: Union of the admitted value-type rules over all templates.
    """
    admitted: set[str] = set()
    for template in grammar_templates(info, container, category, depth):
        for fragment in template:
            admitted |= fragment.options
    return frozenset(admitted)
