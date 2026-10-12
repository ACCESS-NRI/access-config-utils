# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Writing a list of a different length into the entry that holds it.

Writing an element of a list changes one token, but a list that grows or shrinks needs
values, and the text between them, that the file does not have. As with adding a key, the
new text is made rather than the nodes: the line holding the entry is written out with a
marker in place of each value, ``list_layout`` lays the new list out around them, and each
candidate is parsed back with the container's rule as the start symbol.

A candidate is used only if the entry it parses into reads back as exactly the list asked
for, and if putting the new entry's children into the old line reproduces the candidate's
text. The second check is what keeps the rest of the line as it was: a trailing separator,
say, may parse as part of the line rather than of the entry, and then the entry alone does
not carry it.

The entry keeps its node, so the references to it, and to any other entry sharing its
line, stay good.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from lark import Token, Tree
from lark.exceptions import UnexpectedInput

from access.config.config_insert import entry_matches
from access.config.grammar_contract import REPEAT_RULE
from access.config.grammar_values import (
    VALUE_TYPE_HANDLER_REGISTRY,
    UnsupportedEntryError,
    select_value_rule,
)
from access.config.list_layout import VALUE_MARKER, ListText, layout_candidates
from access.config.tree_edits import blank_comments, parse_entry_nodes, replace_entry_contents, rewritten_text
from access.config.tree_navigation import is_value_node, locate_entry, node_at
from access.config.tree_reader import read_entries

if TYPE_CHECKING:
    from access.config.grammar_compiled import ParseContext
    from access.config.tree_reader import EntryRef


def resize_entry(
    ref: EntryRef, ctx: ParseContext, key: str, values: Sequence[Any], origins: Sequence[int | None]
) -> EntryRef:
    """Write a list of a different length into the entry *ref* names, where it stands.

    *ref* is a ``key_list`` entry, and the reference returned has the value nodes of the
    rewritten entry. *key* is the normalised key a candidate has to read back as. *origins*
    gives, for each value, the position of the old element it stands in for, in increasing
    order, or ``None`` for a new one.

    Raises:
        ValueError: If fewer than two values are asked for, or the entry's values are not
            written one per node in a single entry.
        TypeError: If a value standing in for an old element is of another type.
        UnsupportedEntryError: If a new value has no type the format can write here, or no
            layout of the list reads back as the values asked for.
    """
    if len(values) < 2:
        raise ValueError(f"A list needs at least two elements: '{key}'")
    entry = _resizable_entry(ref, key)
    container, line, path = locate_entry(entry, ctx.lark.options.start)
    text = ListText.from_marked(
        _marked_text(ctx, container, line, path, blank=False),
        _marked_text(ctx, container, line, path, blank=True),
        len(ref.value_nodes),
    )
    value_texts = _value_texts(ref, ctx, key, values, origins)
    for candidate in layout_candidates(text, value_texts, origins):
        fresh = _read_candidate(ctx, container, line, path, candidate, key, values)
        if fresh is not None:
            replace_entry_contents(entry, fresh)
            return replace(ref, value_nodes=tuple(child for child in entry.children if is_value_node(child)))
    raise UnsupportedEntryError(f"No layout of '{key}' with {len(values)} elements reads back as the list requested")


def _resizable_entry(ref: EntryRef, key: str) -> Tree:
    """Return the entry node writing *ref*'s values one per node, or raise ValueError."""
    reason = None
    if len(ref.entry_nodes) != 1:
        reason = "the file assigns it more than once"
    elif ref.values is not None:
        reason = "its values are placed by array qualifiers"
    elif any(node is None for node in ref.value_nodes):
        reason = "it leaves positions unset"
    elif any(node.data == REPEAT_RULE for node in ref.value_nodes if node is not None):
        reason = "it is written with a repeat count"
    if reason is not None:
        raise ValueError(f"Cannot change the length of '{key}': {reason}; delete it and assign it again")
    return ref.entry_nodes[0]


def _marked_text(ctx: ParseContext, container: Tree, line: Tree, path: Sequence[int], *, blank: bool) -> str:
    """Write out a copy of *line* with a marker in place of each value of its entry.

    With *blank*, every comment is written as nothing, so that the text between the values
    can be copied without copying someone else's comment.
    """
    marked = copy.deepcopy(line)
    entry = node_at(marked, path)
    for index, node in enumerate(child for child in entry.children if is_value_node(child)):
        token = node.children[0]
        assert isinstance(token, Token)
        node.children[0] = token.update(value=VALUE_MARKER.format(index))
    if blank:
        blank_comments(marked)
    return ctx.reconstructor.reconstruct(Tree(container.data, [marked]))


def _value_texts(
    ref: EntryRef, ctx: ParseContext, key: str, values: Sequence[Any], origins: Sequence[int | None]
) -> list[str]:
    """Return the token text to write each new value with.

    A value that stays keeps its token if unchanged, else is written in its old notation.
    A new one takes the notation of its nearest old neighbour, else the format's own.
    """
    texts = []
    for index, (value, origin) in enumerate(zip(values, origins, strict=True)):
        if origin is not None:
            node = ref.value_nodes[origin]
            assert node is not None
            texts.append(rewritten_text(node, value))
            continue
        # Copy a neighbour's notation, so a list of quoted strings gains quoted strings.
        neighbour = _nearest_old(origins, index)
        if neighbour is not None:
            node = ref.value_nodes[neighbour]
            assert node is not None
            handler = VALUE_TYPE_HANDLER_REGISTRY[str(node.data)]
            if handler.type_check(value):
                texts.append(handler.to_token(value, str(node.children[0])))
                continue
        rule = select_value_rule(value, ctx.info.value_rules)
        if rule is None:
            raise UnsupportedEntryError(f"This format cannot store {value!r} in the list '{key}'")
        texts.append(VALUE_TYPE_HANDLER_REGISTRY[rule].to_new_token(value))
    return texts


def _nearest_old(origins: Sequence[int | None], index: int) -> int | None:
    """Return the old position of the nearest value that stays, searching backwards first.

    The result is ``None`` when no value stays at all.
    """
    before = (origins[at] for at in range(index - 1, -1, -1))
    after = (origins[at] for at in range(index + 1, len(origins)))
    return next((origin for origin in (*before, *after) if origin is not None), None)


def _read_candidate(
    ctx: ParseContext,
    container: Tree,
    line: Tree,
    path: Sequence[int],
    candidate: str,
    key: str,
    values: Sequence[Any],
) -> Tree | None:
    """Return the entry a candidate text parses into, if it is the one asked for.

    The result is ``None`` if the candidate does not parse, reads back as something else, or
    changes the line outside the entry.
    """
    try:
        nodes = parse_entry_nodes(ctx.lark, str(container.data), candidate)
        if len(nodes) != 1 or not isinstance(nodes[0], Tree):
            return None
        fresh = node_at(nodes[0], path)
        refs = read_entries(Tree(container.data, [fresh]), ctx)
    except (UnexpectedInput, IndexError, ValueError, TypeError):
        return None
    if not entry_matches(refs, (key,), list(values), ctx):
        return None
    probe = copy.deepcopy(line)
    target = node_at(probe, path)
    target.children = copy.deepcopy(fresh.children)
    target.data = fresh.data
    if ctx.reconstructor.reconstruct(Tree(container.data, [probe])) != candidate:
        return None
    return fresh
