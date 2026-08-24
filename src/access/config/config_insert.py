# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Adding an entry for a key a configuration does not have yet.

Editing an existing key means changing a token. Adding one means producing parse tree nodes
the parser never built, which is a different problem and gets a module of its own.

Nothing here builds a node by hand. ``entry_render`` synthesises the *text* of the entry
from the grammar, ``parse_entry_nodes`` hands that text back to Lark with the enclosing
rule as the start symbol, and Lark therefore produces the nodes -- so rule inlining, ``->``
aliases,
filtered-out anonymous terminals and ``%import`` terminal renaming all take care of
themselves.

Candidates are offered in order of how idiomatic they are, and **every one is validated in
full before the tree is touched**: it must parse, and reading it back must give exactly the
requested key and a value equal and identical in type to the one asked for. That is what
makes ranking a matter of style rather than of correctness, and it is what stops a value
being written in a form the grammar reads back differently -- ``Path("foo")`` would be
written as the bare word ``foo``, which a grammar with an ``identifier`` rule reads back as
a ``str``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from lark import Tree
from lark.exceptions import UnexpectedInput

from access.config.entry_generate import admitted_value_rules
from access.config.entry_render import iter_entry_snippets
from access.config.grammar_contract import KEY_BLOCK, KEY_LIST, KEY_NULL, KEY_VALUE
from access.config.grammar_values import UnsupportedEntryError
from access.config.tree_edits import parse_entry_nodes, splice_entry_nodes
from access.config.tree_navigation import entries_of, entry_insertion_index
from access.config.tree_reader import EntryRef, entry_value, read_entries

if TYPE_CHECKING:
    from access.config.entry_style import EntryStyle
    from access.config.grammar_compiled import ParseContext


def entry_category(value: Any) -> str:
    """Return the kind of entry needed to hold *value*.

    Args:
        value (Any): The value to be stored.

    Returns:
        str: One of the entry category names.
    """
    if isinstance(value, dict):
        return KEY_BLOCK
    if value is None:
        return KEY_NULL
    if isinstance(value, list):
        return KEY_LIST
    return KEY_VALUE


def _same_value(stored: Any, requested: Any) -> bool:
    """Report whether re-reading the written text gave back exactly the requested value.

    The type is compared as well as the value, because several value types share a Python
    type and a written value can be read back as the wrong one: ``Path("foo")`` writes as
    ``foo``, which a grammar with a bare-word value type reads back as a plain ``str``.

    Args:
        stored (Any): The value read back from the candidate text.
        requested (Any): The value the caller asked to store.

    Returns:
        bool: True if the two match in both type and value.
    """
    return type(stored) is type(requested) and bool(stored == requested)


def entry_matches(refs: Mapping[str, EntryRef], key: str, value: Any) -> bool:
    """Report whether a candidate entry, read back from its text, holds what was asked for.

    This is what makes candidate ranking a matter of style rather than of correctness: a
    candidate that parses but means something else fails here and the next one is tried.

    Args:
        refs (Mapping[str, EntryRef]): The entries read back from the candidate.
        key (str): The normalised key expected.
        value (Any): The value requested.

    Returns:
        bool: True if the candidate is an exact match.
    """
    if set(refs) != {key}:
        return False
    ref = refs[key]
    if ref.category != entry_category(value):
        return False
    if ref.category == KEY_BLOCK:
        # A new block starts empty; its contents are added afterwards.
        assert ref.block_node is not None
        return not entries_of(ref.block_node)
    if ref.category == KEY_NULL:
        return True
    stored = entry_value(ref)
    if ref.category == KEY_LIST:
        return len(stored) == len(value) and all(
            _same_value(item, wanted) for item, wanted in zip(stored, value, strict=True)
        )
    return _same_value(stored, value)


def insert_entry(container: Tree, ctx: ParseContext, key: str, raw_key: str, value: Any, style: EntryStyle) -> EntryRef:
    """Write a new entry into *container* and return where it landed.

    Args:
        container (Tree): The container rule node to add the entry to.
        ctx (ParseContext): The compiled grammar and the per-parser settings.
        key (str): The normalised key, used to check what the candidate read back as.
        raw_key (str): The key as the caller wrote it, used for the file text.
        value (Any): The value to store. A ``dict`` creates an empty block; its contents are
            added afterwards, by the caller, one key at a time.
        style (EntryStyle): Whitespace to copy from a neighbouring entry.

    Returns:
        EntryRef: Where the new entry lives in the parse tree.

    Raises:
        ValueError: If *raw_key* is not a usable key name, or *value* is a list too short
            for the format to express as one.
        UnsupportedEntryError: If no candidate produced the requested value.
    """
    if not isinstance(raw_key, str) or not raw_key.isidentifier():
        raise ValueError(f"Not a valid configuration key: {raw_key!r}")

    category = entry_category(value)
    if category == KEY_LIST and len(value) < 2:
        # Every supported format needs at least two values to be a list rather than a
        # scalar, and writing a scalar would put the wrong type in the dict.
        raise ValueError(f"A new list needs at least two elements: '{key}'")

    container_rule = str(container.data)
    index = entry_insertion_index(container)

    for snippet in iter_entry_snippets(
        ctx.info,
        container_rule,
        category,
        raw_key,
        value,
        style,
        ctx.entry_templates,
        ctx.value_rule_priority,
    ):
        # Read the candidate in a throwaway container, so one that parses but does not mean
        # the right thing leaves no trace. A candidate can fail to parse, or parse into a
        # shape the reader rejects; either way the next one is tried.
        try:
            nodes = parse_entry_nodes(ctx.lark, container_rule, snippet)
            refs = read_entries(Tree(container.data, list(nodes)), ctx)
        except (UnexpectedInput, ValueError, TypeError):
            continue
        if not entry_matches(refs, key, value):
            continue

        splice_entry_nodes(container, nodes, index)
        return refs[key]

    admitted = sorted(admitted_value_rules(ctx.info, container_rule, category))
    raise UnsupportedEntryError(
        f"This format cannot store {value!r} as '{raw_key}' here: there is no way to "
        f"write a {category} in '{container_rule}' with this value"
        + (f" (value types allowed here: {admitted})" if admitted else "")
    )
