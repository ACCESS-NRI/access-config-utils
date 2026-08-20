# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Changing a Lark parse tree without losing the ability to write it back out.

Round-trip fidelity is the constraint every function here works under: the tree that comes
out has to be one the grammar could have produced, or the reconstructor refuses to write
it. That rules out building nodes by hand, which is why a new entry is *parsed* into
existence by ``parse_entry_nodes`` rather than assembled, and it is why deleting an entry
may need ``merge_adjacent_repetitions`` to repair the shape it leaves behind.

Reading a tree is ``tree_navigation``; this module only changes one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from lark import Token, Tree

from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY
from access.config.tree_navigation import AddParent, contains_entry

if TYPE_CHECKING:
    from lark import Lark

    from access.config.grammar_info import GrammarInfo


def update_node_value(rule_node: Tree, value: Any) -> None:
    """Updates the value stored in a value-type rule node.

    The rule node must be a ``Tree`` whose ``.data`` is a key in
    ``VALUE_TYPE_HANDLER_REGISTRY`` (e.g. ``"integer"``, ``"float"``). The handler is
    looked up from that registry.

    Args:
        rule_node (Tree): Value-type rule node to update.
        value (Any): New value.

    Raises:
        TypeError: Raises an exception if the new and old value types do not match.
    """
    handler = VALUE_TYPE_HANDLER_REGISTRY.get(str(rule_node.data))
    if handler is None or not handler.type_check(value):
        raise TypeError("Trying to change value type")
    # The Token storing the value is always the first child of a value-type rule node.
    token = rule_node.children[0]
    assert isinstance(token, Token)
    transformed_value = handler.to_token(value, str(token))
    rule_node.children[0] = token.update(value=transformed_value)


def parse_entry_nodes(lark: Lark, container: str, snippet: str) -> list[Tree | Token]:
    """Parse the text of a new entry into nodes ready to be spliced into a container.

    Parsing the synthesised text with the container rule as the start symbol is what makes
    this safe: the nodes are the ones Lark would have produced for a file containing the
    entry, so there is no need to replicate Lark's rule inlining, aliasing or terminal
    renaming. Text that does not fit the grammar raises here, rather than producing a tree
    that cannot be written back out.

    Args:
        lark (Lark): The compiled parser, which must have *container* among its start
            symbols.
        container (str): Name of the container rule.
        snippet (str): The entry text.

    Returns:
        list[Tree | Token]: The nodes to splice, with ``.parent`` set throughout their
            interiors.

    Raises:
        UnexpectedInput: If *snippet* is not valid in *container*.
    """
    parsed = lark.parse(snippet, start=container)
    children = parsed.children if str(parsed.data) == container else [parsed]

    # Drop nodes that matched no text. A rule able to derive the empty string produces one
    # of these -- Fortran's rule for the text between namelists is one -- and splicing it
    # would put a second empty node beside the container's own, which the grammar cannot
    # derive and the reconstructor therefore refuses to write out. Only the top level is
    # filtered: a block being created is legitimately an empty node, and it is nested rather
    # than top-level.
    nodes = [child for child in children if not (isinstance(child, Tree) and not child.children)]

    for node in nodes:
        if isinstance(node, Tree):
            AddParent().visit(node)
    return nodes


def splice_entry_nodes(container: Tree, nodes: Sequence[Tree | Token], index: int) -> None:
    """Insert freshly parsed entry nodes into a container.

    Kept separate from ``parse_entry_nodes`` so a candidate entry can be validated in
    full before the tree is touched: everything up to this call is reversible, and this
    call is not.

    Args:
        container (Tree): The container rule node to insert into.
        nodes (Sequence[Tree | Token]): Nodes from ``parse_entry_nodes``.
        index (int): Position from ``entry_insertion_index``.
    """
    for node in nodes:
        if isinstance(node, Tree):
            node.parent = container  # type: ignore[attr-defined]
    container.children[index:index] = nodes


def remove_entry_node(entry_node: Tree, info: GrammarInfo) -> None:
    """Unlink an entry from the tree, repairing the shape its removal leaves behind.

    Two repairs, because a grammar can be underivable in two ways once an entry goes. A
    wrapper that *required* the entry cannot derive what is left, so it goes too
    (``prune_empty_wrappers``); and text either side of the entry may now be adjacent where
    the grammar wants it to alternate, so it is rejoined (``merge_adjacent_repetitions``).

    Args:
        entry_node (Tree): The entry rule node to remove.
        info (GrammarInfo): Views over the compiled grammar.
    """
    container: Tree = entry_node.parent  # type: ignore[attr-defined]
    container.children.remove(entry_node)
    container = prune_empty_wrappers(container, info)
    merge_adjacent_repetitions(container, info)


def prune_empty_wrappers(node: Tree, info: GrammarInfo) -> Tree:
    """Remove the nodes that existed only to wrap an entry that has just been deleted.

    A grammar can require an entry where it puts one. Fortran's line rule does: it is an
    assignment followed by an optional comment and newline, so deleting the assignment
    leaves a line the rule cannot derive, and the reconstructor refuses to write the file
    out. Such a wrapper is dropped along with the entry, taking that entry's trailing
    comment with it, since the comment described the line that has gone.

    A repetition rule is never pruned. Those accept any number of children, including none,
    and hold the blank lines and free text that are genuinely part of the file rather than
    part of any one entry. Nor is a node without a parent, which is as far up as the tree
    goes -- pruning does continue past the root a nested configuration was built on, since
    that root sits inside a larger tree that has to stay derivable.

    Args:
        node (Tree): The node the entry was removed from.
        info (GrammarInfo): Views over the compiled grammar, used to spot repetition rules.

    Returns:
        Tree: The nearest surviving container, which is where a repair pass should run.
    """
    while hasattr(node, "parent") and not info.is_repetition_rule(str(node.data)) and not _holds_entry(node):
        parent: Tree = node.parent  # type: ignore[attr-defined]
        parent.children.remove(node)
        node = parent
    return node


def _holds_entry(node: Tree) -> bool:
    """Report whether any of *node*'s children still holds an entry.

    Asked of the children rather than of the node, because a wrapper that is itself an entry
    can still have been emptied: a derived type is a ``key_block`` whose body has just lost
    its only assignment, and ``a%`` on its own is not something the grammar can derive.
    """
    return any(contains_entry(child) for child in node.children)


def merge_adjacent_repetitions(container: Tree, info: GrammarInfo) -> None:
    """Rejoin sibling nodes that became adjacent because an entry between them was removed.

    A grammar can require entries and the text around them to alternate. Fortran's does:
    the text between two namelists is one node, so removing a namelist leaves two such
    nodes side by side, which the start rule cannot derive and the reconstructor therefore
    refuses to write out. Merging them restores a shape the grammar accepts, and preserves
    the text exactly, since the merged node holds both sets of children in order.

    Only rules that are a plain repetition are merged, because only those accept the
    combined children. Everything else is left alone.

    Args:
        container (Tree): The container rule node an entry was just removed from.
        info (GrammarInfo): Views over the compiled grammar, used to identify the rules
            that may be merged.
    """
    index = 1
    while index < len(container.children):
        previous, current = container.children[index - 1], container.children[index]
        if (
            isinstance(previous, Tree)
            and isinstance(current, Tree)
            and previous.data == current.data
            and info.is_repetition_rule(str(current.data))
        ):
            for child in current.children:
                if isinstance(child, Tree):
                    child.parent = previous  # type: ignore[attr-defined]
            previous.children.extend(current.children)
            del container.children[index]
        else:
            index += 1
