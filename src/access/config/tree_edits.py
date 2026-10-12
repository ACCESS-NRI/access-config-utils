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

from access.config.grammar_contract import COMMENT_RULES, REPEAT_RULE
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY, identical_values
from access.config.tree_navigation import AddParent, contains_entry, repeat_parts, token_value

if TYPE_CHECKING:
    from lark import Lark

    from access.config.grammar_compiled import GrammarInfo


def update_node_value(rule_node: Tree, value: Any) -> None:
    """Updates the value stored in a value-type rule node.

    The rule node must be a ``Tree`` whose ``.data`` is a key in
    ``VALUE_TYPE_HANDLER_REGISTRY`` (e.g. ``"integer"``, ``"float"``). The handler is
    looked up from that registry.

    Args:
        rule_node (Tree): Value-type rule node to update.
        value (Any): New value.

    A node that already holds *value* is left as it is, so writing a list leaves the
    elements that did not change written exactly as they were -- ``1.0d0`` stays
    ``1.0d0`` -- and a value that cannot be written anew, such as a NaN, can still be
    written over itself.

    Raises:
        TypeError: Raises an exception if the new and old value types do not match.
    """
    text = rewritten_text(rule_node, value)
    # The Token storing the value is always the first child of a value-type rule node.
    token = rule_node.children[0]
    assert isinstance(token, Token)
    if text != token:
        rule_node.children[0] = token.update(value=text)


def rewritten_text(rule_node: Tree, value: Any) -> str:
    """Return the token text that writes *value* over a value-type node.

    The node's own text if it already holds *value*; see ``ValueTypeHandler.rewrite``.

    Raises:
        TypeError: If the node is not a value-type node, or holds another type.
    """
    handler = VALUE_TYPE_HANDLER_REGISTRY.get(str(rule_node.data))
    if handler is None:
        raise TypeError("Trying to change value type")
    return handler.rewrite(value, str(rule_node.children[0]))


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
    # of these -- a rule for the text between blocks may be one -- and splicing it
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


def replace_entry_contents(entry: Tree, fresh: Tree) -> None:
    """Make *entry* hold what *fresh* holds, keeping the node itself.

    An entry node is what the references to an entry name, and what a line holding other
    entries keeps in its place, so a rewritten entry keeps its node and takes on the
    children of a freshly parsed one instead. The children are moved, not copied.

    Args:
        entry (Tree): The entry rule node in the parse tree.
        fresh (Tree): A freshly parsed entry rule node to take the contents of.
    """
    for child in fresh.children:
        if isinstance(child, Tree):
            child.parent = entry  # type: ignore[attr-defined]
    entry.children = fresh.children
    entry.data = fresh.data


def blank_comments(node: Tree) -> None:
    """Write every comment at or below *node* as nothing, so its text is not copied.

    Used on a copy of part of a file whose layout a new entry or element is to take on,
    without taking on someone else's comment. A carriage return the comment ran up to is
    kept: a comment rule may stop only at the line feed, and the line it ends has to keep
    its Windows line ending.

    Args:
        node (Tree): The rule node to blank the comments of, changed in place.
    """
    for subtree in node.iter_subtrees():
        if subtree.data in COMMENT_RULES:
            subtree.children = [
                child.update(value="\r" if child.endswith("\r") else "") if isinstance(child, Token) else child
                for child in subtree.children
            ]


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

    A grammar can require an entry where it puts one. A line rule may: it is an
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

    A grammar can require entries and the text around them to alternate: where the text
    between two blocks is one node, removing a block leaves two such
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


# --- Writing a list through the nodes backing it ---
#
# One value per node, written in place. A repeat is the exception: it backs several elements
# at once, so it is written as a whole or not at all.


def write_values(refs: Sequence[Tree | None], values: Sequence[Any]) -> None:
    """Write a whole list of values through the nodes backing it.

    One value per node, written in place. Two things complicate that, and both come from a
    list whose elements are not written one for one in the file:

    - A repeat backs several elements at once, so ``6*9.0`` is six of them. Giving them all
      the same new value rewrites the repeated token, ``6*9.0`` to ``6*9.5``; any other
      change to one of them would split the run, which cannot be done in place.
    - An array can have positions no entry wrote, which have no node to write through.

    The write is all or nothing: every value is checked, and every token text worked out,
    before the first node changes.

    Args:
        refs (Sequence[Tree | None]): The node backing each element, a repeat appearing once
            per element it covers and ``None`` where the file wrote nothing.
        values (Sequence[Any]): The new value for each element.

    Raises:
        ValueError: If a value was assigned to a position the file never wrote, or to part
            of a repeat.
        TypeError: If a value is not of the type of the element it replaces, or the format
            cannot write it.
    """
    runs: dict[int, tuple[Tree, list[Any]]] = {}
    for ref, value in zip(refs, values, strict=True):
        if ref is None:
            if value is not None:
                raise ValueError(
                    "Cannot write to an array position the file leaves unset; assign a whole new list instead"
                )
        else:
            # Keyed on identity: a Tree hashes by its contents, so two nodes written the
            # same way would collide.
            runs.setdefault(id(ref), (ref, []))[1].append(value)

    writes = [write for node, run in runs.values() if (write := _run_text(node, run)) is not None]
    for target, text in writes:
        token = target.children[0]
        assert isinstance(token, Token)
        if text != token:
            target.children[0] = token.update(value=text)


def _run_text(node: Tree, values: Sequence[Any]) -> tuple[Tree, str] | None:
    """Return the node to write and the token text to write over it, if anything changes.

    Args:
        node (Tree): A value-type rule node, or a repeat node.
        values (Sequence[Any]): The new value of each element the node backs: one, unless
            it is a repeat.

    Returns:
        tuple[Tree, str] | None: The value-type node holding the token, and its new text;
            ``None`` if the node already holds the values.

    Raises:
        ValueError: If the values are not all alike for a repeat.
        TypeError: If a value is not of the type of the node, or the format cannot write it.
    """
    if node.data != REPEAT_RULE:
        return node, rewritten_text(node, values[0])
    _, inner = repeat_parts(node)
    if inner is None:
        # A run of nulls has no type to write a value in.
        if any(value is not None for value in values):
            raise TypeError("Trying to change value type")
        return None
    if all(identical_values(token_value(inner), value) for value in values):
        return None
    if not all(identical_values(values[0], value) for value in values):
        raise ValueError(
            f"Cannot change part of the repeat {node.children[0]}: it stands for {len(values)} elements, "
            "and only all of them together can be given another value; delete it and assign it again"
        )
    return inner, rewritten_text(inner, values[0])
