# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Finding one's way around a Lark parse tree.

Everything here reads a tree and reports what is in it. Nothing here changes one; that is
``tree_edits``. The questions asked are always the same three: where are the entries, which
children are comment lines, and where does a new entry belong.

Lark parse tree structure
-------------------------
A Lark parse tree is made up of two node types that correspond directly to
grammar elements:

``Tree``: produced by a **grammar rule** (lowercase name in the grammar)
    - ``tree.data`` (``str``): the rule name, e.g. ``"key_value"``, ``"integer"``
    - ``tree.children`` (``list``): child nodes, each a ``Tree`` or ``Token``

``Token``: produced by a **grammar terminal** (UPPERCASE name in the grammar)
    - ``Token`` is a subclass of ``str``, so it compares and behaves like the
      matched text directly (e.g. ``token == "42"``).
    - ``token.type`` (``str``): the terminal name, e.g. ``"CNAME"``, ``"INT"``
    - Tokens are always **leaf nodes**, as such they have no children.

Because ``Token`` inherits from ``str``, code that only needs the matched text
can use a ``Token`` directly as a string without calling ``.value``.

Lark provides no way to walk *up* a tree, so ``AddParent`` annotates every rule node with a
``.parent`` back-reference as soon as it is parsed. Every rule name looked for here comes
from ``grammar_contract``, which states in full what a grammar has to provide.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lark import Token, Tree, Visitor

from access.config.grammar_contract import BLOCK_RULE, COMMENT_RULES, ENTRY_CATEGORIES, REPEAT_RULE
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY


class AddParent(Visitor):
    """Lark visitor that annotates every ``Tree`` node with a ``.parent`` back-reference.

    ``Token`` children (terminal leaves) are skipped and only ``Tree`` children (rule nodes)
    receive the ``.parent`` attribute. This enables upward traversal of the parse tree,
    which Lark does not provide natively.

    Only ever apply this to a freshly parsed tree. Visiting a tree that already carries the
    attribute trips the assertion below, so a new entry is annotated on its own before
    being spliced in, never by re-visiting the tree it is spliced into. Note also that
    deleting an entry does not clear ``.parent`` on the node it removes, and that the
    assertion is a development aid which ``python -O`` strips.
    """

    def __default__(self, tree: Tree) -> None:
        for subtree in tree.children:
            if isinstance(subtree, Tree):
                assert not hasattr(subtree, "parent")
                subtree.parent = tree  # type: ignore


def entry_nodes(node: Tree) -> list[Tree]:
    """Return the entry rule nodes at or below *node*, not entering a nested block.

    Args:
        node (Tree): Any rule node.

    Returns:
        list[Tree]: The entry nodes found, in the order written.
    """
    if node.data in ENTRY_CATEGORIES:
        return [node]
    found = []
    for child in node.children:
        if isinstance(child, Tree) and child.data != BLOCK_RULE:
            found += entry_nodes(child)
    return found


def entries_of(container: Tree) -> list[Tree]:
    """Return the entry rule nodes held directly by *container*, in the order written.

    Args:
        container (Tree): A container rule node: the parse tree root, or a ``block`` node.

    Returns:
        list[Tree]: The entry nodes, not descending into a nested block.
    """
    return [entry for child in container.children if isinstance(child, Tree) for entry in entry_nodes(child)]


def contains_entry(node: Tree | Token) -> bool:
    """Report whether a container child holds a configuration entry.

    Args:
        node (Tree | Token): A child of a container rule node.

    Returns:
        bool: True if an entry rule node is at or below *node*.
    """
    return isinstance(node, Tree) and bool(entry_nodes(node))


def _holds_comment(node: Tree) -> bool:
    """Report whether a comment rule node sits at or below *node*."""
    if node.data in COMMENT_RULES:
        return True
    return any(isinstance(child, Tree) and _holds_comment(child) for child in node.children)


def is_comment_line(node: Tree | Token) -> bool:
    """Report whether a container child is a line holding nothing but a comment.

    A comment-only line and a blank line are the same rule in every grammar, told apart by
    whether a comment node hangs below it. A child that holds an entry is never a comment
    line, however many comments it carries: those belong to the entry's own line and travel
    with it.

    Args:
        node (Tree | Token): A child of a container rule node.

    Returns:
        bool: True if *node* is a line holding only a comment.
    """
    return isinstance(node, Tree) and not contains_entry(node) and _holds_comment(node)


def entry_insertion_index(container: Tree) -> int:
    """Return the index in ``container.children`` at which a new entry belongs.

    One past the last child that holds an entry, so a new entry follows the last existing
    assignment and comes *before* any trailing blank lines or the token closing a block.
    Simply appending would put it after all of those, which for a block would place it
    outside the block entirely.

    The search is over the container's *direct* children even though the test for holding an
    entry looks deeper: in a Fortran namelist one child can be a whole line carrying several
    assignments, and a new entry has to go after that line rather than inside it.

    **A run of comment lines starting at that position is stepped over**, up to the first
    blank line, the next entry, or the end of the container. A comment there either carries
    on from the trailing comment of the line above -- MOM6 documents most of its parameters
    that way, beginning on the parameter's own line and continuing below it -- or heads what
    follows. Either way the new entry belongs after the whole run, and stopping at a blank
    line is what puts it *between* two comment blocks rather than inside the second one.

    Args:
        container (Tree): A container rule node: the parse tree root, or a ``"block"`` node.

    Returns:
        int: The insertion index; ``0`` when the container holds neither an entry nor a
            leading comment.
    """
    index = 0
    for position, child in enumerate(container.children):
        if contains_entry(child):
            index = position + 1
    while index < len(container.children) and is_comment_line(container.children[index]):
        index += 1
    return index


# --- Values ---
#
# A value node is usually one value, but a repeat is several: "6*9.0" is six elements
# written once. Reading that out is a question about a node, so it lives here; undoing it
# when one of those elements is written is a change, and lives in ``tree_edits``.


def is_value_node(node: Tree | Token) -> bool:
    """Report whether a node holds one or more values of an entry.

    Args:
        node (Tree | Token): A child of an entry rule node.

    Returns:
        bool: True for a value-type rule node or a repeat of one.
    """
    return isinstance(node, Tree) and (node.data in VALUE_TYPE_HANDLER_REGISTRY or node.data == REPEAT_RULE)


def repeat_parts(node: Tree) -> tuple[int, Tree | None]:
    """Split a repeat node into its count and the value-type node it repeats.

    Args:
        node (Tree): A repeat rule node.

    Returns:
        tuple[int, Tree | None]: The count, and the repeated node or ``None`` for ``n*``.
    """
    count = int(str(node.children[0]).rstrip("*"))
    inner = node.children[1] if len(node.children) > 1 else None
    assert inner is None or isinstance(inner, Tree)
    return count, inner


def node_values(node: Tree) -> list[Any]:
    """Return the values a value node contributes to its entry, in order.

    One for an ordinary value-type node, and *n* for ``n*v`` -- all equal, and all ``None``
    when the repeat names no value.

    Args:
        node (Tree): A value-type rule node or a repeat node.

    Returns:
        list[Any]: The values.
    """
    if node.data == REPEAT_RULE:
        count, inner = repeat_parts(node)
        return [None if inner is None else token_value(inner)] * count
    return [token_value(node)]


def token_value(node: Tree) -> Any:
    """Convert the token of a value-type rule node into a Python value."""
    return VALUE_TYPE_HANDLER_REGISTRY[str(node.data)].from_token(str(node.children[0]))


def index_positions(node: Tree | None, count: int) -> list[int] | None:
    """Return the position each of *count* values takes, or ``None`` if they follow on.

    Reads the qualifier's own text: ``(3)`` names where the values start, ``(2:5)`` a range,
    and ``(1:7:2)`` a strided one, which leaves gaps between the values it places. An
    implicit bound (``(2:)``, ``(:5)``, ``(:)``) is filled in from how many values there
    are. Positions are as written, so they may start anywhere, including at zero or below.

    A qualifier names where an entry's values *begin*, not how many it may have: Fortran's
    ``v(3) = 1, 2, 3`` fills three positions from the third.

    Args:
        node (Tree | None): The index rule node, or ``None`` for an unqualified assignment.
        count (int): How many values the entry holds.

    Returns:
        list[int] | None: The position of each value, or ``None`` when there is no qualifier
            or it is one this does not model.
    """
    if node is None:
        return None
    spec = str(node.children[0])
    if "," in spec:
        # A multidimensional qualifier addresses an array of arrays, which is a shape this
        # flat model cannot hold. Left alone, it stays part of the key.
        return None
    bounds = spec.split(":")
    if len(bounds) > 3:
        return None
    try:
        start = int(bounds[0]) if bounds[0].strip() else None
        stop = int(bounds[1]) if len(bounds) > 1 and bounds[1].strip() else None
        step = int(bounds[2]) if len(bounds) > 2 and bounds[2].strip() else 1
    except ValueError:
        return None
    if step == 0:
        return None
    if len(bounds) == 1:
        if start is None:
            return None
        first, step = start, 1
    else:
        first = start if start is not None else (stop - (count - 1) * step if stop is not None else 1)
    return [first + offset * step for offset in range(count)]


def place_indexed(
    slots: dict[int, tuple[Any, Tree | None]],
    positions: list[int] | None,
    values: Sequence[Any],
    refs: Sequence[Tree | None],
) -> None:
    """Record where an entry's values sit in the array its key names.

    Args:
        slots (dict[int, tuple[Any, Tree | None]]): Value and backing node by position,
            added to in place. A position already taken is overwritten, as the later
            assignment wins.
        positions (list[int] | None): Where the values go, or ``None`` to follow on from the
            first position Fortran numbers.
        values (Sequence[Any]): The entry's values.
        refs (Sequence[Tree | None]): The node backing each value, ``None`` where the entry
            wrote the position without a value.
    """
    if positions is None:
        # An unqualified assignment to an array otherwise written with indices starts at the
        # first position Fortran numbers, which is one.
        positions = [1 + offset for offset in range(len(values))]
    for position, value, ref in zip(positions, values, refs, strict=True):
        slots[position] = (value, ref)


def flatten_slots(slots: dict[int, tuple[Any, Tree | None]]) -> tuple[list[Any], list[Tree | None]]:
    """Lay the recorded positions out as a list, filling any gap with a null.

    Positions are as the file wrote them, so they can start at any number and a stride or a
    sparse set of indices leaves holes. The holes become ``None``, backed by no node at all,
    which is what makes them unwritable.

    Args:
        slots (dict[int, tuple[Any, Tree | None]]): Value and backing node by position.

    Returns:
        tuple[list[Any], list[Tree | None]]: The values in order, and the node behind each.
    """
    order = range(min(slots), max(slots) + 1)
    filled = [slots.get(position, (None, None)) for position in order]
    return [value for value, _ in filled], [ref for _, ref in filled]
