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

from lark import Token, Tree, Visitor

from access.config.grammar_contract import BLOCK_RULE, COMMENT_RULES, ENTRY_CATEGORIES


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
