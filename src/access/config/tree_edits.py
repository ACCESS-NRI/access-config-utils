# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Changing a Lark parse tree without losing the ability to write it back out.

Round-trip fidelity is the constraint every function here works under: the tree that comes
out has to be one the grammar could have produced, or the reconstructor refuses to write
it. That rules out building nodes by hand, which is why a new entry is *parsed* into
existence by ``parse_entry_nodes`` rather than assembled, and it is why deleting an entry
may need ``prune_empty_wrappers`` and ``merge_adjacent_repetitions`` to repair the shape it
leaves behind. There are two repairs because a grammar can be underivable in two ways once
an entry goes:

- A wrapper that *required* the entry is left empty. A line rule is an assignment followed
  by an optional comment and newline, so without the assignment it derives nothing. The
  wrapper is dropped along with the entry, and with it the entry's trailing comment, which
  described the line that has gone. A repetition rule is never dropped: it accepts any
  number of children, none included, and holds the blank lines and free text that belong to
  the file rather than to any one entry.
- Entries and the text around them may have to alternate. Where the text between two blocks
  is one node, removing a block leaves two such nodes side by side, which the start rule
  cannot derive. Merging them restores a shape the grammar accepts, and keeps the text
  exactly, since the merged node holds both sets of children in order.

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
    """Update the value stored in a value-type rule node.

    A node that already holds *value* is left as it is, so writing a list leaves the
    elements that did not change written exactly as they were (``1.0d0`` stays ``1.0d0``),
    and a value that cannot be written anew, such as a NaN, can be written over itself.

    Raises:
        TypeError: If the node is not a value-type node, or *value* is of another type.
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

    *container* must be among the parser's start symbols. The nodes' interiors have
    ``.parent`` set; ``splice_entry_nodes`` sets it on the nodes themselves.

    Raises:
        UnexpectedInput: If *snippet* is not valid in *container*.
    """
    parsed = lark.parse(snippet, start=container)
    children = parsed.children if str(parsed.data) == container else [parsed]

    # Drop nodes that matched no text. A rule able to derive the empty string produces one
    # of these -- a rule for the text between blocks may be one -- and splicing it would put
    # a second empty node beside the container's own, which the grammar cannot derive and
    # the reconstructor therefore refuses to write out. Only the top level is filtered: a
    # block being created is legitimately an empty node, and it is nested, not top-level.
    nodes = [child for child in children if not (isinstance(child, Tree) and not child.children)]

    for node in nodes:
        if isinstance(node, Tree):
            AddParent().visit(node)
    return nodes


def splice_entry_nodes(container: Tree, nodes: Sequence[Tree | Token], index: int) -> None:
    """Insert freshly parsed entry nodes into a container, at child position *index*.

    Kept separate from ``parse_entry_nodes`` so a candidate entry can be validated in full
    before the tree is touched: everything before this call is reversible, and this is not.
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
    """
    for subtree in node.iter_subtrees():
        if subtree.data in COMMENT_RULES:
            subtree.children = [
                child.update(value="\r" if child.endswith("\r") else "") if isinstance(child, Token) else child
                for child in subtree.children
            ]


def remove_entry_node(entry_node: Tree, info: GrammarInfo) -> None:
    """Unlink an entry from the tree, repairing the shape its removal leaves behind.

    Two repairs, explained in the module docstring: ``prune_empty_wrappers`` for a wrapper
    that *required* the entry, and ``merge_adjacent_repetitions`` for the text either side.
    """
    container: Tree = entry_node.parent  # type: ignore[attr-defined]
    container.children.remove(entry_node)
    container = prune_empty_wrappers(container, info)
    merge_adjacent_repetitions(container, info)


def prune_empty_wrappers(node: Tree, info: GrammarInfo) -> Tree:
    """Remove the nodes that existed only to wrap an entry that has just been deleted.

    Climb from *node*, the one the entry was removed from, dropping each node that holds no
    entry, and return the nearest survivor: the container where a repair pass should run.
    """
    # The climb stops at a repetition rule, which may be empty, and at a node with no
    # ``.parent``, the root of the whole tree. The root of a nested configuration has one,
    # since it sits in a larger tree that has to stay derivable: pruning goes on past it.
    while hasattr(node, "parent") and not info.is_repetition_rule(str(node.data)) and not _holds_entry(node):
        parent: Tree = node.parent  # type: ignore[attr-defined]
        parent.children.remove(node)
        node = parent
    return node


def _holds_entry(node: Tree) -> bool:
    """Report whether any of *node*'s children still holds an entry.

    Asked of the children, not the node: a wrapper that is itself an entry can still be
    emptied, as a derived type's ``key_block`` is, and ``a%`` on its own cannot be derived.
    """
    return any(contains_entry(child) for child in node.children)


def merge_adjacent_repetitions(container: Tree, info: GrammarInfo) -> None:
    """Rejoin sibling nodes that became adjacent because an entry between them was removed.

    Adjacent siblings of one plain repetition rule become a single node holding both sets of
    children in order, so the text is kept exactly. Everything else is left alone.
    """
    index = 1
    while index < len(container.children):
        previous, current = container.children[index - 1], container.children[index]
        # Only a plain repetition rule accepts the children of two nodes combined.
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


def write_values(refs: Sequence[Tree | None], values: Sequence[Any]) -> None:
    """Write a whole list of values through the nodes backing it.

    One value per node, written in place. Two things complicate that, and both come from a
    list whose elements are not written one for one in the file:

    - A repeat backs several elements at once, so ``6*9.0`` is six of them, and it appears
      in *refs* once for each. Giving them all the same new value rewrites the repeated
      token, ``6*9.0`` to ``6*9.5``; any other change to one of them would split the run,
      which cannot be done in place.
    - An array can have positions no entry wrote, which have no node to write through: they
      are ``None`` in *refs*.

    The write is all or nothing: every value is checked, and every token text worked out,
    before the first node changes.

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
    """Return the node to write and its new token text, or ``None`` if nothing changes.

    *values* are those of the elements *node* backs: one, or all of a repeat's, in which
    case the node returned is the one it repeats. Raises what ``write_values`` documents.
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
