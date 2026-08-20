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
from lark.exceptions import UnexpectedInput

from access.config.grammar_contract import KEY_LIST, KEY_VALUE, REPEAT_RULE
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY
from access.config.tree_navigation import (
    AddParent,
    contains_entry,
    is_value_node,
    node_values,
    repeat_parts,
)

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


# --- Writing a list through the nodes backing it ---
#
# Usually one value per node, written in place. A repeat is the exception: it backs several
# elements at once, so writing one of them splits the run and the node has to be replaced.


def _runs_of(values: Sequence[Any]) -> list[tuple[int, Any]]:
    """Group equal neighbouring values, so they can be written back as repeat counts.

    Args:
        values (Sequence[Any]): The values to group.

    Returns:
        list[tuple[int, Any]]: The length and value of each run, in order.
    """
    runs: list[tuple[int, Any]] = []
    for value in values:
        if runs and type(runs[-1][1]) is type(value) and runs[-1][1] == value:
            runs[-1] = (runs[-1][0] + 1, value)
        else:
            runs.append((1, value))
    return runs


def _source_token(node: Tree) -> tuple[str, str]:
    """Return the value rule and token text a value was written with.

    Args:
        node (Tree): A value-type rule node or a repeat node.

    Returns:
        tuple[str, str]: The rule name and the token text.

    Raises:
        TypeError: If *node* is a repeat of null values, which names no type to write in.
    """
    if node.data == REPEAT_RULE:
        _, inner = repeat_parts(node)
        if inner is None:
            raise TypeError("Trying to change value type")
        node = inner
    return str(node.data), str(node.children[0])


def _value_texts(node: Tree, values: Sequence[Any]) -> list[str]:
    """Write the values one node holds as token text, regrouping equal ones into repeats.

    The values are written in the notation of the node they replace, so an edit does not
    restyle what is around it, and a run that stays constant stays written as ``n*v`` rather
    than being expanded.

    Args:
        node (Tree): The node currently holding the values.
        values (Sequence[Any]): The values to write in its place.

    Returns:
        list[str]: The text of each run, in order.

    Raises:
        TypeError: If a value does not match the type of the node it replaces.
    """
    if all(value is None for value in values):
        # A run of nulls needs no notation, and a repeat of them is the only way to write
        # one: there is no value type involved, so nothing has to be preserved.
        return [f"{len(values)}*"]

    rule, token_text = _source_token(node)
    handler = VALUE_TYPE_HANDLER_REGISTRY[rule]
    texts = []
    for count, value in _runs_of(values):
        if not handler.type_check(value):
            raise TypeError("Trying to change value type")
        text = handler.to_token(value, token_text)
        texts.append(f"{count}*{text}" if count > 1 else text)
    return texts


def _parse_values(lark: Lark, texts: Sequence[str]) -> list[Tree | Token]:
    """Parse value texts into the nodes and separators an entry would hold them in.

    The nodes are not built by hand: the text is handed back to Lark with an entry rule as
    the start symbol, exactly as adding a key does, so rule inlining and aliasing take care
    of themselves.

    Args:
        lark (Lark): The compiled parser.
        texts (Sequence[str]): Token text for each value, in the notation wanted.

    Returns:
        list[Tree | Token]: The value nodes, with the grammar's own separator between them.
    """
    rule = KEY_LIST if len(texts) > 1 else KEY_VALUE
    # Whether an entry rule ends at its last value or takes the rest of the line with it is
    # up to the format -- Fortran's stops, MOM6's and NUOPC's do not -- so offer both.
    assignment = f"K = {', '.join(texts)}"
    try:
        fresh = lark.parse(assignment, start=rule)
    except UnexpectedInput:
        fresh = lark.parse(assignment + "\n", start=rule)

    lo, hi = _value_span(fresh)
    region = fresh.children[lo:hi]
    for node in region:
        if isinstance(node, Tree):
            AddParent().visit(node)
    return region


def replace_value_node(entry: Tree, node: Tree, texts: Sequence[str], lark: Lark) -> list[Tree]:
    """Replace one value node of an entry with the nodes needed to hold *texts*.

    Writing one element of ``6*9.0`` splits the run, so a single node becomes three
    (``2*9.0, 8.0, 3*9.0``). Only that node is replaced, so whatever the entry writes
    between its other values -- a comment, a line break -- is left exactly as it was.

    Args:
        entry (Tree): The entry rule node holding *node*.
        node (Tree): The value node to replace.
        texts (Sequence[str]): Token text for each value that takes its place.
        lark (Lark): The compiled parser.

    Returns:
        list[Tree]: The value nodes now standing in for it, in order.
    """
    region = _parse_values(lark, texts)
    position = entry.children.index(node)
    for fresh in region:
        if isinstance(fresh, Tree):
            fresh.parent = entry  # type: ignore[attr-defined]
    entry.children[position : position + 1] = region
    return [fresh for fresh in region if is_value_node(fresh)]


def recategorise_entry(entry: Tree, texts: Sequence[str], lark: Lark) -> list[Tree]:
    """Replace an entry's whole value region, changing what kind of entry it is.

    One value is a scalar assignment and several are a list one, so an entry that held a
    repeat in the position of a single value becomes a list entry once that repeat is split
    into different values. Only the value region is taken from the freshly parsed entry, so
    the indentation, key, separator and anything after the values stay as they were.

    Args:
        entry (Tree): The entry rule node to rewrite.
        texts (Sequence[str]): Token text for each value, in the notation wanted.
        lark (Lark): The compiled parser.

    Returns:
        list[Tree]: The value nodes now in the entry, in order.
    """
    region = _parse_values(lark, texts)
    lo, hi = _value_span(entry)
    for fresh in region:
        if isinstance(fresh, Tree):
            fresh.parent = entry  # type: ignore[attr-defined]
    entry.children[lo:hi] = region
    # The children are the ones Lark built for that rule, so the node is the shape the
    # grammar and the reconstructor expect of it.
    entry.data = KEY_LIST if len(texts) > 1 else KEY_VALUE
    return [fresh for fresh in region if is_value_node(fresh)]


def _value_span(entry: Tree) -> tuple[int, int]:
    """Return the half-open range of *entry*'s children from its first value to its last."""
    positions = [index for index, child in enumerate(entry.children) if is_value_node(child)]
    return positions[0], positions[-1] + 1


def _entry_value_nodes(entry: Tree) -> list[Tree]:
    """Return the value nodes an entry holds, in the order it writes them."""
    return [child for child in entry.children if is_value_node(child)]


def write_values(refs: Sequence[Tree], values: Sequence[Any], lark: Lark) -> list[Tree]:
    """Write a whole list of values through the nodes backing it.

    Usually one value per node, written in place. A repeat is the exception: it backs
    several elements at once, so writing one of them splits the run and that node has to be
    replaced rather than updated.

    Args:
        refs (Sequence[Tree]): The node backing each element, a repeat appearing once per
            element it covers.
        values (Sequence[Any]): The new value for each element.
        lark (Lark): The compiled parser.

    Returns:
        list[Tree]: The node now backing each element.
    """
    entry = _entry_of(refs[0])
    nodes = _entry_value_nodes(entry)
    if not any(node.data == REPEAT_RULE for node in nodes):
        for node, value in zip(refs, values, strict=True):
            update_node_value(node, value)
        return list(refs)

    by_node: dict[int, list[int]] = {}
    for position, node in enumerate(refs):
        by_node.setdefault(id(node), []).append(position)

    # An entry that held a repeat where a single value goes becomes a list entry once the
    # repeat is split, and that is a change to the entry itself rather than to one node.
    texts_by_node = [(node, _value_texts(node, [values[at] for at in by_node[id(node)]])) for node in nodes]
    if str(entry.data) == KEY_VALUE and sum(len(texts) for _, texts in texts_by_node) > 1:
        fresh = recategorise_entry(entry, [text for _, texts in texts_by_node for text in texts], lark)
    else:
        fresh = [new for node, texts in texts_by_node for new in replace_value_node(entry, node, texts, lark)]

    return [node for node in fresh for _ in node_values(node)]


def _entry_of(node: Tree) -> Tree:
    """Return the entry rule node a value node belongs to.

    A value node is always a direct child of its entry, so this is just its parent. Which
    entries hold a key is tracked by the reader rather than derived here, since a key can be
    written by several of them.

    Args:
        node (Tree): A value-type rule node or a repeat node.

    Returns:
        Tree: The entry rule node holding it.
    """
    return node.parent  # type: ignore[attr-defined]
