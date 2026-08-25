# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Reading a parse tree into the entries it holds, and the nodes those entries live in.

This is the one place that turns a tree into data. It walks a container rule node and, for
every entry directly below it, records an ``EntryRef``: which category the entry is, the
node the whole entry hangs from, and the nodes holding its values. Everything an edit needs
to find later is in there, so nothing downstream has to search the tree again.

The reader deliberately stops at data. It does **not** build the dict that will hold the
values -- ``entry_value`` turns a reference into a Python value, and wrapping a list or a
block in something dict-like is the job of the layer that owns the dict. Keeping that out of
here is what lets the tree layer sit below the data layer rather than beside it.

Every rule name looked for here -- the four entry categories, ``key``, ``block`` -- comes
from ``grammar_contract``, which states in full what a grammar has to provide and why. The
shapes relied on follow from it: a key rule node has a ``key`` child whose first child is
the ``Token`` holding the key text, and a value-type rule node has a single ``Token`` child
holding the value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from lark import Token, Tree
from lark.visitors import Interpreter

from access.config.grammar_contract import (
    INDEX_RULE,
    KEY_BLOCK,
    KEY_LIST,
    KEY_NULL,
    KEY_RULE,
    KEY_VALUE,
)
from access.config.tree_navigation import (
    flatten_slots,
    index_positions,
    is_value_node,
    node_values,
    place_indexed,
)

if TYPE_CHECKING:
    from access.config.grammar_compiled import ParseContext


@dataclass(frozen=True)
class EntryRef:
    """Where in the parse tree one key's entry lives.

    An entry is edited by mutating the nodes named here, so a reference has to survive as
    long as the configuration does. It stays valid across edits to *other* keys, because
    inserting or removing an entry never rebuilds the nodes of its neighbours.

    Args:
        category (str): One of ``ENTRY_CATEGORIES``, saying which shape this entry has.
        entry_nodes (tuple[Tree, ...]): Every node that wrote this key, in the order the
            file has them. Usually one. A file may assign a key more than once, and only
            the last assignment reaches the configuration -- but *all* of them have to go
            when the key is deleted, or it comes back the next time the file is read.
        value_nodes (tuple[Tree | None, ...]): The node backing each of the entry's values,
            in order. One for ``key_value``, one per element for ``key_list``, none for
            ``key_block`` and ``key_null``. A repeat node appears once per element it
            covers. ``None`` marks an array position no entry wrote, which therefore has no
            value and cannot be written through.
        block_nodes (tuple[Tree, ...]): The ``block`` nodes holding the entry's contents,
            for a ``key_block``; empty otherwise. More than one when the file writes the
            block several times and the format reads those as separate records rather than
            merging them -- see ``ConfigParser.repeated_blocks``. ``block_node`` reads the
            first, for the callers that only ever have one.
        addable (bool): Whether a new entry can be spliced into a block node. False for a
            block merged from several in the file, whose node is not itself in the parse
            tree, and for one held by a rule that is not a Lark start symbol.
        values (tuple[Any, ...] | None): The values, for an array gathered from several
            indexed entries; ``None`` for every other entry, whose values are read straight
            off *value_nodes*.

            Only a gathered array needs them stated. Everywhere else a node and an element
            correspond closely enough to derive one from the other -- a repeat node stands
            for the run of equal elements that follows it. Once positions come from
            qualifiers that is no longer true: the same repeat can back positions 1 and 3
            with something else at 2, and reading it at each would multiply its values.
    """

    category: str
    entry_nodes: tuple[Tree, ...]
    value_nodes: tuple[Tree | None, ...] = ()
    block_nodes: tuple[Tree, ...] = ()
    addable: bool = True
    values: tuple[Any, ...] | None = None

    @property
    def block_node(self) -> Tree | None:
        """Tree | None: The first block node, or ``None`` if this is not a ``key_block``."""
        return self.block_nodes[0] if self.block_nodes else None


def merge_blocks(first: Tree, second: Tree) -> Tree:
    """Combine two block nodes into one that reads as though they were written together.

    The result is a node the parse tree does not contain, holding the *real* children of
    both. That is what makes it safe: the entries inside it are the ones Lark built, so
    updating a value through this node mutates the original tree, and the file is written
    back with only that value changed. Nothing is copied, and the two source blocks stay
    exactly where they were.

    It is deliberately never given to ``AddParent``: its children keep the ``.parent`` they
    already had, pointing into the real tree, so an edit through this node reaches the file.

    Args:
        first (Tree): Block node seen first, or a merged node built from earlier ones.
        second (Tree): Block node to merge into it.

    Returns:
        Tree: A block node holding the children of both, in order.
    """
    return Tree(first.data, [*first.children, *second.children])


def entry_value(ref: EntryRef) -> Any:
    """Return the Python value an entry holds.

    A ``key_block`` has no value of its own: its contents are the entries of ``block_node``,
    which the caller reads separately and wraps in whatever it uses for a nested
    configuration. Ask this only about the other three categories.

    Args:
        ref (EntryRef): The entry to read, of any category but ``key_block``.

    Returns:
        Any: The scalar, the list of values, or ``None`` for a valueless entry.
    """
    if ref.category == KEY_NULL:
        return None
    if ref.values is not None:
        return list(ref.values)
    values = element_values(ref.value_nodes)
    return values if ref.category == KEY_LIST else values[0]


def element_values(nodes: Sequence[Tree | None]) -> list[Any]:
    """Return one value per element, from the node backing each.

    Two nodes do not correspond one for one with elements. A repeat appears once per element
    it covers, so reading it once per appearance would multiply its values; and an array
    position no entry wrote has no node at all, which is a ``None`` value.

    Args:
        nodes (Sequence[Tree | None]): The node backing each element, in order.

    Returns:
        list[Any]: One value per element.
    """
    values: list[Any] = []
    for index, node in enumerate(nodes):
        if node is None:
            values.append(None)
        elif index == 0 or node is not nodes[index - 1]:
            values.extend(node_values(node))
    return values


def read_entries(container: Tree, ctx: ParseContext) -> dict[str, EntryRef]:
    """Read the entries a container holds, in the order they are written.

    Args:
        container (Tree): The container rule node to read: the parse tree root, a ``block``
            node, or a throwaway node wrapping candidate entry text.
        ctx (ParseContext): Supplies the key normalisation this format uses.

    Returns:
        dict[str, EntryRef]: One reference per key, keyed by the normalised key name.

    Raises:
        ValueError: If an entry node holds no ``key`` rule, or the wrong number of values
            for its category.
        TypeError: If a ``key`` rule node does not begin with a ``Token``.
    """
    return EntryReader(ctx).read(container)


class EntryReader(Interpreter):
    """Lark interpreter collecting an ``EntryRef`` per entry of a container.

    A Lark ``Transformer`` would be the usual choice, but it replaces rule nodes with
    transformed values, destroying the original tree. Using an ``Interpreter`` instead keeps
    references to the original rule nodes so that they can be mutated later to support
    round-trip editing. The ``Interpreter`` also skips visiting child rule nodes
    automatically, so each callback handles an entire entry subtree in one call -- which is
    also what stops the reader descending into a nested block, whose entries belong to that
    block and are read separately when it is opened.

    The callbacks have to be named after the entry categories, because that is how Lark
    dispatches; a test pins the two lists together.

    Args:
        ctx (ParseContext): Supplies the key normalisation this format uses.
    """

    _refs: dict[str, EntryRef]  # References collected while traversing the tree.
    _ctx: ParseContext  # Compiled grammar and per-parser settings.
    # Value and backing node by array position, for the keys written with a qualifier.
    _slots: dict[str, dict[int, tuple[Any, Tree | None]]]

    def __init__(self, ctx: ParseContext) -> None:
        self._ctx = ctx
        super().__init__()

    def read(self, container: Tree) -> dict[str, EntryRef]:
        """Visit a container and return the references to the entries below it.

        Args:
            container (Tree): Container rule node to read.

        Returns:
            dict[str, EntryRef]: One reference per key, keyed by the normalised key name.
        """
        self._refs = {}
        self._slots = {}
        self.visit(container)
        return self._refs

    def _record(self, key: str, ref: EntryRef) -> None:
        """Store *ref* for *key*, keeping the entry nodes of any earlier assignment.

        A later assignment wins: its category and its value nodes are what the
        configuration reports. What it does not do is replace the record of which entries
        wrote the key, since deleting the key has to remove every one of them.

        Args:
            key (str): The normalised key.
            ref (EntryRef): The reference just read.
        """
        previous = self._refs.get(key)
        if previous is not None:
            ref = replace(ref, entry_nodes=previous.entry_nodes + ref.entry_nodes)
        self._refs[key] = ref

    @staticmethod
    def _index_node(tree: Tree) -> Tree | None:
        """Return the array qualifier of an entry, if it has one."""
        return next((child for child in tree.children if isinstance(child, Tree) and child.data == INDEX_RULE), None)

    def _place(self, key: str, tree: Tree, values: Sequence[Any], nodes: Sequence[Tree | None]) -> str | None:
        """Merge an indexed entry into the array its key names, or say where to store it.

        An indexed assignment writes part of an array, so ``v(1) = 1`` and ``v(2) = 2`` are
        two entries making up a single two-element ``v``. The positions are as written, and
        any left unwritten -- by a stride, or by indices that skip -- read as null. An
        unqualified assignment to a key already written with indices joins the same array.

        Args:
            key (str): The normalised key.
            tree (Tree): The entry rule node.
            values (Sequence[Any]): The entry's values.
            nodes (Sequence[Tree | None]): The node backing each value, ``None`` for a
                position the entry wrote without one.

        Returns:
            str | None: ``None`` once the entry has been merged, or the key to store it
                under as an ordinary entry.
        """
        index = self._index_node(tree)
        positions = index_positions(index, len(values))
        if index is not None and positions is None:
            # A qualifier this does not model, a multidimensional one. Keeping it in the key
            # is what stops "v(1,1)" and "v(3,3)" from being read as the same key.
            return f"{key}({index.children[0]})"
        if index is None and key not in self._slots:
            return key
        if key not in self._slots:
            self._slots[key] = self._seed_slots(key)
        place_indexed(self._slots[key], positions, values, nodes)
        merged_values, merged_nodes = flatten_slots(self._slots[key])
        self._record(key, EntryRef(KEY_LIST, (tree,), tuple(merged_nodes), values=tuple(merged_values)))
        return None

    def _seed_slots(self, key: str) -> dict[int, tuple[Any, Tree | None]]:
        """Lay out what a key holds as array positions, so an indexed entry can join it.

        Needed when a file writes the unqualified assignment first, as in ``v = 1`` followed
        by ``v(2) = 2``: the values already read have to take up the positions they would
        have occupied before the indexed one can be placed beside them.

        Args:
            key (str): The normalised key.

        Returns:
            dict[int, tuple[Any, Tree | None]]: Value and backing node by position.
        """
        ref = self._refs.get(key)
        if ref is None:
            return {}
        if ref.category == KEY_NULL:
            # A valueless assignment backs no value; as a position of an array that is a
            # hole, so it keeps no node either.
            return {1: (None, None)}
        values = list(entry_value(ref)) if ref.category == KEY_LIST else [entry_value(ref)]
        return {1 + offset: pair for offset, pair in enumerate(zip(values, ref.value_nodes, strict=True))}

    def _get_key(self, tree: Tree) -> str:
        """Given an entry rule node, extract and return the key name.

        Finds the ``"key"`` rule node among *tree*'s children, then reads the ``Token``
        (terminal) that is its first child and that token's text is the key name.

        Args:
            tree (Tree): An entry rule node (e.g. ``Tree`` with ``.data == "key_value"``).

        Returns:
            str: The key name, normalised for this format's key case-sensitivity.

        Raises:
            ValueError: If no ``"key"`` rule node is found among the children.
            TypeError: If no ``Token`` is found as the first child of the ``"key"`` rule
                node.
        """
        key_rules = [child.children for child in tree.children if child.data == KEY_RULE]
        if len(key_rules) == 0:
            raise ValueError("No 'key' rule nodes found among children of key rule node")
        else:
            # Multiple "key" rule nodes are possible (e.g., if the tree is a key_block
            # storing one of more key_values) but the correct one is always the first.
            key_rule = key_rules[0]

        # The token holding the key name is the first child of the "key" rule node.
        key_token = key_rule[0]
        if not isinstance(key_token, Token):
            raise TypeError("No key found.")

        return self._ctx.normalise_key(key_token.value)

    @staticmethod
    def _value_nodes(tree: Tree) -> tuple[Tree, ...]:
        """Return the value-type rule nodes among an entry's children.

        Args:
            tree (Tree): A ``"key_value"`` or ``"key_list"`` rule node.

        Returns:
            tuple[Tree, ...]: The value-type rule nodes, in the order written.

        Raises:
            ValueError: If there are none.
        """
        found = [child for child in tree.children if is_value_node(child)]
        if not found:
            raise ValueError("No values found in Tree")
        # A repeat node holds several values at once, so it appears once per value it
        # covers and the references are no longer one-to-one with the nodes in the tree.
        return tuple(node for node in found for _ in node_values(node))

    def key_value(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_value"`` rule nodes.

        Args:
            tree (Tree): Rule node produced by the ``"key_value"`` grammar rule.

        Raises:
            ValueError: If the node holds more than one value, which means the grammar
                aliased it to the wrong category.
        """
        nodes = self._value_nodes(tree)
        stored = self._place(self._get_key(tree), tree, element_values(nodes), nodes)
        if stored is None:
            return
        distinct = {id(node) for node in nodes}
        if len(distinct) > 1:
            raise ValueError("More than one value found in Tree")
        # "x = 3*1" is written where a single value goes but means three, so it reads as a
        # list: the category follows the number of values, not the rule that wrote them.
        category = KEY_VALUE if len(nodes) == 1 else KEY_LIST
        self._record(stored, EntryRef(category, (tree,), nodes))

    def key_list(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_list"`` rule nodes.

        Args:
            tree (Tree): Rule node produced by the ``"key_list"`` grammar rule.
        """
        nodes = self._value_nodes(tree)
        stored = self._place(self._get_key(tree), tree, element_values(nodes), nodes)
        if stored is not None:
            self._record(stored, EntryRef(KEY_LIST, (tree,), nodes))

    def key_block(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_block"`` rule nodes.

        A key can name more than one block, and what that means depends on which rule holds
        the contents. A derived type spells one component per line, so ``a%b = 1`` and
        ``a%c = 2`` are two entries under the single key ``A``: those are merged, since
        dropping the earlier ones loses data silently.

        A format may instead read a repeated block as *separate records*, and a Fortran
        namelist does -- ``READ`` stops at the first group of the name, so no single read
        ever sees the merge. Then each occurrence is kept, and the caller is handed a list
        of blocks. Only the format's primary block rule can repeat this way; see
        ``ConfigParser.repeated_blocks``.

        Args:
            tree (Tree): Rule node produced by the ``"key_block"`` grammar rule.

        Raises:
            ValueError: If the key was already given a value. Merging would build a block
                out of a value node and quietly drop the value.
        """
        key = self._get_key(tree)
        index = self._index_node(tree)
        if index is not None:
            # An indexed derived type, "a(1)%b". Its elements are not gathered into a list
            # of blocks, so the qualifier stays in the key -- otherwise "a(1)%b" and
            # "a(2)%b" would read as one key and the first would be lost.
            key = f"{key}({index.children[0]})"
        for child in tree.children:
            if isinstance(child, Tree) and str(child.data) in self._ctx.block_rules:
                previous = self._refs.get(key)
                if previous is not None and previous.category != KEY_BLOCK:
                    raise ValueError(f"'{key}' is assigned a value and used as a block")
                self._record(key, self._block_ref(tree, child, previous))
                return

    def _block_ref(self, tree: Tree, body: Tree, previous: EntryRef | None) -> EntryRef:
        """Build the reference for a block, folding it into any earlier one for the key.

        Args:
            tree (Tree): The ``key_block`` node.
            body (Tree): The child rule node holding the block's contents.
            previous (EntryRef | None): The reference already recorded for this key.

        Returns:
            EntryRef: The reference to record.
        """
        primary = str(body.data) == self._ctx.block_rules[0]
        if previous is None:
            # Only a block held by the format's primary block rule has a body a new entry
            # can be written into: that is the only Lark start symbol.
            return EntryRef(KEY_BLOCK, (tree,), block_nodes=(body,), addable=primary)
        if primary and self._ctx.repeated_blocks == "separate":
            # Another record of the same block. Every occurrence is a real node in the tree,
            # so each stays addable in its own right.
            return EntryRef(KEY_BLOCK, (tree,), block_nodes=(*previous.block_nodes, body), addable=True)
        # Merged: the result is a node the parse tree does not contain, so nothing can be
        # spliced into it and have that reach the file.
        merged = merge_blocks(previous.block_nodes[-1], body)
        return EntryRef(KEY_BLOCK, (tree,), block_nodes=(*previous.block_nodes[:-1], merged), addable=False)

    def key_null(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_null"`` rule nodes.

        A valueless assignment can carry an array qualifier too, and then it sets one
        position of an array the other entries fill in.

        Args:
            tree (Tree): Rule node produced by the ``"key_null"`` grammar rule.
        """
        key = self._get_key(tree)
        if self._index_node(tree) is not None or key in self._slots:
            # A valueless assignment can carry a qualifier too, and then it sets one
            # position of an array the other entries fill in: "v(2) = 3" beside "v(1) ="
            # is [None, 3], not a bare None that discards the 3.
            stored = self._place(key, tree, (None,), (None,))
            if stored is None:
                return
            key = stored
        self._record(key, EntryRef(KEY_NULL, (tree,)))
