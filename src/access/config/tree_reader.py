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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from lark import Token, Tree
from lark.visitors import Interpreter

from access.config.grammar_contract import KEY_BLOCK, KEY_LIST, KEY_NULL, KEY_RULE, KEY_VALUE
from access.config.tree_navigation import is_value_node, node_values

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
        value_nodes (tuple[Tree, ...]): The value-type rule nodes holding the entry's
            values, in order. One for ``key_value``, one per element for ``key_list``,
            none for ``key_block`` and ``key_null``. These come from the last entry, the
            one whose value the configuration holds.
        block_node (Tree | None): The ``block`` node holding the entry's contents, for a
            ``key_block``; ``None`` otherwise.
        addable (bool): Whether a new entry can be spliced into *block_node*. False for a
            block merged from several in the file, whose node is not itself in the parse
            tree, and for one held by a rule that is not a Lark start symbol.
    """

    category: str
    entry_nodes: tuple[Tree, ...]
    value_nodes: tuple[Tree, ...] = ()
    block_node: Tree | None = None
    addable: bool = True


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
    # A repeat node appears once per element it covers, so reading it once per appearance
    # would multiply the values. Take each distinct node's values in order instead.
    values: list[Any] = []
    for index, node in enumerate(ref.value_nodes):
        if index == 0 or node is not ref.value_nodes[index - 1]:
            values.extend(node_values(node))
    return values if ref.category == KEY_LIST else values[0]


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
        distinct = {id(node) for node in nodes}
        if len(distinct) > 1:
            raise ValueError("More than one value found in Tree")
        # "x = 3*1" is written where a single value goes but means three, so it reads as a
        # list: the category follows the number of values, not the rule that wrote them.
        category = KEY_VALUE if len(nodes) == 1 else KEY_LIST
        self._record(self._get_key(tree), EntryRef(category, (tree,), nodes))

    def key_list(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_list"`` rule nodes.

        Args:
            tree (Tree): Rule node produced by the ``"key_list"`` grammar rule.
        """
        self._record(self._get_key(tree), EntryRef(KEY_LIST, (tree,), self._value_nodes(tree)))

    def key_block(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_block"`` rule nodes.

        A key can name more than one block. A derived type spells one component per line, so
        ``a%b = 1`` and ``a%c = 2`` are two entries under the single key ``A``, and the same
        happens when a file repeats a namelist group name. The blocks are merged rather than
        the last one winning, since dropping the earlier ones loses data silently.

        Args:
            tree (Tree): Rule node produced by the ``"key_block"`` grammar rule.

        Raises:
            ValueError: If the key was already given a value. Merging would build a block
                out of a value node and quietly drop the value.
        """
        key = self._get_key(tree)
        for child in tree.children:
            if isinstance(child, Tree) and str(child.data) in self._ctx.block_rules:
                previous = self._refs.get(key)
                if previous is not None and previous.category != KEY_BLOCK:
                    raise ValueError(f"'{key}' is assigned a value and used as a block")
                merged = child if previous is None else merge_blocks(previous.block_node, child)
                # Only an unmerged block held by the format's primary block rule has a body
                # a new entry can be written into: that is the only Lark start symbol.
                addable = merged is child and str(child.data) == self._ctx.block_rules[0]
                self._record(key, EntryRef(KEY_BLOCK, (tree,), block_node=merged, addable=addable))
                return

    def key_null(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_null"`` rule nodes.

        Args:
            tree (Tree): Rule node produced by the ``"key_null"`` grammar rule.
        """
        self._record(self._get_key(tree), EntryRef(KEY_NULL, (tree,)))
