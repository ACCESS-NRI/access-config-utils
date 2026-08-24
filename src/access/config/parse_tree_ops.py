# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Functions and classes for manipulating Lark parse trees.

This module contains the low-level tree operations used by the configuration
parser: updating node values, locating rule nodes, annotating parent references,
and converting a parse tree into a dictionary of values and references.

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

Every rule name this module looks for -- the four entry categories, ``key``,
``block``, ``ws`` -- comes from ``grammar_contract``, which states in full what a
grammar has to provide and why. The shapes relied on here follow from it: a key
rule node has a ``key`` child whose first child is the ``Token`` holding the key
text, and a value-type rule node has a single ``Token`` child holding the value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from lark import Token, Tree, Visitor
from lark.visitors import Interpreter

from access.config.entry_synthesis import contains_entry, is_comment_line
from access.config.grammar_contract import BLOCK_RULE, ENTRY_CATEGORIES, KEY_RULE
from access.config.grammar_info import GrammarInfo
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY

if TYPE_CHECKING:
    from lark import Lark

    from access.config.parser import ParseContext


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


def find_rule_node(ref: list[Tree] | Tree) -> Tree:
    """Given a parse-tree reference for a key, return the corresponding key rule node.

    References are stored differently depending on the key rule type:

    - ``key_list``: *ref* is a list of value-type rule nodes (one per list element);
      the key rule node is the ``.parent`` of any element.
    - ``key_null``: *ref* is the key rule node itself.
    - ``key_value`` / ``key_block``: *ref* is a value-type rule node (child of the key
      rule node); the key rule node is its ``.parent``.

    Args:
        ref (list[Tree] | Tree): A value-type rule node, a list of value-type rule nodes,
            or a key rule node, as stored in ``_refs``.

    Returns:
        Tree: The key rule node (``Tree`` whose ``.data`` is an entry category).
    """
    if isinstance(ref, list):
        return ref[0].parent  # type: ignore[attr-defined]
    elif hasattr(ref, "data") and ref.data in ENTRY_CATEGORIES:
        # Only a ``key_null`` ref is the key rule node itself. For ``key_value`` and
        # ``key_block`` the ref is a **child** of that node -- a value-type rule node or a
        # ``block`` -- so its ``.data`` is never an entry category.
        return ref
    else:
        return ref.parent  # type: ignore[attr-defined]


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


class AddParent(Visitor):
    """Lark visitor that annotates every ``Tree`` node with a ``.parent`` back-reference.

    ``Token`` children (terminal leaves) are skipped and only ``Tree`` children (rule nodes)
    receive the ``.parent`` attribute. This enables upward traversal of the parse tree,
    which Lark does not provide natively.

    Only ever apply this to a freshly parsed tree. Visiting a tree that already carries the
    attribute trips the assertion below, so a new entry is annotated on its own before
    being spliced in, never by re-visiting the tree it is spliced into. Note also that
    ``__delitem__`` does not clear ``.parent`` on the node it removes, and that the
    assertion is a development aid which ``python -O`` strips.
    """

    def __default__(self, tree: Tree) -> None:
        for subtree in tree.children:
            if isinstance(subtree, Tree):
                assert not hasattr(subtree, "parent")
                subtree.parent = tree  # type: ignore


class ConfigToDict(Interpreter):
    """Interpreter to be used by Lark to create a dict holding the config data and the
    corresponding dict of references to rule nodes in the parse tree.

    A Lark ``Transformer`` would be the usual choice, but it replaces rule nodes with
    transformed values, destroying the original tree. Using an ``Interpreter`` instead lets
    us retain references to the original rule nodes so that they can be mutated later to
    support round-trip editing. The ``Interpreter`` also skips visiting child rule nodes
    automatically, so each callback handles an entire key rule subtree in one call.

    While processing blocks, instances of this class need extra information to instantiate
    a ``Config``. That information is carried by the parse context.

    Note that visiting a tree builds a *new* ``Config`` for every block it contains, so
    re-visiting a whole tree would leave any previously handed-out block ``Config`` detached
    from the dict. Insertion therefore visits only the newly parsed subtree and merges the
    result.

    Args:
        ctx (ParseContext): The compiled grammar and the per-parser settings.
    """

    _data: dict[str, Any]  # Config data accumulated while traversing the tree.
    _refs: dict[str, list[Tree] | Tree]  # References to rule nodes in the parse tree, keyed by config key.
    # For key_list: a list of value-type rule nodes (one per element).
    # For key_value / key_block / key_null: a single rule node.
    _ctx: ParseContext  # Compiled grammar and per-parser settings.

    def __init__(self, ctx: ParseContext) -> None:
        self._ctx = ctx
        super().__init__()

    def visit(self, tree: Tree) -> tuple[dict[str, Any], dict[str, list[Tree] | Tree]]:
        """Visit the entire tree and return the parsed values and their tree references.

        The first dictionary holds the parsed values; the second holds, for each config
        key, a reference to the corresponding rule node (or list of rule nodes) in the
        parse tree.

        Args:
            tree (Tree): Root rule node to visit.

        Returns:
            tuple[dict[str, Any], dict[str, list[Tree] | Tree]]: Dict of parsed values,
                dict of rule node references.
        """
        self._data = {}
        self._refs = {}
        super().visit(tree)
        return self._data, self._refs

    def _get_key(self, tree: Tree) -> str:
        """Given a key rule node, extract and return the key name.

        Finds the ``"key"`` rule node among *tree*'s children, then reads the ``Token``
        (terminal) that is its first child and that token's text is the key name.

        Args:
            tree (Tree): A key rule node (e.g. ``Tree`` with ``.data == "key_value"``).

        Returns:
            str: The key name (uppercased if keys are case-insensitive).

        Raises:
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
        if isinstance(key_token, Token):
            key = key_token.value
        else:
            raise TypeError("No key found.")

        if self._ctx.case_sensitive_keys:
            return key
        else:
            return key.upper()

    def _transform_values(self, children: list[Tree]) -> tuple[list[Any], list[Tree]]:
        """Given the children of a ``key_value`` or ``key_list`` node, convert the values.

        Filters *children* to those whose ``.data`` is registered in
        ``VALUE_TYPE_HANDLER_REGISTRY`` (i.e. value-type rule nodes such as ``"integer"``
        or ``"float"``), then reads the ``Token`` child of each to obtain the matched text
        and converts it to a Python value.

        Args:
            children (list[Tree]): Child nodes of a ``"key_value"`` or ``"key_list"``
                rule node.

        Returns:
            tuple[list[Any], list[Tree]]: List of Python values, list of the corresponding
                value-type rule nodes (for storing in ``_refs``).

        Raises:
            ValueError: If no value-type rule nodes are found among the children.
        """
        value_rule_nodes = [child for child in children if child.data in VALUE_TYPE_HANDLER_REGISTRY]
        if len(value_rule_nodes) == 0:
            raise ValueError("No values found in Tree")
        values = [VALUE_TYPE_HANDLER_REGISTRY[node.data].from_token(str(node.children[0])) for node in value_rule_nodes]
        return values, value_rule_nodes

    def _transform_value(self, children: list[Tree]) -> tuple[Any, Tree]:
        """Given the children of a ``"key_value"`` rule node, convert the single value.

        Args:
            children (list[Tree]): Child nodes of a ``"key_value"`` rule node.

        Returns:
            tuple[Any, Tree]: The Python value and the corresponding value-type rule node.

        Raises:
            ValueError: If more than one value-type rule node is found.
        """
        values, value_rule_nodes = self._transform_values(children)
        if len(value_rule_nodes) > 1:
            raise ValueError("More than one value found in Tree")
        return values[0], value_rule_nodes[0]

    def key_list(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_list"`` rule nodes.

        Args:
            tree (Tree): Rule node produced by the ``"key_list"`` grammar rule.
        """
        key = self._get_key(tree)
        self._data[key], self._refs[key] = self._transform_values(tree.children)

    def key_value(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_value"`` rule nodes.

        Args:
            tree (Tree): Rule node produced by the ``"key_value"`` grammar rule.
        """
        key = self._get_key(tree)
        self._data[key], self._refs[key] = self._transform_value(tree.children)

    def key_block(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_block"`` rule nodes.

        Args:
            tree (Tree): Rule node produced by the ``"key_block"`` grammar rule.
        """
        # Import here to avoid circular dependency (Config -> ConfigToDict -> Config)
        from access.config.parser import Config as ConfigImpl

        key = self._get_key(tree)
        for child in tree.children:
            if child.data == BLOCK_RULE:
                self._data[key] = ConfigImpl(child, self._ctx)
                self._refs[key] = child
                return
            else:
                pass

    def key_null(self, tree: Tree) -> None:
        """Interpreter callback for ``"key_null"`` rule nodes.

        Args:
            tree (Tree): Rule node produced by the ``"key_null"`` grammar rule.
        """
        key = self._get_key(tree)
        self._data[key] = None
        self._refs[key] = tree
