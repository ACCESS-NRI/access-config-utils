# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Reading a parse tree into the values it holds, and the nodes those values live in.

This is the one place that turns a tree into data. It walks a container rule node and, for
every entry below it, records the Python value and a reference back to the node holding it,
so that assigning to a key later can mutate exactly that node.

Every rule name looked for here -- the four entry categories, ``key``, ``block`` -- comes
from ``grammar_contract``, which states in full what a grammar has to provide and why. The
shapes relied on follow from it: a key rule node has a ``key`` child whose first child is
the ``Token`` holding the key text, and a value-type rule node has a single ``Token`` child
holding the value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lark import Token, Tree
from lark.visitors import Interpreter

from access.config.grammar_contract import BLOCK_RULE, ENTRY_CATEGORIES, KEY_RULE
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY

if TYPE_CHECKING:
    from access.config.grammar_compiled import ParseContext


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
