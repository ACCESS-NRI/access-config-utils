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

Grammar structure assumptions
------------------------------
All grammars used with this module must follow these conventions:

- Key-value assignments use rules named (or aliased to) ``"key_value"``,
  ``"key_list"``, ``"key_block"``, or ``"key_null"``.
- The rule identifying the key name must be named ``"key"``, must have a
  terminal (e.g. ``CNAME``) as its first child and that terminal's text is the
  key string.
- Value types (e.g. integers, floats, logicals) are expressed as dedicated
  rules whose names are registered in ``VALUE_TYPE_HANDLER_REGISTRY``. Each
  such **value-type rule node** has a single ``Token`` child holding the
  matched text.
- Whitespace that must be preserved for round-trip fidelity is captured in an
  explicit ``ws`` rule rather than discarded with ``%ignore``.
"""

from __future__ import annotations

from typing import Any

from lark import Token, Tree, Visitor
from lark.reconstruct import Reconstructor
from lark.visitors import Interpreter

from access.config.parser_types import VALUE_TYPE_HANDLER_REGISTRY


def update_node_value(rule_node: Tree, value: Any) -> None:
    """Updates the value stored in a value-type rule node.

    The rule node must be a ``Tree`` whose ``.data`` is a key in ``VALUE_TYPE_HANDLER_REGISTRY``
    (e.g. ``"integer"``, ``"float"``). The handler is looked up from that registry.

    Args:
        rule_node (Tree): Value-type rule node to update.
        value (Any): New value.

    Raises:
        TypeError: Raises an exception if the new and old value types do not match.
    """
    data_name: str | None = getattr(rule_node, "data", None)
    handler = VALUE_TYPE_HANDLER_REGISTRY.get(data_name)
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
        ref: A value-type rule node, a list of value-type rule nodes, or a key rule node,
            as stored in ``_refs``.

    Returns:
        Tree: The key rule node (``Tree`` whose ``.data`` starts with ``"key_"``).
    """
    if isinstance(ref, list):
        return ref[0].parent  # type: ignore[attr-defined]
    elif hasattr(ref, "data") and ref.data.startswith("key_"):
        # Note that, although ``key_value`` and ``key_block`` contain ``key_`` in their names, the refs stored for these
        # types are **children** of the key rule node, so their ``.data`` does not start with ``"key_"``.
        return ref
    else:
        return ref.parent  # type: ignore[attr-defined]


class AddParent(Visitor):
    """Lark visitor that annotates every ``Tree`` node with a ``.parent`` back-reference.

    ``Token`` children (terminal leaves) are skipped and only ``Tree`` children (rule nodes)
    receive the ``.parent`` attribute. This enables upward traversal of the parse tree,
    which Lark does not provide natively.
    """

    def __default__(self, tree: Tree) -> None:
        for subtree in tree.children:
            if isinstance(subtree, Tree):
                assert not hasattr(subtree, "parent")
                subtree.parent = tree  # type: ignore


class ConfigToDict(Interpreter):
    """Interpreter to be used by Lark to create a dict holding the config data and the corresponding dict of references
    to rule nodes in the parse tree.

    A Lark ``Transformer`` would be the usual choice, but it replaces rule nodes with transformed values, destroying
    the original tree.  Using an ``Interpreter`` instead lets us retain references to the original rule nodes so that
    they can be mutated later to support round-trip editing.  The ``Interpreter`` also skips visiting child rule nodes
    automatically, so each callback handles an entire key rule subtree in one call.

    While processing blocks, instances of this class need extra information to instantiate a ``Config``. We store that
    extra information as private class arguments.

    Args:
        reconstructor (Reconstructor): Lark reconstructor created from the parser.
        case_sensitive_keys (bool): Are keys case-sensitive?
    """

    _data: dict[str, Any]  # Config data accumulated while traversing the tree.
    _refs: dict[str, list[Tree] | Tree]  # References to rule nodes in the parse tree, keyed by config key.
    # For key_list: a list of value-type rule nodes (one per element).
    # For key_value / key_block / key_null: a single rule node.
    _reconstructor: Reconstructor  # Lark reconstructor.
    _case_sensitive_keys: bool  # Are keys case-sensitive?

    def __init__(self, reconstructor: Reconstructor, case_sensitive_keys: bool) -> None:
        self._reconstructor = reconstructor
        self._case_sensitive_keys = case_sensitive_keys
        super().__init__()

    def visit(self, tree: Tree) -> tuple[dict[str, Any], dict[str, list[Tree] | Tree]]:
        """Visit the entire tree and return two dictionaries: one holding the parsed values and the other holding,
        for each config key, a reference to the corresponding rule node (or list of rule nodes) in the parse tree.

        Args:
            tree (Tree): Root rule node to visit.

        Returns:
            tuple[dict[str, Any], dict[str, list[Tree] | Tree]]: Dict of parsed values, dict of rule node references.
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

        Raises:
            TypeError: If no ``Token`` is found as the first child of the ``"key"`` rule node.

        Returns:
            str: The key name (uppercased if keys are case-insensitive).

        """
        key_rules = [child.children for child in tree.children if child.data == "key"]
        if len(key_rules) == 0:
            raise ValueError("No 'key' rule nodes found among children of key rule node")
        else:
            # Multiple "key" rule nodes are possible (e.g., if the tree is a key_block storing one of more key_values)
            # but the correct one should always be the first one.
            key_rule = key_rules[0]

        # The token holding the key name is the first child of the "key" rule node.
        key_token = key_rule[0]
        if isinstance(key_token, Token):
            key = key_token.value
        else:
            raise TypeError("No key found.")

        if self._case_sensitive_keys:
            return key
        else:
            return key.upper()

    def _transform_values(self, children: list[Tree]) -> tuple[list[Any], list[Tree]]:
        """Given the child nodes of a ``"key_value"`` or ``"key_list"`` rule node, extract and convert the values.

        Filters *children* to those whose ``.data`` is registered in ``VALUE_TYPE_HANDLER_REGISTRY``
        (i.e. value-type rule nodes such as ``"integer"`` or ``"float"``), then reads the ``Token``
        child of each to obtain the matched text and converts it to a Python value.

        Args:
            children (List[Tree]): Child nodes of a ``"key_value"`` or ``"key_list"`` rule node.

        Raises:
            ValueError: If no value-type rule nodes are found among the children.

        Returns:
            Tuple[List[Any], List[Tree]]: List of Python values, list of the corresponding
                value-type rule nodes (for storing in ``_refs``).
        """
        value_rule_nodes = [child for child in children if child.data in VALUE_TYPE_HANDLER_REGISTRY]
        if len(value_rule_nodes) == 0:
            raise ValueError("No values found in Tree")
        values = [VALUE_TYPE_HANDLER_REGISTRY[node.data].from_token(str(node.children[0])) for node in value_rule_nodes]
        return values, value_rule_nodes

    def _transform_value(self, children: list[Tree]) -> tuple[Any, Tree]:
        """Given the child nodes of a ``"key_value"`` rule node, extract and convert the single value.

        Args:
            children (List[Tree]): Child nodes of a ``"key_value"`` rule node.

        Raises:
            ValueError: If more than one value-type rule node is found.

        Returns:
            Tuple[Any, Tree]: The Python value and the corresponding value-type rule node.
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
            if child.data == "block":
                self._data[key] = ConfigImpl(child, self._reconstructor, self._case_sensitive_keys)
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
