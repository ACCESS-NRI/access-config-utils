# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Functions and classes for manipulating Lark parse trees.

This module contains the low-level tree operations used by the configuration
parser: updating node values, locating rule nodes, annotating parent references,
and converting a parse tree into a dictionary of values and references.
"""

from __future__ import annotations

from typing import Any

from lark import Token, Tree, Visitor
from lark.reconstruct import Reconstructor
from lark.visitors import Interpreter

from access.config.parser_types import VALUE_TYPE_HANDLER_REGISTRY


def update_node_value(branch: Tree, value: Any) -> None:
    """Updates the value stored in a Lark tree branch.

    The branch should store a value rule of the appropriate type.  Uses the
    ``VALUE_TYPE_HANDLER_REGISTRY`` to look up the appropriate handler.

    Args:
        branch (Tree): Tree branch to update.
        value (Any): New value.

    Raises:
        TypeError: Raises an exception if the new and old value types do not match.
    """
    data_name: str | None = getattr(branch, "data", None)
    if data_name is None:
        raise TypeError("Trying to change value type")
    handler = VALUE_TYPE_HANDLER_REGISTRY.get(data_name)
    if handler is None or not handler.type_check(value):
        raise TypeError("Trying to change value type")
    token = branch.children[0]
    assert isinstance(token, Token)
    transformed_value = handler.to_token(value, str(token))
    branch.children[0] = token.update(value=transformed_value)


def find_rule_node(ref: list[Tree] | Tree) -> Tree:
    """Given a parse-tree reference for a key, return the corresponding rule node.

    Different rule types store refs differently:
        - ``key_list``: *ref* is a list of value nodes; the rule is the parent of any element.
        - ``key_null``: *ref* is the rule node itself.
        - ``key_value`` / ``key_block``: *ref* is a child node; the rule is its parent.

        Args:
            ref: The reference to find the rule for.

        Returns:
            Tree: The rule node.
    """
    if isinstance(ref, list):
        return ref[0].parent  # type: ignore[attr-defined]
    elif hasattr(ref, "data") and ref.data.startswith("key_"):
        return ref
    else:
        return ref.parent  # type: ignore[attr-defined]


class AddParent(Visitor):
    """Lark visitor that adds to every node in the tree a reference to its parent."""

    def __default__(self, tree: Tree) -> None:
        for subtree in tree.children:
            if isinstance(subtree, Tree):
                assert not hasattr(subtree, "parent")
                subtree.parent = tree  # type: ignore


class ConfigToDict(Interpreter):
    """Interpreter to be used by Lark to create a dict holding the config data and the corresponding dict of references
    to the branches of the parse tree.

    When using Lark, the usual way to transform the parse tree into something else is to use a Transformer.
    Here we use an Interpreter instead, as this allows us to create a dict of references to the branches of the
    original tree. The Interpreter will also skip visiting sub-branches, allowing us to handle entire branches in a
    single function.

    While processing blocks, instances of this class need extra information to instantiate a ``Config``. We store that
    extra information as private class arguments.

    Args:
        reconstructor (Reconstructor): Lark reconstructor created from the parser.
        case_sensitive_keys (bool): Are keys case-sensitive?
    """

    _data: dict[str, Any]  # Private dictionary used to store the config data while traversing the tree.
    _refs: dict[str, list[Tree] | Tree]  # Private dictionary used to store the references while traversing the tree.
    _reconstructor: Reconstructor  # Lark reconstructor.
    _case_sensitive_keys: bool  # Are keys case-sensitive?

    def __init__(self, reconstructor: Reconstructor, case_sensitive_keys: bool) -> None:
        self._reconstructor = reconstructor
        self._case_sensitive_keys = case_sensitive_keys
        super().__init__()

    def visit(self, tree: Tree) -> tuple[dict[str, Any], dict[str, list[Tree] | Tree]]:
        """Visit the entire tree and return two dictionaries: one holding the parsed items and the other one holding,
        for each parsed item, a reference to the corresponding tree branch.

        Args:
            tree (Tree): Tree to visit.

        Returns:
            tuple[dict[str, Any], dict[str, list[Tree] | Tree]]: Dict holding the parsed values, dict holding the
            references to the branches.
        """
        self._data = {}
        self._refs = {}
        super().visit(tree)
        return self._data, self._refs

    def _get_key(self, tree: Tree) -> str:
        """Given a tree, look for the token storing a key name and return it.

        Args:
            tree (Tree): Lark tree storing a "key" rule.

        Raises:
            TypeError: If no key is found.

        Returns:
            str: The key.

        """
        key_node = [child.children[0] for child in tree.children if child.data == "key"][0]
        if isinstance(key_node, Token):
            key = key_node.value
        else:
            raise TypeError("No key found.")

        if self._case_sensitive_keys:
            return key
        else:
            return key.upper()

    def _transform_values(self, children: list[Tree]) -> tuple[list[Any], list[Tree]]:
        """Given the children of a "key_value" or a "key_list" rule, extract and transform the corresponding values.

        Args:
            children (List[Tree]): List of Lark trees containing the values to process. These should be the children
                of a "key_value" or a "key_list" rule.

        Raises:
            ValueError: If no values are found.

        Returns:
            Tuple[List[Any], List[Tree]]: List of transformed values, list of tree branches storing the corresponding
                values.
        """
        refs = [child for child in children if child.data in VALUE_TYPE_HANDLER_REGISTRY]
        if len(refs) == 0:
            raise ValueError("No values found in Tree")
        values = [VALUE_TYPE_HANDLER_REGISTRY[child.data].from_token(str(child.children[0])) for child in refs]
        return values, refs

    def _transform_value(self, children: list[Tree]) -> tuple[Any, Tree]:
        """Given the children of a "key_value" rule, extract and transform the corresponding value.

        Args:
            children (List[Tree]): List of Lark branches containing the value to process. These should be the children
                of a "key_value" rule.

        Raises:
            ValueError: If more than one value is found.

        Returns:
            Tuple[Any, Tree]: Transformed value, tree branch storing the corresponding value.
        """
        values, refs = self._transform_values(children)
        if len(refs) > 1:
            raise ValueError("More than one value found in Tree")
        return values[0], refs[0]

    def key_list(self, tree: Tree) -> None:
        """Function to process "key_list" rules.

        Args:
            tree (Tree): Lark tree storing a "key_list" rule.
        """
        key = self._get_key(tree)
        self._data[key], self._refs[key] = self._transform_values(tree.children)

    def key_value(self, tree: Tree) -> None:
        """Function to process "key_value" rules.

        Args:
            tree (Tree): Lark tree storing a "key_value" rule.
        """
        key = self._get_key(tree)
        self._data[key], self._refs[key] = self._transform_value(tree.children)

    def key_block(self, tree: Tree) -> None:
        """Function to process "key_block" rules.

        Args:
            tree (Tree): Lark tree storing a "key_block" rule.
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
        """Function to process "key_null" rules.

        Args:
            tree (Tree): Lark tree storing a "key_null" rule.
        """
        key = self._get_key(tree)
        self._data[key] = None
        self._refs[key] = tree
