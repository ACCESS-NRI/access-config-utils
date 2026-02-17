# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Classes and utilities to build configuration parsers using Lark.

The classes implemented in this module make a few assumptions about the files to be parsed. The main assumption is that
the contents of the files can be mapped into a Python dictionary, that is, they consist of a series of key-value
assignments. Values can either be scalars, lists, or dictionaries. The supported types of scalars are defined in a
common grammar, in the config.lark file.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lark import Lark, Token, Tree, Visitor
from lark.reconstruct import Reconstructor
from lark.visitors import Interpreter

from access.config.parser_types import VALUE_TYPE_HANDLER_REGISTRY


class ConfigList(list):
    """A list subclass that keeps the parse tree in sync when individual elements are modified.

    When an element is updated via index assignment (e.g., ``config["key"][i] = new_value``), the corresponding
    node in the Lark parse tree is also updated so that round-trip reconstruction reflects the change.

    Args:
        data: The list data.
        refs: List of references to the corresponding parse tree value nodes.
    """

    _refs: list  # References to the value nodes of the parse tree

    def __init__(self, data: list, refs: list) -> None:
        super().__init__(data)
        self._refs = refs

    def __setitem__(self, index, value) -> None:
        """Override to update both the list element(s) and the parse tree node(s).

        Supports both integer indices and slices. When using slices, the number of
        elements assigned must match the number of elements in the slice (i.e. the
        list length cannot change).

        Args:
            index: Integer index or slice of the element(s) to update.
            value: New value (or iterable of values for slices).

        Raises:
            ValueError: If a slice assignment would change the list length.
        """
        if isinstance(index, slice):
            refs_slice = self._refs[index]
            values = list(value)
            if len(values) != len(refs_slice):
                raise ValueError(f"Slice assignment would change list length from {len(refs_slice)} to {len(values)}")
            for ref, v in zip(refs_slice, values, strict=True):
                _update_node_value(ref, v)
            super().__setitem__(index, values)
        else:
            _update_node_value(self._refs[index], value)
            super().__setitem__(index, value)


def _update_node_value(branch: Tree, value: Any) -> None:
    """Updates the value stored in a Lark tree branch.

    The branch should store a value rule of the appropriate type.  Uses the
    ``_value_type_handlers`` registry to look up the appropriate handler.

    Args:
        branch (Tree): Tree branch to update.
        value (Any): New value.

    Raises:
        TypeError: Raises an exception if the new and old value types do not match.
    """
    handler = VALUE_TYPE_HANDLER_REGISTRY.get(getattr(branch, "data", None))
    if handler is None or not handler.type_check(value):
        raise TypeError("Trying to change value type")
    transformed_value = handler.to_token(value, branch.children[0])
    branch.children[0] = branch.children[0].update(value=transformed_value)  # type: ignore


def _find_rule_node(ref: Any) -> Tree:
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
        return ref[0].parent
    elif hasattr(ref, "data") and ref.data.startswith("key_"):
        return ref
    else:
        return ref.parent


class Config(dict):
    """Class inheriting from dict used to store the contents of parsed configuration files.

    For each entry we keep a reference to the corresponding branch in the parse tree so that we can update it when
    changing the contents of the dict. This is then done by overriding the __setitem__ and __delitem__ methods.

    The class also adds support for case-insensitive keys by overriding the appropriate dict methods.

    Args:
        tree (Tree): the parse tree, as returned by Lark.
        reconstructor (Reconstructor): the Lark reconstructor built from the grammar.
        case_sensitive_keys (bool): Are keys case-sensitive?
    """

    _tree: Tree  # The full parse tree
    _refs: dict  # References to the nodes of the parse tree
    _reconstructor: Reconstructor  # Lark reconstrutor used for round-trip parsing
    _case_sensitive_keys: bool  # Are the dict keys case insensitive?

    def __init__(self, tree: Tree, reconstructor: Reconstructor, case_sensitive_keys: bool) -> None:
        self._tree = tree
        self._reconstructor = reconstructor
        self._case_sensitive_keys = case_sensitive_keys
        interpreter = ConfigToDict(reconstructor, case_sensitive_keys)
        data, self._refs = interpreter.visit(self._tree)
        # Wrap list values in ConfigList so that element-level updates keep the parse tree in sync
        for key in data:
            if isinstance(data[key], list):
                data[key] = ConfigList(data[key], self._refs[key])
        super().__init__(data)

    # --- Key normalisation (SRP) ---

    def _normalize_key(self, key: str) -> str:
        """Normalise a key according to the case-sensitivity setting.

        Args:
            key (str): The raw key.

        Returns:
            str: The normalised key.
        """
        return key if self._case_sensitive_keys else key.upper()

    # --- Tree update helpers (SRP) ---

    def _update_list_value(self, key: str, value: list) -> ConfigList:
        """Validate and apply a whole-list replacement to the parse tree.

        Args:
            key (str): The normalised key.
            value (list): The new list of values.

        Returns:
            ConfigList: A ``ConfigList`` wrapping the new values and their tree refs.

        Raises:
            TypeError: If the existing value is not a list.
            ValueError: If the new list has a different length.
        """
        refs = self._refs[key]
        if not isinstance(refs, list):
            raise TypeError(f"Trying to change the type of variable '{key}'")
        if len(refs) != len(value):
            raise ValueError(f"Trying to change the length of list '{key}'")

        for branch, v in zip(refs, value, strict=True):
            _update_node_value(branch, v)

        return ConfigList(value, refs)

    def _reconstruct(self) -> str:
        """Round-trip reconstruct the parse tree into its text representation.

        Returns:
            str: The reconstructed text (trailing newline stripped).
        """
        # The reconstructor modifies the tree in-place, so work on a deep copy
        tree = self._tree.__deepcopy__(None)
        reconstructed = self._reconstructor.reconstruct(tree)
        return reconstructed[:-1] if reconstructed.endswith("\n") else reconstructed

    # --- dict overrides ---

    def __getitem__(self, key: str) -> Any:
        """Override method to get item from dict.

        This method takes into account if keys are case-sensitive or not."""
        return super().__getitem__(self._normalize_key(key))

    def __setitem__(self, key: str, value: Any) -> None:
        """Override method to set item from dict.

        This method takes care of updating the parse tree, so that when writing it back into text it will use the new
        values. To make sure this works correctly, we check that the type of the new value is consistent with the
        current type. This method also takes into account if keys are case-sensitive or not."""

        key = self._normalize_key(key)

        # Currently we only support replacing existing values, not adding new ones
        if key not in self:
            raise KeyError(f"Key doesn't exist: {key}")

        if self[key] is None:
            if value is None:
                return
            else:
                raise TypeError(f"Trying to change the type of variable '{key}'")

        elif isinstance(value, dict):
            raise SyntaxError("Trying to assign a new value to an entire block")

        elif isinstance(value, list):
            value = self._update_list_value(key, value)

        else:
            _update_node_value(self._refs[key], value)

        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Override method to del item from dict."""

        key = self._normalize_key(key)

        # Remove item from the dict
        super().__delitem__(key)

        # Remove the corresponding rule from the parse tree
        rule = _find_rule_node(self._refs[key])
        rule.parent.children.remove(rule)

        # Finally remove reference to the branch storing the value
        del self._refs[key]

    def __str__(self) -> str:
        """Override method to print dict contents to a string."""
        if dict(self) == {}:
            return ""
        return self._reconstruct()


class AddParent(Visitor):
    """Lark visitor that adds to every node in the tree a reference to its parent."""

    def __default__(self, tree):
        for subtree in tree.children:
            if isinstance(subtree, Tree):
                assert not hasattr(subtree, "parent")
                subtree.parent = tree  # type: ignore


class ConfigParser(ABC):
    """Lark-based configuration parser base class.

    The parsers built by extending this class are meant for files that share a common structure. In this case, the
    files must be made by a series of key-value assignments that can be mapped onto a Python dict. Each key-value
    assignment can therefore be one of three types:
      - scalar (e.g, 'a=1')
      - list/array (e.g., 'a=1,2,3')
      - block/dict containing other key-value assignments (e.g., 'blk: b=1, c=2')

    Because the resulting parse trees are all processed using the ConfigToDict Interpreter, all grammars must follow
    the same structure and use the same names for the relevant rules:
      - Key-value assignment rules must be named (or have an alias with that name): "key_value", "key_list" and
        "key_block".
      - Only rules from the "config.lark" file should be used when defining the supported scalar values in the
        assignment rules.
      - The rule defining what a key is must be named "key". Note that the "config.lark" file contains a "key" rule
        that should work for most cases.
      - Empty assignments (e.g., 'a=') are supported and the corresponding rule must be named "key_null".

    This class is made abstract to prevent instantiation, as it requires a Lark grammar to be provided in order to work
    correctly.
    """

    @property
    @abstractmethod
    def grammar(self) -> str:
        """The grammar is a property of the parser.

        Returns:
            str: The parser grammar.
        """

    @property
    @abstractmethod
    def case_sensitive_keys(self) -> bool:
        """Property indicating if the configuration uses case-sensitive keys or not.

        Returns:
            bool: Are the keys case-sensitive?
        """

    def parse(self, stream) -> Config:
        """Parse the given text.

        Args:
            stream (str): The text to parse.

        Returns:
            Config: instance of the Config class storing the parsed data.
        """
        parser = Lark(self.grammar, import_paths=[Path(__file__).parent], maybe_placeholders=False)

        # Parse text. Here we add a newline character to simplify the writting of the grammars, as otherwise one would
        # have to explicitly take into account the case where the text does no end with a newline.
        tree = parser.parse(stream + "\n")

        AddParent().visit(tree)
        reconstructor = Reconstructor(parser)
        return Config(tree, reconstructor, self.case_sensitive_keys)


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

    _data: dict  # Private dictionary used to store the config data while traversing the tree.
    _refs: dict  # Private dictionary used to store the references while traversing the tree.
    _reconstructor: Reconstructor  # Lark reconstructor.
    _case_sensitive_keys: bool  # Are keys case-sensitive?

    def __init__(self, reconstructor: Reconstructor, case_sensitive_keys: bool) -> None:
        self._reconstructor = reconstructor
        self._case_sensitive_keys = case_sensitive_keys
        super().__init__()

    def visit(self, tree: Tree) -> tuple[dict[str, Any], dict[str, Tree]]:
        """Visit the entire tree and return two dictionaries: one holding the parsed items and the other one holding,
        for each parsed item, a reference to the corresponding tree branch.

        Args:
            tree (Tree): Tree to visit.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Tree]]: Dict holding the parsed values, dict holding the references to the
            branches.
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
        values = [VALUE_TYPE_HANDLER_REGISTRY[child.data].from_token(child.children[0]) for child in refs]
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
        key = self._get_key(tree)
        for child in tree.children:
            if child.data == "block":
                self._data[key] = Config(child, self._reconstructor, self._case_sensitive_keys)
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
