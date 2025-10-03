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


def _float_to_str(value: float, token: Token) -> str:
    """Given a float and a Lark token, convert the float to a string using the same notation as used in the token.
    This is to handle cases where the old Fortran notation is used (e.g.: 1.0d10 or 1.0D10).

    Args:
        value (float): float to be converted
        token (Token): Lark token holding the float

    Returns:
        str: the float as a string
    """
    for c in token:
        if c in ["D", "d", "E", "e"]:
            return str(value).replace("e", c)
    else:
        return str(value)


_is_same_type = {
    "logical": lambda value: type(value) is bool,
    "bool": lambda value: type(value) is bool,
    "integer": lambda value: type(value) is int,
    "float": lambda value: type(value) is float,
    "double": lambda value: type(value) is float,
    "complex": lambda value: type(value) is complex,
    "double_complex": lambda value: type(value) is complex,
    "identifier": lambda value: type(value) is str and value.isidentifier(),
    "string": lambda value: type(value) is str,
    "path": lambda value: isinstance(value, Path),
}
"""This dict provides, for each type of value we know about, a lambda function that checks if a value is of the
expected type."""

_value_transformers = {
    "logical": lambda token: str(token).lower() == ".true.",
    "bool": lambda token: str(token) == "True",
    "integer": lambda token: int(token),
    "float": lambda token: float(token),
    "double": lambda token: float(token.replace("D", "E").replace("d", "e")),
    "complex": lambda token: complex(*map(float, token.strip("()").split(","))),
    "double_complex": lambda token: complex(
        *map(float, token.replace("D", "E").replace("d", "e").strip("()").split(","))
    ),
    "identifier": lambda token: str(token),
    "string": lambda token: token[1:-1],
    "path": lambda token: Path(token),
}
"""This dict provides, for each type of value we know about, a lambda function that converts a Lark token into a
variable of the appropriate type."""

_value_inverse_transformers = {
    "logical": lambda value, token: ".true." if value else ".false.",
    "bool": lambda value, token: "True" if value else "False",
    "integer": lambda value, token: value,
    "float": lambda value, token: _float_to_str(value, token),
    "double": lambda value, token: _float_to_str(value, token),
    "complex": lambda value, token: "("
    + _float_to_str(value.real, token)
    + ", "
    + _float_to_str(value.imag, token)
    + ")",
    "double_complex": lambda value, token: "("
    + _float_to_str(value.real, token)
    + ", "
    + _float_to_str(value.imag, token)
    + ")",
    "identifier": lambda value, token: value,
    "string": lambda value, token: token[0] + value + token[-1],
    "path": lambda value, token: str(value),
}
"""This dict provides, for each type of value we know about, a lambda function that updates a Lark token with a given
value. Note that Lark tokens inherit from the str class."""


def _update_node_value(branch: Tree, value: Any) -> None:
    """Updates the value stored in a Lark tree branch.

    The branch should store a value rule of the appropriate type.

    Args:
        branch (Tree): Tree branch to update.
        value (Any): New value.

    Raises:
        TypeError: Raises an exception if the new and old value types do not match.
    """
    if not hasattr(branch, "data") or not _is_same_type[branch.data](value):
        raise TypeError("Trying to change value type")
    transformed_value = _value_inverse_transformers[branch.data](value, branch.children[0])
    branch.children[0] = branch.children[0].update(value=transformed_value)  # type: ignore


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
        super().__init__(data)

    def __getitem__(self, key: str) -> Any:
        """Override method to get item from dict.

        This method takes into account if keys are case-sensitive or not."""
        if self._case_sensitive_keys:
            return super().__getitem__(key)
        else:
            return super().__getitem__(key.upper())

    def __setitem__(self, key: str, value: Any) -> None:
        """Override method to set item from dict.

        This method takes care of updating the parse tree, so that when writing it back into text it will use the new
        values. To make sure this works correctly, we check that the type of the new value is consistent with the
        current type. This method also takes into account if keys are case-sensitive or not."""

        if not self._case_sensitive_keys:
            key = key.upper()

        # Currently we only support replacing existing values, not adding new ones
        if key not in self:
            raise (KeyError(f"Key doesn't exist: {key}"))

        if self[key] is None:
            if value is None:
                return
            else:
                raise TypeError(f"Trying to change the type of variable '{key}'")

        elif isinstance(value, dict):
            raise (SyntaxError("Trying to assign a new value to an entire block"))

        elif isinstance(value, list):
            tree = self._refs[key]
            if not isinstance(tree, list):
                raise TypeError(f"Trying to change the type of variable '{key}'")
            if len(tree) != len(value):
                raise ValueError(f"Trying to change the length of list '{key}'")

            for branch, v in zip(tree, value):
                _update_node_value(branch, v)

        else:
            _update_node_value(self._refs[key], value)

        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Override method to del item from dict."""

        if not self._case_sensitive_keys:
            key = key.upper()

        # Remove item from the dict
        super().__delitem__(key)

        # To update the parse tree, we need to remove the entire branch that defines the removed item.
        # First we get the parent of the branch where the value is stored. This is because each value is a child node of
        # a rule (e.g. the "key_value" rule).
        rule = self._refs[key].parent
        # Next we remove the entire rule from the tree. That is done be removing it from the parent's list of children.
        rule.parent.children.remove(rule)

        # Finally remove reference to the branch storing the value
        del self._refs[key]

    def __str__(self) -> str:
        """Override method to print dict contents to a string."""
        if dict(self) == {}:
            return ""
        else:
            # The reconstructor will modify the tree in-place, so we need to make a deep copy of it beforehand
            tree = self._tree.__deepcopy__(None)

            # Reconstruct
            reconstructed = self._reconstructor.reconstruct(tree)

            # Remove last character if it's a new line (this is to remove the newline character added when parsing the
            # original text)
            return reconstructed[:-1] if reconstructed.endswith("\n") else reconstructed


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
        refs = [child for child in children if child.data in _value_transformers]
        if len(refs) == 0:
            raise ValueError("No values found in Tree")
        values = [_value_transformers[child.data](child.children[0]) for child in refs]
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
