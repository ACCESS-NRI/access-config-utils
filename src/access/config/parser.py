# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Classes and utilities to build configuration parsers using Lark.

The classes implemented in this module make a few assumptions about the files to be
parsed. The main assumption is that the contents of the files can be mapped into a Python
dictionary, that is, they consist of a series of key-value assignments. Values can either
be scalars, lists, or dictionaries. The supported types of scalars are defined in a common
grammar, in the config.lark file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, SupportsIndex

from lark import Tree
from lark.exceptions import UnexpectedInput

from access.config.config_insert import insert_entry
from access.config.entry_style import EntryStyle, probe_entry_style, probe_sibling_block_style
from access.config.grammar_compiled import ParseContext, compile_grammar
from access.config.grammar_contract import (
    KEY_BLOCK,
    KEY_LIST,
    KEY_VALUE,
    START_RULE,
)
from access.config.grammar_values import VALUE_RULE_PRIORITY
from access.config.tree_edits import (
    remove_entry_node,
    update_node_value,
)
from access.config.tree_navigation import AddParent
from access.config.tree_reader import EntryRef, entry_value, read_entries


class ConfigList(list):
    """A list subclass keeping the parse tree in sync when individual elements change.

    When an element is updated via index assignment (e.g. ``config["key"][i] = new_value``),
    the corresponding value-type rule node in the Lark parse tree is also updated so that
    round-trip reconstruction reflects the change.

    Each element is paired with the parse tree node holding it, so only operations that
    preserve the elements and their order are supported. Those that would add, remove or
    reorder them (``append``, ``extend``, ``insert``, ``remove``, ``pop``, ``clear``,
    ``sort``, ``reverse``, ``del`` and ``+=``) raise ``NotImplementedError`` rather than
    silently leaving the list and the file disagreeing. Assign a whole new list instead
    (``config["key"] = [...]``), which rewrites every element.

    Args:
        data (list[Any]): The list data.
        refs (list[Tree]): References to the value-type rule nodes, one per list element.
    """

    _refs: list[Tree]  # Value-type rule nodes from the parse tree, one per list element.

    def __init__(self, data: list[Any], refs: list[Tree]) -> None:
        super().__init__(data)
        self._refs = refs

    def __setitem__(self, index: SupportsIndex | slice, value: Any) -> None:
        """Override to update both the list element(s) and the parse tree node(s).

        Supports both integer indices and slices. When using slices, the number of
        elements assigned must match the number of elements in the slice (i.e. the
        list length cannot change).

        Args:
            index (SupportsIndex | slice): Integer index or slice of the element(s) to
                update.
            value (Any): New value (or iterable of values for slices). The type of each
                value must match the type already stored in the corresponding value-type
                rule node.

        Raises:
            ValueError: If a slice assignment would change the list length.
        """
        if isinstance(index, slice):
            refs_slice = self._refs[index]
            values = list(value)
            if len(values) != len(refs_slice):
                raise ValueError(f"Slice assignment would change list length from {len(refs_slice)} to {len(values)}")
            for ref, v in zip(refs_slice, values, strict=True):
                update_node_value(ref, v)
            super().__setitem__(index, values)
        else:
            update_node_value(self._refs[index], value)
            super().__setitem__(index, value)

    def _unsupported(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject a list operation the parse tree cannot follow.

        Args:
            *args (Any): Ignored.
            **kwargs (Any): Ignored.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Adding, removing or reordering the elements of a configuration list is not "
            "supported; assign a whole new list to the key instead"
        )

    # Every operation that would change the elements or their order, and so break the
    # pairing with the parse tree nodes holding them.
    append = _unsupported  # type: ignore[assignment]
    extend = _unsupported  # type: ignore[assignment]
    insert = _unsupported  # type: ignore[assignment]
    remove = _unsupported  # type: ignore[assignment]
    pop = _unsupported  # type: ignore[assignment]
    clear = _unsupported  # type: ignore[assignment]
    sort = _unsupported  # type: ignore[assignment]
    reverse = _unsupported  # type: ignore[assignment]
    __delitem__ = _unsupported  # type: ignore[assignment]
    __iadd__ = _unsupported  # type: ignore[assignment]
    __imul__ = _unsupported  # type: ignore[assignment]


class Config(dict):
    """Class inheriting from dict used to store the contents of parsed configuration files.

    For each entry we keep a reference to the corresponding rule node in the parse tree so
    that we can update it when changing the contents of the dict. This is then done by
    overriding the __setitem__ and __delitem__ methods.

    The class also adds support for case-insensitive keys by overriding the appropriate
    dict methods.

    Args:
        tree (Tree): The parse tree, as returned by Lark. Its root is the container that new
            entries are added to.
        ctx (ParseContext): The compiled grammar and the per-parser settings.
    """

    _tree: Tree  # The full parse tree
    _refs: dict[str, EntryRef]  # Where each key's entry lives in the parse tree
    _ctx: ParseContext  # Compiled grammar and per-parser settings
    # Whitespace for a new entry when this container holds none to copy from. Set on a block
    # created by an assignment, which is empty and so has no style of its own yet.
    _fallback_style: EntryStyle

    def __init__(self, tree: Tree, ctx: ParseContext) -> None:
        self._tree = tree
        self._ctx = ctx
        self._fallback_style = EntryStyle()
        self._refs = read_entries(tree, ctx)
        super().__init__({key: self._wrap(ref) for key, ref in self._refs.items()})

    def _wrap(self, ref: EntryRef) -> Any:
        """Return the object standing for an entry in the dict.

        A list and a block are both handed out as something that keeps the parse tree in
        step: a ``ConfigList`` pairs each element with the node holding it, and a nested
        ``Config`` does the same for the entries of a block.

        Args:
            ref (EntryRef): The entry to represent.

        Returns:
            Any: The scalar, a ``ConfigList``, a nested ``Config``, or ``None``.
        """
        if ref.category == KEY_BLOCK:
            assert ref.block_node is not None
            return Config(ref.block_node, self._ctx)
        value = entry_value(ref)
        return ConfigList(value, list(ref.value_nodes)) if ref.category == KEY_LIST else value

    # --- Tree update helpers (SRP) ---

    def _update_list_value(self, key: str, value: list[Any]) -> ConfigList:
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
        nodes = self._refs[key].value_nodes
        if self._refs[key].category != KEY_LIST:
            raise TypeError(f"Trying to change the type of variable '{key}'")
        if len(nodes) != len(value):
            raise ValueError(f"Trying to change the length of list '{key}'")

        for rule_node, v in zip(nodes, value, strict=True):
            update_node_value(rule_node, v)

        return ConfigList(value, list(nodes))

    # --- Key addition (SRP) ---

    def _insert_new_key(self, key: str, raw_key: str, value: Any) -> None:
        """Add an entry for a key that is not in the configuration yet.

        The entry itself is written by ``insert_entry``; what is decided here is the
        whitespace it should copy, and what has to happen afterwards to the dict.

        The whitespace comes from the neighbours in this container, or -- when there are
        none, as in a block just created -- from the style handed to the block when it was
        created. A block is filled through this same path, one key at a time, so nesting to
        any depth works.

        Args:
            key (str): The normalised key, used for the dict entry.
            raw_key (str): The key as the caller wrote it, used for the file text.
            value (Any): The value to store.

        Raises:
            ValueError: If *raw_key* is not a usable key name, or *value* is a list too
                short for the format to express as one.
            UnsupportedEntryError: If no candidate produced the requested value.
        """
        container = self._tree
        style = probe_entry_style(container)
        if not style.has_donor:
            style = self._fallback_style

        ref = insert_entry(container, self._ctx, key, raw_key, value, style)
        self._refs[key] = ref
        stored = self._wrap(ref)
        dict.__setitem__(self, key, stored)

        if ref.category == KEY_BLOCK:
            # The new block is empty, so nothing in it can say how its contents should be
            # laid out. Hand it the style of the nearest sibling block, falling back to this
            # container's own, so that a block nested in a new block inherits it too rather
            # than dropping back to the canonical layout.
            assert isinstance(stored, Config)
            sibling = probe_sibling_block_style(container)
            stored._fallback_style = sibling if sibling.has_donor else self._fallback_style

            for block_key, block_value in value.items():
                stored[block_key] = block_value

    def _reconstruct(self) -> str:
        """Round-trip reconstruct the parse tree into its text representation.

        Returns:
            str: The reconstructed text (trailing newline stripped).
        """
        # The reconstructor modifies the tree in-place, so work on a deep copy
        tree = self._tree.__deepcopy__(None)
        reconstructed = self._ctx.reconstructor.reconstruct(tree)
        return reconstructed[:-1] if reconstructed.endswith("\n") else reconstructed

    # --- dict overrides ---

    def __getitem__(self, key: str) -> Any:
        """Override method to get item from dict.

        This method takes into account if keys are case-sensitive or not.
        """
        return super().__getitem__(self._ctx.normalise_key(key))

    def __setitem__(self, key: str, value: Any) -> None:
        """Override method to set item from dict.

        This method takes care of updating the parse tree, so that when writing it back
        into text it will use the new values. To make sure this works correctly, we check
        that the type of the new value is consistent with the current type. This method
        also takes into account if keys are case-sensitive or not.

        A key that is not present yet is added, provided the format can express it: a
        scalar, a list of two or more values, ``None`` where the format allows a valueless
        assignment, or a ``dict`` to create a block and fill it. The new entry is written
        after the last existing one, indented to match it. A block created this way has no
        entry of its own to match, so its contents follow the nearest sibling block instead.

        Args:
            key (str): The key to assign to. For a new key, the name is written into the
                file as given, while the dict entry uses the normalised form.
            value (Any): The new value.

        Raises:
            TypeError: If the new value's type does not match the existing one, or -- as
                ``UnsupportedEntryError`` -- if the format cannot express the new value.
            ValueError: If the length of an existing list would change, or a new list has
                fewer than two elements, or a new key name is unusable.
            SyntaxError: If a ``dict`` is assigned to a key that already exists.
        """

        raw_key = key
        key = self._ctx.normalise_key(key)

        if key not in self:
            self._insert_new_key(key, raw_key, value)
            return

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
            ref = self._refs[key]
            if ref.category != KEY_VALUE:
                raise TypeError(f"Trying to change the type of variable '{key}'")
            update_node_value(ref.value_nodes[0], value)

        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Override method to del item from dict."""

        key = self._ctx.normalise_key(key)

        # Remove item from the dict
        super().__delitem__(key)

        # Remove the entry from the parse tree. Doing so can leave the nodes either side
        # of it adjacent, in a shape the grammar cannot derive, which remove_entry_node
        # repairs.
        remove_entry_node(self._refs[key].entry_node, self._ctx.info)

        # Remove the rule node reference
        del self._refs[key]

    # --- Remaining mapping protocol ---
    #
    # ``dict`` implements these in C without going through ``__getitem__``,
    # ``__setitem__`` or ``__delitem__``, so inheriting them would skip both key
    # normalisation and the parse tree update. That is not merely inconvenient: a ``pop``
    # followed by an assignment would leave the removed entry in the tree and then add a
    # second one, writing a file with a duplicate key.

    def __contains__(self, key: object) -> bool:
        """Override method to test key membership, honouring case-insensitive keys."""
        if isinstance(key, str):
            key = self._ctx.normalise_key(key)
        return super().__contains__(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Override method to get an item with a default, honouring key case."""
        return super().get(self._ctx.normalise_key(key), default)

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        """Override method to assign several items, keeping the parse tree in sync."""
        items = other.items() if hasattr(other, "keys") else other
        for key, value in items:
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Override method to add an item only if absent, keeping the parse tree in sync."""
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key: str, /, *default: Any) -> Any:
        """Override method to remove an item, keeping the parse tree in sync."""
        if key in self:
            value = self[key]
            del self[key]
            return value
        if default:
            return default[0]
        raise KeyError(key)

    def popitem(self) -> tuple[str, Any]:
        """Override method to remove the last item, keeping the parse tree in sync."""
        key = next(reversed(list(self)))
        return key, self.pop(key)

    def clear(self) -> None:
        """Override method to remove every item, keeping the parse tree in sync."""
        for key in list(self):
            del self[key]

    def __ior__(self, other: Any) -> Config:  # type: ignore[override, misc]
        """Override in-place merge, keeping the parse tree in sync."""
        self.update(other)
        return self

    def __str__(self) -> str:
        """Override method to print dict contents to a string.

        A configuration with no entries left is still written out, because the parse tree
        may hold comments and blank lines that are genuinely part of the file. Only a
        grammar that cannot derive a container without entries falls back to the empty
        string, since for such a format there is no text left to write.
        """
        if dict(self) == {}:
            try:
                return self._reconstruct()
            except UnexpectedInput:
                return ""
        return self._reconstruct()


class ConfigParser(ABC):
    """Lark-based configuration parser base class.

    A parser built by extending this class is for a file that can be mapped onto a Python
    dict: a series of key-value assignments, where a value is a scalar (``a=1``), a list
    (``a=1,2,3``) or a block of further assignments (``blk: b=1, c=2``).

    Subclassing means supplying a Lark grammar and saying whether keys are case-sensitive.
    The grammar has to be written against the naming contract stated in full in
    ``grammar_contract`` -- the rule names there are what lets one set of machinery read,
    edit and extend every format.

    A parser may steer the text of a *new* entry through the ``entry_templates`` and
    ``value_rule_priority`` properties, but the defaults derive everything from the grammar
    and are normally enough.

    This class is abstract to prevent instantiation, as it requires a grammar in order to
    work at all.
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

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        """Optional hand-written templates for the text of a new entry.

        Keyed by container rule name and entry category, e.g. ``("block", "key_value")``.
        The value is ordinary text with ``{key}``, ``{value}``, ``{block}`` and ``{ws}``
        markers, where ``{ws}`` marks whitespace that should follow the style of a
        neighbouring entry.

        A template applies **only when there is no neighbouring entry to copy a style
        from**. Matching the entries already in the file always wins, because a minimal diff
        matters more than any declared convention; what a template decides is how the very
        first entry of its kind should look. Override this when a format's conventional
        layout is not what the grammar-derived text happens to be -- an indented block body,
        say, which nothing in the grammar implies.

        Even then it is a preference, not a replacement: synthesis falls back to the
        grammar-derived candidates if the template no longer fits the grammar.

        Returns:
            Mapping[tuple[str, str], str]: The templates; empty by default.
        """
        return {}

    @property
    def value_rule_priority(self) -> Sequence[str]:
        """Order in which value-type rules are tried when writing a value for a new key.

        Several value types accept the same Python type, so the order decides how a value is
        written: whether a ``bool`` becomes ``.true.`` or ``True``, or a ``float`` uses a
        Fortran ``D`` exponent. Only rules the grammar admits are considered.

        Returns:
            Sequence[str]: The rule names, most preferred first.
        """
        return VALUE_RULE_PRIORITY

    def parse(self, stream: str) -> Config:
        """Parse the given text.

        Args:
            stream (str): The text to parse.

        Returns:
            Config: instance of the Config class storing the parsed data.
        """
        ctx = ParseContext(
            compile_grammar(self.grammar),
            self.case_sensitive_keys,
            self.entry_templates,
            tuple(self.value_rule_priority),
        )

        # Parse text. Here we add a newline character to simplify the writting of the
        # grammars, as otherwise one would have to explicitly take into account the case
        # where the text does no end with a newline. The start symbol has to be named
        # explicitly, because the parser has more than one.
        tree = ctx.lark.parse(stream + "\n", start=START_RULE)

        AddParent().visit(tree)
        return Config(tree, ctx)
