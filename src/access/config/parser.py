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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, SupportsIndex

from lark import Lark, Tree
from lark.exceptions import UnexpectedInput
from lark.reconstruct import Reconstructor

from access.config.entry_synthesis import (
    BLOCK_RULE,
    EntryStyle,
    GrammarInfo,
    admitted_value_rules,
    iter_entry_snippets,
    probe_entry_style,
    probe_sibling_block_style,
)
from access.config.parse_tree_ops import (
    AddParent,
    ConfigToDict,
    entry_insertion_index,
    find_rule_node,
    parse_entry_nodes,
    splice_entry_nodes,
    update_node_value,
)
from access.config.parser_types import VALUE_RULE_PRIORITY, UnsupportedEntryError

START_RULE = "start"
"""Name of the top-level rule, fixed by the grammar contract."""


@dataclass(frozen=True)
class CompiledGrammar:
    """A compiled grammar and the artefacts derived from it.

    Compiling a grammar is far more expensive than parsing a configuration file with it, and
    the same artefacts are needed again whenever a key is added, so they are built once and
    cached.

    Args:
        lark (Lark): The compiled parser.
        reconstructor (Reconstructor): Writes a parse tree back out as text.
        info (GrammarInfo): Views over the grammar, used to synthesise new entries.
    """

    lark: Lark
    reconstructor: Reconstructor
    info: GrammarInfo


@dataclass(frozen=True)
class ParseContext:
    """Everything a ``Config`` needs in order to edit itself, beyond the parse tree.

    Args:
        grammar (CompiledGrammar): The compiled grammar and its derived artefacts.
        case_sensitive_keys (bool): Are the dict keys case-sensitive?
        entry_templates (Mapping[tuple[str, str], str]): Per-parser overrides for the
            text of a new entry, keyed by container rule name and entry category.
        value_rule_priority (tuple[str, ...]): Order in which value-type rules are tried for
            a new value.
    """

    grammar: CompiledGrammar
    case_sensitive_keys: bool
    entry_templates: Mapping[tuple[str, str], str]
    value_rule_priority: tuple[str, ...]

    @property
    def lark(self) -> Lark:
        """Lark: The compiled parser."""
        return self.grammar.lark

    @property
    def reconstructor(self) -> Reconstructor:
        """Reconstructor: The reconstructor for this grammar."""
        return self.grammar.reconstructor

    @property
    def info(self) -> GrammarInfo:
        """GrammarInfo: Grammar views used to synthesise new entries."""
        return self.grammar.info


_GRAMMAR_CACHE: dict[str, CompiledGrammar] = {}
"""Compiled grammars, keyed by grammar text.

Keyed on the text rather than on the parser class, because ``ConfigParser.grammar`` is a
property and so could differ between instances of one class, and because parsers sharing a
grammar should share the compiled result. Concurrent first use may compile twice and discard
one result, which is wasteful but harmless.
"""


def clear_grammar_cache() -> None:
    """Discard every cached compiled grammar.

    Only needed by tests that change a grammar between parses.
    """
    _GRAMMAR_CACHE.clear()


def _entry_category(value: Any) -> str:
    """Return the kind of entry needed to hold *value*.

    Args:
        value (Any): The value to be stored.

    Returns:
        str: One of the entry category names.
    """
    if isinstance(value, dict):
        return "key_block"
    if value is None:
        return "key_null"
    if isinstance(value, list):
        return "key_list"
    return "key_value"


def _same_value(stored: Any, requested: Any) -> bool:
    """Report whether re-reading the written text gave back exactly the requested value.

    The type is compared as well as the value, because several value types share a Python
    type and a written value can be read back as the wrong one: ``Path("foo")`` writes as
    ``foo``, which a grammar with a bare-word value type reads back as a plain ``str``.

    Args:
        stored (Any): The value read back from the candidate text.
        requested (Any): The value the caller asked to store.

    Returns:
        bool: True if the two match in both type and value.
    """
    return type(stored) is type(requested) and bool(stored == requested)


def _entry_matches(data: Mapping[str, Any], key: str, value: Any) -> bool:
    """Report whether a candidate entry, read back from its text, holds what was asked for.

    This is what makes candidate ranking a matter of style rather than of correctness: a
    candidate that parses but means something else fails here and the next one is tried.

    Args:
        data (Mapping[str, Any]): The values read back from the candidate.
        key (str): The normalised key expected.
        value (Any): The value requested.

    Returns:
        bool: True if the candidate is an exact match.
    """
    if set(data) != {key}:
        return False
    stored = data[key]
    if isinstance(value, dict):
        # A new block starts empty; its contents are added afterwards.
        return isinstance(stored, Config) and not stored
    if value is None:
        return stored is None
    if isinstance(value, list):
        return (
            isinstance(stored, list)
            and len(stored) == len(value)
            and all(_same_value(item, wanted) for item, wanted in zip(stored, value, strict=True))
        )
    return _same_value(stored, value)


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
    _refs: dict[str, list[Tree] | Tree]
    # References to rule nodes in the parse tree, keyed by config key:
    #   scalar keys  → a value-type rule node (Tree whose .data is in the handler registry)
    #   list keys    → a list of value-type rule nodes (one per element)
    #   block keys   → the "block" rule node (Tree whose .data == "block")
    _ctx: ParseContext  # Compiled grammar and per-parser settings
    # Whitespace for a new entry when this container holds none to copy from. Set on a block
    # created by an assignment, which is empty and so has no style of its own yet.
    _fallback_style: EntryStyle

    def __init__(self, tree: Tree, ctx: ParseContext) -> None:
        self._tree = tree
        self._ctx = ctx
        self._fallback_style = EntryStyle()
        interpreter = ConfigToDict(ctx)
        data, self._refs = interpreter.visit(self._tree)
        # Wrap list values in ConfigList so element-level updates keep the tree in sync
        for key in data:
            if isinstance(data[key], list):
                refs = self._refs[key]
                assert isinstance(refs, list)
                data[key] = ConfigList(data[key], refs)
        super().__init__(data)

    # --- Key normalisation (SRP) ---

    def _normalize_key(self, key: str) -> str:
        """Normalise a key according to the case-sensitivity setting.

        Args:
            key (str): The raw key.

        Returns:
            str: The normalised key.
        """
        return key if self._ctx.case_sensitive_keys else key.upper()

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
        refs = self._refs[key]
        if not isinstance(refs, list):
            raise TypeError(f"Trying to change the type of variable '{key}'")
        if len(refs) != len(value):
            raise ValueError(f"Trying to change the length of list '{key}'")

        for rule_node, v in zip(refs, value, strict=True):
            update_node_value(rule_node, v)

        return ConfigList(value, refs)

    # --- Key addition (SRP) ---

    def _insert_new_key(self, key: str, raw_key: str, value: Any) -> None:
        """Add an entry for a key that is not in the configuration yet.

        Works by synthesising the *text* of the entry, handing it back to Lark with this
        container's rule as the start symbol, and splicing the resulting nodes in. Candidate
        texts are tried in order of how idiomatic they are, and each is validated in full --
        it must parse, and reading it back must give exactly the requested value -- before
        the tree is modified. A candidate the grammar reads differently from how it was
        meant, such as a bare word a format happens to read as a path, is discarded rather
        than written out.

        The whitespace of the entry is copied from its neighbours in this container, or --
        when there are none, as in a block this method has just created -- from the style
        handed to the block when it was created.

        Args:
            key (str): The normalised key, used for the dict entry.
            raw_key (str): The key as the caller wrote it, used for the file text.
            value (Any): The value to store.

        Raises:
            ValueError: If *raw_key* is not a usable key name, or *value* is a list too
                short for the format to express as one.
            UnsupportedEntryError: If no candidate produced the requested value.
        """
        if not isinstance(raw_key, str) or not raw_key.isidentifier():
            raise ValueError(f"Not a valid configuration key: {raw_key!r}")

        category = _entry_category(value)
        if category == "key_list" and len(value) < 2:
            # Every supported format needs at least two values to be a list rather than a
            # scalar, and writing a scalar would put the wrong type in the dict.
            raise ValueError(f"A new list needs at least two elements: '{key}'")

        container = self._tree
        container_rule = str(container.data)
        ctx = self._ctx
        style = probe_entry_style(container)
        if not style.has_donor:
            style = self._fallback_style
        index = entry_insertion_index(container)

        for snippet in iter_entry_snippets(
            ctx.info,
            container_rule,
            category,
            raw_key,
            value,
            style,
            ctx.entry_templates,
            ctx.value_rule_priority,
        ):
            # Interpret the candidate in a throwaway container, so one that parses but does
            # not mean the right thing leaves no trace. A candidate can fail to parse, or
            # parse into a shape the interpreter rejects; either way the next one is tried.
            try:
                nodes = parse_entry_nodes(ctx.lark, container_rule, snippet)
                data, refs = ConfigToDict(ctx).visit(Tree(container.data, list(nodes)))
            except (UnexpectedInput, ValueError, TypeError):
                continue
            if not _entry_matches(data, key, value):
                continue

            splice_entry_nodes(container, nodes, index)
            stored = data[key]
            if isinstance(stored, list):
                element_refs = refs[key]
                assert isinstance(element_refs, list)
                stored = ConfigList(stored, element_refs)
            self._refs[key] = refs[key]
            dict.__setitem__(self, key, stored)

            if category == "key_block":
                # The new block is empty, so nothing in it can say how its contents should
                # be laid out. Hand it the style of the nearest sibling block, falling back
                # to this container's own, so that a block nested in a new block inherits it
                # too rather than dropping back to the canonical layout.
                assert isinstance(stored, Config)
                sibling = probe_sibling_block_style(container)
                stored._fallback_style = sibling if sibling.has_donor else self._fallback_style

                # Fill the block through the same path, one key at a time, so nesting to any
                # depth works.
                for block_key, block_value in value.items():
                    stored[block_key] = block_value
            return

        admitted = sorted(admitted_value_rules(ctx.info, container_rule, category))
        raise UnsupportedEntryError(
            f"This format cannot store {value!r} as '{raw_key}' here: there is no way to "
            f"write a {category} in '{container_rule}' with this value"
            + (f" (value types allowed here: {admitted})" if admitted else "")
        )

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
        return super().__getitem__(self._normalize_key(key))

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
        key = self._normalize_key(key)

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
            if not isinstance(ref, Tree):
                raise TypeError(f"Trying to change the type of variable '{key}'")
            update_node_value(ref, value)

        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Override method to del item from dict."""

        key = self._normalize_key(key)

        # Remove item from the dict
        super().__delitem__(key)

        # Remove the key rule node from the parse tree
        key_rule_node = find_rule_node(self._refs[key])
        key_rule_node.parent.children.remove(key_rule_node)  # type: ignore[attr-defined]

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
            key = self._normalize_key(key)
        return super().__contains__(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Override method to get an item with a default, honouring key case."""
        return super().get(self._normalize_key(key), default)

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

    The parsers built by extending this class are meant for files that share a common
    structure. In this case, the files must be made by a series of key-value assignments
    that can be mapped onto a Python dict. Each key-value assignment can therefore be one
    of three types:
      - scalar (e.g, 'a=1')
      - list/array (e.g., 'a=1,2,3')
      - block/dict containing other key-value assignments (e.g., 'blk: b=1, c=2')

    Because the resulting parse trees are all processed using the ``ConfigToDict``
    Interpreter, whose callbacks are named after grammar rules, all grammars must follow
    the same structure and use the same rule names:

      - Key-value assignment rules must be named (or aliased to): ``"key_value"``,
        ``"key_list"``, and ``"key_block"``. The ``ConfigToDict`` Interpreter dispatches
        to methods of those exact names.
      - Only value-type rules from ``"config.lark"`` should be used for scalar values.
        Their names must be registered in ``VALUE_TYPE_HANDLER_REGISTRY``.
      - The rule that captures the key name must be named ``"key"``. Its first child must
        be a ``Token`` (terminal) whose text is the key string. The ``"config.lark"`` file
        provides a ``"key"`` rule suitable for most cases.
      - Empty assignments (e.g., ``a=``) are supported. The corresponding rule must be
        named ``"key_null"``.

    Two further requirements exist so that keys can be *added* to a parsed file. Adding a
    key means creating parse tree nodes the parser never built, and the way that is done is
    to synthesise the text of the new entry and hand it back to Lark with the enclosing rule
    as the start symbol. That requires the enclosing rules to be nameable:

      - The start rule must be named ``"start"`` and must **not** be transparent: write
        ``start:``, never ``?start:``. A transparent rule collapses when it has a single
        child, so a file with one entry would have that entry as the root of the tree,
        leaving nothing to add to.
      - A ``"key_block"``'s contents must be held by a rule *named* ``"block"``, not by one
        aliased to it with ``-> block``. Lark resolves start symbols against rule names,
        while a node carries the alias, and the two have to agree.

    A parser may steer the synthesised text through the ``entry_templates`` and
    ``value_rule_priority`` properties, but the defaults derive everything from the grammar
    and are normally enough.

    This class is made abstract to prevent instantiation, as it requires a Lark grammar to
    be provided in order to work correctly.
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

        A template is a preference, not a replacement: it is tried first, and synthesis
        falls back to the grammar-derived candidates if it no longer fits the grammar.
        Override this only when the derived text is not what the format conventionally looks
        like.

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

    def _compile(self) -> CompiledGrammar:
        """Compile the grammar, or return the cached result for it.

        The parser is built with ``"block"`` as an additional start symbol, so the text of
        an entry can be parsed in the context it will live in. That is only possible when
        the grammar has such a rule, so its presence is probed first; a format without
        blocks keeps a single start symbol.

        Returns:
            CompiledGrammar: The compiled parser and the artefacts derived from it.
        """
        grammar = self.grammar
        compiled = _GRAMMAR_CACHE.get(grammar)
        if compiled is None:
            lark = self._build(grammar, [START_RULE])
            if any(str(rule.origin.name) == BLOCK_RULE for rule in lark.rules):
                lark = self._build(grammar, [START_RULE, BLOCK_RULE])
            compiled = CompiledGrammar(lark, Reconstructor(lark), GrammarInfo.from_lark(lark))
            _GRAMMAR_CACHE[grammar] = compiled
        return compiled

    @staticmethod
    def _build(grammar: str, start: list[str]) -> Lark:
        """Build a Lark parser for a grammar with the given start symbols.

        Args:
            grammar (str): The grammar text.
            start (list[str]): Names of the rules usable as start symbols.

        Returns:
            Lark: The compiled parser.
        """
        return Lark(grammar, import_paths=[Path(__file__).parent], maybe_placeholders=False, start=start)

    def parse(self, stream: str) -> Config:
        """Parse the given text.

        Args:
            stream (str): The text to parse.

        Returns:
            Config: instance of the Config class storing the parsed data.
        """
        ctx = ParseContext(
            self._compile(), self.case_sensitive_keys, self.entry_templates, tuple(self.value_rule_priority)
        )

        # Parse text. Here we add a newline character to simplify the writting of the
        # grammars, as otherwise one would have to explicitly take into account the case
        # where the text does no end with a newline. The start symbol has to be named
        # explicitly, because the parser has more than one.
        tree = ctx.lark.parse(stream + "\n", start=START_RULE)

        AddParent().visit(tree)
        return Config(tree, ctx)
