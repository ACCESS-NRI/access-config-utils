# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""The dict and list a parsed configuration is handed to the caller as.

``Config`` is a ``dict`` and ``ConfigList`` is a ``list``: reading one is exactly reading
the builtin. What they add is that every *write* is mirrored into the parse tree, so that
writing the configuration back out reflects it -- and that a key can be spelled in whatever
case the format allows.

Neither class touches a ``Tree``. ``Config`` holds a ``ConfigStore``, which owns the tree
and does the work; what is left here is dict semantics: normalising a key, deciding whether
an assignment is an update or an addition, and wrapping what the store reports back into
something that itself keeps the tree in step -- a ``ConfigList`` for a list, a nested
``Config`` for a block.

The mapping methods ``dict`` implements in C are all overridden, because they do not go
through ``__setitem__`` / ``__delitem__`` and would therefore skip both the key
normalisation and the tree update.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NoReturn, SupportsIndex

from access.config.grammar_contract import KEY_BLOCK, KEY_LIST
from access.config.tree_edits import write_values
from access.config.tree_reader import entry_value

if TYPE_CHECKING:
    from lark import Tree

    from access.config.config_store import ConfigStore
    from access.config.grammar_compiled import ParseContext
    from access.config.tree_reader import EntryRef


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
    A repeat count is the one case where a node backs more than one element: ``6*9.0`` is
    six elements written once. Assigning to any of them splits the run, so the notation
    survives an edit -- the six become ``2*9.0, 8.0, 3*9.0``, not six separate values.

    Args:
        data (list[Any]): The list data.
        nodes (Sequence[Tree]): The node backing each element. A repeat node appears once
            per element it covers, so this is one entry per element, not per node.
        ctx (ParseContext): The compiled grammar, needed to parse the replacement for a
            repeat run that an assignment has split.
    """

    _nodes: list[Tree]  # The parse tree node backing each list element.
    _ctx: ParseContext  # Compiled grammar and per-parser settings.

    def __init__(self, data: list[Any], nodes: Sequence[Tree], ctx: ParseContext) -> None:
        super().__init__(data)
        self._nodes = list(nodes)
        self._ctx = ctx

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
        updated = list(self)
        if isinstance(index, slice):
            values = list(value)
            if len(values) != len(self._nodes[index]):
                raise ValueError(
                    f"Slice assignment would change list length from {len(self._nodes[index])} to {len(values)}"
                )
            updated[index] = values
        else:
            updated[index] = value

        # The whole list is written, not just the elements assigned to: a repeat node backs
        # several of them at once, so the run it covers is rewritten as a unit. Writing a
        # value back over itself reproduces its own token, so untouched ones do not move.
        self._nodes[:] = write_values(self._nodes, updated, self._ctx.lark)
        # Not *value*: for a slice it may have been a one-shot iterable, already drained
        # above, and taking it a second time would leave the list short.
        super().__setitem__(index, values if isinstance(index, slice) else value)

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


class ConfigBlockList(list):
    """The occurrences of a block the file writes more than once, one ``Config`` each.

    A format may read a repeated block as separate records rather than merging them -- a
    format may, where a reader stops at the first block of the name and so no single read
    ever sees the merge. Each element here is a nested ``Config`` over one
    occurrence, and writing through it reaches that occurrence alone.

    Only the elements are editable, not the list. Adding or removing a whole block is a
    change to the file's structure that nothing here knows how to write -- where the new
    block would go, and how it should be spaced -- so those operations raise
    ``NotImplementedError`` rather than leaving the list and the file disagreeing.

    Args:
        blocks (Sequence[Config]): One configuration per occurrence, in the order written.
    """

    def _unsupported(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject an operation that would add, remove or replace a whole block.

        Args:
            *args (Any): Ignored.
            **kwargs (Any): Ignored.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Adding, removing or replacing a whole occurrence of a repeated block is not "
            "supported; edit the entries of one occurrence, or delete the key to remove all "
            "of them"
        )

    append = _unsupported  # type: ignore[assignment]
    extend = _unsupported  # type: ignore[assignment]
    insert = _unsupported  # type: ignore[assignment]
    remove = _unsupported  # type: ignore[assignment]
    pop = _unsupported  # type: ignore[assignment]
    clear = _unsupported  # type: ignore[assignment]
    sort = _unsupported  # type: ignore[assignment]
    reverse = _unsupported  # type: ignore[assignment]
    __setitem__ = _unsupported  # type: ignore[assignment]
    __delitem__ = _unsupported  # type: ignore[assignment]
    __iadd__ = _unsupported  # type: ignore[assignment]
    __imul__ = _unsupported  # type: ignore[assignment]


class Config(dict):
    """A dict holding the contents of a parsed configuration file.

    Every write is mirrored into the parse tree the store owns, so that ``str(config)``
    reproduces the file with the change and nothing else. Keys are matched however the
    format matches them: case-insensitively for some formats, exactly for others.

    Args:
        store (ConfigStore): The parse tree of one container and the entries in it.
    """

    _store: ConfigStore  # The parse tree behind this dict.
    # The configuration this one is a block of, and the key it is stored under. None for the
    # configuration of a whole file. Used to drop a block that has been emptied and cannot
    # be written empty -- a derived type is one, since "a%" on its own means nothing.
    _parent: tuple[Config, str] | None

    def __init__(self, store: ConfigStore) -> None:
        self._store = store
        self._parent = None
        super().__init__({key: self._wrap(key, ref) for key, ref in store.refs.items()})

    def _wrap(self, key: str, ref: EntryRef) -> Any:
        """Return the object standing for an entry in the dict.

        A list and a block are both handed out as something that keeps the parse tree in
        step: a ``ConfigList`` pairs each element with the node holding it, and a nested
        ``Config`` does the same for the entries of a block. A block the file writes more
        than once, in a format that reads those as separate records, becomes a
        ``ConfigBlockList`` of one ``Config`` per occurrence.

        Args:
            key (str): The normalised key the entry is stored under.
            ref (EntryRef): The entry to represent.

        Returns:
            Any: The scalar, a ``ConfigList``, a nested ``Config``, a ``ConfigBlockList``,
                or ``None``.
        """
        if ref.category == KEY_BLOCK:
            blocks = [self._block(key, ref, index) for index in range(len(ref.block_nodes))]
            return blocks[0] if len(blocks) == 1 else ConfigBlockList(blocks)
        value = entry_value(ref)
        return ConfigList(value, ref.value_nodes, self._store.ctx) if ref.category == KEY_LIST else value

    def _block(self, key: str, ref: EntryRef, index: int) -> Config:
        """Return the nested configuration for one occurrence of a block."""
        block = Config(self._store.child(ref, index))
        # Tell it where it lives, so emptying it can remove it from here.
        block._parent = (self, key)
        return block

    def __getitem__(self, key: str) -> Any:
        """Override method to get item from dict.

        This method takes into account if keys are case-sensitive or not.
        """
        return super().__getitem__(self._store.ctx.normalise_key(key))

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
        key = self._store.ctx.normalise_key(key)

        if key not in self:
            self._add(key, raw_key, value)
        elif self[key] is None:
            if value is not None:
                raise TypeError(f"Trying to change the type of variable '{key}'")
        elif isinstance(value, dict):
            raise SyntaxError("Trying to assign a new value to an entire block")
        else:
            nodes = self._store.replace(key, value)
            if isinstance(value, list):
                # Update the list already stored rather than making a new one. Writing a
                # list can replace the nodes behind it, and a caller holding the old one
                # would otherwise go on writing through nodes no longer in the tree.
                stored = dict.get(self, key)
                assert isinstance(stored, ConfigList)
                list.__setitem__(stored, slice(None), value)
                stored._nodes[:] = nodes
            else:
                stored = value
            super().__setitem__(key, stored)

    def _add(self, key: str, raw_key: str, value: Any) -> None:
        """Add an entry for a key that is not in the configuration yet.

        A block arrives empty, because nothing in it can say how its contents should be
        spaced until it has some. It is handed the style of its nearest sibling and then
        filled through this same path, one key at a time, so nesting to any depth works.

        Args:
            key (str): The normalised key, used for the dict entry.
            raw_key (str): The key as the caller wrote it, used for the file text.
            value (Any): The value to store.
        """
        ref = self._store.add(key, raw_key, value)
        stored = self._wrap(key, ref)
        super().__setitem__(key, stored)

        if ref.category == KEY_BLOCK:
            assert isinstance(stored, Config)
            stored._store.fallback_style = self._store.created_block_style()
            for block_key, block_value in value.items():
                stored[block_key] = block_value

    def __delitem__(self, key: str) -> None:
        """Override method to del item from dict, keeping the parse tree in sync."""
        key = self._store.ctx.normalise_key(key)
        super().__delitem__(key)
        self._store.remove(key)
        self._drop_if_unwritable()

    def _drop_if_unwritable(self) -> None:
        """Remove this configuration from its parent if it is empty and unwritable.

        A block whose rule is a plain repetition may hold nothing -- an empty block is
        still a block. One written a component per line may not: once the last component
        goes there is nothing left to write, and the lines
        have already been removed from the tree. Leaving the empty block in the parent's
        dict would make it disagree with the file.
        """
        if self or self._parent is None:
            return
        if self._store.ctx.info.is_repetition_rule(str(self._store.tree.data)):
            return
        parent, key = self._parent
        dict.__delitem__(parent, key)
        parent._store.refs.pop(key, None)
        self._parent = None

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
            key = self._store.ctx.normalise_key(key)
        return super().__contains__(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Override method to get an item with a default, honouring key case."""
        return super().get(self._store.ctx.normalise_key(key), default)

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
        """Override method to write the configuration back out as text."""
        return self._store.render()
