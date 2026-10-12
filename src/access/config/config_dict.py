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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NoReturn, SupportsIndex

from access.config.grammar_contract import KEY_BLOCK, KEY_LIST, KEY_VALUE
from access.config.grammar_values import UnsupportedEntryError
from access.config.tree_reader import entry_value

if TYPE_CHECKING:
    from access.config.config_store import ConfigStore
    from access.config.tree_reader import EntryRef


_Leftovers = list[tuple[int, list[tuple[str, Any]]]]
"""Items of a dict still to add after a deeper key was written for its first: for each
block, how many keys deep it is, and its other items, as the caller spelled their keys."""


class ConfigList(list):
    """A list subclass keeping the parse tree in sync with every change made to it.

    Reading one is reading a builtin list. Every write -- an element, a slice, or any of
    the methods that add, remove or reorder elements -- goes through the store holding the
    list's entry first, and changes the list only once the file has taken it, so the two
    never disagree. A write the file cannot take raises and leaves both as they were.

    Writing an element changes one token, in the notation it was written in. A list that
    grows or shrinks is laid out afresh around the elements that stay: those keep their
    text, comments included, and a new element is written like its neighbours, on their
    line or on a line of its own. A list keeps at least two elements, and one whose values
    are written with repeat counts, array qualifiers or in more than one assignment can be
    rewritten but not resized. The elements a repeat count stands for are rewritten only
    together, all alike.

    A list of records, which the file writes one entry per record, keeps at least one
    instead, and hands each record out as a ``ConfigRecord``: a dict that is read only,
    changed by assigning a dict to its place in the list.

    A copy is a plain list, which is what a copy of the data is.

    Args:
        data (list[Any]): The list data.
        store (ConfigStore): The store holding the list's entry.
        key (str): The normalised key the list is stored under.
    """

    _store: ConfigStore  # The store holding the list's entry.
    _key: str  # The key the list is stored under in that store.

    def __init__(self, data: list[Any], store: ConfigStore, key: str) -> None:
        super().__init__()
        self._store = store
        self._key = key
        list.extend(self, _held(data))

    def _write(self, values: list[Any], origins: list[int | None] | None = None) -> None:
        """Write *values* through the store, then hold them.

        Args:
            values (list[Any]): The new elements.
            origins (list[int | None] | None): For a list of a different length, the
                position of the old element each new one stands in for, ``None`` for a new
                one. ``None`` for a list of the same length, written position by position.
        """
        plain = [_plain(value) for value in values]
        self._store.replace(self._key, plain, origins)
        self._hold(plain)

    def _hold(self, values: list[Any]) -> None:
        """Hold *values*, which the file already does, in place of what the list held."""
        list.__setitem__(self, slice(None), _held(values))

    def __setitem__(self, index: SupportsIndex | slice, value: Any) -> None:
        """Override to write the element or elements through the parse tree.

        A slice may be given a different number of elements, which changes the length of
        the list, as it does for a builtin list.

        Args:
            index (SupportsIndex | slice): Integer index or slice of the element(s) to
                update.
            value (Any): New value, or an iterable of values for a slice.
        """
        if isinstance(index, slice):
            value = list(value)
        updated = list(self)
        updated[index] = value
        if len(updated) == len(self):
            self._write(updated)
        else:
            origins: list[int | None] = list(range(len(self)))
            origins[index] = [None] * len(value)
            self._write(updated, origins)

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        """Override to remove the element or elements from the parse tree too."""
        updated = list(self)
        del updated[index]
        origins: list[int | None] = list(range(len(self)))
        del origins[index]
        self._write(updated, origins)

    def append(self, value: Any) -> None:
        """Override to add the element to the parse tree too."""
        self[len(self) :] = [value]

    def extend(self, values: Iterable[Any]) -> None:
        """Override to add the elements to the parse tree too."""
        self[len(self) :] = values

    def insert(self, index: SupportsIndex, value: Any) -> None:
        """Override to add the element to the parse tree too."""
        self[index:index] = [value]

    def pop(self, index: SupportsIndex = -1) -> Any:
        """Override to remove the element from the parse tree too."""
        value = self[index]
        del self[index]
        return value

    def remove(self, value: Any) -> None:
        """Override to remove the element from the parse tree too."""
        del self[self.index(value)]

    def clear(self) -> None:
        """Override to remove every element, which a list cannot do: it keeps two."""
        del self[:]

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        """Override to write the reordered elements through the parse tree."""
        self._write(sorted(self, key=key, reverse=reverse))

    def reverse(self) -> None:
        """Override to write the reversed elements through the parse tree."""
        self._write(self[::-1])

    def __iadd__(self, values: Iterable[Any]) -> ConfigList:  # type: ignore[override, misc]
        """Override in-place concatenation, adding the elements to the parse tree too."""
        self.extend(values)
        return self

    def __imul__(self, count: SupportsIndex) -> ConfigList:  # type: ignore[override, misc]
        """Override in-place repetition: the elements after the first run are new ones."""
        updated = list(self) * count
        origins = [*range(len(self)), *[None] * (len(updated) - len(self))] if updated else []
        self._write(updated, origins)
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        """Copy and pickle as a plain list, a record as a plain dict: no parse tree."""
        return list, ([_plain(value) for value in self],)


class ConfigRecord(dict):
    """One record of a list the file writes one entry per record, as a dict.

    An FMS field table writes a method that way: ``"profile_type", "fixed",
    "surface_value=1.e30"`` is the record ``{"value": "fixed", "surface_value": 1e30}``.
    A record is read only, since changing one of its parameters is writing the whole entry:
    it is replaced by assigning a dict to its place in the list, as in
    ``methods[0] = {**methods[0], "surface_value": 2e30}``.

    A copy is a plain dict, which is what a copy of the data is.
    """

    def _refuse(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Reject a write to the record.

        Args:
            *args (Any): Ignored.
            **kwargs (Any): Ignored.

        Raises:
            TypeError: Always.
        """
        raise TypeError("A record is replaced whole: assign a dict to its place in the list")

    __setitem__ = _refuse  # type: ignore[assignment]
    __delitem__ = _refuse  # type: ignore[assignment]
    __ior__ = _refuse  # type: ignore[assignment]
    update = _refuse  # type: ignore[assignment]
    setdefault = _refuse  # type: ignore[assignment]
    pop = _refuse  # type: ignore[assignment]
    popitem = _refuse  # type: ignore[assignment]
    clear = _refuse  # type: ignore[assignment]

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        """Copy and pickle as a plain dict: the parse tree is not data."""
        return dict, (dict(self),)


def _plain(value: Any) -> Any:
    """Return *value* as plain data: a record as a ``dict``, anything else as it is."""
    return dict(value) if isinstance(value, ConfigRecord) else value


def _held(values: Iterable[Any]) -> list[Any]:
    """Return what stands for *values* in a list: a record for a dict, else the value."""
    return [ConfigRecord(value) if isinstance(value, dict) else value for value in values]


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
    # configuration of a whole file. Used to bring the configurations above this one back in
    # step after a deletion here removed more of the tree than the entry itself.
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
        return ConfigList(value, self._store, key) if ref.category == KEY_LIST else value

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
        after the last existing one, written like it; in a block written one component per
        line, after the last line of that block. A block created this way has no entry of
        its own to match, so its contents follow the nearest sibling block instead.

        Args:
            key (str): The key to assign to. For a new key, the name is written into the
                file as given, while the dict entry uses the normalised form.
            value (Any): The new value.

        Raises:
            TypeError: If the new value's type does not match the existing one, or -- as
                ``UnsupportedEntryError`` -- if the format cannot express the new value.
            ValueError: If a list would have fewer than two elements, or could not change
                length, or a new key name is unusable.
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
            if isinstance(value, list):
                value = [_plain(item) for item in value]
            self._store.replace(key, value)
            if isinstance(value, list):
                # Update the list already stored rather than making a new one, so that a
                # caller holding it sees what the file now holds.
                stored = dict.get(self, key)
                assert isinstance(stored, ConfigList)
                stored._hold(value)
            else:
                stored = value
            super().__setitem__(key, stored)

    def _add(self, key: str, raw_key: str, value: Any) -> None:
        """Add an entry for a key that is not in the configuration yet.

        The entry is written by the nearest configuration whose block has a body in the
        file. That is this one, usually; for a block merged from several in the file, or
        written one component per line as a Fortran derived type is, it is the one holding
        it, which writes a line carrying the whole key path -- ``a%d = 1`` for ``d`` in
        ``a``. The configurations on that path are then read again, in place, so that a
        caller holding one sees the new key.

        A ``dict`` the format cannot write as a block of its own is written one level
        deeper: the Fortran ``nml["g"]["t"] = {"a": 1}`` is the line ``t%a = 1``. What else
        the dict holds is then added to the block that line created.

        A block arrives empty, because nothing in it can say how its contents should be
        spaced until it has some. It is handed the style of its nearest sibling and then
        filled through this same path, one key at a time, so nesting to any depth works.

        A list of dicts is a list of records, which the file writes one entry per record:
        the first is written as the entry, and the others are added beside it, each a copy
        of the one before. Should one of them be refused, the key goes again.

        Args:
            key (str): The normalised key, used for the dict entry.
            raw_key (str): The key as the caller wrote it, used for the file text.
            value (Any): The value to store.
        """
        records = isinstance(value, list) and len(value) > 0 and all(isinstance(item, dict) for item in value)
        anchor, path = self._nearest_addable()
        first = _plain(value[0]) if records else value
        keys, ref, written, rest = anchor._write_path([*path, raw_key], first, KEY_VALUE if records else None)
        if len(keys) == 1:
            dict.__setitem__(self, key, self._wrap(key, ref))
        else:
            anchor._reread(keys)

        if records:
            if len(value) > 1:
                try:
                    anchor._at(keys)._write(list(value), [0, *[None] * (len(value) - 1)])
                except (ValueError, TypeError):
                    del anchor._at(keys[:-1])[keys[-1]]
                    raise
            return
        if isinstance(written, dict):
            block = anchor._at(keys)
            block._store.fallback_style = anchor._store.created_block_style(ref.entry_nodes[0])
            for block_key, block_value in written.items():
                block[block_key] = block_value
        for depth, items in reversed(rest):
            owner = anchor._at(keys[:depth])
            for item_key, item_value in items:
                owner[item_key] = item_value

    def _nearest_addable(self) -> tuple[Config, list[str]]:
        """Return the nearest configuration a new line can go in, and the keys down to this.

        Returns:
            tuple[Config, list[str]]: This configuration, or the nearest one holding it
                whose block has a body in the file; and the keys leading from that one down
                to this, empty if it is this.
        """
        anchor, path = self, []
        while not anchor._store.addable and anchor._parent is not None:
            anchor, key = anchor._parent
            path.append(key)
        return anchor, path[::-1]

    def _write_path(
        self, raw_keys: list[str], value: Any, category: str | None = None
    ) -> tuple[list[str], EntryRef, Any, _Leftovers]:
        """Write *value* at a key path below this configuration, deeper if need be.

        A non-empty ``dict`` the format refuses as a block is written as its first item one
        level down instead, and so on, until something is written or there is nothing left
        to go down into. A record, written with a *category* of its own, never is.

        Args:
            raw_keys (list[str]): The key path as the caller wrote it.
            value (Any): The value to store.
            category (str | None): The category of entry to write, where *value* does not
                say: ``key_value`` for a record, which is a ``dict``.

        Returns:
            tuple[list[str], EntryRef, Any, _Leftovers]: The key path written, normalised,
                the new entry, the value written at the end of it, and the items of each
                dict gone down through that are still to add.

        Raises:
            UnsupportedEntryError: The refusal of the path first asked for, if no level
                down could be written either.
        """
        rest: _Leftovers = []
        refusal = None
        while True:
            try:
                ref = self._store.add_path(raw_keys, value, category)
                return [self._store.ctx.normalise_key(raw) for raw in raw_keys], ref, value, rest
            except UnsupportedEntryError as error:
                refusal = refusal or error
                if category is not None or not isinstance(value, dict) or not value:
                    raise refusal from None
            (raw, inner), *others = value.items()
            rest.append((len(raw_keys), others))
            raw_keys, value = [*raw_keys, raw], inner

    def _at(self, path: Sequence[str]) -> Any:
        """Return what the keys of *path* lead to from this configuration."""
        found: Any = self
        for key in path:
            found = dict.__getitem__(found, key)
        return found

    def __delitem__(self, key: str) -> None:
        """Override method to del item from dict, keeping the parse tree in sync."""
        key = self._store.ctx.normalise_key(key)
        super().__delitem__(key)
        self._store.remove(key)
        self._resync()

    def _is_anchor(self) -> bool:
        """Report whether a deletion below this configuration leaves its container alone.

        Deleting an entry also removes the wrappers the grammar cannot derive empty, and
        that pruning stops only at a node with no parent or at a plain repetition: an empty
        block is still a block. A configuration over such a node, held in the parse tree
        rather than merged from several, is therefore still where it was.

        Returns:
            bool: True for the configuration of a whole file, or of a block whose container
                no deletion can remove.
        """
        store = self._store
        return self._parent is None or (store.addable and store.ctx.info.is_repetition_rule(str(store.tree.data)))

    def _resync(self) -> None:
        """Bring the configurations this one is nested in back in step with the tree.

        A block written a component per line, as a derived type is, holds nothing once its
        last component goes, and the lines writing it are already out of the tree -- and so
        may be the lines of the blocks it was nested in. The configurations above it are
        read again from the nearest one whose container survived, down the path to this
        one, so that a block the file no longer has leaves the dict too, and the references
        left to it no longer name nodes that are gone.
        """
        path: list[str] = []
        anchor = self
        while not anchor._is_anchor():
            assert anchor._parent is not None
            anchor, key = anchor._parent
            path.append(key)
        if path:
            anchor._reread(path[::-1])

    def _reread(self, path: Sequence[str], ref: EntryRef | None = None) -> None:
        """Read this configuration again after the tree changed below it, then down *path*.

        What the file no longer has leaves the dict. The configurations already on the path
        are kept, so that a caller holding one sees the change, and are pointed at their
        blocks as read afresh: a block merged from several in the file is a node the reader
        builds, and reading again builds a new one. Where the path leaves them, the rest is
        new and is wrapped whole.

        Args:
            path (Sequence[str]): Keys leading down to the configuration the change was made
                in, starting with a key of this one; for an entry just added, down to it.
            ref (EntryRef | None): The reference its parent read afresh for this block, to
                point this configuration at; ``None`` for the one the reading starts from.
        """
        refs = self._store.reread(ref)
        for vanished in [key for key in self if key not in refs]:
            child = dict.__getitem__(self, vanished)
            if isinstance(child, Config):
                child._parent = None
            dict.__delitem__(self, vanished)
        if path and path[0] in refs:
            key = path[0]
            child = dict.get(self, key)
            if isinstance(child, Config):
                child._reread(path[1:], refs[key])
            elif key not in self:
                dict.__setitem__(self, key, self._wrap(key, refs[key]))

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
