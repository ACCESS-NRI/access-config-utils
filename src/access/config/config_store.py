# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""The parse tree behind one configuration, and the operations that keep it in step.

A parsed configuration is two things at once: a dict a caller reads and writes, and a parse
tree that has to keep saying the same thing so the file can be written back out unchanged.
``ConfigStore`` is the second of those. It owns one container rule node, the reference to
every entry below it, the parse context, and the whitespace a block created this session
should inherit -- and it exposes the four things that can happen to an entry: read it,
replace its value, add it, remove it.

Splitting it out is what leaves ``Config`` as a dict and nothing else. The store knows
nothing about ``dict``; ``Config`` knows nothing about ``Tree``.

A block gets a store of its own, made by ``child``, which is also where the created-block
whitespace is handed down. That is the only reason a store knows about its neighbours.

Every *key* a method takes is a normalised one, the form ``refs`` is keyed by. ``add_path``
is the exception: it takes the path as the caller wrote it, which is what the text of the
new entry spells, and normalises it itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from lark import Tree
from lark.exceptions import UnexpectedInput

from access.config.config_insert import insert_entry
from access.config.config_resize import resize_entry
from access.config.entry_style import (
    EntryStyle,
    donor_template,
    indent_of,
    probe_entry_style,
    probe_sibling_block_style,
)
from access.config.grammar_contract import KEY_LIST, KEY_VALUE
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY, UnsupportedEntryError
from access.config.list_layout import align_values
from access.config.tree_edits import remove_entry_node, update_node_value, write_values
from access.config.tree_navigation import entry_chain, entry_insertion_index, entry_nodes, key_token, line_index
from access.config.tree_reader import EntryRef, entry_value, read_entries

if TYPE_CHECKING:
    from access.config.grammar_compiled import ParseContext


class ConfigStore:
    """The parse tree of one container, and the entries currently in it.

    Args:
        tree (Tree): The container rule node: the root of a parsed file, or a ``block``.
        ctx (ParseContext): The compiled grammar and the per-parser settings.
        addable (bool): Whether a new entry can be spliced into *tree*. False for a block
            merged from several in the file, whose root is not itself in the parse tree, so
            splicing into it would change nothing that gets written out.
    """

    tree: Tree  # The container rule node this store owns.
    ctx: ParseContext  # Compiled grammar and per-parser settings.
    refs: dict[str, EntryRef]  # Where each key's entry lives, keyed by normalised key.
    # The style for a new entry when this container holds no entry to copy. Set on a block
    # created by an assignment, which is empty and so has no style of its own yet.
    fallback_style: EntryStyle
    addable: bool  # Whether a new entry can be spliced into this container

    def __init__(self, tree: Tree, ctx: ParseContext, addable: bool = True) -> None:
        self.tree = tree
        self.ctx = ctx
        self.addable = addable
        self.fallback_style = EntryStyle()
        self.refs = read_entries(tree, ctx)

    def child(self, ref: EntryRef, index: int = 0) -> ConfigStore:
        """Return the store for the contents of a block held by this container.

        *ref* is a ``key_block`` reference read from this container. *index* picks the
        occurrence to open of a block the file writes more than once and this format reads
        as separate records; it is zero for every other block, which has exactly one.
        """
        return ConfigStore(ref.block_nodes[index], self.ctx, addable=ref.addable)

    def replace(self, key: str, value: Any, origins: Sequence[int | None] | None = None) -> None:
        """Write a new value into an entry that already exists.

        A scalar, or a list of the length it has, is written in place, node by node. A list
        of a different length is laid out afresh, with each new value matched to the old one
        it stands in for, so that what stays keeps its place. A list of records, which the
        file writes one entry per record, changes length by gaining and losing entries; see
        ``_resize_records``.

        For a list of a different length, *origins* gives the position of the old element
        each new value stands in for, or ``None`` for a new one; where the caller does not
        know, it is worked out by aligning the two lists.

        Raises:
            TypeError: If the new value's type does not match what the entry holds, or a
                value standing in for an old element is of another type.
            ValueError: If fewer than two values are asked for (none, for a list of
                records), or the entry's values are not written one per node in a single
                entry.
            UnsupportedEntryError: If the format cannot write the list asked for.
        """
        ref = self.refs[key]
        if ref.category != (KEY_LIST if isinstance(value, list) else KEY_VALUE):
            raise TypeError(f"Trying to change the type of variable '{key}'")
        if not isinstance(value, list):
            node = ref.value_nodes[0]
            # A None node means an array position the file never wrote, which only a
            # key_list entry can have; a key_value always has the one node it names.
            assert node is not None
            update_node_value(node, value)
        elif len(ref.value_nodes) == len(value):
            write_values(ref.value_nodes, value)
        else:
            origins = align_values(entry_value(ref), value) if origins is None else origins
            if ref.collected:
                self._resize_records(key, value, origins)
            else:
                self.refs[key] = resize_entry(ref, self.ctx, key, value, origins)

    def _resize_records(self, key: str, values: Sequence[Any], origins: Sequence[int | None]) -> None:
        """Change how many records a key holds, an entry per record, all or nothing.

        Each new record is written beside the one before it, or before the old first if it
        comes first, as a copy of that neighbour; *origins* is as in ``replace``.
        """
        # Check everything first, and have a new record that cannot be written take the ones
        # written before it back out: a refused list must leave the file as it was.
        nodes = self._check_records(key, values, origins)
        entries = self.refs[key].entry_nodes
        written = self._insert_records(key, values, origins)
        for value, origin in zip(values, origins, strict=True):
            if origin is not None:
                update_node_value(nodes[origin], value)
        for entry in entries:
            if not any(entry is kept for kept in written):
                remove_entry_node(entry, self.ctx.info)
        self.reread()

    def _check_records(self, key: str, values: Sequence[Any], origins: Sequence[int | None]) -> list[Tree]:
        """Return the value node of each old record, refusing a list that cannot be written.

        It runs before anything is changed. An empty list is refused because it would delete
        the key, and a deletion is not an edit.
        """
        if not values:
            raise ValueError(f"A list of records keeps at least one: delete '{key}' to remove them all")
        nodes = [node for node in self.refs[key].value_nodes if node is not None]
        handler = VALUE_TYPE_HANDLER_REGISTRY[str(nodes[0].data)]
        if not all(handler.type_check(value) for value in values):
            raise TypeError(f"Trying to change the type of a record of '{key}'")
        if any(origin is None for origin in origins) and not self.addable:
            raise UnsupportedEntryError(f"Cannot add a record to '{key}': this block has no body in the file for it")
        return nodes

    def _insert_records(self, key: str, values: Sequence[Any], origins: Sequence[int | None]) -> list[Tree]:
        """Write each new record, and return the entry holding every record, in order.

        A new record that cannot be written takes the ones written before it back out, and
        the error ``insert_entry`` raised propagates.
        """
        entries = self.refs[key].entry_nodes
        written: list[Tree] = []
        try:
            for value, origin in zip(values, origins, strict=True):
                if origin is not None:
                    written.append(entries[origin])
                else:
                    neighbour = written[-1] if written else entries[0]
                    written.append(self._insert_record(value, neighbour, after=bool(written)))
        except (ValueError, TypeError):
            for entry in written:
                if not any(entry is old for old in entries):
                    remove_entry_node(entry, self.ctx.info)
            raise
        return written

    def _insert_record(self, value: Any, neighbour: Tree, *, after: bool) -> Tree:
        """Write a record beside another of the same key, as a copy of it.

        The copy takes the neighbour's spacing, spelling of the key and notation of each
        parameter the two share. It goes after *neighbour* if *after*, and before it if not.
        """
        line = line_index(self.tree, neighbour)
        index = entry_insertion_index(self.tree, line) if after else line
        template = donor_template(self.tree.children[line], neighbour, self.ctx.reconstructor)
        style = EntryStyle(indent_of(neighbour), (((1, KEY_VALUE), template),)) if template else EntryStyle()
        ref = insert_entry(self.tree, self.ctx, [str(key_token(neighbour))], value, style, index, KEY_VALUE)
        return ref.entry_nodes[0]

    def add_path(self, raw_keys: Sequence[str], value: Any, category: str | None = None) -> EntryRef:
        """Write an entry for a key path this container does not have yet.

        One key is a plain entry, written after the last entry of the container. More are a
        chain, as in the Fortran ``a%b = 1``: a block written one component per line, each
        line repeating its key. Such a line goes after the last one sharing the most leading
        keys with it, so that a type's components stay together, and spells those keys the
        way that line does.

        The layout comes from the entries above the new one, or -- when there are none, as
        in a block created this session -- from the style handed down when it was created.

        *value* is stored under the innermost key; a ``dict`` creates an empty block, or a
        record if *category* is ``key_value``. The reference returned is to the new entry,
        read on its own: for a chain, to its outermost key, whose block holds the rest.

        Raises:
            ValueError: If a key is unusable, or *value* is a list too short to write.
            UnsupportedEntryError: If the format cannot express *value* here, or this
                container has no body in the file for a new entry to go in.
        """
        if not self.addable:
            raise UnsupportedEntryError(
                f"Cannot add '{raw_keys[-1]}': this block has no body in the file for a new entry to go in. "
                "It is either merged from several blocks, or written one component per line"
            )
        keys = [self.ctx.normalise_key(key) for key in raw_keys]
        index, spelled = self._insertion_point(keys)
        style = probe_entry_style(self.tree, self.ctx.reconstructor, upto=index)
        if style.indent is None:
            style = self.fallback_style
        ref = insert_entry(self.tree, self.ctx, [*spelled, *raw_keys[len(spelled) :]], value, style, index, category)
        if len(keys) == 1:
            self.refs[keys[0]] = ref
        else:
            # The new line may extend a block the file already writes, which reads afresh.
            self.reread()
        return ref

    def _insertion_point(self, keys: Sequence[str]) -> tuple[int, list[str]]:
        """Return where an entry for *keys* goes, and how the keys it shares are spelled.

        That is an index among the children of the container, and the leading keys shared
        with the line the entry follows, as that line spells them.
        """
        if len(keys) == 1:
            return entry_insertion_index(self.tree), []
        after, spelled = None, []
        for position, line in enumerate(self.tree.children):
            for entry in entry_nodes(line) if isinstance(line, Tree) else []:
                shared = []
                for link, key in zip(entry_chain(entry), keys, strict=False):
                    token = key_token(link)
                    if self.ctx.normalise_key(token) != key:
                        break
                    shared.append(str(token))
                if shared and len(shared) >= len(spelled):
                    after, spelled = position, shared
        return entry_insertion_index(self.tree, after), spelled

    def created_block_style(self, entry: Tree | None = None) -> EntryStyle:
        """Return the style the contents of a block just created should follow.

        A new block is empty, so nothing in it can say how its contents should be laid out.
        The nearest sibling block above it answers instead, falling back to this container's
        own style so that a block nested in a new block inherits it too rather than dropping
        back to the template's own indentation.

        *entry* is the entry just written, whose block is new; ``None`` searches every block
        of the container, from the last.
        """
        upto = None if entry is None else line_index(self.tree, entry)
        sibling = probe_sibling_block_style(self.tree, self.ctx.reconstructor, upto)
        return sibling if sibling.indent is not None else self.fallback_style

    def remove(self, key: str) -> None:
        """Unlink every entry that wrote a key, and forget its reference.

        There is more than one when the file assigned the key twice. Only the last
        assignment reached the configuration, so removing only that one would leave the
        earlier entries in the file and the key would come back the next time it was read.
        """
        for entry_node in self.refs[key].entry_nodes:
            remove_entry_node(entry_node, self.ctx.info)
        del self.refs[key]

    def reread(self, ref: EntryRef | None = None) -> dict[str, EntryRef]:
        """Read the entries of the container again, after the tree has changed below it.

        Deleting an entry can take more than the entry with it: the wrappers a grammar
        cannot derive empty go too, up to a container that may be empty. A container above
        the one the deletion was made in may therefore have lost an entry, or hold
        references to nodes that are no longer in the tree.

        The store of a nested configuration is given *ref*, the ``key_block`` reference for
        its block as just read from the container above. The fresh references replace
        ``refs`` and are returned.
        """
        if ref is not None:
            # A block merged from several in the file is a node the reader builds, not one
            # in the parse tree, so reading the container above builds a new one: follow it,
            # or this store goes on reading the old one.
            assert ref.block_node is not None
            self.tree, self.addable = ref.block_node, ref.addable
        self.refs = read_entries(self.tree, self.ctx)
        return self.refs

    def render(self) -> str:
        """Write the tree back out as text, less the trailing newline ``parse`` added.

        A container with no entries left is still written out, because the tree may hold
        comments and blank lines that are genuinely part of the file. Only a grammar that
        cannot derive a container without entries falls back to the empty string, since for
        such a format there is no text left to write.
        """
        if self.refs:
            return self._reconstruct()
        try:
            return self._reconstruct()
        except UnexpectedInput:
            return ""

    def _reconstruct(self) -> str:
        """Write the tree out, less the trailing newline ``parse`` added."""
        # The reconstructor modifies the tree in place, so work on a deep copy.
        text = self.ctx.reconstructor.reconstruct(self.tree.__deepcopy__(None))
        return text[:-1] if text.endswith("\n") else text
