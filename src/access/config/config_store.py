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
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from lark import Tree
from lark.exceptions import UnexpectedInput

from access.config.config_insert import insert_entry
from access.config.entry_style import EntryStyle, probe_entry_style, probe_sibling_block_style
from access.config.grammar_contract import KEY_LIST, KEY_VALUE
from access.config.grammar_values import UnsupportedEntryError
from access.config.tree_edits import remove_entry_node, update_node_value, write_values
from access.config.tree_reader import EntryRef, read_entries

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
    # Whitespace for a new entry when this container holds none to copy from. Set on a block
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

        Args:
            ref (EntryRef): A ``key_block`` reference read from this container.
            index (int): Which occurrence to open, for a block the file writes more than
                once and this format reads as separate records. Zero for every other block,
                which has exactly one.

        Returns:
            ConfigStore: A store over that block's own contents.
        """
        return ConfigStore(ref.block_nodes[index], self.ctx, addable=ref.addable)

    def replace(self, key: str, value: Any) -> tuple[Tree, ...]:
        """Write a new value into the nodes of an entry that already exists.

        Args:
            key (str): The normalised key.
            value (Any): The new value: a scalar, or a whole list of the same length.

        Returns:
            tuple[Tree, ...]: The value-type rule nodes now holding the value.

        Raises:
            TypeError: If the new value's type does not match what the entry holds.
            ValueError: If a list of a different length was given.
        """
        ref = self.refs[key]
        if isinstance(value, list):
            if ref.category != KEY_LIST:
                raise TypeError(f"Trying to change the type of variable '{key}'")
            if len(ref.value_nodes) != len(value):
                raise ValueError(f"Trying to change the length of list '{key}'")
            written = tuple(write_values(ref.value_nodes, value, self.ctx.lark))
            self.refs[key] = replace(ref, value_nodes=written)
            return written
        else:
            if ref.category != KEY_VALUE:
                raise TypeError(f"Trying to change the type of variable '{key}'")
            update_node_value(ref.value_nodes[0], value)
        return ref.value_nodes

    def add(self, key: str, raw_key: str, value: Any) -> EntryRef:
        """Write an entry for a key this container does not have yet.

        The whitespace comes from the entries already in this container, or -- when there
        are none, as in a block created this session -- from the style handed down when it
        was created.

        Args:
            key (str): The normalised key, used to check what the candidate read back as.
            raw_key (str): The key as the caller wrote it, used for the file text.
            value (Any): The value to store. A ``dict`` creates an empty block.

        Returns:
            EntryRef: Where the new entry landed.

        Raises:
            ValueError: If *raw_key* is unusable, or *value* is a list too short to write.
            UnsupportedEntryError: If the format cannot express *value* here, or this
                container has no body in the file for a new entry to go in.
        """
        if not self.addable:
            raise UnsupportedEntryError(
                f"Cannot add '{key}': this block has no body in the file for a new entry to go in. "
                "It is either merged from several blocks, or written one component per line"
            )
        style = probe_entry_style(self.tree)
        if not style.has_donor:
            style = self.fallback_style
        ref = insert_entry(self.tree, self.ctx, key, raw_key, value, style)
        self.refs[key] = ref
        return ref

    def created_block_style(self) -> EntryStyle:
        """Return the style the contents of a block just created should follow.

        A new block is empty, so nothing in it can say how its contents should be laid out.
        The nearest sibling block answers instead, falling back to this container's own
        style so that a block nested in a new block inherits it too rather than dropping
        back to the canonical layout.

        Returns:
            EntryStyle: The style to hand to the new block's store.
        """
        sibling = probe_sibling_block_style(self.tree)
        return sibling if sibling.has_donor else self.fallback_style

    def remove(self, key: str) -> None:
        """Unlink every entry that wrote a key, and forget its reference.

        There is more than one when the file assigned the key twice. Only the last
        assignment reached the configuration, so removing only that one would leave the
        earlier entries in the file and the key would come back the next time it was read.

        Args:
            key (str): The normalised key.
        """
        for entry_node in self.refs[key].entry_nodes:
            remove_entry_node(entry_node, self.ctx.info)
        del self.refs[key]

    def render(self) -> str:
        """Write the tree back out as text.

        A container with no entries left is still written out, because the tree may hold
        comments and blank lines that are genuinely part of the file. Only a grammar that
        cannot derive a container without entries falls back to the empty string, since for
        such a format there is no text left to write.

        Returns:
            str: The reconstructed text, with the trailing newline ``parse`` added removed.
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
