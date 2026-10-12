# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the parse tree behind one configuration.

The package functions the store calls are faked, so each of its five operations can be
checked for what it does and what it delegates. Two things stay real: the ``Reconstructor``,
because writing text back out is lark's job and standing in for it would test nothing, and
the parse trees, which come from a small grammar for the same reason.

The store is the only place that knows a tree is involved at all, so this is where the
delegation is worth pinning: which style an addition is given, and which node a removal
unlinks.
"""

from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import patch

import pytest
from conftest import entry_node, value_node
from lark import Tree
from lark.exceptions import UnexpectedInput

from access.config import config_store
from access.config.config_store import ConfigStore
from access.config.entry_style import EntryStyle
from access.config.grammar_compiled import ParseContext, compile_grammar
from access.config.grammar_values import UnsupportedEntryError
from access.config.tree_navigation import AddParent
from access.config.tree_reader import EntryRef

GRAMMAR = """
    start: line*
    ?line: scalar | listed | blk | chained | record | empty_line
    scalar: ws* key ws* "=" ws* value line_end -> key_value
    listed: ws* key ws* "=" ws* value (ws* "," ws* value)+ line_end -> key_list
    blk: key "<" line_end block ">" line_end -> key_block
    block: (scalar | listed | empty_line)*
    // A block written one component per line, as a Fortran derived type is: "a%b = 1".
    chained: ws* key "%" link -> key_block
    link: component | chained
    component: key ws* "=" ws* value line_end -> key_value
    // One record of a list, each written by an entry of its own: r: "name", "a=1".
    record: ws* key colon ws* named_assignments line_end -> key_value
    colon: ws* ":"
    ?value: integer | string
    empty_line: line_end
    line_end: ws* NEWLINE

    %import config.key
    %import config.integer
    %import config.string
    %import config.named_assignments
    %import config.ws
    %import config.NEWLINE
"""


@pytest.fixture(scope="module")
def ctx() -> ParseContext:
    """Return a context over the grammar above, with a real reconstructor."""
    return ParseContext(compile_grammar(GRAMMAR), True, {}, ("integer", "string", "named_assignments"))


@pytest.fixture(scope="module")
def chain_ctx() -> ParseContext:
    """Return a context reading ``link`` as a block too, with keys that ignore case."""
    return ParseContext(compile_grammar(GRAMMAR), False, {}, ("integer", "string"), ("block", "link"))


@pytest.fixture
def store(ctx) -> ConfigStore:
    """Return a store over a parsed two-entry file."""
    tree = ctx.lark.parse("a = 1\nb = 2\n", start="start")
    AddParent().visit(tree)
    return ConfigStore(tree, ctx)


class TestReading:
    """What a store knows about its container as soon as it is built."""

    def test_reads_the_entries_on_construction(self, store) -> None:
        """Test that the references are collected once, not on demand."""
        assert list(store.refs) == ["a", "b"]
        assert store.refs["a"].category == "key_value"

    def test_starts_with_no_inherited_style(self, store) -> None:
        """Test that only a block created this session is given one."""
        assert store.fallback_style == EntryStyle()

    def test_a_block_gets_a_store_of_its_own(self, ctx) -> None:
        """Test that the contents of a block are a container in their own right."""
        tree = ctx.lark.parse("blk<\n  x = 1\n>\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)

        child = store.child(store.refs["blk"])

        assert child.tree is store.refs["blk"].block_node
        assert list(child.refs) == ["x"]
        assert child.ctx is store.ctx


class TestReplace:
    """Writing a new value into an entry that already exists."""

    def test_a_scalar(self, store) -> None:
        """Test the ordinary edit."""
        store.replace("a", 42)

        assert store.render() == "a = 42\nb = 2"

    def test_a_whole_list(self, ctx) -> None:
        """Test that every element is rewritten in place."""
        tree = ctx.lark.parse("a = 1, 2\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)

        store.replace("a", [7, 8])

        assert store.render() == "a = 7, 8"

    def test_refuses_a_list_for_a_scalar(self, store) -> None:
        """Test that the category is fixed by the parse."""
        with pytest.raises(TypeError, match="change the type"):
            store.replace("a", [1, 2])

    def test_refuses_a_scalar_for_a_list(self, ctx) -> None:
        """Test the same check the other way round."""
        tree = ctx.lark.parse("a = 1, 2\n", start="start")
        AddParent().visit(tree)

        with pytest.raises(TypeError, match="change the type"):
            ConfigStore(tree, ctx).replace("a", 1)

    @pytest.mark.parametrize(
        ("value", "written"),
        [([1, 2, 3], "a = 1, 2, 3"), ([2, 9, 1], "a = 2, 9, 1"), ([1, 2, 3, 4], "a = 1, 2, 3, 4")],
    )
    def test_a_list_of_a_different_length_is_resized(self, ctx, value, written) -> None:
        """Test that a list grows or shrinks around the elements that stay."""
        tree = ctx.lark.parse("a = 1, 2\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)

        store.replace("a", value)

        assert store.render() == written
        assert len(store.refs["a"].value_nodes) == len(value)

    def test_resizing_an_entry_that_is_not_a_list(self, store) -> None:
        """Test that only a list has a length to change."""
        with pytest.raises(TypeError, match="change the type"):
            store.replace("a", [1, 2], [0, None])


class TestAdd:
    """Handing an insertion the style it should follow, and recording what came back."""

    @pytest.fixture
    def spy(self) -> Iterator[tuple[list[dict], EntryRef]]:
        """Patch insertion, recording its arguments and returning a chosen reference."""
        calls: list[dict] = []
        node = entry_node("key_value", "z", value_node())
        ref = EntryRef("key_value", (node,), (node.children[1],))

        def fake_insert(container, ctx, raw_keys, value, style, index, category):
            calls.append({"container": container, "raw_keys": raw_keys, "value": value, "style": style})
            return ref

        with patch.object(config_store, "insert_entry", fake_insert):
            yield calls, ref

    def test_records_the_reference_it_is_given(self, store, spy) -> None:
        """Test that the new entry joins the store's own references."""
        _, ref = spy

        assert store.add_path(("z",), 3) is ref
        assert store.refs["z"] is ref

    def test_passes_the_container_and_the_key_as_written(self, store, spy) -> None:
        """Test that the key travels as the caller spelled it, for the text of the entry."""
        calls, _ = spy
        store.add_path(("z",), 3)

        assert calls[0]["container"] is store.tree
        assert calls[0]["raw_keys"] == ["z"]

    def test_offers_the_indent_of_the_entries_already_there(self, ctx, spy) -> None:
        """Test that a container with entries dictates how the new one is indented."""
        calls, _ = spy
        tree = ctx.lark.parse("  a = 1\n    b = 2\n", start="start")
        AddParent().visit(tree)

        ConfigStore(tree, ctx).add_path(("z",), 3)

        assert calls[0]["style"].indent == "    "

    def test_offers_a_copy_of_the_entries_already_there(self, store, spy) -> None:
        """Test that the store passes its reconstructor on, so whole entries are copied."""
        calls, _ = spy
        store.add_path(("z",), 3)

        assert calls[0]["style"].donor(1, "key_value") is not None

    def test_falls_back_to_the_inherited_style(self, ctx, spy) -> None:
        """Test the block created this session, which has no entry of its own to copy.

        Its contents would otherwise be written with no indentation at all, foreign to a
        file whose blocks are indented.
        """
        calls, _ = spy
        tree = ctx.lark.parse("blk<\n>\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)
        block = store.child(store.refs["blk"])
        block.fallback_style = EntryStyle(indent="    ")

        block.add_path(("z",), 3)

        assert calls[0]["style"].indent == "    "


class TestAddingAChain:
    """A key path written as one line, beside the lines writing the same block."""

    @staticmethod
    def store_of(ctx: ParseContext, text: str) -> ConfigStore:
        """Return a store over *text*."""
        tree = ctx.lark.parse(text, start="start")
        AddParent().visit(tree)
        return ConfigStore(tree, ctx)

    def test_goes_after_the_last_line_sharing_its_keys(self, chain_ctx) -> None:
        """Test that a component joins the others, spelling the type the way they do."""
        store = self.store_of(chain_ctx, "a%b = 1\na%c = 2\nx = 3\n")

        ref = store.add_path(("A", "d"), 4)

        assert store.render() == "a%b = 1\na%c = 2\na%d = 4\nx = 3"
        # The new line on its own; the store reads it together with the other two.
        assert len(ref.entry_nodes) == 1
        assert len(store.refs["A"].entry_nodes) == 3

    def test_follows_the_line_sharing_the_most_keys(self, chain_ctx) -> None:
        """Test that a line sharing two keys wins over a later one sharing only one."""
        store = self.store_of(chain_ctx, "a%b%c = 1\na%d = 2\n")

        store.add_path(("A", "B", "e"), 3)

        assert store.render() == "a%b%c = 1\na%b%e = 3\na%d = 2"

    def test_a_chain_sharing_no_key_goes_after_the_last_entry(self, chain_ctx) -> None:
        """Test a component of a type the file does not write yet."""
        store = self.store_of(chain_ctx, "a%b = 1\nx = 3\n")

        store.add_path(("n", "a"), 1)

        assert store.render() == "a%b = 1\nx = 3\nn%a = 1"


def rec(a: int) -> dict:
    """Return the record ``"f", "a=<a>"``."""
    return {"value": "f", "a": a}


class TestResizingRecords:
    """A list of records, written one entry per record, growing and shrinking."""

    @staticmethod
    def store_of(ctx: ParseContext, text: str) -> ConfigStore:
        """Return a store over *text*."""
        tree = ctx.lark.parse(text, start="start")
        AddParent().visit(tree)
        return ConfigStore(tree, ctx)

    def test_a_new_record_goes_after_the_one_before_it_as_a_copy(self, ctx) -> None:
        """Test that the new line is spaced and punctuated like its neighbour."""
        store = self.store_of(ctx, 'r:  "f","a=1"\nr:  "f","a=2"\nx = 1\n')

        store.replace("r", [rec(1), rec(2), rec(3)], [0, 1, None])

        assert store.render() == 'r:  "f","a=1"\nr:  "f","a=2"\nr:  "f","a=3"\nx = 1'
        assert len(store.refs["r"].entry_nodes) == 3

    def test_a_new_first_record_goes_before_the_old_first(self, ctx) -> None:
        """Test a record put at the front of the list."""
        store = self.store_of(ctx, 'x = 1\nr: "f", "a=1"\n')

        store.replace("r", [rec(0), rec(1)], [None, 0])

        assert store.render() == 'x = 1\nr: "f", "a=0"\nr: "f", "a=1"'

    def test_the_records_that_go_are_removed_and_those_that_stay_written(self, ctx) -> None:
        """Test a list shrinking to one record, which also changes."""
        store = self.store_of(ctx, 'r: "f", "a=1"\nr: "f", "a=2"\nx = 1\n')

        store.replace("r", [rec(5)], [1])

        assert store.render() == 'r: "f", "a=5"\nx = 1'

    @pytest.mark.parametrize(
        ("values", "origins", "error"),
        [
            pytest.param([], [], ValueError, id="none-left"),
            pytest.param([rec(1), 3], [0, None], TypeError, id="not-a-record"),
        ],
    )
    def test_refuses_a_list_it_cannot_write_and_changes_nothing(self, ctx, values, origins, error) -> None:
        """Test the checks made before anything is written."""
        store = self.store_of(ctx, 'r: "f", "a=1"\n')

        with pytest.raises(error):
            store.replace("r", values, origins)

        assert store.render() == 'r: "f", "a=1"'

    def test_refuses_a_new_record_where_there_is_no_body(self, ctx) -> None:
        """Test a block merged from several, which a record could only be written beside."""
        store = self.store_of(ctx, 'r: "f", "a=1"\n')
        store.addable = False

        with pytest.raises(UnsupportedEntryError, match="no body"):
            store.replace("r", [rec(1), rec(2)], [0, None])

    def test_takes_back_the_records_it_wrote_when_one_cannot_be(self, ctx) -> None:
        """Test that a list the file cannot take leaves it as it was."""
        store = self.store_of(ctx, 'r: "f", "a=1"\nx = 1\n')
        real = ConfigStore._insert_record
        calls: list[int] = []

        def second_fails(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise UnsupportedEntryError("refused")
            return real(self, *args, **kwargs)

        with patch.object(ConfigStore, "_insert_record", second_fails), pytest.raises(UnsupportedEntryError):
            store.replace("r", [rec(1), rec(2), rec(3)], [0, None, None])

        assert store.render() == 'r: "f", "a=1"\nx = 1'


class TestUnaddableBlocks:
    """A container with no body in the file for a new entry to go into."""

    def test_refuses_an_addition(self, ctx) -> None:
        """Test that such a store says so rather than splicing where nothing is written."""
        tree = ctx.lark.parse("a = 1\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx, addable=False)

        with pytest.raises(UnsupportedEntryError, match="no body in the file"):
            store.add_path(("z",), 1)

    def test_a_child_inherits_the_flag_from_its_reference(self, ctx) -> None:
        """Test that ``child`` passes on what the reader decided about the block."""
        tree = ctx.lark.parse("blk<\n>\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)
        ref = store.refs["blk"]

        assert store.child(ref).addable
        assert not store.child(replace(ref, addable=False)).addable


class TestCreatedBlockStyle:
    """The style handed down to a block this session created."""

    def test_taken_from_a_sibling_block(self, ctx) -> None:
        """Test that a new block is laid out like the block next to it."""
        tree = ctx.lark.parse("old<\n  p = 1\n>\n", start="start")
        AddParent().visit(tree)

        assert ConfigStore(tree, ctx).created_block_style().indent == "  "

    def test_carries_a_copy_of_the_sibling_entries(self, ctx) -> None:
        """Test that the first entry of a new block is written like those of the last."""
        tree = ctx.lark.parse("old<\n  p = 1\n>\n", start="start")
        AddParent().visit(tree)

        assert ConfigStore(tree, ctx).created_block_style().donor(1, "key_value") is not None

    def test_taken_from_the_block_above_the_one_created(self, ctx) -> None:
        """Test that a block written among others copies the one above it, not the last."""
        tree = ctx.lark.parse("old<\n    p = 1\n>\nnew<\n>\nlast<\n  q = 1\n>\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)

        assert store.created_block_style(store.refs["new"].entry_nodes[0]).indent == "    "

    def test_falls_back_to_this_container_own(self, ctx) -> None:
        """Test that a block nested in a new block inherits rather than going canonical."""
        tree = ctx.lark.parse("blk<\n>\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)
        store.fallback_style = EntryStyle(indent="\t")

        assert store.created_block_style().indent == "\t"


class TestRemove:
    """Taking an entry out of the tree, and out of the references."""

    def test_unlinks_the_entry_and_forgets_it(self, store) -> None:
        """Test both halves, since a reference left behind would be edited into nothing."""
        store.remove("a")

        assert list(store.refs) == ["b"]
        assert store.render() == "b = 2"

    def test_unlinks_every_entry_that_wrote_the_key(self, ctx) -> None:
        """Test the file that assigns a key twice, where only the last is in the dict.

        Removing just that one would leave the earlier line behind, and the key would come
        back the next time the file was read.
        """
        tree = ctx.lark.parse("a = 1\na = 2\nb = 3\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)
        assert len(store.refs["a"].entry_nodes) == 2

        store.remove("a")

        assert store.render() == "b = 3"

    def test_delegates_to_the_tree_edit(self, store) -> None:
        """Test that removal goes through the repair path, not a bare list removal.

        A grammar can require entries and the text around them to alternate, so what is left
        behind may need rejoining; the store must not do the unlink itself.
        """
        seen: list[Tree] = []
        entries = list(store.refs["a"].entry_nodes)

        with patch.object(config_store, "remove_entry_node", lambda node, info: seen.append(node)):
            store.remove("a")

        assert seen == entries


class TestReread:
    """Reading the container again, after the tree changed below it."""

    def test_reports_and_keeps_the_fresh_references(self, store) -> None:
        """Test that an entry gone from the tree is gone from the references."""
        container = store.tree
        container.children.remove(store.refs["a"].entry_nodes[0])

        refs = store.reread()

        assert list(refs) == ["b"]
        assert store.refs is refs

    def test_follows_the_block_a_reference_names(self, ctx) -> None:
        """Test that a nested store is pointed at the block just read for it."""
        tree = ctx.lark.parse("blk<\n  x = 1\n>\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)
        child = store.child(store.refs["blk"])
        merged = replace(store.refs["blk"], block_nodes=(Tree("block", []),), addable=False)

        child.reread(merged)

        assert child.tree is merged.block_node
        assert child.addable is False


class TestRender:
    """Writing the tree back out."""

    def test_reproduces_the_file_less_the_newline_parse_added(self, store) -> None:
        """Test the round trip, which is the property the whole package exists for."""
        assert store.render() == "a = 1\nb = 2"

    def test_works_on_a_copy(self, store) -> None:
        """Test that rendering twice gives the same text.

        The reconstructor mutates the tree it is given, so the store has to hand it a deep
        copy or a second call would see a mangled tree.
        """
        assert store.render() == store.render()

    def test_a_container_with_entries_left_reports_a_failure(self, ctx) -> None:
        """Test that a tree that cannot be written out raises rather than coming back empty.

        Silently returning nothing would lose a file's contents; the empty-string fallback
        is only for a configuration with no entries at all.
        """
        tree = ctx.lark.parse("a = 1\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)

        with patch.object(store.ctx.reconstructor, "reconstruct", _refuse), pytest.raises(UnexpectedInput):
            store.render()

    def test_an_emptied_container_still_writes_its_comments(self, ctx) -> None:
        """Test that what was never an entry survives every entry being deleted.

        Blank lines and comments are genuinely part of the file, so a configuration emptied
        of keys is not an empty file.
        """
        tree = ctx.lark.parse("a = 1\n\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)

        store.remove("a")

        assert store.render() == ""
        assert store.refs == {}

    def test_an_emptied_container_a_grammar_cannot_derive_writes_nothing(self, ctx) -> None:
        """Test the fallback for a format whose start rule requires at least one entry.

        There is no text left to write for such a grammar, so the empty string is the honest
        answer rather than an error.
        """
        tree = ctx.lark.parse("a = 1\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)
        store.remove("a")

        with patch.object(store.ctx.reconstructor, "reconstruct", _refuse):
            assert store.render() == ""


def _refuse(tree: Tree) -> str:
    """Stand in for a reconstructor that cannot derive the tree it was given."""
    raise UnexpectedInput()


def test_render_strips_only_one_trailing_newline(ctx) -> None:
    """Test that a file ending in a blank line keeps it.

    ``parse`` appends exactly one newline, so exactly one comes back off; a second would eat
    a blank line the file really had.
    """
    tree = ctx.lark.parse("a = 1\n\n", start="start")
    AddParent().visit(tree)

    assert ConfigStore(tree, ctx).render() == "a = 1\n"
