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

import pytest
from conftest import entry_node, value_node
from lark import Tree
from lark.exceptions import UnexpectedInput

from access.config import config_store
from access.config.config_store import ConfigStore
from access.config.entry_style import EntryStyle
from access.config.grammar_compiled import ParseContext, compile_grammar
from access.config.tree_navigation import AddParent
from access.config.tree_reader import EntryRef

GRAMMAR = """
    start: line*
    ?line: scalar | listed | blk | empty_line
    scalar: ws* key ws* "=" ws* value line_end -> key_value
    listed: ws* key ws* "=" ws* value (ws* "," ws* value)+ line_end -> key_list
    blk: key "<" line_end block ">" line_end -> key_block
    block: (scalar | listed | empty_line)*
    ?value: integer | string
    empty_line: line_end
    line_end: ws* NEWLINE

    %import config.key
    %import config.integer
    %import config.string
    %import config.ws
    %import config.NEWLINE
"""


@pytest.fixture(scope="module")
def ctx() -> ParseContext:
    """Return a context over the grammar above, with a real reconstructor."""
    return ParseContext(compile_grammar(GRAMMAR), True, {}, ("integer", "string"))


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
        """Test the ordinary edit, and that the node holding the value is handed back."""
        nodes = store.replace("a", 42)

        assert nodes == store.refs["a"].value_nodes
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

    def test_refuses_a_list_of_a_different_length(self, ctx) -> None:
        """Test that a list cannot grow or shrink, since each element has its own node."""
        tree = ctx.lark.parse("a = 1, 2\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)

        for value in ([1], [1, 2, 3]):
            with pytest.raises(ValueError, match="change the length"):
                store.replace("a", value)


class TestAdd:
    """Handing an insertion the style it should follow, and recording what came back."""

    @pytest.fixture
    def spy(self, monkeypatch):
        """Patch insertion, recording its arguments and returning a chosen reference."""
        calls: list[dict] = []
        node = entry_node("key_value", "z", value_node())
        ref = EntryRef("key_value", (node,), (node.children[1],))

        def fake_insert(container, ctx, key, raw_key, value, style):
            calls.append({"container": container, "key": key, "raw_key": raw_key, "value": value, "style": style})
            return ref

        monkeypatch.setattr(config_store, "insert_entry", fake_insert)
        return calls, ref

    def test_records_the_reference_it_is_given(self, store, spy) -> None:
        """Test that the new entry joins the store's own references."""
        _, ref = spy

        assert store.add("z", "z", 3) is ref
        assert store.refs["z"] is ref

    def test_passes_the_container_and_the_key_as_written(self, store, spy) -> None:
        """Test that both spellings travel: the normalised key, and the one for the file."""
        calls, _ = spy
        store.add("Z", "z", 3)

        assert calls[0]["container"] is store.tree
        assert (calls[0]["key"], calls[0]["raw_key"]) == ("Z", "z")

    def test_offers_the_style_of_the_entries_already_there(self, store, spy) -> None:
        """Test that a container with entries dictates how the new one is spaced."""
        calls, _ = spy
        store.add("z", "z", 3)

        assert calls[0]["style"].has_donor
        # One run per whitespace slot the neighbour used: either side of the "=".
        assert calls[0]["style"].key_pads == (" ", " ")

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
        block.fallback_style = EntryStyle(indent="    ", has_donor=True)

        block.add("z", "z", 3)

        assert calls[0]["style"].indent == "    "


class TestCreatedBlockStyle:
    """The style handed down to a block this session created."""

    def test_taken_from_a_sibling_block(self, ctx) -> None:
        """Test that a new block is laid out like the block next to it."""
        tree = ctx.lark.parse("old<\n  p = 1\n>\n", start="start")
        AddParent().visit(tree)

        assert ConfigStore(tree, ctx).created_block_style().indent == "  "

    def test_falls_back_to_this_container_own(self, ctx) -> None:
        """Test that a block nested in a new block inherits rather than going canonical."""
        tree = ctx.lark.parse("blk<\n>\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)
        store.fallback_style = EntryStyle(indent="\t", has_donor=True)

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

    def test_delegates_to_the_tree_edit(self, store, monkeypatch) -> None:
        """Test that removal goes through the repair path, not a bare list removal.

        A grammar can require entries and the text around them to alternate, so what is left
        behind may need rejoining; the store must not do the unlink itself.
        """
        seen: list[Tree] = []
        monkeypatch.setattr(config_store, "remove_entry_node", lambda node, info: seen.append(node))
        entries = list(store.refs["a"].entry_nodes)

        store.remove("a")

        assert seen == entries


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

    def test_a_container_with_entries_left_reports_a_failure(self, ctx, monkeypatch) -> None:
        """Test that a tree that cannot be written out raises rather than coming back empty.

        Silently returning nothing would lose a file's contents; the empty-string fallback
        is only for a configuration with no entries at all.
        """
        tree = ctx.lark.parse("a = 1\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)
        monkeypatch.setattr(store.ctx.reconstructor, "reconstruct", _refuse)

        with pytest.raises(UnexpectedInput):
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

    def test_an_emptied_container_a_grammar_cannot_derive_writes_nothing(self, ctx, monkeypatch) -> None:
        """Test the fallback for a format whose start rule requires at least one entry.

        There is no text left to write for such a grammar, so the empty string is the honest
        answer rather than an error.
        """
        tree = ctx.lark.parse("a = 1\n", start="start")
        AddParent().visit(tree)
        store = ConfigStore(tree, ctx)
        store.remove("a")
        monkeypatch.setattr(store.ctx.reconstructor, "reconstruct", _refuse)

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
