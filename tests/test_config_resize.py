# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for writing a list of a different length into the entry that holds it.

The grammar below is small but real, because what is under test is the round trip through
Lark: the text written out with markers, the candidates parsed back, and the check that a
candidate changes nothing outside the entry. Its lines wrap their entry, so that the line
ending sits outside the entry and the way down from the line to the entry is not empty, and
a list may continue onto the next line after a comma, so that both layouts are reachable.

The candidates are scripted by patching ``layout_candidates`` only where a rejection branch
needs a candidate no real layout produces.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest
from lark import Tree

from access.config import config_resize
from access.config.config_resize import resize_entry
from access.config.config_store import ConfigStore
from access.config.grammar_compiled import ParseContext, compile_grammar
from access.config.grammar_values import UnsupportedEntryError
from access.config.list_layout import align_values
from access.config.tree_navigation import AddParent
from access.config.tree_reader import EntryRef, entry_value

GRAMMAR = """
    start: (line | empty_line)*
    line: listed line_end
    listed: ws* key eq ws* value (comma ws* value | comma line_end ws* value)+ -> key_list
    eq: ws* "="
    comma: ws* ","
    ?value: integer | string
    empty_line: line_end
    line_end: (comment | ws*) NEWLINE

    %import config.key
    %import config.integer
    %import config.string
    %import config.comment
    %import config.ws
    %import config.NEWLINE
"""


@pytest.fixture(scope="module")
def ctx() -> ParseContext:
    """Return a context over the grammar above, with a real reconstructor."""
    return ParseContext(compile_grammar(GRAMMAR), True, {}, ("integer", "string"))


def store_of(ctx: ParseContext, text: str) -> ConfigStore:
    """Return a store over *text*, parsed as the grammar's start rule."""
    tree = ctx.lark.parse(text, start="start")
    AddParent().visit(tree)
    return ConfigStore(tree, ctx)


def resize(store: ConfigStore, values: list, origins: list | None = None) -> EntryRef:
    """Resize the list under ``a``, matching old and new values unless told how."""
    ref = store.refs["a"]
    if origins is None:
        origins = align_values(entry_value(ref), values)
    store.refs["a"] = resize_entry(ref, store.ctx, "a", values, origins)
    return store.refs["a"]


class TestResizing:
    """Lists that grow and shrink, laid out around the elements that stay."""

    @pytest.mark.parametrize(
        ("values", "written"),
        [
            pytest.param([1, 2, 3], "a = 1, 2, 3", id="append"),
            pytest.param([0, 1, 2], "a = 0, 1, 2", id="prepend"),
            pytest.param([1, 9, 2], "a = 1, 9, 2", id="insert"),
        ],
    )
    def test_an_inline_list(self, ctx, values, written) -> None:
        """Test a list on one line, where a new element copies the inline separator."""
        store = store_of(ctx, "a = 1, 2\n")

        resize(store, values)

        assert store.render() == written

    def test_a_list_a_line_per_element(self, ctx) -> None:
        """Test that the comment closing the list stays on its last line.

        The comment is in the line ending, outside the entry, so only the reading where the
        new last element takes the closing line end leaves the line as it was.
        """
        store = store_of(ctx, "a = 1,\n    2  # two\n")

        resize(store, [1, 2, 3])

        assert store.render() == "a = 1,\n    2,\n    3  # two"

    def test_a_comment_between_elements_is_not_copied(self, ctx) -> None:
        """Test that a new element ends its line like the others, without their comment."""
        store = store_of(ctx, "a = 1,  # one\n    2,\n    3\n")

        resize(store, [1, 2, 3, 4], [0, 1, 2, None])

        assert store.render() == "a = 1,  # one\n    2,\n    3,\n    4"

    def test_a_windows_line_end_survives_a_comment_before_it(self, ctx) -> None:
        """Test that a new element ends its line as the others do, carriage return and all.

        The comment rule runs up to the line feed, so it holds the carriage return, and
        blanking the comment must not take that too. The last line feed is the one the
        store drops on writing out, as it drops the newline ``parse`` adds.
        """
        store = store_of(ctx, "a = 1,  # one\r\n    2\r\n")

        resize(store, [1, 2, 3])

        assert store.render() == "a = 1,  # one\r\n    2,\r\n    3\r"

    def test_keeps_the_entry_node(self, ctx) -> None:
        """Test that the references to the entry, and so the dict's, stay good."""
        store = store_of(ctx, "a = 1, 2\n")
        entry = store.refs["a"].entry_nodes[0]

        ref = resize(store, [1, 2, 3])

        assert ref.entry_nodes == (entry,)
        assert [node.parent for node in ref.value_nodes] == [entry] * 3

    def test_a_new_value_of_another_type(self, ctx) -> None:
        """Test that a value its neighbours cannot hold takes the format's preference."""
        store = store_of(ctx, "a = 1, 2\n")

        resize(store, [1, 2, "x"], [0, 1, None])

        assert store.render() == 'a = 1, 2, "x"'

    def test_a_changed_value_keeps_its_place(self, ctx) -> None:
        """Test that a value taking an old element's place is written in its notation."""
        store = store_of(ctx, "a = 1, 2\n")

        resize(store, [1, 7, 3], [0, 1, None])

        assert store.render() == "a = 1, 7, 3"

    def test_unchanged_values_keep_their_text(self, ctx) -> None:
        """Test that a value standing in for an equal old one is written as it was."""
        store = store_of(ctx, "a = 'x', 2\n")

        resize(store, ["x", 2, 3], [0, 1, None])

        assert store.render() == "a = 'x', 2, 3"


class TestRefused:
    """What cannot be resized, and why."""

    def test_fewer_than_two_elements(self, ctx) -> None:
        """Test that a list keeps at least two elements, since one is a scalar."""
        store = store_of(ctx, "a = 1, 2\n")

        with pytest.raises(ValueError, match="at least two elements"):
            resize(store, [1], [0])

    @pytest.mark.parametrize(
        ("change", "reason"),
        [
            pytest.param(lambda ref: replace(ref, entry_nodes=ref.entry_nodes * 2), "more than once", id="twice"),
            pytest.param(lambda ref: replace(ref, values=(1, 2)), "array qualifiers", id="qualified"),
            pytest.param(lambda ref: replace(ref, value_nodes=(None, ref.value_nodes[1])), "unset", id="skipped"),
            pytest.param(
                lambda ref: replace(ref, value_nodes=(Tree("repeat", []), ref.value_nodes[1])),
                "repeat count",
                id="repeat",
            ),
        ],
    )
    def test_values_not_written_one_per_node(self, ctx, change, reason) -> None:
        """Test each shape whose values are not one token apiece in a single entry."""
        store = store_of(ctx, "a = 1, 2\n")
        store.refs["a"] = change(store.refs["a"])

        with pytest.raises(ValueError, match=reason):
            resize(store, [1, 2, 3], [0, 1, None])
        assert store.render() == "a = 1, 2"

    def test_an_element_standing_in_for_one_of_another_type(self, ctx) -> None:
        """Test that a value keeps the type of the element whose place it takes."""
        store = store_of(ctx, "a = 1, 2\n")

        with pytest.raises(TypeError, match="change value type"):
            resize(store, ["x", 2, 3], [0, 1, None])

    def test_a_new_value_no_type_can_hold(self, ctx) -> None:
        """Test that a value the format has no notation for is refused."""
        store = store_of(ctx, "a = 1, 2\n")

        with pytest.raises(UnsupportedEntryError, match="cannot store"):
            resize(store, [1, 2, object()], [0, 1, None])


class TestCandidates:
    """The checks a candidate text has to pass before it is used."""

    def test_the_first_candidate_that_passes_every_check_is_used(self, ctx) -> None:
        """Test each reason a candidate is passed over, then one that is used.

        Two lines are two nodes; a blank line reads back as no list; a comment added to the
        line ending is outside the entry, and so changes the line, not the list.
        """
        store = store_of(ctx, "a = 1, 2\n")
        candidates = ["a = 1, 2, 3\nb = 4, 5\n", "\n", "a = 1, 2, 3  # new\n", "a = 1, 2, 3\n"]

        with patch.object(config_resize, "layout_candidates", lambda *args: candidates):
            resize(store, [1, 2, 3], [0, 1, None])

        assert store.render() == "a = 1, 2, 3"

    @pytest.mark.parametrize(
        "candidate",
        [
            pytest.param("a = = 1\n", id="does-not-parse"),
            pytest.param("a = 1, 2, 4\n", id="other-values"),
            pytest.param("a = 1, 2, 3  # new\n", id="changes-the-line-outside-the-entry"),
        ],
    )
    def test_no_candidate_reads_back(self, ctx, candidate) -> None:
        """Test that a list no layout can write is refused, leaving the file alone.

        The last candidate reads back as the list asked for, but its comment is in the line
        ending, outside the entry. Putting the new entry into the old line would leave that
        comment out, so the line would not be written as the candidate says: that is refused
        rather than written differently.
        """
        store = store_of(ctx, "a = 1, 2\n")

        with (
            patch.object(config_resize, "layout_candidates", lambda *args: [candidate]),
            pytest.raises(UnsupportedEntryError, match="No layout"),
        ):
            resize(store, [1, 2, 3], [0, 1, None])
        assert store.render() == "a = 1, 2"

    def test_a_candidate_the_reader_rejects(self, ctx) -> None:
        """Test that a candidate whose entry cannot be read is passed over."""
        store = store_of(ctx, "a = 1, 2\n")

        def refuse(tree, context):
            raise ValueError("unreadable")

        with (
            patch.object(config_resize, "read_entries", refuse),
            pytest.raises(UnsupportedEntryError),
        ):
            resize(store, [1, 2, 3], [0, 1, None])
