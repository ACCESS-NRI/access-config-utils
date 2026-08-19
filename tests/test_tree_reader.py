# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reading a parse tree into the entries it holds.

No grammar is compiled. The reader's only package collaborator is the context, which it asks
one thing -- how to normalise a key -- and every input it takes is a tree, built here
directly.

That is what makes the error paths reachable. Each one is a shape a grammar can produce and
the reader cannot use: an entry aliased to the wrong category, a ``key`` rule whose first
child is not a terminal. Through a real parser each needs a grammar contrived to be wrong;
here each is three lines.
"""

import pytest
from conftest import FakeContext, entry_node, key_node, value_node, ws_node
from lark import Token, Tree

from access.config.tree_reader import EntryReader, EntryRef, entry_value, read_entries


def container(*entries: Tree) -> Tree:
    """Return a start node holding *entries*."""
    return Tree("start", list(entries))


class TestEntryValue:
    """Turning a reference into the Python value it stands for."""

    def test_a_scalar(self) -> None:
        """Test the single-value case, read through the handler its rule names."""
        node = entry_node("key_value", "a", value_node("integer", "42"))
        assert entry_value(EntryRef("key_value", (node,), (node.children[1],))) == 42

    def test_a_list(self) -> None:
        """Test that a list comes back in the order written, each element converted."""
        values = (value_node("integer", "1"), value_node("float", "2.5"))
        node = entry_node("key_list", "a", *values)

        assert entry_value(EntryRef("key_list", (node,), values)) == [1, 2.5]

    def test_a_valueless_entry(self) -> None:
        """Test that ``key_null`` reads as ``None``, having no value node at all."""
        assert entry_value(EntryRef("key_null", (entry_node("key_null", "a"),))) is None


class TestReadEntries:
    """Collecting one reference per key below a container."""

    def test_reads_each_category(self) -> None:
        """Test that all four entry kinds are recognised and given the right shape."""
        block = Tree("block", [])
        refs = read_entries(
            container(
                entry_node("key_value", "a", value_node()),
                entry_node("key_list", "b", value_node(), value_node()),
                entry_node("key_block", "c", block),
                entry_node("key_null", "d"),
            ),
            FakeContext(),
        )

        assert [ref.category for ref in refs.values()] == ["key_value", "key_list", "key_block", "key_null"]
        assert list(refs) == ["a", "b", "c", "d"]
        assert refs["c"].block_node is block

    def test_records_the_nodes_an_edit_will_need(self) -> None:
        """Test that a reference names both the whole entry and the values inside it.

        The entry node is what deletion unlinks; the value nodes are what assignment writes
        into.
        """
        value = value_node("integer", "1")
        entry = entry_node("key_value", "a", value)

        ref = read_entries(container(entry), FakeContext())["a"]

        assert ref.entry_nodes == (entry,)
        assert ref.value_nodes == (value,)
        assert ref.block_node is None

    def test_ignores_whitespace_and_punctuation_among_the_children(self) -> None:
        """Test that only value-type rules count as values, whatever else is there."""
        value = value_node()
        entry = Tree("key_value", [ws_node(), key_node("a"), ws_node(), Tree("equal", []), value])

        assert read_entries(container(entry), FakeContext())["a"].value_nodes == (value,)

    def test_does_not_descend_into_a_block(self) -> None:
        """Test that the entries of a block belong to it, read separately when opened.

        Re-reading a whole tree would otherwise build a second set of references for every
        nested entry, leaving any block handed out earlier detached from the dict.
        """
        inner = entry_node("key_value", "inner", value_node())
        outer = entry_node("key_block", "blk", Tree("block", [inner]))

        assert list(read_entries(container(outer), FakeContext())) == ["blk"]
        assert list(read_entries(Tree("block", [inner]), FakeContext())) == ["inner"]

    def test_a_later_key_replaces_an_earlier_one(self) -> None:
        """Test the duplicate-key case, where the last entry in the file is the one kept."""
        first, second = value_node("integer", "1"), value_node("integer", "2")
        refs = read_entries(
            container(entry_node("key_value", "a", first), entry_node("key_value", "a", second)),
            FakeContext(),
        )

        assert refs["a"].value_nodes == (second,)

    def test_but_every_entry_that_wrote_the_key_is_remembered(self) -> None:
        """Test that a key assigned twice records both entries, in the order written.

        Only the last value reaches the configuration, but deleting the key has to take
        both lines with it or the key returns the next time the file is read.
        """
        early = entry_node("key_value", "a", value_node("integer", "1"))
        late = entry_node("key_value", "a", value_node("integer", "2"))

        refs = read_entries(container(early, late), FakeContext())

        assert refs["a"].entry_nodes == (early, late)

    def test_entries_of_different_categories_accumulate_too(self) -> None:
        """Test that the record spans categories, since the last assignment sets the shape.

        A file may write a key as a scalar and then as a list. The reference is the list --
        that is what the configuration holds -- and both entries still have to go.
        """
        scalar = entry_node("key_value", "a", value_node())
        listed = entry_node("key_list", "a", value_node(), value_node())

        refs = read_entries(container(scalar, listed), FakeContext())

        assert refs["a"].category == "key_list"
        assert refs["a"].entry_nodes == (scalar, listed)

    def test_an_empty_container_reads_as_no_entries(self) -> None:
        """Test the block that was empty in the file, and the file that is only comments."""
        assert read_entries(Tree("block", []), FakeContext()) == {}


class TestKeyNormalisation:
    """The one thing the reader asks of its context."""

    def test_a_case_sensitive_format_keeps_the_spelling(self) -> None:
        """Test the default, which is every format but the Fortran namelist."""
        entry = entry_node("key_value", "MixedCase", value_node())

        assert list(read_entries(container(entry), FakeContext())) == ["MixedCase"]

    def test_a_case_insensitive_format_normalises(self) -> None:
        """Test the namelist, where two spellings of a key have to be one entry."""
        entry = entry_node("key_value", "MixedCase", value_node())

        assert list(read_entries(container(entry), FakeContext(case_sensitive=False))) == ["MIXEDCASE"]

    def test_two_spellings_become_one_entry(self) -> None:
        """Test the consequence: a file naming a key twice in different case has it once."""
        refs = read_entries(
            container(
                entry_node("key_value", "dt", value_node("integer", "1")),
                entry_node("key_value", "DT", value_node("integer", "2")),
            ),
            FakeContext(case_sensitive=False),
        )

        assert list(refs) == ["DT"]


class TestRejectedShapes:
    """Trees a grammar can produce and the reader cannot use.

    Each is an aliasing mistake in a grammar. Reporting them is what turns a silent
    misreading -- a list stored as a scalar, an entry with no key -- into a failed parse.
    """

    def test_a_key_value_holding_more_than_one_value(self) -> None:
        """Test the rule that should have been aliased ``key_list``."""
        entry = entry_node("key_value", "a", value_node(), value_node())

        with pytest.raises(ValueError, match="More than one value"):
            read_entries(container(entry), FakeContext())

    def test_a_key_value_holding_no_value(self) -> None:
        """Test the rule that should have been aliased ``key_null``."""
        with pytest.raises(ValueError, match="No values found"):
            read_entries(container(entry_node("key_value", "a")), FakeContext())

    def test_a_key_list_holding_no_value(self) -> None:
        """Test the same check on the list callback, which shares the helper."""
        with pytest.raises(ValueError, match="No values found"):
            read_entries(container(entry_node("key_list", "a")), FakeContext())

    def test_an_entry_with_no_key_rule(self) -> None:
        """Test the rule matching a value with nothing naming it."""
        with pytest.raises(ValueError, match="No 'key' rule nodes found"):
            read_entries(container(Tree("key_value", [value_node()])), FakeContext())

    def test_a_key_rule_whose_first_child_is_not_a_terminal(self) -> None:
        """Test the contract that the key text is a ``Token``, not a nested rule."""
        entry = Tree("key_value", [Tree("key", [Tree("integer", [])]), value_node()])

        with pytest.raises(TypeError, match="No key found"):
            read_entries(container(entry), FakeContext())

    def test_a_block_entry_whose_body_is_not_a_block_node(self) -> None:
        """Test that an unrecognised body is skipped rather than misread.

        ``key_block`` scans its children for a rule named ``block`` and reports nothing if
        there is none, which is how a grammar aliasing the body shows up as a missing key.
        """
        entry = entry_node("key_block", "blk", Tree("body", []))

        assert read_entries(container(entry), FakeContext()) == {}

    def test_the_first_key_rule_is_the_entry_own(self) -> None:
        """Test that an entry holding a nested key takes the outer one.

        A ``key_block`` node has the keys of its contents below it, and the first in
        order is always the block's own.
        """
        inner = entry_node("key_value", "inner", value_node())
        entry = entry_node("key_block", "outer", Tree("block", [inner]))

        assert list(read_entries(container(entry), FakeContext())) == ["outer"]


def test_the_reader_can_be_reused() -> None:
    """Test that one reader reads two containers without carrying entries between them."""
    reader = EntryReader(FakeContext())

    first = reader.read(container(entry_node("key_value", "a", value_node())))
    second = reader.read(container(entry_node("key_value", "b", value_node())))

    assert list(first) == ["a"]
    assert list(second) == ["b"]


def test_a_token_child_of_the_container_is_skipped() -> None:
    """Test that a bare terminal among the container's children is not visited as a rule."""
    entry = entry_node("key_value", "a", value_node())

    assert list(read_entries(Tree("start", [Token("NEWLINE", "\n"), entry]), FakeContext())) == ["a"]
