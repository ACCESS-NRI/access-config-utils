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

from access.config.tree_reader import (
    EntryReader,
    EntryRef,
    element_values,
    entry_value,
    merge_blocks,
    read_entries,
)


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


class TestBlocksWrittenMoreThanOnce:
    """A key naming several blocks, as a derived type spread over lines does."""

    def test_merge_blocks_holds_the_children_of_both(self) -> None:
        """Test that the merged node is a view, holding the real children in order.

        Nothing is copied: an edit through this node reaches the nodes Lark built, so the
        file is written back with only the value that changed.
        """
        first = Tree("block", [entry_node("key_value", "b", value_node())])
        second = Tree("block", [entry_node("key_value", "c", value_node())])

        merged = merge_blocks(first, second)

        assert merged.data == "block"
        assert merged.children == [*first.children, *second.children]
        # The sources are untouched, and the children are the same objects.
        assert merged.children[0] is first.children[0]
        assert len(first.children) == 1

    def test_two_blocks_under_one_key_are_merged(self) -> None:
        """Test that both contribute their entries rather than the last one winning."""
        first = entry_node("key_block", "a", Tree("block", [entry_node("key_value", "b", value_node())]))
        second = entry_node("key_block", "a", Tree("block", [entry_node("key_value", "c", value_node())]))

        refs = read_entries(container(first, second), FakeContext())

        assert list(refs) == ["a"]
        assert len(refs["a"].block_node.children) == 2
        assert refs["a"].entry_nodes == (first, second)

    def test_a_merged_block_cannot_be_added_to(self) -> None:
        """Test that a node the parse tree does not contain is marked unaddable.

        Splicing into it would change nothing that gets written out.
        """
        first = entry_node("key_block", "a", Tree("block", []))
        second = entry_node("key_block", "a", Tree("block", []))

        assert read_entries(container(first), FakeContext())["a"].addable
        assert not read_entries(container(first, second), FakeContext())["a"].addable

    def test_block_node_reads_the_first_of_several(self) -> None:
        """Test the convenience view, for the callers that only ever have one block."""
        body = Tree("block", [])
        refs = read_entries(container(entry_node("key_block", "a", body)), FakeContext())

        assert refs["a"].block_node is body
        assert EntryRef("key_value", ()).block_node is None


class TestRepeatedBlocksKeptApart:
    """A format that reads a block written twice as two records rather than one merge."""

    @staticmethod
    def separate() -> FakeContext:
        """Return a context whose primary block rule keeps its repetitions apart."""
        return FakeContext(repeated_blocks="separate")

    def test_each_occurrence_is_kept(self) -> None:
        """Test that both bodies are recorded, in the order the file writes them."""
        first_body = Tree("block", [entry_node("key_value", "b", value_node())])
        second_body = Tree("block", [entry_node("key_value", "c", value_node())])
        first = entry_node("key_block", "a", first_body)
        second = entry_node("key_block", "a", second_body)

        refs = read_entries(container(first, second), self.separate())

        assert refs["a"].block_nodes == (first_body, second_body)
        assert refs["a"].entry_nodes == (first, second)

    def test_every_occurrence_stays_addable(self) -> None:
        """Test that a kept occurrence is a real node in the tree, so it has a body.

        This is what a merged block cannot offer: its node is not in the parse tree, so
        splicing an entry into it would change nothing that gets written out.
        """
        first = entry_node("key_block", "a", Tree("block", []))
        second = entry_node("key_block", "a", Tree("block", []))

        assert read_entries(container(first, second), self.separate())["a"].addable

    def test_a_secondary_block_rule_merges_whatever_the_policy(self) -> None:
        """Test that only the primary block rule repeats as records.

        Repeating one of the others means something else: a Fortran derived type spells one
        component per line, so two of them are two writings of a single key.
        """
        first = entry_node("key_block", "a", Tree("dtype_body", [entry_node("key_value", "b", value_node())]))
        second = entry_node("key_block", "a", Tree("dtype_body", [entry_node("key_value", "c", value_node())]))
        ctx = FakeContext(block_rules=("block", "dtype_body"), repeated_blocks="separate")

        refs = read_entries(container(first, second), ctx)

        assert len(refs["a"].block_nodes) == 1
        assert len(refs["a"].block_node.children) == 2
        assert not refs["a"].addable

    def test_a_secondary_block_rule_cannot_be_added_to(self) -> None:
        """Test that only the format's primary block rule accepts a new entry.

        It is the only Lark start symbol, so it is the only body a synthesised entry can be
        parsed into. A Fortran derived type is held by the other kind.
        """
        entry = entry_node("key_block", "a", Tree("dtype_body", []))
        ctx = FakeContext(block_rules=("block", "dtype_body"))

        assert not read_entries(container(entry), ctx)["a"].addable

    def test_a_key_both_assigned_and_used_as_a_block_is_refused(self) -> None:
        """Test that the two cannot be merged, since that would discard the value.

        Building a block around a value node drops the value with no error, and deleting the
        key afterwards leaves the configuration disagreeing with the file.
        """
        scalar = entry_node("key_value", "a", value_node())
        block = entry_node("key_block", "a", Tree("block", []))

        with pytest.raises(ValueError, match="assigned a value and used as a block"):
            read_entries(container(scalar, block), FakeContext())


def indexed(category: str, key: str, spec: str, *children: Tree) -> Tree:
    """Return an entry node carrying an array qualifier, the ``(3)`` of ``v(3) = 1``."""
    return Tree(category, [key_node(key), Tree("index", [Token("INDEX", spec)]), *children])


def repeat(count: int, inner: Tree | None = None) -> Tree:
    """Return a repeat node: ``n*v``, or ``n*`` when nothing is repeated."""
    children: list[Tree | Token] = [Token("REPEAT_COUNT", f"{count}*")]
    if inner is not None:
        children.append(inner)
    return Tree("repeat", children)


class TestElementValues:
    """One value per element, from the node backing each."""

    def test_a_repeat_expands_once_however_often_it_appears(self) -> None:
        """Test that a run written once is read once, not once per element it covers."""
        node = repeat(3, value_node("integer", "1"))

        assert element_values([node, node, node]) == [1, 1, 1]

    def test_a_hole_is_a_null(self) -> None:
        """Test the array position no entry wrote, which has no node behind it."""
        assert element_values([value_node(text="1"), None, value_node(text="3")]) == [1, None, 3]


class TestGatheringAnIndexedArray:
    """Several indexed entries making up one array."""

    def test_two_entries_become_one_list(self) -> None:
        """Test the case the whole feature exists for: ``v(1) = 1`` beside ``v(2) = 2``."""
        first = indexed("key_value", "v", "1", value_node(text="1"))
        second = indexed("key_value", "v", "2", value_node(text="2"))

        refs = read_entries(container(first, second), FakeContext())

        assert list(refs) == ["v"]
        assert refs["v"].category == "key_list"
        assert entry_value(refs["v"]) == [1, 2]
        # Deleting the key has to take both lines with it.
        assert refs["v"].entry_nodes == (first, second)

    def test_the_lines_need_not_be_in_order(self) -> None:
        """Test that position comes from the qualifier, not from where the line sits."""
        later = indexed("key_value", "v", "2", value_node(text="2"))
        earlier = indexed("key_value", "v", "1", value_node(text="1"))

        refs = read_entries(container(later, earlier), FakeContext())

        assert entry_value(refs["v"]) == [1, 2]

    def test_a_position_no_entry_wrote_reads_as_null(self) -> None:
        """Test the gap a sparse set of indices leaves."""
        refs = read_entries(
            container(
                indexed("key_value", "v", "1", value_node(text="1")),
                indexed("key_value", "v", "3", value_node(text="3")),
            ),
            FakeContext(),
        )

        assert entry_value(refs["v"]) == [1, None, 3]
        assert refs["v"].value_nodes[1] is None

    def test_a_qualifier_says_where_the_values_begin(self) -> None:
        """Test that ``v(3) = 1, 2, 3`` fills three positions from the third."""
        entry = indexed("key_list", "v", "3", value_node(text="1"), value_node(text="2"), value_node(text="3"))

        refs = read_entries(container(entry), FakeContext())

        assert entry_value(refs["v"]) == [1, 2, 3]

    def test_an_unqualified_entry_joins_the_same_array(self) -> None:
        """Test that a plain assignment beside indexed ones starts at position one."""
        refs = read_entries(
            container(
                indexed("key_value", "v", "2", value_node(text="2")),
                entry_node("key_value", "v", value_node(text="1")),
            ),
            FakeContext(),
        )

        assert entry_value(refs["v"]) == [1, 2]

    def test_an_unqualified_entry_first_is_seeded_into_the_array(self) -> None:
        """Test the other order: the values already read take their positions first."""
        refs = read_entries(
            container(
                entry_node("key_list", "v", value_node(text="1"), value_node(text="2")),
                indexed("key_value", "v", "3", value_node(text="3")),
            ),
            FakeContext(),
        )

        assert entry_value(refs["v"]) == [1, 2, 3]

    def test_a_valueless_entry_sets_one_position(self) -> None:
        """Test that ``v(1) =`` beside ``v(2) = 3`` is ``[None, 3]``, not a bare null.

        Storing the null as the whole value would discard the 3.
        """
        refs = read_entries(
            container(
                indexed("key_value", "v", "2", value_node(text="3")),
                indexed("key_null", "v", "1"),
            ),
            FakeContext(),
        )

        assert entry_value(refs["v"]) == [None, 3]

    def test_a_valueless_entry_joins_an_array_already_started(self) -> None:
        """Test the unqualified valueless assignment to a key written with indices."""
        refs = read_entries(
            container(
                indexed("key_value", "v", "2", value_node(text="3")),
                entry_node("key_null", "v"),
            ),
            FakeContext(),
        )

        assert entry_value(refs["v"]) == [None, 3]

    def test_a_valueless_entry_read_first_becomes_position_one(self) -> None:
        """Test ``v =`` followed by ``v(2) = 3``, where the null is seeded into the array.

        The valueless assignment was read as an ordinary entry, so its value has to take the
        position it would have occupied before the indexed one can be placed beside it.
        """
        refs = read_entries(
            container(
                entry_node("key_null", "v"),
                indexed("key_value", "v", "2", value_node(text="3")),
            ),
            FakeContext(),
        )

        assert entry_value(refs["v"]) == [None, 3]

    def test_a_repeat_spread_across_positions_is_not_multiplied(self) -> None:
        """Test the case that makes a gathered array carry its own values.

        The repeat backs positions 1 and 3 with something else at 2, so reading it at each
        appearance would give five values instead of three.
        """
        run = repeat(2, value_node("integer", "1"))
        # One repeat child covering two elements, as "v(1:3:2) = 2*1" parses.
        strided = indexed("key_value", "v", "1:3:2", run)
        middle = indexed("key_value", "v", "2", value_node(text="7"))

        refs = read_entries(container(strided, middle), FakeContext())

        assert entry_value(refs["v"]) == [1, 7, 1]


class TestQualifiersLeftInTheKey:
    """The three shapes not gathered, which keep their text in the key instead."""

    def test_a_multidimensional_index(self) -> None:
        """Test that ``a(1,1)`` and ``a(3,3)`` stay two keys rather than becoming one."""
        refs = read_entries(
            container(
                indexed("key_value", "a", "1,1", value_node(text="1")),
                indexed("key_value", "a", "3,3", value_node(text="2")),
            ),
            FakeContext(),
        )

        assert sorted(refs) == ["a(1,1)", "a(3,3)"]

    def test_an_index_on_a_derived_type(self) -> None:
        """Test that ``a(1)%b`` keeps its qualifier, so ``a(2)%b`` is a different key."""
        refs = read_entries(
            container(
                indexed("key_block", "a", "1", Tree("block", [])),
                indexed("key_block", "a", "2", Tree("block", [])),
            ),
            FakeContext(),
        )

        assert sorted(refs) == ["a(1)", "a(2)"]

    def test_a_valueless_entry_with_a_qualifier_it_cannot_model(self) -> None:
        """Test that the decorated key is used for a valueless assignment too."""
        refs = read_entries(container(indexed("key_null", "a", "1,1")), FakeContext())

        assert list(refs) == ["a(1,1)"]
        assert refs["a(1,1)"].category == "key_null"


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
