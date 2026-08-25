# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for finding one's way around a parse tree.

``tree_navigation`` imports nothing but rule names, so there is nothing to fake here. The
trees are built directly instead of parsed: these functions read ``.data`` and ``.children``
and nothing else, so a literal is a faithful input, and it is the only way to lay out the
awkward arrangements -- a comment run split by a blank line, a container holding no entry at
all -- without a grammar contrived to produce each one.
"""

import pytest
from conftest import entry_node, value_node, ws_node
from lark import Token, Tree

from access.config.tree_navigation import (
    AddParent,
    contains_entry,
    entries_of,
    entry_insertion_index,
    entry_nodes,
    flatten_slots,
    index_positions,
    is_comment_line,
    is_value_node,
    node_values,
    place_indexed,
    repeat_parts,
    token_value,
)


def line(*children: Tree | Token) -> Tree:
    """Return a line node wrapping whatever it holds, as a grammar's line rule would."""
    return Tree("empty_line", list(children))


def comment_line(text: str = "# note", rule: str = "comment") -> Tree:
    """Return a line holding nothing but a comment."""
    return line(Tree(rule, [Token("COMMENT", text)]))


def scalar(key: str = "a") -> Tree:
    """Return a ``key_value`` entry node."""
    return entry_node("key_value", key, value_node())


class TestAddParent:
    """The ``.parent`` back-reference, which lark does not provide."""

    def test_annotates_every_rule_node(self) -> None:
        """Test that each rule node below the root learns which node holds it."""
        inner, leaf = Tree("key", [Token("CNAME", "a")]), Tree("integer", [Token("V", "1")])
        entry = Tree("key_value", [inner, leaf])
        root = Tree("start", [entry])

        AddParent().visit(root)

        assert entry.parent is root
        assert inner.parent is entry
        assert leaf.parent is entry

    def test_leaves_tokens_alone(self) -> None:
        """Test that terminals are skipped: only rule nodes are ever walked upwards."""
        token = Token("CNAME", "a")
        AddParent().visit(Tree("key", [token]))

        assert not hasattr(token, "parent")

    def test_refuses_a_node_parented_twice(self) -> None:
        """Test the assertion guarding against a tree node reused in two places.

        A node object appearing under two parents makes ``.parent`` a lie, and the insertion
        path relies on it to find the container an entry sits in. Re-visiting an annotated
        tree is the way that happens by accident.
        """
        root = Tree("start", [Tree("key_value", [])])
        AddParent().visit(root)

        with pytest.raises(AssertionError):
            AddParent().visit(root)


class TestFindingEntries:
    """Which nodes count as an entry, and where the search stops."""

    def test_entry_nodes_finds_one_at_the_top(self) -> None:
        """Test that a node that is itself an entry is returned as one."""
        node = scalar()
        assert entry_nodes(node) == [node]

    def test_entry_nodes_looks_through_a_line(self) -> None:
        """Test that an entry nested inside a line node is still found.

        A Fortran namelist puts several assignments on one line, so the entry is not a
        direct child of the container.
        """
        first, second = scalar("a"), scalar("b")
        assert entry_nodes(line(first, Token("COMMA", ","), second)) == [first, second]

    def test_entry_nodes_stops_at_a_nested_block(self) -> None:
        """Test that the entries of a block belong to the block, not to what holds it."""
        inner = scalar("x")
        block = entry_node("key_block", "blk", Tree("block", [inner]))

        # The block entry itself is found; what is inside it is not.
        assert entry_nodes(block) == [block]
        assert entries_of(Tree("start", [block])) == [block]
        assert entries_of(Tree("block", [inner])) == [inner]

    def test_contains_entry(self) -> None:
        """Test the predicate telling a child that holds an entry from one that does not."""
        assert contains_entry(scalar())
        assert contains_entry(line(scalar()))
        assert not contains_entry(line())
        assert not contains_entry(comment_line())
        # A token child is never an entry.
        assert not contains_entry(Token("NEWLINE", "\n"))


class TestCommentLines:
    """Telling a comment-only line from a blank one, and from an entry that carries one."""

    @pytest.mark.parametrize("rule", ["comment", "fortran_comment"])
    def test_a_line_holding_only_a_comment(self, rule) -> None:
        """Test that either comment rule marks its line as a comment line."""
        assert is_comment_line(comment_line(rule=rule))

    def test_a_blank_line_is_not_one(self) -> None:
        """Test that a blank line, the same rule, is told apart by the missing node."""
        assert not is_comment_line(line())

    def test_an_entry_with_a_trailing_comment_is_not_one(self) -> None:
        """Test that a comment belonging to an entry does not make its line a comment.

        MOM6 documents nearly every parameter with a comment starting on its own line, so
        getting this wrong would make a new entry land in the middle of the documentation.
        """
        assert not is_comment_line(line(scalar(), Tree("comment", [Token("C", "# why")])))

    def test_a_token_is_not_one(self) -> None:
        """Test that a terminal child is never a comment line."""
        assert not is_comment_line(Token("NEWLINE", "\n"))

    def test_a_comment_nested_deeper_still_counts(self) -> None:
        """Test that the search for a comment node goes all the way down."""
        assert is_comment_line(line(Tree("line_end", [Tree("comment", [Token("C", "# deep")])])))


class TestEntryInsertionIndex:
    """Where a new entry belongs among a container's children."""

    def test_empty_container(self) -> None:
        """Test that the first entry of an empty container goes at the front."""
        assert entry_insertion_index(Tree("start", [])) == 0

    def test_container_with_no_entry_but_a_leading_comment(self) -> None:
        """Test that a header comment is stepped over even with no entry to start from.

        The scan starts at zero, so a file that is nothing but a comment block still has its
        first entry placed after the block rather than above it.
        """
        assert entry_insertion_index(Tree("start", [comment_line(), comment_line()])) == 2

    def test_after_the_last_entry(self) -> None:
        """Test that a new entry follows the last one, before any trailing blank line.

        Appending instead would put it after the blank line -- and, in a block, outside the
        block altogether.
        """
        container = Tree("start", [scalar("a"), scalar("b"), line()])
        assert entry_insertion_index(container) == 2

    def test_steps_over_a_trailing_comment_run(self) -> None:
        """Test that a comment run touching the insertion point is stepped over.

        One past the entry is the second line of a comment that began on the entry's own
        line, so the entry belongs after the whole run rather than inside it.
        """
        container = Tree("start", [scalar("a"), comment_line(), comment_line()])
        assert entry_insertion_index(container) == 3

    def test_stops_at_a_blank_line(self) -> None:
        """Test that a blank line ends the run, putting the entry between two comments."""
        container = Tree("start", [scalar("a"), comment_line(), line(), comment_line()])
        assert entry_insertion_index(container) == 2

    def test_stops_at_the_next_entry(self) -> None:
        """Test that the run ends at an entry, so the index is one past the *last* entry."""
        container = Tree("start", [scalar("a"), comment_line(), scalar("b"), line()])
        assert entry_insertion_index(container) == 3


def repeat(count: int, inner: Tree | None = None) -> Tree:
    """Return a repeat node: ``n*v``, or ``n*`` when nothing is repeated."""
    children: list[Tree | Token] = [Token("REPEAT_COUNT", f"{count}*")]
    if inner is not None:
        children.append(inner)
    return Tree("repeat", children)


class TestReadingValues:
    """What a value node contributes to its entry.

    One value for an ordinary node, and *n* for ``n*v`` -- which is why a node and a list
    element are no longer one to one.
    """

    def test_a_plain_value_node_is_one_value(self) -> None:
        """Test the ordinary case, where the node and the element correspond."""
        node = value_node("integer", "7")

        assert is_value_node(node)
        assert token_value(node) == 7
        assert node_values(node) == [7]

    def test_a_repeat_is_the_value_that_many_times(self) -> None:
        """Test ``3*1``, which is written where one value goes and means three."""
        node = repeat(3, value_node("integer", "1"))

        assert is_value_node(node)
        assert node_values(node) == [1, 1, 1]

    def test_a_repeat_of_nulls_names_no_value(self) -> None:
        """Test the bare ``n*`` form, which stands for that many unset elements."""
        assert node_values(repeat(2)) == [None, None]

    def test_repeat_parts_splits_the_count_from_what_it_repeats(self) -> None:
        """Test the two shapes a repeat node comes in."""
        inner = value_node("float", "9.0")

        assert repeat_parts(repeat(6, inner)) == (6, inner)
        assert repeat_parts(repeat(4)) == (4, None)

    def test_what_is_not_a_value_node(self) -> None:
        """Test that only a value-type rule or a repeat counts."""
        assert not is_value_node(Tree("key", [Token("CNAME", "a")]))
        assert not is_value_node(ws_node())
        assert not is_value_node(Token("NEWLINE", "\n"))


def index_node(spec: str) -> Tree:
    """Return an array qualifier node holding *spec*, the ``1:3:2`` of ``v(1:3:2)``."""
    return Tree("index", [Token("INDEX", spec)])


class TestIndexPositions:
    """Working out where an indexed entry's values land in the array its key names."""

    @pytest.mark.parametrize(
        ("spec", "count", "expected"),
        [
            ("3", 1, [3]),
            # A qualifier says where the values *begin*, not how many there may be.
            ("3", 3, [3, 4, 5]),
            ("2:5", 4, [2, 3, 4, 5]),
            ("1:7:2", 4, [1, 3, 5, 7]),
            # Implicit bounds are filled in from how many values there are.
            ("2:", 3, [2, 3, 4]),
            (":5", 3, [3, 4, 5]),
            (":", 2, [1, 2]),
            # Positions are as written, so they may start anywhere.
            ("0", 2, [0, 1]),
            ("-1", 2, [-1, 0]),
            ("5:1:-2", 3, [5, 3, 1]),
        ],
    )
    def test_a_qualifier_it_models(self, spec, count, expected) -> None:
        """Test each shape of qualifier this understands."""
        assert index_positions(index_node(spec), count) == expected

    def test_no_qualifier_at_all(self) -> None:
        """Test the unqualified assignment, whose values simply follow on."""
        assert index_positions(None, 3) is None

    @pytest.mark.parametrize("spec", ["1,1", "1:2:3:4", "n", "1:n", "1:5:0", ""])
    def test_a_qualifier_it_does_not_model(self, spec) -> None:
        """Test the ones left alone, which keep their text in the key instead.

        A multidimensional index addresses an array of arrays, which this flat model cannot
        hold; a non-numeric bound names nothing it can place; a zero stride goes nowhere;
        and an empty qualifier names no position at all. Reporting ``None`` is what stops
        ``v(1,1)`` and ``v(3,3)`` reading as one key.
        """
        assert index_positions(index_node(spec), 2) is None


class TestPlacingAndFlattening:
    """Recording where each entry's values sit, then laying the array out."""

    def test_places_at_the_positions_given(self) -> None:
        """Test the ordinary indexed entry."""
        slots: dict[int, tuple[object, Tree | None]] = {}
        first, second = value_node(text="1"), value_node(text="2")

        place_indexed(slots, [1, 2], [1, 2], [first, second])

        assert slots == {1: (1, first), 2: (2, second)}

    def test_an_unqualified_entry_starts_at_one(self) -> None:
        """Test that an unqualified assignment joins the array Fortran would number from."""
        slots: dict[int, tuple[object, Tree | None]] = {}
        node = value_node()

        place_indexed(slots, None, [7], [node])

        assert slots == {1: (7, node)}

    def test_a_later_entry_overwrites_the_position(self) -> None:
        """Test that the last assignment to a position wins, as it does for a whole key."""
        slots: dict[int, tuple[object, Tree | None]] = {1: (1, value_node())}
        later = value_node(text="9")

        place_indexed(slots, [1], [9], [later])

        assert slots == {1: (9, later)}

    def test_flatten_fills_the_gaps_with_nulls(self) -> None:
        """Test that a position no entry wrote reads as null, backed by no node.

        Having no node is what makes it unwritable: there is nowhere in the file to put a
        value for it.
        """
        first, third = value_node(text="1"), value_node(text="3")

        values, nodes = flatten_slots({1: (1, first), 3: (3, third)})

        assert values == [1, None, 3]
        assert nodes == [first, None, third]

    def test_flatten_starts_at_the_lowest_position_written(self) -> None:
        """Test that the array spans what the file wrote, wherever that begins."""
        values, _ = flatten_slots({5: (5, value_node()), 7: (7, value_node())})

        assert values == [5, None, 7]
