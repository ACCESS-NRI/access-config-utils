# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for changing a parse tree without losing the ability to write it back out.

Most of this needs no grammar: the nodes are built directly and the one package
collaborator, ``GrammarInfo.is_repetition_rule``, is faked so that merging can be asked
about any rule, regardless of what a real grammar happens to define.

``parse_entry_nodes`` is the exception. It takes a ``Lark`` and calls it, so the parser is
real -- and the branch worth pinning, the empty node a rule deriving the empty string
produces, exists only because Lark produces it.
"""

import pytest
from conftest import entry_node, value_node
from lark import Token, Tree
from lark.exceptions import UnexpectedInput

from access.config.grammar_compiled import compile_grammar
from access.config.tree_edits import (
    _runs_of,
    _source_token,
    _value_texts,
    merge_adjacent_repetitions,
    parse_entry_nodes,
    prune_empty_wrappers,
    remove_entry_node,
    replace_value_node,
    splice_entry_nodes,
    update_node_value,
    write_values,
)
from access.config.tree_navigation import AddParent, is_value_node


class FakeInfo:
    """Stands in for ``GrammarInfo`` where only the merge test is consulted.

    Args:
        repetitions: Names of the rules to report as a plain repetition.
    """

    def __init__(self, *repetitions: str) -> None:
        self._repetitions = set(repetitions)

    def is_repetition_rule(self, name: str) -> bool:
        """Report whether *name* was declared a repetition."""
        return name in self._repetitions


class TestUpdateNodeValue:
    """Writing a new value into the token a value-type node holds."""

    def test_replaces_the_token_text(self) -> None:
        """Test the ordinary edit, which is what an assignment to an existing key does."""
        node = value_node("integer", "1")

        update_node_value(node, 42)

        assert str(node.children[0]) == "42"

    def test_keeps_the_notation_of_the_token_it_replaces(self) -> None:
        """Test that a Fortran ``D`` exponent survives an edit, not becoming ``e``."""
        node = value_node("double", "1.0D3")

        update_node_value(node, 2e-20)

        assert str(node.children[0]) == "2D-20"

    def test_leaves_the_token_object_intact(self) -> None:
        """Test that a fresh ``Token`` is put in place rather than the old one mutated.

        ``Token`` is a ``str`` subclass and therefore immutable; the node's child list is
        what changes.
        """
        node = value_node("integer", "1")
        original = node.children[0]

        update_node_value(node, 2)

        assert node.children[0] is not original
        assert original == "1"

    def test_refuses_a_value_of_another_type(self) -> None:
        """Test that the type is fixed by the parse and cannot be changed by assignment."""
        with pytest.raises(TypeError, match="Trying to change value type"):
            update_node_value(value_node("integer", "1"), "text")

    def test_refuses_a_node_that_is_not_a_value_type(self) -> None:
        """Test the guard against being handed the wrong node entirely."""
        with pytest.raises(TypeError, match="Trying to change value type"):
            update_node_value(Tree("key", [Token("CNAME", "a")]), 1)


class TestSpliceEntryNodes:
    """Putting freshly parsed nodes into a container."""

    def test_inserts_at_the_index_given(self) -> None:
        """Test that the nodes land where the insertion index said, not at the end."""
        first, last = entry_node("key_value", "a"), Tree("empty_line", [])
        container = Tree("start", [first, last])
        fresh = entry_node("key_value", "z")

        splice_entry_nodes(container, [fresh], 1)

        assert container.children == [first, fresh, last]

    def test_adopts_what_it_inserts(self) -> None:
        """Test that the spliced nodes learn which container now holds them."""
        container = Tree("start", [])
        fresh = entry_node("key_value", "z")

        splice_entry_nodes(container, [fresh, Token("NEWLINE", "\n")], 0)

        assert fresh.parent is container

    def test_inserts_several_nodes_in_order(self) -> None:
        """Test that an entry spanning more than one node keeps its pieces together."""
        container = Tree("start", [])
        nodes = [entry_node("key_value", "z"), Token("NEWLINE", "\n")]

        splice_entry_nodes(container, nodes, 0)

        assert container.children == nodes


class TestRemoveEntryNode:
    """Unlinking an entry, and repairing the shape that leaves behind."""

    def test_unlinks_the_entry_from_its_container(self) -> None:
        """Test the ordinary deletion, where nothing else needs doing."""
        kept, doomed = entry_node("key_value", "a"), entry_node("key_value", "b")
        container = Tree("start", [kept, doomed])
        AddParent().visit(container)

        remove_entry_node(doomed, FakeInfo())

        assert container.children == [kept]

    def test_repairs_the_runs_the_removal_left_adjacent(self) -> None:
        """Test the Fortran case: text either side of a namelist becomes one run.

        The grammar requires entries and the text between them to alternate, so leaving two
        text nodes side by side gives a tree the start rule cannot derive.
        """
        before, after = Tree("random_text", [Token("T", "x")]), Tree("random_text", [Token("T", "y")])
        doomed = entry_node("key_block", "nml")
        container = Tree("start", [before, doomed, after])
        AddParent().visit(container)

        remove_entry_node(doomed, FakeInfo("random_text"))

        assert len(container.children) == 1
        assert [str(child) for child in container.children[0].children] == ["x", "y"]


class TestPruneEmptyWrappers:
    """Dropping the nodes that existed only to hold an entry now deleted."""

    def test_removes_a_wrapper_the_grammar_cannot_derive_empty(self) -> None:
        """Test the Fortran line rule: an assignment plus an optional comment and newline.

        Once the assignment goes there is nothing the rule can derive, so the line goes with
        it -- taking the trailing comment, which described the line that has gone.
        """
        line = Tree("nml_line", [Tree("fortran_comment", [Token("C", "! note")])])
        container = Tree("block", [line, entry_node("key_value", "kept", value_node())])
        AddParent().visit(container)

        surviving = prune_empty_wrappers(line, FakeInfo("block"))

        assert container.children == [container.children[0]]
        assert surviving is container

    def test_keeps_a_wrapper_that_still_holds_an_entry(self) -> None:
        """Test the line holding two assignments, which keeps the one that remains."""
        kept = entry_node("key_value", "b", value_node())
        line = Tree("nml_line", [kept])
        container = Tree("block", [line])
        AddParent().visit(container)

        assert prune_empty_wrappers(line, FakeInfo("block")) is line
        assert line.children == [kept]

    def test_never_prunes_a_repetition_rule(self) -> None:
        """Test that a container accepting any number of children is left alone.

        Those hold the blank lines and free text that belong to the file rather than to any
        one entry, so an empty one is not a shape the grammar rejects.
        """
        empty = Tree("block", [])
        container = Tree("start", [empty])
        AddParent().visit(container)

        assert prune_empty_wrappers(empty, FakeInfo("block")) is empty
        assert container.children == [empty]

    def test_climbs_until_something_survives(self) -> None:
        """Test that a wrapper emptied by pruning its own child goes too."""
        inner = Tree("nml_line", [])
        outer = Tree("group_body", [inner])
        container = Tree("start", [outer])
        AddParent().visit(container)

        surviving = prune_empty_wrappers(inner, FakeInfo("start"))

        assert container.children == []
        assert surviving is container

    def test_stops_at_a_node_with_no_parent(self) -> None:
        """Test the top of the tree, which has nothing to be removed from."""
        root = Tree("start", [])

        assert prune_empty_wrappers(root, FakeInfo()) is root


class TestMergeAdjacentRepetitions:
    """Rejoining sibling nodes of a rule that accepts any number of children."""

    def test_merges_two_and_reparents_what_moves(self) -> None:
        """Test that the surviving node owns the children it absorbs.

        Exercised with rule nodes rather than tokens, since only a ``Tree`` carries the
        ``.parent`` that has to be corrected.
        """
        left, right = Tree("block", [Tree("x", [])]), Tree("block", [Tree("y", [])])
        container = Tree("start", [left, right, Tree("equal", [])])
        AddParent().visit(container)

        merge_adjacent_repetitions(container, FakeInfo("block"))

        assert len(container.children) == 2
        assert [child.data for child in container.children[0].children] == ["x", "y"]
        assert container.children[0].children[1].parent is left
        # A rule that is not a repetition is left alone.
        assert container.children[1].data == "equal"

    def test_merges_a_run_of_more_than_two(self) -> None:
        """Test that merging continues from where it left off, not skipping a pair."""
        nodes = [Tree("text", [Token("T", letter)]) for letter in "abc"]
        container = Tree("start", nodes)
        AddParent().visit(container)

        merge_adjacent_repetitions(container, FakeInfo("text"))

        assert len(container.children) == 1
        assert [str(child) for child in container.children[0].children] == ["a", "b", "c"]

    def test_leaves_nodes_of_different_rules_alone(self) -> None:
        """Test that only siblings of the *same* repetition rule are candidates."""
        container = Tree("start", [Tree("text", []), Tree("other", [])])
        AddParent().visit(container)

        merge_adjacent_repetitions(container, FakeInfo("text", "other"))

        assert len(container.children) == 2

    def test_leaves_a_rule_that_is_not_a_repetition_alone(self) -> None:
        """Test the check that stops a rule being handed children it cannot accept."""
        container = Tree("start", [Tree("entry", []), Tree("entry", [])])
        AddParent().visit(container)

        merge_adjacent_repetitions(container, FakeInfo())

        assert len(container.children) == 2

    def test_ignores_tokens_between_nodes(self) -> None:
        """Test that a terminal sibling is not mistaken for a mergeable node."""
        container = Tree("start", [Tree("text", []), Token("NEWLINE", "\n"), Tree("text", [])])
        AddParent().visit(container)

        merge_adjacent_repetitions(container, FakeInfo("text"))

        assert len(container.children) == 3


class TestParseEntryNodes:
    """Turning the text of a new entry into the nodes Lark would have produced for it."""

    GRAMMAR = """
        start: line*
        ?line: entry | filler
        entry: key "=" integer NEWLINE -> key_value
        // A rule able to derive the empty string, as Fortran's inter-namelist text is.
        filler: FILLER*
        FILLER: /#[^\\n]*\\n/

        %import config.key
        %import config.integer
        %import config.NEWLINE
    """

    @pytest.fixture(scope="class")
    def lark(self):
        """Fixture returning the compiled parser for the grammar above."""
        return compile_grammar(self.GRAMMAR).lark

    def test_produces_the_nodes_of_the_container_rule(self, lark) -> None:
        """Test that parsing with the container as the start symbol yields its children.

        This is what makes hand-building nodes unnecessary: rule inlining, aliases and
        terminal renaming all come out right because Lark did them.
        """
        [node] = [n for n in parse_entry_nodes(lark, "start", "a=1\n") if isinstance(n, Tree)]

        assert node.data == "key_value"

    def test_annotates_the_nodes_it_returns(self, lark) -> None:
        """Test that the interiors carry ``.parent``, ready to be spliced in."""
        [node] = [n for n in parse_entry_nodes(lark, "start", "a=1\n") if isinstance(n, Tree)]

        assert all(child.parent is node for child in node.children if isinstance(child, Tree))

    def test_drops_a_node_that_matched_no_text(self, lark) -> None:
        """Test the filter, without which a second empty node joins the container's own.

        A rule able to derive the empty string produces one of these. Splicing it would
        give the container two, a shape the grammar cannot derive and the reconstructor
        refuses to write out.
        """
        nodes = parse_entry_nodes(lark, "start", "a=1\n")

        assert not any(isinstance(node, Tree) and not node.children for node in nodes)

    def test_raises_on_text_the_grammar_does_not_accept(self, lark) -> None:
        """Test that a bad candidate fails here, before anything is spliced."""
        with pytest.raises(UnexpectedInput):
            parse_entry_nodes(lark, "start", "!!! not an entry\n")


# The entry rules are *named* key_value and key_list, not aliased to them: Lark resolves a
# start symbol against a rule name, and writing values needs them as start symbols.
REPEAT_GRAMMAR = """
    start: line*
    ?line: key_value | key_list
    key_value: key ws* "=" ws* value line_end
    key_list: key ws* "=" ws* value (ws* "," ws* value)+ line_end
    ?value: repeat | scalar
    repeat: REPEAT_COUNT scalar?
    REPEAT_COUNT: /[0-9]+\\*/
    ?scalar: integer | float
    line_end: ws* NEWLINE

    %import config.key
    %import config.integer
    %import config.float
    %import config.ws
    %import config.NEWLINE
"""


class TestWritingRepeats:
    """Splitting a repeat run when one of the elements it covers is written.

    A repeat backs several list elements at once, so writing one of them cannot be an
    in-place token update: the run has to be re-spelled and the node replaced.
    """

    @pytest.fixture(scope="class")
    def lark(self):
        """Fixture returning a parser whose entry rules are start symbols."""
        return compile_grammar(REPEAT_GRAMMAR).lark

    @staticmethod
    def _entry(lark, text: str) -> Tree:
        """Parse one assignment and return its entry node, annotated."""
        tree = lark.parse(text, start="start")
        AddParent().visit(tree)
        return next(node for node in tree.iter_subtrees() if str(node.data) in ("key_value", "key_list"))

    def test_runs_of_groups_equal_neighbours(self) -> None:
        """Test the grouping that lets a constant run stay written as ``n*v``."""
        assert _runs_of([9.0, 9.0, 8.0, 9.0, 9.0, 9.0]) == [(2, 9.0), (1, 8.0), (3, 9.0)]
        assert _runs_of([1]) == [(1, 1)]
        assert _runs_of([]) == []

    def test_runs_of_separates_equal_values_of_different_types(self) -> None:
        """Test that ``1`` and ``1.0`` are not one run, since they write differently."""
        assert _runs_of([1, 1.0]) == [(1, 1), (1, 1.0)]

    def test_source_token_reads_through_a_repeat(self, lark) -> None:
        """Test that the notation comes from the repeated value, not the count."""
        entry = self._entry(lark, "x = 3*1\n")
        node = next(child for child in entry.children if is_value_node(child))

        assert _source_token(node) == ("integer", "1")

    def test_source_token_refuses_a_repeat_of_nulls(self, lark) -> None:
        """Test that ``n*`` names no type, so there is no notation to write a value in."""
        entry = self._entry(lark, "x = 2*\n")
        node = next(child for child in entry.children if is_value_node(child))

        with pytest.raises(TypeError, match="change value type"):
            _source_token(node)

    def test_value_texts_regroups_into_repeats(self, lark) -> None:
        """Test that equal neighbours are written back as a count rather than expanded."""
        entry = self._entry(lark, "x = 6*9.0\n")
        node = next(child for child in entry.children if is_value_node(child))

        assert _value_texts(node, [9.0, 9.0, 8.0, 9.0, 9.0, 9.0]) == ["2*9.0", "8.0", "3*9.0"]

    def test_value_texts_writes_a_run_of_nulls_as_a_bare_count(self, lark) -> None:
        """Test the one case needing no notation at all."""
        entry = self._entry(lark, "x = 2*\n")
        node = next(child for child in entry.children if is_value_node(child))

        assert _value_texts(node, [None, None, None]) == ["3*"]

    def test_value_texts_refuses_a_value_of_another_type(self, lark) -> None:
        """Test that the type is fixed by what the file already holds."""
        entry = self._entry(lark, "x = 3*1\n")
        node = next(child for child in entry.children if is_value_node(child))

        with pytest.raises(TypeError, match="change value type"):
            _value_texts(node, [1, "text", 1])

    def test_replace_value_node_splits_a_run_in_place(self, lark) -> None:
        """Test that only the node being split is replaced.

        Whatever the entry writes between its other values -- a comment, a line break -- is
        left exactly as it was, because the surrounding children are untouched.
        """
        entry = self._entry(lark, "x = 6*9.0\n")
        node = next(child for child in entry.children if is_value_node(child))

        fresh = replace_value_node(entry, node, ["2*9.0", "8.0", "3*9.0"], lark)

        assert len(fresh) == 3
        assert all(new.parent is entry for new in fresh)
        assert node not in entry.children

    def test_write_values_updates_in_place_when_there_is_no_repeat(self, lark) -> None:
        """Test the ordinary path, where each node holds exactly one element."""
        entry = self._entry(lark, "x = 1, 2\n")
        nodes = [child for child in entry.children if is_value_node(child)]

        written = write_values(nodes, [7, 8], lark)

        assert written == nodes
        assert [str(node.children[0]) for node in nodes] == ["7", "8"]

    def test_write_values_splits_the_run_it_has_to(self, lark) -> None:
        """Test the whole path: six elements written once, one of them changed."""
        entry = self._entry(lark, "x = 6*9.0\n")
        node = next(child for child in entry.children if is_value_node(child))
        # A repeat appears once per element it covers.
        refs = [node] * 6

        written = write_values(refs, [9.0, 9.0, 8.0, 9.0, 9.0, 9.0], lark)

        # Still one reference per element, but now backed by three nodes: 2*9.0, 8.0, 3*9.0.
        assert len(written) == 6
        assert len({id(node) for node in written}) == 3

    def test_write_values_leaves_a_list_entry_a_list(self, lark) -> None:
        """Test splitting a repeat inside an entry that is already a list.

        Nothing about the entry changes here -- only the node holding the run -- so the
        cheaper path that replaces one node is taken rather than rewriting the whole entry.
        """
        entry = self._entry(lark, "x = 1, 3*2\n")
        nodes = [child for child in entry.children if is_value_node(child)]
        # The first node backs one element, the repeat backs three.
        refs = [nodes[0], nodes[1], nodes[1], nodes[1]]

        written = write_values(refs, [1, 2, 5, 2], lark)

        assert str(entry.data) == "key_list"
        assert len(written) == 4

    def test_write_values_recategorises_a_scalar_entry(self, lark) -> None:
        """Test that an entry holding one repeat becomes a list entry once it is split.

        One value is a scalar assignment and several are a list one, so splitting the repeat
        changes what kind of entry the node is.
        """
        entry = self._entry(lark, "x = 3*1\n")
        node = next(child for child in entry.children if is_value_node(child))

        write_values([node] * 3, [1, 2, 3], lark)

        assert str(entry.data) == "key_list"

    def test_refuses_a_value_for_a_position_the_file_never_wrote(self, lark) -> None:
        """Test the hole in a gathered array, which has no node to write through.

        A position no indexed entry mentioned is read as null and stays that way; giving it
        a value would mean inventing a line to hold it.
        """
        entry = self._entry(lark, "x = 1\n")
        node = next(child for child in entry.children if is_value_node(child))

        with pytest.raises(ValueError, match="leaves unset"):
            write_values([node, None], [1, 2], lark)

        # Leaving it unset is fine; nothing is being written there.
        assert write_values([node, None], [1, None], lark) == [node, None]

    def test_refuses_an_entry_whose_positions_another_overrode(self, lark) -> None:
        """Test the array whose entries overlap, where one line cannot be rewritten alone.

        The repeat holds three values but only two of its positions survive in the list, so
        re-spelling that entry would change how many values it writes.
        """
        entry = self._entry(lark, "x = 3*1\n")
        node = next(child for child in entry.children if is_value_node(child))

        with pytest.raises(ValueError, match="assigns some of its positions twice"):
            write_values([node, node], [1, 2], lark)

    def test_groups_interleaved_entries_by_the_entry_that_wrote_them(self, lark) -> None:
        """Test that an entry is rewritten once with everything it holds.

        An array written with indices can have its entries interleaved, so grouping by
        adjacency would rewrite the first entry twice and lose a value each time.
        """
        first = self._entry(lark, "x = 1, 3\n")
        second = self._entry(lark, "y = 2\n")
        outer = [child for child in first.children if is_value_node(child)]
        inner = next(child for child in second.children if is_value_node(child))

        # Positions 0 and 2 come from one entry, position 1 from another.
        written = write_values([outer[0], inner, outer[1]], [7, 8, 9], lark)

        assert [str(node.children[0]) for node in written] == ["7", "8", "9"]
