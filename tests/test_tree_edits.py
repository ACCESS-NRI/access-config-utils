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
    merge_adjacent_repetitions,
    parse_entry_nodes,
    remove_entry_node,
    splice_entry_nodes,
    update_node_value,
)
from access.config.tree_navigation import AddParent


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
