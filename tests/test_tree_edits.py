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
from access.config.grammar_values import UnsupportedEntryError
from access.config.tree_edits import (
    blank_comments,
    merge_adjacent_repetitions,
    parse_entry_nodes,
    prune_empty_wrappers,
    remove_entry_node,
    replace_entry_contents,
    splice_entry_nodes,
    update_node_value,
    write_values,
)
from access.config.tree_navigation import AddParent, is_value_node


class FakeInfo:
    """Stands in for ``GrammarInfo`` where only the merge test is consulted.

    *repetitions* names the rules to report as a plain repetition.
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

        ``Token`` is an immutable ``str`` subclass; the node's child list is what changes.
        """
        node = value_node("integer", "1")
        original = node.children[0]

        update_node_value(node, 2)

        assert node.children[0] is not original
        assert original == "1"

    def test_leaves_a_token_already_holding_the_value(self) -> None:
        """Test that writing a value over itself keeps the notation it was written in."""
        node = value_node("double", "1.0d0")
        original = node.children[0]

        update_node_value(node, 1.0)

        assert node.children[0] is original

    def test_writes_a_nan_over_itself(self) -> None:
        """Test that an unchanged NaN, which cannot be written anew, is no obstacle."""
        node = value_node("float", "nan")

        update_node_value(node, float("nan"))

        assert str(node.children[0]) == "nan"

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


class TestReplaceEntryContents:
    """Giving an entry the contents of a freshly parsed one, keeping the node itself."""

    def test_takes_the_children_and_the_rule_of_the_fresh_entry(self) -> None:
        """Test that the node keeps its identity and adopts what it now holds."""
        entry = entry_node("key_list", "a", value_node(text="1"), value_node(text="2"))
        fresh = entry_node("key_value", "a", value_node(text="3"))
        fresh_children = list(fresh.children)

        replace_entry_contents(entry, fresh)

        assert entry.data == "key_value"
        assert entry.children == fresh_children
        assert all(child.parent is entry for child in entry.children)


class TestBlankComments:
    """Writing the comments in a copy of part of a file as nothing."""

    def test_blanks_every_comment_and_nothing_else(self) -> None:
        """Test comments at any depth, of either rule, beside text that has to stay."""
        line = Tree(
            "line",
            [
                entry_node("key_value", "a", value_node()),
                Tree("line_end", [Tree("comment", [Token("COMMENT", "  # one")]), Token("NEWLINE", "\n")]),
                Tree("fortran_comment", [Token("COMMENT", "! two")]),
            ],
        )

        blank_comments(line)

        assert [str(token) for token in line.scan_values(lambda _: True)] == ["a", "1", "", "\n", ""]

    def test_keeps_the_carriage_return_a_comment_ran_up_to(self) -> None:
        """Test a comment rule stopping only at the line feed, in a Windows file."""
        comment = Tree("comment", [Token("COMMENT", "# one\r"), Tree("unusual", [])])

        blank_comments(comment)

        assert comment.children == [Token("COMMENT", "\r"), Tree("unusual", [])]


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

        Entries and text must alternate, or the start rule cannot derive the tree.
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

        Once the assignment goes the line goes too, with the comment that described it.
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

        It holds the blank lines and free text of the file, so an empty one is derivable.
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

        Rule nodes, not tokens: only a ``Tree`` carries the ``.parent`` to be corrected.
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

        Rule inlining, aliases and terminal renaming come out right because Lark did them.
        """
        [node] = [n for n in parse_entry_nodes(lark, "start", "a=1\n") if isinstance(n, Tree)]

        assert node.data == "key_value"

    def test_annotates_the_nodes_it_returns(self, lark) -> None:
        """Test that the interiors carry ``.parent``, ready to be spliced in."""
        [node] = [n for n in parse_entry_nodes(lark, "start", "a=1\n") if isinstance(n, Tree)]

        assert all(child.parent is node for child in node.children if isinstance(child, Tree))

    def test_drops_a_node_that_matched_no_text(self, lark) -> None:
        """Test the filter, without which a second empty node joins the container's own.

        A rule that derives the empty string makes one; two would be underivable.
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


class TestWritingValues:
    """Writing a list through the nodes behind it, a repeat only as a whole.

    A repeat backs several elements, so writing one cannot be an in-place token update:
    the whole repeat is written, or the write is refused.
    """

    @pytest.fixture(scope="class")
    def lark(self):
        """Fixture returning a parser whose entry rules are start symbols."""
        return compile_grammar(REPEAT_GRAMMAR).lark

    @staticmethod
    def _nodes(lark, text: str) -> list[Tree]:
        """Parse one assignment and return the value nodes of its entry, annotated."""
        tree = lark.parse(text, start="start")
        AddParent().visit(tree)
        entry = next(node for node in tree.iter_subtrees() if str(node.data) in ("key_value", "key_list"))
        return [child for child in entry.children if is_value_node(child)]

    def test_updates_in_place_when_there_is_no_repeat(self, lark) -> None:
        """Test the ordinary path, where each node holds exactly one element."""
        nodes = self._nodes(lark, "x = 1, 2\n")

        write_values(nodes, [7, 8])

        assert [str(node.children[0]) for node in nodes] == ["7", "8"]

    def test_leaves_an_element_that_did_not_change_as_written(self, lark) -> None:
        """Test that an element that is not given a new value keeps the token it had."""
        nodes = self._nodes(lark, "x = 1.0, 2.5\n")
        untouched = nodes[1].children[0]

        write_values(nodes, [3.0, 2.5])

        assert nodes[1].children[0] is untouched
        assert str(nodes[0].children[0]) == "3.0"

    def test_a_repeat_given_one_new_value_is_rewritten_whole(self, lark) -> None:
        """Test that ``6*9.0`` becomes ``6*9.5``: the count stays, the token changes."""
        [node] = self._nodes(lark, "x = 6*9.0\n")

        write_values([node] * 6, [9.5] * 6)

        assert str(node.children[0]) == "6*"
        assert str(node.children[1].children[0]) == "9.5"

    def test_a_repeat_in_a_list_is_written_beside_the_other_elements(self, lark) -> None:
        """Test an entry mixing a plain element and a repeat: both are written."""
        first, run = self._nodes(lark, "x = 1, 3*2\n")

        write_values([first, run, run, run], [5, 7, 7, 7])

        assert str(first.children[0]) == "5"
        assert str(run.children[1].children[0]) == "7"

    def test_a_repeat_left_alone_is_not_written(self, lark) -> None:
        """Test that a repeat given the values it holds keeps its token as it is."""
        [node] = self._nodes(lark, "x = 3*1\n")
        token = node.children[1].children[0]

        write_values([node] * 3, [1, 1, 1])

        assert node.children[1].children[0] is token

    def test_part_of_a_repeat_cannot_be_written(self, lark) -> None:
        """Test that one element of a run is not given another value on its own.

        That would split the run: the caller is told to delete the key and assign again.
        """
        [node] = self._nodes(lark, "x = 6*9.0\n")

        with pytest.raises(ValueError, match=r"part of the repeat 6\*.*delete it and assign it again"):
            write_values([node] * 6, [9.0, 9.0, 8.0, 9.0, 9.0, 9.0])

        assert str(node.children[1].children[0]) == "9.0"

    def test_a_repeat_given_a_value_of_another_type_is_refused(self, lark) -> None:
        """Test that the type is fixed by what the file already holds."""
        [node] = self._nodes(lark, "x = 3*1\n")

        with pytest.raises(TypeError, match="change value type"):
            write_values([node] * 3, ["text"] * 3)

    def test_a_repeat_of_nulls_holds_no_type_to_write_a_value_in(self, lark) -> None:
        """Test that ``n*`` can be left alone but not given a value."""
        [node] = self._nodes(lark, "x = 2*\n")

        write_values([node, node], [None, None])
        with pytest.raises(TypeError, match="change value type"):
            write_values([node, node], [1, 1])

    def test_refuses_a_value_for_a_position_the_file_never_wrote(self, lark) -> None:
        """Test the hole in an array, which has no node to write through.

        A position no entry mentioned stays null; a value would mean inventing a line.
        """
        [node] = self._nodes(lark, "x = 1\n")

        with pytest.raises(ValueError, match="leaves unset"):
            write_values([node, None], [1, 2])

        # Leaving it unset is fine; nothing is being written there.
        write_values([node, None], [3, None])
        assert str(node.children[0]) == "3"

    def test_a_value_the_file_refuses_leaves_every_node_as_it_was(self, lark) -> None:
        """Test that a write is all or nothing.

        The first element is fine and the second is not: the first must not be written.
        """
        nodes = self._nodes(lark, "x = 1, 2\n")

        with pytest.raises(TypeError, match="change value type"):
            write_values(nodes, [5, "text"])

        assert [str(node.children[0]) for node in nodes] == ["1", "2"]

    def test_a_value_the_format_cannot_write_leaves_every_node_as_it_was(self, lark) -> None:
        """Test that the text of every value is worked out before any node changes.

        A NaN has the right type, so it passes the check on that, but no format writes one.
        """
        nodes = self._nodes(lark, "x = 1.0, 2.0\n")

        with pytest.raises(UnsupportedEntryError):
            write_values(nodes, [5.0, float("nan")])

        assert [str(node.children[0]) for node in nodes] == ["1.0", "2.0"]

    def test_a_value_refused_in_a_repeat_leaves_the_elements_before_it_as_they_were(self, lark) -> None:
        """Test that a repeat is checked before anything is written, like a plain node.

        The repeat comes second and is refused for its type; the plain one stays unwritten.
        """
        [plain] = self._nodes(lark, "x = 1\n")
        [repeat] = self._nodes(lark, "y = 2*2\n")

        with pytest.raises(TypeError, match="change value type"):
            write_values([plain, repeat, repeat], [7, "a", "a"])

        assert str(plain.children[0]) == "1"
