# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for how a new entry should be written so that it looks like its neighbours.

Nothing is faked. Copying a whole entry writes a parse tree out and so needs a grammar's
reconstructor, and the indentation of an entry and the style of a block are read off the
same trees, so they are all parsed from text. The Fortran namelist grammar has every shape
a copy has to deal with: a line wrapping its entry or not, a chain, an array qualifier, a
repeat count, a skipped position and a block. The MOM6 grammar adds an entry holding its own
comment, and the FMS field table an entry that is a chain of three keys.
"""

import pytest
from conftest import entry_node, value_node, ws_node
from lark import Token, Tree

from access.config.entry_fragments import EntryTemplate
from access.config.entry_style import (
    EntryStyle,
    donor_template,
    indent_of,
    probe_entry_style,
    probe_sibling_block_style,
)
from access.config.field_table import FieldTableParser
from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_compiled import CompiledGrammar, compile_grammar
from access.config.mom6_input import MOM6InputParser
from access.config.tree_navigation import entries_of


@pytest.fixture(scope="module")
def fortran() -> CompiledGrammar:
    """Return the compiled Fortran namelist grammar."""
    return compile_grammar(FortranNMLParser().grammar)


@pytest.fixture(scope="module")
def mom6() -> CompiledGrammar:
    """Return the compiled MOM6 parameter file grammar."""
    return compile_grammar(MOM6InputParser().grammar)


@pytest.fixture(scope="module")
def field_table() -> CompiledGrammar:
    """Return the compiled FMS field table grammar."""
    return compile_grammar(FieldTableParser().grammar)


def group_body(compiled: CompiledGrammar, text: str) -> Tree:
    """Return the body of the one namelist group *text* holds."""
    group = entries_of(compiled.lark.parse(text, start="start"))[0]
    return next(child for child in group.children if isinstance(child, Tree) and child.data == "block")


def shown(template: EntryTemplate | None) -> str | None:
    """Return a template as text, with ``{K0}`` for a key of level 0 and ``{V}`` a value."""
    if template is None:
        return None
    marks = {"key": "{{K{level}}}", "value": "{{V}}"}
    return "".join(
        fragment.text if fragment.kind == "literal" else marks[fragment.kind].format(level=fragment.level)
        for fragment in template
    )


class TestIndentOf:
    """The whitespace the line of an entry starts with."""

    def test_is_the_whitespace_before_the_key(self) -> None:
        """Test the ordinary case, one run of whitespace in front of the key."""
        entry = Tree("key_value", [ws_node("  "), Tree("key", [Token("CNAME", "a")]), value_node()])

        assert indent_of(entry) == "  "

    def test_is_empty_for_an_entry_with_none(self) -> None:
        """Test that no indentation is the empty string, which is not ``None``."""
        assert indent_of(entry_node("key_value", "a", value_node())) == ""

    def test_joins_every_run_before_the_key(self) -> None:
        """Test a grammar that writes the whitespace before a key as several nodes."""
        entry = Tree("key_value", [ws_node(" "), ws_node("  "), Tree("key", [Token("CNAME", "a")])])

        assert indent_of(entry) == "   "

    def test_stops_at_the_key(self) -> None:
        """Test that whitespace after the key, around the assignment, is not indentation."""
        entry = Tree("key_value", [ws_node("  "), Tree("key", [Token("CNAME", "a")]), ws_node(" "), value_node()])

        assert indent_of(entry) == "  "

    def test_ignores_children_that_are_tokens(self) -> None:
        """Test that punctuation kept as a bare terminal is not whitespace."""
        entry = Tree("key_value", [ws_node(" "), Token("AMP", "&"), Tree("key", [Token("CNAME", "a")])])

        assert indent_of(entry) == " "


class TestProbeEntryStyle:
    """Reading the indentation off the entries a container already holds."""

    def test_reports_no_indent_for_an_empty_container(self, fortran) -> None:
        """Test the block that was empty in the file: nothing above shows an indentation."""
        assert probe_entry_style(Tree("block", []), fortran.reconstructor) == EntryStyle()

    def test_reads_the_indent_of_the_last_entry(self, fortran) -> None:
        """Test that the entry just above is the one copied, whatever the ones before."""
        style = probe_entry_style(group_body(fortran, "&g\n  a = 1\n    b = 2\n/\n"), fortran.reconstructor)

        assert style.indent == "    "

    def test_an_unindented_entry_shows_an_empty_indent(self, fortran) -> None:
        """Test that an unindented file is told apart from a container with no entry."""
        style = probe_entry_style(group_body(fortran, "&g\na = 1\n/\n"), fortran.reconstructor)

        assert style.indent == ""

    def test_reads_only_the_entries_above_where_the_new_one_goes(self, fortran) -> None:
        """Test an entry written among others, copying the one above, not the last."""
        body = group_body(fortran, "&g\n  a = 1\n    b = 2\n/\n")

        assert probe_entry_style(body, fortran.reconstructor, upto=1).indent == "  "
        assert probe_entry_style(body, fortran.reconstructor, upto=0) == EntryStyle()


class TestProbeSiblingBlockStyle:
    """The style a block created this session inherits from the blocks around it."""

    def test_takes_it_from_a_sibling_block(self, fortran) -> None:
        """Test that a new block is laid out like the block next to it."""
        tree = fortran.lark.parse("&old\n  a = 1\n/\n&new\n/\n", start="start")

        style = probe_sibling_block_style(tree, fortran.reconstructor)

        assert style.indent == "  "

    def test_takes_the_nearest_block_able_to_donate(self, fortran) -> None:
        """Test that the last block with entries wins, an empty one being passed over."""
        tree = fortran.lark.parse("&far\n    a = 1\n/\n&near\n  b = 1\n/\n&new\n/\n", start="start")

        assert probe_sibling_block_style(tree, fortran.reconstructor).indent == "  "

    def test_reports_no_indent_without_a_usable_block(self, fortran) -> None:
        """Test both ways there is nothing to copy: no block at all, and an empty one."""
        empty_block = fortran.lark.parse("&new\n/\n", start="start")

        assert probe_sibling_block_style(Tree("start", []), fortran.reconstructor) == EntryStyle()
        assert probe_sibling_block_style(empty_block, fortran.reconstructor) == EntryStyle()

    def test_takes_the_block_at_the_end_of_a_chain(self, field_table) -> None:
        """Test an FMS field table entry: a name block in a model in a type."""
        tree = field_table.lark.parse('"tracer", "ocean_mod", "age"\n  "longname", "x" /\n', start="start")

        assert probe_sibling_block_style(tree, field_table.reconstructor).indent == "  "

    def test_searches_upwards_from_the_block_created(self, fortran) -> None:
        """Test that the block above a new one wins over a later one."""
        tree = fortran.lark.parse("&above\n    a = 1\n/\n&new\n/\n&below\n  b = 1\n/\n", start="start")
        new = tree.children.index(entries_of(tree)[1])

        assert probe_sibling_block_style(tree, fortran.reconstructor, upto=new).indent == "    "

    def test_ignores_a_block_entry_whose_body_is_not_a_block_node(self, fortran) -> None:
        """Test the contract: the body must be a rule *named* ``block``, not just nested."""
        odd = entry_node("key_block", "blk", Tree("body", [entry_node("key_value", "a", value_node())]))

        assert probe_sibling_block_style(Tree("start", [odd]), fortran.reconstructor) == EntryStyle()


class TestDonorTemplate:
    """Copying an existing entry whole, as the template for a new one."""

    def test_writes_markers_for_the_key_and_the_value(self, fortran) -> None:
        """Test that everything else is fixed text, and the value's notation is kept."""
        template = probe_entry_style(group_body(fortran, "&g\n  name = 'abc'\n/\n"), fortran.reconstructor).donor(
            1, "key_value"
        )

        assert shown(template) == "  {K0} = {V}\n"
        assert [fragment.notation for fragment in template if fragment.kind == "value"] == [("fortran_string", "'abc'")]

    def test_leaves_out_a_comment_in_the_entry(self, mom6) -> None:
        """Test that a new entry does not repeat someone else's comment."""
        tree = mom6.lark.parse("A = 1 ! one\n", start="start")

        assert shown(probe_entry_style(tree, mom6.reconstructor).donor(1, "key_value")) == "{K0} = {V}\n"

    def test_takes_only_the_line_break_from_a_line_wrapping_the_entry(self, fortran) -> None:
        """Test that a separator ending the line stays with it, as the comment does."""
        body = group_body(fortran, "&g\n  a = 1, ! one\n/\n")

        assert shown(probe_entry_style(body, fortran.reconstructor).donor(1, "key_value")) == "  {K0} = {V}\n"

    def test_keeps_a_windows_line_break(self, fortran, mom6) -> None:
        """Test that a carriage return survives, even one a comment ran up to."""
        body = group_body(fortran, "&g\r\n  a = 1 ! one\r\n/\r\n")
        tree = mom6.lark.parse("A = 1 ! one\r\n", start="start")

        assert shown(probe_entry_style(body, fortran.reconstructor).donor(1, "key_value")) == "  {K0} = {V}\r\n"
        assert shown(probe_entry_style(tree, mom6.reconstructor).donor(1, "key_value")) == "{K0} = {V}\r\n"

    def test_an_entry_on_a_line_with_no_line_break(self, fortran) -> None:
        """Test a group written on one line, whose entries are followed by the next."""
        body = group_body(fortran, "&g a = 1 /\n")

        assert shown(probe_entry_style(body, fortran.reconstructor).donor(1, "key_value")) == " {K0} = {V}"

    def test_marks_each_key_of_a_chain_by_its_level(self, fortran) -> None:
        """Test a derived-type component, which only a chain of the same depth can copy."""
        style = probe_entry_style(group_body(fortran, "&g\n  a%b = 1\n/\n"), fortran.reconstructor)

        assert shown(style.donor(2, "key_value")) == "  {K0}%{K1} = {V}\n"
        assert style.donor(1, "key_value") is None

    def test_leaves_out_an_array_qualifier(self, fortran) -> None:
        """Test that a new entry is not written as one element of an array."""
        body = group_body(fortran, "&g\n  v(3) = 1\n/\n")

        assert shown(probe_entry_style(body, fortran.reconstructor).donor(1, "key_value")) == "  {K0} = {V}\n"

    def test_empties_a_block(self, fortran) -> None:
        """Test that a new group copies the delimiters of the last, not its contents."""
        tree = fortran.lark.parse("$g\n  a = 1\n$end\n", start="start")

        assert shown(probe_entry_style(tree, fortran.reconstructor).donor(1, "key_block")) == "${K0}\n$end\n"

    def test_keeps_the_first_separator_and_the_tail_of_a_list(self, fortran) -> None:
        """Test that a list is cut down to the two values ``expand_value_slots`` grows."""
        body = group_body(fortran, "&g\n  a = 1, 2,3\n/\n")

        template = probe_entry_style(body, fortran.reconstructor).donor(1, "key_list")

        assert shown(template) == "  {K0} = {V}, {V}\n"
        assert [fragment.notation for fragment in template if fragment.kind == "value"] == [
            ("integer", "1"),
            ("integer", "2"),
        ]

    @pytest.mark.parametrize(
        ("text", "category"),
        [
            pytest.param("&g\n  a = 1,\n      2\n/\n", "key_list", id="broken-list"),
            pytest.param("&g\n  a = 3*1\n/\n", "key_value", id="repeat"),
            pytest.param("&g\n  a = 1, , 3\n/\n", "key_list", id="skipped-position"),
        ],
    )
    def test_refuses_an_entry_a_new_one_should_not_copy(self, fortran, text, category) -> None:
        """Test the shapes that belong to the values of an entry, not to its layout."""
        assert probe_entry_style(group_body(fortran, text), fortran.reconstructor).donor(1, category) is None

    def test_refuses_an_entry_holding_several(self) -> None:
        """Test an entry that is not a chain, which no key path can write.

        Refused before anything is written out, so no reconstructor is needed.
        """
        nested = Tree("body", [entry_node("key_value", "b", value_node()), entry_node("key_value", "c", value_node())])
        entry = entry_node("key_block", "a", nested)

        assert donor_template(entry, entry, None) is None


class TestProbingDonors:
    """Which entries a container offers to copy, and when."""

    def test_copies_the_last_entry_of_each_kind(self, fortran) -> None:
        """Test that a scalar and a list are each copied from the nearest of their kind."""
        body = group_body(fortran, "&g\n  a = 1\n  b = 'x'\n  c = 1, 2\n/\n")

        style = probe_entry_style(body, fortran.reconstructor)

        assert [signature for signature, _ in style.donors] == [(1, "key_list"), (1, "key_value")]
        assert style.donor(1, "key_value")[-2].notation == ("fortran_string", "'x'")

    def test_skips_a_line_holding_several_entries(self, fortran) -> None:
        """Test that an entry sharing its line is not copied: that line is set for both."""
        body = group_body(fortran, "&g\n  a = 1\n  b = 2, c = 'x'\n/\n")

        template = probe_entry_style(body, fortran.reconstructor).donor(1, "key_value")

        assert template[-2].notation == ("integer", "1")

    def test_tries_the_entry_above_one_that_cannot_be_copied(self, fortran) -> None:
        """Test that a list skipping a position does not stop the one above being copied."""
        body = group_body(fortran, "&g\n  a = 1, 2\n  b = 1, , 3\n/\n")

        assert shown(probe_entry_style(body, fortran.reconstructor).donor(1, "key_list")) == "  {K0} = {V}, {V}\n"

    def test_a_created_block_copies_the_entries_of_its_sibling(self, fortran) -> None:
        """Test that the first entry of a new group is written like those of the last."""
        tree = fortran.lark.parse("&g\n  a = 'x'\n/\n", start="start")

        style = probe_sibling_block_style(tree, fortran.reconstructor)

        assert shown(style.donor(1, "key_value")) == "  {K0} = {V}\n"
