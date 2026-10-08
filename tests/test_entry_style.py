# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for how a new entry should be written so that it looks like its neighbours.

Nothing is faked: ``entry_style`` reads templates and parse trees, and both are built here
directly. That is what makes the cases separable -- a container whose only entry has no
value, a scalar and a list disagreeing about spacing, a template with an indent slot and one
without. A grammar gives you the shapes it happens to derive, not the one under test.

The exception is copying a whole entry, which writes a parse tree out and so needs a
grammar's reconstructor. Those trees are parsed from text by the Fortran namelist grammar,
which has every shape a copy has to deal with: a line wrapping its entry or not, a chain,
an array qualifier, a repeat count, a skipped position and a block. The MOM6 grammar adds
an entry holding its own comment.
"""

import pytest
from conftest import entry_node, value_node, ws_node
from lark import Token, Tree

from access.config.entry_fragments import EntryTemplate, Fragment
from access.config.entry_style import (
    EntryStyle,
    canonical_ws,
    classify_slots,
    donor_template,
    probe_entry_style,
    probe_sibling_block_style,
    style_matches,
)
from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_compiled import CompiledGrammar, compile_grammar
from access.config.mom6_input import MOM6InputParser
from access.config.tree_navigation import entries_of

KEY, WS, VALUE, BLOCK = Fragment("key"), Fragment("ws"), Fragment("value"), Fragment("block")


def literal(text: str) -> Fragment:
    """Return a fragment of fixed text."""
    return Fragment("literal", text)


# ``  KEY = value\n``: an indent slot, two around the assignment, one before the newline.
LINE = (WS, KEY, WS, literal("="), WS, VALUE, WS, literal("\n"))

# ``  KEY = a, b\n``: the same, with an element separator between two values.
LIST_LINE = (WS, KEY, WS, literal("="), WS, VALUE, literal(","), WS, VALUE, WS, literal("\n"))


@pytest.fixture(scope="module")
def fortran() -> CompiledGrammar:
    """Return the compiled Fortran namelist grammar."""
    return compile_grammar(FortranNMLParser().grammar)


@pytest.fixture(scope="module")
def mom6() -> CompiledGrammar:
    """Return the compiled MOM6 parameter file grammar."""
    return compile_grammar(MOM6InputParser().grammar)


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


def styled_entry(indent: str = "", *pads: str, key: str = "a", values: int = 1) -> Tree:
    """Return an entry node whose whitespace runs are *indent* then *pads*, in order.

    Args:
        indent: Whitespace before the key.
        *pads: One run before each value, in order.
        key: The key name.
        values: How many values the entry holds.
    """
    children: list[Tree] = [ws_node(indent)] if indent else []
    children.append(Tree("key", [Token("CNAME", key)]))
    for index in range(values):
        if index < len(pads):
            children.append(ws_node(pads[index]))
        children.append(value_node())
    return Tree("key_value" if values == 1 else "key_list", children)


class TestCanonicalWs:
    """What a whitespace placeholder renders as when there is no entry to copy from."""

    @pytest.mark.parametrize("index", [0, 6])
    def test_empty_at_either_end_of_the_line(self, index) -> None:
        """Test that a canonical entry carries no indentation and no trailing whitespace."""
        assert canonical_ws(LINE, index) == ""

    @pytest.mark.parametrize("index", [2, 4])
    def test_a_single_space_where_the_grammar_allows_one(self, index) -> None:
        """Test the readable default, which is what makes a new entry ``KEY = value``."""
        assert canonical_ws(LINE, index) == " "

    @pytest.mark.parametrize("separator", [",", ":"])
    def test_empty_before_a_separator_following_a_value(self, separator) -> None:
        """Test that a list reads as ``a, b`` rather than ``a , b``, any separator alike."""
        template = (VALUE, WS, literal(separator), WS, VALUE)
        assert canonical_ws(template, 1) == ""
        # The slot after the separator is ordinary readable whitespace.
        assert canonical_ws(template, 3) == " "

    def test_empty_either_side_of_a_block(self) -> None:
        """Test that the body of a block is not padded, since it starts on its own line."""
        assert canonical_ws((KEY, WS, BLOCK, WS, literal("\n")), 1) == ""
        assert canonical_ws((KEY, WS, BLOCK, WS, literal("\n")), 3) == ""

    def test_a_separator_that_is_not_punctuation_keeps_its_space(self) -> None:
        """Test that a word-like separator is spaced, as ``1 and 2`` would need to be."""
        assert canonical_ws((VALUE, WS, literal("and"), WS, VALUE), 1) == " "

    def test_empty_before_any_comma(self) -> None:
        """Test that a comma is not spaced off whatever it follows: ``"k", "v"``."""
        assert canonical_ws((literal('"'), WS, literal(","), WS, literal('"')), 1) == ""

    def test_empty_between_punctuation_and_a_key(self) -> None:
        """Test that a chain of keys reads as ``a%b`` rather than ``a% b``."""
        assert canonical_ws((KEY, literal("%"), WS, KEY), 2) == ""


class TestClassifySlots:
    """Grouping the whitespace placeholders by what each one spaces out."""

    def test_splits_a_scalar_line_into_its_regions(self) -> None:
        """Test that indent, assignment padding and the trailing slot are told apart."""
        slots = classify_slots(LINE)

        assert slots.leading == (0,)
        assert slots.key == (2, 4)
        # The "=" is the first fixed text after the key, so one slot falls before it.
        assert slots.key_split == 1
        # A scalar has no second value, so nothing spaces out an element separator.
        assert slots.element == ()

    def test_finds_the_element_region_of_a_list(self) -> None:
        """Test that only slots between the first two values are element spacing."""
        slots = classify_slots(LIST_LINE)

        assert slots.key == (2, 4)
        assert slots.element == (7,)

    def test_excludes_slots_fixed_by_the_line_structure(self) -> None:
        """Test that a slot rendering empty canonically is not offered a copied run.

        The trailing slot before the newline is at index 6, and it is absent from every
        region: a line never ends with copied whitespace, however the donor was spaced.
        """
        slots = classify_slots(LINE)
        assert 6 not in slots.key + slots.element + slots.leading

    def test_an_entry_with_no_value_spaces_out_its_key(self) -> None:
        """Test the ``KEY =`` case, where everything after the key is key padding."""
        slots = classify_slots((KEY, WS, literal("="), WS, literal("\n")))

        assert slots.key == (1,)
        assert slots.element == ()

    def test_a_template_with_no_fixed_text_after_the_key(self) -> None:
        """Test that with no assignment to split on, every key slot precedes it."""
        slots = classify_slots((KEY, WS, VALUE))

        assert slots.key == (1,)
        assert slots.key_split == 1

    def test_a_template_with_no_key(self) -> None:
        """Test that a template holding no key has no leading region to indent."""
        assert classify_slots((WS, VALUE, literal("\n"))).leading == (0,)


class TestStyleMatches:
    """Whether a template can reproduce the runs a donor demonstrated."""

    def test_anything_matches_when_there_is_no_donor(self) -> None:
        """Test the short circuit: with nothing to copy, no template can fail to match."""
        assert style_matches(LINE, EntryStyle())
        assert style_matches((KEY,), EntryStyle())

    def test_matches_when_the_slot_counts_line_up(self) -> None:
        """Test the ordinary case, one recorded run per stylable slot."""
        style = EntryStyle(indent="  ", key_pads=(" ", " "), key_split=1, has_donor=True)
        assert style_matches(LINE, style)

    def test_rejects_runs_that_fall_on_the_wrong_side_of_the_assignment(self) -> None:
        """Test that agreeing counts are not enough; the split has to agree too.

        ``LINE`` puts one slot before the ``=`` and one after. A donor recording both runs
        before it wrote something else, and reproducing them here would move them.
        """
        assert not style_matches(LINE, EntryStyle(key_pads=(" ", " "), key_split=2, has_donor=True))

    def test_a_style_with_no_key_runs_recorded_matches_anything(self) -> None:
        """Test that the split constrains nothing when there is no run to place."""
        assert style_matches(LINE, EntryStyle(indent="  ", key_split=2, has_donor=True))

    def test_rejects_a_template_with_nowhere_to_put_the_indent(self) -> None:
        """Test what a MOM6 assignment presents: no leading slot, so no indent."""
        style = EntryStyle(indent="    ", has_donor=True)

        assert not style_matches((KEY, WS, literal("="), WS, VALUE, literal("\n")), style)
        # With no indent recorded, the same template is fine.
        assert style_matches((KEY, WS, literal("="), WS, VALUE, literal("\n")), EntryStyle(has_donor=True))

    def test_rejects_a_mismatched_number_of_runs(self) -> None:
        """Test that a donor spaced differently from the template is not guessed at."""
        assert not style_matches(LINE, EntryStyle(key_pads=(" ",), has_donor=True))
        assert not style_matches(LIST_LINE, EntryStyle(element_pads=(" ", " "), has_donor=True))


class TestProbeEntryStyle:
    """Reading the runs back off the entries a container already holds."""

    def test_reports_no_donor_for_an_empty_container(self) -> None:
        """Test the block that was empty in the file, which is left canonical."""
        assert probe_entry_style(Tree("block", [])) == EntryStyle()

    def test_reads_the_indent_and_the_assignment_padding(self) -> None:
        """Test the runs taken from a single well-formed neighbour."""
        style = probe_entry_style(Tree("block", [styled_entry("  ", " ")]))

        assert style.has_donor
        assert style.indent == "  "
        assert style.key_pads == (" ",)

    def test_takes_the_last_entry_that_demonstrates_each_kind(self) -> None:
        """Test that a scalar neighbour does not dictate how a list is spaced, or vice
        versa.

        The list here is spaced and comes first; the scalar after it is written tight.
        Element spacing has to come from the list and assignment spacing from the scalar,
        or a new entry of either kind comes out looking like the other.
        """
        spaced_list = styled_entry("", " ", " ", key="lst", values=2)
        tight_scalar = Tree("key_value", [Tree("key", [Token("CNAME", "v")]), value_node()])

        style = probe_entry_style(Tree("start", [spaced_list, tight_scalar]))

        assert style.key_pads == ()
        assert style.element_pads == (" ",)

    def test_falls_back_to_the_last_entry_when_none_has_a_value(self) -> None:
        """Test a container whose only entry is valueless, as ``KEY =`` is.

        With no value to measure against, everything after the key counts as key padding and
        nothing says how elements are separated.
        """
        valueless = Tree("key_null", [ws_node("  "), Tree("key", [Token("CNAME", "a")]), ws_node(" ")])

        style = probe_entry_style(Tree("block", [valueless]))

        assert style.has_donor
        assert style.indent == "  "
        assert style.key_pads == (" ",)
        assert style.element_pads is None

    def test_reads_through_a_transparent_assignment_rule(self) -> None:
        """Test the ``eq`` rule, which binds the run before ``=`` to the literal.

        Left as a bare ``ws*`` beside the one after ``=``, a single run is ambiguous about
        which side it fell on and the reconstructor moves it. The runs inside the rule count
        as though written beside the key, and the rule marks where the literal fell.
        """
        # "a =1": one run before the "=", none after.
        entry = Tree(
            "key_value",
            [Tree("key", [Token("CNAME", "a")]), Tree("eq", [ws_node(" ")]), value_node()],
        )

        style = probe_entry_style(Tree("start", [entry]))

        assert style.key_pads == (" ",)
        assert style.key_split == 1

    def test_a_run_after_the_assignment_is_counted_on_the_other_side(self) -> None:
        """Test the mirror case ``a= 1``: the same one run, split the other way."""
        entry = Tree(
            "key_value",
            [Tree("key", [Token("CNAME", "a")]), Tree("eq", []), ws_node(" "), value_node()],
        )

        style = probe_entry_style(Tree("start", [entry]))

        assert style.key_pads == (" ",)
        assert style.key_split == 0

    def test_a_block_entry_does_not_donate_assignment_spacing(self) -> None:
        """Test that a block is passed over as a donor for the spacing around an assignment.

        It shows the indentation but carries no assignment, so taking its empty run would
        write ``KEY=value`` into a file that spaces its assignments.
        """
        assignment = Tree(
            "key_value",
            [Tree("key", [Token("CNAME", "a")]), Tree("eq", [ws_node(" ")]), ws_node(" "), value_node()],
        )
        block = entry_node("key_block", "blk", Tree("block", []))

        style = probe_entry_style(Tree("start", [assignment, block]))

        assert style.key_pads == (" ", " ")

    def test_reads_only_the_entries_above_where_the_new_one_goes(self) -> None:
        """Test an entry written among others, copying the one above, not the last."""
        container = Tree("block", [styled_entry("  ", " "), styled_entry("    ", " ", key="b")])

        assert probe_entry_style(container, upto=1).indent == "  "
        assert probe_entry_style(container, upto=0) == EntryStyle()

    def test_ignores_children_that_are_tokens(self) -> None:
        """Test that punctuation kept as a bare terminal is not a whitespace run."""
        entry = Tree("key_value", [Tree("key", [Token("CNAME", "a")]), Token("EQ", "="), value_node()])

        assert probe_entry_style(Tree("start", [entry])).key_pads == ()


class TestProbeSiblingBlockStyle:
    """The style a block created this session inherits from the blocks around it."""

    def test_takes_it_from_a_sibling_block(self) -> None:
        """Test that a new block is laid out like the block next to it."""
        sibling = entry_node("key_block", "old", Tree("block", [styled_entry("  ", " ")]))

        style = probe_sibling_block_style(Tree("start", [sibling]))

        assert style.has_donor
        assert style.indent == "  "

    def test_takes_the_nearest_block_able_to_donate(self) -> None:
        """Test that the last block with entries wins, an empty one being passed over."""
        far = entry_node("key_block", "far", Tree("block", [styled_entry("    ", " ", key="x")]))
        near = entry_node("key_block", "near", Tree("block", [styled_entry("  ", " ", key="y")]))
        empty = entry_node("key_block", "new", Tree("block", []))

        assert probe_sibling_block_style(Tree("start", [far, near, empty])).indent == "  "

    def test_reports_no_donor_without_a_usable_block(self) -> None:
        """Test both ways there is nothing to copy: no block at all, and an empty one."""
        entries_only = Tree("start", [styled_entry("", " ")])
        empty_block = Tree("start", [entry_node("key_block", "blk", Tree("block", []))])

        assert probe_sibling_block_style(entries_only) == EntryStyle()
        assert probe_sibling_block_style(empty_block) == EntryStyle()

    def test_takes_the_block_at_the_end_of_a_chain(self) -> None:
        """Test an FMS field table entry: a name block in a model in a type."""
        name = entry_node("key_block", "age", Tree("block", [styled_entry("  ", " ")]))
        model = entry_node("key_block", "ocean_mod", Tree("name_body", [name]))
        kind = entry_node("key_block", "tracer", Tree("model_body", [model]))

        assert probe_sibling_block_style(Tree("start", [kind])).indent == "  "

    def test_searches_upwards_from_the_block_created(self) -> None:
        """Test that the block above a new one wins over a later one."""
        above = entry_node("key_block", "above", Tree("block", [styled_entry("    ", " ", key="x")]))
        new = entry_node("key_block", "new", Tree("block", []))
        below = entry_node("key_block", "below", Tree("block", [styled_entry("  ", " ", key="y")]))

        assert probe_sibling_block_style(Tree("start", [above, new, below]), upto=1).indent == "    "

    def test_ignores_a_block_entry_whose_body_is_not_a_block_node(self) -> None:
        """Test the contract: the body must be a rule *named* ``block``, not just nested."""
        odd = entry_node("key_block", "blk", Tree("body", [styled_entry("  ", " ")]))

        assert probe_sibling_block_style(Tree("start", [odd])) == EntryStyle()


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

    def test_copies_nothing_without_a_reconstructor(self, fortran) -> None:
        """Test that the whitespace can still be read on its own."""
        style = probe_entry_style(group_body(fortran, "&g\n  a = 1\n/\n"))

        assert style.has_donor
        assert style.donors == ()

    def test_a_created_block_copies_the_entries_of_its_sibling(self, fortran) -> None:
        """Test that the first entry of a new group is written like those of the last."""
        tree = fortran.lark.parse("&g\n  a = 'x'\n/\n", start="start")

        style = probe_sibling_block_style(tree, fortran.reconstructor)

        assert shown(style.donor(1, "key_value")) == "  {K0} = {V}\n"
