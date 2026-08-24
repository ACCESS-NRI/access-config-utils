# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the whitespace a new entry should be written with.

Nothing is faked: ``entry_style`` reads templates and parse trees, and both are built here
directly. That is what makes the cases separable -- a container whose only entry has no
value, a scalar and a list disagreeing about spacing, a template with an indent slot and one
without. A grammar gives you the shapes it happens to derive, not the one under test.
"""

import pytest
from conftest import entry_node, value_node, ws_node
from lark import Token, Tree

from access.config.entry_fragments import Fragment
from access.config.entry_style import (
    EntryStyle,
    canonical_ws,
    classify_slots,
    probe_entry_style,
    probe_sibling_block_style,
    style_matches,
)

KEY, WS, VALUE, BLOCK = Fragment("key"), Fragment("ws"), Fragment("value"), Fragment("block")


def literal(text: str) -> Fragment:
    """Return a fragment of fixed text."""
    return Fragment("literal", text)


# ``  KEY = value\n``: an indent slot, two around the assignment, one before the newline.
LINE = (WS, KEY, WS, literal("="), WS, VALUE, WS, literal("\n"))

# ``  KEY = a, b\n``: the same, with an element separator between two values.
LIST_LINE = (WS, KEY, WS, literal("="), WS, VALUE, literal(","), WS, VALUE, WS, literal("\n"))


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

    def test_empty_before_a_separator_following_a_value(self) -> None:
        """Test that a list reads as ``a, b`` rather than ``a , b``."""
        template = (VALUE, WS, literal(","), WS, VALUE)
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


class TestClassifySlots:
    """Grouping the whitespace placeholders by what each one spaces out."""

    def test_splits_a_scalar_line_into_its_regions(self) -> None:
        """Test that indent, assignment padding and the trailing slot are told apart."""
        slots = classify_slots(LINE)

        assert slots.leading == (0,)
        assert slots.key == (2, 4)
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
        style = EntryStyle(indent="  ", key_pads=(" ", " "), element_pads=None, has_donor=True)
        assert style_matches(LINE, style)

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

    def test_ignores_a_block_entry_whose_body_is_not_a_block_node(self) -> None:
        """Test the contract: the body must be a rule *named* ``block``, not just nested."""
        odd = entry_node("key_block", "blk", Tree("body", [styled_entry("  ", " ")]))

        assert probe_sibling_block_style(Tree("start", [odd])) == EntryStyle()
