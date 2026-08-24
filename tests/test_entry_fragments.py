# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the template type and the operations on one template alone.

``entry_fragments`` imports nothing but rule names, so nothing is faked here and no grammar
is compiled. Every template is written out fragment by fragment, which is what lets the
ranking terms be tested one at a time: a real grammar hands you whatever derivations it has,
never the pair that differs in exactly the term under test.
"""

import pytest

from access.config.entry_fragments import (
    Fragment,
    compile_template,
    count_fragments,
    expand_value_slots,
    neighbour_fragment,
    shape_key,
)

KEY, WS, VALUE, BLOCK = Fragment("key"), Fragment("ws"), Fragment("value"), Fragment("block")


def literal(text: str) -> Fragment:
    """Return a fragment of fixed text."""
    return Fragment("literal", text)


# ``KEY = value`` followed by a newline: the shape most grammars derive for a scalar entry.
LINE = (KEY, WS, literal("="), WS, VALUE, literal("\n"))


def test_count_fragments() -> None:
    """Test counting the fragments of one kind."""
    assert count_fragments(LINE, "ws") == 2
    assert count_fragments(LINE, "value") == 1
    assert count_fragments(LINE, "block") == 0


class TestNeighbourFragment:
    """Looking out from a position for the nearest fragment that is not whitespace."""

    def test_skips_whitespace_in_both_directions(self) -> None:
        """Test that whitespace placeholders are passed over: they may render empty."""
        assert neighbour_fragment(LINE, 2, -1) is KEY
        assert neighbour_fragment(LINE, 2, 1) is VALUE

    def test_reports_nothing_past_either_end(self) -> None:
        """Test the ends: this is how a placeholder starting a line is recognised."""
        assert neighbour_fragment(LINE, 0, -1) is None
        assert neighbour_fragment((WS, WS), 0, 1) is None

    def test_accepts_an_index_one_past_the_end(self) -> None:
        """Test the call ``shape_key`` makes to find the last fragment of a template."""
        assert neighbour_fragment(LINE, len(LINE), -1) == literal("\n")


class TestShapeKey:
    """The style-independent ranking terms, smaller being preferred."""

    def test_prefers_a_template_that_ends_the_line(self) -> None:
        """Test the first term, which stops an entry running into what follows it."""
        assert shape_key(LINE)[0] == 0
        assert shape_key((KEY, literal("="), VALUE))[0] == 1

    def test_a_trailing_whitespace_slot_does_not_hide_the_newline(self) -> None:
        """Test that the newline is looked for past a whitespace placeholder at the end.

        The comment on the implementation is precise about this: the *last fragment* has to
        be the newline, not the last literal, or a list continued across lines would
        score as a complete line.
        """
        assert shape_key((KEY, literal("="), VALUE, literal("\n"), WS))[0] == 0
        assert shape_key((KEY, literal("="), VALUE, literal(",\n"), VALUE))[0] == 1

    def test_prefers_a_block_body_on_its_own_line(self) -> None:
        """Test the second term, which leaves a new block room for its first keys."""
        assert shape_key((KEY, literal("<\n"), BLOCK, literal(">\n")))[1] == 0
        assert shape_key((KEY, literal("<"), BLOCK, literal(">\n")))[1] == 1

    def test_a_block_at_the_start_has_nothing_before_it(self) -> None:
        """Test the case where the block placeholder has no preceding fragment at all."""
        assert shape_key((BLOCK, KEY, literal("\n")))[1] == 1

    def test_a_template_with_no_block_passes_the_block_term(self) -> None:
        """Test that a scalar template is not penalised by a term about blocks."""
        assert shape_key(LINE)[1] == 0

    def test_counts_newlines_other_than_the_one_ending_the_entry(self) -> None:
        """Test the third term, which is what stops a blank line being added."""
        assert shape_key(LINE)[2] == 0
        assert shape_key((KEY, literal("="), VALUE, literal("\n\n")))[2] == 1

    def test_measures_fixed_text_ignoring_whitespace(self) -> None:
        """Test the fourth term, which is what rejects an optional trailing separator."""
        assert shape_key(LINE)[3] == len("=")
        assert shape_key((KEY, literal("="), VALUE, literal(","), literal("\n")))[3] == len("=,")

    def test_prefers_more_whitespace_slots_and_fewer_keys(self) -> None:
        """Test the last two terms: spare whitespace is free, a repeated key is not."""
        assert shape_key(LINE)[4] == -2
        assert shape_key(LINE)[5] == 1
        assert shape_key((KEY, literal("%\n"), BLOCK, literal("%"), KEY, literal("\n")))[5] == 2

    def test_orders_a_realistic_pair(self) -> None:
        """Test the terms together, on the choice a Fortran list actually presents."""
        one_line = (KEY, WS, literal("="), WS, VALUE, literal(","), WS, VALUE, literal("\n"))
        continued = (KEY, WS, literal("="), WS, VALUE, literal(",\n"), WS, VALUE, literal("\n"))

        assert shape_key(one_line) < shape_key(continued)


class TestExpandValueSlots:
    """Growing a two-value list template to hold as many values as were asked for."""

    def test_repeats_the_separator_between_the_placeholders(self) -> None:
        """Test that the grammar's own separator is what gets repeated."""
        template = (KEY, literal("="), VALUE, literal(","), WS, VALUE, literal("\n"))

        grown = expand_value_slots(template, 4)

        assert [f.kind for f in grown] == [
            "key", "literal",
            "value", "literal", "ws",
            "value", "literal", "ws",
            "value", "literal", "ws",
            "value", "literal",
        ]  # fmt: skip
        assert count_fragments(grown, "value") == 4

    def test_two_values_is_the_template_unchanged(self) -> None:
        """Test the shortest list a format can express, where nothing needs repeating."""
        template = (KEY, literal("="), VALUE, literal(","), VALUE, literal("\n"))
        assert expand_value_slots(template, 2) == template

    def test_keeps_what_surrounds_the_values(self) -> None:
        """Test that the text before the first value and after the last is left alone."""
        template = (KEY, literal("=("), VALUE, literal(","), VALUE, literal(")\n"))

        grown = expand_value_slots(template, 3)

        assert grown[0] is KEY
        assert grown[1] == literal("=(")
        assert grown[-1] == literal(")\n")


class TestCompileTemplate:
    """Compiling the template text a parser may declare for a new entry."""

    def test_splits_text_around_its_markers(self) -> None:
        """Test that the markers become placeholders and the rest becomes fixed text."""
        template = compile_template("{key} = {value}\n", "key_value", frozenset({"integer"}))

        assert [(f.kind, f.text) for f in template] == [
            ("key", ""),
            ("literal", " = "),
            ("value", ""),
            ("literal", "\n"),
        ]
        # The admitted value rules are carried on the value placeholder, for rendering.
        assert template[2].options == frozenset({"integer"})

    def test_handles_a_block_and_a_repeated_key(self) -> None:
        """Test a block template, whose key is written twice and whose body is a slot."""
        template = compile_template("{key}%\n{block}%{key}\n", "key_block", frozenset())

        assert [f.kind for f in template] == ["key", "literal", "block", "literal", "key", "literal"]

    def test_text_with_no_marker_at_all(self) -> None:
        """Test the branch that consumes the remainder once no marker is left."""
        with pytest.raises(ValueError, match="key"):
            compile_template("just text", "key_null", frozenset())

    def test_a_marker_at_the_very_start(self) -> None:
        """Test that a template opening on a marker emits no empty literal before it."""
        template = compile_template("{key}=", "key_null", frozenset())
        assert [f.kind for f in template] == ["key", "literal"]

    @pytest.mark.parametrize(
        ("text", "category", "message"),
        [
            ("{key}={value}", "key_null", "exactly 0"),
            ("{key}={value}", "key_list", "exactly 2"),
            ("{key}=", "key_block", "'{block}'"),
            ("={value}", "key_value", "'{key}'"),
            ("{key}={block}", "key_null", "'{block}'"),
        ],
    )
    def test_rejects_markers_that_do_not_fit_the_category(self, text, category, message) -> None:
        """Test that a template is checked against what its category requires.

        A list needs two value markers because the text between them is what identifies the
        element separator; a block needs exactly one body marker; every template needs a
        key.
        """
        with pytest.raises(ValueError, match=message):
            compile_template(text, category, frozenset())
