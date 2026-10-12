# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the template type and the operations on one template alone.

``entry_fragments`` imports nothing but rule names, so nothing is faked here and no grammar
is compiled. Every template is written out fragment by fragment, which is what lets each
operation be tested on exactly the shape it is about.
"""

import pytest

from access.config.entry_fragments import Fragment, compile_template, expand_value_slots, neighbour_fragment

KEY, WS, VALUE = Fragment("key"), Fragment("ws"), Fragment("value")


def literal(text: str) -> Fragment:
    """Return a fragment of fixed text."""
    return Fragment("literal", text)


# ``KEY = value`` followed by a newline: the shape most grammars derive for a scalar entry.
LINE = (KEY, WS, literal("="), WS, VALUE, literal("\n"))


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
        """Test the call that finds the last fragment of a template."""
        assert neighbour_fragment(LINE, len(LINE), -1) == literal("\n")


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
        assert [f.kind for f in grown].count("value") == 4

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
    """Compiling the template text a parser declares for a new entry."""

    def test_splits_text_around_its_markers(self) -> None:
        """Test that the markers become placeholders and the rest becomes fixed text."""
        template = compile_template("{key} = {value}\n", "key_value", 1, None)

        assert [(f.kind, f.text) for f in template] == [
            ("key", ""),
            ("literal", " = "),
            ("value", ""),
            ("literal", "\n"),
        ]

    def test_a_key_written_twice(self) -> None:
        """Test a block template, whose key closes it as well as opening it."""
        template = compile_template("{key}%\n%{key}\n", "key_block", 1, None)

        assert [f.kind for f in template] == ["key", "literal", "key", "literal"]

    def test_text_with_no_marker_at_all(self) -> None:
        """Test that a template without a key is refused."""
        with pytest.raises(ValueError, match="key"):
            compile_template("just text", "key_null", 1, None)

    def test_a_marker_at_the_very_start(self) -> None:
        """Test that a template opening on a marker emits no empty literal before it."""
        template = compile_template("{key}=", "key_null", 1, None)
        assert [f.kind for f in template] == ["key", "literal"]

    @pytest.mark.parametrize(
        ("text", "category", "message"),
        [
            ("{key}={value}", "key_null", "exactly 0"),
            ("{key}={value}", "key_list", "exactly 2"),
            ("{key}={value}{value}", "key_value", "exactly 1"),
            ("={value}", "key_value", "'{key}'"),
        ],
    )
    def test_rejects_markers_that_do_not_fit_the_category(self, text, category, message) -> None:
        """Test that a template is checked against what its category requires.

        A list needs two value markers because the text between them is what identifies the
        element separator, and every template needs a key.
        """
        with pytest.raises(ValueError, match=message):
            compile_template(text, category, 1, None)

    @pytest.mark.parametrize(("depth", "written"), [(1, "A"), (2, "A%B"), (3, "A%B%C")])
    def test_keys_writes_every_key_of_a_chain(self, depth, written) -> None:
        """Test that ``{keys:SEP}`` writes one key per level, parted by SEP."""
        template = compile_template("{keys:%} =\n", "key_null", depth, None)

        assert [(f.kind, f.level) for f in template if f.kind == "key"] == [("key", level) for level in range(depth)]
        assert "".join(f.text if f.kind == "literal" else "A%B%C"[2 * f.level] for f in template) == written + " =\n"

    def test_a_separator_may_hold_any_text_but_a_closing_brace(self) -> None:
        """Test the field table header, whose keys are parted by ``", "`` inside quotes."""
        template = compile_template('"{keys:", "}"/\n', "key_block", 3, None)

        assert [f.text for f in template if f.kind == "literal"] == ['"', '", "', '", "', '"/\n']

    def test_a_template_for_one_key_cannot_write_a_chain(self) -> None:
        """Test that ``{key}`` is no template at a depth of two: it names the one key."""
        assert compile_template("{key} = 1\n", "key_null", 2, None) is None

    @pytest.mark.parametrize(
        ("text", "indent", "written"),
        [
            pytest.param(" {key} =\n", None, " ", id="keeps-its-own"),
            pytest.param(" {key} =\n", "    ", "    ", id="replaces-its-own"),
            pytest.param(" {key} =\n", "", "", id="removes-its-own"),
            pytest.param("{key} =\n", "  ", "  ", id="adds-one"),
            pytest.param('\n  "{key}" =\n', "    ", '\n    "', id="after-a-line-break"),
            pytest.param("&{key}\n", "  ", "  &", id="before-a-sigil"),
        ],
    )
    def test_indent_replaces_the_whitespace_the_first_keys_line_starts_with(self, text, indent, written) -> None:
        """Test that the file's indentation takes the place of the template's own."""
        template = compile_template(text, "key_null", 1, indent)

        key = next(index for index, fragment in enumerate(template) if fragment.kind == "key")
        assert "".join(f.text for f in template[:key]) == written
