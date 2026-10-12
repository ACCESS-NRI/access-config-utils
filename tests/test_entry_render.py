# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for choosing a template and writing the text of a new entry.

Nothing is faked: the template copied from a neighbour is built here fragment by fragment,
and the one a parser declares is a string, so each test says exactly what it is offered and
asserts what is written from it.
"""

from unittest.mock import patch

import pytest

from access.config.entry_fragments import Fragment
from access.config.entry_render import iter_entry_snippets, render
from access.config.entry_style import EntryStyle
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY, ValueTypeHandler

KEY = Fragment("key")
NUMERIC = frozenset({"integer", "float", "string", "logical", "identifier"})
PRIORITY = ("logical", "integer", "float", "string", "identifier")


def literal(text: str) -> Fragment:
    """Return a fragment of fixed text."""
    return Fragment("literal", text)


def snippets(category, keys, value, style, declared=None, admitted=NUMERIC, priority=PRIORITY) -> list[str]:
    """Return every candidate ``iter_entry_snippets`` yields, in order."""
    return list(iter_entry_snippets(category, keys, value, style, declared, admitted, priority))


def copied(signature: tuple[int, str], *fragments: Fragment) -> EntryStyle:
    """Return a style carrying *fragments* as the template copied for *signature*."""
    return EntryStyle(indent="", donors=((signature, tuple(fragments)),))


# ``KEY = value\n``, the shape most formats write for a scalar entry.
LINE = (KEY, literal(" = "), Fragment("value"), literal("\n"))


class TestRender:
    """Substituting the key and the values into a template."""

    def test_writes_the_text_around_the_placeholders(self) -> None:
        """Test that fixed text stays where it is and the placeholders take their texts."""
        assert render(LINE, ["A"], ["1"]) == "A = 1\n"

    def test_writes_each_value_in_turn(self) -> None:
        """Test a list, whose values are taken in the order they were given."""
        template = (KEY, literal(" = "), Fragment("value"), literal(", "), Fragment("value"))

        assert render(template, ["A"], ["1", "2"]) == "A = 1, 2"

    def test_writes_each_key_of_a_chain_where_its_level_says(self) -> None:
        """Test a derived-type component, whose keys are written one after the other."""
        template = (KEY, literal("%"), Fragment("key", level=1), literal(" = 1\n"))
        assert render(template, ["A", "B"], []) == "A%B = 1\n"

    def test_writes_a_key_more_than_once(self) -> None:
        """Test a block that is closed by the key that opened it."""
        template = (KEY, literal("%\n%"), KEY, literal("\n"))
        assert render(template, ["K"], []) == "K%\n%K\n"


class TestValueChoice:
    """How a value is written when several rules accept its Python type."""

    LINE = "{key} = {value}\n"
    LIST = "{key} = {value}, {value}\n"

    def test_offers_every_accepting_rule_most_preferred_first(self) -> None:
        """Test that the preferred rendering comes first but the others still follow.

        The caller validates each candidate, so a preferred form that reads back as the
        wrong type has to be followed by an alternative rather than being the only offer.
        """
        texts = snippets("key_value", ("K",), "word", EntryStyle(), self.LINE, priority=("identifier", "string"))

        assert texts == ["K = word\n", 'K = "word"\n']

    def test_yields_nothing_when_no_admitted_rule_accepts_the_value(self) -> None:
        """Test that an unrepresentable value yields no candidate, not a bad one."""
        texts = snippets("key_value", ("K",), "text", EntryStyle(), self.LINE, frozenset({"integer"}), ("integer",))

        assert texts == []

    def test_writes_a_list_of_one_type_with_the_declared_separator(self) -> None:
        """Test that the template is grown using the text between its placeholders."""
        texts = snippets("key_list", ("K",), [1, 2, 3], EntryStyle(), self.LIST, priority=("integer",))

        assert texts == ["K = 1, 2, 3\n"]

    def test_writes_a_list_whose_elements_differ_in_type(self) -> None:
        """Test the per-value fallback, for a list no single rule covers.

        No one rule accepts both an integer and a string, so each element is given the rule
        its own type needs.
        """
        texts = snippets("key_list", ("K",), [1, "two"], EntryStyle(), self.LIST, priority=("integer", "string"))

        assert texts == ['K = 1, "two"\n']

    def test_a_valueless_entry_renders_with_no_value_at_all(self) -> None:
        """Test the ``key_null`` path, where the template has no value placeholder."""
        assert snippets("key_null", ("K",), None, EntryStyle(), "{key} =\n") == ["K =\n"]


class TestDeclaredTemplate:
    """The template a parser declares for a kind of entry the file has none to copy of."""

    DECLARED = " {key}: {value}\n"

    def test_is_offered_when_the_file_shows_no_indent(self) -> None:
        """Test that the declared template is written where nothing above can be copied."""
        assert snippets("key_value", ("K",), 1, EntryStyle(), self.DECLARED) == [" K: 1\n"]

    def test_takes_the_indent_of_the_entry_above_and_then_its_own(self) -> None:
        """Test that the file's indentation is tried first, and the template's own after it.

        A grammar that allows no indentation there refuses the first, and the second is the
        one that parses.
        """
        style = EntryStyle(indent="    ")

        assert snippets("key_value", ("K",), 1, style, self.DECLARED) == ["    K: 1\n", " K: 1\n"]

    def test_an_entry_above_with_no_indent_removes_the_default(self) -> None:
        """Test that an unindented file gets an unindented entry, the default after it."""
        assert snippets("key_value", ("K",), 1, EntryStyle(indent=""), self.DECLARED) == ["K: 1\n", " K: 1\n"]

    def test_writes_every_key_of_a_chain(self) -> None:
        """Test that ``{keys:SEP}`` serves a key path of any depth."""
        texts = snippets("key_value", ("A", "B"), 1, EntryStyle(), "{keys:%} = {value}\n", priority=("integer",))

        assert texts == ["A%B = 1\n"]

    def test_a_template_for_one_key_is_not_offered_for_a_chain(self) -> None:
        """Test that a template writing one key is not used for a key path of two."""
        assert snippets("key_value", ("A", "B"), 1, EntryStyle(), self.DECLARED) == []

    def test_nothing_is_offered_for_a_kind_with_no_template_and_no_neighbour(self) -> None:
        """Test that a format declaring nothing has nothing to write a first entry from."""
        assert snippets("key_value", ("K",), 1, EntryStyle()) == []


class TestCopiedTemplate:
    """The template copied from a neighbouring entry, offered ahead of the declared one."""

    # ``K: value``, where the declared template writes ``K = value``.
    COPIED = (KEY, literal(": "), Fragment("value", notation=("string", "'old'")), literal("\n"))
    DECLARED = "{key} = {value}\n"

    def test_is_offered_first_and_keeps_the_notation(self) -> None:
        """Test that the copy comes first, its value quoted as the neighbour's was."""
        style = copied((1, "key_value"), *self.COPIED)

        texts = snippets("key_value", ("K",), "x", style, self.DECLARED, priority=("string",))

        assert texts == ["K: 'x'\n", 'K = "x"\n']

    def test_writes_a_value_of_another_rule_afresh(self) -> None:
        """Test that the copy admits what the format does, not what its neighbour had."""
        style = copied((1, "key_value"), *self.COPIED)

        texts = snippets("key_value", ("K",), 1, style, None, frozenset({"integer"}), ("integer",))

        assert texts == ["K: 1\n"]

    @pytest.mark.parametrize(
        "signature",
        [pytest.param((1, "key_list"), id="other-category"), pytest.param((2, "key_value"), id="other-depth")],
    )
    def test_is_only_offered_for_the_kind_it_was_copied_from(self, signature) -> None:
        """Test that a list does not dictate how a scalar is written, nor a chain a key."""
        style = copied(signature, *self.COPIED)

        texts = snippets("key_value", ("K",), "x", style, self.DECLARED, priority=("string",))

        assert texts == ['K = "x"\n']

    def test_writes_a_chain_a_key_per_level(self) -> None:
        """Test the copy of a derived-type component, for a key path of the same depth."""
        template = (KEY, literal("%"), Fragment("key", level=1), literal(" = "), Fragment("value"), literal("\n"))
        style = copied((2, "key_value"), *template)

        assert snippets("key_value", ("A", "B"), 1, style, None, priority=("integer",)) == ["A%B = 1\n"]

    def test_falls_back_where_the_notation_cannot_hold_the_value(self) -> None:
        """Test that a value the copied notation refuses is written as new ones are."""

        def particular(value: str, token: str) -> str:
            if token:
                raise TypeError("Not in this notation")
            return value

        handler = ValueTypeHandler(
            type_check=lambda value: isinstance(value, str),
            from_token=str,
            to_token=particular,
        )
        style = copied((1, "key_value"), KEY, literal(": "), Fragment("value", notation=("particular", "old")))

        with patch.dict(VALUE_TYPE_HANDLER_REGISTRY, {"particular": handler}):
            texts = snippets("key_value", ("K",), "x", style, None, frozenset({"particular"}), ("particular",))

        assert texts == ["K: x"]
