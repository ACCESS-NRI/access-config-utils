# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for choosing a template and writing the text of a new entry.

The templates a grammar could derive are supplied here instead of derived, by faking
``grammar_templates`` and the two other package functions this module calls. That is what
lets ranking be tested at all: it is a choice *between* templates, and a real grammar offers
whichever pair it happens to admit rather than two differing in one term.

``info`` is passed as ``None`` throughout: with generation faked, nothing looks at it.
"""

from collections.abc import Iterator
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from access.config import entry_render
from access.config.entry_fragments import Fragment
from access.config.entry_render import iter_entry_snippets, render
from access.config.entry_style import EntryStyle
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY, ValueTypeHandler

KEY, WS, BLOCK = Fragment("key"), Fragment("ws"), Fragment("block")
NUMERIC = frozenset({"integer", "float", "string", "logical", "identifier"})


def literal(text: str) -> Fragment:
    """Return a fragment of fixed text."""
    return Fragment("literal", text)


def value(options: frozenset[str] = NUMERIC) -> Fragment:
    """Return a value placeholder admitting *options*."""
    return Fragment("value", options=options)


# ``KEY = value\n``, the shape most grammars derive for a scalar entry.
LINE = (KEY, WS, literal("="), WS, value(), literal("\n"))
# ``KEY = a, b\n``, the two-value form every list template starts from.
LIST_LINE = (KEY, WS, literal("="), WS, value(), literal(","), WS, value(), literal("\n"))


@pytest.fixture
def offer() -> Iterator[object]:
    """Return a function declaring the templates generation should be taken to have found.

    Patches the module under test, not the module the names come from: every import in this
    package is a ``from X import name``, so each consumer holds its own reference. The
    patches are entered on the stack rather than in a ``with`` here, because the values
    they return are only known once the returned function is called; the stack unwinds them
    when the fixture tears down.
    """
    with ExitStack() as stack:

        def _offer(*templates, admitted: frozenset[str] = NUMERIC):
            stack.enter_context(
                patch.multiple(
                    entry_render,
                    grammar_templates=lambda info, c, cat: tuple(templates),
                    admitted_value_rules=lambda info, c, cat: admitted,
                )
            )

        yield _offer


def snippets(*args, **kwargs) -> list[str]:
    """Return every candidate ``iter_entry_snippets`` yields, in order."""
    return list(iter_entry_snippets(None, "start", *args, **kwargs))


def copied(signature: tuple[int, str], *fragments: Fragment) -> EntryStyle:
    """Return a style carrying *fragments* as the template copied for *signature*."""
    return EntryStyle(has_donor=True, donors=((signature, tuple(fragments)),))


class TestRender:
    """Substituting the key, the values and the whitespace into a template."""

    def test_canonical_layout_without_a_donor(self) -> None:
        """Test the default spacing: one space where allowed, none at either end."""
        assert render(LINE, ["A"], ["1"], EntryStyle()) == "A = 1\n"

    def test_reproduces_a_donor_run_per_slot(self) -> None:
        """Test that the recorded runs are used, one per stylable slot."""
        style = EntryStyle(indent="", key_pads=("  ", "  "), key_split=1, has_donor=True)
        assert render(LINE, ["A"], ["1"], style) == "A  =  1\n"

    def test_a_style_whose_runs_fall_the_other_side_is_ignored(self) -> None:
        """Test that runs are dropped rather than moved across the assignment.

        The donor put both runs before its ``=``; this template has one either side.
        Writing them here would space the entry unlike the neighbour it was copied from, so
        the canonical layout is used instead.
        """
        style = EntryStyle(key_pads=("  ", "  "), key_split=2, has_donor=True)
        assert render(LINE, ["A"], ["1"], style) == "A = 1\n"

    def test_a_style_offering_the_wrong_number_of_runs_is_ignored(self) -> None:
        """Test that a mismatch falls back to the canonical layout rather than guessing."""
        style = EntryStyle(key_pads=(" ", " ", " "), has_donor=True)
        assert render(LINE, ["A"], ["1"], style) == "A = 1\n"

    def test_places_the_indent_immediately_before_the_key(self) -> None:
        """Test that of several leading slots, the last one carries the indentation."""
        template = (WS, WS, KEY, literal("=\n"))
        assert render(template, ["A"], [], EntryStyle(indent="  ", has_donor=True)) == "  A=\n"

    def test_ignores_a_block_placeholder(self) -> None:
        """Test that the contents of a new block render as nothing; they are added after."""
        assert render((KEY, literal("<"), BLOCK, literal(">")), ["K"], [], EntryStyle()) == "K<>"

    def test_writes_each_key_of_a_chain_where_its_level_says(self) -> None:
        """Test a derived-type component, whose keys are written one after the other."""
        template = (KEY, literal("%"), Fragment("key", level=1), literal(" = 1\n"))
        assert render(template, ["A", "B"], [], EntryStyle()) == "A%B = 1\n"


class TestRanking:
    """Which of the offered templates is written, when several would do."""

    def test_prefers_a_template_that_ends_the_line(self, offer) -> None:
        """Test that a form running into what follows it loses to one that terminates."""
        offer((KEY, literal("="), value()), LINE)
        assert snippets("key_value", ("K",), 1, EntryStyle())[0] == "K = 1\n"

    def test_prefers_less_fixed_text(self, offer) -> None:
        """Test the term rejecting an optional trailing separator, as Fortran allows."""
        offer((KEY, WS, literal("="), WS, value(), literal(",\n")), LINE)
        assert snippets("key_value", ("K",), 1, EntryStyle())[0] == "K = 1\n"

    def test_style_only_decides_between_equals(self, offer) -> None:
        """Test that matching a neighbour ranks below the amount of fixed text.

        Ranked any higher, a grammar with an optional trailing separator would have one
        added purely to make the whitespace runs line up.
        """
        with_separator = (KEY, WS, literal("="), WS, value(), WS, literal(",\n"))
        offer(with_separator, LINE)

        # The style has three runs, which only the separator form can place.
        style = EntryStyle(key_pads=(" ", " "), has_donor=True)
        assert snippets("key_value", ("K",), 1, style)[0] == "K = 1\n"

    def test_style_breaks_a_tie(self, offer) -> None:
        """Test that between templates writing the same text, the donor match wins."""
        one_slot = (KEY, literal(" ="), WS, value(), literal("\n"))
        offer(one_slot, LINE)

        style = EntryStyle(key_pads=("  ", "  "), key_split=1, has_donor=True)
        assert snippets("key_value", ("K",), 1, style)[0] == "K  =  1\n"

    def test_generation_order_is_the_final_tie_break(self, offer) -> None:
        """Test that two indistinguishable templates come out in the order offered."""
        offer(LINE, tuple(LINE))
        assert snippets("key_value", ("K",), 1, EntryStyle()) == ["K = 1\n", "K = 1\n"]


class TestValueChoice:
    """How a value is written when several rules accept its Python type."""

    def test_offers_every_accepting_rule_most_preferred_first(self, offer) -> None:
        """Test that the preferred rendering comes first but the others still follow.

        The caller validates each candidate, so a preferred form that reads back as the
        wrong type has to be followed by an alternative rather than being the only offer.
        """
        offer(LINE)
        priority = ("identifier", "string")

        assert snippets("key_value", ("K",), "word", EntryStyle(), None, priority) == ["K = word\n", 'K = "word"\n']

    def test_yields_nothing_when_no_admitted_rule_accepts_the_value(self, offer) -> None:
        """Test that an unrepresentable value yields no candidate, not a bad one."""
        offer(LINE, admitted=frozenset({"integer"}))
        assert snippets("key_value", ("K",), "text", EntryStyle(), None, ("integer",)) == []

    def test_writes_a_list_of_one_type_with_the_grammar_separator(self, offer) -> None:
        """Test that the template is grown using the text between its placeholders."""
        offer(LIST_LINE)
        assert snippets("key_list", ("K",), [1, 2, 3], EntryStyle(), None, ("integer",)) == ["K = 1, 2, 3\n"]

    def test_writes_a_list_whose_elements_differ_in_type(self, offer) -> None:
        """Test the per-value fallback, for a list no single rule covers.

        No one rule accepts both an integer and a string, so each element is given the rule
        its own type needs.
        """
        offer(LIST_LINE)
        priority = ("integer", "string")

        assert snippets("key_list", ("K",), [1, "two"], EntryStyle(), None, priority) == ['K = 1, "two"\n']

    def test_a_valueless_entry_renders_with_no_value_at_all(self, offer) -> None:
        """Test the ``key_null`` path, where the template has no value placeholder."""
        offer((KEY, WS, literal("=\n")))
        assert snippets("key_null", ("K",), None, EntryStyle()) == ["K =\n"]


class TestParserTemplateOverride:
    """The template a parser may declare for a container it wants laid out its own way."""

    OVERRIDE = {("start", "key_value"): " {key}: {value}\n"}

    def test_is_offered_first_when_nothing_can_supply_a_style(self, offer) -> None:
        """Test that the declared template wins where no neighbour can supply a style."""
        offer(LINE)

        assert snippets("key_value", ("K",), 1, EntryStyle(), self.OVERRIDE)[0] == " K: 1\n"

    def test_is_not_offered_when_a_neighbour_can(self, offer) -> None:
        """Test that copying the entries already in the file beats a declared convention.

        A template is fixed text, so it cannot reproduce a neighbour's whitespace; offered
        unconditionally it would space a new entry unlike the one directly above it.
        """
        offer(LINE)

        assert snippets("key_value", ("K",), 1, EntryStyle(has_donor=True), self.OVERRIDE)[0] == "K = 1\n"

    def test_a_declaration_for_another_container_is_ignored(self, offer) -> None:
        """Test that the override is looked up by container and category together."""
        offer(LINE)
        elsewhere = {("block", "key_value"): " {key}: {value}\n"}

        assert snippets("key_value", ("K",), 1, EntryStyle(), elsewhere)[0] == "K = 1\n"

    def test_the_generated_candidates_still_follow(self, offer) -> None:
        """Test that the override is a preference, not a replacement.

        If it no longer fits the grammar the caller simply moves on, so the derived text has
        to be offered behind it.
        """
        offer(LINE)

        assert snippets("key_value", ("K",), 1, EntryStyle(), self.OVERRIDE) == [" K: 1\n", "K = 1\n"]


class TestCopiedTemplate:
    """The template copied from a neighbouring entry, offered ahead of the grammar's."""

    # ``K: value``, where the grammar's own template writes ``K = value``.
    COPIED = (KEY, literal(": "), Fragment("value", notation=("string", "'old'")), literal("\n"))

    def test_is_offered_first_and_keeps_the_notation(self, offer) -> None:
        """Test that the copy comes first, its value quoted as the neighbour's was."""
        offer(LINE)

        texts = snippets("key_value", ("K",), "x", copied((1, "key_value"), *self.COPIED), None, ("string",))

        assert texts == ["K: 'x'\n", 'K = "x"\n']

    def test_writes_a_value_of_another_rule_afresh(self, offer) -> None:
        """Test that the copy admits what the container does, not what its neighbour had."""
        offer(admitted=frozenset({"integer"}))

        texts = snippets("key_value", ("K",), 1, copied((1, "key_value"), *self.COPIED), None, ("integer",))

        assert texts == ["K: 1\n"]

    @pytest.mark.parametrize(
        "signature",
        [pytest.param((1, "key_list"), id="other-category"), pytest.param((2, "key_value"), id="other-depth")],
    )
    def test_is_only_offered_for_the_kind_it_was_copied_from(self, offer, signature) -> None:
        """Test that a list does not dictate how a scalar is written, nor a chain a key."""
        offer(LINE)

        texts = snippets("key_value", ("K",), "x", copied(signature, *self.COPIED), None, ("string",))

        assert texts == ['K = "x"\n']

    def test_writes_a_chain_a_key_per_level(self, offer) -> None:
        """Test the copy of a derived-type component, for a key path of the same depth."""
        offer()
        template = (KEY, literal("%"), Fragment("key", level=1), literal(" = "), value(), literal("\n"))

        texts = snippets("key_value", ("A", "B"), 1, copied((2, "key_value"), *template), None, ("integer",))

        assert texts == ["A%B = 1\n"]

    def test_falls_back_where_the_notation_cannot_hold_the_value(self, offer) -> None:
        """Test that a value the copied notation refuses is written as new ones are."""

        def particular(value: str, token: str) -> str:
            if token:
                raise TypeError("Not in this notation")
            return value

        handler = ValueTypeHandler(
            type_check=lambda value: isinstance(value, str),
            from_token=str,
            to_token=particular,
            seed_token=lambda value: "",
        )
        offer(admitted=frozenset({"particular"}))
        style = copied((1, "key_value"), KEY, literal(": "), Fragment("value", notation=("particular", "old")))

        with patch.dict(VALUE_TYPE_HANDLER_REGISTRY, {"particular": handler}):
            assert snippets("key_value", ("K",), "x", style, None, ("particular",)) == ["K: x"]


def test_snippet_cap_is_enforced(offer) -> None:
    """Test that the number of candidates offered for one insertion is bounded."""
    with patch.object(entry_render, "_MAX_SNIPPETS", 2):
        offer(LINE, LINE, LINE, LINE)

        assert len(snippets("key_value", ("K",), 1, EntryStyle())) == 2
