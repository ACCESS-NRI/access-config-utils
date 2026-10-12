# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the views over a compiled grammar.

Nothing is faked here, because there would be nothing left: ``GrammarInfo`` reads
``lark.rules``, and those are lark's own objects. Standing in for them would mean asserting
against a guess at what Lark produces -- and the shapes that matter are exactly the ones it
invents, such as the ``__x_star_N`` helper behind a repetition.

The grammars are therefore real, but small and written for the question at hand.
"""

import pytest

from access.config.grammar_compiled import compile_grammar
from access.config.grammar_info import GrammarInfo

# A grammar with one of each shape the views have to tell apart: a plain repetition of each
# kind, a rule with a longer expansion, a rule that is a single terminal, and a choice of
# value types. Every rule hangs off ``start``, because Lark drops what it cannot reach.
GRAMMAR = """
    start: entry* only_terminal? repeated plus_repeated two_ways?
    entry: key equal value line_end
    equal: "="
    only_terminal: EQUALS
    EQUALS: "=="
    ?value: integer | float
    repeated: integer*
    plus_repeated: nested+
    nested: integer
    two_ways: integer | float | key equal
    line_end: ws* NEWLINE

    %import config.key
    %import config.integer
    %import config.float
    %import config.ws
    %import config.NEWLINE
"""


@pytest.fixture(scope="module")
def info() -> GrammarInfo:
    """Fixture returning the views over the grammar above."""
    return compile_grammar(GRAMMAR).info


def test_from_lark_groups_the_rules_and_finds_the_value_types() -> None:
    """Test that a compiled parser is turned into the two views everything else uses."""
    compiled = compile_grammar(GRAMMAR)
    info = GrammarInfo.from_lark(compiled.lark)

    assert "entry" in info.rules_by_origin
    assert all(str(rule.origin.name) == "entry" for rule in info.rules_by_origin["entry"])
    # The value types some rule uses, wherever it uses them; ``key`` is not one.
    assert info.value_rules == frozenset({"integer", "float"})


def test_compared_by_identity() -> None:
    """Test that two views over the same grammar are distinct keys.

    A caller memoising something per grammar keys it on the instance, so equality has to be
    identity or two grammars would share a cache entry.
    """
    compiled = compile_grammar(GRAMMAR)
    first, second = GrammarInfo.from_lark(compiled.lark), GrammarInfo.from_lark(compiled.lark)

    assert first != second
    assert len({first, second}) == 2


class TestIsRepetitionRule:
    """Which rules accept the combined children of two adjacent nodes."""

    @pytest.mark.parametrize("name", ["repeated", "plus_repeated"])
    def test_a_plain_repetition(self, info, name) -> None:
        """Test both forms, which Lark compiles to a ``_star_``/``_plus_`` helper."""
        assert info.is_repetition_rule(name)

    def test_a_rule_with_a_longer_expansion(self, info) -> None:
        """Test that a rule of several symbols cannot absorb another node's children."""
        assert not info.is_repetition_rule("entry")

    def test_a_rule_whose_alternative_is_a_single_terminal(self, info) -> None:
        """Test the branch rejecting a one-symbol expansion that is a terminal."""
        assert not info.is_repetition_rule("only_terminal")

    def test_a_rule_with_a_mixture_of_alternatives(self, info) -> None:
        """Test that one non-repetition alternative disqualifies the whole rule."""
        assert not info.is_repetition_rule("two_ways")

    def test_a_name_that_is_not_a_rule(self, info) -> None:
        """Test the lookup miss, which is how a caller may ask about any node it holds."""
        assert not info.is_repetition_rule("not_a_rule")


class TestValueRules:
    """The value types a grammar has, which is what a new value may be written as."""

    def test_the_value_types_every_rule_mentions(self, info) -> None:
        """Test that a value type counts wherever it appears in an expansion.

        ``integer`` and ``float`` appear in a choice, ``integer`` also in a repetition and
        on its own; ``key`` and ``equal`` are rules of the grammar but no value type.
        """
        assert info.value_rules == frozenset({"integer", "float"})

    def test_a_grammar_with_no_value_type_has_none(self) -> None:
        """Test that nothing is invented for a grammar that holds no value."""
        keys_only = compile_grammar("""
            start: key*

            %import config.key
            %import common.WS
            %ignore WS
        """).info

        assert keys_only.value_rules == frozenset()
