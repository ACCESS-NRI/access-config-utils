# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the views over a compiled grammar.

Nothing is faked here, because there would be nothing left: ``GrammarInfo`` reads
``lark.rules`` and ``lark.terminals``, and those are lark's own objects. Standing in for
them would mean asserting against a guess at what Lark produces -- and the shapes that
matter are exactly the ones it invents, such as the ``__x_star_N`` helper behind a
repetition.

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


def test_from_lark_groups_the_rules_and_indexes_the_terminals() -> None:
    """Test that a compiled parser is turned into the two lookups everything else uses."""
    compiled = compile_grammar(GRAMMAR)
    info = GrammarInfo.from_lark(compiled.lark)

    assert "entry" in info.rules_by_origin
    assert all(str(rule.origin.name) == "entry" for rule in info.rules_by_origin["entry"])
    # Terminals are keyed by name, under whatever name the import left them with.
    assert "EQUALS" in info.terminals
    assert any(name.endswith("NEWLINE") for name in info.terminals)


def test_compared_by_identity() -> None:
    """Test that two views over the same grammar are distinct keys.

    Template generation memoises against the instance, so equality has to be identity or two
    grammars would share a cache entry.
    """
    compiled = compile_grammar(GRAMMAR)
    first, second = GrammarInfo.from_lark(compiled.lark), GrammarInfo.from_lark(compiled.lark)

    assert first != second
    assert len({first, second}) == 2


class TestSampleTerminal:
    """The text to emit for a terminal, when an entry is being written from scratch."""

    def test_a_literal_terminal_is_reproduced_from_its_pattern(self, info) -> None:
        """Test the ordinary case: punctuation the grammar spells out."""
        assert info.sample_terminal("EQUALS") == "=="

    @pytest.mark.parametrize(("suffix", "text"), [("NEWLINE", "\n"), ("WS_INLINE", " ")])
    def test_whitespace_terminals_have_declared_samples(self, info, suffix, text) -> None:
        """Test the two regular-expression terminals worth reproducing.

        They are matched on the name *suffix*, because an ``%import`` renames a terminal:
        ``WS_INLINE`` arrives as ``config__WS_INLINE``.
        """
        [name] = [name for name in info.terminals if name == suffix or name.endswith(f"__{suffix}")]

        assert name != suffix or "__" not in name
        assert info.sample_terminal(name) == text

    def test_any_other_regular_expression_terminal_cannot_be_reproduced(self, info) -> None:
        """Test the pruning that stops a derivation inventing text.

        A terminal with no sample returns ``None`` and the derivation needing it is
        discarded, which is how an entry never comes out carrying a made-up comment.
        """
        [signed_int] = [name for name in info.terminals if name.endswith("SIGNED_INT")]

        assert info.sample_terminal(signed_int) is None


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
    """Recognising a position in the grammar that holds a value."""

    def test_a_value_type_resolves_to_itself(self, info) -> None:
        """Test the base case, a rule that is registered as a value type."""
        assert info.value_rules("integer") == frozenset({"integer"})

    def test_a_choice_between_value_types_resolves_to_the_union(self, info) -> None:
        """Test the ``?value: integer | float`` shape every grammar uses."""
        assert info.value_rules("value") == frozenset({"integer", "float"})

    def test_a_rule_with_a_longer_expansion_is_not_a_value_position(self, info) -> None:
        """Test that a rule holding more than one symbol resolves to nothing."""
        assert info.value_rules("entry") == frozenset()

    def test_a_choice_that_is_not_all_value_types_resolves_to_nothing(self, info) -> None:
        """Test that one non-value alternative disqualifies the position."""
        assert info.value_rules("two_ways") == frozenset()

    def test_a_name_that_is_not_a_rule(self, info) -> None:
        """Test the lookup miss, reached while walking a rule's own symbols."""
        assert info.value_rules("not_a_rule") == frozenset()

    def test_the_answer_is_memoised(self, info) -> None:
        """Test that the same question is not resolved twice.

        The walk runs on every key addition, and the answer cannot change for a grammar
        already compiled.
        """
        first = info.value_rules("value")
        assert info.value_rules("value") is first

    def test_a_recursive_rule_terminates(self) -> None:
        """Test the guard against a rule reachable from itself.

        Without it the walk would not terminate; with it the cycle resolves to nothing,
        which is right, since a self-referential rule is not a value position.
        """
        recursive = compile_grammar("""
            start: loop
            loop: nested
            nested: loop | integer

            %import config.integer
        """).info

        assert recursive.value_rules("loop") == frozenset()
