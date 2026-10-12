# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for turning grammar text into the artefacts everything else works with.

Nothing is faked. The module's whole job is to build a ``Lark`` and a ``Reconstructor``, so
standing in for either would leave nothing to test -- and the one branch worth pinning, the
probe deciding whether ``block`` can be a second start symbol, is only observable in what
Lark accepts afterwards. The same goes for ``GrammarInfo``, the views over a compiled
grammar: it reads ``lark.rules``, which are lark's own objects, and the shapes that matter
are exactly the ones Lark invents, such as the ``__x_star_N`` helper behind a repetition.

The grammars are real, and as short as the question allows.
"""

import pytest
from lark import Lark
from lark.exceptions import ConfigurationError
from lark.reconstruct import Reconstructor

from access.config.grammar_compiled import (
    GRAMMAR_DIR,
    CompiledGrammar,
    GrammarInfo,
    ParseContext,
    compile_grammar,
)
from access.config.grammar_values import VALUE_RULE_PRIORITY

WITH_BLOCK = """
    start: entry*
    entry: key "<" block ">" -> key_block
    block: inner*
    inner: key "=" integer

    %import config.key
    %import config.integer
    %import common.WS
    %ignore WS
"""

WITHOUT_BLOCK = """
    start: entry*
    entry: key "=" integer -> key_value

    %import config.key
    %import config.integer
    %import common.WS
    %ignore WS
"""


@pytest.fixture(autouse=True)
def _isolate_the_cache():
    """Give every test a cache of its own, so none can see another's compiled grammar."""
    compile_grammar.cache_clear()
    yield
    compile_grammar.cache_clear()


class TestCompileGrammar:
    """Building the three artefacts, and reusing them."""

    def test_builds_the_parser_the_reconstructor_and_the_views(self) -> None:
        """Test that one call yields everything a parsed configuration needs."""
        compiled = compile_grammar(WITHOUT_BLOCK)

        assert isinstance(compiled, CompiledGrammar)
        assert isinstance(compiled.lark, Lark)
        assert isinstance(compiled.reconstructor, Reconstructor)
        assert isinstance(compiled.info, GrammarInfo)

    def test_resolves_imports_from_the_shared_grammar(self) -> None:
        """Test that ``%import config...`` finds ``config.lark`` beside the package."""
        assert (GRAMMAR_DIR / "config.lark").is_file()
        assert "key" in compile_grammar(WITHOUT_BLOCK).info.rules_by_origin

    def test_the_same_text_is_compiled_once(self) -> None:
        """Test the cache, since compiling costs far more than parsing with the result."""
        assert compile_grammar(WITHOUT_BLOCK) is compile_grammar(WITHOUT_BLOCK)

    def test_different_text_gets_its_own_result(self) -> None:
        """Test that the key is the grammar text, so two formats do not share a parser."""
        assert compile_grammar(WITHOUT_BLOCK) is not compile_grammar(WITH_BLOCK)


class TestStartSymbols:
    """The probe deciding whether a grammar gets ``block`` as a second start symbol."""

    def test_a_grammar_with_a_block_rule_can_parse_one(self) -> None:
        """Test that an entry can be parsed in the container it will live in.

        This is what makes adding a key to a block possible at all: the synthesised text is
        handed back to Lark with ``block`` as the start symbol.
        """
        lark = compile_grammar(WITH_BLOCK).lark

        assert lark.parse("a=1", start="block").data == "block"

    def test_a_grammar_without_one_has_no_block_start_symbol(self) -> None:
        """Test that ``block`` is not declared for a format that has no such rule.

        Lark rejects a start symbol naming a rule that does not exist, so the probe is what
        lets a blockless format compile at all.
        """
        lark = compile_grammar(WITHOUT_BLOCK).lark

        assert lark.parse("a=1", start="start").data == "start"
        with pytest.raises(ConfigurationError):
            lark.parse("a=1", start="block")

    def test_the_start_symbol_must_always_be_named(self) -> None:
        """Test the consequence of having more than one: Lark will not guess.

        Every ``parse`` call in the package passes ``start=`` for this reason.
        """
        with pytest.raises(ConfigurationError):
            compile_grammar(WITH_BLOCK).lark.parse("a<b=1>")


class TestParseContext:
    """The compiled grammar paired with what the parser class declares."""

    @pytest.fixture
    def ctx(self) -> ParseContext:
        """Return a context over a real compiled grammar."""
        return ParseContext(
            compile_grammar(WITHOUT_BLOCK), True, {("start", "key_value"): "{key}={value}"}, ("integer",)
        )

    def test_exposes_the_artefacts_of_its_grammar(self, ctx) -> None:
        """Test the three accessors, saving every caller a reach through ``.grammar``."""
        assert ctx.lark is ctx.grammar.lark
        assert ctx.reconstructor is ctx.grammar.reconstructor
        assert ctx.info is ctx.grammar.info

    def test_carries_what_the_parser_declared(self, ctx) -> None:
        """Test that the per-format settings travel with the grammar they apply to."""
        assert ctx.case_sensitive_keys is True
        assert ctx.entry_templates == {("start", "key_value"): "{key}={value}"}
        assert ctx.value_rule_priority == ("integer",)

    @pytest.mark.parametrize(
        ("text", "accepted"),
        [
            pytest.param("a_b", True, id="identifier"),
            pytest.param("a-b", False, id="hyphen"),
            pytest.param("2fast", False, id="leading-digit"),
            pytest.param("", False, id="empty"),
            pytest.param(" a", False, id="ignored-whitespace"),
            pytest.param("a b", False, id="two-names"),
        ],
    )
    def test_accepts_key_asks_the_grammar(self, ctx, text, accepted) -> None:
        """Test that a name is valid when the format's ``key`` rule reads it back whole."""
        assert ctx.accepts_key(text) is accepted

    def test_accepts_key_follows_a_wider_key_rule(self) -> None:
        """Test that a format whose names hold hyphens and slashes accepts them."""
        grammar = """
            start: entry*
            entry: key "=" integer -> key_value
            key: NAME
            NAME: /[A-Za-z0-9_\\-\\/]+/
            %import config.integer
            %import common.WS
            %ignore WS
        """
        ctx = ParseContext(compile_grammar(grammar), True, {}, VALUE_RULE_PRIORITY)

        assert ctx.accepts_key("horizontal-advection-scheme")
        assert ctx.accepts_key("ocean_age_tracer/global")

    def test_accepts_key_without_a_key_rule(self) -> None:
        """Test that a grammar with no ``key`` rule, and so no entries, accepts no name."""
        ctx = ParseContext(compile_grammar("start: integer\n%import config.integer"), True, {}, VALUE_RULE_PRIORITY)

        assert not ctx.accepts_key("a")

    def test_normalise_key_leaves_a_case_sensitive_key_alone(self, ctx) -> None:
        """Test the format that matches keys exactly, which is all but the namelist."""
        assert ctx.normalise_key("MixedCase") == "MixedCase"

    def test_normalise_key_uppercases_a_case_insensitive_one(self) -> None:
        """Test the namelist, where two spellings of a key have to be the same entry.

        The spelling the caller used is kept only for the file text of a brand-new entry.
        """
        ctx = ParseContext(compile_grammar(WITHOUT_BLOCK), False, {}, VALUE_RULE_PRIORITY)

        assert ctx.normalise_key("MixedCase") == "MIXEDCASE"
        assert ctx.normalise_key("MIXEDCASE") == "MIXEDCASE"

    def test_normalise_key_lowercases_when_the_parser_says_so(self) -> None:
        """Test the field table, whose reader lowercases names, as its dict does."""
        ctx = ParseContext(compile_grammar(WITHOUT_BLOCK), False, {}, VALUE_RULE_PRIORITY, key_case="lower")

        assert ctx.normalise_key("MixedCase") == "mixedcase"
        assert ctx.normalise_key("mixedcase") == "mixedcase"


# A grammar with one of each shape the views have to tell apart: a plain repetition of each
# kind, a rule with a longer expansion, a rule that is a single terminal, and a choice of
# value types. Every rule hangs off ``start``, because Lark drops what it cannot reach.
INFO_GRAMMAR = """
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
    return compile_grammar(INFO_GRAMMAR).info


def test_from_lark_groups_the_rules_and_finds_the_value_types() -> None:
    """Test that a compiled parser is turned into the two views everything else uses."""
    compiled = compile_grammar(INFO_GRAMMAR)
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
    compiled = compile_grammar(INFO_GRAMMAR)
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
