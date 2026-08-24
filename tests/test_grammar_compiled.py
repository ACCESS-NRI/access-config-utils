# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for turning grammar text into the artefacts everything else works with.

Nothing is faked. The module's whole job is to build a ``Lark`` and a ``Reconstructor``, so
standing in for either would leave nothing to test -- and the one branch worth pinning, the
probe deciding whether ``block`` can be a second start symbol, is only observable in what
Lark accepts afterwards.

The grammars are real, and as short as the question allows.
"""

import pytest
from lark import Lark
from lark.exceptions import ConfigurationError
from lark.reconstruct import Reconstructor

from access.config.grammar_compiled import (
    GRAMMAR_DIR,
    CompiledGrammar,
    ParseContext,
    clear_grammar_cache,
    compile_grammar,
)
from access.config.grammar_info import GrammarInfo
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
    clear_grammar_cache()
    yield
    clear_grammar_cache()


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

    def test_clearing_the_cache_forces_a_recompile(self) -> None:
        """Test the escape hatch a test needs when it changes a grammar between parses."""
        first = compile_grammar(WITHOUT_BLOCK)
        clear_grammar_cache()

        assert compile_grammar(WITHOUT_BLOCK) is not first


class TestStartSymbols:
    """The probe deciding whether a grammar gets ``block`` as a second start symbol."""

    def test_a_grammar_with_a_block_rule_can_parse_one(self) -> None:
        """Test that an entry can be parsed in the container it will live in.

        This is what makes adding a key to a block possible at all: the synthesised text is
        handed back to Lark with ``block`` as the start symbol.
        """
        lark = compile_grammar(WITH_BLOCK).lark

        assert lark.parse("a=1", start="block").data == "block"

    def test_a_grammar_without_one_keeps_a_single_start_symbol(self) -> None:
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
