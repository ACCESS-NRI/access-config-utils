# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Turning grammar text into the artefacts the rest of the package works with.

Compiling a grammar costs far more than parsing a file with it, and the same artefacts are
needed again on every key addition, so ``compile_grammar`` builds them once and caches them:
the ``Lark`` parser, the ``Reconstructor`` that writes a tree back out, and the
``GrammarInfo`` views.

``ParseContext`` pairs those artefacts with the settings the parser class declares -- key
case-sensitivity, hand-written entry templates, value-rule priority. It is what a parsed
configuration carries around so that it can still edit itself long after ``parse`` returned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lark import Lark
from lark.reconstruct import Reconstructor

from access.config.grammar_contract import BLOCK_RULE, KEY_LIST, KEY_VALUE, START_RULE
from access.config.grammar_info import GrammarInfo

OPTIONAL_START_RULES: tuple[str, ...] = (BLOCK_RULE, KEY_VALUE, KEY_LIST)
"""Rules made start symbols as well, when the grammar defines them under those names.

Text is handed back to Lark rather than turned into nodes by hand wherever nodes are needed,
and the start symbol says what the text has to be valid as: a block for a new entry, an
entry for the values written into one. A rule reached only through an alias is not included,
since Lark resolves start symbols against rule names.
"""

GRAMMAR_DIR: Path = Path(__file__).parent
"""Directory holding ``config.lark``, the shared grammar every format grammar imports from.

Exported so that a test or a doctest building a parser by hand resolves an ``%import``
the same way the package does, rather than hardcoding the path.
"""


@dataclass(frozen=True)
class CompiledGrammar:
    """A compiled grammar and the artefacts derived from it.

    Args:
        lark (Lark): The compiled parser.
        reconstructor (Reconstructor): Writes a parse tree back out as text.
        info (GrammarInfo): Views over the grammar, used to synthesise new entries.
    """

    lark: Lark
    reconstructor: Reconstructor
    info: GrammarInfo


@dataclass(frozen=True)
class ParseContext:
    """Everything a parsed configuration needs in order to edit itself, beyond its tree.

    Args:
        grammar (CompiledGrammar): The compiled grammar and its derived artefacts.
        case_sensitive_keys (bool): Are the dict keys case-sensitive?
        entry_templates (Mapping[tuple[str, str], str]): Per-parser overrides for the
            text of a new entry, keyed by container rule name and entry category. Used
            only where no neighbouring entry can supply a style.
        value_rule_priority (tuple[str, ...]): Order in which value-type rules are tried for
            a new value.
        block_rules (tuple[str, ...]): Names of the rules holding block contents, the one
            new entries are written into first.
        repeated_blocks (str): What a file writing the same block twice means, ``"merge"``
            or ``"separate"``. Applies only to the first of *block_rules*; see
            ``ConfigParser.repeated_blocks``.
    """

    grammar: CompiledGrammar
    case_sensitive_keys: bool
    entry_templates: Mapping[tuple[str, str], str]
    value_rule_priority: tuple[str, ...]
    block_rules: tuple[str, ...] = (BLOCK_RULE,)
    repeated_blocks: str = "merge"

    @property
    def lark(self) -> Lark:
        """Lark: The compiled parser."""
        return self.grammar.lark

    @property
    def reconstructor(self) -> Reconstructor:
        """Reconstructor: The reconstructor for this grammar."""
        return self.grammar.reconstructor

    @property
    def info(self) -> GrammarInfo:
        """GrammarInfo: Grammar views used to synthesise new entries."""
        return self.grammar.info

    def normalise_key(self, key: str) -> str:
        """Return the form of *key* used as the dict key.

        A case-insensitive format stores its keys uppercased, so that two spellings of one
        key are the same entry. The spelling the caller used is kept only for the file text
        of a brand-new entry.

        Args:
            key (str): The key as written.

        Returns:
            str: The normalised key.
        """
        return key if self.case_sensitive_keys else key.upper()


_GRAMMAR_CACHE: dict[str, CompiledGrammar] = {}
"""Compiled grammars, keyed by grammar text.

Keyed on the text rather than on the parser class, because ``ConfigParser.grammar`` is a
property and so could differ between instances of one class, and because parsers sharing a
grammar should share the compiled result. Concurrent first use may compile twice and discard
one result, which is wasteful but harmless.
"""


def clear_grammar_cache() -> None:
    """Discard every cached compiled grammar.

    Only needed by tests that change a grammar between parses.
    """
    _GRAMMAR_CACHE.clear()


def compile_grammar(grammar: str) -> CompiledGrammar:
    """Compile a grammar, or return the cached result for it.

    The parser is built with ``"block"`` as an additional start symbol, so the text of an
    entry can be parsed in the context it will live in. That is only possible when the
    grammar has such a rule, so its presence is probed first; a format without blocks
    keeps a single start symbol.

    Because the result has more than one start symbol, **every** ``parse()`` call on it must
    name one explicitly, or Lark raises ``ConfigurationError``.

    Args:
        grammar (str): The grammar text.

    Returns:
        CompiledGrammar: The compiled parser and the artefacts derived from it.
    """
    compiled = _GRAMMAR_CACHE.get(grammar)
    if compiled is None:
        lark = _build(grammar, [START_RULE])
        defined = {str(rule.origin.name) for rule in lark.rules}
        extra = [name for name in OPTIONAL_START_RULES if name in defined]
        if extra:
            lark = _build(grammar, [START_RULE, *extra])
        compiled = CompiledGrammar(lark, Reconstructor(lark), GrammarInfo.from_lark(lark))
        _GRAMMAR_CACHE[grammar] = compiled
    return compiled


def _build(grammar: str, start: list[str]) -> Lark:
    """Build a Lark parser for a grammar with the given start symbols.

    Args:
        grammar (str): The grammar text.
        start (list[str]): Names of the rules usable as start symbols.

    Returns:
        Lark: The compiled parser.
    """
    return Lark(grammar, import_paths=[GRAMMAR_DIR], maybe_placeholders=False, start=start)
