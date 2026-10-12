# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Turning grammar text into the artefacts the rest of the package works with.

Compiling a grammar costs far more than parsing a file with it, and the same artefacts are
needed again on every key addition, so ``compile_grammar`` builds them once and caches them:
the ``Lark`` parser, the ``Reconstructor`` that writes a tree back out, and the
``GrammarInfo`` views. A ``Lark`` instance exposes its compiled rules as a flat list, which
answers none of the questions the rest of the package actually asks -- is this rule a plain
repetition, which value types does the format have -- so ``GrammarInfo`` indexes the list
once and answers both. It is held on an instance rather than in a module-level cache so
that it dies with the compiled grammar it describes.

``ParseContext`` pairs those artefacts with the settings the parser class declares -- key
case-sensitivity, hand-written entry templates, value-rule priority. It is what a parsed
configuration carries around so that it can still edit itself long after ``parse`` returned.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lark import Lark, Token
from lark.exceptions import ConfigurationError, UnexpectedInput
from lark.grammar import Rule
from lark.reconstruct import Reconstructor

from access.config.grammar_contract import BLOCK_RULE, KEY_RULE, START_RULE
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY

OPTIONAL_START_RULES: tuple[str, ...] = (BLOCK_RULE, KEY_RULE)
"""Rules made start symbols as well, when the grammar defines them under those names.

Text is handed back to Lark rather than turned into nodes by hand wherever nodes are needed,
and the start symbol says what the text has to be valid as: a block for a new entry, a key
for the name a new entry is written under. A rule reached only through an alias is not
included, since Lark resolves start symbols against rule names.
"""

GRAMMAR_DIR: Path = Path(__file__).parent
"""Directory holding ``config.lark``, the shared grammar every format grammar imports from.

Exported so that a test or a doctest building a parser by hand resolves an ``%import``
the same way the package does, rather than hardcoding the path.
"""


@dataclass(eq=False)
class GrammarInfo:
    """Pre-computed views over a compiled grammar.

    Compared and hashed by identity, so that a caller wanting to memoise something else per
    grammar can key it on the instance.

    Args:
        rules_by_origin (Mapping[str, Sequence[Rule]]): Compiled rules, grouped by the name
            of the rule they define.
        value_rules (frozenset[str]): The value-type rules the grammar has: the names in
            ``VALUE_TYPE_HANDLER_REGISTRY`` that some rule uses.

    ``is_repetition_rule`` says whether a rule is a plain repetition, which is what tells a
    container that may be empty from one that may not.
    """

    rules_by_origin: Mapping[str, Sequence[Rule]]
    value_rules: frozenset[str]

    @classmethod
    def from_lark(cls, lark: Lark) -> GrammarInfo:
        """Build the views for a compiled Lark instance.

        Args:
            lark (Lark): The compiled parser.

        Returns:
            GrammarInfo: The pre-computed views.
        """
        rules_by_origin: dict[str, list[Rule]] = {}
        used: set[str] = set()
        for rule in lark.rules:
            rules_by_origin.setdefault(str(rule.origin.name), []).append(rule)
            used.update(str(symbol.name) for symbol in rule.expansion)
        return cls(rules_by_origin, frozenset(used & VALUE_TYPE_HANDLER_REGISTRY.keys()))

    def is_repetition_rule(self, name: str) -> bool:
        """Report whether a rule is a plain repetition of something, such as ``a: b*``.

        Two adjacent nodes of such a rule can be merged into one holding both sets of
        children, because the rule accepts any number of them. That is what lets the text
        either side of a deleted entry be rejoined -- see ``merge_adjacent_repetitions``.

        Recognised by shape: every alternative is either empty or the single repetition
        helper rule Lark generates for this rule, as ``random_text: (/.+/|NEWLINE)*``
        compiles to ``random_text: __random_text_star_5 |``.

        Args:
            name (str): Rule name.

        Returns:
            bool: True if the rule is a plain repetition.
        """
        rules = self.rules_by_origin.get(name)
        if not rules:
            return False
        prefixes = (f"__{name}_star_", f"__{name}_plus_")
        for rule in rules:
            if not rule.expansion:
                continue
            if len(rule.expansion) != 1 or rule.expansion[0].is_term:
                return False
            if not str(rule.expansion[0].name).startswith(prefixes):
                return False
        return True


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
        entry_templates (Mapping[tuple[str, str], str]): The text of a new entry, keyed by
            container rule name and entry category. Used where the file has no entry of the
            kind to copy.
        value_rule_priority (tuple[str, ...]): Order in which value-type rules are tried for
            a new value.
        block_rules (tuple[str, ...]): Names of the rules holding block contents, the one
            new entries are written into first.
        repeated_blocks (str): What a file writing the same block twice means, ``"merge"``
            or ``"separate"``. Applies only to the first of *block_rules*; see
            ``ConfigParser.repeated_blocks``.
        key_case (str): The case keys are stored in when they are not case-sensitive,
            ``"upper"`` or ``"lower"``; see ``ConfigParser.key_case``.
    """

    grammar: CompiledGrammar
    case_sensitive_keys: bool
    entry_templates: Mapping[tuple[str, str], str]
    value_rule_priority: tuple[str, ...]
    block_rules: tuple[str, ...] = (BLOCK_RULE,)
    repeated_blocks: str = "merge"
    key_case: str = "upper"

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

    def accepts_key(self, text: str) -> bool:
        """Report whether *text* is a name this format can write a key under.

        Asked of the grammar's own ``key`` rule, so a format whose names may hold a hyphen
        or a slash accepts them, and one restricted to identifiers refuses everything else.
        The name has to be the whole text, as a single token: a grammar ignoring whitespace
        would otherwise accept ``" a"`` and write ``a``.

        Args:
            text (str): The name as the caller wrote it.

        Returns:
            bool: True if *text* reads back as exactly that key.
        """
        try:
            tree = self.lark.parse(text, start=KEY_RULE)
        except (UnexpectedInput, ConfigurationError):
            return False
        children = tree.children
        return len(children) == 1 and isinstance(children[0], Token) and str(children[0]) == text

    def normalise_key(self, key: str) -> str:
        """Return the form of *key* used as the dict key.

        A case-insensitive format stores its keys in one case, upper unless the parser says
        lower, so that two spellings of one key are the same entry. The spelling the caller
        used is kept only for the file text of a brand-new entry.

        Args:
            key (str): The key as written.

        Returns:
            str: The normalised key.
        """
        if self.case_sensitive_keys:
            return key
        return key.lower() if self.key_case == "lower" else key.upper()


@functools.cache
def compile_grammar(grammar: str) -> CompiledGrammar:
    """Compile a grammar, or return the result already compiled for the same text.

    The result is cached on the text rather than on the parser class, because
    ``ConfigParser.grammar`` is a property and so could differ between instances of one
    class, and because parsers sharing a grammar should share the compiled result.
    Concurrent first use may compile twice and discard one result, which is wasteful but
    harmless. ``compile_grammar.cache_clear()`` forgets every result.

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
    lark = _build(grammar, [START_RULE])
    defined = {str(rule.origin.name) for rule in lark.rules}
    extra = [name for name in OPTIONAL_START_RULES if name in defined]
    if extra:
        lark = _build(grammar, [START_RULE, *extra])
    return CompiledGrammar(lark, Reconstructor(lark), GrammarInfo.from_lark(lark))


def _build(grammar: str, start: list[str]) -> Lark:
    """Build a Lark parser for a grammar with the given start symbols.

    Args:
        grammar (str): The grammar text.
        start (list[str]): Names of the rules usable as start symbols.

    Returns:
        Lark: The compiled parser.
    """
    return Lark(grammar, import_paths=[GRAMMAR_DIR], maybe_placeholders=False, start=start)
