# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Pre-computed views over a compiled grammar.

A ``Lark`` instance exposes its compiled rules and terminals as flat lists, which answer
none of the questions the rest of the package actually asks: what text does this terminal
stand for, is this rule a plain repetition, is this position a value. ``GrammarInfo``
indexes those lists once and answers all three.

The views are held on an instance rather than in a module-level cache so that they die with
the compiled grammar they describe, and they are memoised because the same questions are
asked again on every key addition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lark.grammar import Rule
from lark.lexer import PatternStr, TerminalDef

from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY

if TYPE_CHECKING:
    from lark import Lark

_TERMINAL_SAMPLES: Mapping[str, str] = {
    "NEWLINE": "\n",
    "WS_INLINE": " ",
}
"""Text to emit for regular-expression terminals that carry no text of their own.

Terminals whose pattern is a plain string are reproduced from the pattern, and terminals
holding a key or a value are filled in from the caller's data. That leaves only whitespace
and end-of-line. Any other regular-expression terminal has no entry here, and a derivation
needing one is discarded -- which is how a derivation that would have to invent the text of
a comment, or of a keyword closing a block, is pruned (a Fortran namelist has both: a
comment is free text, and ``&end`` is an alternative to ``/``).

Matched on the terminal name *suffix*, because Lark renames terminals imported from another
grammar: ``WS_INLINE`` becomes ``config__WS_INLINE``.
"""


@dataclass(eq=False)
class GrammarInfo:
    """Pre-computed views over a compiled grammar.

    Compared and hashed by identity, so that a caller wanting to memoise something else per
    grammar can key it on the instance -- the entry-template generator does exactly that.

    Args:
        rules_by_origin (Mapping[str, Sequence[Rule]]): Compiled rules, grouped by the name
            of the rule they define.
        terminals (Mapping[str, TerminalDef]): Terminal definitions, keyed by name.

    Examples:
        >>> from access.config.grammar_compiled import compile_grammar
        >>> from access.config import MOM6InputParser
        >>> info = compile_grammar(MOM6InputParser().grammar).info
        >>> sorted(info.value_rules("value"))
        ['bool', 'float', 'integer', 'string']
        >>> info.value_rules("key_value")
        frozenset()
        >>> info.is_repetition_rule("block")
        True
    """

    rules_by_origin: Mapping[str, Sequence[Rule]]
    terminals: Mapping[str, TerminalDef]
    # Value rules already resolved, keyed by rule name. Held here rather than in a
    # module-level cache so the memo dies with the compiled grammar.
    _value_rules: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_lark(cls, lark: Lark) -> GrammarInfo:
        """Build the views for a compiled Lark instance.

        Args:
            lark (Lark): The compiled parser.

        Returns:
            GrammarInfo: The pre-computed views.
        """
        rules_by_origin: dict[str, list[Rule]] = {}
        for rule in lark.rules:
            rules_by_origin.setdefault(str(rule.origin.name), []).append(rule)
        return cls(rules_by_origin, {terminal.name: terminal for terminal in lark.terminals})

    def sample_terminal(self, name: str) -> str | None:
        """Return the text to emit for a terminal, or ``None`` if it cannot be reproduced.

        Args:
            name (str): Terminal name, possibly renamed by an ``%import``. A compiled
                grammar defines every terminal its rules refer to, so the name is known.

        Returns:
            str | None: The text, or ``None`` when the terminal is a regular expression with
                no sample text available.
        """
        terminal = self.terminals[name]
        if isinstance(terminal.pattern, PatternStr):
            return terminal.pattern.value
        for suffix, text in _TERMINAL_SAMPLES.items():
            if name == suffix or name.endswith(f"__{suffix}"):
                return text
        return None

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

    def value_rules(self, name: str) -> frozenset[str]:
        """Return the value-type rules reachable from *name* by single-symbol rules only.

        This is what identifies a placeholder position for a value. A rule that is itself a
        value type (``"integer"``) resolves to itself; a rule that is a plain choice between
        value types (``?value: logical | integer | ...``) resolves to the union of its
        alternatives; anything with a longer expansion is not a value position and resolves
        to the empty set.

        Args:
            name (str): Rule name.

        Returns:
            frozenset[str]: Names of the admitted value-type rules, empty if *name* is not a
                value position.
        """
        cached = self._value_rules.get(name)
        if cached is None:
            cached = self._compute_value_rules(name, set())
            self._value_rules[name] = cached
        return cached

    def _compute_value_rules(self, name: str, visited: set[str]) -> frozenset[str]:
        if name in VALUE_TYPE_HANDLER_REGISTRY:
            return frozenset({name})
        rules = self.rules_by_origin.get(name)
        if not rules or name in visited:
            return frozenset()
        visited.add(name)
        found: set[str] = set()
        for rule in rules:
            if len(rule.expansion) != 1 or rule.expansion[0].is_term:
                return frozenset()
            reachable = self._compute_value_rules(str(rule.expansion[0].name), visited)
            if not reachable:
                return frozenset()
            found |= reachable
        return frozenset(found)
