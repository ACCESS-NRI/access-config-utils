# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Pre-computed views over a compiled grammar.

A ``Lark`` instance exposes its compiled rules as a flat list, which answers none of the
questions the rest of the package actually asks: is this rule a plain repetition, which
value types does the format have. ``GrammarInfo`` indexes the list once and answers both.

The views are held on an instance rather than in a module-level cache so that they die with
the compiled grammar they describe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lark.grammar import Rule

from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY

if TYPE_CHECKING:
    from lark import Lark


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
