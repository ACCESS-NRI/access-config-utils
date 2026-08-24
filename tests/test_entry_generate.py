# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deriving entry templates from a grammar.

Nothing is faked. The walk reads real ``Rule`` and ``Symbol`` objects, and the shapes that
matter -- the helper rule behind a repetition most of all -- are ones Lark invents, so
standing in for a ``GrammarInfo`` would mean asserting against a guess at what it produces.

The three shipped grammars answer most of the questions. Where none of them has the shape
under test, a short grammar written for it does.
"""

from conftest import CANONICAL, PARSERS

from access.config.entry_generate import admitted_value_rules, grammar_templates
from access.config.grammar_compiled import compile_grammar
from access.config.grammar_contract import ENTRY_CATEGORIES


def test_unsupported_combinations(infos) -> None:
    """Test that a container and category a grammar cannot express yields no template."""
    for grammar in PARSERS:
        for container in ("start", "block"):
            for category in ENTRY_CATEGORIES:
                templates = grammar_templates(infos[grammar], container, category)
                if (grammar, container, category) in CANONICAL:
                    assert templates, (grammar, container, category)
                else:
                    assert not templates, (grammar, container, category)


def test_template_counts_are_bounded(infos) -> None:
    """Test that generation stays small on every real grammar and container."""
    for grammar in PARSERS:
        for container in ("start", "block"):
            for category in ENTRY_CATEGORIES:
                assert len(grammar_templates(infos[grammar], container, category)) <= 64


def test_templates_are_cached(infos) -> None:
    """Test that the templates for a container and category are computed once."""
    info = infos["mom6_input"]
    first = grammar_templates(info, "start", "key_value")
    assert grammar_templates(info, "start", "key_value") is first


def test_value_rules(infos) -> None:
    """Test the detection of the grammar positions that hold a value."""
    info = infos["mom6_input"]

    # A rule that is itself a value type resolves to itself.
    assert info.value_rules("integer") == frozenset({"integer"})
    # A plain choice between value types resolves to the union of its alternatives.
    assert info.value_rules("value") == frozenset({"bool", "integer", "float", "string"})
    # A rule with a longer expansion is not a value position, and neither is a missing name.
    assert info.value_rules("key_value") == frozenset()
    assert info.value_rules("not_a_rule") == frozenset()


def test_admitted_value_rules(infos) -> None:
    """Test the value types reported as usable in a given container and category."""
    assert admitted_value_rules(infos["mom6_input"], "start", "key_value") == frozenset(
        {"bool", "integer", "float", "string"}
    )
    # NUOPC has no quoted-string type, which is why a string with a space cannot be written.
    nuopc = admitted_value_rules(infos["nuopc_config"], "start", "key_value")
    assert "string" not in nuopc
    assert {"identifier", "path"} <= nuopc
    # Nothing is admitted where the entry cannot be expressed at all.
    assert admitted_value_rules(infos["fortran_nml"], "start", "key_value") == frozenset()


def test_generation_ignores_unreproducible_terminals(infos) -> None:
    """Test that no candidate invents text for a regular-expression terminal.

    Comments and Fortran's ``&end`` namelist terminator are matched by patterns with no
    sample text, so a derivation needing one is discarded. If that pruning ever stopped
    working, generated entries would carry invented comment text.
    """
    for grammar, container, category in CANONICAL:
        for template in grammar_templates(infos[grammar], container, category):
            literals = "".join(fragment.text for fragment in template if fragment.kind == "literal")
            assert "!" not in literals
            assert "#" not in literals
            assert "&end" not in literals


def test_a_derivation_with_the_wrong_number_of_values_is_discarded() -> None:
    """Test that a rule aliased to the wrong category yields no template.

    ``mistyped`` says ``key_value`` but admits a second value, so its derivation carries two
    value placeholders where a scalar entry needs one. Writing it would produce an entry the
    reader then refuses, so it is dropped at generation instead.

    No shipped grammar has this shape, because it is a mistake; the check exists so that a
    grammar making it fails visibly rather than emitting entries nothing can read back.
    """
    info = compile_grammar("""
        start: line*
        ?line: mistyped | plain
        mistyped: key "=" value (":" value)+ NEWLINE -> key_value
        plain: key "=" value NEWLINE -> key_value

        %import config.key
        %import config.integer
        %import config.NEWLINE
        ?value: integer
    """).info

    templates = grammar_templates(info, "start", "key_value")

    # Only the single-value form survives, and every survivor holds exactly one value.
    assert templates
    assert all(sum(1 for fragment in t if fragment.kind == "value") == 1 for t in templates)
