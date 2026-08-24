# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the synthesis of the text of a new configuration entry.

These exercise the module directly, against the three shipped grammars, so that a change in
how a candidate is generated, ranked or rendered is caught here rather than only through the
end-to-end tests in ``test_config.py`` and the per-format modules.
"""

from pathlib import Path

import pytest
from lark import Lark, Tree

from access.config.entry_synthesis import (
    EntryStyle,
    iter_entry_snippets,
    probe_entry_style,
    probe_sibling_block_style,
    render,
)
from access.config.entry_templates import (
    Fragment,
    admitted_value_rules,
    compile_template,
    expand_value_slots,
    grammar_templates,
)
from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_contract import ENTRY_CATEGORIES
from access.config.grammar_info import GrammarInfo
from access.config.mom6_input import MOM6InputParser
from access.config.nuopc_config import NUOPCParser

PARSERS = {
    "fortran_nml": FortranNMLParser,
    "mom6_input": MOM6InputParser,
    "nuopc_config": NUOPCParser,
}


def _compile(parser_class: type) -> Lark:
    """Build a multi-start parser for one of the shipped grammars."""
    return Lark(
        parser_class().grammar,
        import_paths=[Path("src/access/config")],
        maybe_placeholders=False,
        start=["start", "block"],
    )


@pytest.fixture(scope="module")
def larks() -> dict[str, Lark]:
    """Fixture returning a compiled parser per shipped grammar."""
    return {name: _compile(cls) for name, cls in PARSERS.items()}


@pytest.fixture(scope="module")
def infos(larks: dict[str, Lark]) -> dict[str, GrammarInfo]:
    """Fixture returning the grammar views per shipped grammar."""
    return {name: GrammarInfo.from_lark(lark) for name, lark in larks.items()}


# The entry text each grammar produces for each container and category it supports, with
# no neighbouring entry to copy a style from. Every combination absent from this table is
# one the grammar cannot express, and is asserted empty by test_unsupported_combinations.
CANONICAL = {
    ("nuopc_config", "start", "key_value"): "NEWKEY: 7\n",
    ("nuopc_config", "start", "key_list"): "NEWKEY: 1 2 3\n",
    ("nuopc_config", "start", "key_block"): "NEWKEY::\n::\n",
    ("nuopc_config", "block", "key_value"): "NEWKEY = 7\n",
    ("nuopc_config", "block", "key_list"): "NEWKEY = 1:2:3\n",
    ("mom6_input", "start", "key_value"): "NEWKEY = 7\n",
    ("mom6_input", "start", "key_list"): "NEWKEY = 1, 2, 3\n",
    ("mom6_input", "start", "key_block"): "NEWKEY%\n%NEWKEY\n",
    ("mom6_input", "block", "key_value"): "NEWKEY = 7\n",
    ("mom6_input", "block", "key_list"): "NEWKEY = 1, 2, 3\n",
    ("fortran_nml", "start", "key_block"): "&NEWKEY\n/\n",
    ("fortran_nml", "block", "key_value"): "NEWKEY = 7\n",
    ("fortran_nml", "block", "key_list"): "NEWKEY = 1, 2, 3\n",
    ("fortran_nml", "block", "key_null"): "NEWKEY =\n",
}

VALUES = {"key_value": 7, "key_list": [1, 2, 3], "key_null": None, "key_block": {}}


@pytest.mark.parametrize(("case", "expected"), CANONICAL.items(), ids=lambda item: item)
def test_canonical_snippet(infos, larks, case, expected) -> None:
    """Test the entry text produced for each supported container and category.

    The first candidate has to be both the expected text and valid in that container, so
    these assertions pin the whole generate-rank-render pipeline against the real grammars.
    """
    grammar, container, category = case
    snippets = iter_entry_snippets(infos[grammar], container, category, "NEWKEY", VALUES[category], EntryStyle())

    first = next(iter(snippets))
    assert first == expected

    # The text must parse in the container it was generated for, and yield that category.
    parsed = larks[grammar].parse(first, start=container)
    categories = {node.data for node in parsed.iter_subtrees()} | {parsed.data}
    assert category in categories


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


def test_value_type_rendering(infos) -> None:
    """Test the notation each grammar uses for a value written into a new entry.

    These pin ``VALUE_RULE_PRIORITY`` as it applies to the real formats: a namelist writes
    Fortran logicals and has no bare-word type, while MOM6 writes Python-style booleans.
    """
    fortran = infos["fortran_nml"]
    for value, expected in [
        (True, "K = .true.\n"),
        (7, "K = 7\n"),
        (1.5, "K = 1.5\n"),
        (3.0 + 4.0j, "K = (3.0, 4.0)\n"),
        ("text", 'K = "text"\n'),
    ]:
        assert next(iter(iter_entry_snippets(fortran, "block", "key_value", "K", value, EntryStyle()))) == expected

    mom6 = infos["mom6_input"]
    assert next(iter(iter_entry_snippets(mom6, "start", "key_value", "K", True, EntryStyle()))) == "K = True\n"

    nuopc = infos["nuopc_config"]
    assert next(iter(iter_entry_snippets(nuopc, "start", "key_value", "K", True, EntryStyle()))) == "K: .true.\n"
    assert next(iter(iter_entry_snippets(nuopc, "start", "key_value", "K", "datm", EntryStyle()))) == "K: datm\n"


def test_no_candidate_for_unrepresentable_value(infos) -> None:
    """Test that a value no admitted rule accepts produces no candidate at all."""
    # NUOPC has neither a quoted string nor any other type accepting this value.
    assert list(iter_entry_snippets(infos["nuopc_config"], "start", "key_value", "K", "a string", EntryStyle())) == []
    # MOM6 has no complex type.
    assert list(iter_entry_snippets(infos["mom6_input"], "start", "key_value", "K", 1 + 2j, EntryStyle())) == []


def test_value_rule_fallback(infos) -> None:
    """Test that every accepting value type is offered, not only the most preferred one.

    A grammar admitting both a quoted string and a bare word offers both renderings, so a
    caller whose preferred one reads back as the wrong type can fall through to the next.
    """
    snippets = list(iter_entry_snippets(infos["nuopc_config"], "start", "key_value", "K", "datm", EntryStyle()))
    assert "K: datm\n" in snippets


def test_style_is_copied_from_neighbours(infos) -> None:
    """Test that a neighbour's indentation and spacing are reproduced."""
    info = infos["fortran_nml"]
    style = EntryStyle(indent="  ", key_pads=(" ", " "), element_pads=(" ",), has_donor=True)

    assert next(iter(iter_entry_snippets(info, "block", "key_value", "NEW", 42, style))) == "  NEW = 42\n"
    assert next(iter(iter_entry_snippets(info, "block", "key_list", "NEW", [1, 2, 3], style))) == "  NEW = 1, 2, 3\n"
    assert next(iter(iter_entry_snippets(info, "block", "key_null", "NEW", None, style))) == "  NEW =\n"


def test_style_without_padding(infos) -> None:
    """Test that a neighbour written without spaces is matched without spaces."""
    info = infos["mom6_input"]
    tight = EntryStyle(indent="", key_pads=(), element_pads=(), has_donor=True)

    assert next(iter(iter_entry_snippets(info, "start", "key_value", "NEW", 1, tight))) == "NEW=1\n"
    assert next(iter(iter_entry_snippets(info, "start", "key_list", "NEW", [1, 2], tight))) == "NEW=1,2\n"


def test_style_indent_ignored_when_unsupported(infos) -> None:
    """Test that indentation is dropped where the grammar does not allow it.

    A MOM6 assignment cannot be indented, so an indented neighbour must not produce an entry
    that fails to parse.
    """
    info = infos["mom6_input"]
    style = EntryStyle(indent="    ", key_pads=(" ", " "), element_pads=None, has_donor=True)

    assert next(iter(iter_entry_snippets(info, "start", "key_value", "NEW", 1, style))) == "NEW = 1\n"


def test_probe_entry_style(larks) -> None:
    """Test the whitespace read back from the entries of a real container."""
    tree = larks["fortran_nml"].parse("&L\n  Var = 4\n  LIST = 1, 2, 3\n/\n", start="start")
    block = next(node for node in tree.iter_subtrees() if str(node.data) == "block")

    style = probe_entry_style(block)
    assert style.has_donor
    assert style.indent == "  "
    assert style.key_pads == (" ", " ")
    assert style.element_pads == (" ",)


def test_probe_entry_style_separates_the_two_kinds(larks) -> None:
    """Test that a scalar neighbour does not dictate how a list is spaced.

    The last entry here is a scalar written without spaces around its separator, and the
    only list is spaced. Element spacing has to come from the list, and assignment spacing
    from the scalar, or a new entry of either kind comes out looking like the other.
    """
    tree = larks["mom6_input"].parse("BLIST = 1, 2, 3\nVAR=4\n", start="start")

    style = probe_entry_style(tree)
    assert style.key_pads == ()
    assert style.element_pads == (" ",)


def test_probe_entry_style_without_entries(larks) -> None:
    """Test that a container holding no entries reports no style to copy."""
    tree = larks["mom6_input"].parse("B%\n%B\n", start="start")
    block = next(node for node in tree.iter_subtrees() if str(node.data) == "block")

    assert probe_entry_style(block) == EntryStyle()


def test_probe_entry_style_without_values(larks) -> None:
    """Test the style read from an entry that has a key but no value."""
    tree = larks["fortran_nml"].parse("&L\n  NOVALUE =\n/\n", start="start")
    block = next(node for node in tree.iter_subtrees() if str(node.data) == "block")

    style = probe_entry_style(block)
    assert style.has_donor
    assert style.indent == "  "
    # With no value, everything after the key spaces out the key, and nothing says how
    # elements are separated.
    assert style.element_pads is None


def test_probe_sibling_block_style(larks) -> None:
    """Test the style a new block takes from the blocks already in its container."""
    tree = larks["fortran_nml"].parse("&L\n  Var = 4\n/\n&EMPTY\n/\n", start="start")

    style = probe_sibling_block_style(tree)
    assert style.has_donor
    assert style.indent == "  "
    assert style.key_pads == (" ", " ")


def test_probe_sibling_block_style_takes_the_nearest(larks) -> None:
    """Test that the last block able to donate wins, as the last entry does for a style.

    A new block is written at the end of its container, so the block just above it is its
    closest neighbour. An empty one in between says nothing and is passed over.
    """
    tree = larks["fortran_nml"].parse("&A\n    Far = 1\n/\n&B\n  Near = 2\n/\n&C\n/\n", start="start")

    assert probe_sibling_block_style(tree).indent == "  "


def test_probe_sibling_block_style_without_blocks(larks) -> None:
    """Test that a container with no block to copy from reports no style.

    Both cases have to report it: a container holding no block at all, and one whose only
    block is empty -- which is what a block just created looks like.
    """
    entries_only = larks["mom6_input"].parse("A = 1\n", start="start")
    empty_block = larks["mom6_input"].parse("B%\n%B\n", start="start")

    assert probe_sibling_block_style(entries_only) == EntryStyle()
    assert probe_sibling_block_style(empty_block) == EntryStyle()


def test_expand_value_slots() -> None:
    """Test that a two-value template grows by repeating its element separator."""
    template = (
        Fragment("key"),
        Fragment("literal", "="),
        Fragment("value"),
        Fragment("literal", ","),
        Fragment("ws"),
        Fragment("value"),
    )

    grown = expand_value_slots(template, 4)
    assert sum(1 for fragment in grown if fragment.kind == "value") == 4
    assert render(grown, "K", ["1", "2", "3", "4"], EntryStyle()) == "K=1, 2, 3, 4"


def test_compile_template() -> None:
    """Test the compilation of a parser-declared template."""
    template = compile_template("{key} = {value}\n", "key_value", frozenset({"integer"}))
    assert render(template, "K", ["1"], EntryStyle()) == "K = 1\n"

    block = compile_template("{key}%\n{block}%{key}\n", "key_block", frozenset())
    assert render(block, "K", [], EntryStyle()) == "K%\n%K\n"


def test_compile_template_validation() -> None:
    """Test that a template whose markers do not fit its category is rejected."""
    # Wrong number of values for the category.
    with pytest.raises(ValueError):
        compile_template("{key}={value}", "key_null", frozenset())
    with pytest.raises(ValueError):
        compile_template("{key}={value}", "key_list", frozenset())
    # A block template needs exactly one block marker.
    with pytest.raises(ValueError):
        compile_template("{key}=", "key_block", frozenset())
    # Every template needs a key.
    with pytest.raises(ValueError):
        compile_template("={value}", "key_value", frozenset())


def test_snippet_cap_is_enforced(infos, monkeypatch) -> None:
    """Test that the number of candidates offered for one insertion is bounded."""
    from access.config import entry_synthesis

    monkeypatch.setattr(entry_synthesis, "_MAX_SNIPPETS", 2)

    snippets = list(iter_entry_snippets(infos["mom6_input"], "start", "key_value", "K", 1, EntryStyle()))
    assert len(snippets) == 2


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


def test_ranking_prefers_a_complete_line(infos) -> None:
    """Test that the chosen entry ends the line it is written on.

    A Fortran assignment can be written with no line terminator, since the terminator sits
    on the enclosing line rule. An entry rendered that way would run into what follows it.
    """
    info = infos["fortran_nml"]
    first = next(iter(iter_entry_snippets(info, "block", "key_value", "K", 1, EntryStyle())))
    assert first.endswith("\n")


def test_ranking_avoids_a_trailing_separator(infos) -> None:
    """Test that an optional trailing separator is not added.

    Fortran allows a comma after the last value on a line, and that form has exactly the
    whitespace runs a spaced neighbour wants. Style must not be allowed to buy it.
    """
    style = EntryStyle(indent="  ", key_pads=(" ", " "), element_pads=(" ",), has_donor=True)
    first = next(iter(iter_entry_snippets(infos["fortran_nml"], "block", "key_list", "K", [1, 2], style)))

    assert first == "  K = 1, 2\n"
    assert not first.rstrip("\n").endswith(",")


def test_ranking_avoids_a_line_continuation(infos) -> None:
    """Test that a list is written on one line.

    Fortran can separate two values with a comma and a newline, which is a legal list but
    reads as a continuation. The entry should stay on the line it started on.
    """
    first = next(iter(iter_entry_snippets(infos["fortran_nml"], "block", "key_list", "K", [1, 2, 3], EntryStyle())))
    assert first.count("\n") == 1


def test_ranking_puts_a_new_block_body_on_its_own_line(infos) -> None:
    """Test that a new namelist group leaves its body room.

    ``&NAME/`` is a legal empty namelist, but a group written that way has nowhere to put
    the keys that are about to be added to it.
    """
    first = next(iter(iter_entry_snippets(infos["fortran_nml"], "start", "key_block", "K", {}, EntryStyle())))
    assert first == "&K\n/\n"


def test_render_ignores_a_block_placeholder() -> None:
    """Test that the contents of a new block render as nothing."""
    template = (Fragment("key"), Fragment("literal", "<"), Fragment("block"), Fragment("literal", ">"))
    assert render(template, "K", [], EntryStyle()) == "K<>"


def test_entry_nodes_do_not_cross_into_a_nested_block(larks) -> None:
    """Test that the entries of a container exclude those of a block inside it."""
    tree = larks["mom6_input"].parse("A = 1\nB%\nX = 1\n%B\n", start="start")

    style = probe_entry_style(tree)
    assert style.has_donor
    # The block's own entry must not be visible to the outer container's insertion index.
    block = next(node for node in tree.iter_subtrees() if str(node.data) == "block")
    assert isinstance(block, Tree)
