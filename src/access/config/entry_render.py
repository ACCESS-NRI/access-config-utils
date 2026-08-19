# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Choosing how to write a new entry, and writing it.

Adding a key to a parsed configuration file means producing parse tree nodes that the parser
never built. Rather than assembling ``Tree`` and ``Token`` objects by hand -- which means
replicating Lark's rule inlining, alias handling, filtered-out anonymous terminals and
``%import`` terminal renaming -- synthesis produces the **text** of the new entry. The
caller hands that text back to Lark with the container rule as the start symbol, so Lark
builds nodes that are by construction the ones it would have produced for a file containing
the entry.

The text is derived from the compiled grammar itself, so a new format gets entry synthesis
for free.

The pipeline
------------

1. **Generate.** Collect every way the grammar can derive one entry of the requested
   category, as an ``EntryTemplate``. That is ``entry_generate``, which also filters out any
   derivation not holding exactly one entry of the right category, or the wrong number of
   values for it.
2. **Rank.** Order the survivors so the most idiomatic rendering wins: a complete line, a
   block whose contents start on their own line, no blank lines, no trailing separator, and
   the whitespace of a neighbouring entry where that costs nothing.
3. **Render.** Substitute the key, the value and the whitespace, then let the caller
   validate the result by parsing it.

Because step 3 is validated by parsing, ranking only decides which *valid* rendering is
used. It can change how the output looks, never whether it is correct.

Examples:
    Synthesis is reached through the ordinary mapping interface; a new key is written in the
    format's own syntax, spaced like the entries around it.

    >>> from access.config import MOM6InputParser
    >>> config = MOM6InputParser().parse("A = 1\\n")
    >>> config["B"] = [2, 3]
    >>> print(str(config), end="")
    A = 1
    B = 2, 3
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from access.config.entry_fragments import (
    EntryTemplate,
    compile_template,
    expand_value_slots,
    shape_key,
)
from access.config.entry_generate import admitted_value_rules, grammar_templates
from access.config.entry_style import EntryStyle, canonical_ws, classify_slots, style_matches
from access.config.grammar_contract import KEY_LIST, KEY_VALUE
from access.config.grammar_info import GrammarInfo
from access.config.grammar_values import (
    VALUE_RULE_PRIORITY,
    VALUE_TYPE_HANDLER_REGISTRY,
    select_value_rule,
)

_MAX_SNIPPETS = 64
"""Cap on the candidate texts offered for a single insertion."""

# --- Ranking ---


def _rank_key(template: EntryTemplate, style: EntryStyle, order: int) -> tuple[int, ...]:
    """Return the sort key for a complete template; smaller is preferred.

    The terms of ``shape_key``, with a preference for reproducing a neighbouring entry's
    whitespace, and generation order last so the result is deterministic.

    Matching the neighbouring style ranks *below* the amount of fixed text, so style can
    only decide between templates that write the same thing. Ranked any higher, a grammar
    with an optional trailing separator would have one added purely to make the whitespace
    runs line up.

    Args:
        template (EntryTemplate): The template to score.
        style (EntryStyle): Whitespace of a neighbouring entry.
        order (int): Position in generation order, as a final tie-break.

    Returns:
        tuple[int, ...]: The sort key.
    """
    ends_line, block_ok, newlines, literal_length, negated_ws, keys = shape_key(template)
    mismatch = 0 if style_matches(template, style) else 1
    return (ends_line, block_ok, newlines, literal_length, mismatch, negated_ws, keys, order)


def _ranked_templates(info: GrammarInfo, container: str, category: str, style: EntryStyle) -> list[EntryTemplate]:
    """Return the templates for a new entry, most idiomatic first.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        container (str): Container rule name.
        category (str): Entry category.
        style (EntryStyle): Whitespace of a neighbouring entry, used as a preference.

    Returns:
        list[EntryTemplate]: The ranked templates.
    """
    templates = grammar_templates(info, container, category)
    scored = [(_rank_key(template, style, order), template) for order, template in enumerate(templates)]
    scored.sort(key=lambda item: item[0])
    return [template for _, template in scored]


# --- Rendering ---


def _ws_texts(template: EntryTemplate, style: EntryStyle) -> dict[int, str]:
    """Return the text for every whitespace placeholder of *template*."""
    texts = {index: canonical_ws(template, index) for index, fragment in enumerate(template) if fragment.kind == "ws"}
    if not style.has_donor:
        return texts

    slots = classify_slots(template)
    if style.indent and slots.leading:
        # The indentation sits immediately before the key.
        texts[slots.leading[-1]] = style.indent
    # Only reproduce the key runs where the template puts them on the side they came from.
    matched = style.key_pads is not None and slots.key_split == style.key_split
    for indices, pads in ((slots.key if matched else (), style.key_pads), (slots.element, style.element_pads)):
        if pads is not None and len(indices) == len(pads):
            for index, pad in zip(indices, pads, strict=True):
                texts[index] = pad
    return texts


def render(template: EntryTemplate, key: str, value_texts: Sequence[str], style: EntryStyle) -> str:
    """Render a template into the text of an entry.

    Args:
        template (EntryTemplate): The template, already grown to the right number of values.
        key (str): The key name, written as given.
        value_texts (Sequence[str]): Token text for each value, in order.
        style (EntryStyle): Whitespace to copy from a neighbouring entry.

    Returns:
        str: The entry text.

    Examples:
        With no entry to copy from, the whitespace placeholders take a canonical layout: one
        space where the grammar allows it for readability, none at either end of the line.

        >>> from access.config.entry_fragments import Fragment
        >>> template = (Fragment("key"), Fragment("ws"), Fragment("literal", "="),
        ...             Fragment("ws"), Fragment("value"), Fragment("literal", "\\n"))
        >>> render(template, "A", ["1"], EntryStyle())
        'A = 1\\n'

        Given a donor style, the runs it recorded are reproduced instead. One pad per
        stylable slot: this template has two between the key and the value, either side of
        the ``=``, and a style offering a different number is ignored rather than guessed
        at.

        >>> style = EntryStyle(indent="", key_pads=("  ", "  "), has_donor=True)
        >>> render(template, "A", ["1"], style)
        'A  =  1\\n'
    """
    ws_texts = _ws_texts(template, style)
    values = iter(value_texts)
    pieces = []
    for index, fragment in enumerate(template):
        if fragment.kind == "literal":
            pieces.append(fragment.text)
        elif fragment.kind == "ws":
            pieces.append(ws_texts[index])
        elif fragment.kind == "key":
            pieces.append(key)
        elif fragment.kind == "value":
            pieces.append(next(values))
    return "".join(pieces)


# --- Public entry point ---


def iter_entry_snippets(
    info: GrammarInfo,
    container: str,
    category: str,
    key: str,
    value: object,
    style: EntryStyle,
    overrides: Mapping[tuple[str, str], str] | None = None,
    priority: Sequence[str] = VALUE_RULE_PRIORITY,
) -> Iterator[str]:
    """Yield candidate texts for a new entry, most preferred first.

    A parser-declared override for this container and category is offered first, but only
    when *style* has no donor. A template is fixed text, so it cannot reproduce a
    neighbour's whitespace; offered unconditionally it would space a new entry unlike the
    entry directly above it, which is worse than the canonical layout it was meant to
    replace. What it is for is the case where nothing nearby can say how the entry should
    look.

    Even then the override is a preference, not a replacement: if it no longer parses --
    because the grammar changed under it -- the caller simply moves on to the generated
    candidates.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        container (str): Name of the rule the entry will become a child of.
        category (str): One of ``ENTRY_CATEGORIES``.
        key (str): The key name, written as given.
        value (object): The value; a sequence of values for ``key_list``, unused for
            ``key_block`` and ``key_null``.
        style (EntryStyle): Whitespace of a neighbouring entry.
        overrides (Mapping[tuple[str, str], str] | None): Parser-declared templates, keyed
            by container and category.
        priority (Sequence[str]): Order in which to try the admitted value-type rules.

    Yields:
        str: Candidate entry text, to be validated by parsing.
    """
    templates = _ranked_templates(info, container, category, style)

    override = (overrides or {}).get((container, category))
    if override is not None and not style.has_donor:
        options = admitted_value_rules(info, container, category)
        templates.insert(0, compile_template(override, category, options))

    emitted = 0
    for template in templates:
        for text in _render_candidates(template, key, value, style, category, priority):
            yield text
            emitted += 1
            if emitted >= _MAX_SNIPPETS:
                return


def _values_for(category: str, value: object) -> list[object]:
    """Return the individual values a *category* entry has to write."""
    if category == KEY_LIST:
        return list(value)  # type: ignore[call-overload]
    if category == KEY_VALUE:
        return [value]
    return []


def _rule_choices(values: Sequence[object], options: frozenset[str], priority: Sequence[str]) -> Iterator[list[str]]:
    """Yield ways of writing *values*, a value-type rule per value, most preferred first.

    Every admitted rule accepting all the values is offered, not just the most preferred
    one, because the preferred rendering is not always readable back as the same type: a
    grammar with a bare-word value type reads ``True`` as a boolean, so a string of that
    text has to be written quoted instead. The caller validates each candidate and moves on,
    so the priority order expresses a preference and never a restriction.

    Args:
        values (Sequence[object]): The values to write.
        options (frozenset[str]): Value-type rules admitted in this position.
        priority (Sequence[str]): Order in which to consider rules.

    Yields:
        list[str]: One rule name per value.
    """
    offered: list[list[str]] = []
    for rule_name in priority:
        handler = VALUE_TYPE_HANDLER_REGISTRY.get(rule_name)
        if rule_name in options and handler is not None and all(handler.type_check(item) for item in values):
            offered.append([rule_name] * len(values))
            yield offered[-1]

    # A list may hold values of different types, which no single rule covers.
    per_value = [select_value_rule(item, options, priority) for item in values]
    if all(name is not None for name in per_value) and per_value not in offered:
        yield [name for name in per_value if name is not None]


def _render_candidates(
    template: EntryTemplate,
    key: str,
    value: object,
    style: EntryStyle,
    category: str,
    priority: Sequence[str],
) -> Iterator[str]:
    """Yield renderings of *template*, one per way of writing the value."""
    values = _values_for(category, value)
    if category == KEY_LIST:
        template = expand_value_slots(template, len(values))
    slots = [fragment for fragment in template if fragment.kind == "value"]
    if not slots:
        yield render(template, key, [], style)
    else:
        # Every slot of one entry admits the same value types, so the first speaks for all.
        for rule_names in _rule_choices(values, slots[0].options, priority):
            texts = [
                VALUE_TYPE_HANDLER_REGISTRY[name].to_new_token(item)
                for name, item in zip(rule_names, values, strict=True)
            ]
            yield render(template, key, texts, style)
