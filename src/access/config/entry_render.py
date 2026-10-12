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

The pipeline
------------

1. **Template.** A new entry is written as a copy of the nearest entry of its kind in the
   file, if there is one; see ``entry_style.donor_template``. Otherwise it is written from
   the template its parser declares in ``ConfigParser.entry_templates``, indented like the
   entry above where the file shows one.
2. **Render.** Substitute the key, the value and the whitespace, once for each way the
   value can be written: every value-type rule the format has that accepts it, most
   preferred first.

The caller validates each candidate by parsing it and reading it back, so what is offered
first is a matter of style: it can change how the output looks, never whether it is correct.
Synthesis is reached through the ordinary mapping interface: assigning a key the
configuration does not have writes it in the format's own syntax, spaced like the entries
around it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from access.config.entry_fragments import EntryTemplate, Fragment, compile_template, expand_value_slots
from access.config.entry_style import EntryStyle, canonical_ws, classify_slots
from access.config.grammar_contract import KEY_LIST, KEY_VALUE
from access.config.grammar_values import VALUE_TYPE_HANDLER_REGISTRY, select_value_rule

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


def render(template: EntryTemplate, keys: Sequence[str], value_texts: Sequence[str], style: EntryStyle) -> str:
    """Render a template into the text of an entry.

    Args:
        template (EntryTemplate): The template, already grown to the right number of values.
        keys (Sequence[str]): The key names, outermost first, written as given. One for a
            plain entry; a chain such as ``a%b = 1`` takes one per level.
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
        >>> render(template, ["A"], ["1"], EntryStyle())
        'A = 1\\n'

        Given a donor style, the runs it recorded are reproduced instead. One pad per
        stylable slot: this template has two between the key and the value, either side of
        the ``=``, and a style offering a different number is ignored rather than guessed
        at. ``key_split`` says how many of them fall before the ``=``, since a run recorded
        on the wrong side of it would move the whitespace rather than reproduce it.

        >>> style = EntryStyle(key_pads=("  ", "  "), key_split=1, has_donor=True)
        >>> render(template, ["A"], ["1"], style)
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
            pieces.append(keys[fragment.level])
        elif fragment.kind == "value":
            pieces.append(next(values))
    return "".join(pieces)


# --- Public entry point ---


def iter_entry_snippets(
    category: str,
    keys: Sequence[str],
    value: object,
    style: EntryStyle,
    declared: str | None,
    admitted: frozenset[str],
    priority: Sequence[str],
) -> Iterator[str]:
    """Yield candidate texts for a new entry, most preferred first.

    A template copied from a neighbouring entry of the same kind comes first: it is the one
    thing that writes the new entry exactly as the file writes the others, punctuation and
    value notation included.

    The template the parser declares for this kind of entry comes next, written with the
    indentation of the entry above where the file shows one, and then with its own. It is
    how the first entry of a kind is written when the file has none to copy.

    Neither is more than a preference: if one does not parse, or reads back as something
    else, the caller simply moves on.

    Args:
        category (str): One of ``ENTRY_CATEGORIES``.
        keys (Sequence[str]): The key names, outermost first, written as given: one for a
            plain entry, one per level for a chain such as ``a%b = 1``.
        value (object): The value; a sequence of values for ``key_list``, unused for
            ``key_block`` and ``key_null``.
        style (EntryStyle): The layout of the neighbouring entries.
        declared (str | None): The text the parser declares for this container and
            category, if it does.
        admitted (frozenset[str]): The value-type rules the format has.
        priority (Sequence[str]): Order in which to try the admitted value-type rules.

    Yields:
        str: Candidate entry text, to be validated by parsing.
    """
    depth = len(keys)
    templates = [style.donor(depth, category)]
    if declared is not None:
        shown = style.indent if style.has_donor else None
        templates += [compile_template(declared, category, depth, indent) for indent in (shown, None)]
    for template in dict.fromkeys(template for template in templates if template is not None):
        yield from _render_candidates(template, keys, value, style, category, admitted, priority)


def _values_for(category: str, value: object) -> list[object]:
    """Return the individual values a *category* entry has to write."""
    if category == KEY_LIST:
        return list(value)  # type: ignore[call-overload]
    if category == KEY_VALUE:
        return [value]
    return []


def _rule_choices(values: Sequence[object], admitted: frozenset[str], priority: Sequence[str]) -> Iterator[list[str]]:
    """Yield ways of writing *values*, a value-type rule per value, most preferred first.

    Every admitted rule accepting all the values is offered, not just the most preferred
    one, because the preferred rendering is not always readable back as the same type: a
    grammar with a bare-word value type reads ``True`` as a boolean, so a string of that
    text has to be written quoted instead. The caller validates each candidate and moves on,
    so the priority order expresses a preference and never a restriction.

    Args:
        values (Sequence[object]): The values to write.
        admitted (frozenset[str]): The value-type rules the format has.
        priority (Sequence[str]): Order in which to consider rules.

    Yields:
        list[str]: One rule name per value.
    """
    offered: list[list[str]] = []
    for rule_name in priority:
        handler = VALUE_TYPE_HANDLER_REGISTRY.get(rule_name)
        if rule_name in admitted and handler is not None and all(handler.type_check(item) for item in values):
            offered.append([rule_name] * len(values))
            yield offered[-1]

    # A list may hold values of different types, which no single rule covers.
    per_value = [select_value_rule(item, admitted, priority) for item in values]
    if all(name is not None for name in per_value) and per_value not in offered:
        yield [name for name in per_value if name is not None]


def _render_candidates(
    template: EntryTemplate,
    keys: Sequence[str],
    value: object,
    style: EntryStyle,
    category: str,
    admitted: frozenset[str],
    priority: Sequence[str],
) -> Iterator[str]:
    """Yield renderings of *template*, one per way of writing the value."""
    values = _values_for(category, value)
    if category == KEY_LIST:
        template = expand_value_slots(template, len(values))
    slots = [fragment for fragment in template if fragment.kind == "value"]
    if not slots:
        yield render(template, keys, [], style)
    else:
        for rule_names in _rule_choices(values, admitted, priority):
            texts = [_value_text(*choice) for choice in zip(rule_names, values, slots, strict=True)]
            yield render(template, keys, texts, style)


def _value_text(rule: str, value: object, slot: Fragment) -> str:
    """Return the token text writing *value* with *rule* into *slot*.

    A slot copied from an existing entry knows how that entry wrote its value, and a value
    written with the same rule takes on its notation: the quote character, the spelling of a
    logical, the exponent letter. Where that notation cannot hold the value, it is written
    the way a brand-new value is.

    Args:
        rule (str): The value-type rule chosen.
        value (object): The value.
        slot (Fragment): The value placeholder.

    Returns:
        str: The token text.
    """
    handler = VALUE_TYPE_HANDLER_REGISTRY[rule]
    if slot.notation is not None and slot.notation[0] == rule:
        try:
            return handler.to_token(value, slot.notation[1])
        except (TypeError, ValueError):
            pass
    return handler.to_new_token(value)
