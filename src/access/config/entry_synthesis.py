# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Synthesis of the text of a new configuration entry.

Adding a key to a parsed configuration file means producing parse tree nodes that the parser
never built. Rather than assembling ``Tree`` and ``Token`` objects by hand -- which means
replicating Lark's rule inlining, alias handling, filtered-out anonymous terminals and
``%import`` terminal renaming -- this module produces the **text** of the new entry. The
caller hands that text back to Lark with the container rule as the start symbol, so Lark
builds nodes that are by construction the ones it would have produced for a file containing
the entry.

The text is derived from the compiled grammar itself, so a new format gets entry synthesis
for free.

The pipeline
------------

1. **Generate.** Collect every way the grammar can derive one entry of the requested
   category, as a tuple of ``Fragment``\\ s -- fixed text interleaved with placeholders for
   the key, a value, whitespace, or the contents of a new block. This is
   ``entry_templates``, which also filters out any derivation that does not hold exactly one
   entry of the right category, or the wrong number of values for it.
2. **Rank.** Order the survivors so the most idiomatic rendering wins: a complete line, a
   block whose contents start on their own line, no blank lines, no trailing separator, and
   the whitespace of a neighbouring entry where that costs nothing.
3. **Render.** Substitute the key, the value and the whitespace, then let the caller
   validate the result by parsing it.

Because step 3 is validated by parsing, ranking only decides which *valid* rendering is
used. It can change how the output looks, never whether it is correct.

This module holds the half of synthesis that depends on the file being edited: the
whitespace copied from its existing entries, the ranking that prefers it, and the rendering
that applies it. The grammar-only half lives in ``entry_templates``.

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
from dataclasses import dataclass

from lark import Token, Tree

from access.config.entry_templates import (
    EntryTemplate,
    GrammarInfo,
    admitted_value_rules,
    compile_template,
    expand_value_slots,
    grammar_templates,
    neighbour_fragment,
    shape_key,
)
from access.config.grammar_contract import (
    BLOCK_RULE,
    COMMENT_RULES,
    ENTRY_CATEGORIES,
    KEY_BLOCK,
    KEY_LIST,
    KEY_RULE,
    KEY_VALUE,
    WS_RULE,
)
from access.config.parser_types import (
    VALUE_RULE_PRIORITY,
    VALUE_TYPE_HANDLER_REGISTRY,
    select_value_rule,
)

_MAX_SNIPPETS = 64
"""Cap on the candidate texts offered for a single insertion."""


@dataclass(frozen=True)
class EntryStyle:
    """Whitespace copied from existing entries, so a new one matches its neighbours.

    Args:
        indent (str): Whitespace before the key.
        key_pads (tuple[str, ...] | None): Whitespace runs between the key and the first
            value, in order, or ``None`` when no entry demonstrates it.
        element_pads (tuple[str, ...] | None): Whitespace runs between the first and second
            values, in order, or ``None`` when no entry demonstrates it. That is the case
            whenever no entry holds two values, so a scalar neighbour does not dictate how a
            list is spaced.
        has_donor (bool): Whether any entry was found to copy from. When false, rendering
            falls back to a canonical layout.
    """

    indent: str = ""
    key_pads: tuple[str, ...] | None = None
    element_pads: tuple[str, ...] | None = None
    has_donor: bool = False


# --- Whitespace policy ---


def _is_punctuation(text: str) -> bool:
    """Report whether *text* is non-empty and made only of punctuation."""
    return text != "" and not any(character.isalnum() or character.isspace() for character in text)


def _canonical_ws(template: EntryTemplate, index: int) -> str:
    """Return the whitespace to emit at *index* when there is no entry to copy from.

    Empty at the start and the end of a line, and before a separator that follows a value,
    so a canonically rendered entry carries no indentation, no trailing whitespace, and
    reads as ``a, b`` rather than ``a , b``. A single space anywhere else, which is where a
    grammar allows optional whitespace for readability, as in ``KEY = value``.

    Args:
        template (EntryTemplate): The template being rendered.
        index (int): Index of the whitespace placeholder.

    Returns:
        str: The whitespace text.
    """
    before = neighbour_fragment(template, index, -1)
    after = neighbour_fragment(template, index, 1)
    if before is None or before.kind == "block" or (before.kind == "literal" and before.text.endswith("\n")):
        return ""
    if after is None or after.kind == "block" or (after.kind == "literal" and after.text.startswith("\n")):
        return ""
    if before.kind == "value" and after.kind == "literal" and _is_punctuation(after.text):
        return ""
    return " "


@dataclass(frozen=True)
class _Slots:
    """Indices of a template's whitespace placeholders, grouped by what they space out.

    Args:
        leading (tuple[int, ...]): Placeholders before the key, which carry the indentation.
        key (tuple[int, ...]): Stylable placeholders between the key and the first value.
        element (tuple[int, ...]): Stylable placeholders between the first two values.
    """

    leading: tuple[int, ...]
    key: tuple[int, ...]
    element: tuple[int, ...]


def _classify_slots(template: EntryTemplate) -> _Slots:
    """Group the whitespace placeholders of *template* into regions.

    Only placeholders that render as a space canonically are collected for the key and
    element regions. The others are fixed by the line structure -- a line never starts or
    ends with copied whitespace -- and must not consume the copied whitespace runs.

    Args:
        template (EntryTemplate): The template to classify.

    Returns:
        _Slots: The grouped indices.
    """
    keys = [index for index, fragment in enumerate(template) if fragment.kind == "key"]
    values = [index for index, fragment in enumerate(template) if fragment.kind == "value"]
    first_key = keys[0] if keys else len(template)
    # With no value to stop at, everything after the key spaces out the key, as in "KEY =".
    key_end = values[0] if values else len(template)
    element_start, element_end = (values[0], values[1]) if len(values) > 1 else (len(template), len(template))

    def stylable(start: int, stop: int) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(start, stop)
            if template[index].kind == "ws" and _canonical_ws(template, index) == " "
        )

    leading = tuple(index for index in range(first_key) if template[index].kind == "ws")
    return _Slots(leading, stylable(first_key, key_end), stylable(element_start, element_end))


def _style_matches(template: EntryTemplate, style: EntryStyle) -> bool:
    """Report whether *template* can reproduce the whitespace *style* was taken from."""
    if not style.has_donor:
        return True
    slots = _classify_slots(template)
    if style.indent and not slots.leading:
        return False
    for indices, pads in ((slots.key, style.key_pads), (slots.element, style.element_pads)):
        if pads is not None and len(indices) != len(pads):
            return False
    return True


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
    mismatch = 0 if _style_matches(template, style) else 1
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
    texts = {index: _canonical_ws(template, index) for index, fragment in enumerate(template) if fragment.kind == "ws"}
    if not style.has_donor:
        return texts

    slots = _classify_slots(template)
    if style.indent and slots.leading:
        # The indentation sits immediately before the key.
        texts[slots.leading[-1]] = style.indent
    for indices, pads in ((slots.key, style.key_pads), (slots.element, style.element_pads)):
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

        >>> from access.config.entry_templates import Fragment
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


# --- Style probing ---


def _entry_nodes(node: Tree) -> list[Tree]:
    """Return the entry rule nodes at or below *node*, not entering a nested block."""
    if node.data in ENTRY_CATEGORIES:
        return [node]
    found = []
    for child in node.children:
        if isinstance(child, Tree) and child.data != BLOCK_RULE:
            found += _entry_nodes(child)
    return found


def contains_entry(node: Tree | Token) -> bool:
    """Report whether a container child holds a configuration entry.

    Args:
        node (Tree | Token): A child of a container rule node.

    Returns:
        bool: True if an entry rule node is at or below *node*.
    """
    return isinstance(node, Tree) and bool(_entry_nodes(node))


def _holds_comment(node: Tree) -> bool:
    """Report whether a comment rule node sits at or below *node*."""
    if node.data in COMMENT_RULES:
        return True
    return any(isinstance(child, Tree) and _holds_comment(child) for child in node.children)


def is_comment_line(node: Tree | Token) -> bool:
    """Report whether a container child is a line holding nothing but a comment.

    A comment-only line and a blank line are the same rule in every grammar, told apart by
    whether a comment node hangs below it. A child that holds an entry is never a comment
    line, however many comments it carries: those belong to the entry's own line and travel
    with it.

    Args:
        node (Tree | Token): A child of a container rule node.

    Returns:
        bool: True if *node* is a line holding only a comment.
    """
    return isinstance(node, Tree) and not contains_entry(node) and _holds_comment(node)


def _entries_of(container: Tree) -> list[Tree]:
    """Return the entry rule nodes held directly by *container*, in the order written.

    Args:
        container (Tree): A container rule node.

    Returns:
        list[Tree]: The entry nodes, not descending into a nested block.
    """
    return [entry for child in container.children if isinstance(child, Tree) for entry in _entry_nodes(child)]


def probe_entry_style(container: Tree) -> EntryStyle:
    """Read the whitespace of the entries in *container*, to match when adding a new one.

    Each kind of whitespace is taken from the last entry that actually demonstrates it: the
    indentation and the spacing around the assignment from the last entry with both a key
    and a value, and the spacing around the element separator from the last entry with two
    values. So a scalar neighbour does not dictate how a list is spaced, and a list
    neighbour does not stop a scalar being spaced like the scalars around it.

    Falls back to a canonical layout when the container holds no entries at all, which is
    the case for a block that was empty in the file. A block that was *just created* has
    nothing to copy from either, but is not left canonical: see
    ``probe_sibling_block_style``.

    Args:
        container (Tree): The container rule node the new entry will be added to.

    Returns:
        EntryStyle: The whitespace to copy, or a style with ``has_donor`` false.

    Examples:
        The effect, through the public interface: an unusually spaced neighbour is matched
        rather than normalised, because a minimal diff matters more than tidiness.

        >>> from access.config import MOM6InputParser
        >>> config = MOM6InputParser().parse("A   =   1\\n")
        >>> config["B"] = 2
        >>> print(str(config), end="")
        A   =   1
        B   =   2
    """
    entries = _entries_of(container)
    if not entries:
        return EntryStyle()

    shapes = [_shape_of(entry) for entry in entries]
    with_value = [shape for shape in shapes if shape[2]]
    indent, key_pads, _ = with_value[-1] if with_value else shapes[-1]
    with_elements = [shape for shape in shapes if len(shape[2]) > 1]
    element_pads = with_elements[-1][2][1] if with_elements else None
    return EntryStyle(indent, key_pads, element_pads, has_donor=True)


def probe_sibling_block_style(container: Tree) -> EntryStyle:
    """Read the whitespace used inside the blocks of *container*, for a block just created.

    A block created by an assignment starts empty, so ``probe_entry_style`` finds nothing in
    it to copy and its contents would be written with no indentation at all -- foreign to a
    file whose other blocks are indented. The style is therefore taken from a sibling block
    instead, searching from the nearest one upwards, since that is where the new block was
    added. This mirrors ``probe_entry_style`` taking its style from the *last* entry: in
    both cases the closest neighbour wins.

    Only the rule named ``BLOCK_RULE`` is looked at, and only as a direct child of the
    entry, which is the same contract ``ConfigToDict.key_block`` relies on to find the body
    of a block.

    Args:
        container (Tree): The container rule node the new block was added to.

    Returns:
        EntryStyle: The whitespace to copy, or a style with ``has_donor`` false when no
            sibling block holds an entry -- including the new block itself, which is empty.

    Examples:
        A new block is laid out like the block next to it, not like a fresh file.

        >>> from access.config import NUOPCParser
        >>> config = NUOPCParser().parse("OLD::\\n  P = 1\\n::\\n")
        >>> config["NEW"] = {"X": 2}
        >>> print(str(config), end="")
        OLD::
          P = 1
        ::
        NEW::
          X = 2
        ::
    """
    for entry in reversed(_entries_of(container)):
        if entry.data != KEY_BLOCK:
            continue
        for child in entry.children:
            if isinstance(child, Tree) and child.data == BLOCK_RULE:
                style = probe_entry_style(child)
                if style.has_donor:
                    return style
    return EntryStyle()


def _shape_of(entry: Tree) -> tuple[str, tuple[str, ...], list[tuple[str, ...]]]:
    """Split the whitespace runs of an entry rule node into the parts a style is built from.

    Args:
        entry (Tree): An entry rule node.

    Returns:
        tuple[str, tuple[str, ...], list[tuple[str, ...]]]: The indentation, the runs
            between the key and the first value, and the runs before each value in turn --
            so the second element, when present, is the element separator.
    """
    kinds: list[tuple[str, str]] = []
    for child in (child for child in entry.children if isinstance(child, Tree)):
        if child.data == WS_RULE:
            kinds.append(("ws", "".join(str(token) for token in child.children)))
        elif child.data == KEY_RULE:
            kinds.append((KEY_RULE, ""))
        elif child.data in VALUE_TYPE_HANDLER_REGISTRY:
            kinds.append(("value", ""))

    def runs(start: int, stop: int) -> tuple[str, ...]:
        return tuple(text for kind, text in kinds[start:stop] if kind == "ws")

    keys = [index for index, (kind, _) in enumerate(kinds) if kind == KEY_RULE]
    key_at = keys[0] if keys else len(kinds)
    values = [index for index, (kind, _) in enumerate(kinds) if kind == "value"]

    # The run before each value, measured from the key for the first and from the previous
    # value thereafter. With no value at all, everything after the key spaces out the key.
    if values:
        boundaries = [key_at, *values[:-1]]
        gaps = [runs(previous, index) for previous, index in zip(boundaries, values, strict=True)]
        key_pads = gaps[0]
    else:
        gaps = []
        key_pads = runs(key_at, len(kinds))
    return "".join(runs(0, key_at)), key_pads, gaps


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
