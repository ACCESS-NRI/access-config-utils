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

1. **Generate.** Walk the compiled rules reachable from the container rule and collect every
   way of deriving one entry of the requested category, as a tuple of ``Fragment``\\ s. A
   fragment is either fixed text or a placeholder for the key, a value, whitespace, or the
   contents of a new block.
2. **Filter.** Discard derivations that do not hold exactly one entry of the right category,
   or the wrong number of values for it.
3. **Rank.** Order the survivors so the most idiomatic rendering wins: a complete line, a
   block whose contents start on their own line, no blank lines, no trailing separator, and
   the whitespace of a neighbouring entry where that costs nothing.
4. **Render.** Substitute the key, the value and the whitespace, then let the caller
   validate the result by parsing it.

Because step 4 is validated by parsing, ranking only decides which *valid* rendering is
used. It can change how the output looks, never whether it is correct.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from lark import Token, Tree
from lark.grammar import Rule, Symbol
from lark.lexer import PatternStr, TerminalDef

from access.config.parser_types import (
    VALUE_RULE_PRIORITY,
    VALUE_TYPE_HANDLER_REGISTRY,
    select_value_rule,
)

if TYPE_CHECKING:
    from lark import Lark

ENTRY_CATEGORIES = ("key_value", "key_list", "key_block", "key_null")
"""Rule names (or aliases) that ``ConfigToDict`` recognises as a configuration entry."""

KEY_RULE = "key"
"""Name of the rule holding a key name, fixed by the grammar contract."""

BLOCK_RULE = "block"
"""Name of the rule holding the contents of a block, fixed by the grammar contract."""

COMMENT_RULES = ("comment", "fortran_comment")
"""Names of the rules holding the text of a comment, fixed by the grammar contract.

``config.lark`` defines exactly these two -- ``#`` for the resource-file dialects, ``!`` for
the Fortran ones -- and both are rules rather than terminals, so the name is what identifies
a comment. The terminal it wraps is an anonymous regular expression, which Lark names
``__ANON_0``.
"""

VALUE_SLOT_COUNTS: Mapping[str, int] = {
    "key_value": 1,
    "key_list": 2,
    "key_block": 0,
    "key_null": 0,
}
"""Number of value placeholders a template for each category must have.

``key_list`` needs exactly two. The second is what makes the element separator visible, so
that it can be replicated for a list of any length -- see ``expand_value_slots``.
"""

_TERMINAL_SAMPLES: Mapping[str, str] = {
    "NEWLINE": "\n",
    "WS_INLINE": " ",
}
"""Text to emit for regular-expression terminals that carry no text of their own.

Terminals whose pattern is a plain string are reproduced from the pattern, and terminals
holding a key or a value are filled in from the caller's data. That leaves only whitespace
and end-of-line. Any other regular-expression terminal has no entry here, and a derivation
needing one is discarded -- which is how derivations that would have to invent comment text,
or Fortran's ``&end`` namelist terminator, are pruned.

Matched on the terminal name *suffix*, because Lark renames terminals imported from another
grammar: ``WS_INLINE`` becomes ``config__WS_INLINE``.
"""

_MAX_ALTERNATIVES = 64
"""Cap on the derivations kept per rule, to bound generation on a large grammar.

The derivations are sorted by ``_simplicity_key`` before the cap is applied, so what
survives is what ranking would have chosen anyway. Truncating in generation order instead
would let one prolific alternative decide the outcome before ranking sees the others.
"""

_MAX_COMBINATIONS = 4096
"""Cap on the combinations examined per rule alternative, including those pruned."""

_MAX_SNIPPETS = 64
"""Cap on the candidate texts offered for a single insertion."""


FragmentKind = Literal["literal", "ws", "key", "value", "block"]


@dataclass(frozen=True)
class Fragment:
    """One piece of the text of an entry.

    Args:
        kind (FragmentKind): ``"literal"`` for fixed text, or a placeholder: ``"ws"`` for
            optional whitespace, ``"key"`` for the key name, ``"value"`` for a value,
            ``"block"`` for the contents of a new (and therefore empty) block.
        text (str): The text, for a ``"literal"`` fragment.
        options (frozenset[str]): Names of the value-type rules allowed here, for a
            ``"value"`` fragment.
    """

    kind: FragmentKind
    text: str = ""
    options: frozenset[str] = frozenset()


EntryTemplate = tuple[Fragment, ...]
"""One way of writing an entry, as an ordered sequence of fragments."""


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


@dataclass
class GrammarInfo:
    """Pre-computed views over a compiled grammar, used to generate entry templates.

    Args:
        rules_by_origin (Mapping[str, Sequence[Rule]]): Compiled rules, grouped by the name
            of the rule they define.
        terminals (Mapping[str, TerminalDef]): Terminal definitions, keyed by name.
    """

    rules_by_origin: Mapping[str, Sequence[Rule]]
    terminals: Mapping[str, TerminalDef]
    # Memoised results, keyed by rule name and by container/category respectively. Held here
    # rather than in a module-level cache so they die with the compiled grammar.
    _value_rules: dict[str, frozenset[str]] = field(default_factory=dict)
    _templates: dict[tuple[str, str], tuple[EntryTemplate, ...]] = field(default_factory=dict)

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


# --- Template generation ---

# A partial derivation: the fragments produced so far, and the entry categories within them.
_Derivation = tuple[EntryTemplate, tuple[str, ...]]


def _expand_symbol(info: GrammarInfo, symbol: Symbol, stack: tuple[str, ...], category: str) -> list[_Derivation]:
    """Return the derivations for a single grammar symbol."""
    name = str(symbol.name)
    if symbol.is_term:
        text = info.sample_terminal(name)
        # A terminal that cannot be reproduced prunes the whole derivation.
        return [] if text is None else [((Fragment("literal", text),), ())]
    if name == KEY_RULE:
        return [((Fragment("key"),), ())]
    if name == BLOCK_RULE:
        # A block being created is empty; its contents are added afterwards, key by key.
        return [((Fragment("block"),), ())]
    options = info.value_rules(name)
    if options:
        return [((Fragment("value", options=options),), ())]
    if name == "ws":
        return [((Fragment("ws"),), ())]
    return _expand_rule(info, name, stack, category)


def _expand_rule(info: GrammarInfo, name: str, stack: tuple[str, ...], category: str) -> list[_Derivation]:
    """Return the derivations for a rule that hold at most one entry, of *category*.

    A rule already on the stack is not re-entered. Lark turns every EBNF repetition into a
    left-recursive helper rule, so cutting recursion here keeps the derivations finite and
    gives exactly the shapes wanted: ``ws*`` contributes at most one whitespace placeholder,
    a repeated value contributes exactly two value placeholders, and a container contributes
    exactly one entry.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        name (str): Name of the rule to expand.
        stack (tuple[str, ...]): Rules already being expanded, to cut recursion.
        category (str): The entry category being generated, used to prune early.

    Returns:
        list[_Derivation]: The derivations, simplest first, capped at ``_MAX_ALTERNATIVES``.
    """
    if name in stack:
        return []
    inner = (*stack, name)
    derivations: list[_Derivation] = []
    for rule in info.rules_by_origin.get(name, ()):
        seen_name = str(rule.alias) if rule.alias else name
        seen = (seen_name,) if seen_name in ENTRY_CATEGORIES else ()
        if _has_excess_entries(seen, category):
            continue
        for fragments, categories in _expand_expansion(info, rule.expansion, inner, category):
            derivations.append((fragments, seen + categories))
    derivations.sort(key=lambda derivation: _simplicity_key(derivation[0]))
    return derivations[:_MAX_ALTERNATIVES]


def _has_excess_entries(categories: tuple[str, ...], category: str) -> bool:
    """Report whether a partial derivation has already gone wrong, entry-wise.

    Pruning as soon as a derivation holds a foreign entry, or a second one of the wanted
    category, is what keeps generation small. Without it a rule such as Fortran's
    ``nml_line`` -- which allows several assignments on one line -- swamps the derivation
    limit with forms that would all be discarded later, and the single-assignment form is
    never reached.

    Args:
        categories (tuple[str, ...]): Entry categories seen in the derivation so far.
        category (str): The entry category being generated.

    Returns:
        bool: True if the derivation can no longer produce exactly one *category* entry.
    """
    return any(item != category for item in categories) or categories.count(category) > 1


def _expand_expansion(
    info: GrammarInfo, expansion: Sequence[Symbol], stack: tuple[str, ...], category: str
) -> list[_Derivation]:
    """Return the derivations for one alternative, as a product over its symbols."""
    per_symbol = []
    for symbol in expansion:
        derivations = _expand_symbol(info, symbol, stack, category)
        if not derivations:
            return []
        per_symbol.append(derivations)

    combined: list[_Derivation] = []
    for combination in itertools.islice(itertools.product(*per_symbol), _MAX_COMBINATIONS):
        fragments: EntryTemplate = ()
        categories: tuple[str, ...] = ()
        for part_fragments, part_categories in combination:
            fragments += part_fragments
            categories += part_categories
        if _has_excess_entries(categories, category):
            continue
        combined.append((fragments, categories))
    return combined


def _is_usable(template: EntryTemplate, categories: tuple[str, ...], category: str) -> bool:
    """Report whether a derivation writes exactly one well-formed entry of *category*.

    Anything else about the derivation is left to the caller's validation, which reads the
    candidate back and compares it with what was asked for.

    Args:
        template (EntryTemplate): The fragments of the derivation.
        categories (tuple[str, ...]): Entry categories within it.
        category (str): The entry category wanted.

    Returns:
        bool: True if the derivation is usable.
    """
    if Counter(categories) != Counter({category: 1}):
        return False
    if _count(template, "value") != VALUE_SLOT_COUNTS[category]:
        return False
    return _count(template, "key") >= 1


def _count(template: EntryTemplate, kind: FragmentKind) -> int:
    """Return the number of fragments of the given kind in *template*."""
    return sum(1 for fragment in template if fragment.kind == kind)


def grammar_templates(info: GrammarInfo, container: str, category: str) -> tuple[EntryTemplate, ...]:
    """Return every way of writing one *category* entry inside *container*, unranked.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        container (str): Name of the rule whose children the entry will become.
        category (str): One of ``ENTRY_CATEGORIES``.

    Returns:
        tuple[EntryTemplate, ...]: The usable templates, empty if the grammar cannot express
            such an entry in that container.
    """
    cached = info._templates.get((container, category))
    if cached is None:
        seen: dict[EntryTemplate, None] = {}
        for template, categories in _expand_rule(info, container, (), category):
            if _is_usable(template, categories, category):
                seen[template] = None
        cached = tuple(seen)
        info._templates[(container, category)] = cached
    return cached


def admitted_value_rules(info: GrammarInfo, container: str, category: str) -> frozenset[str]:
    """Return the value-type rules that can appear in a new *category* entry in *container*.

    Args:
        info (GrammarInfo): Views over the compiled grammar.
        container (str): Container rule name.
        category (str): Entry category.

    Returns:
        frozenset[str]: Union of the admitted value-type rules over all templates.
    """
    admitted: set[str] = set()
    for template in grammar_templates(info, container, category):
        for fragment in template:
            admitted |= fragment.options
    return frozenset(admitted)


# --- Whitespace policy ---


def _neighbour(template: EntryTemplate, index: int, step: int) -> Fragment | None:
    """Return the nearest fragment on one side of *index* that is not a whitespace slot."""
    position = index + step
    while 0 <= position < len(template):
        if template[position].kind != "ws":
            return template[position]
        position += step
    return None


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
    before = _neighbour(template, index, -1)
    after = _neighbour(template, index, 1)
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


def _block_contents_start_a_line(template: EntryTemplate) -> bool:
    """Report whether the contents of a new block would begin on their own line.

    Fortran's namelist rule makes the newline after the group name optional, so ``&NAME/``
    is a legal if unusual empty namelist. Preferring the form whose body is on its own line
    is what leaves room for the keys about to be added to it.

    Args:
        template (EntryTemplate): The template to inspect.

    Returns:
        bool: True if every block placeholder is preceded by a newline, or there is none.
    """
    for index, fragment in enumerate(template):
        if fragment.kind != "block":
            continue
        before = _neighbour(template, index, -1)
        if before is None or before.kind != "literal" or not before.text.endswith("\n"):
            return False
    return True


def _shape_key(template: EntryTemplate) -> tuple[int, ...]:
    """Return the style-independent terms of the sort key; smaller is preferred.

    In order of precedence:

    1. The entry should be a complete line. Grammars that ignore whitespace have no newlines
       at all, so this is a preference rather than a filter.
    2. A new block's contents should start on their own line.
    3. Fewest newlines *other than* the one ending the entry, so no blank lines are added.
    4. Least fixed text, which rejects an optional trailing separator such as ``KEY = 1,``.
    5. Most whitespace placeholders, since unwanted ones render empty anyway.
    6. Fewest repetitions of the key.

    Args:
        template (EntryTemplate): The template to score.

    Returns:
        tuple[int, ...]: The sort key.
    """
    literals = [fragment.text for fragment in template if fragment.kind == "literal"]
    # The *last* fragment has to be the newline. Testing the last literal instead would
    # accept a template whose only newline separates two values, as a continued list does.
    last = _neighbour(template, len(template), -1)
    ends_line = last is not None and last.kind == "literal" and last.text.endswith("\n")
    return (
        0 if ends_line else 1,
        0 if _block_contents_start_a_line(template) else 1,
        sum(text.count("\n") for text in literals) - (1 if ends_line else 0),
        sum(len(text.strip()) for text in literals),
        -_count(template, "ws"),
        _count(template, "key"),
    )


def _simplicity_key(template: EntryTemplate) -> tuple[int, ...]:
    """Return a style-independent preference for a derivation; smaller is simpler.

    The same terms ranking uses, so the derivations surviving the generation cap are the
    ones ranking would have chosen anyway.

    Args:
        template (EntryTemplate): The template to score.

    Returns:
        tuple[int, ...]: The sort key.
    """
    return _shape_key(template)


def _rank_key(template: EntryTemplate, style: EntryStyle, order: int) -> tuple[int, ...]:
    """Return the sort key for a complete template; smaller is preferred.

    The terms of ``_shape_key``, with a preference for reproducing a neighbouring entry's
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
    ends_line, block_ok, newlines, literal_length, negated_ws, keys = _shape_key(template)
    mismatch = 0 if _style_matches(template, style) else 1
    return (ends_line, block_ok, newlines, literal_length, mismatch, negated_ws, keys, order)


def ranked_templates(info: GrammarInfo, container: str, category: str, style: EntryStyle) -> list[EntryTemplate]:
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


def expand_value_slots(template: EntryTemplate, count: int) -> EntryTemplate:
    """Grow a two-value list template so that it holds *count* values.

    The fragments between the two value placeholders are the element separator, so repeating
    them along with the placeholder produces a list of any length using the grammar's own
    separator.

    Args:
        template (EntryTemplate): A template with exactly two value placeholders.
        count (int): Number of values wanted, at least two.

    Returns:
        EntryTemplate: The grown template.
    """
    values = [index for index, fragment in enumerate(template) if fragment.kind == "value"]
    first, second = values[0], values[1]
    separator = template[first + 1 : second]
    repeated = (separator + (template[second],)) * (count - 1)
    return template[: first + 1] + repeated + template[second + 1 :]


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
    """
    for entry in reversed(_entries_of(container)):
        if entry.data != "key_block":
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
        if child.data == "ws":
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


# --- Parser-declared overrides ---

_MARKERS: Mapping[str, Fragment] = {
    "{key}": Fragment("key"),
    "{value}": Fragment("value"),
    "{block}": Fragment("block"),
    "{ws}": Fragment("ws"),
}


def compile_template(text: str, category: str, options: frozenset[str]) -> EntryTemplate:
    """Compile a hand-written entry template declared by a parser.

    The template is ordinary text with ``{key}``, ``{value}``, ``{block}`` and ``{ws}``
    markers. ``{ws}`` marks whitespace that should follow a neighbouring entry's style;
    literal spaces are left alone.

    Args:
        text (str): The template text.
        category (str): The entry category it is for.
        options (frozenset[str]): Value-type rules admitted in this position.

    Returns:
        EntryTemplate: The compiled template.

    Raises:
        ValueError: If the markers do not match what *category* requires.
    """
    fragments: list[Fragment] = []
    rest = text
    while rest:
        found = [(rest.index(name), name) for name in _MARKERS if name in rest]
        if not found:
            fragments.append(Fragment("literal", rest))
            break
        position, name = min(found)
        if position:
            fragments.append(Fragment("literal", rest[:position]))
        marker = _MARKERS[name]
        fragments.append(Fragment("value", options=options) if marker.kind == "value" else marker)
        rest = rest[position + len(name) :]

    template = tuple(fragments)
    expected = VALUE_SLOT_COUNTS[category]
    if _count(template, "value") != expected:
        raise ValueError(f"A {category} template needs exactly {expected} '{{value}}' markers")
    if _count(template, "block") != (1 if category == "key_block" else 0):
        raise ValueError(f"A {category} template has the wrong number of '{{block}}' markers")
    if _count(template, "key") < 1:
        raise ValueError("An entry template needs at least one '{key}' marker")
    return template


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

    A parser-declared override for this container and category is offered first, then the
    grammar-derived templates in ranked order. The override is a preference, not a
    replacement: if it no longer parses -- because the grammar changed under it -- the
    caller simply moves on to the generated candidates.

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
    templates = ranked_templates(info, container, category, style)

    override = (overrides or {}).get((container, category))
    if override is not None:
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
    if category == "key_list":
        return list(value)  # type: ignore[call-overload]
    if category == "key_value":
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
    if category == "key_list":
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
