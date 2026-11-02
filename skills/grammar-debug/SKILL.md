---
name: grammar-debug
description: Diagnose Lark parse failures and round-trip mismatches in access-config-utils parsers. Use when a config file fails to parse (UnexpectedToken/UnexpectedCharacters/UnexpectedEOF), when str(config) does not reproduce the input, when a key is missing from the parsed dict, or when editing a value corrupts the output.
---

# Debugging grammars and round-trips

Each section below is one failure mode with a distinct cause. Identify which one you have before
touching the grammar — blind grammar edits usually trade one failure for another.

## 0. Reproduce in isolation

Cut the input down to the smallest snippet that still fails, then work from a scratch script:

```python
from access.config.fortran_nml import FortranNMLParser   # or the parser under test
from access.config.grammar_compiled import compile_grammar

p = FortranNMLParser()
lark = compile_grammar(p.grammar).lark   # built exactly as parse() builds it
# start= is required: the parser has more than one start symbol, so an unqualified parse()
# raises ConfigurationError. parse() appends a newline, so do the same.
tree = lark.parse(text + "\n", start="start")
print(tree.pretty())
```

`tree.pretty()` is the primary diagnostic. Almost every issue below is visible in it.

## 1. It does not parse

`UnexpectedToken` / `UnexpectedCharacters` / `UnexpectedEOF` from Lark.

- Check the reported line/column against the grammar rule that should cover it.
- `UnexpectedEOF` on otherwise-valid input usually means a rule requires a trailing `NEWLINE` that
  is not there. Remember `parse()` adds exactly one `"\n"` — reproduce that in your scratch script
  or you will chase a phantom.
- Bare words claimed by the wrong value rule: `identifier` and `path` overlap heavily. Reorder the
  `?value:` alternatives, or tighten the terminal.
- Whitespace: if the grammar captures `ws` explicitly (it should), then *every* place the format
  permits a space needs a `ws*`. A missing one shows up as an unexpected-token error on a space.
- Blank and comment-only lines need their own alternative in the line rule.

## 2. It parses, but a key is missing from the dict

The tree is fine; `EntryReader` never dispatched. Its callbacks are named exactly `key_value`,
`key_list`, `key_block`, `key_null` — check `tree.pretty()` for the rule name actually produced:

- A rule named something else needs the alias arrow: `... -> key_value`.
- A transparent rule (`?name:`) with a single child is **inlined away** by Lark, so its name never
  reaches the interpreter. This is intentional for pass-through rules but fatal for a rule the
  interpreter must see.
- `ValueError: No values found in Tree` — the children contain no rule whose name is in
  `VALUE_TYPE_HANDLER_REGISTRY`. You used a rule name outside the registry, or wrote the value
  inline as a terminal instead of via a `config.lark` value rule.
- `ValueError: No 'key' rule nodes found...` — the key is not wrapped in a rule named `key`.
- `TypeError: No key found.` — the `key` rule's first child is a `Tree`, not a `Token`.
- A `key_block` whose nested content is missing: `key_block` must have a child whose rule is one
  of the parser's `block_rules` (`block` unless it declares more); `EntryReader.key_block` scans
  for one and silently records nothing if none is there.
- Blocks that read fine but take no new key: the block rule is only an alias (`-> block`).
  `compile_grammar` adds `block` as a start symbol only for a rule *named* `block`, so a new
  entry cannot be parsed in that context. Define a real `block` rule.

## 3. It parses, but `str(config) != input`

The reconstruction dropped or added text. Compare with a diff, not by eye:

```python
import difflib
print("".join(difflib.unified_diff(text.splitlines(True), str(cfg).splitlines(True))))
```

- Missing spaces → whitespace was `%ignore`d or a `ws*` is absent from that position.
- Missing blank lines → no `empty_line` alternative.
- Missing comments → `comment` / `fortran_comment` not wired into `line_end` and the line rule.
- Missing/extra trailing newline → `Config.__str__` strips exactly one trailing `"\n"`; the grammar
  should not also consume or emit one.
- Text written in *another rule's* syntax → two rules are aliased to the same entry category and
  admit the same children. The reconstructor chooses a rule by matching a node's children, so it
  cannot tell them apart by where the node sits, and anonymous string tokens (`"="`, `":"`) have
  been filtered out of the tree before it looks. Capture the punctuation that distinguishes the two
  syntaxes in a rule of its own — `assign: "="` in `nuopc_config.py`. Expect the collision to show
  up for only *some* inputs: NUOPC's table lists were written with the top-level separator at two
  elements but not at three, and its table scalars only when nothing spaced out the `=`.

The reconstructor mutates the tree, which is why `ConfigStore._reconstruct` deep-copies first. If you
call `Reconstructor.reconstruct` yourself in a scratch script, copy the tree too or subsequent
operations will see a mangled tree and you will misdiagnose.

## 4. Editing a value corrupts the output

The parse and round-trip are fine, but assignment writes to the wrong place.

- `TypeError: Trying to change value type` (from `update_node_value` or the value handler) means
  the new value fails the handler's `type_check` for the rule node's type. This is intended
  behaviour: types are fixed by the parse. Confirm the source value's rule really is the type you
  think.
- `update_node_value` writes to `rule_node.children[0]` — it assumes the value-type rule node has
  its `Token` as the first child. A value rule with a different shape breaks silently.
- Wrong node updated → the wrong node was named in the `EntryRef`. `ConfigStore.refs[key]` holds
  `value_nodes` (one for `key_value`; one per element for `key_list`, where a repeat node appears
  once per element it covers and `None` marks an array position nothing wrote; none otherwise),
  `block_nodes` for a `key_block` (more than one when `repeated_blocks` is `"separate"`),
  `entry_nodes` for the whole entries (more than one when the file assigns a key twice, which is
  what `__delitem__` unlinks), and `values` for a Fortran array gathered from indexed entries.
  Print `cfg._store.refs[key]`, on the `Config` holding the key, to see which is which.
- A list of another length refused: `ValueError` "A list needs at least two elements", or
  "Cannot change the length of ...: <reason>" for a list written with a repeat count, array
  qualifiers, skipped positions or in several assignments. `UnsupportedEntryError` "No layout of
  ... reads back as the list requested" means every layout `list_layout` offered failed the
  parse-and-read-back check; look at the separator and line breaks of the entry being resized.
- `AssertionError` in `AddParent.__default__` means a `Tree` was reached twice, i.e. the same node
  object appears in two places. Any code building tree nodes must create a **fresh object** for
  every position, never reuse one instance.

## 5. Transparent start rules

The start rule must be `start:`, never `?start:`. A transparent rule collapses when it has a single
child, so a config containing one item ends up with that item as the tree root: there is no container
node, `.parent` walks behave differently, and key addition has nothing to add to. If a bug only
reproduces for single-entry files, this is why. Test with one entry as well as several.

## 6. Adding a key fails or looks wrong

`UnsupportedEntryError` means no candidate text both parsed and read back as the requested value. To
see what was tried:

```python
from access.config.entry_render import iter_entry_snippets
from access.config.entry_style import EntryStyle, probe_entry_style
cfg = config["grp"]   # the Config the key is added to; config itself for the top level
ctx = cfg._store.ctx
declared = ctx.entry_templates.get((str(cfg._store.tree.data), "key_value"))
for snippet in iter_entry_snippets("key_value", ("NEWKEY",), 1, EntryStyle(), declared, ctx.info.value_rules):
    print(repr(snippet))
```

The keys are a path: one key for a plain entry, one per level for a chain such as the Fortran
`a%b = 1` or a field table header. With a real style instead of `EntryStyle()`, the first candidate
is a copy of the nearest entry of the same kind, if the container has one. To see what was copied:

```python
style = probe_entry_style(cfg._store.tree, ctx.reconstructor)
for signature, template in style.donors:   # signature: (depth, category)
    print(signature, [(f.kind, f.text, f.level, f.notation) for f in template])
```

An entry on a line shared with another, one written with a repeat count or a skipped position,
and a list broken after its first value are never copied.

An empty list means the parser declares no template for that kind of entry in that container
(`ctx.entry_templates`) and the file has no entry of the kind to copy. Check the container rule name
(`cfg._store.tree.data`) against the keys of `entry_templates`: it is the rule holding the entry,
`start` for the top level and `block` or another of `block_rules` inside one. A template is a
string with `{key}`, `{keys:SEP}` and `{value}` markers, and a list template needs two `{value}`s;
one that is malformed raises `ValueError` the first time it is used.

Candidates that *are* offered but all rejected mean either they do not parse in that container, or
they read back as a different value: a bare word written for a `Path` is read back as an
`identifier`, so the value type changes and the candidate is refused. That refusal is correct — the
alternative would be a file whose text and dict disagree.

If a key is added but formatted oddly, the culprit is the donor or the template. A donor is the
nearest entry of the same kind, copied whole — its spacing and punctuation included — so a ragged
neighbour gives a ragged entry. Where the container has no entry of the kind, the text is what the
parser declared. The only thing read from the file besides the donor is the indentation of the entry
above (`EntryStyle.indent`), tried first, and then the template's own.

If a key is added in the wrong *place*, the rule is in `entry_insertion_index`: one past the last
child that holds an entry, then forward over any run of comment lines starting there, stopping at
the first blank line, entry, or the container's end. Two consequences to check before suspecting a
bug. A comment on an entry's own line is part of that entry's node, so it never separates from it —
and it is why the run is stepped over at all, since one-past-the-entry would otherwise be the second
line of that comment. A blank line is *not* a comment line (same `empty_line` rule, no
`comment`/`fortran_comment` node below it), so it stops the run and the new key goes above it.
Fortran between namelist groups is the exception: those lines are one `random_text` blob with no
comment node in it, so nothing is stepped over there.

A block created by an assignment is the one container that has no entry above it to read an indent
from: it is empty, so `probe_sibling_block_style(container, reconstructor)` gives it the style of
the nearest block above it that has entries, and `ConfigStore.fallback_style` holds that until the
block has an entry of its own to copy. Two consequences when a new block looks wrong: the donor is
the *nearest* sibling, not the prevailing convention of the file, and a donated indent the grammar
has no leading-whitespace slot for makes that candidate fail to parse, so the template's own indent
is used instead.

## Before finishing

Add the reproducing input as a regression test in the relevant `tests/test_<format>.py`, then run
the gate in [`skills/check/SKILL.md`](../check/SKILL.md) in full — grammar changes routinely break
a different format's expectations through the shared `config.lark`.
