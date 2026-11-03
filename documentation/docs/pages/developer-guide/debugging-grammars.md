# Debugging a grammar

A grammar can fail in several distinct ways, each with its own cause: a file that does not parse,
a key missing from the dict, a round trip that differs, an edit written in the wrong place, and a
key that cannot be added. Find out which one it is before changing the grammar: an edit made by
trial and error usually trades one failure for another.

## Reproduce it in isolation

Cut the file down to the smallest text that still fails, and look at the parse tree. Parse it as
`parse` does, with a newline added and the start rule named, since a grammar has several start
symbols:

```py
from access.config import FortranNMLParser
from access.config.grammar_compiled import compile_grammar

lark = compile_grammar(FortranNMLParser().grammar).lark
print(lark.parse("&g\n  x = 1\n/\n" + "\n", start="start").pretty())
```

```text
start
  random_text
  key_block
    nml_start
      amp_open
    key	g
    line_end
    block
      nml_line
        key_value
          ws
          key	x
          eq
            ws
          ws
          integer	1
        line_end
    nml_end
      slash_close
      line_end
  random_text
```

`pretty()` is the first thing to look at: nearly every problem below shows in it.

## It does not parse

Lark raises `UnexpectedCharacters`, `UnexpectedToken` or `UnexpectedEOF`, with the line and column
it stopped at. Compare them with the rule that should cover that place.

- `UnexpectedEOF` on a text that looks right usually means a rule requires a final newline that is
  not there. `parse` adds exactly one, so add it in a reproduction too.
- A bare word claimed by the wrong value-type rule: `identifier` and `path` both match bare words.
  Reorder the alternatives of `?value`, or make the terminal stricter.
- An unexpected space: since whitespace is captured rather than ignored, every place the format
  allows a space needs a `ws*`.
- A blank or comment line refused: it needs an alternative of its own in the line rule.

## A key is missing from the dict

The tree is fine, but the reader never saw the entry: it looks for rules named `key_value`,
`key_list`, `key_block` and `key_null`, so check in the tree which name the entry has.

- A rule of another name needs the alias arrow: `... -> key_value`.
- A transparent rule (`?name:`) with a single child is inlined away by Lark, and its name never
  reaches the reader.
- `ValueError: No values found in Tree`: no child is a value-type rule of the registry. The value
  is written as a terminal, or with a rule whose name is not registered.
- `ValueError: No 'key' rule nodes found`: the key is not held by a rule named `key`.
- `TypeError: No key found.`: the first child of the `key` rule is a tree, not the key's token.
- A block whose entries are missing: the `key_block` has no child whose rule is one of the parser's
  `block_rules`.
- A block that reads but takes no new key: its rule is aliased `-> block` rather than named
  `block`, so it cannot be a start symbol, and a new entry cannot be parsed in it.

## The round trip differs

Compare the texts with a diff rather than by eye. Here a grammar discards the whitespace around
the `=` with `%ignore`, which the reconstructor then cannot write back:

```python
import difflib
from collections.abc import Mapping

from access.config.grammar_contract import KEY_VALUE, START_RULE
from access.config.parser import ConfigParser


class LeakyParser(ConfigParser):
    @property
    def grammar(self) -> str:
        return """
start: lines*
?lines: key_value | empty_line
key_value: key "=" value NEWLINE
?value: integer
empty_line: NEWLINE
%import config.key
%import config.integer
%import config.NEWLINE
%import common.WS_INLINE
%ignore WS_INLINE
"""

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        return {(START_RULE, KEY_VALUE): "{key} = {value}\n"}


text = "steps = 100\nlayout  =  4\n"
config = LeakyParser().parse(text)
print("".join(difflib.unified_diff(text.splitlines(True), str(config).splitlines(True), "file", "str(config)")))
```

```text
--- file
+++ str(config)
@@ -1,2 +1,2 @@
-steps = 100
-layout  =  4
+steps=100
+layout=4
```

- Missing spaces: the whitespace is `%ignore`d, or a `ws*` is missing from that place.
- Missing blank lines or comments: the line rule has no alternative for them, or `line_end` does
  not take a comment.
- A missing or extra final newline: `str` removes exactly the one `parse` added, so the grammar
  should neither consume nor write another.
- An entry written in *another rule's* syntax: two rules aliased to the same category have the
  same children once Lark has dropped their punctuation, and the reconstructor, which picks a rule
  by the children of a node, cannot tell them apart. Keep the punctuation that distinguishes them
  in a rule of its own, as `assign: "="` is in `nuopc_config`. Such a collision may show for some
  texts only: NUOPC's table lists were written in the top-level syntax with two elements, but not
  with three.

The reconstructor changes the tree it writes, which is why the store reconstructs a copy. Calling
`Reconstructor.reconstruct` in a script of one's own, copy the tree first, or what follows sees a
tree that has been changed.

## An edit is written in the wrong place

The parse and the round trip are fine, but an assignment changes the wrong text.

- `TypeError: Trying to change value type`: the new value fails the type check of the rule the old
  value was read with. Types are fixed by the parse, so check which rule really holds the value.
- The value is written as the first child of its value-type node, so a value-type rule with any
  other shape breaks quietly.
- `AssertionError` in `AddParent`: a tree node appears twice in the tree. Code building nodes has
  to make a new object for every position, never reuse one.

When the wrong node changes, look at the entry's references, held by the store of the
configuration the key is in:

```python
from access.config import FortranNMLParser

nml = FortranNMLParser().parse("&g\n  v = 1, 2\n  v(4) = 4\n/\n")
ref = nml["g"]._store.refs["V"]
print(ref.category, ref.values, len(ref.entry_nodes), len(ref.value_nodes))
```

```text
key_list (1, 2, None, 4) 2 4
```

`entry_nodes` are the entries writing the key, two here; `value_nodes` are the nodes of its
values, `None` where no entry writes a position; and `block_nodes` are the nodes of a block, more
than one where the block is written more than once.

## A key cannot be added, or looks wrong

`UnsupportedEntryError` means no candidate both parsed and read back as the value asked for. The
candidates can be listed:

```python
from access.config.entry_render import iter_entry_snippets
from access.config.entry_style import EntryStyle

group = nml["g"]
ctx = group._store.ctx
declared = ctx.entry_templates.get((str(group._store.tree.data), "key_value"))
for snippet in iter_entry_snippets("key_value", ("NEWKEY",), 1.5, EntryStyle(), declared, ctx.info.value_rules):
    print(repr(snippet))
```

```text
'NEWKEY = 1.5\n'
'NEWKEY = 1.5\n'
'NEWKEY = 1.5\n'
```

There is one candidate for each value-type rule the format has that accepts the value, here three
that happen to spell it alike. With the style read from the file,
`probe_entry_style(container, reconstructor)`, rather than an empty `EntryStyle()`, a copy of the
nearest entry of the kind comes first.

- No candidate at all: the parser declares no template for that kind of entry in that container,
  and the file has no entry of the kind to copy. The container rule is `group._store.tree.data`,
  `start` at the top level, and the keys of `entry_templates` have to name it.
- Candidates, all refused: they do not parse in that container, or they read back as another
  value. A bare word written for a `Path` reads back as an `identifier`, a `str`, and is rightly
  refused: the alternative is a file whose text and dict disagree.
- An entry formatted oddly: the donor is the nearest entry of its kind, copied whole, so a ragged
  neighbour gives a ragged entry; without one, the text is the parser's template.
- An entry in the wrong place: it goes one past the last entry of its container, then over the
  comment lines starting there, up to a blank line. A blank line is not a comment line, so a new key
  goes above it. A block created by an assignment takes its style from the nearest sibling block
  above it.

## Before finishing

Add the text that reproduced the problem as a test in `tests/test_<format>.py`, and run the whole
test suite: the grammars share `config.lark`, so a change made for one format can break another.
