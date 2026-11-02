---
name: add-parser
description: Add a round-trip parser for a new configuration file format to access-config-utils. Use when asked to support a new config file type (a new namelist dialect, resource file, key-value format, …), when writing or extending a Lark grammar under src/access/config/, or when a new ConfigParser subclass is needed.
---

# Adding a configuration file parser

A parser is a `ConfigParser` subclass supplying a Lark `grammar` string and the text of the first
entry of each kind it can write (`entry_templates`). Three more properties have defaults, to
override where they do not fit: `key_case`, `block_rules` and `repeated_blocks` (see [Key
addition](#key-addition)). Everything else — dict construction, round-trip reconstruction,
in-place editing, key addition, list resizing — is inherited. The cost of that leverage is that the
grammar must follow an exact naming contract, because `EntryReader` (a Lark `Interpreter`)
dispatches on rule names.

A format a standard library already reads and writes (YAML, JSON) needs none of this: wrap the
library instead, as `yaml_config.py` and `json_config.py` do.

## Steps

1. **Collect real sample files first.** Round-trip fidelity is the product; a grammar written from a
   format description will miss the comment styles, continuation lines and stray whitespace that
   appear in practice. Put representative samples into the test module as fixtures.

2. **Create `src/access/config/<format>.py`** with the Apache-2.0 header, a module docstring
   describing the format (cite the spec/URL — see `nuopc_config.py` for the standard set by this
   repo), and the subclass:

   ```python
   class MyFormatParser(ConfigParser):
       @property
       def grammar(self) -> str:
           return """
   start: lines*
   ...
   """

       @property
       def entry_templates(self) -> Mapping[tuple[str, str], str]:
           return {("start", "key_value"): "{key} = {value}\n"}
   ```

3. **Write the grammar against the naming contract** (see below).

4. **Export it** from `src/access/config/__init__.py` (the import and the `__all__` entry), and
   list it among the supported formats in `README.md` and in the repository layout of `AGENTS.md`.

5. **Write `tests/test_<format>.py`** as described under [Tests](#tests). It must take the new
   module to 100% coverage on its own.

6. Run the gate in [`skills/check/SKILL.md`](../check/SKILL.md).

## The grammar contract

`EntryReader` has callbacks named `key_value`, `key_list`, `key_block` and `key_null`. A rule only
gets visited if it carries one of those names, so use Lark's alias arrow when the natural rule name
differs:

```lark
rfile_key_value: ws* key ":" ws* value line_end -> key_value
```

Requirements:

- **`key`** — import from `config.lark`, or define a rule named `key` when the format's keys need
  another terminal (`field_table.py` does, for keys holding hyphens and slashes). Its first child
  must be the `Token` holding the key text.
- **`key_value`** — exactly one value-type rule node among its children.
- **`key_list`** — two or more value-type rule nodes.
- **`key_block`** — must contain a child whose rule is one of `block_rules` (just `block` unless
  the parser declares more), which holds the nested assignments. The nested `Config` is built from
  that node. The first of `block_rules` must be *named* `block`; do not alias another rule to it
  with `-> block`. `compile_grammar` adds `block` as a start symbol only for a rule of that name,
  and key addition parses the text of a new entry with it.
- **`key_null`** — for valueless assignments (`a=`). Carries no value node.
- **`start`** — must be named `start` and must **not** be transparent: write `start:`, never
  `?start:`. A transparent rule collapses when it has a single child, so a file with one entry would
  have that entry as the root of the tree, leaving no container for key addition to add to. This also
  removes a whole class of single-entry-only bugs.
- **Values** — only value-type rules whose names are keys of `VALUE_TYPE_HANDLER_REGISTRY`; a
  rule under any other name will not be recognised as a value. Import them from `config.lark`
  (`logical`, `bool`, `integer`, `float`, `double`, `complex`, `double_complex`, `identifier`,
  `string`, `fortran_string`, `path`, and the more particular ones: `text_line` for a whole line
  of free text, `inferred_string`/`inferred_text` for a value typed by its spelling as the FMS
  field manager types one, `named_assignments` for a record of a name and parameters). A format
  may define one itself under a registry name, with exactly one `Token` child, as `fortran_nml.py`
  does for `bare_exponent` and `fortran_identifier`. Collect the subset your format allows into a
  transparent `?value:` rule. A value type whose handler `collects` makes an entry one record of a
  list the key holds, an entry per record.

Order the alternatives in `?value` from most specific to least: `identifier` and `path` both match
bare words, so a plain token can be claimed by the wrong rule if the order is wrong. Check what the
existing grammars do before inventing an order.

The order in `?value` governs *parsing*. Which rule a **newly added** value is written as is decided
separately, by the order of `VALUE_TYPE_HANDLER_REGISTRY` in `grammar_values.py`, filtered down to
the rules your grammar admits. So a format with `logical` writes `.true.` while one with `bool`
writes `True`, with no work on your part. Add a byte-exact test per value type your format admits,
since those tests are what pin the choice. A parser cannot reorder it; a spelling a format insists
on belongs in a template, or in a value-type rule of its own.

## Key addition

A new entry is written as a copy of the nearest entry of its kind in the file. The first entry of a
kind has nothing to copy, so **declare the text of one for every kind of entry your grammar can
write**, keyed by container rule name and entry category:

```python
@property
def entry_templates(self) -> Mapping[tuple[str, str], str]:
    return {
        ("start", "key_value"): "{key} = {value}\n",
        ("start", "key_list"): "{key} = {value}, {value}\n",
        ("start", "key_block"): "{key}<\n>\n",
    }
```

Markers are `{key}` (one key), `{keys:SEP}` (every key of a chain, parted by SEP: the Fortran
`a%b = 1` is `{keys:%} = {value}`, one template for any depth) and `{value}`. A `key_list` template
needs exactly two `{value}` markers: what lies between them is the element separator, and it is
repeated for longer lists. A block is written empty and its contents are added afterwards. The
whitespace the first line starts with is the template's own indentation, which that of the entry
above replaces where the file shows one (NUOPC writes a table row as ` {key} = {value}\n`). The text
is parsed back with the enclosing rule as the start symbol and read back, and only a candidate that
means what was asked for is spliced in; a kind with no template and no entry to copy raises
`UnsupportedEntryError`. Satisfy the `start` and `block` requirements above. A new key has to be a
name the grammar's own `key` rule reads back as itself (`ParseContext.accepts_key`), so a `key`
terminal admitting hyphens or slashes is all it takes to allow them.

A format whose keys are not case-sensitive sets `key_case`: `"keep"` is the default (two spellings
are two keys), `"upper"` is what a Fortran namelist does, and `"lower"` is for a format whose own
reader lower-cases names. A format with more than one kind of block lists them in `block_rules`, the
one new entries are written into first; a block held by any other has no body of its own, and a key
added to it is written as a line carrying the whole key path (a Fortran `a%b = 1`, a field table
header).

A block written twice is merged into one by default (`repeated_blocks = "merge"`). A format whose
own reader stops at the first block of a name, as Fortran's `READ` does, sets `"separate"`: each
occurrence is then kept and handed out as its own block, in a `ConfigBlockList`.

## Whitespace and round-tripping

**Do not `%ignore WS`** for whitespace that must be reproduced. Import the explicit `ws` rule from
`config.lark` and place `ws*` where the format allows spacing:

```lark
key_value: ws* key ws* "=" ws* value
line_end: (comment|ws*) NEWLINE
empty_line: line_end
```

Blank lines and comment-only lines need their own alternative in the line rule, or they will be
dropped from the reconstruction. `config.lark` provides both `comment` (`#`) and `fortran_comment`
(`!`).

`ConfigParser.parse` appends `"\n"` to the input before parsing, so the grammar may assume every
line — including the last — ends with `NEWLINE`. `Config.__str__` strips that newline back off.

## Imports from config.lark

Import each terminal and rule you use explicitly; there is no wildcard:

```lark
%import config.key
%import config.integer
%import config.ws
%import config.NEWLINE
```

Lark prefixes imported terminals internally (`CNAME` → `config__CNAME`). This is invisible for
parsing and reconstruction, but matters to any code that inspects `token.type`.

## Tests

Mirror `tests/test_fortran_nml.py` / `tests/test_nuopc_config.py`:

- a module-scoped `parser` fixture,
- a fixture holding the expected `dict` for a sample file,
- **round-trip test**: `assert str(parser.parse(text)) == text`,
- **content test**: the parsed `Config` compares equal to the expected dict,
- **edit test**: change a scalar, a list element, and a nested block value; assert the output text
  differs only where expected,
- **addition test**: add a key of every category the format supports (scalar, list, block, and
  valueless if the grammar has `key_null`), at the top level and inside a block, and assert the whole
  file byte-exactly against a fixture — that is what pins indentation, the separator chosen for each
  container, and where the new entry lands relative to trailing blank lines and block terminators,
- **resize test**: grow and shrink a list, byte-exactly — the elements that stay keep their text and
  a new one is written like its neighbours,
- **error tests**: malformed input raises the relevant `lark.exceptions` error; changing a value's
  type raises `TypeError`; shrinking a list below two elements, or resizing one written with
  repeat counts or array qualifiers, raises `ValueError`; assigning to an **existing** block raises
  `SyntaxError`; and for each category and value type your format does *not* admit, adding it
  raises `UnsupportedEntryError`,
- for case-insensitive formats, assert keys are stored in the format's `key_case` and lookup works
  in any case,
- if the format sets `repeated_blocks = "separate"`, a file writing a block twice,
- add the format to `PARSERS` in `tests/conftest.py`, with its rows in `CANONICAL` (and
  `CHAIN_CANONICAL` for chains) — one per template it declares, which
  `test_a_shipped_parser_declares_a_template_for_every_entry_it_can_write` checks is all of them —
  and parametrise a `test_canonical_entry_text` over `canonical_rows("<format>")` (and over
  `chain_rows` for chains), asserting with `assert_canonical_entry`.

Type-annotate the tests (`-> None`) and give each a "Test that ..." docstring.

Cover every value type the grammar admits — the value handlers are selected by rule name, so an
untested type is an untested code path.

Pair every byte-exact assertion with `assert dict(parser.parse(str(config))) == dict(config)`. Text
alone does not prove the values survive: several value types share a Python type, so a value can be
written with exactly the text you expected and still read back as the wrong type.

## If it does not parse

Work through [`skills/grammar-debug/SKILL.md`](../grammar-debug/SKILL.md) rather than mutating the
grammar by trial and error.
