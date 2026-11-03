# Adding a parser

A parser for a new format is a subclass of `ConfigParser` that declares two things: a
[Lark](https://lark-parser.readthedocs.io/) grammar of the format, and the text of the first entry
of each kind it can write. Everything else is inherited: reading the parse tree into a dict,
writing it back, editing values, adding keys and resizing lists. What makes that possible is that
the grammar follows a naming contract, since the shared code finds entries and values by the names
of the rules that hold them.

A format that a library already reads and writes, as YAML and JSON are, does not need a grammar:
wrap the library instead, as `yaml_config` and `json_config` do, and give the wrapper the same
`parse` and `str`.

## A first parser

Here is a complete parser, for files of `key = value` lines with `#` comments:

```python
from collections.abc import Mapping

from access.config.grammar_contract import KEY_LIST, KEY_VALUE, START_RULE
from access.config.parser import ConfigParser


class KeyValueParser(ConfigParser):
    """Parser for files of ``key = value`` lines, with ``#`` comments."""

    @property
    def grammar(self) -> str:
        return """
start: lines*

?lines: key_value
      | key_list
      | empty_line

key_value: key eq ws* value line_end
key_list: key eq ws* value (ws* "," ws* value)+ line_end
eq: ws* "="

?value: integer
      | float
      | string

empty_line: line_end
line_end: (comment | ws*) NEWLINE

%import config.key
%import config.integer
%import config.float
%import config.string
%import config.comment
%import config.ws
%import config.NEWLINE
"""

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        return {
            (START_RULE, KEY_VALUE): "{key} = {value}\n",
            (START_RULE, KEY_LIST): "{key} = {value}, {value}\n",
        }
```

It reads, writes back and edits a file, and adds keys, like any of the shipped parsers:

```python
text = "# run settings\nsteps = 100  # per run\nname = 'test'\n"
config = KeyValueParser().parse(text)
print(dict(config), str(config) == text)

config["steps"] = 200
config["dt"] = 1800.0
config["layout"] = [4, 2]
print(str(config))
```

```text
{'steps': 100, 'name': 'test'} True
# run settings
steps = 200  # per run
name = 'test'
dt = 1800.0
layout = 4, 2
```

and it refuses what its grammar cannot write, naming what it can:

```python
from access.config import UnsupportedEntryError

try:
    config["restart"] = True
except UnsupportedEntryError as error:
    print(error)
```

```text
This format cannot store True as 'restart' here: there is no way to write a key_value in 'start' with this value (value types this format writes: ['float', 'integer', 'string'])
```

## The grammar contract

The names below are what the shared code looks for, and are defined once, in
[`grammar_contract`](../api/internals/grammar.md#access.config.grammar_contract), whose constants
(`START_RULE`, `KEY_VALUE` and the rest) a parser imports rather than spelling the names out.

- **The entries.** Every assignment is a rule named, or aliased with `->`, `key_value` for one
  value, `key_list` for two or more, `key_block` for a block of entries, or `key_null` for a key
  with no value. Aliasing is the normal case, since a format may write one category in several
  ways: `rfile_key_value: ws* key ":" ws* value line_end -> key_value`.
- **The key.** The rule holding a key is named `key`, and its first child is the token holding the
  key's text. `config.lark` has one that suits most formats; a format whose keys may hold other
  characters defines its own, as `field_table` does for keys with hyphens and slashes. Whether a
  new key is a usable name is asked of this rule, so it is all that decides which names are
  allowed.
- **The values.** A scalar is written with one of the *value-type rules* registered in
  `VALUE_TYPE_HANDLER_REGISTRY`, each with a single token: `integer`, `float`, `double`, `complex`,
  `logical`, `bool`, `string`, `fortran_string`, `identifier`, `path`, and the more particular
  ones the shipped formats need, all in `config.lark`. A rule of any other name is not a value. A
  format may define a value-type rule of its own, under a registered name, as `fortran_nml` does.
- **The blocks.** A `key_block` holds its entries in a rule *named* `block`, not one aliased to
  it, since a new entry of a block is parsed with `block` as the start symbol.
- **The start rule.** The top-level rule is named `start`, and is not transparent: `start:`,
  never `?start:`. A transparent rule with one child is inlined away, so a file with a single entry
  would have no container for a new entry to go in.
- **Whitespace.** Whitespace that has to survive a round trip is captured in the `ws` rule, never
  discarded with `%ignore`. The whitespace before an `=` goes into a rule of its own with the `=`,
  `eq` above: Lark drops anonymous literals from the tree, so with `ws* "=" ws*` it would record
  which spaces exist but not on which side of the `=` each was, and the reconstructor would move
  them, writing `A= 1` back as `A =1`.
- **Comments and blank lines** need an alternative of their own in the line rule, or the
  reconstruction drops them: `empty_line` above, with `comment` (`#`) or `fortran_comment` (`!`)
  from `config.lark`.
- **Distinguishable rules.** Two rules aliased to the same category must have different children.
  The reconstructor picks the rule it writes a node with by its children, after Lark has dropped
  the anonymous punctuation, so punctuation that tells two syntaxes apart is kept in a rule of its
  own, as `assign: "="` is in `nuopc_config`.

`parse` adds a newline to the text, so a grammar may assume that every line ends with one.
Terminals and rules are imported from `config.lark` one at a time, with `%import config.<name>`;
Lark renames an imported terminal (`CNAME` becomes `config__CNAME`), which matters only to code
that looks at `token.type`.

The order of the alternatives of `?value` decides how a value is *read*: put the more specific
rules first, since `identifier` and `path`, for instance, both match bare words. How a *new* value
is written is decided separately, by the order of `VALUE_TYPE_HANDLER_REGISTRY`, among the rules
the grammar admits. So a format with `logical` writes `.true.` and one with `bool` writes `True`,
with nothing for the parser to do. A parser cannot change that order; a spelling a format insists
on belongs in a template or in a value-type rule of its own.

## Templates of new entries

A new entry is written as a copy of the nearest entry of its kind, and the first of a kind, which
has nothing to copy, is written from the parser's `entry_templates`. It maps a container rule and
an entry category to the text of the entry, with markers:

- `{key}` for the key;
- `{keys:SEP}` for every key of a chain, parted by `SEP`: the Fortran `a%b = 1` is
  `{keys:%} = {value}`, one template for a key path of any depth;
- `{value}` for a value. A list's template has two, and the text between them is the separator
  repeated for longer lists.

The whitespace a template starts with is its own indentation, which the indentation of the entry
above replaces where the file has one: NUOPC's template of a table row starts with a single space
before `{key} = {value}`, which the first row of a table is written with. A block is written
empty, `"{key}%\n%{key}\n"` in MOM6, and its entries are added to it afterwards.

Declare a template for every kind of entry the grammar can write: a kind with no template and no
entry to copy raises `UnsupportedEntryError`. The tests check that the shipped parsers declare one
for every kind.

## The other properties

Three more properties have defaults, to be overridden where they do not fit:

- `key_case`: `"keep"`, the default, for a format whose keys are case-sensitive; `"upper"` or
  `"lower"` for one whose reader matches names in any case, as Fortran's (`"upper"`) and the FMS
  field manager's (`"lower"`) do.
- `block_rules`: the rules holding the entries of a block, `("block",)` by default. A format with
  blocks written one component per line, such as Fortran's derived types, lists the rule holding
  them too, and a key added to such a block is written as a line carrying its whole key path.
- `repeated_blocks`: what a block written twice means. `"merge"`, the default, reads one block
  holding both; `"separate"` keeps each as a block of its own, in a `ConfigBlockList`, for a format
  whose reader stops at the first block of a name, as Fortran's `READ` does.

## Steps

1. **Collect real files first.** The product is an exact round trip, and a grammar written from a
   description of the format misses the comment styles, continuation lines and stray whitespace
   real files have. Keep samples of them as test fixtures.
2. **Write the module**, `src/access/config/<format>.py`, with the licence header, a docstring
   describing the format and citing its specification, and the parser class. `nuopc_config` sets
   the standard for the docstring.
3. **Export the parser** from `access/config/__init__.py`, in its imports and `__all__`, and add it
   to the formats listed in the README, in the repository layout of `AGENTS.md`, and in these pages:
   a page of its own under *Configuration files*, a row in the tables of the
   [home page](../index.md) and of [Reading and writing](../user-guide/reading-and-writing.md), and
   a `:::` block on the [Parsers](../api/parsers.md) page of the reference.
4. **Write its tests**, `tests/test_<format>.py`, which take the module to 100% coverage on their
   own:
    - a round trip of each sample file, and the dict it reads as;
    - edits of a scalar, an element of a list and a value in a block, asserting the whole text;
    - the addition of a key of every category the format writes, at the top level and in a block,
      asserting the whole text, which pins the indentation, the separators and where the entry
      lands;
    - growing and shrinking a list;
    - the errors: malformed text, a change of type, a list too short, assigning to a block, and
      every value type or category the format cannot write;
    - a byte-exact test for each value type the grammar admits, since value handlers are chosen by
      rule name, and an untested type is an untested code path;
    - the format's rows in `CANONICAL` (and `CHAIN_CANONICAL`, for chains) in `tests/conftest.py`,
      one per template it declares.

    Pair each byte-exact assertion with a check that the text parses back to the same dict:
    several value types share a Python type, so a value can be written with exactly the text
    expected and still read back as another type.

5. **Run the checks** in [Contributing](contributing.md#before-committing).

If the grammar does not parse a file, or a round trip differs, work through
[Debugging a grammar](debugging-grammars.md) rather than changing the grammar until it does.
