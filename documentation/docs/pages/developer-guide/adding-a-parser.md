# Adding a parser

Do you need to read a configuration format that the package does not support? You can add a parser
for it. A parser is a subclass of `ConfigParser` that declares two things: a
[Lark](https://lark-parser.readthedocs.io/) grammar of the format, and the text of the first entry
of each kind that it can write. Everything else is inherited: reading the parse tree into a dict,
writing it back, editing values, adding keys and resizing lists. This works because the grammar
follows a naming contract. The shared code finds the entries and the values by the names of the
rules that hold them. In this tutorial we write a small parser, and then look at the contract and at
the steps for adding a parser to the package.

Note that a format that a library already reads and writes, as YAML and JSON are, does not need a
grammar. Wrap the library instead, as `yaml_config` and `json_config` do, and give the wrapper the
same `parse` and `str`.

## A first parser

We start with a complete parser, for files of `key = value` lines with `#` comments:

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

Like any of the parsers of the package, it reads a file, writes it back, edits it and adds keys. Let
us try it on a short text:

```python
text = "# run settings\nsteps = 100  # per run\nname = 'test'\n"
config = KeyValueParser().parse(text)
print(dict(config), str(config) == text)

config["steps"] = 200
config["dt"] = 1800.0
config["layout"] = [4, 2]
print(str(config))
```

This prints the dict and the result of the round trip, and then the text after our edits:

```text
{'steps': 100, 'name': 'test'} True
# run settings
steps = 200  # per run
name = 'test'
dt = 1800.0
layout = 4, 2
```

It also refuses what its grammar cannot write, and says what it can write. Our grammar has no
logical values, so we try to add one:

```python
from access.config import UnsupportedEntryError

try:
    config["restart"] = True
except UnsupportedEntryError as error:
    print(error)
```

This prints:

```text
This format cannot store True as 'restart' here: there is no way to write a key_value in 'start' with this value (value types this format writes: ['float', 'integer', 'string'])
```

## The grammar contract

The shared code looks for the names below. They are defined once, in
[`grammar_contract`](../api/internals/grammar.md#access.config.grammar_contract). A parser imports
its constants (`START_RULE`, `KEY_VALUE` and the rest) instead of spelling the names out.

- **The entries.** Every assignment is a rule that is named after its category, or aliased to it
  with `->`. The categories are `key_value` for one value, `key_list` for two or more, `key_block`
  for a block of entries, and `key_null` for a key with no value. Aliasing is the normal case,
  because a format may write one category in several ways: `rfile_key_value: ws* key ":" ws* value
  line_end -> key_value`.
- **The key.** The rule that holds a key is named `key`, and its first child is the token with the
  text of the key. `config.lark` has a `key` rule that suits most formats. A format whose keys can
  contain other characters defines its own, as `field_table` does for keys with hyphens and slashes.
  This rule alone decides which names are allowed, because a new key is checked against it.
- **The values.** A scalar is written with one of the *value-type rules* registered in
  `VALUE_TYPE_HANDLER_REGISTRY`, each with a single token. These are `integer`, `float`, `double`,
  `complex`, `logical`, `bool`, `string`, `fortran_string`, `identifier` and `path`, together with
  the more particular ones that the formats of the package need. They are all in `config.lark`. A
  rule with any other name is not a value. A format can define a value-type rule of its own, under a
  registered name, as `fortran_nml` does.
- **The blocks.** A `key_block` holds its entries in a rule *named* `block`, and not in one aliased
  to it, because a new entry of a block is parsed with `block` as the start symbol.
- **The start rule.** The top-level rule is named `start`, and it is not transparent: write
  `start:`, never `?start:`. A transparent rule with one child is inlined away, so a file with a
  single entry would have no container for a new entry to go in.
- **Whitespace.** Whitespace that has to survive a round trip is captured in the `ws` rule, and
  never discarded with `%ignore`. The whitespace before an `=` goes into a rule of its own, together
  with the `=`. This is the `eq` rule of our parser. The reason is that Lark drops anonymous
  literals from the tree. With `ws* "=" ws*`, the tree would record which spaces exist, but not on
  which side of the `=` each one was. The reconstructor would then move them, and write `A= 1` back
  as `A =1`.
- **Comments and blank lines** need an alternative of their own in the line rule, otherwise the
  reconstruction drops them. In our parser this is `empty_line`, with `comment` (`#`) from
  `config.lark`. For `!` comments, use `fortran_comment`.
- **Distinguishable rules.** Two rules aliased to the same category must have different children.
  The reconstructor picks the rule that it writes a node with by the children of the node, after
  Lark has dropped the anonymous punctuation. Punctuation that tells two syntaxes apart is therefore
  kept in a rule of its own, as `assign: "="` is in `nuopc_config`.

`parse` adds a newline to the text, so a grammar can assume that every line ends with one. Terminals
and rules are imported from `config.lark` one at a time, with `%import config.<name>`. Lark renames
an imported terminal (`CNAME` becomes `config__CNAME`), which only matters to code that looks at
`token.type`.

The order of the alternatives of `?value` decides how a value is *read*. Put the more specific rules
first, because `identifier` and `path`, for instance, both match bare words. How a *new* value is
written is decided separately, by the order of `VALUE_TYPE_HANDLER_REGISTRY`, among the rules that
the grammar admits. A format with `logical` therefore writes `.true.` and one with `bool` writes
`True`, with nothing for the parser to do. A parser cannot change that order. If a format insists on
a particular spelling, put it in a template or in a value-type rule of its own.

## Templates of new entries

A new entry is written as a copy of the nearest entry of its kind. The first entry of a kind has
nothing to copy, so it is written from the `entry_templates` of the parser. This property maps a
container rule and an entry category to the text of the entry, with markers:

- `{key}` for the key;
- `{keys:SEP}` for every key of a chain, separated by `SEP`. The Fortran `a%b = 1` is `{keys:%} =
  {value}`, which is one template for a key path of any depth;
- `{value}` for a value. The template of a list has two, and the text between them is the separator,
  which is repeated for longer lists.

The whitespace at the start of a template is its own indentation. Where the file has an entry above,
the indentation of that entry replaces it. For example, the NUOPC template of a table row starts
with a single space before `{key} = {value}`, and the first row of a table is written with it. A
block is written empty, as `"{key}%\n%{key}\n"` in MOM6, and its entries are added to it afterwards.

Declare a template for every kind of entry that the grammar can write. A kind with no template and
no entry to copy raises `UnsupportedEntryError`. The tests check that the parsers of the package
declare one for every kind.

## The other properties

Three more properties have defaults. Override them where the defaults do not fit:

- `key_case`: `"keep"`, the default, for a format whose keys are case-sensitive. Use `"upper"` or
  `"lower"` for a format whose reader matches names in any case, as Fortran (`"upper"`) and the FMS
  field manager (`"lower"`) do.
- `block_rules`: the rules that hold the entries of a block, `("block",)` by default. A format with
  blocks written with one component on each line, such as the derived types of Fortran, also lists
  the rule that holds them. A key added to such a block is written as a line that carries its whole
  key path.
- `repeated_blocks`: what a block written twice means. With `"merge"`, the default, it is read as
  one block that holds both. With `"separate"`, each is kept as a block of its own, in a
  `ConfigBlockList`. This is for a format whose reader stops at the first block of a name, as the
  `READ` of Fortran does.

## Steps

1. **Collect real files first.** The product is an exact round trip. A grammar written from a
   description of the format misses the comment styles, continuation lines and stray whitespace that
   real files have. Keep samples of them as test fixtures.
2. **Write the module**, `src/access/config/<format>.py`, with the licence header, a docstring
   describing the format and citing its specification, and the parser class. `nuopc_config` sets
   the standard for the docstring.
3. **Export the parser** from `access/config/__init__.py`, in its imports and in `__all__`. Add it
   to the formats listed in the README and in the *Repository layout* section of `AGENTS.md`. In
   these pages, give it a page of its own under *Configuration files*, a row in the tables of the
   [home page](../index.md) and of [Reading and writing](../user-guide/reading-and-writing.md), and
   a `:::` block on the [Parsers](../api/parsers.md) page of the reference.
4. **Write its tests**, `tests/test_<format>.py`, which take the module to 100% coverage on their
   own:
    - a round trip of each sample file, and the dict it reads as;
    - edits of a scalar, an element of a list and a value in a block, asserting the whole text;
    - the addition of a key of every category that the format writes, at the top level and in a
      block, asserting the whole text. This pins the indentation, the separators and where the entry
      lands;
    - growing and shrinking a list;
    - the errors: malformed text, a change of type, a list too short, assigning to a block, and
      every value type or category the format cannot write;
    - a byte-exact test for each value type that the grammar admits. Value handlers are chosen by
      rule name, so an untested type is an untested code path;
    - the format's rows in `CANONICAL` (and `CHAIN_CANONICAL`, for chains) in `tests/conftest.py`,
      one per template it declares.

    Pair each byte-exact assertion with a check that the text parses back to the same dict. Several
    value types share a Python type, so a value can be written with exactly the expected text and
    still be read back as another type.

5. **Run the checks** in [Contributing](contributing.md#before-committing).

If the grammar does not parse a file, or a round trip differs, work through [Debugging a
grammar](debugging-grammars.md). Do not change the grammar by trial and error until it works.
