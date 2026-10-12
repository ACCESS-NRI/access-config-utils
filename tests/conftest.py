# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for the parser tests.

Three things are shared.

The **fakes and node builders** are what lets a test file isolate its own module. Every
module's test file fakes the collaborators it imports from elsewhere in ``access.config``,
and never fakes anything from lark: ``Tree`` and ``Token`` are built for real here, because
handing a function its input is not mocking it, and it is what makes the awkward branches
reachable. ``test_parser.py`` is the one exception -- it drives the whole stack on purpose.

The **shipped grammars** are compiled once, for the five modules whose collaborator is a
lark object and so cannot be faked.

The **toy grammar** is what ``test_parser.py`` drives and what those five use when any real
grammar will do. It is deliberately unlike any real format: it exercises every value type at
once, every entry category, and three rules aliased to the wrong category.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pytest
from lark import Token, Tree
from lark.exceptions import UnexpectedInput

from access.config.entry_render import iter_entry_snippets
from access.config.entry_style import EntryStyle
from access.config.field_table import FieldTableParser
from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_compiled import compile_grammar
from access.config.grammar_values import UnsupportedEntryError
from access.config.mom6_input import MOM6InputParser
from access.config.nuopc_config import NUOPCParser
from access.config.nuopc_runseq import NUOPCRunSeqParser
from access.config.parser import ConfigParser
from access.config.tree_reader import EntryRef

PARSERS = {
    "fortran_nml": FortranNMLParser,
    "mom6_input": MOM6InputParser,
    "nuopc_config": NUOPCParser,
    "nuopc_runseq": NUOPCRunSeqParser,
    "field_table": FieldTableParser,
}

# The entry text each parser writes for each container and category it supports, with no
# neighbouring entry to copy a style from: what it declares in ``entry_templates``. A
# combination absent from this table is one the parser declares nothing for.
CANONICAL = {
    ("nuopc_config", "start", "key_value"): "NEWKEY: 7\n",
    ("nuopc_config", "start", "key_list"): "NEWKEY: 1 2 3\n",
    ("nuopc_config", "start", "key_block"): "NEWKEY::\n::\n",
    ("nuopc_config", "start", "key_null"): "NEWKEY:\n",
    ("nuopc_config", "block", "key_value"): " NEWKEY = 7\n",
    ("nuopc_config", "block", "key_list"): " NEWKEY = 1:2:3\n",
    ("nuopc_config", "block", "key_null"): " NEWKEY =\n",
    ("mom6_input", "start", "key_value"): "NEWKEY = 7\n",
    ("mom6_input", "start", "key_list"): "NEWKEY = 1, 2, 3\n",
    ("mom6_input", "start", "key_block"): "NEWKEY%\n%NEWKEY\n",
    ("mom6_input", "block", "key_value"): "NEWKEY = 7\n",
    ("mom6_input", "block", "key_list"): "NEWKEY = 1, 2, 3\n",
    ("fortran_nml", "start", "key_block"): "&NEWKEY\n/\n",
    ("fortran_nml", "block", "key_value"): "NEWKEY = 7\n",
    ("fortran_nml", "block", "key_list"): "NEWKEY = 1, 2, 3\n",
    ("fortran_nml", "block", "key_null"): "NEWKEY =\n",
    ("nuopc_runseq", "start", "key_value"): "NEWKEY::\nATM\n::\n",
    ("nuopc_runseq", "start", "key_list"): "NEWKEY::\n@3600\nATM\n@\n::\n",
    ("nuopc_runseq", "start", "key_null"): "NEWKEY::\n::\n",
    # A method line starts with its line break, and is indented.
    ("field_table", "block", "key_value"): '\n  "NEWKEY", "7"',
    ("field_table", "block", "key_list"): "\n  NEWKEY = 1, 2, 3",
}

# The same for a chain of keys, keyed by how many keys there are: the Fortran ``a%b = 1``
# is a block for ``a`` holding an entry for ``b``. Written with the keys "A", "B" and "C".
# Absent combinations are again ones the grammar cannot express.
CHAIN_CANONICAL = {
    ("fortran_nml", "block", "key_value", 2): "A%B = 7\n",
    ("fortran_nml", "block", "key_list", 2): "A%B = 1, 2, 3\n",
    ("fortran_nml", "block", "key_null", 2): "A%B =\n",
    ("fortran_nml", "block", "key_value", 3): "A%B%C = 7\n",
    # A field table header: a field type, a model and a field, opening a block of methods.
    ("field_table", "start", "key_block", 3): '"A", "B", "C"/\n',
}

VALUES = {"key_value": 7, "key_list": [1, 2, 3], "key_null": None, "key_block": {}}

# Values for a format whose only value type is a line of text, which holds no number.
TEXT_VALUES = {"key_value": "ATM", "key_list": ["@3600", "ATM", "@"], "key_null": None}


def canonical_rows(grammar: str) -> list[tuple[str, str, str]]:
    """Return ``(container, category, expected)`` for one grammar's rows of ``CANONICAL``.

    Each format's test module parametrises over its own rows, so the table stays stated once
    while the assertions live with the format they describe.
    """
    return [(container, category, text) for (name, container, category), text in CANONICAL.items() if name == grammar]


def chain_rows(grammar: str) -> list[tuple[str, str, int, str]]:
    """Return ``(container, category, depth, expected)`` for one grammar's chain rows."""
    return [
        (container, category, depth, text)
        for (name, container, category, depth), text in CHAIN_CANONICAL.items()
        if name == grammar
    ]


def assert_canonical_entry(
    parser: ConfigParser,
    container: str,
    category: str,
    expected: str,
    values: Mapping[str, Any] = VALUES,
    keys: Sequence[str] = ("NEWKEY",),
) -> None:
    """Assert the entry text a parser writes for a kind of entry the file has none of.

    The first candidate that parses in that container has to be the expected text, which
    pins the template the parser declares, the choice of a value type and the rendering
    against a real grammar. A candidate that does not parse is passed over, as the caller
    passes it over.

    Args:
        parser: The parser, which declares the template and supplies the grammar.
        container: Name of the container rule the entry is written in.
        category: The entry category, of the innermost entry for a chain.
        expected: The exact text expected.
        values: The value to write per category, for a format that cannot hold the usual.
        keys: The key path, one key per entry of the chain.
    """
    compiled = compile_grammar(parser.grammar)
    candidates = iter_entry_snippets(
        category,
        keys,
        values[category],
        EntryStyle(),
        parser.entry_templates.get((container, category)),
        compiled.info.value_rules,
        tuple(parser.value_rule_priority),
    )
    for text in candidates:
        try:
            parsed = compiled.lark.parse(text, start=container)
        except UnexpectedInput:
            continue
        # The text must yield that category.
        categories = {node.data for node in parsed.iter_subtrees()} | {parsed.data}
        assert category in categories
        assert text == expected
        return
    pytest.fail("No candidate parses")


TOY_GRAMMAR = """
    // Made-up grammar taylored to test the different building blocks
    // of the configuration parsers.
    start: (key_block|outer_key_value|outer_key_list|key_null|wrong_key_value1|wrong_key_value2|wrong_key_value3)+

    key_null: key equal
    outer_key_value: key equal value -> key_value
    outer_key_list: key equal value ("," value)* -> key_list
    equal: "="
    key_block: key "<" block ">"
    block: (block_key_value| block_key_list)*
    block_key_value: key colon value -> key_value
    block_key_list: key colon value ("|" value)* -> key_list
    colon: ":"
 
    // This rule will trip the interpreter because there is more than one value
    // (the alias should therefore be key_list)
    ?wrong_key_value1:  key equal value (":" value)* -> key_value

    // This rule will trip the interpreter because there is no value
    // (the alias should therefore be key_null)
    ?wrong_key_value2:  key colon equal -> key_value

    // This rule will trip the interpreter because there is no "key" child node
    ?wrong_key_value3:  "!" value -> key_value

    ?value: logical
         | bool
         | integer
         | float
         | double
         | complex
         | double_complex
         | identifier
         | string
         | path

    %import config.key
    %import config.logical
    %import config.bool
    %import config.integer
    %import config.float
    %import config.double
    %import config.complex
    %import config.double_complex
    %import config.identifier
    %import config.string
    %import config.path

    %import common.WS
    %ignore WS
"""


# What the toy grammar writes for each kind of entry it has: whitespace is discarded, so
# the entries are written tight.
TOY_TEMPLATES = {
    ("start", "key_null"): "{key}=",
    ("start", "key_value"): "{key}={value}",
    ("start", "key_list"): "{key}={value},{value}",
    ("start", "key_block"): "{key}<>",
    ("block", "key_value"): "{key}:{value}",
    ("block", "key_list"): "{key}:{value}|{value}",
}


class Parser(ConfigParser):
    """Parser using the grammar defined above, with case sensitive keys."""

    @property
    def case_sensitive_keys(self) -> bool:
        return True

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        return TOY_TEMPLATES

    @property
    def grammar(self) -> str:
        return TOY_GRAMMAR


@pytest.fixture(scope="module")
def parser():
    """Fixture instantiating the parser"""
    return Parser()


class CaseParser(ConfigParser):
    """Parser using the grammar defined above, with case insensitive keys."""

    @property
    def case_sensitive_keys(self) -> bool:
        return False

    @property
    def entry_templates(self) -> Mapping[tuple[str, str], str]:
        return TOY_TEMPLATES

    @property
    def grammar(self) -> str:
        return TOY_GRAMMAR


@pytest.fixture(scope="module")
def case_parser():
    return CaseParser()


# --- Real lark nodes, built directly ---
#
# Not fakes: these are the classes the parser itself produces. Building one is constructing
# the input to the function under test, which is what lets a module be exercised without
# compiling a grammar to reach the shape wanted.


def value_node(rule: str = "integer", text: str = "1") -> Tree:
    """Return a value-type rule node: a rule ``Tree`` over a single ``Token``."""
    return Tree(rule, [Token("VALUE", text)])


def key_node(name: str = "a") -> Tree:
    """Return a ``key`` rule node holding *name*."""
    return Tree("key", [Token("CNAME", name)])


def entry_node(category: str, key: str = "a", *children: Tree | Token) -> Tree:
    """Return an entry rule node: a ``key`` child followed by whatever the entry holds.

    Args:
        category: One of ``ENTRY_CATEGORIES``, used as the node's rule name.
        key: The key name.
        *children: The rest of the entry -- value nodes, a ``block``, whitespace.
    """
    return Tree(category, [key_node(key), *children])


def ws_node(text: str = " ") -> Tree:
    """Return a ``ws`` rule node holding *text*."""
    return Tree("ws", [Token("WS_INLINE", text)])


# --- Fakes for collaborators inside the package ---
#
# Hand-rolled rather than ``Mock(spec=...)``: the house style declares instance state as
# bare annotations, so those names are not class attributes and ``spec`` rejects every one
# of them. Each fake records what it was asked to do, so a test can assert the arguments it
# received rather than a call count.


class RepetitionInfo:
    """Stands in for ``GrammarInfo``, reporting every rule a plain repetition.

    That is what a namelist group's block is, so a deletion inside a fake nested
    configuration stays there, as it does inside a real group. It has no value type, unless
    a test gives it some.
    """

    value_rules: frozenset[str] = frozenset()

    def is_repetition_rule(self, name: str) -> bool:
        """Report that *name* is a plain repetition."""
        return True


class FakeContext:
    """Stands in for a ``ParseContext`` where only its settings are needed."""

    def __init__(
        self,
        *,
        case_sensitive: bool = True,
        info: Any = None,
        lark: Any = None,
        block_rules: tuple[str, ...] = ("block",),
        repeated_blocks: str = "merge",
        key_case: str = "upper",
    ) -> None:
        self.case_sensitive_keys = case_sensitive
        self.key_case = key_case
        self.info = info if info is not None else RepetitionInfo()
        self.lark = lark
        self.entry_templates: dict[tuple[str, str], str] = {}
        self.value_rule_priority: tuple[str, ...] = ()
        self.block_rules = block_rules
        self.repeated_blocks = repeated_blocks

    def accepts_key(self, text: str) -> bool:
        """Accept the names a grammar using ``config.key`` would: identifiers."""
        return text.isidentifier()

    def normalise_key(self, key: str) -> str:
        """Normalise as a real ``ParseContext`` would."""
        if self.case_sensitive_keys:
            return key
        return key.lower() if self.key_case == "lower" else key.upper()


class FakeStore:
    """Stands in for a ``ConfigStore``, recording what ``Config`` asks of it.

    Args:
        refs: The entries the store reports, keyed by normalised key.
        ctx: The context to expose; a case-sensitive ``FakeContext`` by default.
        added: References ``add_path`` should return, keyed by key, or by key path for a
            chain, for a test adding one.
        refused: Key paths ``add_path`` refuses, as a format with no syntax for them would.
        text: What ``render`` returns.
        reread: References ``reread`` should report, as though the tree had changed;
            the ones already held when not given.
    """

    def __init__(
        self,
        refs: dict[str, EntryRef] | None = None,
        *,
        ctx: Any = None,
        added: dict[Any, EntryRef] | None = None,
        refused: Collection[tuple[str, ...]] = (),
        text: str = "<rendered>",
        reread: dict[str, EntryRef] | None = None,
    ) -> None:
        self.refs = refs if refs is not None else {}
        self.ctx = ctx if ctx is not None else FakeContext()
        self.fallback_style = EntryStyle()
        self.tree = Tree("block", [])
        self.addable = True
        self.calls: list[tuple[str, Any]] = []
        self._added = added or {}
        self._refused = set(refused)
        self._text = text
        self._reread = reread

    def child(self, ref: EntryRef, index: int = 0) -> "FakeStore":
        """Return a store over the occurrence of the block *ref* names."""
        self.calls.append(("child", (ref, index) if index else ref))
        return FakeStore(ctx=self.ctx)

    def replace(self, key: str, value: Any, origins: Any = None) -> None:
        """Record the replacement, and the old element each new value stands in for."""
        call = (key, value)
        self.calls.append(("replace", call if origins is None else (*call, list(origins))))

    def add_path(self, keys: Sequence[str], raw_keys: Sequence[str], value: Any, category: Any = None) -> EntryRef:
        """Record the addition and return the reference declared for it, or refuse it.

        The category is recorded only when one is given, as it is for a record.
        """
        call = (tuple(keys), tuple(raw_keys), value)
        self.calls.append(("add_path", call if category is None else (*call, category)))
        if tuple(keys) in self._refused:
            raise UnsupportedEntryError(f"Refused {list(keys)}")
        self.refs[keys[0]] = self._added[keys[0] if len(keys) == 1 else tuple(keys)]
        return self.refs[keys[0]]

    def created_block_style(self, entry: Tree | None = None) -> EntryStyle:
        """Return the style a block just created should inherit."""
        self.calls.append(("created_block_style", entry))
        return EntryStyle(indent="  ")

    def remove(self, key: str) -> None:
        """Record the removal and forget the reference."""
        self.calls.append(("remove", key))
        del self.refs[key]

    def reread(self, ref: EntryRef | None = None) -> dict[str, EntryRef]:
        """Record the request, and the reference the store was pointed at, if any."""
        self.calls.append(("reread", ref))
        if self._reread is not None:
            self.refs = self._reread
        return self.refs

    def render(self) -> str:
        """Return the canned text."""
        self.calls.append(("render", None))
        return self._text

    def asked(self, name: str) -> list[Any]:
        """Return the arguments of every call to *name*, in order."""
        return [args for called, args in self.calls if called == name]
