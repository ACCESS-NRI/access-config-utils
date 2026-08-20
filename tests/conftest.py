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

from typing import Any

import pytest
from lark import Lark, Token, Tree

from access.config.entry_render import iter_entry_snippets
from access.config.entry_style import EntryStyle
from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_compiled import compile_grammar
from access.config.grammar_info import GrammarInfo
from access.config.mom6_input import MOM6InputParser
from access.config.nuopc_config import NUOPCParser
from access.config.parser import ConfigParser
from access.config.tree_reader import EntryRef

PARSERS = {
    "fortran_nml": FortranNMLParser,
    "mom6_input": MOM6InputParser,
    "nuopc_config": NUOPCParser,
}

# The entry text each grammar produces for each container and category it supports, with
# no neighbouring entry to copy a style from. Every combination absent from this table is
# one the grammar cannot express, and is asserted empty by test_unsupported_combinations.
CANONICAL = {
    ("nuopc_config", "start", "key_value"): "NEWKEY: 7\n",
    ("nuopc_config", "start", "key_list"): "NEWKEY: 1 2 3\n",
    ("nuopc_config", "start", "key_block"): "NEWKEY::\n::\n",
    ("nuopc_config", "block", "key_value"): "NEWKEY = 7\n",
    ("nuopc_config", "block", "key_list"): "NEWKEY = 1:2:3\n",
    ("mom6_input", "start", "key_value"): "NEWKEY = 7\n",
    ("mom6_input", "start", "key_list"): "NEWKEY = 1, 2, 3\n",
    ("mom6_input", "start", "key_block"): "NEWKEY%\n%NEWKEY\n",
    ("mom6_input", "block", "key_value"): "NEWKEY = 7\n",
    ("mom6_input", "block", "key_list"): "NEWKEY = 1, 2, 3\n",
    ("fortran_nml", "start", "key_block"): "&NEWKEY\n/\n",
    ("fortran_nml", "block", "key_value"): "NEWKEY = 7\n",
    ("fortran_nml", "block", "key_list"): "NEWKEY = 1, 2, 3\n",
    ("fortran_nml", "block", "key_null"): "NEWKEY =\n",
}

VALUES = {"key_value": 7, "key_list": [1, 2, 3], "key_null": None, "key_block": {}}


def canonical_rows(grammar: str) -> list[tuple[str, str, str]]:
    """Return ``(container, category, expected)`` for one grammar's rows of ``CANONICAL``.

    Each format's test module parametrises over its own rows, so the table stays stated once
    while the assertions live with the format they describe.
    """
    return [(container, category, text) for (name, container, category), text in CANONICAL.items() if name == grammar]


def assert_canonical_entry(compiled: Any, container: str, category: str, expected: str) -> None:
    """Assert the entry text a grammar produces with no neighbour to copy a style from.

    The first candidate has to be both the expected text and valid in that container, so
    this pins the whole generate-rank-render pipeline against a real grammar.

    Args:
        compiled: The ``CompiledGrammar`` for the format.
        container: Name of the container rule the entry is generated for.
        category: The entry category.
        expected: The exact text expected.
    """
    snippets = iter_entry_snippets(compiled.info, container, category, "NEWKEY", VALUES[category], EntryStyle())
    first = next(iter(snippets))
    assert first == expected

    # The text must parse in the container it was generated for, and yield that category.
    parsed = compiled.lark.parse(first, start=container)
    categories = {node.data for node in parsed.iter_subtrees()} | {parsed.data}
    assert category in categories


@pytest.fixture(scope="session")
def larks() -> dict[str, Lark]:
    """Fixture returning a compiled parser per shipped grammar."""
    return {name: compile_grammar(cls().grammar).lark for name, cls in PARSERS.items()}


@pytest.fixture(scope="session")
def infos() -> dict[str, GrammarInfo]:
    """Fixture returning the grammar views per shipped grammar."""
    return {name: compile_grammar(cls().grammar).info for name, cls in PARSERS.items()}


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


class Parser(ConfigParser):
    """Parser using the grammar defined above, with case sensitive keys."""

    @property
    def case_sensitive_keys(self) -> bool:
        return True

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


class FakeContext:
    """Stands in for a ``ParseContext`` where only its settings are needed."""

    def __init__(
        self,
        *,
        case_sensitive: bool = True,
        info: Any = None,
        lark: Any = None,
        block_rules: tuple[str, ...] = ("block",),
    ) -> None:
        self.case_sensitive_keys = case_sensitive
        self.info = info
        self.lark = lark
        self.entry_templates: dict[tuple[str, str], str] = {}
        self.value_rule_priority: tuple[str, ...] = ()
        self.block_rules = block_rules

    def normalise_key(self, key: str) -> str:
        """Normalise as a real ``ParseContext`` would."""
        return key if self.case_sensitive_keys else key.upper()


class FakeStore:
    """Stands in for a ``ConfigStore``, recording what ``Config`` asks of it.

    Args:
        refs: The entries the store reports, keyed by normalised key.
        ctx: The context to expose; a case-sensitive ``FakeContext`` by default.
        added: References ``add`` should return, keyed by key, for a test adding one.
        text: What ``render`` returns.
    """

    def __init__(
        self,
        refs: dict[str, EntryRef] | None = None,
        *,
        ctx: Any = None,
        added: dict[str, EntryRef] | None = None,
        text: str = "<rendered>",
    ) -> None:
        self.refs = refs if refs is not None else {}
        self.ctx = ctx if ctx is not None else FakeContext()
        self.fallback_style = EntryStyle()
        self.calls: list[tuple[str, Any]] = []
        self._added = added or {}
        self._text = text

    def child(self, ref: EntryRef) -> "FakeStore":
        """Return a store over the block *ref* names."""
        self.calls.append(("child", ref))
        return FakeStore(ctx=self.ctx)

    def replace(self, key: str, value: Any) -> tuple[Tree, ...]:
        """Record the replacement and return the entry's value nodes."""
        self.calls.append(("replace", (key, value)))
        return self.refs[key].value_nodes

    def add(self, key: str, raw_key: str, value: Any) -> EntryRef:
        """Record the addition and return the reference declared for *key*."""
        self.calls.append(("add", (key, raw_key, value)))
        self.refs[key] = self._added[key]
        return self.refs[key]

    def created_block_style(self) -> EntryStyle:
        """Return the style a block just created should inherit."""
        self.calls.append(("created_block_style", None))
        return EntryStyle(indent="  ", has_donor=True)

    def remove(self, key: str) -> None:
        """Record the removal and forget the reference."""
        self.calls.append(("remove", key))
        del self.refs[key]

    def render(self) -> str:
        """Return the canned text."""
        self.calls.append(("render", None))
        return self._text

    def asked(self, name: str) -> list[Any]:
        """Return the arguments of every call to *name*, in order."""
        return [args for called, args in self.calls if called == name]
