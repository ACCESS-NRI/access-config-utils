# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for the parser tests.

Two things are shared. The modules covering entry synthesis all work against the three
shipped grammars, so the compiled parsers and their views are built once here, and
``CANONICAL`` -- the whole truth about what each grammar can express -- is stated once, so
that the module asserting what gets *written* and the module asserting what gets *generated*
cannot drift apart.

The rest is the made-up grammar the other parser tests run against. It is deliberately
unlike any real format: it exercises every value type at once, every entry category, and
three rules aliased to the wrong category, which is how the reader's own error paths are
reached.
"""

import pytest
from lark import Lark

from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_compiled import compile_grammar
from access.config.grammar_info import GrammarInfo
from access.config.mom6_input import MOM6InputParser
from access.config.nuopc_config import NUOPCParser
from access.config.parser import ConfigParser

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
