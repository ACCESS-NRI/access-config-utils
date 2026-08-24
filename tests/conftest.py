# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the entry-synthesis tests.

The four modules covering synthesis all work against the three shipped grammars, so the
compiled parsers and their views are built once here. ``CANONICAL`` is shared for another
reason: it is the whole truth about what each grammar can express, so the module asserting
what gets *written* and the module asserting what gets *generated* have to agree on it.
"""

import pytest
from lark import Lark

from access.config.fortran_nml import FortranNMLParser
from access.config.grammar_compiled import compile_grammar
from access.config.grammar_info import GrammarInfo
from access.config.mom6_input import MOM6InputParser
from access.config.nuopc_config import NUOPCParser

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
