# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path

from access.config.parser import ConfigParser

grammar = """
    // Made-up grammar taylored to test the different building blocks
    // of the configuration parsers.
    start: (key_block|outer_key_value|outer_key_list|key_null|wrong_key_value1|wrong_key_value2)+

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
        return grammar


@pytest.fixture(scope="module")
def parser():
    """Fixture instantiating the parser"""
    return Parser()


def test_config_type_logical(parser):
    """Test transformation and sanity checks of logical values"""
    config = parser.parse("a=.True. b=.FALSE.")
    assert config["a"]
    assert not config["b"]

    config["a"] = False
    assert str(config) == "a=.false.b=.FALSE."

    with pytest.raises(TypeError):
        config["a"] = "z"


def test_config_type_bool(parser):
    """Test transformation and sanity checks of bool values"""
    config = parser.parse("a=True b=False")
    assert config["a"]
    assert not config["b"]

    config["a"] = False
    assert str(config) == "a=False b=False"

    with pytest.raises(TypeError):
        config["a"] = "z"


def test_config_type_integer(parser):
    """Test transformation and sanity checks of integer values"""
    config = parser.parse("a=1 b=-2")
    assert config["a"] == 1
    assert config["b"] == -2

    config["a"] = 2
    assert str(config) == "a=2 b=-2"

    with pytest.raises(TypeError):
        config["a"] = "z"


def test_config_type_float(parser):
    """Test transformation and sanity checks of float values"""
    config = parser.parse("a=1.0e-2 b=-2.0E2 c=10.0")
    assert config["a"] == 1.0e-2
    assert config["b"] == -2.0e2
    assert config["c"] == 10.0

    config["a"] = 10.0e20
    config["b"] = 2e-10
    config["c"] = -20.0
    assert str(config) == "a=1e+21 b=2E-10 c=-20.0"

    with pytest.raises(TypeError):
        config["a"] = "z"


def test_config_type_double(parser):
    """Test transformation and sanity checks of Fortran double values"""
    config = parser.parse("a=1.0d-2 b=-2.0D2")
    assert config["a"] == 1.0e-2
    assert config["b"] == -2.0e2

    config["a"] = 10.0e20
    config["b"] = -4.0e-20
    assert str(config) == "a=1d+21 b=-4D-20"

    with pytest.raises(TypeError):
        config["a"] = "z"


def test_config_type_complex(parser):
    """Test transformation and sanity checks of complex values"""
    config = parser.parse("a=(1.0e-2, -2.0) b=(1.0, -2.0E2)")
    assert config["a"] == 1.0e-2 - 2.0j
    assert config["b"] == 1.0 - 2.0e2j

    config["a"] = 1.0e20 + 1.0j
    config["b"] = -2.0 + 1.0e-20j
    assert (str(config)) == "a=(1e+20, 1.0)b=(-2.0, 1E-20)"

    with pytest.raises(TypeError):
        config["a"] = "z"


def test_config_type_double_complex(parser):
    """Test transformation and sanity checks of Fortran double complex values"""
    config = parser.parse("a = (1.0d-2, -2.0) b = (1.0, -2.0D2)")
    assert config["a"] == 1.0e-2 - 2.0j
    assert config["b"] == 1.0 - 2.0e2j

    config["a"] = 1.0e20 + 1.0j
    config["b"] = -2.0 + 1.0e-20j
    assert (str(config)) == "a=(1d+20, 1.0)b=(-2.0, 1D-20)"

    with pytest.raises(TypeError):
        config["a"] = "z"


def test_config_type_identifier(parser):
    """Test transformation and sanity checks of identifier-like values"""
    config = parser.parse("a=Word b=a2_b1")
    assert config["a"] == "Word"
    assert config["b"] == "a2_b1"

    config["a"] = "new_word"
    assert str(config) == "a=new_word b=a2_b1"

    with pytest.raises(TypeError):
        config["a"] = 1

    with pytest.raises(TypeError):
        config["a"] = "a string"

    with pytest.raises(TypeError):
        config["a"] = "1"


def test_config_type_string(parser):
    """Test transformation and sanity checks of string values"""

    # Double-quoted strings
    config = parser.parse("a='string'b='another string'")
    assert config["a"] == "string"
    assert config["b"] == "another string"

    config["a"] = "new string"
    assert str(config) == "a='new string'b='another string'"

    # Single-quoted strings
    config = parser.parse('a="string" b="another string"')
    assert config["a"] == "string"
    assert config["b"] == "another string"

    config["a"] = "new string"
    assert str(config) == 'a="new string"b="another string"'

    # Sanity check
    with pytest.raises(TypeError):
        config["a"] = 1


def test_config_type_path(parser):
    """Test transformation and sanity checks of path-like values"""
    config = parser.parse("a=./file.txt b=/dir/subdir c=./")
    assert config["a"] == Path("./file.txt")
    assert config["b"] == Path("/dir/subdir")
    assert config["c"] == Path("./")

    config["a"] = Path("/another/path/file.bak")
    assert str(config) == "a=/another/path/file.bak b=/dir/subdir c=./"

    with pytest.raises(TypeError):
        config["a"] = 1


def test_config_type_null(parser):
    """Test transformation and sanity checks of null values"""
    config = parser.parse("a= b=")
    assert config["a"] == None
    assert config["b"] == None
    assert str(config) == "a=b="

    config["b"] = None
    assert str(config) == "a=b="

    # Changing null values not yet implemented
    with pytest.raises(TypeError):
        config["a"] = 1


def test_config_list(parser):
    """Test transformation and sanity checks for lists"""
    config = parser.parse("a = 1, 2, 3 b=4,5,6")

    assert config["a"] == [1, 2, 3]
    assert config["b"] == [4, 5, 6]

    # List reconstruction
    assert str(config) == "a=1,2,3 b=4,5,6"

    # List modification
    config["a"] = [10, 11, 12]
    assert config["a"] == [10, 11, 12]
    assert str(config) == "a=10,11,12 b=4,5,6"

    # Assigning scalar to list
    with pytest.raises(TypeError):
        config["a"] = 1

    # Change type of list item
    with pytest.raises(TypeError):
        config["a"] = ["A", 11, 12]

    # Changing list length
    with pytest.raises(ValueError):
        config["a"] = [10, 11]
    with pytest.raises(ValueError):
        config["a"] = [10, 11, 12, 13]

    # Assigning list to scalar
    with pytest.raises(TypeError):
        config = parser.parse("a = 1")
        config["a"] = [1, 2]


def test_config_block(parser):
    """Test transformation and sanity checks of blocks"""
    config = parser.parse("block < a:2 b:4|5|6 >")

    assert config["block"]["a"] == 2
    assert config["block"]["b"] == [4, 5, 6]
    assert dict(config["block"]) == {"a": 2, "b": [4, 5, 6]}

    # Block content reconstruction
    assert str(config) == "block<a:2 b:4|5|6>"
    assert str(config["block"]) == "a:2 b:4|5|6"

    # Block modification
    config["block"]["b"] = [4, 10, 6]
    assert config["block"]["b"] == [4, 10, 6]
    assert str(config["block"]) == "a:2 b:4|10|6"

    # Directly updating an entire block is not allowed
    with pytest.raises(SyntaxError):
        config["block"] = {"a": 4, "b": [10, 11, 12]}


def test_config_del(parser):
    """Test for deleting items from config"""

    # Delete key-value
    config = parser.parse("c=1 block < a:2 b:4|5|6 >")
    del config["c"]
    assert dict(config) == {"block": {"a": 2, "b": [4, 5, 6]}}
    assert str(config) == "block<a:2 b:4|5|6>"

    # Delete key-value inside block
    config = parser.parse("c=1 block < a:2 b:4|5|6 >")
    del config["block"]["a"]
    assert dict(config) == {"c": 1, "block": {"b": [4, 5, 6]}}
    assert str(config) == "c=1 block<b:4|5|6>"

    # Delete block
    config = parser.parse("c=1 block < a:2 b:4|5|6 >")
    del config["block"]
    assert dict(config) == {"c": 1}
    assert str(config) == "c=1"

    # Delete everything
    config = parser.parse("c=1 block < a:2 b:4|5|6 >")
    del config["block"]
    del config["c"]
    assert dict(config) == {}
    assert str(config) == ""


def test_config_invalid_rules(parser):
    """Test for rules that are incompatible with the assumptions made in the interpreter"""

    # key_value rule that returns more than one value
    with pytest.raises(ValueError):
        parser.parse("a = 1:2\n")

    # key_value rule that returns no value
    with pytest.raises(ValueError):
        parser.parse("a := \n")


def test_config_invalid_operations(parser):
    """Test operations that are not supported by the configurations as stored in an Config instance"""
    config = parser.parse("a=1")

    # Adding a new item
    with pytest.raises(KeyError):
        config["z"] = "a string"


class CaseParser(ConfigParser):
    """Parser using the grammar defined above, with case insensitive keys."""

    @property
    def case_sensitive_keys(self) -> bool:
        return False

    @property
    def grammar(self) -> str:
        return grammar


@pytest.fixture(scope="module")
def case_parser():
    return CaseParser()


def test_config_case_insensitive(case_parser):
    """Test config files whose keys are case insensitive"""
    config = case_parser.parse("a = 1\n")

    assert config["a"] == config["A"]

    config["a"] = 2
    assert config["A"] == 2

    config["A"] = 3
    assert config["a"] == 3

    del config["a"]
    assert "a" not in config
    assert "A" not in config


class WrongParser(ConfigParser):
    """Parser with an incorrect grammar where keys are not terminals."""

    @property
    def case_sensitive_keys(self) -> bool:
        return False

    @property
    def grammar(self) -> str:
        return """
    start: key_value

    key_value: key "=" value
    key: integer

    ?value: logical

    %import config.logical
    %import config.integer

    %import common.WS
    %ignore WS
"""


@pytest.fixture()
def wrong_parser():
    """Fixture instantiating the incorrect parser."""
    return WrongParser()


def test_config_wrong_key_rule(wrong_parser):
    """Test incorrect case where keys are not terminals"""
    with pytest.raises(TypeError):
        wrong_parser.parse("10=.true.")
