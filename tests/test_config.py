# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from conftest import TOY_GRAMMAR
from lark import Tree

from access.config.grammar_compiled import clear_grammar_cache
from access.config.grammar_contract import ENTRY_CATEGORIES
from access.config.parser import ConfigParser
from access.config.tree_edits import merge_adjacent_repetitions
from access.config.tree_navigation import AddParent
from access.config.tree_reader import EntryReader


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
    assert config["a"] is None
    assert config["b"] is None
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

    # Update a single element by index
    config["a"][1] = 20
    assert config["a"] == [1, 20, 3]
    assert str(config) == "a=1,20,3 b=4,5,6"

    # Update using negative index
    config["b"][-1] = 60
    assert config["b"] == [4, 5, 60]
    assert str(config) == "a=1,20,3 b=4,5,60"

    # Replace list with new list of same length and type
    config["a"] = [10, 11, 12]
    assert config["a"] == [10, 11, 12]
    assert str(config) == "a=10,11,12 b=4,5,60"

    # After whole-list replacement, element updates still work
    config["a"][1] = 80
    assert config["a"] == [10, 80, 12]
    assert str(config) == "a=10,80,12 b=4,5,60"

    # Type mismatch when updating a single element
    with pytest.raises(TypeError):
        config["a"][0] = "string"

    # Slice assignment
    config["a"] = [10, 11, 12]
    config["a"][0:2] = [1, 2]
    assert config["a"] == [1, 2, 12]
    assert str(config) == "a=1,2,12 b=4,5,60"

    # Slice assignment with step
    config["a"] = [10, 20, 30]
    config["a"][::2] = [100, 300]
    assert config["a"] == [100, 20, 300]
    assert str(config) == "a=100,20,300 b=4,5,60"

    # Slice assignment with wrong length
    with pytest.raises(ValueError):
        config["a"][0:2] = [1]

    # Slice assignment with wrong type
    with pytest.raises(TypeError):
        config["a"][0:1] = ["string"]

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


def test_config_list_element_update_in_block(parser):
    """Test updating individual elements in a list inside a block"""
    config = parser.parse("block < a:2 b:4|5|6 >")

    config["block"]["b"][1] = 50
    assert config["block"]["b"] == [4, 50, 6]
    assert str(config["block"]) == "a:2 b:4|50|6"


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

    # Delete key-list
    config = parser.parse("c=1 a=1,2,3")
    del config["a"]
    assert dict(config) == {"c": 1}
    assert str(config) == "c=1"

    # Delete key-list inside block
    config = parser.parse("c=1 block < a:2 b:4|5|6 >")
    del config["block"]["b"]
    assert dict(config) == {"c": 1, "block": {"a": 2}}
    assert str(config) == "c=1 block<a:2>"

    # Delete key-null
    config = parser.parse("c=1 a=")
    del config["a"]
    assert dict(config) == {"c": 1}
    assert str(config) == "c=1"


def test_config_invalid_rules(parser):
    """Test for rules that are incompatible with the assumptions made in the interpreter"""

    # key_value rule that returns more than one value
    with pytest.raises(ValueError):
        parser.parse("a = 1:2\n")

    # key_value rule that returns no value
    with pytest.raises(ValueError):
        parser.parse("a := \n")

    # key_value rule that has no "key" child node
    with pytest.raises(ValueError):
        parser.parse("! 1\n")


def test_config_invalid_operations(parser):
    """Test operations not supported by the configurations stored in a Config instance"""
    config = parser.parse("a=1")

    # Reading or removing a key that does not exist. Assigning to one adds it.
    with pytest.raises(KeyError):
        config["z"]
    with pytest.raises(KeyError):
        del config["z"]

    # A block can be created, but not replaced once it exists.
    config = parser.parse("blk<a:2>")
    with pytest.raises(SyntaxError):
        config["blk"] = {"a": 3}


def _check_added(parser, config, expected):
    """Assert the text of *config*, and that reading it back gives the same values.

    The text alone does not prove the values survive: several value types share a Python
    type, so a value can be written with exactly the expected text and still read back as a
    different type.
    """
    assert str(config) == expected
    reparsed = parser.parse(expected)
    assert dict(reparsed) == dict(config)
    # Writing out what was just read back must not drift.
    assert str(reparsed) == expected


def test_config_add_key_value(parser):
    """Test adding a new scalar key"""
    config = parser.parse("a=1 b=2")
    config["z"] = 3

    assert config["z"] == 3
    _check_added(parser, config, "a=1 b=2 z=3")


def test_config_add_key_value_types(parser):
    """Test the value type chosen for a new key, for every type the grammar admits.

    This grammar admits both members of every ambiguous pair, so these expectations are the
    specification of ``VALUE_RULE_PRIORITY``: ``logical`` over ``bool``, ``float`` over
    ``double``, ``complex`` over ``double_complex`` and ``string`` over ``identifier``.
    """
    config = parser.parse("a=1")
    config["t1"] = True
    config["t2"] = 7
    config["t3"] = 1.5
    config["t4"] = 3 + 4j
    config["t5"] = "word"
    config["t6"] = Path("/x/y")

    # Note the missing spaces: the reconstructor separates two items only when the
    # characters either side of the join would otherwise run together into one word.
    _check_added(parser, config, 'a=1 t1=.true.t2=7 t3=1.5 t4=(3.0, 4.0)t5="word"t6=/x/y')


def test_config_add_key_list(parser):
    """Test adding a new list key, and that its elements stay individually editable"""
    config = parser.parse("a=1")
    config["z"] = [1, 2, 3]
    _check_added(parser, config, "a=1 z=1,2,3")

    # The new list is a ConfigList over the new nodes, so element updates reach the tree.
    config["z"][1] = 9
    _check_added(parser, config, "a=1 z=1,9,3")

    config["z"] = [4, 5, 6]
    _check_added(parser, config, "a=1 z=4,5,6")

    with pytest.raises(ValueError):
        config["z"] = [1, 2]


def test_config_add_heterogeneous_list(parser):
    """Test adding a list whose elements are of different types"""
    config = parser.parse("a=1")
    config["z"] = [1, "two", 3.5]

    assert config["z"] == [1, "two", 3.5]
    _check_added(parser, config, 'a=1 z=1,"two",3.5')


def test_config_add_key_null(parser):
    """Test adding a new key with no value"""
    config = parser.parse("a=1")
    config["z"] = None

    assert config["z"] is None
    _check_added(parser, config, "a=1 z=")

    # A valueless entry still cannot become a valued one.
    with pytest.raises(TypeError):
        config["z"] = 1

    del config["z"]
    assert "z" not in config
    _check_added(parser, config, "a=1")


def test_config_add_key_block(parser):
    """Test creating a new block and filling it"""
    config = parser.parse("a=1")
    config["z"] = {"p": 1, "q": [2, 3]}

    assert dict(config["z"]) == {"p": 1, "q": [2, 3]}
    _check_added(parser, config, "a=1 z<p:1 q:2|3>")

    # The block is a live Config, so it can be edited and added to afterwards.
    config["z"]["p"] = 5
    config["z"]["r"] = 6
    _check_added(parser, config, "a=1 z<p:5 q:2|3 r:6>")


def test_config_add_empty_key_block(parser):
    """Test that a block can be created empty and filled later"""
    config = parser.parse("a=1")
    config["z"] = {}
    _check_added(parser, config, "a=1 z<>")

    config["z"]["p"] = 1
    _check_added(parser, config, "a=1 z<p:1>")


def test_config_add_into_block(parser):
    """Test adding keys inside a block, which uses a different syntax from the top level"""
    config = parser.parse("blk<a:2>")

    config["blk"]["b"] = 3
    assert str(config["blk"]) == "a:2 b:3"
    _check_added(parser, config, "blk<a:2 b:3>")

    config["blk"]["c"] = [3, 4]
    _check_added(parser, config, "blk<a:2 b:3 c:3|4>")

    # A key added at the top level of the same file uses the top-level syntax.
    config["d"] = 0
    _check_added(parser, config, "blk<a:2 b:3 c:3|4>d=0")


def test_config_add_into_empty_block(parser):
    """Test adding a key to an empty block, where there is no entry to copy a style from"""
    config = parser.parse("blk< >")
    assert dict(config["blk"]) == {}

    config["blk"]["x"] = 1
    _check_added(parser, config, "blk<x:1>")


def test_config_add_after_delete(parser):
    """Test that deleting a key and adding it again leaves exactly one entry"""
    config = parser.parse("a=1 b=2")
    del config["a"]
    config["a"] = 9

    # The re-added key goes to the end; what matters is that the original entry is gone.
    _check_added(parser, config, "b=2 a=9")

    # Deleting the only key of a block, then adding one back.
    config = parser.parse("blk<a:2>")
    del config["blk"]["a"]
    config["blk"]["c"] = 3
    _check_added(parser, config, "blk<c:3>")


def test_config_add_into_emptied_config(parser):
    """Test adding a key after every key has been deleted.

    ``__str__`` reports an empty configuration as the empty string even though the parse
    tree still holds whatever was not an entry, so the first addition brings that back.
    """
    config = parser.parse("a=1 b=2")
    for key in list(config):
        del config[key]
    assert str(config) == ""

    config["n"] = 1
    _check_added(parser, config, "n=1")


def test_config_add_string_quote_choice(parser):
    """Test that a new string is quoted with a character it does not itself contain"""
    config = parser.parse("a=1")

    config["y"] = "plain"
    config["z"] = 'has a " inside'

    assert config["z"] == 'has a " inside'
    _check_added(parser, config, 'a=1 y="plain"z=\'has a " inside\'')


def test_config_add_key_errors(parser):
    """Test the ways adding a key can fail"""
    config = parser.parse("a=1")

    # Keys have to be usable as a name in the file.
    for key in ("", "a b", "2fast", "a-b", "a.b"):
        with pytest.raises(ValueError):
            config[key] = 1

    # A list needs at least two elements to be written as one.
    for value in ([], [1]):
        with pytest.raises(ValueError):
            config["z"] = value

    # No value type can hold any of these.
    for value in (object(), float("inf"), float("nan")):
        with pytest.raises(TypeError):
            config["z"] = value

    # A quoted string cannot hold both quote characters, since values are not escaped.
    with pytest.raises(TypeError):
        config["z"] = "has 'one' and \"the other\""

    # This grammar has no valueless assignment inside a block.
    config = parser.parse("blk<a:2>")
    with pytest.raises(TypeError):
        config["blk"]["z"] = None

    # None of the above may have changed the file.
    assert str(config) == "blk<a:2>"


def test_config_add_idempotent(parser):
    """Test that writing a configuration out is repeatable and stable across a re-parse"""
    config = parser.parse("a=1 blk<b:2>")
    config["z"] = [1, 2]
    config["blk"]["y"] = 3

    first = str(config)
    assert str(config) == first

    reparsed = parser.parse(first)
    reparsed["w"] = 4
    again = parser.parse(str(reparsed))
    assert str(again) == str(reparsed)


def test_config_add_no_node_reuse(parser):
    """Test that assigning one key's value to another does not make them share tree nodes"""
    config = parser.parse("x=1 y=2,3")

    config["p"] = config["x"]
    config["q"] = config["y"]
    assert config._refs["p"] is not config._refs["x"]
    assert config._refs["q"] is not config._refs["y"]

    config["p"] = 9
    config["q"][0] = 9
    assert config["x"] == 1
    assert config["y"] == [2, 3]
    _check_added(parser, config, "x=1 y=2,3 p=9 q=9,3")


def test_config_dict_methods(parser):
    """Test the dict methods ``dict`` implements without going through __setitem__.

    Inheriting them would skip both key normalisation and the parse tree update. A ``pop``
    followed by an assignment is the damaging case: the removed entry would stay in the tree
    and a second one would be added, writing a file with the key twice.
    """
    config = parser.parse("a=1 b=2")

    assert config.pop("a") == 1
    assert str(config) == "b=2"
    config["a"] = 5
    _check_added(parser, config, "b=2 a=5")

    assert config.get("a") == 5
    assert config.get("nope") is None
    assert config.get("nope", "fallback") == "fallback"

    assert config.pop("nope", "fallback") == "fallback"
    with pytest.raises(KeyError):
        config.pop("nope")

    config.update({"c": 3}, d=4)
    _check_added(parser, config, "b=2 a=5 c=3 d=4")

    assert config.setdefault("c", 99) == 3
    assert config.setdefault("e", 5) == 5
    _check_added(parser, config, "b=2 a=5 c=3 d=4 e=5")

    config |= {"f": 6}
    _check_added(parser, config, "b=2 a=5 c=3 d=4 e=5 f=6")

    assert config.popitem() == ("f", 6)
    _check_added(parser, config, "b=2 a=5 c=3 d=4 e=5")

    config.clear()
    assert str(config) == ""


def test_config_list_unsupported_operations(parser):
    """Test that list operations the parse tree cannot follow are refused"""
    config = parser.parse("a=1,2,3")
    values = config["a"]

    for operation in (
        lambda: values.append(4),
        lambda: values.extend([4]),
        lambda: values.insert(0, 4),
        lambda: values.remove(1),
        lambda: values.pop(),
        lambda: values.clear(),
        lambda: values.sort(),
        lambda: values.reverse(),
        lambda: values.__delitem__(0),
        lambda: values.__iadd__([4]),
        lambda: values.__imul__(2),
    ):
        with pytest.raises(NotImplementedError):
            operation()

    assert str(config) == "a=1,2,3"


def test_config_grammar_cache(parser):
    """Test that a grammar is compiled once and that the cache can be cleared"""
    first = parser.parse("a=1")
    second = parser.parse("a=1")
    # Compiling costs far more than parsing, so the result is shared.
    assert first._ctx.grammar is second._ctx.grammar

    clear_grammar_cache()
    assert parser.parse("a=1")._ctx.grammar is not first._ctx.grammar


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


class TemplateParser(ConfigParser):
    """Parser overriding the text used for a new entry, to exercise the override path."""

    @property
    def case_sensitive_keys(self) -> bool:
        return True

    @property
    def grammar(self) -> str:
        return TOY_GRAMMAR

    @property
    def entry_templates(self):
        # This grammar discards whitespace with %ignore, so a template can only spell the
        # same text the grammar itself derives. What these pin is that the override is
        # compiled and used at all; that a template can *change* the text is covered
        # against a real format, in test_nuopc_config.py.
        return {("start", "key_value"): "{key}={value}", ("block", "key_value"): "{key}:{value}"}

    @property
    def value_rule_priority(self):
        # Prefer the bare identifier over the quoted string, reversing the default.
        return ["identifier", "string", "logical", "bool", "integer", "float", "double"]


def test_config_entry_template_override():
    """Test that a parser-declared template and value priority are used"""
    parser = TemplateParser()

    config = parser.parse("a=1")
    config["z"] = "word"
    # The overridden priority picks identifier, so the value is not quoted.
    _check_added(parser, config, "a=1 z=word")

    config = parser.parse("blk<a:2>")
    config["blk"]["b"] = 3
    _check_added(parser, config, "blk<a:2 b:3>")


def test_config_entry_template_used_without_a_donor():
    """Test that the declared template is compiled and used for an entry with no neighbour.

    A template applies only where nothing nearby can supply a style, so an empty block is
    the case that reaches it: the block holds no entry to copy from, and no sibling block
    either.
    """
    parser = TemplateParser()

    config = parser.parse("a=1 blk<>")
    config["blk"]["b"] = 3
    _check_added(parser, config, "a=1 blk<b:3>")


def test_config_add_value_rule_fallback():
    """Test that a value whose preferred rendering reads back as another type falls back.

    With ``identifier`` preferred over ``string``, the string ``"True"`` would be written as
    the bare word ``True``, which this grammar reads back as a boolean. That candidate is
    rejected and the quoted form is used instead, so the priority order can never produce a
    wrong value.
    """
    parser = TemplateParser()

    config = parser.parse("a=1")
    config["z"] = "True"
    assert config["z"] == "True"
    _check_added(parser, config, 'a=1 z="True"')

    # A string that is not a valid bare word likewise falls back to being quoted.
    config = parser.parse("a=1")
    config["z"] = "a string"
    _check_added(parser, config, 'a=1 z="a string"')


class BrokenTemplateParser(TemplateParser):
    """Parser whose override does not fit the grammar, to exercise the fall-back."""

    @property
    def entry_templates(self):
        # Declared for the block, because that is the container an empty one gives us: a
        # template is only ever tried where there is no neighbouring entry to copy from.
        return {("block", "key_value"): "{key} <<>> {value}"}


def test_config_entry_template_fallback():
    """Test that an override no longer fitting the grammar falls back to the derived text"""
    parser = BrokenTemplateParser()

    config = parser.parse("a=1 blk<>")
    config["blk"]["b"] = 3
    _check_added(parser, config, "a=1 blk<b:3>")


class BlocklessParser(ConfigParser):
    """Parser for a grammar with no block rule, so it keeps a single start symbol."""

    @property
    def case_sensitive_keys(self) -> bool:
        return True

    @property
    def grammar(self) -> str:
        return """
    start: key_value+

    key_value: key equal value
    equal: "="

    ?value: integer

    %import config.key
    %import config.integer

    %import common.WS
    %ignore WS
"""


def test_config_blockless_grammar():
    """Test that a grammar without a block rule still compiles and accepts new keys.

    The parser is built with ``block`` as a second start symbol only when the grammar has
    such a rule, since Lark rejects a start symbol naming a rule that does not exist.
    """
    parser = BlocklessParser()
    config = parser.parse("a=1")

    config["z"] = 2
    _check_added(parser, config, "a=1 z=2")


def test_config_string_quote_switching(parser):
    """Test that updating a string picks a quote character the value does not contain.

    Values are wrapped verbatim rather than escaped, so keeping the original quote would
    produce a line that no longer parses.
    """
    config = parser.parse("a='x'b=\"y\"")

    # The quote already in use is kept when the value allows it.
    config["a"] = "plain"
    assert str(config) == "a='plain'b=\"y\""

    # It is switched when the value contains it.
    config["a"] = "has ' inside"
    _check_added(parser, config, 'a="has \' inside"b="y"')

    config["b"] = 'has " inside'
    _check_added(parser, config, "a=\"has ' inside\"b='has \" inside'")

    # A value containing both cannot be written at all, and nothing is changed.
    with pytest.raises(TypeError):
        config["a"] = "has ' and \""
    _check_added(parser, config, "a=\"has ' inside\"b='has \" inside'")


def test_config_non_finite_floats(parser):
    """Test that infinities and NaNs are refused rather than written as bare words.

    Python writes them as ``inf`` and ``nan``, which no supported format reads back as a
    number, so writing one would silently produce a file that no longer means the same.
    """
    config = parser.parse("a=1.0 b=(1.0, 2.0) c=1.0d0")

    for key in ("a", "b", "c"):
        for value in (float("inf"), float("-inf"), float("nan")):
            with pytest.raises(TypeError):
                config[key] = value if key != "b" else complex(value, 0.0)

    # None of the refused assignments changed the file.
    assert str(config) == "a=1.0 b=(1.0, 2.0)c=1.0d0"


def test_config_is_repetition_rule(parser):
    """Test the shape test that decides which rules may be merged after a deletion"""
    info = parser.parse("a=1")._ctx.info

    # A rule that is a plain repetition of something.
    assert info.is_repetition_rule("block")
    # A rule whose alternative is a single terminal, and one with a longer expansion.
    assert not info.is_repetition_rule("equal")
    assert not info.is_repetition_rule("key_value")
    # A name that is not a rule at all.
    assert not info.is_repetition_rule("not_a_rule")


def test_config_merge_adjacent_repetitions(parser):
    """Test that merging rejoins the children of two adjacent nodes and re-parents them.

    Exercised here with a repetition rule whose children are rule nodes rather than tokens,
    so that the merged children end up owned by the node that survives.
    """
    config = parser.parse("a=1")
    info = config._ctx.info

    left, right = Tree("block", [Tree("x", [])]), Tree("block", [Tree("y", [])])
    container = Tree("start", [left, right, Tree("equal", [])])
    AddParent().visit(container)

    merge_adjacent_repetitions(container, info)

    assert len(container.children) == 2
    assert [child.data for child in container.children[0].children] == ["x", "y"]
    # The moved child now belongs to the node that absorbed it.
    assert container.children[0].children[1].parent is left
    # A rule that is not a repetition is left alone.
    assert container.children[1].data == "equal"


def test_grammar_contract_has_an_interpreter_callback_per_category() -> None:
    """Test that ``EntryReader`` implements every entry category the contract declares.

    This is the one duplication of the category names that cannot be removed: Lark's
    ``Interpreter`` dispatches on a node's rule name to a method of that name, so the
    callbacks have to be spelled out. Adding a category without its callback would leave
    entries of that category silently unvisited, so the two lists are pinned together here
    instead of by construction.
    """
    for category in ENTRY_CATEGORIES:
        assert callable(getattr(EntryReader, category, None)), category
