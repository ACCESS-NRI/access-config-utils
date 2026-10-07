# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest

from access.config.yaml_config import DEFAULT_INDENT, YAMLConfig, YAMLParser, guess_indent


@pytest.fixture(scope="module")
def parser():
    """Fixture instantiating the parser."""
    return YAMLParser()


@pytest.fixture()
def simple_yaml_config():
    """Fixture returning a dictionary storing a payu config file."""
    return {
        "project": "x77",
        "ncpus": 48,
        "jobfs": "10GB",
        "mem": "192GB",
        "walltime": "01:00:00",
        "jobname": "1deg_jra55do_ryf",
        "model": "access-om3",
        "exe": "/some/path/to/access-om3-MOM6-CICE6",
        "input": [
            "/some/path/to/inputs/1deg/mom",
            "/some/path/to/inputs/1deg/cice",
            "/some/path/to/inputs/1deg/share",
        ],
    }


@pytest.fixture()
def simple_yaml_config_file():
    """Fixture returning the contents of a simple payu config file."""
    return """project: x77
ncpus: 48
jobfs: 10GB
mem: 192GB
walltime: 01:00:00
jobname: 1deg_jra55do_ryf
model: access-om3
exe: /some/path/to/access-om3-MOM6-CICE6
input:
- /some/path/to/inputs/1deg/mom
- /some/path/to/inputs/1deg/cice
- /some/path/to/inputs/1deg/share
"""


@pytest.fixture()
def yaml_config_file():
    """Fixture returning the contents of a more complex payu config file."""
    return """# PBS configuration

# If submitting to a different project to your default, uncomment line below
# and change project code as appropriate; also set shortpath below
project: x77

# Force payu to always find, and save, files in this scratch project directory
# (you may need to add the corresponding PBS -l storage flag in sync_data.sh)

ncpus: 48
jobfs: 10GB
mem: 192GB

walltime: 01:00:00
jobname: 1deg_jra55do_ryf

model: access-om3

exe: /some/path/to/access-om3-MOM6-CICE6
input:
    - /some/path/to/inputs/1deg/mom   # MOM6 inputs
    - /some/path/to/inputs/1deg/cice  # CICE inputs
    - /some/path/to/inputs/1deg/share # shared inputs

"""


@pytest.fixture()
def modified_yaml_config_file():
    """Fixture returning the previous payu config file, with some modifications."""
    return """# PBS configuration

# If submitting to a different project to your default, uncomment line below
# and change project code as appropriate; also set shortpath below
project: x77

# Force payu to always find, and save, files in this scratch project directory
# (you may need to add the corresponding PBS -l storage flag in sync_data.sh)

ncpus: 64
jobfs: 10GB
mem: 192GB

walltime: 01:00:00
jobname: 1deg_jra55do_ryf

model: access-om3

input:
    - /some/other/path/to/inputs/1deg/mom # MOM6 inputs
    - /some/path/to/inputs/1deg/cice  # CICE inputs
    - /some/path/to/inputs/1deg/share # shared inputs

"""


def test_read_yaml_config(parser, simple_yaml_config, simple_yaml_config_file):
    """Test parsing of a simple file."""
    config = parser.parse(simple_yaml_config_file)

    assert config == simple_yaml_config


def test_round_trip_yaml_config(parser, yaml_config_file, modified_yaml_config_file):
    """Test round-trip parsing of a more complex file with mutation of the config."""
    config = parser.parse(yaml_config_file)

    config["ncpus"] = 64
    config["input"][0] = "/some/other/path/to/inputs/1deg/mom"
    del config["exe"]

    assert modified_yaml_config_file == str(config)


def test_round_trip_unmodified(parser, yaml_config_file):
    """An untouched file is written back out exactly as it was read."""
    assert str(parser.parse(yaml_config_file)) == yaml_config_file


def test_round_trip_keeps_quotes_and_long_lines(parser) -> None:
    """Quoting and long lines survive the round trip, and so does the indentation."""
    text = """jobname: "1deg_jra55_ryf"
queue: 'normal'
modules:
  use:
    - /g/data/vk83/modules
submodels:
  - name: ocean
    input:
      - /g/data/vk83/configurations/inputs/access-om3/share/grids/global.1deg/2020.10.22/topog.nc
"""
    config = parser.parse(text)
    assert str(config) == text

    config["queue"] = "express"
    config["submodels"][0]["name"] = "ocn"
    assert str(config) == text.replace("normal", "express").replace("name: ocean", "name: ocn")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("a: 1\nb: 2\n", DEFAULT_INDENT, id="flat"),
        pytest.param("a:\n    b: 1\n", {**DEFAULT_INDENT, "mapping": 4}, id="mapping"),
        pytest.param("a:\n- x\n", {**DEFAULT_INDENT, "sequence": 2, "offset": 0}, id="indentless-sequence"),
        pytest.param("a:\n    - x\n", {**DEFAULT_INDENT, "sequence": 6, "offset": 4}, id="offset-sequence"),
        pytest.param(
            "a:\n  b:\n  - x\nc:\n  d: 1\n",
            {"mapping": 2, "sequence": 2, "offset": 0},
            id="mapping-and-sequence",
        ),
        pytest.param(
            "a:\n  - name: x\n    b:\n       c: 1\n",
            {"mapping": 3, "sequence": 4, "offset": 2},
            id="mapping-in-sequence-entry",
        ),
        pytest.param(
            "# comment:\n\na:   # trailing comment\n   # comment\n   b: 1\n",
            {**DEFAULT_INDENT, "mapping": 3},
            id="comments-and-blank-lines",
        ),
        pytest.param("a: b:\n  c\n", DEFAULT_INDENT, id="value-ending-in-colon"),
    ],
)
def test_guess_indent(text: str, expected: dict[str, int]) -> None:
    """The indentation is read off the first nested mapping and block sequence."""
    assert guess_indent(text) == expected


@pytest.fixture()
def config(parser):
    """Fixture returning a small configuration to mutate."""
    return parser.parse("a: 1\nb: 2\nc: 3\n")


def test_pop(config) -> None:
    """pop removes the key from the text, and honours a default for a missing key."""
    assert config.pop("b") == 2
    assert config.pop("missing", None) is None
    with pytest.raises(KeyError):
        config.pop("missing")
    assert str(config) == "a: 1\nc: 3\n"


def test_popitem(config) -> None:
    """popitem removes the last key from the text, and raises on an empty configuration."""
    assert config.popitem() == ("c", 3)
    assert str(config) == "a: 1\nb: 2\n"
    config.clear()
    with pytest.raises(KeyError):
        config.popitem()


def test_clear(config) -> None:
    """clear removes every key from the text."""
    config.clear()
    assert config == {}
    assert str(config) == "{}\n"


def test_update(config) -> None:
    """update writes every item, whether given as a mapping, pairs or keywords."""
    config.update({"a": 10}, b=20)
    config.update([("d", 4)])
    config |= {"c": 30}
    assert config == {"a": 10, "b": 20, "c": 30, "d": 4}
    assert str(config) == "a: 10\nb: 20\nc: 30\nd: 4\n"


def test_setdefault(config) -> None:
    """setdefault writes only a key that is absent."""
    assert config.setdefault("a", 10) == 1
    assert config.setdefault("d", 4) == 4
    assert str(config) == "a: 1\nb: 2\nc: 3\nd: 4\n"


def test_yaml_config_default_dumper(parser) -> None:
    """A YAMLConfig made without a YAML instance dumps with ruamel.yaml's defaults."""
    config = YAMLConfig(parser.parse("a:\n    - x\n").map)
    assert str(config) == "a:\n- x\n"
