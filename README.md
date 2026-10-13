# access-config-utils

![CI](https://github.com/ACCESS-NRI/access-config-utils/actions/workflows/ci.yml/badge.svg) [![codecov](https://codecov.io/github/ACCESS-NRI/access-config-utils/graph/badge.svg?token=KtmrCtSyMv)](https://codecov.io/github/ACCESS-NRI/access-config-utils) [![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://opensource.org/license/apache-2-0) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## About

A Python package providing various utilities to work with configurations for the models developed at [ACCESS-NRI](https://github.com/ACCESS-NRI).

## Key Features

Parsers:
- Various parsers for configuration files used across ACCESS-NRI models:
  - Fortran namelists (`FortranNMLParser`)
  - MOM6 parameter files, `MOM_input` and `MOM_override` (`MOM6InputParser`)
  - NUOPC run configurations, `nuopc.runconfig` (`NUOPCParser`)
  - NUOPC run sequences, `nuopc.runseq` (`NUOPCRunSeqParser`)
  - FMS field tables in the legacy format, `field_table` (`FieldTableParser`)
  - YAML (`YAMLParser`) and JSON (`JSONParser`)
- Support for round-trip parsing, preserving comments and formatting
- Simple API

## Documentation

Coming soon.

## Installation

### Using pip

You can install the latest release directly from PyPI:
```shell
pip install access-config-utils
```

### From source

If you prefer to install from source:
```shell
git clone https://github.com/ACCESS-NRI/access-config-utils.git
cd access-config-utils
pip install .
```

## Usage

### Configuration files parsers

Here is a simple example of how to parse a text and modify its contents.

To parse some text, one just needs to call the `parse` function of the appropriate parser:
```python
from access.config import FortranNMLParser

text = '''&data_nml
parameterA = 1
parameterB = 'abc'
/'''

config = FortranNMLParser().parse(text)
print(config)
```
```python
&data_nml
parameterA = 1
parameterB = 'abc'
/
```

The parsed content can then be modified like a Python dict. Everything not touched — comments,
blank lines, spacing — is reproduced exactly:
```python
config["data_nml"]["parameterA"] = 2
print(config)
```
```python
&data_nml
parameterA = 2
parameterB = 'abc'
/
```

New keys can be added too, and are written like the entries around them: the same spacing,
delimiters, quotes and line endings. Each value is written in whichever notation the format uses,
so a `bool` becomes a Fortran logical here:
```python
config["data_nml"]["parameterC"] = True
config["data_nml"]["parameterD"] = [1, 2, 3]
config["new_nml"] = {"parameterE": "def"}
print(config)
```
```python
&data_nml
parameterA = 2
parameterB = 'abc'
parameterC = .true.
parameterD = 1, 2, 3
/
&new_nml
parameterE = "def"
/
```

A value the format cannot express raises `UnsupportedEntryError` rather than writing a file that
no longer means what the dict says — for instance `None` in a format with no valueless assignment,
or a string containing a space in one with no quoted-string syntax.

A list value behaves like a list: elements can be changed, added, removed and reordered, by
assignment or with `append`, `extend`, `insert`, `pop`, `sort` and the rest. The elements that stay
keep their text, comments included, and a new element is written like its neighbours, on their
line or on a line of its own:
```python
config = FortranNMLParser().parse('''&data_nml
parameterD = 1,
             2  ! two
/''')
config["data_nml"]["parameterD"].append(3)
print(config)
```
```python
&data_nml
parameterD = 1,
             2,
             3  ! two
/
```

A few caveats on the dict and list behaviour:
- A list keeps at least two elements, since a single value is a scalar, and a scalar does not
  become a list or the other way round.
- A list written with repeat counts (`3*1.0`), array qualifiers (`x(2) = …`) or skipped
  positions, or assigned more than once, can have its elements changed but not its length:
  delete the key and assign it again instead. The elements a repeat count stands for can only
  be changed together, all to the same value (`6*9.0` to `6*9.5`): to change one of them,
  delete the key and assign it again.
- Replacing an existing block wholesale raises `SyntaxError` — a block can be created and edited,
  but not swapped out.

Every parser works the same way; only the layout of the dict follows the format. An FMS field
table, for instance, reads as the FMS YAML field table lays it out. A header is three levels of
keys, and a method of three strings is a record, one of a list of the records the file writes for
it, holding its name under `"value"`:
```python
from access.config import FieldTableParser

config = FieldTableParser().parse('''"TRACER", "atmos_mod", "sphum"
    "longname", "specific humidity"
    "profile_type", "fixed", "surface_value=3.e-6" /
''')
print(dict(config["tracer"]["atmos_mod"]["sphum"]))
```
```python
{'longname': 'specific humidity', 'profile_type': [{'value': 'fixed', 'surface_value': 3e-06}]}
```
A record is read only, and is changed by assigning a dict to its place in the list. A new field
is written as a whole entry, after the last of its type and model:
```python
profile = config["tracer"]["atmos_mod"]["sphum"]["profile_type"]
profile[0] = {**profile[0], "surface_value": 1e-5}
config["tracer"]["atmos_mod"]["liq_wat"] = {"longname": "cloud liquid specific humidity"}
print(config)
```
```python
"TRACER", "atmos_mod", "sphum"
    "longname", "specific humidity"
    "profile_type", "fixed", "surface_value=1e-05" /
"TRACER", "atmos_mod", "liq_wat"
    "longname", "cloud liquid specific humidity" /
```

A JSON file holds plain data, so its dict is edited directly, and the text is written again from it
once it changes. The file's own layout is kept, down to which lists it wrote on one line:
```python
from access.config import JSONParser

config = JSONParser().parse('''{
    "name": "run",
    "levels": [1, 2, 3],
    "files": [
        "a.nc",
        "b.nc"
    ]
}
''')
config["dt"] = 900
config["levels"].append(4)
print(config)
```
```python
{
    "name": "run",
    "levels": [1, 2, 3, 4],
    "files": [
        "a.nc",
        "b.nc"
    ],
    "dt": 900
}
```

## Development installation

If you intend to contribute or modify the package, it is recommended to work inside a virtual environment.

1. Create and activate a virtual environment
```shell
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

2. Install in editable mode with development and test dependencies
```shell
pip install -e ".[devel,test]"
```
This will install the package in editable mode, meaning changes to the source code are reflected immediately without reinstallation. Development dependencies such as testing tools will also be installed.

3. Run the test suite
```shell
pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you’d like to add features, fix bugs, or improve documentation.

For significant contributions, we recommend discussing proposed changes in an issue before opening a pull request.

## License

This project is licensed under the Apache 2.0 License.
