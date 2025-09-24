# access-config-utils

![CI](https://github.com/ACCESS-NRI/access-config-utils/actions/workflows/ci.yml/badge.svg) [![codecov](https://codecov.io/github/ACCESS-NRI/access-config-utils/graph/badge.svg?token=KtmrCtSyMv)](https://codecov.io/github/ACCESS-NRI/access-config-utils) [![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://opensource.org/license/apache-2-0) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## About

A Python package providing various utilities to work with configurations for the models developed at [ACCESS-NRI](https://github.com/ACCESS-NRI).

## Key Features

Parsers:
- Various parsers for configuration files used across ACCESS-NRI models
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

The parsed content can then be modified just like any Python dict:
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
