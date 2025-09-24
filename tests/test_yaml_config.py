# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest

from access.config.yaml_config import YAMLParser


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
    """Fixture returning the contents the previous payu config file after introducing some modifications."""
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
- /some/path/to/inputs/1deg/cice      # CICE inputs
- /some/path/to/inputs/1deg/share     # shared inputs

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
