import pytest
from unittest.mock import mock_open, patch
from pathlib import Path

from access.parsers.payu_config_yaml import read_payu_config_yaml, write_payu_config_yaml


@pytest.fixture()
def simple_payu_config():
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
def simple_payu_config_file():
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
def complex_payu_config_file():
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
def modified_payu_config_file():
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

exe: /some/path/to/access-om3-MOM6-CICE6
input:
- /some/other/path/to/inputs/1deg/mom # MOM6 inputs
- /some/path/to/inputs/1deg/cice      # CICE inputs
- /some/path/to/inputs/1deg/share     # shared inputs

"""


@patch("pathlib.Path.is_file", new=lambda file: True)
def test_read_payu_config(simple_payu_config, simple_payu_config_file):
    with patch("io.open", mock_open(read_data=simple_payu_config_file)) as m:
        config = read_payu_config_yaml(file_name="simple_payu_config_file")

        assert config == simple_payu_config


def test_write_payu_config(simple_payu_config, simple_payu_config_file):
    with patch("io.open", mock_open()) as m:
        write_payu_config_yaml(simple_payu_config, Path("config_file"))

        assert simple_payu_config_file == "".join(call.args[0] for call in m().write.mock_calls)


@patch("pathlib.Path.is_file", new=lambda file: True)
def test_round_trip_payu_config(complex_payu_config_file, modified_payu_config_file):
    with patch("io.open", mock_open(read_data=complex_payu_config_file)) as m:
        config = read_payu_config_yaml(file_name="complex_config_file")

        config["ncpus"] = 64
        config["input"][0] = "/some/other/path/to/inputs/1deg/mom"
        write_payu_config_yaml(config, Path("some_other_config_file"))

        assert modified_payu_config_file == "".join(call.args[0] for call in m().write.mock_calls)


def test_read_missing_payu_config():
    with pytest.raises(FileNotFoundError):
        read_payu_config_yaml(file_name="garbage")
