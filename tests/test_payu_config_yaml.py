import pytest
import filecmp

from test_utils import MockFile
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
def simple_payu_config_file(tmp_path):
    file = tmp_path / "simple_payu_config_file.yaml"
    payu_file_str = """project: x77
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
    return MockFile(file, payu_file_str)


@pytest.fixture()
def complex_payu_config_file(tmp_path):
    file = tmp_path / "complex_payu_config_file.yaml"
    payu_file_str = """# PBS configuration

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
    return MockFile(file, payu_file_str)


@pytest.fixture()
def modified_payu_config_file(tmp_path):
    file = tmp_path / "modified_payu_config_file.yaml"
    payu_file_str = """# PBS configuration

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
    return MockFile(file, payu_file_str)


def test_read_payu_config(tmp_path, simple_payu_config, simple_payu_config_file):
    config_from_file = read_payu_config_yaml(file_name=simple_payu_config_file.file)

    assert config_from_file == simple_payu_config


def test_write_payu_config(tmp_path, simple_payu_config, simple_payu_config_file):
    file = tmp_path / "config_file"
    write_payu_config_yaml(simple_payu_config, file)

    assert filecmp.cmp(file, simple_payu_config_file.file)


def test_round_trip_payu_config(tmp_path, complex_payu_config_file, modified_payu_config_file):
    config = read_payu_config_yaml(complex_payu_config_file.file)
    config["ncpus"] = 64
    config["input"][0] = "/some/other/path/to/inputs/1deg/mom"
    write_payu_config_yaml(config, tmp_path / "config.yaml")

    assert filecmp.cmp(tmp_path / "config.yaml", modified_payu_config_file.file)


def test_read_missing_payu_config():
    with pytest.raises(FileNotFoundError):
        read_payu_config_yaml(file_name="garbage")
