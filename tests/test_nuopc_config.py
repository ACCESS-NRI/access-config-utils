import pytest
import filecmp

from test_utils import MockFile
from access.parsers.nuopc_config import read_nuopc_config, write_nuopc_config


@pytest.fixture()
def simple_nuopc_config():
    return dict(
        DRIVER_attributes={
            "Verbosity": "off",
            "cime_model": "cesm",
            "logFilePostFix": ".log",
            "pio_blocksize": -1,
            "pio_rearr_comm_enable_hs_comp2io": True,
            "pio_rearr_comm_enable_hs_io2comp": False,
            "reprosum_diffmax": -1.0e-8,
            "wv_sat_table_spacing": 1.0,
            "wv_sat_transition_start": 20.0,
        },
        COMPONENTS=["atm", "ocn"],
        ALLCOMP_attributes={
            "ATM_model": "datm",
            "GLC_model": "sglc",
            "OCN_model": "mom",
            "ocn2glc_levels": "1:10:19:26:30:33:35",
        },
    )


@pytest.fixture()
def simple_nuopc_config_file(tmp_path):
    file = tmp_path / "simple_config_file"
    resource_file_str = """DRIVER_attributes::
  Verbosity = off
  cime_model = cesm
  logFilePostFix = .log
  pio_blocksize = -1
  pio_rearr_comm_enable_hs_comp2io = .true.
  pio_rearr_comm_enable_hs_io2comp = .false.
  reprosum_diffmax = -1.000000D-08
  wv_sat_table_spacing = 1.000000D+00
  wv_sat_transition_start = 2.000000D+01
::

COMPONENTS: atm ocn
ALLCOMP_attributes::
  ATM_model = datm
  GLC_model = sglc
  OCN_model = mom
  ocn2glc_levels = 1:10:19:26:30:33:35
::

"""
    return MockFile(file, resource_file_str)


@pytest.fixture()
def invalid_nuopc_config_file(tmp_path):
    file = tmp_path / "invalid_config_file"
    resource_file_str = """DRIVER_attributes::
  Verbosity: off
  cime_model - cesm
::

COMPONENTS::: atm ocn
"""
    return MockFile(file, resource_file_str)


def test_read_nuopc_config(tmp_path, simple_nuopc_config, simple_nuopc_config_file):
    config_from_file = read_nuopc_config(file_name=simple_nuopc_config_file.file)

    assert config_from_file == simple_nuopc_config


def test_write_nuopc_config(tmp_path, simple_nuopc_config, simple_nuopc_config_file):
    file = tmp_path / "config_file"
    write_nuopc_config(simple_nuopc_config, file)

    assert filecmp.cmp(file, simple_nuopc_config_file.file)


def test_read_invalid_nuopc_config_file(tmp_path, invalid_nuopc_config_file):
    with pytest.raises(ValueError):
        read_nuopc_config(file_name=invalid_nuopc_config_file.file)


def test_read_missing_nuopc_config_file():
    with pytest.raises(FileNotFoundError):
        read_nuopc_config(file_name="garbage")
