import pytest
from unittest.mock import mock_open, patch
from pathlib import Path

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
def simple_nuopc_config_file():
    return """DRIVER_attributes::
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


@pytest.fixture()
def invalid_nuopc_config_file():
    return """DRIVER_attributes::
  Verbosity: off
  cime_model - cesm
::

COMPONENTS::: atm ocn
"""


@patch("pathlib.Path.is_file", new=lambda file: True)
def test_read_nuopc_config(simple_nuopc_config, simple_nuopc_config_file):
    with patch("builtins.open", mock_open(read_data=simple_nuopc_config_file)) as m:
        config = read_nuopc_config(file_name="simple_nuopc_config_file")

        assert config == simple_nuopc_config


def test_write_nuopc_config(simple_nuopc_config, simple_nuopc_config_file):
    with patch("builtins.open", mock_open()) as m:
        write_nuopc_config(simple_nuopc_config, Path("config_file"))

        assert simple_nuopc_config_file == "".join(call.args[0] for call in m().write.mock_calls)


@patch("pathlib.Path.is_file", new=lambda file: True)
def test_read_invalid_nuopc_config_file(invalid_nuopc_config_file):
    with patch("builtins.open", mock_open(read_data=invalid_nuopc_config_file)) as m:
        with pytest.raises(ValueError):
            read_nuopc_config(file_name="invalid_nuopc_config_file")


def test_read_missing_nuopc_config_file():
    with pytest.raises(FileNotFoundError):
        read_nuopc_config(file_name="garbage")
