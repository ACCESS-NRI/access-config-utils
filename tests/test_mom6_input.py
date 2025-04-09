import pytest
from unittest.mock import mock_open, patch
from pathlib import Path

from access.parsers.mom6_input import Mom6Input, write_mom6_input, read_mom6_input


@pytest.fixture()
def simple_mom6_input():
    return {
        "REGRIDDING_COORDINATE_MODE": "ZSTAR",
        "N_SMOOTH": 4,
        "INCORRECT_DIRECTIVE": 2,
        "IGNORED_DIRECTIVE": 3,
        "DT": 1800.0,
        "BOOL": True,
    }


@pytest.fixture()
def simple_mom6_input_file():
    return """BOOL = True
DT = 1800.0
IGNORED_DIRECTIVE = 3
INCORRECT_DIRECTIVE = 2
N_SMOOTH = 4
REGRIDDING_COORDINATE_MODE = 'ZSTAR'
"""


@pytest.fixture()
def complex_mom6_input_file():
    return """
/* This is a comment
   spanning two lines */
REGRIDDING_COORDINATE_MODE = Z*
KPP%
N_SMOOTH = 4
%KPP

#COMMENT_DIRECTIVE = 1
# INCORRECT_DIRECTIVE = 2
#override IGNORED_DIRECTIVE = 3
DT = 1800.0  ! This is a comment
! This is another comment
!COMMENTED_VAR = 3
TO_BE_REMOVED = 10.0 
BOOL = True
"""


@pytest.fixture()
def modified_mom6_input_file():
    return """


REGRIDDING_COORDINATE_MODE = Z*
KPP%
N_SMOOTH = 4
%KPP

#COMMENT_DIRECTIVE = 1
# INCORRECT_DIRECTIVE = 2
#override IGNORED_DIRECTIVE = 3
DT = 900.0  ! This is a comment
! This is another comment
!COMMENTED_VAR = 3
BOOL = True


ADDED_VAR = 32
"""


@patch("pathlib.Path.is_file", new=lambda file: True)
def test_read_mom6_input(simple_mom6_input, simple_mom6_input_file):
    with patch("builtins.open", mock_open(read_data=simple_mom6_input_file)) as m:
        config = read_mom6_input(file_name="simple_mom6_input_file")

        assert config == simple_mom6_input


def test_write_mom6_input(simple_mom6_input, simple_mom6_input_file):
    with patch("pathlib.Path.open", mock_open()) as m:
        write_mom6_input(simple_mom6_input, Path("config_file"))

        assert simple_mom6_input_file == "".join(call.args[0] for call in m().write.mock_calls)


@patch("pathlib.Path.is_file", new=lambda file: True)
def test_round_trip_mom6_input(complex_mom6_input_file, modified_mom6_input_file):
    with patch("builtins.open", mock_open(read_data=complex_mom6_input_file)) as m1:
        config = read_mom6_input(file_name="complex_config_file")

        config["dt"] = 900.0
        config["ADDED_VAR"] = 1
        config["ADDED_VAR"] = 32
        del config["N_SMOOTH"]
        config["N_SMOOTH"] = 4
        del config["TO_BE_REMOVED"]

        with patch("pathlib.Path.open", mock_open()) as m:
            write_mom6_input(config, Path("some_other_config_file"))

            assert config["ADDED_VAR"] == 32
            assert modified_mom6_input_file == "".join(call.args[0] for call in m().write.mock_calls)


def test_read_missing_mom6_file():
    with pytest.raises(FileNotFoundError):
        Mom6Input(file_name="garbage")
