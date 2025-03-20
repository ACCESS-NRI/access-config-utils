import pytest
import filecmp

from test_utils import MockFile
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
def simple_mom6_input_file(tmp_path):
    file = tmp_path / "simple_mom6_input_file"
    mom6_input_str = """BOOL = True
DT = 1800.0
IGNORED_DIRECTIVE = 3
INCORRECT_DIRECTIVE = 2
N_SMOOTH = 4
REGRIDDING_COORDINATE_MODE = 'ZSTAR'
"""
    return MockFile(file, mom6_input_str)


@pytest.fixture()
def complex_mom6_input_file(tmp_path):
    file = tmp_path / "complex_mom6_input_file"
    mom6_input_str = """
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
    return MockFile(file, mom6_input_str)


@pytest.fixture()
def modified_mom6_input_file(tmp_path):
    file = tmp_path / "modified_mom6_input_file"
    mom6_input_str = """


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
    return MockFile(file, mom6_input_str)


def test_read_mom6_input(tmp_path, simple_mom6_input, simple_mom6_input_file):
    mom6_input_from_file = read_mom6_input(file_name=simple_mom6_input_file.file)

    assert mom6_input_from_file == simple_mom6_input


def test_write_mom6_input(tmp_path, simple_mom6_input, simple_mom6_input_file):
    file = tmp_path / "MOM_input"
    write_mom6_input(simple_mom6_input, file)

    assert filecmp.cmp(file, simple_mom6_input_file.file)


def test_round_trip_mom6_input(tmp_path, complex_mom6_input_file, modified_mom6_input_file):
    mom6_input_from_file = Mom6Input(file_name=complex_mom6_input_file.file)
    mom6_input_from_file["dt"] = 900.0
    mom6_input_from_file["ADDED_VAR"] = 1
    mom6_input_from_file["ADDED_VAR"] = 32
    del mom6_input_from_file["N_SMOOTH"]
    mom6_input_from_file["N_SMOOTH"] = 4
    del mom6_input_from_file["TO_BE_REMOVED"]

    write_mom6_input(mom6_input_from_file, tmp_path / "MOM_input_new")

    assert mom6_input_from_file["ADDED_VAR"] == 32
    assert filecmp.cmp(tmp_path / "MOM_input_new", modified_mom6_input_file.file)


def test_read_missing_mom6_file():
    with pytest.raises(FileNotFoundError):
        Mom6Input(file_name="garbage")
