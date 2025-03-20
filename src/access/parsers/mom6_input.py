"""Utilities to handle MOM6 parameter files.

The MOM6 parameter file format is described here:

https://mom6.readthedocs.io/en/main/api/generated/pages/Runtime_Parameter_System.html#mom6-parameter-file-syntax

It has similarities with a Fortran namelist, but with some notable differences:
 - no opening nor closing clauses ('&NAME' and '\')
 - usage of an override directive ('#override')
 - some character, like '*', are allowed in the MOM6 parameter files, but not in namelists
We have also found MOM6 parameter files with C-style comments in files used by CESM. These are ignored by MOM6, but
are actually not part of the specifications.

However, it is possible to preprocess the file to make it a conforming Fortran namelist and then use the f90nml
package to read it. Similarly, one can use the f90nml package to write the file and then postprocess it.

This means that the path from a MOM6 parameter file to a Python dictionary requires the following steps:
 1. read file and preprocess it to handle the directives and the C-style comments.
 2. add opening and closing namelist clauses
 3. parse the file with f90nml, which returns a Namelist object
 4. convert the Namelist object to a Python dictionary

Similarly, to get write a Python dictionary as a MOM6 parameter file, one requires the following steps:
 1. convert the Python dictionary into a Namelist object
 2. write the Namelist object to a file
 3. remove opening and closing namelist clauses

 In the following, we use the following naming conventions:
  - 'mom6_input': the contents of the parameter file as a Python dictionary
  - 'mom6_input_str': the contents of the parameter file, stored as a string
  - 'nml_str': the contents of the file, patched to make it a conforming namelist, stored as a string

We then have utility functions to convert from one representation to another:
  - nml_str -> mom6_input (_nml_str_to_mom6_input)
  - mom6_input -> nml_str (_mom6_input_to_nml_str)
  - mom6_input_str -> nml_str (_mom6_input_str_to_nml_str + _patch_mom6_input_str)
  - nml_str -> mom6_input_str (_nml_str_to_mom6_input_str + _unpatch_mom6_input_str)

For round-trip parsing, one needs to keep track of the changes done to the file to make it a conforming Fortran
namelist and then undo those changes. Since we use the f90mnml parser ability to patch a file as it is read, we also
need to keep the original nml_str and a dictionary with all the changes done to mom6_input. We do this by introducing
a MOM6Input class that extends the dict class.
"""

from pathlib import Path
import re
from io import StringIO

import f90nml


def _patch_mom6_input_str(mom6_input_str: str) -> tuple[str, dict]:
    """Modify the contents of a MOM6 file into a Fortran namelist format readable by f90nml.

    Currently, the "#override" directive is not properly supported. When parsing the file, we will treat variables with
    this directive as normal variables (i.e., we will pretend the directive is not there), but when writing the file
    back, the directive will be preserved. This might introduce unexpected changes.

    Also includes fixes for some non-standard things we have come across. In particular:
       - C style comments (/* This is a comment */). These are added by CESM/CIME. We simply remove them and do not put
         them back when writing to a file.
       - "#" before a variable declaration (without the "override"). Some experiments suggest the following behaviour
         from the MOM6 parser: "# variable = 1" is equivalent to "variable = 1", while "#variable = 1" is equivalent to
         "!variable = 1". We try to handle them accordingly and to preserve them when writing the file back.
         (Reference: https://github.com/COSIMA/mom6-panan/commit/80e4a872f2b24f2e41da87439dd342df0c643d00#r130376163)

    The changes are recorded as a "patch", which is a dictionary: the keys are the line numbers where changes
    were made, while the values are tuples containing a keyword describing the type of change and, optionally, a string.

    Args:
        mom6_input_str (str): Contents of the MOM6 parameter file to patch.

    Returns:
        tuple: Contents of the patched MOM6 parameter file and the patch that was applied.
    """
    # Define several patterns that need to be matched
    comment_pattern = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
    zstar_pattern = re.compile(r"Z\*")
    block_pattern = re.compile(r"KPP%|%KPP|CVMix_CONVECTION%|%CVMix_CONVECTION|CVMIX_DDIFF%|%CVMIX_DDIFF")
    override_directive_pattern = re.compile(r"^(#override\s*?)")
    incorrect_directive_pattern = re.compile(r"^(#\s+)")
    comment_directive_pattern = re.compile(r"^#((?!override)\w+\b\s*=\s*\w+$)")

    # Modify the input while recording the changes
    patch = {}
    output = ""
    lines = mom6_input_str.split("\n")
    for i in range(len(lines)):
        line = lines[i] + "\n"
        if zstar_pattern.search(line):
            patch[i] = ("zstar", line)
            output += zstar_pattern.sub("ZSTAR", line)
        elif block_pattern.search(line):
            patch[i] = ("block", line)
            output += block_pattern.sub("", line)
        elif override_directive_pattern.search(line):
            patch[i] = ("override", override_directive_pattern.match(line).group(0))
            output += override_directive_pattern.sub("", line)
        elif incorrect_directive_pattern.search(line):
            patch[i] = (
                "incorrect directive",
                incorrect_directive_pattern.match(line).group(0),
            )
            output += incorrect_directive_pattern.sub("", line)
        elif comment_directive_pattern.search(line):
            patch[i] = ("comment_directive", line)
            output += "\n"
        else:
            output += line

    # Remove all C-style comments. These are not recorded and will not be undone.
    def replace_comment(match):
        return "\n" * match.group().count("\n")

    output = comment_pattern.sub(replace_comment, output)

    return output, patch


def _unpatch_mom6_input_str(mom6_input_str: str, patch: dict = None) -> str:
    """Undo the changes that were done to a MOM6 parameter file to make it into a conforming Fortran namelist.

    Args:
        mom6_input_str (str): Contents of the MOM6 parameter file to unpatch.
        patch (dict):  A dict containing the patch to revert.

    Returns:
        str: Unpatched contents of the MOM6 parameter  file.
    """
    output = ""
    lines = mom6_input_str.split("\n")[1:-2]
    for i in range(len(lines)):
        line = lines[i] + "\n"
        if i in patch:
            if patch[i][0] == "block":
                output += patch[i][1]
            elif patch[i][0] == "zstar":
                output += re.sub(r"ZSTAR", "Z*", line)
            elif patch[i][0] == "override":
                output += patch[i][1] + line
            elif patch[i][0] == "incorrect directive":
                output += patch[i][1] + line
            elif patch[i][0] == "comment_directive":
                output += patch[i][1]
        else:
            line = line.lstrip() if line != "\n" else line
            output += line
    return output


def _mom6_input_str_to_nml_str(mom6_input_str: str) -> str:
    """Convert the MOM6 parameter file to a conforming Fortran namelist.

    Args:
        mom6_input_str (str): Contents of the MOM6 parameter file.

    Returns:
        str: Fortran namelist.
    """
    return "&mom6\n" + mom6_input_str + "\n/"


def _nml_str_to_mom6_input_str(nml_str: str) -> str:
    """Convert a Fortran namelist into a MOM6 parameter file.

    Args:
        nml_str (str): Fortran namelist.

    Returns:
        str: MOM6 parameter file.
    """
    lines = nml_str.split("\n")
    lines = lines[1:-2]
    return "\n".join(lines)


def _mom6_input_to_nml_str(mom6_input: dict) -> str:
    """Convert MOM6 parameters stored in a dictionary into a Fortran namelist.

    Args:
        mom6_input (dict): Dictionary of MOM6 parameters.

    Returns:
        str: Fortran namelist.
    """
    output_file = StringIO("")
    nml = f90nml.Namelist({"mom6": mom6_input})
    nml.uppercase = True
    nml.false_repr = "False"
    nml.true_repr = "True"
    nml.indent = 0
    nml.write(output_file)
    return output_file.getvalue()


def _nml_str_to_mom6_input(nml_str: str) -> dict:
    """Convert MOM6 parameters stored as a Fortran namelist into a dictionary.

    Args:
        nml_str (str): Fortran namelist.

    Returns:
        dict: Dictionary of MOM6 parameters.
    """
    parser = f90nml.Parser()
    nml = parser.reads(nml_str)
    nml.uppercase = True
    return dict(nml.todict()["mom6"])


class Mom6Input(dict):
    """Class to read, store, modify and write a MOM6 parameter file.

    This class is used to enable round-trip parsing of MOM6 parameter files.
    It overrides the dict methods to:
      - stored all the keys in upper case
      - keep track of the changes done to the original dictionary

    It also stores the "patch" that was applied to the mom6_input_str to convert it to a conforming Fortran namelist.
    This is used to "undo" the changes when writing the file.
    """

    # Patched contents of the file to make it look like proper f90 namelist
    _mom6_input_str_patched = None

    # Dictionary containing information that can be used to reconstruct the original file from the output of f90nml
    _file_patch = {}

    # A record of all the changes done to the dictionary that can be passed to f90nml to do round-trip parsing
    _nml_patch = None

    # A record of keys that have been deleted from the dictionary
    _deleted_keys = []

    def __init__(self, file_name: str = None):
        """Read NOM6 parameters from file.

        Args:
            file_name (str): Name of file to read.
        """
        # Open file and read contents
        file = Path(file_name)
        if not file.is_file():
            raise FileNotFoundError(f"File not found: {file.as_posix()}")

        with open(file, "r") as f:
            mom6_input_str = f.read()

        # Convert file contents to dictionary
        self._mom6_input_str_patched, self._file_patch = _patch_mom6_input_str(mom6_input_str)
        nml_str = _mom6_input_str_to_nml_str(self._mom6_input_str_patched)
        mom6_input = _nml_str_to_mom6_input(nml_str)

        # Initialize class dictionary
        super().__init__(mom6_input)
        self._keys_to_upper()

        # Initialize nml patch
        self._nml_patch = {"mom6": {}}

    def __setitem__(self, key, value):
        """Override method to add item to dict.

        This method takes into account that all keys should be stored in uppercase. It also adds the new item to the
        namelist patch used for round-trip parsing.
        """
        super().__setitem__(key.upper(), value)

        if key.upper() in self._deleted_keys:
            self._deleted_keys.remove(key.upper())

        if self._nml_patch:
            self._nml_patch["mom6"][key.upper()] = value

    def __getitem__(self, key):
        """Override method to get item from dict, taking into account all keys are stored in uppercase."""
        return super().__getitem__(key.upper())

    def __delitem__(self, key):
        """Override method to delete item from dict, so that all keys are stored in uppercase."""
        self._deleted_keys.append(key.upper())
        super().__delitem__(key.upper())

    def write(self, file: Path):
        """Write contents of MOM6Input to a file.

        Args:
            file (Path): File to write to.
        """
        # Streams to pass to f90nml
        nml_file = StringIO(_mom6_input_str_to_nml_str(self._mom6_input_str_patched))
        tmp_file = StringIO("")

        parser = f90nml.Parser()
        parser.read(nml_file, self._nml_patch, tmp_file)
        mom6_input_str = _unpatch_mom6_input_str(tmp_file.getvalue(), self._file_patch)

        # Change keys to uppercase using a regex substitution, as there seems to be no way of doing this with f90nml
        # when applying a nml patch.
        mom6_input_str = re.sub(r"((?<=^)|(?<=\n))(\w+)", lambda pat: pat.group(2).upper(), mom6_input_str)

        # Explicitly removed keys from string
        for key in self._deleted_keys:
            mom6_input_str = re.sub(r"\s*" + f"{key}" + r"\s*=\s*\S*\s*\n", r"\n", mom6_input_str)

        file.write_text(mom6_input_str)

    def _keys_to_upper(self):
        """Change all keys in dictionary to uppercase."""
        for key in list(self.keys()):
            if not key.isupper():
                self[key.upper()] = self.pop(key)


def read_mom6_input(file_name: str) -> Mom6Input:
    """Read the contents of a MOM6 parameter file and return its contents as an instance of the MOM6Input class.

    Args:
        file_name: Name of MOM6 parameter file to read.

    Returns:
        MOM6Input: Contents of parameter file.
    """
    return Mom6Input(file_name)


def write_mom6_input(mom_input: [dict | Mom6Input], file: Path):
    """Write MOM6 parameters stored either as a dict of a MOM6Input to a file.

    Args:
        mom_input (dict|MOM6Input): MOM6 parameters.
        file (Path): File to write to.
    """
    if isinstance(mom_input, Mom6Input):
        mom_input.write(file)
    else:
        nml_str = _mom6_input_to_nml_str(mom_input)
        mom6_input_str = _nml_str_to_mom6_input_str(nml_str) + "\n"
        file.write_text(mom6_input_str)
