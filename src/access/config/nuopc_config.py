# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Parser for NUOPC configuration files.

The `nuopc.runconfig` files used by the CESM driver, and thus by the NUOPC-based ACCESS models, are a mixture of
formats. At the top-level, one has the Resource Files as implemented in ESMF. From the ESMF documentation:

    A Resource File (RF) is a text file consisting of list of label-value pairs. There is a limit of 1024 characters per
    line and the Resource File can contain a maximum of 200 records. Each label should be followed by some data, the
    value. An example Resource File follows. It is the file used in the example below.

     # This is an example Resource File.
     # It contains a list of <label,value> pairs.
     # The colon after the label is required.

     # The values after the label can be an list.
     # Multiple types are authorized.

      my_file_names:         jan87.dat jan88.dat jan89.dat  # all strings
      constants:             3.1415   25                    # float and integer
      my_favorite_colors:    green blue 022


     # Or, the data can be a list of single value pairs.
     # It is simplier to retrieve data in this format:

      radius_of_the_earth:   6.37E6
      parameter_1:           89
      parameter_2:           78.2
      input_file_name:       dummy_input.nc

     # Or, the data can be located in a table using the following
     # syntax:

      my_table_name::
       1000     3000     263.0
        925     3000     263.0
        850     3000     263.0
        700     3000     269.0
        500     3000     287.0
        400     3000     295.8
        300     3000     295.8
      ::

    Note that the colon after the label is required and that the double colon is required to declare tabular data.

See https://earthsystemmodeling.org/docs/release/ESMF_8_6_0/ESMF_refdoc/node6.html#SECTION06090000000000000000 for
further details.


The CESM driver then uses tables as defined in Resource Files to store lists of key-value pairs instead of simple
values:

    DRIVER_attributes::
     Verbosity = off
     cime_model = cesm
     logFilePostFix = .log
     pio_blocksize = -1
     pio_rearr_comm_enable_hs_comp2io = .true.
     pio_rearr_comm_enable_hs_io2comp = .false.
     reprosum_diffmax = -1.000000D-08
    ::

    ALLCOMP_attributes::
     ATM_model = datm
     GLC_model = sglc
     OCN_model = mom
     ocn2glc_levels = 1:10:19:26:30:33:35
    ::

This format of key-value pairs does not seem to be documented and, although it resembles Fortran namelists, it is not.
For example, the keys are case-sensitive, which is not the case in Fortran namelists. The format used to store arrays
of values is also not the same as in Fortran namelists.
"""

from access.config.parser import ConfigParser


class NUOPCParser(ConfigParser):
    """NUOPC config parser."""

    @property
    def case_sensitive_keys(self) -> bool:
        return True

    @property
    def grammar(self) -> str:
        return """
?start: lines*

?lines: rfile_key_value
      | rfile_key_list
      | rfile_key_block
      | empty_line

rfile_key_value: ws* key ":" ws* value line_end -> key_value
rfile_key_list: ws* key ":" ws* value (ws* value)+ line_end -> key_list
rfile_key_block: ws* key "::" line_end block "::" line_end -> key_block

block: block_line*

?block_line: block_key_value
           | block_key_list
           | empty_line

block_key_value : ws* key ws* "=" ws* value line_end -> key_value
block_key_list : ws* key ws* "=" ws* value (":"value)+ line_end -> key_list

?value: logical
      | integer
      | float
      | double
      | identifier
      | path

empty_line: line_end
line_end: (comment|ws*) NEWLINE

%import config.key
%import config.logical
%import config.integer
%import config.float
%import config.double
%import config.identifier
%import config.path
%import config.comment
%import config.ws
%import config.NEWLINE
"""
