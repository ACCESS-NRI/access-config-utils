# NUOPC run configurations

`NUOPCParser` reads the `nuopc.runconfig` file of a model driven by CMEPS, the NUOPC mediator,
such as ACCESS-OM3. The file is an
[ESMF Resource File](https://earthsystemmodeling.org/docs/release/ESMF_8_6_0/ESMF_refdoc/node6.html#SECTION06090000000000000000),
whose tables the CESM driver uses to hold lists of `key = value` attributes.

```python
from access.config import NUOPCParser

parser = NUOPCParser()
```

## How a run configuration reads

A line `label: value` is an entry of the configuration, and a table, opened by `NAME::` and closed
by a line `::`, is a nested dict of its `key = value` lines. Here is an excerpt of the
`nuopc.runconfig` of an ACCESS-OM3 configuration:

```text title="nuopc.runconfig"
DRIVER_attributes::
     Verbosity = off
     pio_rearr_comm_enable_hs_comp2io = .true.
     reprosum_diffmax = -1.0e-8
     timing_dir = ./timing
     wv_sat_table_spacing = 1.0D0
::

component_list: MED ATM ICE OCN ROF
ALLCOMP_attributes::
     OCN_model = mom
     case_name = access-om3
     ocn2glc_levels = 1:10:19:26:30:33:35
     stop_n = 1
     stop_option = nmonths
::
```

```python
from pathlib import Path

config = parser.parse(Path("nuopc.runconfig").read_text())
print(config["component_list"])
for key, value in config["ALLCOMP_attributes"].items():
    print(key, repr(value))
```

```text
['MED', 'ATM', 'ICE', 'OCN', 'ROF']
OCN_model 'mom'
case_name PosixPath('access-om3')
ocn2glc_levels [1, 10, 19, 26, 30, 33, 35]
stop_n 1
stop_option 'nmonths'
```

Keys are case-sensitive. A list is written with its values parted by blanks at the top level,
and by colons in a table. Comments start with `#`.

## Values

| Written | Read as |
|---|---|
| `1`, `-999` | `int` |
| `-1.0e-8`, `1.0D0` | `float` |
| `.true.`, `.false.` | `bool` |
| `mom`, `nmonths`, `off`, `true`, a bare name | `str` |
| `./timing`, `rpointer.cpl`, `access-om3`, `2denable`, any other bare word | `pathlib.Path` |
| `label:` or `key =`, with nothing after it | `None` |

The format has no quoted strings, so a word is a name or a path depending on how it is spelled. A
name is made of letters, digits and underscores, and does not start with a digit. Any other word
made of those and of `.`, `-` and `/` is a path, and a word holding any other character is not
read. That makes `case_name` a `Path` above, and only the dotted spellings, `.true.` and
`.false.`, logicals: `true` is a name.

## Editing

Since a value's type cannot change, a value read as a `Path` is given a `Path`, and a new value
holding a `/`, a `.` or a `-` has to be one too: there is no way of writing it as a `str`, so
assigning one raises `UnsupportedEntryError`. A `Path` is written as `pathlib` spells it, which
drops a leading `./`:

```python
attributes = config["ALLCOMP_attributes"]
attributes["case_name"] = Path("access-om3-25km")
attributes["stop_n"] = 12
config["DRIVER_attributes"]["timing_dir"] = Path("./timing-12")
config["component_list"].append("WAV")
print(str(config))
```

```text
DRIVER_attributes::
     Verbosity = off
     pio_rearr_comm_enable_hs_comp2io = .true.
     reprosum_diffmax = -1.0e-8
     timing_dir = timing-12
     wv_sat_table_spacing = 1.0D0
::

component_list: MED ATM ICE OCN ROF WAV
ALLCOMP_attributes::
     OCN_model = mom
     case_name = access-om3-25km
     ocn2glc_levels = 1:10:19:26:30:33:35
     stop_n = 12
     stop_option = nmonths
::
```

A dict assigned to a new key is written as a new table, and `None` as an entry with no value:

```python
config["MED_attributes"] = {"history_n": 1, "history_option": "nmonths"}
config["case_label"] = None
print("\n".join(str(config).splitlines()[16:]))
```

```text
MED_attributes::
     history_n = 1
     history_option = nmonths
::
case_label:
```

## What is not read

- Tables nested in tables, and a table closed by an indented `::`.
- Comments starting with anything other than `#`.
- Quoted strings: a value holding a blank, or a character other than letters, digits, `_`, `.`,
  `-` and `/`, cannot be read or written.
