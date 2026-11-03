# Reading and writing

Every format has a parser, imported from `access.config`:

| Format | Parser |
|---|---|
| [Fortran namelists](formats/fortran-namelists.md) | `FortranNMLParser` |
| [MOM6 parameter files](formats/mom6.md) | `MOM6InputParser` |
| [NUOPC run configurations](formats/nuopc-runconfig.md) | `NUOPCParser` |
| [NUOPC run sequences](formats/nuopc-runseq.md) | `NUOPCRunSeqParser` |
| [FMS field tables](formats/field-tables.md) | `FieldTableParser` |
| [YAML](formats/yaml-json.md) | `YAMLParser` |
| [JSON](formats/yaml-json.md#json) | `JSONParser` |

All of them work the same way: `parse` takes the text of a file and returns the configuration it
holds, and `str` turns the configuration back into text. This page describes what the parsers
built on a grammar have in common, which is all of them but the YAML and JSON ones. Those two
wrap a library of their own, and [YAML and JSON](formats/yaml-json.md) describes how they differ.

## Parsing

Here are the first parameters of the `MOM_input` of an ACCESS-OM3 configuration, with MOM6's
documentation of each one in its comments, and one of its parameter blocks:

```text title="MOM_input"
! === module MOM ===
DIABATIC_FIRST = True           !   [Boolean] default = False
                                ! If true, apply diabatic and thermodynamic processes, including buoyancy
                                ! forcing and mass gain or loss, before stepping the dynamics forward.
DT = 1800.0                     !   [s]
                                ! The (baroclinic) dynamics time step.
DT_THERM = 3600.0               !   [s] default = 1800.0
                                ! The thermodynamic and tracer advection time step.
KPP%
N_SMOOTH = 3                    ! default = 0
                                ! The number of times the 1-1-4-1-1 Laplacian filter is applied on OBL depth.
MATCH_TECHNIQUE = "MatchGradient" ! default = "SimpleShapes"
%KPP
```

`parse` takes the text, not the name of the file, so the file is read first:

```python
from pathlib import Path

from access.config import MOM6InputParser

config = MOM6InputParser().parse(Path("MOM_input").read_text())
print(config["DT"], config["DIABATIC_FIRST"], config["KPP"]["MATCH_TECHNIQUE"])
```

```text
1800.0 True MatchGradient
```

The configuration is a `Config`, which is a `dict`, and a block of entries, here MOM6's `KPP%`
block, is a nested `Config`. Everything a dict can be asked, it answers as a dict would:

```python
print(isinstance(config, dict), list(config))
print(len(config["KPP"]), "N_SMOOTH" in config["KPP"], config.get("DTBT", "not set"))
```

```text
True ['DIABATIC_FIRST', 'DT', 'DT_THERM', 'KPP']
2 True not set
```

Values are read as Python values: an `int`, a `float`, a `str`, a `bool`, or `None` for a key
written without a value, as well as a `complex` in a Fortran namelist and a `pathlib.Path` in a
NUOPC run configuration. A value written as a list is a `ConfigList`, which is a `list`. The page
of each format says how its values are spelled and what they are read as.

## Keys

How keys are matched depends on the format, and follows what the model that reads the file does:

| Format | Keys |
|---|---|
| Fortran namelists | Matched in any case, and read in upper case: `config["fms_nml"]` is `config["FMS_NML"]` |
| FMS field tables | Matched in any case, and read in lower case |
| All the others | Matched exactly: `DT` and `dt` are two keys |

A key added to the configuration is written into the file exactly as it was spelled, whatever
case the dict holds it in.

## Writing

`str` gives the text of the configuration. Until something is changed, that is the text it was
parsed from, byte for byte:

```python
text = Path("MOM_input").read_text()
print(str(MOM6InputParser().parse(text)) == text)
```

```text
True
```

After an edit, it is that text with the edit made, and nothing else changed: the comments, the
blank lines, the spacing and the spelling of every value that was not touched are as they were.
[Editing](editing.md) describes what can be changed.

The `str` of a block is the text of its body:

```python
print(str(config["KPP"]))
```

```text
N_SMOOTH = 3                    ! default = 0
                                ! The number of times the 1-1-4-1-1 Laplacian filter is applied on OBL depth.
MATCH_TECHNIQUE = "MatchGradient" ! default = "SimpleShapes"
```

### Writing to a file

To write the configuration back to its file, write its text to a temporary file next to it and
move that over the original. If anything goes wrong while writing, the original is left whole:

```python
path = Path("MOM_input")
config = MOM6InputParser().parse(path.read_text())
config["DT"] = 900.0

temporary = path.with_name(path.name + ".tmp")
temporary.write_text(str(config))
temporary.replace(path)
print(path.read_text().splitlines()[4])
```

```text
DT = 900.0                     !   [s]
```

Only the value was rewritten, so the spaces after it are those of the file, and the comment
starts one column earlier than before.

## Malformed files

A text the format does not allow raises one of Lark's `UnexpectedInput` exceptions, which say
where the problem is:

```python
from lark.exceptions import UnexpectedInput

try:
    MOM6InputParser().parse("DT = 1800.0\nDT_THERM 3600.0\n")
except UnexpectedInput as error:
    print(type(error).__name__, "at line", error.line, "column", error.column)
```

```text
UnexpectedCharacters at line 2 column 10
```

A parser reads only what its format allows, and it refuses the rest rather than guessing. Each
format's page lists what it does not read.
