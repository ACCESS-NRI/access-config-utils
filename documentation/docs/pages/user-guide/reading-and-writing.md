# Reading and writing

Every format has its own parser, which you import from `access.config`:

| Format | Parser |
|---|---|
| [Fortran namelists](formats/fortran-namelists.md) | `FortranNMLParser` |
| [MOM6 parameter files](formats/mom6.md) | `MOM6InputParser` |
| [NUOPC run configurations](formats/nuopc-runconfig.md) | `NUOPCParser` |
| [NUOPC run sequences](formats/nuopc-runseq.md) | `NUOPCRunSeqParser` |
| [FMS field tables](formats/field-tables.md) | `FieldTableParser` |
| [YAML](formats/yaml-json.md) | `YAMLParser` |
| [JSON](formats/yaml-json.md#json) | `JSONParser` |

All the parsers work in the same way. `parse` takes the text of a file and returns the configuration
it holds, and `str` turns the configuration back into text. Here we look at what the parsers that
are built on a grammar have in common. That is all of them except the YAML and JSON parsers, which
wrap a library of their own. [YAML and JSON](formats/yaml-json.md) describes how those two differ.

## Parsing

As an example, we use the first parameters of the `MOM_input` of an ACCESS-OM3 configuration,
together with one of its parameter blocks. The comments are MOM6's documentation of each parameter:

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

Save it as `MOM_input` in the current directory, or adapt the path in the examples to where your
file is. Note that `parse` takes the text of the file, and not its name, so we read the file first:

```python
from pathlib import Path

from access.config import MOM6InputParser

config = MOM6InputParser().parse(Path("MOM_input").read_text())
print(config["DT"], config["DIABATIC_FIRST"], config["KPP"]["MATCH_TECHNIQUE"])
```

This prints:

```text
1800.0 True MatchGradient
```

The configuration is a `Config`, which is a `dict`. A block of entries, such as MOM6's `KPP%` block
here, is a nested `Config`. You can use everything that a dict offers:

```python
print(isinstance(config, dict), list(config))
print(len(config["KPP"]), "N_SMOOTH" in config["KPP"], config.get("DTBT", "not set"))
```

The output is what a dict would give:

```text
True ['DIABATIC_FIRST', 'DT', 'DT_THERM', 'KPP']
2 True not set
```

The values are read as Python values: an `int`, a `float`, a `str` or a `bool`. A key written
without a value is read as `None`. A Fortran namelist can also hold a `complex`, and a NUOPC run
configuration a `pathlib.Path`. A value written as a list is a `ConfigList`, which is a `list`. The
page of each format explains how its values are written and what they are read as.

## Keys

How the keys are matched depends on the format. Each parser does what the model that reads the file
does:

| Format | Keys |
|---|---|
| Fortran namelists | Matched in any case, and read in upper case: `config["fms_nml"]` is `config["FMS_NML"]` |
| FMS field tables | Matched in any case, and read in lower case |
| All the others | Matched exactly: `DT` and `dt` are two keys |

When you add a key to the configuration, it is written into the file exactly as you spelled it,
whatever case the dict holds it in.

## Writing

`str` gives the text of the configuration. As long as you have not changed anything, this is the
text that was parsed, byte for byte. We can check this:

```python
text = Path("MOM_input").read_text()
print(str(MOM6InputParser().parse(text)) == text)
```

This prints:

```text
True
```

After an edit, `str` gives that text with the edit made, and nothing else changed. The comments, the
blank lines, the spacing and the way every other value is written are as they were.
[Editing](editing.md) describes what you can change.

The `str` of a block is the text of its body. For example, for the `KPP` block:

```python
print(str(config["KPP"]))
```

This prints:

```text
N_SMOOTH = 3                    ! default = 0
                                ! The number of times the 1-1-4-1-1 Laplacian filter is applied on OBL depth.
MATCH_TECHNIQUE = "MatchGradient" ! default = "SimpleShapes"
```

### Writing to a file

To write the configuration back to its file, write its text to a temporary file next to the
original, and then move the temporary file over it. This way, if anything goes wrong while writing,
the original is left whole. Here we change the time step and write the file back:

```python
path = Path("MOM_input")
config = MOM6InputParser().parse(path.read_text())
config["DT"] = 900.0

temporary = path.with_name(path.name + ".tmp")
temporary.write_text(str(config))
temporary.replace(path)
print(path.read_text().splitlines()[4])
```

The last line prints the line of the file that we changed:

```text
DT = 900.0                     !   [s]
```

Only the value was rewritten. The spaces after it are therefore those of the file, and the comment
starts one column earlier than before.

## Malformed files

What happens when a file is not valid? A text that the format does not allow raises one of Lark's
`UnexpectedInput` exceptions, which says where the problem is. Here the second line has no `=`:

```python
from lark.exceptions import UnexpectedInput

try:
    MOM6InputParser().parse("DT = 1800.0\nDT_THERM 3600.0\n")
except UnexpectedInput as error:
    print(type(error).__name__, "at line", error.line, "column", error.column)
```

This prints:

```text
UnexpectedCharacters at line 2 column 10
```

A parser only reads what its format allows. It refuses everything else, and does not guess. The page
of each format lists what it does not read.
