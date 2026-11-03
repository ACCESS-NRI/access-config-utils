# MOM6 parameter files

`MOM6InputParser` reads the parameter files of the MOM6 ocean model, such as the `MOM_input` of an
ACCESS-OM3 configuration. The format is described in [MOM6's
documentation](https://mom6.readthedocs.io/en/main/api/generated/pages/Runtime_Parameter_System.html#mom6-parameter-file-syntax).
You create the parser like this:

```python
from access.config import MOM6InputParser

parser = MOM6InputParser()
```

## How a parameter file reads

A parameter file is a list of assignments, one on each line, and the configuration is a dict of
them. A parameter block, which opens with `NAME%` and closes with `%NAME`, is a nested dict. As an
example, we use this file:

```text title="MOM_input"
DT = 1800.0                     !   [s]
                                ! The (baroclinic) dynamics time step.
FRAZIL = True                   !   [Boolean] default = False
KPP%
N_SMOOTH = 3                    ! default = 0
MATCH_TECHNIQUE = "MatchGradient" ! default = "SimpleShapes"
%KPP
```

Save it as `MOM_input` in the current directory, or adapt the path to where your file is. We parse
it and print two parameters and the block:

```python
from pathlib import Path

config = parser.parse(Path("MOM_input").read_text())
print(config["DT"], config["FRAZIL"], dict(config["KPP"]))
```

This prints:

```text
1800.0 True {'N_SMOOTH': 3, 'MATCH_TECHNIQUE': 'MatchGradient'}
```

Keys are case-sensitive, as they are for MOM6. Comments start with `!`, and are kept wherever they
are. This includes the lines of documentation that MOM6 writes under each parameter of the files it
produces.

## Values

| Written | Read as |
|---|---|
| `3` | `int` |
| `1800.0`, `1.0E-03` | `float` |
| `"MatchGradient"`, `'abc'` | `str`; a string is always quoted |
| `True`, `False` | `bool` |
| `48, 40` | a list |

A parameter cannot be written without a value, so assigning `None` raises `UnsupportedEntryError`.
So does a string that contains both quote characters, because MOM6 has no way of escaping a quote. A
new string is written in the quotes of the string it replaces, or of the nearest string above it.

## Editing

A new parameter of a block is written after the last parameter of that block. A dict assigned to a
new key is written as a new block, after the last line of the file. Here we do both:

```python
config["KPP"]["ENHANCE_DIFFUSION"] = False
config["CVMIX_DDIFF"] = {"USE_CVMIX_DDIFF": True, "STRAT_RATIO_DDIFF": 2.55}
print(str(config))
```

This prints:

```text
DT = 1800.0                     !   [s]
                                ! The (baroclinic) dynamics time step.
FRAZIL = True                   !   [Boolean] default = False
KPP%
N_SMOOTH = 3                    ! default = 0
MATCH_TECHNIQUE = "MatchGradient" ! default = "SimpleShapes"
ENHANCE_DIFFUSION = False
%KPP
CVMIX_DDIFF%
USE_CVMIX_DDIFF = True
STRAT_RATIO_DDIFF = 2.55
%CVMIX_DDIFF
```

Deleting a parameter deletes the comment on its line. It does not delete the lines of documentation
below it, which are comments of their own. A parameter written twice is read as the last value it is
given, as MOM6 reads it, and deleting it deletes both lines.

## What is not read

- `#override` directives. A `MOM_override` file, which is made of them, cannot be read.
- C-style comments, `/* ... */`. MOM6 ignores them, and some CESM files have them, but they are
  not part of the format.
- Blocks nested in blocks, an assignment indented from the start of its line, a logical spelled
  in lower case (`true`) and a string that is not quoted.
