# Fortran namelists

`FortranNMLParser` reads Fortran namelists, the format most components of the ACCESS models are
configured with: the `input.nml` of MOM and FMS, the `ice_in` or `cice_in.nml` of CICE, and the
namelists of many other components.

```python
from access.config import FortranNMLParser

parser = FortranNMLParser()
```

## How a namelist reads

A namelist is a sequence of groups, each a block of assignments. The configuration is a dict of
the groups, and each group a dict of its variables:

```fortran title="input.nml"
&ocean_nml
    dt = 1800.0      ! seconds
    layout = 48, 40
    use_ice = .true.
    grid%nx = 360
    grid%ny = 300
/
```

```python
from pathlib import Path

config = parser.parse(Path("input.nml").read_text())
print(dict(config["ocean_nml"]))
```

```text
{'DT': 1800.0, 'LAYOUT': [48, 40], 'USE_ICE': True, 'GRID': {'NX': 360, 'NY': 300}}
```

Fortran does not tell upper case from lower case in names, so the keys are matched in any case,
and are read in upper case. A key that is added is written as it is spelled.

A group opens with `&` or the older `$`, and closes with `/`, `&end` or `$end`. Comments start
with `!`. Text between the groups, which Fortran ignores, is kept in the file but not read.

## Values

| Written | Read as |
|---|---|
| `1`, `-2` | `int` |
| `1.5`, `1.0e-3`, `1.0d-3`, `1+0` | `float` |
| `(1.0, 2.0)` | `complex` |
| `'it''s'`, `"abc"` | `str`; a quote is doubled inside a string |
| `.true.`, `T`, `.FALSE.`, `f` | `bool` |
| `abc`, a bare word | `str` |
| `x =`, nothing at all | `None` |

A new value is written in the notation of the one it replaces, as [Editing](../editing.md)
describes, and a new string is written in quotes: Fortran doubles a quote that is part of the
string, so any string can be written. `inf` and `nan` are read as the floats they are, but cannot
be written, since no namelist reader reads Python's spelling of them back; assigning one raises
`UnsupportedEntryError`.

## Lists

A list is written with its values parted by commas or by blanks: `layout = 48, 40` and
`layout = 48 40` are the same list. Fortran also has several ways of writing part of an array,
and all of them are read:

| Written | Read as |
|---|---|
| `v = 1, , 3` | `[1, None, 3]`: a skipped position is `None` |
| `v = 3*0.5` | `[0.5, 0.5, 0.5]`: a repeat count |
| `v(1) = 1` and `v(3) = 3` | `[1, None, 3]`: indexed assignments make up one list |
| `v(2:3) = 7, 8` | `[7, 8]` |
| `v = 1, 2` and then `v = 3` | `[3, 2]`: a later assignment writes from the first position, and leaves the rest |

A list holds the positions from the first one written to the last, without Fortran's indices,
and a position between them that no entry writes reads as `None`. Such a position has no text in
the file, so it cannot be given a value: assign a whole new list instead.

A list written in any of these ways can have its elements changed, but not its length. The file
then says more than the values: which position each entry writes, or how many times a value is
repeated, and there is no single way of writing a longer or a shorter list that keeps saying it.
Changing the length raises `ValueError`; delete the key and assign it again, and it is written as
one plain list. The elements a repeat count stands for are changed together, all to the same
value, which rewrites the count's value: `3*0.5` becomes `3*0.25`. Giving one of them a value of
its own raises `ValueError`.

Two qualifiers are not gathered into a list, and stay in the key instead: a multidimensional one,
`v(1,2) = 4`, which is the key `V(1,2)`, and one on a derived type, `a(1)%b = 2`, which is the key
`A(1)`. A list of derived types written one element per line, as WW3 writes them
(`INBND_POINT(1) = 2 2 F`), therefore reads as one list of what their components hold.

## Derived types

A variable written with a `%` is a component of a derived type, and the type reads as a nested
dict: `grid%nx = 360` and `grid%ny = 300` are the type `GRID`, holding `NX` and `NY`. A derived type
is only ever written one component per line, so that is how a new component is written, after the
type's last line, and how a dict assigned to a new variable is written:

```python
ocean = config["ocean_nml"]
ocean["grid"]["nz"] = 75
ocean["tracers"] = {"temp": True, "salt": True}
print(str(config))
```

```text
&ocean_nml
    dt = 1800.0      ! seconds
    layout = 48, 40
    use_ice = .true.
    grid%nx = 360
    grid%ny = 300
    grid%nz = 75
    tracers%temp = .true.
    tracers%salt = .true.
/
```

A variable cannot be both a value and a derived type, as in `a = 1` and `a%b = 2` in one group;
such a group raises `ValueError` when it is parsed.

## Groups written more than once

A program reading a group with one `READ` gets the first group of that name, and one reading it in
a loop gets each in turn, so a group written twice is two groups rather than one merged:

```fortran title="diag.nml"
&output_nml
    name = 'daily'
/
&output_nml
    name = 'monthly'
/
```

```python
diag = parser.parse(Path("diag.nml").read_text())
outputs = diag["output_nml"]
print(type(outputs).__name__, [output["name"] for output in outputs])
outputs[1]["name"] = "yearly"
print(str(diag))
```

```text
ConfigBlockList ['daily', 'monthly']
&output_nml
    name = 'daily'
/
&output_nml
    name = 'yearly'
/
```

`config["output_nml"]` is then a `ConfigBlockList`, a list holding one configuration per group,
in the order they are written. The entries of each are edited as any group's are. The list
itself cannot be changed: adding, removing or replacing a whole group raises
`NotImplementedError`, since nothing says where a new one would go. Deleting the key deletes
every group of that name.

## What is not read

- Character substrings, `name(1:4) = 'abcd'`, and a character value continued across lines.
- A group that is not closed: neither the next `&` nor a bare `$` ends one.
- Text a group's body cannot hold raises rather than being taken for free text around the
  groups: a file that would write back with a whole group missing from the configuration is
  worse than one that is refused.

A comparison of this parser with [f90nml](https://github.com/marshallward/f90nml), run against
f90nml's own test suite, is described in [Contributing](../../developer-guide/contributing.md#comparing-with-f90nml).
