# Fortran namelists

`FortranNMLParser` reads Fortran namelists. This is the format that most components of the ACCESS
models are configured with: the `input.nml` of MOM and FMS, the `ice_in` or `cice_in.nml` of CICE,
and the namelists of many other components. You create the parser like this:

```python
from access.config import FortranNMLParser

parser = FortranNMLParser()
```

## How a namelist reads

A namelist is a sequence of groups, and each group is a block of assignments. The configuration is a
dict of the groups, and each group is a dict of its variables. As an example, we use this namelist:

```fortran title="input.nml"
&ocean_nml
    dt = 1800.0      ! seconds
    layout = 48, 40
    use_ice = .true.
    grid%nx = 360
    grid%ny = 300
/
```

Save it as `input.nml` in the current directory, or adapt the path to where your file is. We parse
it and print its group as a plain dict:

```python
from pathlib import Path

config = parser.parse(Path("input.nml").read_text())
print(dict(config["ocean_nml"]))
```

This prints:

```text
{'DT': 1800.0, 'LAYOUT': [48, 40], 'USE_ICE': True, 'GRID': {'NX': 360, 'NY': 300}}
```

Fortran does not tell upper case from lower case in names. The keys are therefore matched in any
case, and read in upper case. A key that you add is written as you spelled it.

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

A new value is written in the same notation as the one it replaces, as [Editing](../editing.md)
describes. A new string is written in quotes. Fortran doubles a quote that is part of the string, so
any string can be written. `inf` and `nan` are read as the floats they are, but they cannot be
written, because no namelist reader reads Python's spelling of them back. Assigning one raises
`UnsupportedEntryError`.

## Lists

In a list, the values are separated by commas or by blanks: `layout = 48, 40` and `layout = 48 40`
are the same list. Fortran also has several ways of writing part of an array, and all of them are
read:

| Written | Read as |
|---|---|
| `v = 1, , 3` | `[1, None, 3]`: a skipped position is `None` |
| `v = 3*0.5` | `[0.5, 0.5, 0.5]`: a repeat count |
| `v(1) = 1` and `v(3) = 3` | `[1, None, 3]`: indexed assignments make up one list |
| `v(2:3) = 7, 8` | `[7, 8]` |
| `v = 1, 2` and then `v = 3` | `[3, 2]`: a later assignment writes from the first position, and leaves the rest |

A list holds the positions from the first one written to the last, without Fortran's indices. A
position between them that no entry writes is read as `None`. Such a position has no text in the
file, so you cannot give it a value: assign a whole new list instead.

A list written in any of these ways can have its elements changed, but not its length. The reason is
that the file then says more than the values: which position each entry writes, or how many times a
value is repeated. There is no single way of writing a longer or a shorter list that still says the
same. Changing the length therefore raises `ValueError`. If you need another length, delete the key
and assign it again, and it is written as one plain list. The elements that a repeat count stands
for are changed together, all to the same value. This rewrites the value of the count: `3*0.5`
becomes `3*0.25`. Giving one of them a value of its own raises `ValueError`.

Two kinds of qualifier are not gathered into a list, and stay in the key instead. The first is a
multidimensional one: `v(1,2) = 4` is the key `V(1,2)`. The second is one on a derived type: `a(1)%b
= 2` is the key `A(1)`. A list of derived types written with one element on each line, as WW3 writes
them (`INBND_POINT(1) = 2 2 F`), is therefore read as one list of what their components hold.

## Derived types

A variable written with a `%` is a component of a derived type, and the type is read as a nested
dict. For example, `grid%nx = 360` and `grid%ny = 300` are the type `GRID`, which holds `NX` and
`NY`. A derived type is always written with one component on each line. A new component is therefore
written in the same way, after the last line of the type, and so is a dict that you assign to a new
variable. Here we add a component to `grid`, and a new derived type:

```python
ocean = config["ocean_nml"]
ocean["grid"]["nz"] = 75
ocean["tracers"] = {"temp": True, "salt": True}
print(str(config))
```

This prints:

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

A variable cannot be both a value and a derived type, as in `a = 1` and `a%b = 2` in the same group.
Such a group raises `ValueError` when it is parsed.

## Groups written more than once

What if a group is written twice? A program that reads a group with one `READ` gets the first group
of that name, and a program that reads it in a loop gets each group in turn. The two groups are
therefore kept as two groups, and are not merged. As an example, we use this namelist:

```fortran title="diag.nml"
&output_nml
    name = 'daily'
/
&output_nml
    name = 'monthly'
/
```

Save it as `diag.nml`. We parse it, look at the group, and change a value in its second occurrence:

```python
diag = parser.parse(Path("diag.nml").read_text())
outputs = diag["output_nml"]
print(type(outputs).__name__, [output["name"] for output in outputs])
outputs[1]["name"] = "yearly"
print(str(diag))
```

This prints:

```text
ConfigBlockList ['daily', 'monthly']
&output_nml
    name = 'daily'
/
&output_nml
    name = 'yearly'
/
```

`diag["output_nml"]` is a `ConfigBlockList`: a list that holds one configuration for each group, in
the order they are written. You edit the entries of each one as you would those of any group. The
list itself cannot be changed. Adding, removing or replacing a whole group raises
`NotImplementedError`, because nothing says where a new group would go. Deleting the key deletes
every group of that name.

## What is not read

- Character substrings, `name(1:4) = 'abcd'`, and a character value continued across lines.
- A group that is not closed: neither the next `&` nor a bare `$` ends one.
- Text that the body of a group cannot hold raises an error. It is not taken for free text around
  the groups, because the configuration would then be missing a whole group, which is worse than
  refusing the file.

This parser has been compared with [f90nml](https://github.com/marshallward/f90nml), by running it
against f90nml's own test suite. The comparison is described in
[Contributing](../../developer-guide/contributing.md#comparing-with-f90nml).
