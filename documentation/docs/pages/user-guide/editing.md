# Editing

A configuration is edited as a dict is: by assigning to a key, deleting one, or changing a list
in place. Each change is made to the text straight away, so `str` always gives the text of the
configuration as it now is. This page edits the first group of the `ice_in` file CICE reads in
an ACCESS-OM3 configuration:

```fortran title="ice_in"
&setup_nml
  conserv_check = .false.
  diagfreq = 960
  dumpfreq = "y"
  histfreq = "d", "m", "x", "x", "x"
  npt = 35040
/
```

What is described here holds for every parser built on a grammar, which is all of them but the
YAML and JSON ones. [YAML and JSON](formats/yaml-json.md) describes those two.

## Changing a value

Assigning to a key the configuration has rewrites its value, and nothing else:

```python
from pathlib import Path

from access.config import FortranNMLParser

config = FortranNMLParser().parse(Path("ice_in").read_text())
setup = config["setup_nml"]
setup["diagfreq"] = 480
setup["dumpfreq"] = "m"
setup["conserv_check"] = True
print(str(config))
```

```text
&setup_nml
  conserv_check = .true.
  diagfreq = 480
  dumpfreq = "m"
  histfreq = "d", "m", "x", "x", "x"
  npt = 35040
/
```

A new value is written in the notation of the one it replaces: with the same quotes, unless the
new string holds that quote character and the other one has to be used; a logical spelled the
same way (`T` becomes `F`, `.false.` becomes `.true.`); and a number with the same exponent letter
when it has one (a `d` stays a `d`).

The new value has to be of the same type as the old one, and is checked exactly: an `int` is not
a `float`, a `bool` is not an `int`, and a NumPy number is not a Python one. A value of another
type raises `TypeError`:

```python
try:
    setup["npt"] = 35040.0
except TypeError as error:
    print(error)
```

```text
Trying to change value type
```

To give a key a value of another type, delete it and assign it again. It is then written as a new
entry, at the end of its block.

## Lists

A value written as a list is a `ConfigList`. It is a `list`, and every way of changing a list
works on it: assigning an element or a slice, `del`, `append`, `extend`, `insert`, `pop`, `remove`,
`clear`, `sort`, `reverse`, `+=` and `*=`:

```python
histfreq = setup["histfreq"]
histfreq[2] = "y"
del histfreq[3:]
histfreq.insert(0, "h")
print(histfreq)
print(str(config).splitlines()[4])
```

```text
['h', 'd', 'm', 'y']
  histfreq = "h", "d", "m", "y"
```

The elements that stay keep their text, comments included, and a new element is written like its
neighbours, on their line or on a line of its own. Here is a list written one element per line:

```fortran title="history.nml"
&setup_nml
  histfreq = 'd',  ! daily
             'm'   ! monthly
/
```

```python
history = FortranNMLParser().parse(Path("history.nml").read_text())
history["setup_nml"]["histfreq"].append("y")
print(str(history))
```

```text
&setup_nml
  histfreq = 'd',  ! daily
             'm',
             'y'   ! monthly
/
```

A comment after an element belongs to it, and goes with it if it is removed. The comment after the
last element is different: it ends the whole entry, and stays at its end, which is why
`! monthly` now follows the new element.

A copy of a `ConfigList`, such as `histfreq[:]` or `list(histfreq)`, is a plain `list`, and
changing it changes nothing in the file.

### What a list cannot become

- A list keeps at least two elements, since a single value is a scalar. One that would be left
  with fewer raises `ValueError`.
- A list does not become a scalar, nor a scalar a list: that is a change of type, and raises
  `TypeError`. Delete the key and assign it again instead.
- In a Fortran namelist, the elements of a list written with a repeat count (`3*1.0`), with array
  qualifiers (`x(2) = ...`), with skipped positions (`1, , 3`) or over several assignments can be
  changed, but there cannot be more or fewer of them: changing the length raises `ValueError`.
  [Fortran namelists](formats/fortran-namelists.md#lists) explains why.

## Adding keys

Assigning to a key the configuration does not have adds an entry for it:

```python
setup["dump_last"] = True
setup["histfreq_n"] = [1, 1, 1, 1]
setup["restart_dir"] = None
config["grid_nml"] = {"grid_type": "tripole", "ncat": 5}
print(str(config))
```

```text
&setup_nml
  conserv_check = .true.
  diagfreq = 480
  dumpfreq = "m"
  histfreq = "h", "d", "m", "y"
  npt = 35040
  dump_last = .true.
  histfreq_n = 1, 1, 1, 1
  restart_dir =
/
&grid_nml
  grid_type = "tripole"
  ncat = 5
/
```

A new entry goes after the last entry of its block, or of the configuration, and never inside a
comment. It is written as a copy of the nearest entry of its kind, with that entry's spacing and
punctuation, and its value in the format's own notation: a `bool` is a Fortran logical here. The
first entry of a kind, which has nothing to copy, is written the way the format's parser
declares, indented like the entry above it.

What can be added:

- a scalar: an `int`, a `float`, a `str`, a `bool`, and whatever else the format reads;
- a list of two or more values;
- `None`, in a format that can write a key without a value;
- a `dict`, which adds a block and fills it, such as a namelist group;
- a list of dicts, which adds records, in an FMS field table.

A dict that the format cannot write as a block is written one level deeper, where it can be: in a
namelist group, `{"a": 1}` under the key `t` is the derived type `t%a = 1`.

A new key is written as it is spelled, and has to be a name the format allows. One it does not
allow raises `ValueError`:

```python
try:
    setup["history file"] = "iceh"
except ValueError as error:
    print(error)
```

```text
Not a valid configuration key: 'history file'
```

A value the format has no way of writing raises `UnsupportedEntryError`, a subclass of
`TypeError`. Its message says which value types the format writes:

```python
from access.config import MOM6InputParser, UnsupportedEntryError

mom_input = MOM6InputParser().parse("DT = 1800.0\n")
try:
    mom_input["DT_THERM"] = None
except UnsupportedEntryError as error:
    print(error)
```

```text
This format cannot store None as 'DT_THERM' here: there is no way to write a key_null in 'start' with this value (value types this format writes: ['bool', 'float', 'integer', 'string'])
```

A block that exists cannot be replaced by assigning a dict to it: that raises `SyntaxError`. Edit
its entries instead, or delete it and assign it again.

## Deleting keys

`del`, `pop`, `popitem` and `clear` remove the entries writing a key, together with the comment on
the same line:

```python
del setup["histfreq_n"]
del setup["restart_dir"]
print(str(config["setup_nml"]))
```

```text
  conserv_check = .true.
  diagfreq = 480
  dumpfreq = "m"
  histfreq = "h", "d", "m", "y"
  npt = 35040
  dump_last = .true.
```

A comment on lines of its own is left where it is: it may describe the entries around it rather
than the one deleted.

## A failed edit changes nothing

Every value is checked before anything is written. An edit that raises leaves the file, and the
list being edited, as they were:

```python
try:
    histfreq[0:2] = ["a", 2]
except TypeError as error:
    print(error)
print(histfreq)
```

```text
Trying to change value type
['h', 'd', 'm', 'y']
```

The first of the two elements could have been changed, but the second could not, so neither was.

## What cannot be done

| Edit | Raises |
|---|---|
| Giving a value another type, or a list in place of a scalar, or a scalar in place of a list | `TypeError` |
| Leaving a list with fewer than two elements | `ValueError` |
| Changing the length of a namelist list written with a repeat count, qualifiers, skipped positions or several assignments | `ValueError` |
| Giving one of the elements a repeat count stands for a value of its own | `ValueError` |
| Writing a position of a namelist array that no entry writes | `ValueError` |
| Adding a key the format does not allow as a name | `ValueError` |
| Adding a value the format cannot write, such as `None` in a MOM6 parameter file, an infinite or NaN `float`, or a string holding both quote characters in a format with no way of escaping them | `UnsupportedEntryError` |
| Assigning a dict to a block that exists | `SyntaxError` |
| Changing a field of a field table record, rather than replacing the record | `TypeError` |

Each format's page adds what is particular to it.
