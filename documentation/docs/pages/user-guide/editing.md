# Editing

You edit a configuration as you would edit a dict: you assign to a key, delete a key, or change a
list in place. Each change is made to the text straight away, so `str` always gives the text of the
configuration as it is now. In the examples we edit the first group of the `ice_in` file that CICE
reads in an ACCESS-OM3 configuration:

```fortran title="ice_in"
&setup_nml
  conserv_check = .false.
  diagfreq = 960
  dumpfreq = "y"
  histfreq = "d", "m", "x", "x", "x"
  npt = 35040
/
```

Save it as `ice_in` in the current directory, or adapt the path in the examples to where your file
is. Everything described here is true for every parser that is built on a grammar. That is all of
them except the YAML and JSON parsers, which are described in [YAML and JSON](formats/yaml-json.md).

## Changing a value

When you assign to a key that the configuration already has, its value is rewritten, and nothing
else. Here we change three values:

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

This prints the whole file, with the three new values:

```text
&setup_nml
  conserv_check = .true.
  diagfreq = 480
  dumpfreq = "m"
  histfreq = "d", "m", "x", "x", "x"
  npt = 35040
/
```

A new value is written in the same notation as the one it replaces. A string keeps the same quotes,
unless the new string contains that quote character, in which case the other one is used. A logical
is spelled in the same way: `T` becomes `F`, and `.false.` becomes `.true.`. A number keeps its
exponent letter when it has one, so a `d` stays a `d`.

The new value has to be of the same type as the old one. The type is checked exactly: an `int` is
not a `float`, a `bool` is not an `int`, and a NumPy number is not a Python one. A value of another
type raises `TypeError`. For example, we try to give an integer a `float`:

```python
try:
    setup["npt"] = 35040.0
except TypeError as error:
    print(error)
```

This prints:

```text
Trying to change value type
```

If you do want to give a key a value of another type, delete the key and assign it again. It is then
written as a new entry, at the end of its block.

## Lists

A value written as a list is a `ConfigList`. It is a `list`, so you can change it in all the usual
ways: by assigning an element or a slice, and with `del`, `append`, `extend`, `insert`, `pop`,
`remove`, `clear`, `sort`, `reverse`, `+=` and `*=`. Here we change one element, delete the last two
and insert a new one at the start:

```python
histfreq = setup["histfreq"]
histfreq[2] = "y"
del histfreq[3:]
histfreq.insert(0, "h")
print(histfreq)
print(str(config).splitlines()[4])
```

This prints the list, and then its line in the file:

```text
['h', 'd', 'm', 'y']
  histfreq = "h", "d", "m", "y"
```

The elements that stay keep their text, comments included. A new element is written like its
neighbours, on their line or on a line of its own. If the elements of the list all have the same
type, the new element has to have that type too. To see where a new element goes, we use a list that
is written with one element on each line:

```fortran title="history.nml"
&setup_nml
  histfreq = 'd',  ! daily
             'm'   ! monthly
/
```

Save it as `history.nml`, and append an element to the list:

```python
history = FortranNMLParser().parse(Path("history.nml").read_text())
history["setup_nml"]["histfreq"].append("y")
print(str(history))
```

This prints:

```text
&setup_nml
  histfreq = 'd',  ! daily
             'm',
             'y'   ! monthly
/
```

A comment after an element belongs to that element, and is removed with it. The comment after the
last element is different. It ends the whole entry and stays at its end, which is why `! monthly`
now follows the new element.

A copy of a `ConfigList`, such as `histfreq[:]` or `list(histfreq)`, is a plain `list`, and changing
it changes nothing in the file. The same is true of the copies that `copy.copy`, `copy.deepcopy` and
`pickle` make, of a list or of a whole configuration. `copy.copy` gives a plain `dict` or `list`
holding the same values. `copy.deepcopy` and `pickle` give plain dicts and lists all the way down.

### What a list cannot become

- A list keeps at least two elements, because a single value is a scalar. Removing elements so that
  fewer than two are left raises `ValueError`.
- A list cannot become a scalar, and a scalar cannot become a list. This is a change of type, and it
  raises `TypeError`. Delete the key and assign it again instead.
- A list whose elements all have the same type keeps that type. Adding an element of another type
  raises `TypeError`. A list of mixed types, such as a Fortran derived type written as a list of its
  components, still accepts an element of any type.
- In a Fortran namelist, some lists can have their elements changed, but not their number: changing
  the length raises `ValueError`. These are the lists written with a repeat count (`3*1.0`), with
  array qualifiers (`x(2) = ...`), with skipped positions (`1, , 3`) or over several assignments.
  [Fortran namelists](formats/fortran-namelists.md#lists) explains why.

## Adding keys

When you assign to a key that the configuration does not have, an entry is added for it. Here we add
three entries to the group, and a whole new group:

```python
setup["dump_last"] = True
setup["histfreq_n"] = [1, 1, 1, 1]
setup["restart_dir"] = None
config["grid_nml"] = {"grid_type": "tripole", "ncat": 5}
print(str(config))
```

This prints:

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
comment. It is written as a copy of the nearest entry of the same kind, with the spacing and
punctuation of that entry. Its value is written in the notation of the format, so a `bool` is a
Fortran logical here. The first entry of a kind has nothing to copy. It is written in the way that
the parser of the format declares, with the indentation of the entry above it.

You can add:

- a scalar: an `int`, a `float`, a `str`, a `bool`, or any other type that the format reads;
- a list of two or more values;
- `None`, in a format that can write a key without a value;
- a `dict`, which adds a block, such as a namelist group, and fills it;
- a list of dicts, which adds records, in an FMS field table.

A dict that the format cannot write as a block is written one level deeper, where it can be. In a
namelist group, for example, `{"a": 1}` under the key `t` is the derived type `t%a = 1`.

A new key is written as you spelled it, and it has to be a name that the format allows. Otherwise it
raises `ValueError`. For example, the name of a namelist variable cannot contain a space:

```python
try:
    setup["history file"] = "iceh"
except ValueError as error:
    print(error)
```

This prints:

```text
Not a valid configuration key: 'history file'
```

A value that the format has no way of writing raises `UnsupportedEntryError`, which is a subclass of
`TypeError`. Its message says which types of value the format writes. For example, a MOM6 parameter
file cannot hold a key without a value:

```python
from access.config import MOM6InputParser, UnsupportedEntryError

mom_input = MOM6InputParser().parse("DT = 1800.0\n")
try:
    mom_input["DT_THERM"] = None
except UnsupportedEntryError as error:
    print(error)
```

This prints:

```text
This format cannot store None as 'DT_THERM' here: there is no way to write a key_null in 'start' with this value (value types this format writes: ['bool', 'float', 'integer', 'string'])
```

You cannot replace an existing block by assigning a dict to it: this raises `SyntaxError`. Edit its
entries instead, or delete the block and assign it again.

## Deleting keys

`del`, `pop`, `popitem` and `clear` remove the entries that write a key, together with the comment
on the same line. Here we delete two of the entries we added:

```python
del setup["histfreq_n"]
del setup["restart_dir"]
print(str(config["setup_nml"]))
```

This prints the body of the group:

```text
  conserv_check = .true.
  diagfreq = 480
  dumpfreq = "m"
  histfreq = "h", "d", "m", "y"
  npt = 35040
  dump_last = .true.
```

A comment on lines of its own is left where it is, because it may describe the entries around it and
not the one you deleted.

## A failed edit changes nothing

Every value is checked before anything is written. An edit that raises therefore leaves the file as
it was, and also the list that was being edited. Here we try to replace two elements, and the second
one has the wrong type:

```python
try:
    histfreq[0:2] = ["a", 2]
except TypeError as error:
    print(error)
print(histfreq)
```

This prints the error, and then the list, which is unchanged:

```text
Trying to change value type
['h', 'd', 'm', 'y']
```

The first of the two elements could have been changed, but the second could not, so neither was.

## What cannot be done

| Edit | Raises |
|---|---|
| Giving a value another type, a list in place of a scalar, or a scalar in place of a list; adding an element of another type to a list whose elements all have one type | `TypeError` |
| Leaving a list with fewer than two elements | `ValueError` |
| Changing the length of a namelist list written with a repeat count, qualifiers, skipped positions or several assignments | `ValueError` |
| Giving one of the elements a repeat count stands for a value of its own | `ValueError` |
| Writing a position of a namelist array that no entry writes | `ValueError` |
| Adding a key the format does not allow as a name | `ValueError` |
| Adding a value the format cannot write, such as `None` in a MOM6 parameter file, an infinite or NaN `float`, or a string holding both quote characters in a format with no way of escaping them | `UnsupportedEntryError` |
| Assigning a dict to a block that exists | `SyntaxError` |
| Changing a field of a field table record, rather than replacing the record | `TypeError` |

The page of each format adds what is particular to it.
