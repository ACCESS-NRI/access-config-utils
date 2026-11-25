# Rose configurations

`RoseConfParser` reads the configuration files of a suite configured with Rose, such as its
`rose-suite.conf` and the `rose-app.conf` of each of its applications. The suites of ACCESS-AM3,
ACCESS-rAM3 and ACCESS-CM3 are configured with Rose, and run with Cylc. The format is described in
[Rose's documentation](https://metomi.github.io/rose/2019.01.8/html/api/configuration/rose-configuration-format.html),
and the parser reads a file as Rose 2019.01 reads it. You create the parser like this:

```python
from access.config import RoseConfParser

parser = RoseConfParser()
```

## How a Rose configuration reads

A Rose configuration is a list of settings, one on each line, each a key and a value parted by an
equals sign. The configuration is a dict of them. A header, `[NAME]`, starts a section, which is a
nested dict. Nothing closes a section, so it runs until the next header. The settings above the
first header belong to no section. As an example, we use this file:

```text title="rose-suite.conf"
# The run directories go to scratch.
root-dir=*=/scratch/$PROJECT/$USER

[jinja2:suite.rc]
MAIN_ATM_PROCX=24
MAIN_ATM_PROCY=24
!!MAIN_IOS_NPROC=0
MAIN_CLOCK='PT1H10M'
SITES='nci',
     ='other'
```

Save it as `rose-suite.conf` in the current directory, or adapt the path to where your file is. We
parse it and print the setting of no section, and then the section:

```python
from pathlib import Path

config = parser.parse(Path("rose-suite.conf").read_text())
print(config["root-dir"])
print(dict(config["jinja2:suite.rc"]))
```

This prints:

```text
*=/scratch/$PROJECT/$USER
{'MAIN_ATM_PROCX': '24', 'MAIN_ATM_PROCY': '24', '!!MAIN_IOS_NPROC': '0', 'MAIN_CLOCK': "'PT1H10M'", 'SITES': "'nci',\n'other'"}
```

Keys and section names are case-sensitive, as they are for Rose. A key holds no whitespace and no
equals sign, so it does not have to be a name: `root-dir{share}` is a key. A section name holds
anything but a bracket, such as `namelist:nlsizes` or `file:app/um/input.nml`.

A comment is a line that starts with `#`, after any blanks, and it is kept wherever it is. A `#`
anywhere else is part of a value. A section written twice is read as one section, which holds the
settings of both. A key written twice is read as the last value it is given, as Rose reads it.

Reading a large file takes a while: a file of 6,000 lines takes a few seconds to parse, and about
as long again to write back out.

## Values

Every value is read as a string: the text after the equals sign, without the blanks around it.
This is because Rose gives a setting no type, and the same text means different things in different
sections. A `[jinja2:suite.rc]` section holds Jinja2 values, a `[namelist:...]` section of a
`rose-app.conf` holds Fortran values, and an `[env]` section holds shell text.

| Written | Read as |
|---|---|
| `MAIN_ATM_PROCX=24` | `"24"` |
| `MAIN_CLOCK='PT1H10M'` | `"'PT1H10M'"`, as the quotes are part of a Jinja2 string |
| `ltimer=.true.` | `".true."` |
| `root-dir=*=/scratch` | `"*=/scratch"`, as only the first equals sign ends the key |
| `KEY=` | `None` |
| A value on several lines | One string, with a line break between the lines |

A value goes on over the indented lines below it, as `SITES` does in the example. Each of these
lines is trimmed, and loses one equals sign that it starts with. This means that a line can keep
the blanks it starts with by putting them after an equals sign. A blank line or a comment between
these lines adds nothing to the value.

A value is written exactly as the string you give. The string must be one line, with no blanks at
either end, because Rose would read anything else back differently. A number, a boolean or a list
raises `TypeError` when it replaces a value, and `UnsupportedEntryError` when it is given to a new
key. An empty string, a string with blanks at either end and a string of several lines raise
`UnsupportedEntryError`. You can still write a value of several lines over itself, and you can
replace it with one line.

`None` and a string are different kinds of entry. A key written as `KEY=` cannot be given a string,
and a key with a value cannot be given `None`: both raise `TypeError`. To change one into the
other, delete the key and assign it again.

## Ignored settings and sections

Rose ignores a setting or a section whose key or name starts with `!` or `!!`. A user marks a
setting ignored with `!`, and a program with `!!`. The marks are part of the key. In the example,
the setting `!!MAIN_IOS_NPROC` is not the setting `MAIN_IOS_NPROC`:

```python
variables = config["jinja2:suite.rc"]
print(variables["!!MAIN_IOS_NPROC"], "MAIN_IOS_NPROC" in variables)
```

This prints:

```text
0 False
```

This means that a setting Rose ignores is never mistaken for one it uses. You edit an ignored
setting under its marked key, and you add one by writing its marks in the key. A new setting does
not take on the marks of the setting it is copied from. To stop ignoring a setting, delete it and
assign it again under the key without its marks. It is then written at the end of its section.

Note that a file holding both `KEY=` and `!KEY=` is read as two keys. Rose reads them as one key,
in the state that is written last.

## Editing

A new setting of a section is written after the last setting of that section. A new setting of no
section is written after the last setting above the first section, because a setting below a
section would be part of it. A dict assigned to a new key is written as a new section, after the
last one. Here we change two settings, and add a setting of no section and a section:

```python
variables["MAIN_ATM_PROCX"] = "16"
variables["MAIN_CLOCK"] = "'PT6H'"
config["root-dir{share}"] = "*=/g/data/$PROJECT/$USER"
config["env"] = {"PROJECT": "tm70"}
print(str(config))
```

This prints:

```text
# The run directories go to scratch.
root-dir=*=/scratch/$PROJECT/$USER
root-dir{share}=*=/g/data/$PROJECT/$USER

[jinja2:suite.rc]
MAIN_ATM_PROCX=16
MAIN_ATM_PROCY=24
!!MAIN_IOS_NPROC=0
MAIN_CLOCK='PT6H'
SITES='nci',
     ='other'

[env]
PROJECT=tm70
```

A new section is written as a copy of the last section, so it is parted from the section above it
in the same way, here by a blank line. The first section of a file has nothing to copy, and is
written after a blank line. The blank lines and the comments right above a header belong to its
section. A new setting is therefore never written between a comment and the header below it, and
deleting a section deletes the comments above it. Deleting the last setting of a section leaves the
section, with nothing in it.

## What is not read

- Three headers that Rose reads: the empty header `[]`, which takes Rose back to the settings of no
  section, a header with blanks before it, and a header with blanks between a bracket and its name.
- The optional configurations of a suite or an application, under `opt/`, are files of their own.
  Each is parsed on its own, and nothing merges it into the configuration it changes.

What can be read is not always possible to add:

- A section whose name holds a blank or an equals sign can be read and edited, but not added. Its
  name raises `ValueError`, as a key's name would.
- A section written twice takes no new setting, because it has no single place to put it. Adding
  one raises `UnsupportedEntryError`.
- A section cannot hold a section, so a dict assigned to a key of a section raises
  `UnsupportedEntryError`.
- The comments right above the first header belong to its section, even when they describe the
  whole file. A new setting of no section is written above them.
