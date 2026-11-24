# YAML and JSON

YAML and JSON already have libraries that read and write them. Their parsers therefore wrap those
libraries, and do not have a grammar of their own: [ruamel.yaml](https://yaml.dev/doc/ruamel.yaml/)
for YAML, and the `json` module of the standard library for JSON. They have the same `parse` and
`str` as the other parsers. The configuration they return, however, is the library's own data. When
you edit it, the rules of the library apply, and not those that [Editing](../editing.md) describes:

- nested mappings and lists are the library's types, not `Config` and `ConfigList`;
- nothing checks the type of a value: any value the library can write can be assigned;
- lists can have any length, and keys any name.

## YAML

`YAMLParser` reads YAML files, such as the `config.yaml` of a Payu experiment. The configuration is
a `YAMLConfig`. This is the round-trip mapping of ruamel.yaml, a `CommentedMap`, and its nested
mappings and lists are `CommentedMap` and `CommentedSeq` objects. As an example, we use this file:

```yaml title="config.yaml"
# PBS configuration
queue: normal
walltime: 03:00:00
jobname: 1deg_jra55_ryf

model: access-om2
submodels:
  - name: atmosphere
    model: yatm
    ncpus: 1

  - name: ocean
    model: mom
    ncpus: 216

collate:
  enable: true
  restart: true
```

Save it as `config.yaml` in the current directory, or adapt the path to where your file is. We parse
it, change three values and print the result:

```python
from pathlib import Path

from access.config import YAMLParser

config = YAMLParser().parse(Path("config.yaml").read_text())
config["walltime"] = "05:00:00"
config["submodels"][1]["ncpus"] = 240
config["collate"]["mpi"] = False
print(str(config))
```

This prints:

```text
# PBS configuration
queue: normal
walltime: 05:00:00
jobname: 1deg_jra55_ryf

model: access-om2
submodels:
  - name: atmosphere
    model: yatm
    ncpus: 1

  - name: ocean
    model: mom
    ncpus: 240

collate:
  enable: true
  restart: true
  mpi: false
```

ruamel.yaml keeps the comments, the order of the keys and the quotes of every string. However, it
does not keep the layout of the file. It writes the whole file with one indentation, and it spells
some values in its own way: `True` as `true`, `null` as nothing, and `[ 1, 2 ]` as `[1, 2]`. `str`
therefore does not return what ruamel.yaml writes, but edits the text that was parsed. A
configuration that has not changed is written back exactly as it was read. After an edit, only the
lines that changed are replaced in the text. The other lines keep their indentation, spacing and
spelling, even in a file that mixes several indentations. As a second example, we use such a file:

```yaml title="mixed.yaml"
sync:
    enable: False   # set to true to sync
    path: null
manifest:
  reproduce:
    exe: True
base_date: [ 1900, 1, 1 ]
```

Save it as `mixed.yaml` in the current directory too. We parse it, add a key to the innermost
mapping and print the result:

```python
config = YAMLParser().parse(Path("mixed.yaml").read_text())
config["manifest"]["reproduce"]["input"] = True
print(str(config))
```

This prints:

```text
sync:
    enable: False   # set to true to sync
    path: null
manifest:
  reproduce:
    exe: True
    input: true
base_date: [ 1900, 1, 1 ]
```

Only the new line differs from the file. It is indented like the lines around it, but it is written
in ruamel.yaml's spelling: `true`, next to the `True` of the line above. Lines that are added or
changed use the line ending of the file, `\n` or `\r\n`. A file that does not end with a newline
still does not end with one after an edit.

Before `str` returns the edited text, it reads the text again and compares it with the
configuration. Sometimes the text does not read as the same data, or a changed line cannot be placed
in it. In that case, ruamel.yaml writes the whole file instead, with the indentation that the parser
detects from the file. This happens, for example, when a list that is written in brackets over
several lines changes. It also happens when an entry changes in a list that is nested directly in a
list, written `- - a`. A file that is indented in the same way throughout, like `config.yaml` above,
then keeps its indentation. A file that mixes several indentations comes back with one of them. In
both cases, every value is spelled in ruamel.yaml's way.

The top level of the file has to be a mapping. Anything else raises `TypeError`.

## JSON

`JSONParser` reads JSON files, such as the `forcing.json` of an ACCESS-OM2 configuration. The
configuration is a `JSONConfig`, which is a `dict` that holds the data as `json` reads it. Nested
objects are `dict`s and arrays are `list`s, and you edit them directly.

JSON has no comments, and its whitespace has no meaning. A file is therefore its data, plus the way
the data is formatted. As long as the data does not change, `str` gives back the text as it was
read. After an edit, the text is written again from the data, with the formatting of the file. A
file written by a program, with one set of `json.dumps` options throughout, is written with those
options, and an edit changes exactly the lines that `json.dumps` writes differently. A file
formatted by hand is written in the closest style that can be detected from it. Each of its lists
keeps its own formatting, on one line or with one element on each line. As an example, we use this
file:

```json title="run.json"
{
    "name": "run",
    "levels": [1, 2, 3],
    "files": [
        "a.nc",
        "b.nc"
    ]
}
```

Save it as `run.json`. We parse it, add a key, and append an element to each list:

```python
from access.config import JSONParser

config = JSONParser().parse(Path("run.json").read_text())
config["dt"] = 900
config["levels"].append(4)
config["files"].append("c.nc")
print(str(config))
```

This prints:

```text
{
    "name": "run",
    "levels": [1, 2, 3, 4],
    "files": [
        "a.nc",
        "b.nc",
        "c.nc"
    ],
    "dt": 900
}
```

The configuration is plain data, so a value that JSON cannot hold is only found when the text is
written. A tuple, a set or a key that is not a string makes `str` raise `TypeError`, with a message
that says where the value is. For example, with a tuple:

```python
config["shape"] = (360, 300)
try:
    str(config)
except TypeError as error:
    print(error)
```

This prints:

```text
tuple at ['shape'] cannot be written as JSON
```

Invalid JSON raises `json.JSONDecodeError`, from the standard library. This includes a trailing
comma after the last element of an object or an array, which some readers accept. A file whose top
level is not an object raises `ValueError`.
