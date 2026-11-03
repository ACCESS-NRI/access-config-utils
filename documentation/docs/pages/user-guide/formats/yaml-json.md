# YAML and JSON

YAML and JSON already have libraries that read and write them, so their parsers wrap those
libraries rather than a grammar of their own: [ruamel.yaml](https://yaml.dev/doc/ruamel.yaml/)
for YAML, and the standard library's `json` for JSON. They have the same `parse` and `str` as the
other parsers, but the configuration they return is the library's own data, and editing it
follows the library's rules rather than those [Editing](../editing.md) describes:

- nested mappings and lists are the library's types, not `Config` and `ConfigList`;
- nothing checks the type of a value: any value the library can write can be assigned;
- lists can have any length, and keys any name.

## YAML

`YAMLParser` reads YAML files, such as the `config.yaml` of a Payu experiment. The configuration
is a `YAMLConfig`, which is ruamel.yaml's round-trip mapping, a `CommentedMap`, with nested
mappings and lists as `CommentedMap` and `CommentedSeq`:

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

```python
from pathlib import Path

from access.config import YAMLParser

config = YAMLParser().parse(Path("config.yaml").read_text())
config["walltime"] = "05:00:00"
config["submodels"][1]["ncpus"] = 240
config["collate"]["mpi"] = False
print(str(config))
```

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

ruamel.yaml keeps the comments, the order of the keys and the quotes of every string. It writes
the whole file again, though, rather than editing it, so what comes back is its own spelling of
the data. The parser reads the indentation the file uses, and writes with that, so a file indented
the same way throughout, like the one above, comes back unchanged apart from the edits. One that
mixes several indentations comes back with one of them, and ruamel.yaml's spelling of some values
replaces the file's: `True` is written `true`, `null` as nothing, and `[ 1, 2 ]` as `[1, 2]`.

The top level of the file has to be a mapping. Anything else raises `TypeError`.

## JSON

`JSONParser` reads JSON files, such as the `forcing.json` of an ACCESS-OM2 configuration. The
configuration is a `JSONConfig`, a `dict` holding the data as `json` reads it: nested objects
are `dict`s and arrays are `list`s, edited directly.

JSON has no comments, and none of its whitespace means anything, so a file is its data and the
way the data was laid out. Until the data changes, `str` gives back the text as it was read. After
an edit, the text is written again from the data, laid out the way the file was: a file written
by a program, with one set of `json.dumps` options throughout, is written with those options,
and an edit changes exactly the lines `json.dumps` writes differently. A file laid out by hand
is written in the closest style that can be read off it, keeping the layout of each of its
lists, on one line or one element per line:

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

```python
from access.config import JSONParser

config = JSONParser().parse(Path("run.json").read_text())
config["dt"] = 900
config["levels"].append(4)
config["files"].append("c.nc")
print(str(config))
```

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

Since the configuration is plain data, a value JSON cannot hold is only found when the text is
written: a tuple, a set or a key that is not a string makes `str` raise `TypeError`, naming where
it is:

```python
config["shape"] = (360, 300)
try:
    str(config)
except TypeError as error:
    print(error)
```

```text
tuple at ['shape'] cannot be written as JSON
```

Invalid JSON raises the standard library's `json.JSONDecodeError`, including a trailing comma
after the last element of an object or an array, which some readers accept. A file whose top
level is not an object raises `ValueError`.
