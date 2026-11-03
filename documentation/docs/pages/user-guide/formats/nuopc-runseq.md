# NUOPC run sequences

`NUOPCRunSeqParser` reads the `nuopc.runseq` file of a model driven by CMEPS, such as ACCESS-OM3.
This file says in which order the components and the mediator run, and how often. It is an ESMF
Resource File with a single table, `runSeq::`. The syntax of the table is described in the [NUOPC
reference manual](https://earthsystemmodeling.org/docs/release/latest/NUOPC_refdoc/), under
`NUOPC_DriverIngestRunSequence`. You create the parser like this:

```python
from access.config import NUOPCRunSeqParser

parser = NUOPCRunSeqParser()
```

## How a run sequence reads

A table is read as a list of its lines. Each line is a string, without its indentation, its trailing
whitespace or its comment. As an example, we use the start of the run sequence of an ACCESS-OM3
configuration. It is shown as CMEPS writes it, with spaces at the end of some lines:

```text title="nuopc.runseq"
runSeq:: 
@3600 
  MED med_phases_aofluxes_run
  MED med_phases_prep_ocn_accum
  MED -> ICE :remapMethod=redist
  ICE
  OCN   # the ocean
  OCN -> MED :remapMethod=redist
  MED med_phases_post_ocn
@ 
::
```

Save it as `nuopc.runseq` in the current directory, or adapt the path to where your file is. We
parse it and print the table:

```python
from pathlib import Path

config = parser.parse(Path("nuopc.runseq").read_text())
print(config["runSeq"])
```

This prints:

```text
['@3600', 'MED med_phases_aofluxes_run', 'MED med_phases_prep_ocn_accum', 'MED -> ICE :remapMethod=redist', 'ICE', 'OCN', 'OCN -> MED :remapMethod=redist', 'MED med_phases_post_ocn', '@']
```

`@3600` opens a time loop of 3600 seconds and `@` closes it. Loops can be nested. A line with the
name of a component runs that component, and a line with a component and a phase runs that phase. A
line `SRC -> DST`, with optional `:option=value` settings, runs a connector. A table with a single
line is read as a string, and an empty table as `None`.

## Editing

You edit the list like any other list. Changing a line rewrites that line alone, and a new line is
indented like its neighbours. Here we shorten the time loop, and add a wave component with its
connector:

```python
run_sequence = config["runSeq"]
run_sequence[0] = "@1800"
run_sequence.insert(5, "WAV")
run_sequence.insert(6, "WAV -> MED :remapMethod=redist")
print(str(config))
```

This prints:

```text
runSeq:: 
@1800 
  MED med_phases_aofluxes_run
  MED med_phases_prep_ocn_accum
  MED -> ICE :remapMethod=redist
  ICE
  WAV
  WAV -> MED :remapMethod=redist
  OCN   # the ocean
  OCN -> MED :remapMethod=redist
  MED med_phases_post_ocn
@ 
::
```

The lines that stay are written as they were, including their comments and trailing spaces. You can
also assign a whole run sequence as a list of strings. The lines that are in both the old and the
new list are kept as they are.

A new line has to be a string. Anything else, such as a number, raises `TypeError`. The string
cannot have a blank at either end, contain a `#` or start with `::`. Such a string raises
`UnsupportedEntryError`, because it would not be read back as the same line.

## What is not read

- Text outside a table, which ESMF ignores.
- A table that is not closed, and a label written as `runSeq:` rather than `runSeq::`.

A label written twice is read as the last of the tables, whereas ESMF finds the first.
