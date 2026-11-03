# NUOPC run sequences

`NUOPCRunSeqParser` reads the `nuopc.runseq` file of a model driven by CMEPS, such as ACCESS-OM3,
which says in which order the components and the mediator run, and how often. The file is an
ESMF Resource File holding one table, `runSeq::`, whose syntax is described in the
[NUOPC reference manual](https://earthsystemmodeling.org/docs/release/latest/NUOPC_refdoc/),
under `NUOPC_DriverIngestRunSequence`.

```python
from access.config import NUOPCRunSeqParser

parser = NUOPCRunSeqParser()
```

## How a run sequence reads

A table reads as a list of its lines, each a string, without its indentation, its trailing
whitespace or its comment. Here is the start of the run sequence of an ACCESS-OM3 configuration,
as CMEPS writes it, with spaces at the end of some lines:

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

```python
from pathlib import Path

config = parser.parse(Path("nuopc.runseq").read_text())
print(config["runSeq"])
```

```text
['@3600', 'MED med_phases_aofluxes_run', 'MED med_phases_prep_ocn_accum', 'MED -> ICE :remapMethod=redist', 'ICE', 'OCN', 'OCN -> MED :remapMethod=redist', 'MED med_phases_post_ocn', '@']
```

`@3600` opens a time loop of 3600 seconds and `@` closes it, and loops may nest. A line naming a
component runs it, a component and a phase runs that phase, and `SRC -> DST`, with optional
`:option=value` settings, runs a connector. A table with a single line reads as a string, and an
empty one as `None`.

## Editing

The list is edited as any list is. Changing a line rewrites that line alone, and a new line is
indented like its neighbours:

```python
run_sequence = config["runSeq"]
run_sequence[0] = "@1800"
run_sequence.insert(5, "WAV")
run_sequence.insert(6, "WAV -> MED :remapMethod=redist")
print(str(config))
```

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

The lines that stay are written as they were, their comments and trailing spaces included. A
whole run sequence can also be assigned as a list of strings, and the lines that are in both are
kept as they are.

A new line has to be a string with no blank at either end, no `#` and not starting with `::`.
Anything else raises `UnsupportedEntryError`, since it would not read back as the same line.

## What is not read

- Text outside a table, which ESMF ignores.
- A table that is not closed, and a label written as `runSeq:` rather than `runSeq::`.

A label written twice reads as the last of the tables, where ESMF finds the first.
