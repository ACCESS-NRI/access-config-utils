# AGENTS.md

Guidance for AI coding agents working in `access-config-utils`.

## What this project is

A Python library from ACCESS-NRI with two largely independent halves:

1. **Round-trip config parsing/editing** — Lark-based parsers for the configuration file
   formats used by ACCESS climate models. The defining requirement is *round-trip fidelity*:
   `str(parser.parse(text)) == text`, including comments, whitespace and formatting, and an
   edit must produce a minimal diff.
2. **Parallel layout enumeration** — a library for enumerating and filtering valid MPI/OpenMP
   layouts (rank counts, thread counts, cartesian process grids, domain decompositions) for a
   tree of model components.

Python >= 3.11 (CI covers 3.11–3.13). Runtime deps: `lark`, `ruamel.yaml`.

## Working here

- **Think before coding.** State your assumptions; where a request is ambiguous, ask rather
  than pick silently. Point out a simpler alternative or a tradeoff, and push back when it is
  warranted.
- **Simplicity first.** Write the least code that solves the problem: no speculative
  features, single-use abstractions, unrequested configurability, or handling of errors that
  cannot happen. A hook nothing overrides is removed, not kept for later, and an unreachable
  branch could not be covered anyway.
- **Exact over approximate.** Round trips are byte for byte, core counts are integers, and the
  layout search is exhaustive; don't trade any of that for a numerical shortcut.
- **Surgical changes.** Every changed line traces to the request. Match the surrounding style;
  don't refactor, reformat or "improve" adjacent code. Remove what your change orphans; mention
  pre-existing dead code rather than deleting it.
- **The documentation follows the code.** The docstrings and the pages of the
  [user guide](documentation/docs/pages/user-guide/) describe the behaviour in detail, so a
  change of behaviour updates the docstrings and the pages that describe it, in the same change:
  grep `src/` and `documentation/docs/pages` for the names you touched. The strict build catches
  a broken reference, not stale prose.
- **Be honest.** Report what you ran and what it showed: a failing test, a check you skipped,
  a test that passes without binding.

## Repository layout

```
src/access/config/
  ── grammar: what a format declares, and what that compiles to ──
  grammar_contract.py   The rule names every grammar must use, and the contract in full
  grammar_values.py     ValueTypeHandler + VALUE_TYPE_HANDLER_REGISTRY (its order is the order
                        a new value's rule is chosen in) + select_value_rule
  grammar_compiled.py   CompiledGrammar, GrammarInfo (the views over a compiled grammar),
                        ParseContext, compile_grammar (cached on the grammar text)
  config.lark           Shared terminals/rules imported by every format grammar

  ── parse tree: reading one, changing one ──
  tree_navigation.py    AddParent, entry_nodes/entries_of, entry_chain, contains_entry,
                        is_comment_line, entry_insertion_index, line_index, key_token,
                        locate_entry/node_at
  tree_edits.py         update_node_value, write_values, parse_entry_nodes,
                        splice_entry_nodes, replace_entry_contents, blank_comments,
                        remove_entry_node, merge_adjacent_repetitions
  tree_reader.py        EntryRef + EntryReader: tree -> one reference per key

  ── entry synthesis: the text of a key the file does not have yet ──
  entry_fragments.py    Fragment/EntryTemplate, compile_template (the text a parser
                        declares), expand_value_slots
  entry_style.py        EntryStyle (an indent and the donors) + probing: the indentation
                        and the whole entries to copy from the file (donor_template)
  entry_render.py       render, iter_entry_snippets: the donor, then the declared template

  ── the data, and the parser ──
  list_layout.py        The text of a list whose length changes: ListText, align_values,
                        layout_candidates (text only, no tree)
  config_insert.py      insert_entry: synthesise, validate, splice
  config_resize.py      resize_entry: a list of another length written into its entry
  config_store.py       ConfigStore: the tree behind one container, kept in step
  config_dict.py        Config (dict subclass), ConfigList and ConfigRecord (read only)
  parser.py             ConfigParser (ABC)

  ── formats ──
  fortran_nml.py        FortranNMLParser   (case-insensitive keys, &name ... / blocks)
  mom6_input.py         MOM6InputParser    (case-sensitive, % blocks)
  nuopc_config.py       NUOPCParser        (case-sensitive, :: ESMF resource-file blocks)
  nuopc_runseq.py       NUOPCRunSeqParser  (case-sensitive, one table per label, a line per
                        element)
  field_table.py        FieldTableParser   (keys lower-cased, headers as chains of three keys,
                        methods of three strings as records)
  yaml_config.py        YAMLParser         (thin wrapper over ruamel.yaml)
  json_config.py        JSONParser         (thin wrapper over json, its layout detected)

  parallel_domain.py      Domain, DomainDecompositionSpec
  parallel_mpi_grid.py    MPICartesianGrid + iter_grids
  parallel_component.py   ParallelComponent (+ iter_decompositions), ComponentLayout,
                          CoreSharing, LocalConstraint, GroupConstraint
  parallel_constraints.py 10 concrete constraints, in four families
  parallel_allocation_strategies.py
                          AllocationStrategy (+ RootAllocation and Fixed/Weighted/Free/
                          WholeRange-Allocation), and the sibling-group scheduling that is
                          inherently mode-aware
  parallel_layouts.py     iter_layouts (the entry point) and the recursive, memoised
                          search

scripts/run_f90nml_tests.py
                        f90nml's own test suite run against FortranNMLParser
                        (--source <f90nml sdist>); a diagnostic report, not a gate

documentation/          The MkDocs site, published on Read the Docs: pages in docs/pages/,
                        the API reference generated from the docstrings in docs/pages/api/
```

The [developer guide](documentation/docs/pages/developer-guide/index.md) explains this
structure, how a file is parsed and edited and how the layout search works, for people; this
file is the complete list of the rules, for an agent changing the code. The API reference is
generated from the docstrings, on pages in `documentation/docs/pages/api/` that are each a run
of `::: access.config.<module>` blocks, one page per concept, with the parser modules that are
not public API under `api/internals/`. A new module needs a block on the right page and a row in
the table of `api/index.md`; a new page needs an entry in the `nav` of
`documentation/mkdocs.yml`, or the strict build fails.

**The parser modules form four concept groups, and the imports run strictly one way through
them**: grammar, then parse tree, then entry synthesis, then the data and the parser class.
There is no cycle anywhere in the package, and no function-local import breaking one. The
`TYPE_CHECKING` imports (in `config_dict`, `config_insert`, `config_resize`, `config_store`,
`tree_edits` and `tree_reader`, the ones in brackets below) point down the same way the
runtime ones do.

Read in topological order, each importing only from lines above it:

```
grammar_contract, grammar_values, list_layout
grammar_compiled      <- contract, values
tree_navigation       <- contract, values
tree_edits            <- contract, values, navigation                    [compiled]
tree_reader           <- contract, values, navigation                    [compiled]
entry_fragments       <- contract
entry_style           <- contract, values, fragments, navigation, tree_edits
entry_render          <- contract, values, fragments, style
config_insert         <- contract, values, tree_*, entry_render          [compiled, style]
config_resize         <- contract, values, tree_*, list_layout, config_insert
                                                                         [compiled]
config_store          <- contract, values, tree_*, entry_style, list_layout, config_insert,
                         config_resize                                   [compiled]
config_dict           <- contract, values, tree_reader                   [config_store]
parser                <- compiled, contract, config_dict, config_store, tree_navigation
<format modules>      <- parser, contract
yaml_config, json_config  -- no package imports
```

Two boundaries in there are load-bearing and easy to erode:

- **`tree_navigation` reads, `tree_edits` changes.** The predicates deciding *where* an entry
  goes (`contains_entry`, `is_comment_line`, `entry_insertion_index`) belong down here; in the
  synthesis module they would make the tree layer import upwards.
- **The reader hands back references, never a `Config`.** `EntryReader` returns one `EntryRef`
  per key; turning a reference into a `ConfigList` or a nested `Config` is `config_dict`'s job.
  A reader that builds the dict brings back a `parser` <-> tree cycle.

The allocation modes and the enumerator live in separate modules, and the dependency runs one
way: `parallel_layouts` imports `parallel_allocation_strategies`, never the reverse. That
direction is the point of the split — **no code path in `parallel_layouts` names a concrete
allocation mode.** The one concrete class it names is `RootAllocation`, as the declared type of
the `allocations` parameter. If you find yourself writing `isinstance(strategy, FreeAllocation)`
or reaching for `.min_cores`/`.n_cores`/`.weight` in `parallel_layouts`, the logic belongs next
door. The mode-aware scheduling (`iter_core_splits`, `iter_shared_core_shares`,
`_group_by_mode`, `resolve_root_strategy`) sits with the modes for the same reason: it is the
only code that must know all of them at once.

`tests/` mirrors `src/` one-to-one, `test_<module>.py` per source module, and each module's
file **takes that module to 100% on its own**. That is stricter than the aggregate 100% and the
ordinary report cannot show it, so check the one you changed:

```shell
m=config_store
python -m pytest tests/test_$m.py -o addopts="" --cov=access.config.$m \
       --cov-report=term-missing --cov-fail-under=100
```

Coverage says a line ran, not that anything would notice it changing, and an over-permissive
mock passes whatever it is asked. To check a test binds, break the code it covers and confirm
*that* test fails, then **clear the bytecode**:
`find . -name __pycache__ -type d -not -path "./.git/*" | xargs rm -rf`. Python checks a `.pyc`
against the source's `(mtime, size)`, so a same-length edit reverted within the same second
(`idle_cores` to `idle_corez` and back) leaves the mutant loaded and tests failing against a
source `git diff` calls clean.

**Fake what comes from `access.config`, never anything from lark.** `Tree`, `Token`, `Lark` and
`Reconstructor` are always the real classes: building a `Tree("key_value", [...])` to hand to a
function is constructing its input, not mocking it, and it is how the awkward branches are
reached. What gets faked is what a module imports from *other modules in this package*. Most
modules need no grammar; `grammar_compiled`, `entry_style`, `tree_edits`, `config_resize` and
`config_store` keep a real one, because their collaborator *is* a lark object.

Four mechanics, all of which bite:

- **`unittest.mock`, never `monkeypatch`.** `patch.object` throughout, and
  `patch.multiple(module, a=..., b=...)` for a run of collaborators.
- **Patch the consumer, not the source.** Every cross-module import is
  `from access.config.X import name`, so each consumer binds its own reference:
  `patch.object(config_dict, "update_node_value", ...)` works, patching
  `tree_edits.update_node_value` does not. Where a module imports a whole module
  (`from access.config import parallel_mpi_grid`), patch that name on the consumer
  (`patch.object(parallel_component, "parallel_mpi_grid")`), not an attribute of the real module,
  which would leak to every other importer.
- **`ExitStack` when the fake is built late.** A fixture whose fakes close over arguments known
  only once its returned function is called (`offer` in `test_config_insert`) cannot use
  `with patch(...)` inside that function: the patch unwinds at once. Enter the patches on a
  `contextlib.ExitStack` the fixture holds, and they last until teardown.
- **`spec` sees methods, not bare-annotation fields.** `Mock(spec=LocalConstraint)` works, and
  is hashable, which constraints in frozen dataclasses need. On the dataclass-shaped types a
  field with no default is only an annotation: `Mock(spec=ComponentLayout).name` and
  `Mock(spec=ConfigStore).refs` raise `AttributeError`, while properties (`idle_cores`,
  `is_leaf`) come through. Use an unspecced `MagicMock` and assign what the code reads — `name`
  by assignment, since `name=` is reserved by `Mock`.

`conftest.py`'s `FakeStore` and `FakeContext` are hand-rolled on purpose: they are stateful
stubs, not spies. `FakeStore.add_path` mutates `refs` (and refuses the key paths it is told to),
`remove` deletes from it, `child` returns a fresh store, and `asked()` backs most assertions in
`test_config_dict.py`; `FakeContext` carries `normalise_key`, `key_case` and `accepts_key`. As
`MagicMock`s they would need that behaviour rebuilt in `side_effect` closures.

**Two files are deliberate exceptions**, one per half, because something has to prove the
isolated pieces still fit together:

- **`test_parser.py`** drives the whole parser stack through `parse()` and the mapping
  interface, so it is both `parser.py`'s test file and that half's integration suite. With the
  format modules' tests, it is what shows the sample files still round-trip byte for byte.
- **`test_parallel_layouts.py`** does the same for the parallel half: real component trees,
  real allocation strategies, real domains and grids, with `test_parallel_doctests.py` behind
  it. Its constraint doubles are about subject rather than isolation — borrowing a rule from
  `parallel_constraints` would test two things at once.

`tests/conftest.py` holds what more than one file needs: the fakes and node builders, the
compiled shipped grammars, the `CANONICAL` and `CHAIN_CANONICAL` tables of the text each shipped
parser declares for the first entry of a kind, for one key and for a chain of them (asserted
from the format modules, where a statement about a real grammar belongs), and the made-up
`TOY_GRAMMAR` and `TOY_TEMPLATES` that `test_parser.py` drives. `HeaderParser`, a miniature
field table for the chains and records the toy cannot write, is in `test_parser.py`.

## Commands

```shell
python -m pytest                # coverage is on by default via pyproject addopts
python -m pytest tests/test_parallel_layouts.py -q --no-cov   # fast single-module run
ruff format .
ruff check .
pre-commit run --all-files      # what CI's formatting job runs
```

The documentation is built in an environment of its own, the way CI builds it:

```shell
pip install -r documentation/requirements.txt
READTHEDOCS_CANONICAL_URL=https://access-config-utils.readthedocs.io/latest/ \
    mkdocs build --strict -f documentation/mkdocs.yml
```

The variable stands in for the one Read the Docs sets, without which the build refuses to run.
The date at the foot of each page comes from git, so a page that has not been committed yet
makes the strict build warn.

CI (`.github/workflows/ci.yml`): pre-commit, PyPA build, conda build, then pytest on
3.11/3.12/3.13 with coverage uploaded to Codecov, and the strict documentation build. Releases
are tag-driven (`v*` → PyPI + anaconda); there is nothing to run locally for a release.

## Conventions

- Apache-2.0 header on every source file: the two-line
  `# Copyright 2025 ACCESS-NRI and contributors...` / `# SPDX-License-Identifier: Apache-2.0`
  block. `__init__.py` is the one without it.
- ruff, line length 120, rules `E,F,B,C4,SIM,PTH,UP,I,N,C901,W505`. `C901` (complexity) is
  enforced — prefer decomposition, and use `# noqa: C901` with a reason only where the function
  is genuinely irreducible.
- `from __future__ import annotations` at the top of new modules.
- Docstrings follow one convention across the whole package — see [Docstring
  conventions](#docstring-conventions) below.
- Type-annotate everything, including test functions (`-> None`).
- Tests: `pytest` with module-scoped fixtures; the `parallel_*` tests group cases in
  `class Test<Thing>:` blocks with no `__init__`. Coverage is **100%** across
  `src/access/config`, and new code keeps it there, error paths included.
- `documentation/` comes from the ACCESS-NRI template,
  [documentation-infra](https://github.com/ACCESS-NRI/documentation-infra), and is kept so that
  it can be updated from there: `hooks/`, `overrides/`, `docs/css/`, `docs/js/`,
  `docs/fontello/`, `docs/assets/logos/`, `requirements.txt` and `.readthedocs.yaml` stay as
  copied, and `mkdocs.yml` and `abbreviations.md` are changed only where the template marks them
  for it and where a comment says why.
- Credit the agent in commit messages, with a `Co-Authored-By:` trailer.

## Docstring conventions

**One style across the whole package: Google.** Do not reintroduce numpydoc (`Parameters` over
a row of dashes), `Attributes:` sections, or Sphinx cross-reference roles.

```python
@classmethod
def iter_grids(cls, ndim: int, n_ranks: int) -> Iterator[MPICartesianGrid]:
    """Yield every ``ndim``-dimensional grid that uses exactly *n_ranks* ranks.

    Args:
        ndim (int): Number of grid dimensions. Must be >= 1.
        n_ranks (int): Total number of MPI ranks. Must be >= 1.

    Yields:
        MPICartesianGrid: Each valid grid, in lexicographic order of its shape.

    Examples:
        >>> [g.shape for g in MPICartesianGrid.iter_grids(2, 4)]
        [(1, 4), (2, 2), (4, 1)]
    """
```

- **Sections** are `Args:` / `Returns:` / `Yields:` / `Raises:` / `Examples:` —
  colon-terminated, never underlined, always in that order. One blank line before each section,
  none after the heading and none between entries. Entries are indented +4 from the heading,
  continuations +8.
- **Width**: docstrings and standalone comments wrap at **92** columns; code keeps
  `line-length = 120`. Ruff's `W505` (`max-doc-length = 92`) fails the build on an over-long doc
  line. 92 rather than 88 so the copyright header line fits unchanged.
- **Types are repeated** in the docstring even though everything is annotated:
  `name (Type): Description.` in `Args:`, a leading `Type: Description.` in
  `Returns:`/`Yields:`, and `ExceptionType: Condition.` in `Raises:`. Spell them as the
  annotation does (`list[int]`, `int | None`). That is for the public API and the `parallel_*`
  modules; the parser half's internals are shorter, as the next bullet says.
- **How much to write in the parser half.** The public API (`Config`, `ConfigList`,
  `ConfigRecord`, `ConfigBlockList`, `ConfigParser` and its properties, the parser classes,
  `YAMLParser`/`YAMLConfig`, `UnsupportedEntryError`) and every module docstring get the full
  treatment. A package-internal function keeps its summary, explanatory paragraphs, `Raises:`
  and `Examples:` and drops `Args:`/`Returns:`/`Yields:`, putting a contract the annotations
  miss (what `None` means, an ordering) in a sentence. A private helper (`_name`) is a summary
  and at most two more lines. **Design rationale is written once, concisely**, in the module
  docstring or a `#` comment beside the code it explains (the deletion repairs in `tree_edits`,
  the layout rules in `list_layout`), not on each helper. A test is a "Test that ..." summary
  plus, only where the reason is not obvious, one line: what the real program does, or which bug
  the case pins. Fixtures and helpers get a line.
- **Constructor parameters are documented on the class**, in its `Args:` section, never on
  `__init__`; that covers dataclass and `NamedTuple` fields. Attributes that are *not*
  constructor parameters get an inline `#` comment instead — see `ConfigStore` in
  `config_store.py` and `ValueTypeHandler` in `grammar_values.py`. **`Enum` members are not
  constructor parameters either**: describe them in the class's prose and give each an inline
  `#` comment, as `CoreSharing` does in `parallel_component.py`.
- **Markup is reST inline only**: ```` ``double backticks`` ```` for code and identifiers,
  `*emphasis*` when referring to a parameter, `**bold**` for emphasis. No `:class:`, `:attr:`,
  `:meth:`, `:exc:`, `:func:` or `:mod:` roles — there is no Sphinx build to render them.
- **The text is rendered as Markdown**, by mkdocstrings in the API reference, so its block
  structure is Markdown's: a blank line before a list, which is otherwise folded into the
  paragraph above it; a list of terms as `- **term**: definition`, never a reST definition list;
  an indented literal block introduced by a single colon, never `::`; and no continuation line
  of an entry starting with `>`, which reads as a blockquote. CI builds the documentation
  strictly, so an `Args:` entry griffe cannot match to the signature fails the build.
- **Prose sub-headings inside a module docstring** (the `Concepts` blocks in the `parallel_*`
  modules, `Lark parse tree structure` in `tree_navigation.py`) keep a reST dash underline
  matched to the title's length. These are prose, not sections. `Notes` and `See Also` are *not*
  sections either — write them as prose, or as a `Note: ...` line.
- **Quoting**: a one-line docstring keeps its text and closing `"""` on the same line and ends
  with a period. A multi-line docstring puts the summary on the opening `"""` line, then a
  blank line, and always closes on a line of its own. One space after a full stop.
- **The `Examples:` blocks in the `parallel_*` modules are executable doctests**, run by
  `tests/test_parallel_doctests.py`. Keep them runnable, and keep `>>>` lines within 92 columns
  by splitting the statement, not wrapping it.

Ruff enforces only the width, and the strict documentation build only what griffe cannot read.
Ruff's `D` (pydocstyle) rules are deliberately **off**: they would fail the build on gaps this
convention does not yet cover — functions that raise without a `Raises:` section, undocumented
`__post_init__` / `is_satisfied` overrides in `parallel_constraints.py`, the missing licence
header in `__init__.py`. Structure is a review matter; line length is not.

## Invariants — parser half

- **Grammar rule names are an API, not cosmetics.** `EntryReader` is a Lark `Interpreter` whose
  callbacks are named after rules, so every grammar must produce rules named (or aliased with
  `->`) `key_value`, `key_list`, `key_block`, `key_null`, and `key`. Aliasing is the normal
  case: `rfile_key_value: ws* key ":" ws* value line_end -> key_value`. The names live in
  `grammar_contract.py`; import them rather than writing the string literal, so a rename stays
  one edit. `EntryReader`'s callbacks are the one place that must spell them out, because that
  is how Lark dispatches, and a test pins them against `ENTRY_CATEGORIES`.
- The `key` rule's first child must be a `Token` holding the key text.
- Scalar values must use value-type rules whose names are registered in
  `VALUE_TYPE_HANDLER_REGISTRY`, normally imported from `config.lark` (Fortran defines
  `bare_exponent` and `fortran_identifier` in its own grammar). A value-type rule node has exactly
  one `Token` child.
- **Never `%ignore` whitespace that must survive round-tripping.** Capture it in the explicit
  `ws` rule from `config.lark`.
- **Two rules aliased to the same entry category must produce distinguishable children.** The
  reconstructor picks the rule to write a node with by matching its children, not by where it
  sits, and Lark drops anonymous string tokens before it looks. NUOPC's `key: value` (top
  level) and `key = value` (inside a `::` table) both reduced to a key, `ws` and a value, so a
  table entry could be written back in the top-level syntax, as a file that no longer parses.
  Keep the distinguishing punctuation in a rule of its own: `assign: "="` in `nuopc_config.py`,
  `separator: ","` in `fortran_nml.py`.
- `ConfigParser.parse` appends a `"\n"` to the input before parsing (so grammars need not
  handle a missing final newline), and `Config.__str__` strips one trailing newline back off.
- **`ConfigStore` owns the tree; `Config` owns the dict.** The store keeps `refs`, one
  `EntryRef` per key — the entry node, the value-type nodes, and the `block` node where there is
  one — and mutates those nodes in place on assignment. `Config` never touches a `Tree`: it
  normalises the key, decides update or addition, and wraps what the store reports into a
  `ConfigList` or a nested `Config`. Do not let either side reach across.
- Editing is restricted, but adding keys and resizing lists are supported. Assigning to a
  **new** key adds an entry — a scalar, a list of two or more values, `None` where the grammar
  has a `key_null` rule, a `dict` to create a block and fill it, or a list of dicts for records.
  What still fails: changing a value's type raises `TypeError`, as does a new element of a list
  whose elements all have one type, unless it has that type too (a mixed list, such as a Fortran
  derived type assigned as a list of its components, takes any); a list shorter than two
  elements (one, for records), or one written with a repeat count, array qualifiers, skipped
  positions or in several assignments, raises `ValueError` on a change of length; so does giving
  one element of a repeat count (`6*9.0`) a value of its own, though giving all of them the same
  one rewrites the repeated token (`6*9.5`); assigning to an **existing** block raises
  `SyntaxError` (you may create a block but not replace one); an unusable key name raises
  `ValueError`; and a value the grammar cannot express raises `UnsupportedEntryError` (a
  `TypeError` subclass). A list write that fails leaves the file and the list as they were:
  every value is checked before any node changes.
- **Indexed Fortran assignments are gathered into one list.** `v(1) = 1` and `v(2) = 2` are the
  two-element `V`; `EntryRef.values` holds what such an array reads as, and each element is
  edited through the node that wrote it. A position no entry writes, or a skipped one
  (`x = 1, , 3`), reads as null with no node behind it, so it can be read but not written. An
  unqualified assignment over a longer entry overlays it from the first position, as Fortran
  does: `x = 1, 2` then `x = 3` is `[3, 2]`. A qualifier the reader does not model, a
  multidimensional one or the index of a derived type (`a(1)%b`), stays in the key, so two of
  them are two keys. The known limit: an array of derived types written one element per line,
  as WW3 does (`INBND_POINT(1) = 2 2 F`, `INBND_POINT(2) = 2 11 T`), reads as one list of what
  its components hold.
- One property says how keys are cased, `ConfigParser.key_case`: `"keep"` (the default, for
  case-sensitive formats), `"upper"` for `FortranNMLParser`, `"lower"` for `FieldTableParser`.
  A parser has to declare nothing but its grammar. A **new** key is written into the file
  exactly as the caller spelled it, while the dict entry is normalised — idempotently, which is
  why `add_path` and `insert_entry` take only the spelled keys. Whether a new key is a usable
  name is asked of the grammar's own `key` rule (`ParseContext.accepts_key`, parsing with
  `start="key"`), never of `str.isidentifier`: a field table key may hold a hyphen or a slash.
- `ConfigParser.repeated_blocks` says what a block written twice means: `"merge"` (the default)
  reads one block holding both, `"separate"` keeps each occurrence as a block of its own, in a
  `ConfigBlockList`. Fortran is `"separate"`, because its `READ` stops at the first group of a
  name.
- `AddParent` asserts that no `Tree` is parented twice — if you build tree nodes by hand, every
  node must be a distinct object, never a shared instance. Never re-visit a tree that has
  already been annotated; the insertion path annotates each new subtree on its own before
  splicing it in.
- `dict` and `list` methods that CPython implements without dispatching to `__setitem__` /
  `__delitem__` are overridden on `Config` (`get`, `__contains__`, `update`, `setdefault`,
  `pop`, `popitem`, `clear`, `|=`) so they honour key normalisation and keep the tree in sync.
  **`ConfigList` is a view over the store**, holding only `(store, key)`: every write — an
  element, a slice, `append`, `pop`, `sort` and the rest — goes through
  `ConfigStore.replace(key, values, origins)` first, and changes the list only once the file
  has taken it. The same length is written in place; another length is laid out afresh by
  `config_resize`/`list_layout` around the elements that stay, given the old position each new
  element stands in for (`origins`: worked out by alignment when a whole list is assigned, exact
  for `append`, `extend` and `insert`, which are slice assignments). Each candidate layout is
  validated by parsing, and the entry keeps its node. **A copy is plain data**, through
  `__reduce_ex__`: a `ConfigList` or a `ConfigBlockList` copies, deep copies and pickles as a
  `list`, a `Config` or a `ConfigRecord` as a `dict`. The default copy of a `dict` subclass
  refills it through `__setitem__`, which would write every entry into the file again.
- **A value type that `collects` makes records.** `named_assignments` (a field table method of
  three strings) is one: the reader gathers every entry of the key into one `KEY_LIST` ref with
  `collected=True`, an entry per record, in file order — one record is a list of one. Each
  record is handed out as a **read-only** `ConfigRecord`: it is replaced whole, by assigning a
  dict to its place in the list (`rows[1] = {**rows[1], "itable": 45}`), and every other write
  raises `TypeError`. Resizing a list of records inserts and removes entries
  (`ConfigStore._resize_records`): a new record is a copy of its neighbour, and the change is
  all-or-nothing.
- **Deleting an entry can leave the tree underivable, so `__delitem__` repairs it.** A grammar
  may require entries and the text between them to alternate — Fortran's does — so removing one
  leaves two runs of text adjacent, which the start rule cannot derive and the reconstructor
  refuses to write out. `merge_adjacent_repetitions` rejoins them, preserving the text exactly.
  It only merges rules that are a plain repetition (`GrammarInfo.is_repetition_rule`), since
  only those accept the combined children; do not loosen that test.
- **A configuration with no entries left still writes out its comments and blank lines.**
  `__str__` falls back to the empty string only when the grammar cannot derive a container
  without entries (a `start: (...)+` form), where there is no text left to write. That keeps
  `str()` monotonic: nothing is hidden and then resurrected by the first addition.
- **Values are wrapped verbatim, never escaped**, so serialisation picks a quote character the
  value does not contain — keeping the one already in use where it can — and raises
  `UnsupportedEntryError` when the value contains both. Non-finite floats raise the same,
  because Python writes them as `inf`/`nan`, which no supported format reads back as a number.
  Both checks live in `grammar_values.py`, so updating a value is exactly as strict as adding
  one.

## Invariants — key addition

- **Adding a key never builds tree nodes by hand.** `entry_render` writes the *text* of the new
  entry — a copy of a neighbour, or the template the parser declares — and `parse_entry_nodes`
  hands it back to Lark with the enclosing rule as the start symbol. Lark therefore produces the
  nodes, so rule inlining, `->` aliases, filtered-out anonymous terminals and `%import` terminal
  renaming (`CNAME` → `config__CNAME`) all take care of themselves.
- **A format declares the text of its first entry of each kind; everything after that is a
  copy.** The text is never derived from the grammar. A parser's `entry_templates` maps
  `(container rule, entry category)` to text with `{key}`, `{keys:SEP}` and `{value}` markers.
  `{keys:SEP}` is every key of a chain parted by SEP, so the Fortran `a%b = 1` is
  `{keys:%} = {value}`, one template for a key path of any depth. A list template has two
  `{value}` markers, and the text between them is the separator longer lists repeat. Every kind
  the grammar can write has to be declared: a kind with no template and no neighbour to copy
  raises `UnsupportedEntryError`, and
  `test_a_shipped_parser_declares_a_template_for_every_entry_it_can_write` pins that for the
  shipped formats against the `CANONICAL` rows. A format that wants a chain written otherwise
  than with one repeated separator gets chains from donors only.
- **A new entry is first written as a copy of the nearest entry of its kind.** `donor_template`
  writes that entry out with markers in place of its keys and values and turns the text between
  them into fixed text, punctuation and all; `probe_entry_style` keeps one per
  `(depth, category)`, from the last entry on a line of its own above the insertion point. It is
  offered ahead of the declared template, and a value written with the same rule as the one
  copied keeps its notation (quotes, logical spelling, exponent letter). Comments, array
  qualifiers and block contents are left out, and a list broken after its first value is not
  copied at all. The only other thing read from the file is the *indentation* of the entry above
  (`EntryStyle.indent`, `None` where there is none), never the spacing inside an entry.
- **The candidates, in order**: the donor; the declared template with the indent of the entry
  above; the declared template with its own indent, the leading whitespace of the line holding
  its first key (NUOPC's table rows are ` {key} = {value}\n`, a field table method
  `\n  "{key}", {value}`). A grammar that refuses the file's indent at that place fails the
  parse check and the next candidate is tried. `UnsupportedEntryError` names the value types the
  format writes — the registry names that appear in the grammar's rules
  (`GrammarInfo.value_rules`), not the whole registry, or the run sequence would write `"ATM"`.
- **A key in a block with no body of its own is written as a chain.** A Fortran derived type
  (`a%b = 1`) and a field table header (`"tracer", "ocean_mod", "age"`) are entries holding
  entries; `entry_chain` follows them and `{keys:SEP}` writes them. A depth the grammar cannot
  write (a one-key field table header) fails the parse check, and `Config._write_path` goes one
  level deeper. `Config._add` walks `_parent` up to the nearest addable store, which writes the
  whole key path after the last line sharing the most leading keys (`ConfigStore.add_path`),
  spelled the way that line spells them; the configurations on the path are then re-read **in
  place** (`ConfigStore.reread(ref)` also points a nested store at the block as read afresh),
  so a reference a caller holds sees the new key. A non-empty dict the format cannot write as a
  block goes one level deeper (`nml["g"]["t"] = {"a": 1}` writes `t%a = 1`).
- Two grammar requirements follow, both checked by the tests rather than by code: the start
  rule must be named `start` and be **non-transparent** (`start:`, never `?start:`), and a
  `key_block`'s contents must be held by a rule *named* `block`, not one aliased with
  `-> block`.
- `Lark` is built with `start=["start", "block"]` (just `["start"]` for a grammar with no
  `block` rule), plus `key` where a grammar defines it (`OPTIONAL_START_RULES`), so **every**
  `parse()` call must pass `start=` explicitly or Lark raises `ConfigurationError`.
- `compile_grammar` is `functools.cache`d on the grammar text, so parsers sharing a grammar
  share the compiled result and a test that changes a grammar between parses needs nothing done
  (`compile_grammar.cache_clear()` forgets them all). `ConfigStore` holds the `ParseContext`
  because insertion needs the `Lark` instance long after parsing. `GRAMMAR_DIR` from the same
  module resolves `%import config...`; do not hardcode that path.
- **Candidate order is style, never correctness.** Every candidate is parsed and read back, and
  accepted only if it yields exactly the requested key and a value equal *and identical in
  type* to the one asked for; otherwise `Path("foo")` would be written as the bare word `foo`,
  which a grammar with an `identifier` rule reads back as a `str`. Nothing is spliced until a
  candidate passes. The same holds for the order of `VALUE_TYPE_HANDLER_REGISTRY`, which picks
  the rule when several accept a value's Python type (`logical` before `bool`, `float` before
  `double`, `string` before `identifier`; the comment above the registry says why): every
  accepting rule is tried in turn, so the order changes how output looks, never what it means.
  A parser cannot override it; `test_parser.py::test_config_add_key_value_types` pins it, and
  the `CANONICAL` rows pin what each real format writes.
- **A new entry never lands inside a comment.** `entry_insertion_index` starts one past the
  last entry-bearing child and steps over any run of comment lines beginning there, stopping at
  a blank line, the next entry, or the end of the container. A *trailing* comment belongs to the
  entry's node, so one past the entry can be the second line of a comment begun on the entry's
  own line, as MOM6 documents nearly every parameter; stopping at a blank line leaves the entry
  *between* two comment blocks. `is_comment_line` tells a comment line from a blank one (both
  are `empty_line`, told apart by a `comment`/`fortran_comment` node below), and a child holding
  an entry is never one. Fortran's top-level comments are the exception: every line between two
  namelist groups is one opaque `random_text` node, so a new group still goes above them.
- **Layout is copied from the nearest donor, never invented.** The entries of the container
  above the insertion point come first (`probe_entry_style(..., upto=)`); a block created by an
  assignment has none, so it takes the style of the nearest sibling block above it — the
  innermost block of a chain, for a field table header (`probe_sibling_block_style`, carried on
  `ConfigStore.fallback_style`). A block that was *empty in the file* is left to the parser's
  template; only a block this session created inherits. Where a grammar admits no leading
  whitespace, the candidate carrying the donated indent fails to parse and is skipped.

## Invariants — parallel layout half

- Every public type is a **frozen dataclass** that validates its arguments in `__post_init__`
  and raises `ValueError` with a message naming the class and field.
- `ComponentLayout` is the resolved result; `ParallelComponent` is the input structure;
  `AllocationStrategy` is a *separate parallel tree*, keyed by component name, describing how
  cores are split. Keeping structure and allocation separate is intentional — don't merge them.
- `AllocationStrategy` is **abstract, and the allocation mode is the subclass**:
  - `FixedAllocation` — `n_cores`, or `core_fraction` of the whole budget (exactly one).
  - `WeightedAllocation` — `weight`, reduced to lowest terms among siblings, which then share
    one integer multiplier.
  - `FreeAllocation` — `min_cores`/`max_cores` and/or `min_core_fraction`/`max_core_fraction`,
    offered at the multiples of `core_step`.
  - `WholeRangeAllocation` — no fields: the whole of a `SHARED` parent's range from the
    sub-component's own `core_offset`.

  A mixed mode is not rejected, it is unrepresentable. The base holds only what every mode
  shares (`thread_range`, `subcomponents`, `local_constraints`, `group_constraints`) and
  declares those `kw_only=True`, which leaves the positional slot free for
  `FixedAllocation(12)`.
- **Fractional bounds are resolved once, before the search.** A fraction is a share of
  `iter_layouts`' `total_cores`, turned into a count by `resolve_root_strategy`
  (`_resolve_fractions`), so the scheduler only ever sees absolute counts and one strategy tree
  serves a whole scaling study. A product within `_FRACTION_REL_TOL` of a whole number is that
  number (`0.29 * 100` is 29 cores, not 28). Where a `FreeAllocation` bound is given both ways,
  the absolute one is a guard rail: `min_cores` floors the resolved minimum, `max_cores` caps
  the resolved maximum. `core_step` offers only the multiples of the step the bounds admit, and
  bounds that admit none are a `ValueError`.
- **The root is a `RootAllocation`, and that is the whole class** — it adds no fields to the
  base. The base's four fields are exactly what a root may state, and every *sizing* field
  lives on a mode, so "the root cannot size itself" is expressed by the type. One `isinstance`
  check in `resolve_root_strategy` enforces it, and it is **required**: there is no mypy in CI
  and the package ships no `py.typed`, so the annotation alone enforces nothing.
  `RootAllocation` declares no `_phase`, so nesting one raises `AttributeError` naming the
  class; its `_iter_core_shares` exists only to satisfy the ABC and raises `TypeError` (not a
  generator, so it fires on the call).
- **There is no default mode.** `_validate_names` requires the two trees to name the same
  components exactly, at every level — unknown names *and* missing ones are both `ValueError`.
  A component the caller has no opinion about carries an explicit `FreeAllocation()`;
  `iter_layouts(allocations=None)` builds that tree via `FreeAllocation._tree_for` under a
  `RootAllocation` that `resolve_root_strategy` supplies. This is why `strategies_for` can index
  `subcomponents` directly. Don't reintroduce a default: it would put a mode's identity back on
  the base.
- **The scheduler is mode-agnostic**, and this is the invariant most worth protecting. Adding a
  mode means writing the class, declaring `_phase`, and implementing `_iter_core_shares`;
  nothing in `iter_core_splits` is edited, and there is no `isinstance` chain to extend. The
  divided-path protocol on `AllocationStrategy` is four members, two of which have defaults:
  - `_phase: ClassVar[int]` — where the mode sits in the schedule (fixed 0, weighted 1, free 2).
    Deliberately has **no default**, so a mode that forgets it fails by name on the first search
    rather than being served last by accident. `WholeRangeAllocation` declares 0 although it is
    never scheduled: `_group_by_mode` sorts on `_phase` before any mode is asked for shares, and
    an undeclared one would fail there instead of with the mode's own explanation.
  - `_iter_core_shares(allocs, budget)` — abstract, one shape for every mode (`FixedAllocation`
    yields its single share or nothing). `allocs` is never empty, and cores reserved for other
    modes are already subtracted from `budget`, so a mode never learns the others exist.
  - `_reserved_cores(allocs)` — defaults to 0, and is **a pruning hint only**: returning 0 is
    always correct and merely costs wasted work, because a group that cannot be served yields
    nothing and the split is abandoned anyway. Only `FreeAllocation` overrides it. Verified
    exhaustively — don't "fix" it into a correctness rule.
  - `_describe_infeasible(allocs, names, budget)` — defaults to `None`. Called only when a group
    yielded nothing, so it is off every hot path. Only `FixedAllocation` overrides it, because a
    fixed group's failure is the one the budget alone cannot explain.
- **Mode semantics live on the mode's class**, not in the enumerator: `FreeAllocation._core_range`,
  `_resolve_fractions`, the protocol members, and `AllocationStrategy.strategies_for` /
  `.resolve_thread_range` on the base. Nothing outside the mode classes should reach into
  `n_cores`/`core_fraction`/`weight`/`min_cores`/`core_step` to re-derive what a mode means.
- Don't bring back an `isinstance` chain for speed: `iter_core_splits` is ~0.7% of a search,
  and timings against the chain it replaced were flat to within noise. The single-group fast
  path (one mode holding every position in order) is what keeps the default all-free tree
  cheap — keep it. The group budget is clamped at zero so a group squeezed out by a later
  phase's reservation cannot report a negative "available" in the diagnostics.
- **Three names are public only because they cross the module boundary**: `iter_core_splits`,
  `iter_shared_core_shares` and `resolve_root_strategy`, plus the accessors
  `.resolve_thread_range` and `.strategies_for`. They are package-internal contract, not caller
  API: Python has no `internal`, and importing a `_`-prefixed name across modules is worse. (The
  one such import, `_occupied` into `parallel_layouts`, is a Housekeeping item.) Everything a
  mode knows about *itself* stays `_`-prefixed and inside `parallel_allocation_strategies`:
  `_validate_names`, `_core_range`, `_resolve_fractions`, `_iter_core_shares`,
  `_shared_core_range`, `_tree_for`, `_group_by_mode`. What a caller touches is the six classes'
  fields plus `iter_layouts`. Doctests run on private methods too, but keep *class-level*
  examples on the public API.
- Code that knows *all* the modes at once sits at module level below them, never on the base
  class, which would then name its own subclasses. `iter_core_splits` is such code, a
  sibling-group algorithm rather than per-node behaviour: weighted siblings share one
  multiplier, so no single strategy can work out its own share. It takes resolved strategies
  (`strategies_for(...)` output), not names.
- **A shared parent bypasses the phase schedule.** `iter_shared_core_shares` is the
  `CoreSharing.SHARED` counterpart of `iter_core_splits`: there is no budget to divide, so it
  asks each sibling for its `_shared_core_range` (of what is left from its `core_offset`) and
  yields the *product*. A mode joins that path by overriding `_shared_core_range` — Fixed, Free
  and WholeRange do; the base returns `None`, which the caller turns into a `ValueError` naming
  the sub-component. Weighted and WholeRange are mirror images, each meaningful on one path
  only: a weight is a share of a *divided* budget, so `WeightedAllocation` keeps the `None`; a
  whole range is all of a *shared* one, so `WholeRangeAllocation._iter_core_shares` raises
  `ValueError` (not a generator). Those two are the errors `iter_layouts` cannot raise eagerly,
  because a strategy's mode is not consulted until its parent is reached.
- `subcomponents` is `field(hash=False)`, so `@dataclass` generates `__eq__`/`__hash__` per
  class: equality compares the sub-tree, the hash skips the (unhashable) mapping, and the
  subtree memo relies on **equality** to separate strategies sharing a bucket. Don't add a
  hand-written hash to "make modes distinct" — `__eq__` already checks the class.
- **Cores are the currency, everywhere.** The root receives `total_cores`; every parent passes
  the cores it holds to its sub-components; every leaf turns its allocation into
  `n_cores // threads_per_rank` ranks. Ranks are never divided, because a rank costs as many
  cores as it runs threads.
- **A parent's sub-components either divide its cores or share them**, and its `CoreSharing`
  says which. `PARTITIONED` (the default): disjoint subsets, so the parent holds at least their
  total, and its `n_ranks` is their total. `SHARED`: one range, run on in turn, so the parent
  must *reach* as far as the furthest sub-component — `max(core_offset + n_cores)`, **not**
  always the biggest one — and its `n_ranks` is the ranks they sit on between them, the union
  of their `(core_offset, n_ranks)` spans (`_occupied`), an overlap counted once.
  `ComponentLayout._combine` applies the same either/or to what a parent *spends*
  (`used_cores`, hence `idle_cores`); `_validate_as_parent` computes the reach and the rank
  total itself, and `_LayoutSearch` the rank total.
- **A shared parent's sub-components must cover its range exactly**, leaving it no idle cores —
  `_validate_shared_coverage` refuses a gap at the front, a hole between two children, or slack
  past the furthest of them. Reaching far enough and covering are different questions; the
  reach check runs first, so an overrun is reported as one ("reach N core(s) into the range")
  rather than as idle cores — keep that order. This is a structural rule, not a tolerance: a
  shared range can be made smaller or started later, so a core in one that nothing runs on is
  paid for and left out of the run. A partitioned parent is under no such rule.
- **`core_offset` is a core index, as ESMF means `rootpe`** (its driver checks
  `rootpe + npes <= ncpus`, with `npes` counted in cores), and contiguous placement — ESMF's
  `pestride = 1` — is assumed because the model cannot express anything else. It is refused on
  a sub-component running more than one core per rank, where a core index and a PE index part
  company, and under a partitioned parent, where each child already starts where the last ended.
  The search offers a sub-component placed at an offset only its layouts running one rank per
  core (`_placeable_layouts`), so a thread range or idle cores below it narrow its layouts
  rather than ending the search with that refusal.
- A component is a **leaf or a parent, never both** (`ParallelComponent` rejects `domain`
  together with `subcomponents`), and that split runs through `ComponentLayout`: a leaf has a
  `threads_per_rank` and spends its cores exactly (`n_ranks × threads_per_rank == n_cores`); a
  parent has `threads_per_rank is None`, no decomposition, `n_ranks` as the bullet above says,
  and may leave cores idle. Thread constraints therefore only apply to leaves.
- Per-component thread counts come from `AllocationStrategy.thread_range`, which **inherits
  down the tree** from `RootAllocation.thread_range`. This is what lets UM run threaded next to
  an unthreaded MOM. The root's is the only one that is never `None`, which is why
  `iter_layouts` has no `thread_range` argument of its own; adding one would give the same
  setting two homes.
- Core splits under a `PARTITIONED` parent may leave cores idle by design; bound the slack with
  `MaxWastedCoreFractionConstraint` on the parent rather than changing the enumerator. On a
  `SHARED` parent the covering rule does that job by pruning: `iter_shared_core_shares` yields
  only the combinations that fill the range (on ACCESS-OM3's tree, a little over half the
  layouts and more than half the time). What is left is still a product — four unconstrained
  siblings on 275 cores is most of 275⁴ — so shrink it with allocations: `WholeRangeAllocation`
  or `FixedAllocation` to pin a sibling, narrow bounds or a `core_step` to thin one.
  `EqualCoresGroupConstraint` says "all the same count" in one rule, but filters the product
  rather than shrinking it.
- One `iter_layouts` call is one `_LayoutSearch`. The two things invariant for a whole search —
  `total_cores` and the subtree memo — are its state; `(component, n_cores, thread_range,
  strategy)` varies with the recursion and is exactly the memo key. That is why the key omits
  `total_cores`: a cache never outlives the budget it was filled under. Don't make the search
  reusable across budgets, and don't hoist the cache to module level.
- `iter_layouts` is the only entry point and is **lazy** — layout counts reach millions at
  production core counts, so nothing in the library materialises a result set; a caller that
  knows its search is small takes `list()` of it. It is deliberately *not* a generator function
  — it validates eagerly and returns a lazy pipeline, so bad arguments raise at the call rather
  than at the first `next()`.
- **Each module logs under its own name**: `access.config.parallel_layouts` (dead-end
  subtrees, unusable thread counts, an empty search) and
  `access.config.parallel_allocation_strategies` (a core split the fixed allocations cannot
  fit). pytest's `caplog` leaves the *root* logger at WARNING, so a test naming the wrong one
  sees no record at all.
- Both modules carry an `__all__`: `parallel_allocation_strategies.__all__` is the abstract
  base, `RootAllocation` and the four modes; `parallel_layouts.__all__` is `iter_layouts` alone
  — the classes are imported from where they are defined, not re-exported. Names in neither
  are an import, module state (`logger`), `_`-prefixed, or one of the cross-module contract
  names above. Ruff's `F822` fails the build on a name in `__all__` that no longer exists, so a
  rename must touch both. `__all__` constrains `import *` only; it does not stop an explicit
  import, and it says nothing about which *methods* are internal.
- `LocalConstraint.is_satisfied(layout)` filters a single component;
  `GroupConstraint.is_satisfied(sub_layouts)` filters siblings and must be attached to the
  **parent's** `group_constraints`. A constraint sees only the layout it judges: a rule stated
  as a share of the whole budget is an allocation bound (`FixedAllocation(core_fraction=...)`,
  `FreeAllocation(max_core_fraction=...)`), resolved to a core count before the search starts.
- A constraint returns `False` when the layout is valid-but-rejected, and **raises
  `ValueError`** when it was applied to something it cannot evaluate (e.g. a domain constraint
  on a component with no decomposition). Do not silently return `False` in that case.
- Group constraints reference siblings by **name**, never by positional index.
- `Domain` (physical grid points) and `MPICartesianGrid` (rank counts) are distinct concepts;
  `DomainDecompositionSpec` pairs them but deliberately does *not* compute per-rank extents or
  remainder distribution — only `mean_local_shape`.

## Task guides

Step-by-step guides for the recurring tasks in this repo live in `skills/`, one directory per
guide with the instructions in `SKILL.md`. They are plain Markdown with a YAML header (`name`,
`description`) — readable directly, and discoverable by any tool that supports that convention.
Read the relevant one *before* starting the task:

| Guide | Read it when |
|---|---|
| [`skills/add-parser/SKILL.md`](skills/add-parser/SKILL.md) | adding a parser for a new configuration file format |
| [`skills/grammar-debug/SKILL.md`](skills/grammar-debug/SKILL.md) | a file fails to parse, a round-trip differs, a key is missing, or an edit corrupts the output |
| [`skills/add-parallel-constraint/SKILL.md`](skills/add-parallel-constraint/SKILL.md) | adding a constraint to the layout enumeration library |
| [`skills/check/SKILL.md`](skills/check/SKILL.md) | finishing a change, before committing or opening a PR |

`.claude/skills` is a symlink to `skills/` so that Claude Code auto-discovers them; the guides
themselves are tool-neutral.

## Housekeeping notes

- `generate_esm1p6_layouts.py` sits untracked at the repo root — decide whether it belongs in
  the package or in `.gitignore` rather than leaving it in limbo.
- Only half the doctests are guarded. `tests/test_parallel_doctests.py` runs the `Examples:`
  blocks of the six `parallel_*` modules, but there is no `--doctest-modules` in `addopts`, so
  the blocks in `entry_fragments.py` and `entry_render.py` pass unwatched. Run
  `python -m pytest --doctest-modules src/access/config/ --no-cov` by hand after touching one
  of those, or extend the doctest module to cover them.
- `parallel_layouts` imports the private `_occupied` from `parallel_component` to compute a
  shared parent's `n_ranks`, which `_validate_as_parent` also computes for itself. A public
  helper in `parallel_component` would give the rule one home and end the private import.
- `iter_layouts`' docstring is behind the code: its `Raises:` names only the
  `WeightedAllocation` lazy error, not `WholeRangeAllocation` under a partitioned parent, and its
  logging note names only the `access.config.parallel_layouts` logger.
- 53 test functions predate the `-> None` convention (44 of them in `test_parser.py`).
- Nothing declares the Python floor: `pyproject.toml` has no `requires-python` and
  `conda/meta.yaml` says `python >=3.10`, while `typing.Self` makes it 3.11.
