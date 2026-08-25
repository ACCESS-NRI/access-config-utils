#!/usr/bin/env python3
# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Run f90nml's own test suite against this package's Fortran namelist parser.

f90nml (https://github.com/marshallward/f90nml) is the reference implementation for reading
Fortran namelists, and its test suite is the widest corpus of real namelist syntax there is.
This script points that suite at ``FortranNMLParser`` and reports what happens. It is a
diagnostic rather than a gate: it prints a report and exits 0 unless it could not run.

Only *reading* is substituted. ``f90nml.read`` and ``f90nml.reads`` are replaced by our
parser, while f90nml's ``Namelist``, its writer, ``patch`` and its CLI are left alone. Their
suite then runs unmodified, and every test that reads a file becomes a test of our reading.

Two runs, because one is not interpretable
------------------------------------------
Upstream declares no expected failures and skips nothing, so every test is meant to pass --
but not every test *can*. An sdist ships only ``*.nml *.txt *.py`` from ``tests/``, so the
handful of tests wanting a ``.json`` or ``.yaml`` fixture fail on any sdist however good the
parser is. Rather than special-case that, the suite is run twice, once untouched and once
shimmed, and only the *difference* is attributed to us. The reverse difference is reported
too: a test that passes with our reader and fails with theirs is worth noticing.

A second group has to come out of the denominator as well. Many upstream tests build a
``Namelist`` from a Python dict and never call a reader, so they pass whatever this parser
does. They are counted separately rather than inflating the pass rate.

Attributing a failure
---------------------
Which bucket a failing test belongs in is decided by what our parser did to the files that
test reads, worked out separately by ``compare_corpus`` -- not by the exception the test
raised, which for an assertion is a multi-line diff that classifies badly. The files a test
reads are recovered from its source with ``ast``.

The distinction that matters is the last bucket: tests such as ``test_indent`` and the
cogroup tests fail only because a ``Namelist`` rebuilt from our values carries none of the
formatting attributes their writer reads, nor the repeated groups it keeps beside the
mapping. Counting those against our parser would overstate the gap.

Usage:
    python scripts/run_f90nml_tests.py --source <extracted-f90nml-sdist>

    An f90nml source tree is not downloaded by this script. To obtain one::

        pip download f90nml --no-binary :all: --no-deps && tar xf f90nml-*.tar.gz
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import subprocess
import sys
import tempfile
import traceback
import unittest
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

TEST_MODULE = "test_f90nml.py"
"""Name of the upstream test module, looked for under ``--source``."""

SOURCE_HINT = f"""A source tree holds tests/{TEST_MODULE} beside the f90nml package. To obtain one:

    pip download f90nml --no-binary :all: --no-deps
    tar xf f90nml-*.tar.gz"""

CANNOT_READ = "cannot read the file"
VALUES_DIFFER = "values differ"
REJECTS_DIFFERENTLY = "rejects differently"
NOT_OUR_READING = "f90nml's writer/API, not us"
WE_ACCEPT = "we accept, f90nml rejects"
BOTH_REJECT = "rejected by both"
AGREE = "agree"

_SEVERITY = (CANNOT_READ, VALUES_DIFFER, WE_ACCEPT, BOTH_REJECT, AGREE)
"""File buckets, worst first. A test is attributed to the worst bucket it touches."""

_TEST_BUCKETS = (CANNOT_READ, VALUES_DIFFER, REJECTS_DIFFERENTLY, NOT_OUR_READING)
"""The buckets a failing test can land in, in the order the report lists them."""

_READING_CALLS = frozenset({"read", "reads", "patch"})
"""Callables whose first namelist argument is an input file, for the ``ast`` scan."""


# --- Values ---


def plain(value: Any) -> Any:
    """Convert a parsed configuration into plain dicts and lists, with lowercased keys.

    f90nml lowercases every key, and ``Namelist`` needs ordinary containers rather than the
    mapping types this package returns.

    Args:
        value (Any): A parsed value, mapping or sequence.

    Returns:
        Any: The same value as built-in types.
    """
    if isinstance(value, Mapping):
        return {str(key).lower(): plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [plain(item) for item in value]
    return value


def normalise(value: Any) -> Any:
    """Return *value* in a form two implementations can be compared by equality.

    ``NaN`` is not equal to itself, so it becomes a sentinel; otherwise a namelist holding
    one would read as a difference on every run.

    A mapping is walked by iterating it rather than through ``items()``, because for f90nml
    the two disagree about a group the file writes twice. ``items()`` reports the merge of
    the occurrences; iteration yields the key once per occurrence, with ``nml[key]``
    advancing through them. The occurrences are what f90nml means -- ``nml[key]`` outside a
    loop returns the list of them -- so a key seen more than once collects into a list,
    which is the shape this package now reads such a group as too.
    """
    if isinstance(value, Mapping):
        grouped: dict[str, list[Any]] = {}
        for key in value:
            grouped.setdefault(str(key).lower(), []).append(normalise(value[key]))
        return {key: items[0] if len(items) == 1 else items for key, items in grouped.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [normalise(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return "<nan>"
    return value


# --- The shim ---


_CALLED: list[str] = []
"""Set by the shim on every call, and cleared per test, to record which tests read."""


def install_shim() -> None:
    """Replace f90nml's readers with this package's parser, leaving the rest of it alone."""
    import f90nml

    from access.config import FortranNMLParser

    parser = FortranNMLParser()

    def reads(text: str) -> Any:
        _CALLED.append("reads")
        return f90nml.Namelist(plain(parser.parse(text)))

    def read(source: Any, *args: Any, **kwargs: Any) -> Any:
        if hasattr(source, "read"):
            return reads(source.read())
        return reads(Path(source).read_text())

    f90nml.read = read
    f90nml.reads = reads


class RecordingResult(unittest.TextTestResult):
    """A test result that also records which tests reached our parser.

    Whether a test exercises this package cannot be read reliably off its source: the CLI
    tests reach the parser through ``f90nml.cli`` without naming it, while others call
    ``.read()`` on a plain file object and only look as though they do. Recording the calls
    as they happen answers it exactly, and needs no heuristic.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reached: set[str] = set()

    def startTest(self, test: unittest.TestCase) -> None:  # noqa: N802 -- unittest's spelling
        _CALLED.clear()
        super().startTest(test)

    def stopTest(self, test: unittest.TestCase) -> None:  # noqa: N802 -- unittest's spelling
        if _CALLED:
            self.reached.add(test.id().split(".")[-1])
        super().stopTest(test)


# --- Locating a source tree ---


def locate(source: Path) -> tuple[Path, Path | None]:
    """Return the upstream tests directory, and the f90nml package beside it if present.

    Accepts either the root of a source tree or the tests directory itself.

    Args:
        source (Path): The ``--source`` argument.

    Returns:
        tuple[Path, Path | None]: The tests directory, and the package directory or ``None``
            when the tree ships no package and the installed f90nml must be used.

    Raises:
        SystemExit: If no upstream test module is found under *source*.
    """
    if not source.is_dir():
        raise SystemExit(f"error: --source {source} is not a directory.\n\n{SOURCE_HINT}")

    candidates = [source / "tests", source]
    tests = next((path for path in candidates if (path / TEST_MODULE).is_file()), None)
    if tests is None:
        looked = " or ".join(str(path / TEST_MODULE) for path in candidates)
        raise SystemExit(f"error: no upstream test module found: looked for {looked}.\n\n{SOURCE_HINT}")

    package = tests.parent / "f90nml"
    return tests, package if (package / "__init__.py").is_file() else None


def staged_copy(tests: Path, package: Path | None, destination: Path) -> Path:
    """Copy the upstream tree into *destination*, and return the staged tests directory.

    The suite writes ``tmp.nml`` beside the corpus and chdirs into a directory named
    ``tests``, so it runs against a copy: ``--source`` is left exactly as it was found.
    """
    staged = destination / "tests"
    shutil.copytree(tests, staged)
    if package is not None:
        shutil.copytree(package, destination / "f90nml")
    return staged


# --- Comparing the corpus, file by file ---


def read_ours(text: str) -> tuple[Any, str | None]:
    """Parse *text* with this package, returning the values or the error."""
    from access.config import FortranNMLParser

    try:
        return FortranNMLParser().parse(text), None
    except Exception as error:  # noqa: BLE001 -- any parse failure is a result, not a crash
        return None, f"{type(error).__name__}: {error}".splitlines()[0]


def read_theirs(path: Path) -> tuple[Any, str | None]:
    """Parse the file at *path* with f90nml, returning the values or the error.

    Warnings are silenced: the corpus is deliberately full of edge cases, and f90nml warns
    on several of them. What it made of the file is recorded in the result either way.
    """
    import f90nml

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f90nml.read(str(path)), None
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}".splitlines()[0]


def compare_file(path: Path) -> dict[str, Any]:
    """Compare one namelist as read by both implementations, and round-trip it through ours.

    Args:
        path (Path): The ``.nml`` file.

    Returns:
        dict[str, Any]: ``bucket`` for the values, ``round_trip`` for byte fidelity, and the
            error each implementation raised where it did.
    """
    text = path.read_text()
    ours, our_error = read_ours(text)
    theirs, their_error = read_theirs(path)

    if our_error and their_error:
        bucket = BOTH_REJECT
    elif our_error:
        bucket = CANNOT_READ
    elif their_error:
        bucket = WE_ACCEPT
    else:
        bucket = AGREE if normalise(plain(ours)) == normalise(theirs) else VALUES_DIFFER

    if our_error:
        # Keep "nobody can read this" apart from "we cannot": only the latter is our gap.
        round_trip = BOTH_REJECT if their_error else CANNOT_READ
    else:
        round_trip = "byte-exact" if str(ours) == text else "differs"

    return {
        "bucket": bucket,
        "round_trip": round_trip,
        "our_error": our_error,
        "their_error": their_error,
    }


def compare_corpus(tests: Path) -> dict[str, dict[str, Any]]:
    """Compare every ``.nml`` file under *tests*, keyed by file name."""
    return {path.name: compare_file(path) for path in sorted(tests.glob("*.nml"))}


# --- Which files a test reads ---


def _callee(node: ast.expr) -> str:
    """Return the trailing name of an attribute or reference, empty for anything else."""
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _nml_arguments(call: ast.Call) -> Iterator[str]:
    """Yield the namelist file names a reading call takes as input."""
    names = [arg.value for arg in call.args if isinstance(arg, ast.Constant) and str(arg.value).endswith(".nml")]
    if _callee(call.func) == "patch":
        # patch(input, changes, output): only the first names a file that gets read.
        names = names[:1]
    yield from names


def _is_rejection(call: ast.Call) -> bool:
    """Report whether *call* asserts that reading raises, as ``assertRaises(E, read, src)``.

    The reader is passed by reference rather than called, so it is an *argument* here and
    the scan for reading calls above cannot see it.
    """
    return _callee(call.func).startswith("assertRaises") and any(_callee(arg) in _READING_CALLS for arg in call.args)


def files_by_test(module: Path) -> dict[str, dict[str, Any]]:
    """Return what each test method reads, recovered statically from the test source.

    The upstream tests name their inputs as string literals, so scanning for reading calls
    recovers the mapping without running anything.

    Args:
        module (Path): The upstream test module.

    Returns:
        dict[str, dict[str, Any]]: Per test name, the ``files`` it reads and whether it
            ``rejects`` -- asserts that reading its input raises.
    """
    found: dict[str, dict[str, Any]] = {}
    for node in ast.walk(ast.parse(module.read_text())):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
            continue
        calls = [inner for inner in ast.walk(node) if isinstance(inner, ast.Call)]
        reading = [call for call in calls if _callee(call.func) in _READING_CALLS]
        found[node.name] = {
            "files": [name for call in reading for name in _nml_arguments(call)],
            "rejects": any(_is_rejection(call) for call in calls),
        }
    return found


def attribute(test: str, reads: Mapping[str, dict[str, Any]], corpus: Mapping[str, dict[str, Any]]) -> str:
    """Return the bucket a failing test belongs in, from what our parser did to its input.

    A test asserting that malformed input is *rejected* can only have failed because we did
    not reject it the same way -- either we accepted it, or we raised something other than
    the ``ValueError`` expected. Both are differences in our reading, so those come first.

    Otherwise the test is attributed to the worst bucket among the files it reads. One that
    reads nothing we could fault exercises f90nml's own writer or ``Namelist`` API, driven
    by values we read correctly.
    """
    entry = reads.get(test, {})
    if entry.get("rejects"):
        return REJECTS_DIFFERENTLY
    buckets = {corpus[name]["bucket"] for name in entry.get("files", ()) if name in corpus}
    worst = next((bucket for bucket in _SEVERITY if bucket in buckets), None)
    return worst if worst in (CANNOT_READ, VALUES_DIFFER) else NOT_OUR_READING


# --- Running the suite ---


def _case_names(suite: unittest.TestSuite) -> Iterator[str]:
    """Yield the method name of every test case in *suite*, however deeply nested."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _case_names(item)
        else:
            yield item.id().split(".")[-1]


def run_child(tests: Path, shimmed: bool, out: Path) -> None:
    """Run the upstream suite in this process and write the result to *out* as JSON.

    The result goes to a file rather than to stdout because upstream's CLI tests replace
    ``sys.stdout`` and never restore it, silently swallowing anything printed afterwards.
    """
    if shimmed:
        install_shim()

    sys.path.insert(0, str(tests))
    module = __import__(Path(TEST_MODULE).stem)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)

    # Before running: TestSuite.run() drops each case as it finishes, to free memory.
    names = sorted(_case_names(suite))
    runner = unittest.TextTestRunner(verbosity=0, stream=StringIO(), resultclass=RecordingResult)
    result = runner.run(suite)

    failed = {}
    for case, report in result.failures + result.errors:
        lines = [line for line in report.strip().splitlines() if line.strip()]
        failed[case.id().split(".")[-1]] = lines[-1].strip()[:200] if lines else ""

    payload = {
        "total": result.testsRun,
        "failed": failed,
        "tests": names,
        "reached": sorted(result.reached),
    }
    out.write_text(json.dumps(payload))


def run_suite(tests: Path, shimmed: bool) -> dict[str, Any]:
    """Run the upstream suite in a subprocess, and return its result.

    A subprocess rather than an in-process run so that the two runs cannot contaminate each
    other through upstream's global state, and so a hard crash in their suite is a
    reportable result rather than the end of this script.

    Args:
        tests (Path): The staged tests directory.
        shimmed (bool): Whether to substitute our parser for f90nml's reader.

    Returns:
        dict[str, Any]: ``total`` and ``failed``, the latter mapping test name to a summary.

    Raises:
        SystemExit: If the child produced no result.
    """
    with tempfile.TemporaryDirectory() as workspace:
        out = Path(workspace) / "result.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--child", str(tests), str(out)]
        if shimmed:
            command.append("--shim")
        process = subprocess.run(command, cwd=tests, capture_output=True, text=True, check=False)
        label = "shimmed" if shimmed else "baseline"
        result = json.loads(out.read_text()) if out.is_file() else {"error": process.stderr.strip()}
        if "error" in result:
            raise SystemExit(f"error: the {label} run failed.\n{result['error']}")
        return result


# --- Reporting ---


def _count_line(label: str, count: int, note: str = "", indent: int = 2) -> str:
    """Format one counted line of the report."""
    return f"{' ' * indent}{label:<{42 - indent}}{count:>4}{'   ' + note if note else ''}"


def _tally(names: Iterable[str], key: Any) -> dict[str, list[str]]:
    """Group *names* by *key*, preserving order within each group."""
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(key(name), []).append(name)
    return grouped


def build_report(
    baseline: dict[str, Any], shimmed: dict[str, Any], corpus: dict[str, dict[str, Any]], reads: Mapping[str, list[str]]
) -> dict[str, Any]:
    """Assemble the diff of the two runs and the corpus comparison into one structure.

    Upstream declares no expected failures and skips nothing, so every test is meant to pass
    where it can run at all. Two groups still have to come out of the denominator before the
    rest means anything: tests that fail *without* our reader, and tests that never called
    our reader at all -- recorded during the shimmed run -- which pass whatever we do.
    """
    unrunnable = sorted(baseline["failed"])
    ours = sorted(set(shimmed["failed"]) - set(unrunnable))
    fixed = sorted(set(unrunnable) - set(shimmed["failed"]))
    reached = set(shimmed["reached"]) | set(ours)
    inert = sorted(name for name in shimmed["tests"] if name not in unrunnable and name not in reached)

    return {
        "total": shimmed["total"],
        "unrunnable": unrunnable,
        "unrunnable_detail": baseline["failed"],
        "missing_fixture": sorted(n for n, why in baseline["failed"].items() if "FileNotFoundError" in why),
        "inert": inert,
        "attributable": _tally(ours, lambda name: attribute(name, reads, corpus)),
        "fixed": fixed,
        "reaching": shimmed["total"] - len(unrunnable) - len(inert),
        "passed": shimmed["total"] - len(unrunnable) - len(inert) - len(ours),
        "files": _tally(sorted(corpus), lambda name: corpus[name]["bucket"]),
        "round_trip": _tally(sorted(corpus), lambda name: corpus[name]["round_trip"]),
        "corpus": corpus,
    }


def _emitter(verbose: bool) -> tuple[Any, Any]:
    """Return an ``emit`` and a ``detail`` writer, both aimed at the real stdout.

    Prefer ``sys.__stdout__``: these helpers are also usable in-process, where upstream's
    CLI tests would have replaced ``sys.stdout``. It is ``None`` where there is none.
    """
    out = sys.__stdout__ or sys.stdout

    def emit(line: str = "") -> None:
        print(line, file=out)

    def detail(names: list[str], indent: int) -> None:
        for name in names if verbose else ():
            emit(f"{' ' * indent}- {name}")

    return emit, detail


def _print_suite(report: dict[str, Any], verbose: bool) -> None:
    """Print the two-run diff over the upstream suite."""
    emit, detail = _emitter(verbose)
    attributable = report["attributable"]
    unrunnable = report["unrunnable"]
    note = "fixture not in the source tree" if len(report["missing_fixture"]) == len(unrunnable) else "see -v"

    emit("upstream suite (upstream declares no expected failures and skips nothing)")
    emit(_count_line("fail without our reader", len(unrunnable), note))
    for name in unrunnable if verbose else ():
        emit(f"      - {name}: {report['unrunnable_detail'][name]}")
    emit(_count_line("never reach our reader", len(report["inert"]), "pass either way"))
    detail(report["inert"], 6)
    emit()
    emit(_count_line("reach our reader", report["reaching"]))
    emit(_count_line("pass", report["passed"], indent=6))
    for bucket in _TEST_BUCKETS:
        names = attributable.get(bucket, [])
        emit(_count_line(f"fail: {bucket}", len(names), indent=6))
        detail(names, 10)
    emit(_count_line("pass with ours, fail upstream", len(report["fixed"]), indent=6))
    detail(report["fixed"], 10)


def print_report(report: dict[str, Any], source: Path, verbose: bool) -> None:
    """Print the whole report: the suite diff, the corpus comparison and the round trip."""
    emit, detail = _emitter(verbose)

    emit()
    emit(f"f90nml at {source} -- {report['total']} tests, {len(report['corpus'])} namelists")
    emit()
    _print_suite(report, verbose)

    for title, tally, order in (
        ("the corpus, read by both", report["files"], _SEVERITY),
        ("round-trip through our parser", report["round_trip"], ("byte-exact", "differs", CANNOT_READ, BOTH_REJECT)),
    ):
        emit()
        emit(title)
        for bucket in order:
            names = tally.get(bucket, [])
            if names:
                emit(_count_line(bucket, len(names)))
                detail(names, 6)
    emit()


# --- Entry point ---


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=SOURCE_HINT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=Path, help="an extracted f90nml sdist or checkout")
    parser.add_argument("-v", "--verbose", action="store_true", help="list the tests and files in each group")
    parser.add_argument("--json", type=Path, metavar="PATH", help="also write the full report as JSON")
    parser.add_argument("--child", nargs=2, metavar=("TESTS", "OUT"), help=argparse.SUPPRESS)
    parser.add_argument("--shim", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run both passes over an upstream source tree and report the difference."""
    args = parse_args(argv)

    if args.child:
        out = Path(args.child[1])
        try:
            run_child(Path(args.child[0]), args.shim, out)
        except Exception:
            # Both streams may belong to upstream's CLI tests by now, so a traceback printed
            # here would vanish. The parent reads it back out of the result file instead.
            out.write_text(json.dumps({"error": traceback.format_exc()}))
            return 1
        return 0

    if args.source is None:
        raise SystemExit(f"error: --source is required.\n\n{SOURCE_HINT}")

    tests, package = locate(args.source.expanduser())
    with tempfile.TemporaryDirectory() as workspace:
        staged = staged_copy(tests, package, Path(workspace))
        if package is not None:
            sys.path.insert(0, workspace)
        corpus = compare_corpus(staged)
        report = build_report(
            run_suite(staged, shimmed=False),
            run_suite(staged, shimmed=True),
            corpus,
            files_by_test(staged / TEST_MODULE),
        )

    print_report(report, args.source, args.verbose)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
