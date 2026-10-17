"""Catalog of real, documented breaking API changes with executable programs.

Each :class:`ApiChange` describes one real API change between two pinned
versions of a Python package. Each :class:`Program` is a small, self-contained
program whose ``old_code`` runs on the legacy pin but FAILS on the updated pin,
and whose ``new_code`` passes on the updated pin. ``test_code`` is the HIDDEN
executable test (never shown to the LLM or any ranker); it imports the program
module as ``solution`` and asserts behaviour, not just absence of exceptions.

Programs are the clustering unit for splits and bootstrap (near-duplicate
surface variants are derived from a program and share its cluster id).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Program:
    name: str                 # short slug, unique within the ApiChange
    old_code: str             # uses legacy API; passes on old pin, fails on new pin
    new_code: str             # reference fix; passes on new pin (hidden from models)
    test_code: str            # hidden tests; `import solution` then assert behaviour
    description: str = ""     # one line: what the program computes


@dataclass(frozen=True)
class SiblingProgram:
    """Near-miss control: uses a related, UNCHANGED API of the same package.

    ``code`` must pass on BOTH the old and new pins without modification.
    """

    name: str
    code: str
    test_code: str
    description: str = ""


@dataclass(frozen=True)
class ApiChange:
    package: str              # pip distribution name, e.g. "numpy"
    api_family: str           # e.g. "numpy.trapz"
    old_api: str              # legacy call/identifier, e.g. "np.trapz"
    new_api: str              # replacement, e.g. "np.trapezoid"
    old_pin: str              # version where old_api works, e.g. "1.26.4"
    new_pin: str              # version where old_api fails, e.g. "2.1.3"
    change_date: str          # ISO date the breaking release was published
    change_url: str           # official changelog / docs URL for the change
    summary: str              # one-line human description
    programs: tuple[Program, ...]
    siblings: tuple[SiblingProgram, ...] = field(default_factory=tuple)
    extra_requirements: tuple[str, ...] = field(default_factory=tuple)  # pinned deps, e.g. ("pandas==2.2.3",)
    python_version: str = "3.11"   # docker base image python version (3.11/3.12/3.13)
    stdlib: bool = False           # True when the "package" is the Python interpreter itself

    @property
    def family_id(self) -> str:
        return f"{self.package}:{self.api_family}"


def validate_change(change: ApiChange) -> list[str]:
    """Static sanity checks (no execution). Returns a list of problems."""
    problems: list[str] = []
    if not change.programs:
        problems.append(f"{change.family_id}: no programs")
    names = [p.name for p in change.programs]
    if len(set(names)) != len(names):
        problems.append(f"{change.family_id}: duplicate program names")
    for p in change.programs:
        if "import solution" not in p.test_code and "from solution" not in p.test_code:
            problems.append(f"{change.family_id}/{p.name}: test_code must import solution")
        if p.old_code.strip() == p.new_code.strip():
            problems.append(f"{change.family_id}/{p.name}: old_code equals new_code")
    for s in change.siblings:
        if "import solution" not in s.test_code and "from solution" not in s.test_code:
            problems.append(f"{change.family_id}/sibling {s.name}: test_code must import solution")
    return problems
