"""Executable, version-pinned API-change repair benchmark construction.

Builds task instances from the first-party catalog (:mod:`sparce.catalog`),
creates negative controls, assigns simulated stream timestamps, performs a
chronological + API-family split at the *cluster* (program) level and writes
public and hidden records separately. Hidden records (reference fixes and
tests) are never passed to generation, retrieval or ranking.

Surface variants (consistent identifier renaming via ``tokenize``) are
near-duplicates by construction and therefore share the program's cluster id;
all paired statistics resample clusters, and cluster counts are reported.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib
import io
import json
import keyword
import math
import tokenize
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from sparce.catalog import ApiChange, validate_change
from sparce.environments import EnvironmentSpec, environment_for

CATALOG_MODULES = ("sparce.catalog.numpy_scipy", "sparce.catalog.pandas_sklearn", "sparce.catalog.misc")

KINDS = ("repair", "previously_correct", "old_version", "near_api", "near_api_old")


@dataclass(frozen=True)
class Task:
    task_id: str
    cluster_id: str
    family_id: str
    package: str
    api_family: str
    package_version: str       # target pin the code must run on
    python_version: str
    env_id: str
    kind: str                  # one of KINDS
    variant: int
    timestamp: str             # simulated stream time (ISO 8601, UTC)
    request_code: str          # code shown to the model
    change_url: str
    license: str
    split: str = ""            # train | val | test
    subset: str = ""           # known_family | heldout_family
    request_status: str = ""   # verifier status of request_code on env (filled after verification)
    request_stderr: str = ""   # sanitized traceback tail of request_code (filled after verification)

    def scope(self) -> dict[str, str]:
        return {"package": self.package, "version": self.package_version}


@dataclass(frozen=True)
class HiddenRecord:
    task_id: str
    reference_code: str        # code expected to pass on env (hidden)
    test_code: str             # hidden executable tests
    old_api: str
    new_api: str


def _sha(*parts: str, n: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf8"))
        h.update(b"\x1f")
    return h.hexdigest()[:n]


def load_catalog(modules: Iterable[str] | None = None) -> list[ApiChange]:
    import os
    if modules is None:
        override = os.environ.get("SPARCE_CATALOG_MODULES")
        modules = tuple(m.strip() for m in override.split(",") if m.strip()) if override else CATALOG_MODULES
    changes: list[ApiChange] = []
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        changes.extend(getattr(mod, "CHANGES", []))
    problems = [p for c in changes for p in validate_change(c)]
    if problems:
        raise ValueError("Catalog problems: " + "; ".join(problems[:10]))
    return changes


# --------------------------------------------------------------------------- renaming
_BUILTINS = set(dir(builtins))
_SUFFIXES = ("_v", "_x", "_data", "_val", "_out", "_tmp", "_obj", "_item", "2", "_cur")


def _collect_protected(code: str) -> set[str]:
    """Names that must NOT be renamed: imports, keyword-argument keys, dunders."""
    protected: set[str] = set()
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    depth = 0
    in_import = False
    for i, t in enumerate(toks):
        if t.type == tokenize.OP:
            if t.string in "([{":
                depth += 1
            elif t.string in ")]}":
                depth = max(0, depth - 1)
        if t.type == tokenize.NAME and t.string in ("import", "from"):
            in_import = True
        if t.type in (tokenize.NEWLINE, tokenize.NL) and depth == 0:
            in_import = False
        if t.type == tokenize.NAME:
            if in_import:
                protected.add(t.string)
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if depth > 0 and nxt is not None and nxt.type == tokenize.OP and nxt.string == "=":
                protected.add(t.string)  # keyword argument key or default parameter name
            if t.string.startswith("__"):
                protected.add(t.string)
    return protected


def rename_identifiers(code: str, test_code: str, seed: str, protected_extra: Iterable[str] = ()) -> tuple[str, str, dict[str, str]]:
    """Deterministically rename user-defined identifiers in code and tests.

    Attribute names (preceded by '.') are left alone, except ``solution.<name>``
    in the tests, which refers to the renamed module-level names.
    """
    protected = _collect_protected(code) | _collect_protected(test_code) | set(protected_extra)
    protected |= _BUILTINS | set(keyword.kwlist) | {"solution", "self", "cls"}
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    names: list[str] = []
    for i, t in enumerate(toks):
        if t.type != tokenize.NAME or t.string in protected or len(t.string) < 3:
            continue
        prev = toks[i - 1] if i > 0 else None
        if prev is not None and prev.type == tokenize.OP and prev.string == ".":
            continue
        if t.string not in names:
            names.append(t.string)
    mapping: dict[str, str] = {}
    taken = set(names) | protected
    for name in names:
        suffix = _SUFFIXES[int(_sha(seed, name), 16) % len(_SUFFIXES)]
        new = name + suffix
        while new in taken:
            new += "_"
        mapping[name] = new
        taken.add(new)

    def apply(src: str, module_attr: bool) -> str:
        out = []
        ts = list(tokenize.generate_tokens(io.StringIO(src).readline))
        for i, t in enumerate(ts):
            s = t.string
            if t.type == tokenize.NAME and s in mapping:
                prev = ts[i - 1] if i > 0 else None
                is_attr = prev is not None and prev.type == tokenize.OP and prev.string == "."
                if not is_attr:
                    s = mapping[s]
                elif module_attr and i >= 2 and ts[i - 2].string == "solution":
                    s = mapping[s]
            out.append(t._replace(string=s))
        return tokenize.untokenize(out)

    return apply(code, False), apply(test_code, True), mapping


# --------------------------------------------------------------------------- timestamps/splits
def simulated_timestamp(change_date: str, cluster_key: str, mean_days: float = 150.0) -> str:
    """Deterministic exponential offset after the breaking release date."""
    u = (int(_sha("ts", cluster_key), 16) % 1_000_000 + 0.5) / 1_000_000
    days = -mean_days * math.log(1 - u)
    base = datetime.fromisoformat(change_date).replace(tzinfo=timezone.utc)
    return (base + timedelta(days=days)).isoformat()


@dataclass
class SplitConfig:
    train_fraction: float = 0.45
    val_fraction: float = 0.15
    heldout_family_fraction: float = 0.2
    seed: int = 42


@dataclass
class BuildResult:
    tasks: list[Task]
    hidden: list[HiddenRecord]
    environments: dict[str, EnvironmentSpec]
    manifest: dict


def _license_for(change: ApiChange) -> str:
    return "first-party benchmark program (MIT); API change documented at " + change.change_url


def build_tasks(changes: list[ApiChange], split: SplitConfig | None = None, *, variants: int = 2) -> BuildResult:
    split = split or SplitConfig()
    tasks: list[Task] = []
    hidden: list[HiddenRecord] = []
    envs: dict[str, EnvironmentSpec] = {}
    rename_failures: list[str] = []

    for change in changes:
        new_env = environment_for(change.package, change.new_pin, change.extra_requirements, change.python_version, change.stdlib)
        old_py = change.old_pin if change.stdlib else change.python_version
        old_env = environment_for(change.package, change.old_pin, change.extra_requirements, old_py, change.stdlib)
        envs[new_env.env_id] = new_env
        envs[old_env.env_id] = old_env
        api_names = {change.old_api.split(".")[-1], change.new_api.split(".")[-1]}
        for program in change.programs:
            cluster = f"{change.family_id}/{program.name}"
            ts = simulated_timestamp(change.change_date, cluster)
            for v in range(variants):
                if v == 0:
                    old_c, new_c, test_c = program.old_code, program.new_code, program.test_code
                else:
                    try:
                        old_c, test_c, mapping = rename_identifiers(program.old_code, program.test_code, f"{cluster}:{v}", api_names)
                        new_c, _, _ = rename_identifiers(program.new_code, program.test_code, f"{cluster}:{v}", api_names)
                    except (tokenize.TokenError, SyntaxError, IndexError) as exc:
                        rename_failures.append(f"{cluster}:{v}:{exc}")
                        continue
                    if old_c == program.old_code:
                        continue
                specs = [
                    ("repair", new_env, old_c, new_c),
                    ("previously_correct", new_env, new_c, new_c),
                    ("old_version", old_env, old_c, old_c),
                ]
                for kind, env, req, ref in specs:
                    tid = _sha(cluster, kind, str(v), env.env_id)
                    tasks.append(Task(tid, cluster, change.family_id, change.package, change.api_family,
                                      env.package_version, env.python_version, env.env_id, kind, v, ts, req,
                                      change.change_url, _license_for(change)))
                    hidden.append(HiddenRecord(tid, ref, test_c, change.old_api, change.new_api))
        for sib in change.siblings:
            cluster = f"{change.family_id}/sib/{sib.name}"
            ts = simulated_timestamp(change.change_date, cluster)
            for v in range(variants):
                if v == 0:
                    code_c, test_c = sib.code, sib.test_code
                else:
                    try:
                        code_c, test_c, _ = rename_identifiers(sib.code, sib.test_code, f"{cluster}:{v}", api_names)
                    except (tokenize.TokenError, SyntaxError, IndexError) as exc:
                        rename_failures.append(f"{cluster}:{v}:{exc}")
                        continue
                for kind, env in (("near_api", new_env), ("near_api_old", old_env)):
                    tid = _sha(cluster, kind, str(v), env.env_id)
                    tasks.append(Task(tid, cluster, change.family_id, change.package, change.api_family,
                                      env.package_version, env.python_version, env.env_id, kind, v, ts, code_c,
                                      change.change_url, _license_for(change)))
                    hidden.append(HiddenRecord(tid, code_c, test_c, change.old_api, change.new_api))

    tasks = assign_splits(tasks, split)
    manifest = {
        "n_tasks": len(tasks),
        "n_clusters": len({t.cluster_id for t in tasks}),
        "n_families": len({t.family_id for t in tasks}),
        "kinds": {k: sum(1 for t in tasks if t.kind == k) for k in KINDS},
        "splits": {s: sum(1 for t in tasks if t.split == s and t.kind == "repair") for s in ("train", "val", "test")},
        "test_subsets": {s: sum(1 for t in tasks if t.split == "test" and t.kind == "repair" and t.subset == s)
                         for s in ("known_family", "heldout_family")},
        "rename_failures": rename_failures,
        "split_config": asdict(split),
        "environments": sorted(envs),
    }
    return BuildResult(tasks, hidden, envs, manifest)


def assign_splits(tasks: list[Task], split: SplitConfig) -> list[Task]:
    families = sorted({t.family_id for t in tasks})
    rng_key = lambda f: int(_sha("heldout", str(split.seed), f), 16)  # noqa: E731
    n_heldout = int(round(split.heldout_family_fraction * len(families)))
    heldout = set(sorted(families, key=rng_key)[:n_heldout])
    clusters: dict[str, str] = {}
    for t in tasks:
        clusters.setdefault(t.cluster_id, t.timestamp)
    # Chronological cutoffs computed on in-scope (non-heldout) repair-bearing clusters.
    in_scope = sorted(((ts, c) for c, ts in clusters.items() if c.split("/")[0] not in heldout and "/sib/" not in c))
    n = len(in_scope)
    n_train = int(split.train_fraction * n)
    n_val = int(split.val_fraction * n)
    train_cut = in_scope[n_train - 1][0] if n_train > 0 else ""
    val_cut = in_scope[n_train + n_val - 1][0] if n_val > 0 else train_cut
    out: list[Task] = []
    for t in tasks:
        fam_heldout = t.family_id in heldout
        ts = clusters[t.cluster_id]
        if fam_heldout:
            s = "test"
        elif ts <= train_cut:
            s = "train"
        elif ts <= val_cut:
            s = "val"
        else:
            s = "test"
        out.append(Task(**{**asdict(t), "split": s, "subset": "heldout_family" if fam_heldout else "known_family"}))
    return out


# --------------------------------------------------------------------------- IO
def write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("w", encoding="utf8") as fh:
        for row in rows:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False)
            fh.write(line + "\n")
            h.update(line.encode("utf8"))
    return "sha256:" + h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_dataset(result: BuildResult, root: Path) -> dict:
    public = root / "tasks.jsonl"
    hidden = root / "hidden" / "hidden.jsonl"
    h_public = write_jsonl(public, [asdict(t) for t in result.tasks])
    h_hidden = write_jsonl(hidden, [asdict(h) for h in result.hidden])
    manifest = {**result.manifest, "tasks_sha256": h_public, "hidden_sha256": h_hidden,
                "environments_detail": {k: v.to_dict() for k, v in result.environments.items()}}
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return manifest


def load_tasks(root: Path) -> tuple[list[Task], dict[str, HiddenRecord]]:
    tasks = [Task(**r) for r in read_jsonl(root / "tasks.jsonl")]
    hidden = {r["task_id"]: HiddenRecord(**r) for r in read_jsonl(root / "hidden" / "hidden.jsonl")}
    return tasks, hidden


def leakage_audit(tasks: list[Task]) -> dict:
    """Checks the split invariants that the protocol requires."""
    by_cluster: dict[str, set[str]] = {}
    by_family_split: dict[str, set[str]] = {}
    for t in tasks:
        by_cluster.setdefault(t.cluster_id, set()).add(t.split)
        by_family_split.setdefault(t.family_id, set()).add(t.split)
    straddling = [c for c, s in by_cluster.items() if len(s) > 1]
    heldout_leak = [f for f, s in by_family_split.items()
                    if any(t.subset == "heldout_family" and t.family_id == f for t in tasks) and s != {"test"}]
    train_max = max((t.timestamp for t in tasks if t.split == "train"), default="")
    test_min = min((t.timestamp for t in tasks if t.split == "test" and t.subset == "known_family"), default="9")
    dup_requests: dict[str, list[str]] = {}
    for t in tasks:
        dup_requests.setdefault(_sha(t.request_code, t.env_id), []).append(t.task_id)
    exact_dupes = [v for v in dup_requests.values() if len(v) > 1]
    return {
        "clusters_straddling_splits": straddling,
        "heldout_family_outside_test": heldout_leak,
        "train_max_timestamp": train_max,
        "known_family_test_min_timestamp": test_min,
        "chronology_ok": train_max < test_min,
        "exact_duplicate_request_groups": len(exact_dupes),
        "ok": not straddling and not heldout_leak and train_max < test_min,
    }
