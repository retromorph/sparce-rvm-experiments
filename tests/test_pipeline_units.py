"""Unit tests: dataset splits/leakage, renaming, memory methods, rules, admission, sandbox contract."""

from __future__ import annotations

import numpy as np
import pytest

from sparce.analysis import admit_event
from sparce.catalog import ApiChange, Program, SiblingProgram, validate_change
from sparce.dataset import SplitConfig, build_tasks, leakage_audit, rename_identifiers
from sparce.embedding import HashEmbedder, joint_text
from sparce.environments import environment_for, make_env_id
from sparce.memory import PairExample, ScopedKNN, ScopedRVM, VersionRules, identifier_tokens, make_method
from sparce.sandbox import RUNNER_SOURCE, SandboxJob, SandboxLimits, docker_available, DockerSandbox


def _change(i: int, package: str = "lib", n: int = 4) -> ApiChange:
    programs = tuple(Program(f"p{j}", f"import {package}\ndef f{j}(x):\n    return {package}.old{i}(x) + {j}\n",
                             f"import {package}\ndef f{j}(x):\n    return {package}.new{i}(x) + {j}\n",
                             f"import solution\nassert solution.f{j}(1) == {j + 1}\n") for j in range(n))
    sibs = (SiblingProgram("s0", f"import {package}\ndef g(x):\n    return {package}.stable(x)\n", "import solution\nassert solution.g(1) == 1\n"),)
    return ApiChange(package, f"{package}.old{i}", f"{package}.old{i}", f"{package}.new{i}", "1.0", "2.0", f"2024-0{1 + i % 9}-01",
                     "https://example.invalid", "x", programs, sibs)


def test_catalog_validation_rejects_bad_programs():
    bad = ApiChange("lib", "lib.a", "a", "b", "1", "2", "2024-01-01", "u", "s",
                    (Program("p", "x = 1\n", "x = 1\n", "print(1)\n"),))
    problems = validate_change(bad)
    assert any("equals" in p for p in problems) and any("import solution" in p for p in problems)


def test_split_invariants_and_leakage_audit():
    changes = [_change(i) for i in range(10)]
    built = build_tasks(changes, SplitConfig(seed=3), variants=2)
    audit = leakage_audit(built.tasks)
    assert audit["ok"], audit
    assert audit["clusters_straddling_splits"] == []
    heldout = {t.family_id for t in built.tasks if t.subset == "heldout_family"}
    assert len(heldout) == 2  # 20% of 10 families
    assert all(t.split == "test" for t in built.tasks if t.family_id in heldout)
    kinds = {t.kind for t in built.tasks}
    assert kinds == {"repair", "previously_correct", "old_version", "near_api", "near_api_old"}
    # hidden records are separate from public tasks and never contain tests in the public record
    assert all(not hasattr(t, "test_code") for t in built.tasks)
    assert len(built.hidden) == len(built.tasks)


def test_rename_variants_are_consistent_and_keep_api_names():
    code = "import numpy as np\ndef area(xs, ys):\n    total = np.trapz(ys, x=xs)\n    return float(total)\n"
    test = "import solution\nassert solution.area([0, 1], [1, 1]) == 1.0\n"
    new_code, new_test, mapping = rename_identifiers(code, test, "seed", {"trapz"})
    assert "trapz" in new_code and "area" not in new_code.replace(mapping["area"], "")
    assert mapping["area"] in new_test and "x=xs" in new_code  # keyword argument key untouched


def test_environment_ids_are_deterministic():
    a = environment_for("numpy", "2.1.3", (), "3.11")
    b = environment_for("numpy", "2.1.3", (), "3.11")
    assert a.env_id == b.env_id and a.image_tag.startswith("sparce-env:")
    assert make_env_id("3.11", ("numpy==2.1.3",)) != make_env_id("3.11", ("numpy==1.26.4",))
    assert "FROM python:3.11-slim" in a.dockerfile() and 'USER 65534' in a.dockerfile()


def _examples(n: int, seed: int, scope: str = "lib==2.0") -> list[PairExample]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        good = rng.random() < 0.5
        base = np.array([1.0, 0.0, 0.0] if good else [0.0, 1.0, 0.0]) + rng.normal(0, 0.05, 3)
        other = np.array([0.0, 1.0, 0.0] if good else [1.0, 0.0, 0.0]) + rng.normal(0, 0.05, 3)
        out.append(PairExample(f"t{seed}-{i}", f"c{i // 2}", "f", "repair", "train", "known_family", scope, "2024",
                               base, other, -0.5, good, not good, evidence_hash="sha256:abc"))
    return out


def test_rvm_scope_gate_and_snapshot_round_trip():
    train = _examples(60, 1)
    m = ScopedRVM(capacity=8, min_support=0.3, margin_threshold=0.0, z_score=1.0)
    m.fit(train, seed=0)
    assert 0 < m.active_count <= 8
    test = _examples(20, 2)
    decisions = [m.decide(e) for e in test]
    acc = np.mean([(d.recommend_alternative and e.a_pass) or (not d.recommend_alternative and e.b_pass) for d, e in zip(decisions, test)])
    assert acc > 0.8
    other_scope = PairExample(**{**test[0].__dict__, "scope": "lib==3.0"})
    assert m.decide(other_scope).reason == "scope_mismatch"
    m2 = ScopedRVM(capacity=8, min_support=0.3, margin_threshold=0.0, z_score=1.0)
    m2.restore(m.snapshot())
    assert all(m2.decide(e).margin == m.decide(e).margin for e in test)
    b = m.bytes_breakdown()
    assert b["covariance"] == m.active_count ** 2 * 8 and b["total"] > b["prototypes"]


def test_knn_variants_capacity_and_bytes():
    train = _examples(60, 3)
    for name in ("knn_cap", "random", "medoids"):
        m = make_method(name, capacity=6, seed=0, params={"k": 3, "threshold": 0.0, "min_support": 0.3})
        m.fit(train, seed=0)
        assert m.active_count == 6 and m.bytes_breakdown()["covariance"] == 0
    full = make_method("knn_full", params={"k": 3, "threshold": 0.0, "min_support": 0.3})
    full.fit(train, seed=0)
    assert full.active_count > 6


def test_rules_are_scope_exact():
    ex = PairExample("t", "c", "f", "repair", "train", "known_family", "numpy==2.1.3", "2024", np.ones(3), np.ones(3), -0.3, True, False,
                     identifier_tokens("import numpy as np\ny = np.trapz(a)"), identifier_tokens("y = np.trapezoid(a)"),
                     identifier_tokens("y = np.trapz(a)"), evidence_hash="sha256:x")
    rules = VersionRules()
    rules.fit([ex])
    assert rules.active_count == 1
    assert rules.decide(ex).recommend_alternative
    assert rules.decide(PairExample(**{**ex.__dict__, "scope": "numpy==1.26.4"})).reason == "scope_mismatch"


def test_admission_rejects_unverified_submissions():
    ex = _examples(1, 5)[0]
    assert admit_event(ex)[0]
    assert admit_event(PairExample(**{**ex.__dict__, "verified": False}))[0] is False
    assert admit_event(PairExample(**{**ex.__dict__, "evidence_hash": ""}))[0] is False


def test_joint_text_is_edit_focused_and_hash_embedder_normalised():
    txt = joint_text("numpy", "2.1.3", "a = np.trapz(y)\n", "AttributeError: module 'numpy' has no attribute 'trapz'", "a = np.trapezoid(y)\n")
    assert "-a = np.trapz(y)" in txt and "+a = np.trapezoid(y)" in txt and "scope: numpy==2.1.3" in txt
    v = HashEmbedder(32).embed([txt, "other"])
    assert v.shape == (2, 32) and np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_runner_source_has_isolation_controls():
    compile(RUNNER_SOURCE, "runner.py", "exec")
    assert "RLIMIT_AS" in RUNNER_SOURCE and "timeout=timeout" in RUNNER_SOURCE
    limits = SandboxLimits()
    assert limits.walltime_seconds == 30 and limits.pids == 128


@pytest.mark.skipif(not docker_available(), reason="docker not available (CI/CPU)")
def test_docker_sandbox_blocks_network_and_detects_failures():
    sb = DockerSandbox("python:3.11-slim", "test")
    jobs = [SandboxJob("ok", "def f():\n    return 1\n", "import solution\nassert solution.f() == 1\n"),
            SandboxJob("bad", "def f():\n    return 2\n", "import solution\nassert solution.f() == 1\n"),
            SandboxJob("net", "import urllib.request\n", "import solution, urllib.request\nurllib.request.urlopen('http://example.com', timeout=3)\n"),
            SandboxJob("slow", "import time\n", "import solution, time\ntime.sleep(60)\n")]
    sb.limits = SandboxLimits(walltime_seconds=3)
    res = {r.job_id: r for r in sb.run(jobs)}
    assert res["ok"].status == "pass" and res["bad"].status == "fail"
    assert res["net"].status == "fail" and res["slow"].status == "timeout"
