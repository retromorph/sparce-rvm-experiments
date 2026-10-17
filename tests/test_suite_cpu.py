"""CPU-only end-to-end test of the suite with fake LLM/embedder/sandbox (no docker, no downloads)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

import sparce.pipeline as pipeline
from sparce.catalog import ApiChange, Program, SiblingProgram
from sparce.embedding import HashEmbedder
from sparce.environments import BuiltImage, environment_for
from sparce.sandbox import SandboxResult, evidence_hash


def _change(i: int, n_programs: int = 6) -> ApiChange:
    programs = []
    for j in range(n_programs):
        old = textwrap.dedent(f'''
            import fakelib
            def compute_{j}(values, offset):
                total = fakelib.oldcall_{i}(values) + offset * {j + 1}
                return total
            ''')
        new = old.replace(f"oldcall_{i}", f"newcall_{i}")
        test = f"import solution\nassert solution.compute_{j}([1, 2], 1) == 3 + {j + 1}\n"
        programs.append(Program(f"p{j}", old, new, test))
    sibs = tuple(SiblingProgram(f"s{j}", f"import fakelib\ndef other_{j}(x):\n    return fakelib.stable_{i}(x) + {j}\n",
                                f"import solution\nassert solution.other_{j}(1) == {j + 1}\n") for j in range(2))
    return ApiChange("fakelib", f"fakelib.oldcall_{i}", f"fakelib.oldcall_{i}", f"fakelib.newcall_{i}", "1.0", "2.0",
                     f"2024-0{1 + i % 8}-15", "https://example.invalid/changes", "fake change", tuple(programs), sibs)


CHANGES = [_change(i) for i in range(8)]


class RuleSandbox:
    """Deterministic fake verifier: on the new pin, code passes iff it uses newcall and not oldcall."""

    def __init__(self, env_id: str, version: str):
        self.env_id, self.version = env_id, version
        self.image_digest = "fake"

    def run(self, jobs, **_):
        out = []
        for j in jobs:
            code = j.solution_code
            if "fakelib" not in code:
                status, tail = "fail", "NameError: fakelib"
            elif self.version == "2.0":
                status = "pass" if ("newcall_" in code and "oldcall_" not in code) or ("stable_" in code and "oldcall_" not in code) else "fail"
                tail = "" if status == "pass" else "AttributeError: module 'fakelib' has no attribute 'oldcall'"
            else:
                status = "pass" if "newcall_" not in code else "fail"
                tail = "" if status == "pass" else "AttributeError: module 'fakelib' has no attribute 'newcall'"
            out.append(SandboxResult(j.job_id, status, True, 0 if status == "pass" else 1, tail, 1.0, "fake", evidence_hash("fake", code, j.test_code, status, tail)))
        return out


class FakeLLM:
    """Greedy keeps the old call ~70% of the time; the sample fixes it ~50% of the time."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def metadata(self):
        return {"model_id": "fake", "revision": "0", "n_parameters": 1}

    def peak_vram_bytes(self):
        return 0

    def generate_candidates(self, task_ids, prompts, cfg):
        from sparce.llm import Generation
        out = []
        for tid, p in zip(task_ids, prompts):
            code = p.split("```python\n")[-1].split("```")[0]
            r = int(tid[:6], 16) % 10
            greedy = code.replace("oldcall_", "newcall_") if r < 3 else code
            sample = code.replace("oldcall_", "newcall_") if r % 2 == 0 else code.replace("oldcall_", "wrongcall_")
            wrong = code.replace("oldcall_", "brokencall_")
            out.append([Generation(greedy, greedy, True, 100, 50, "greedy", "eos", 10.0), Generation(sample, sample, True, 100, 50, "sample", "eos", 10.0),
                        Generation(wrong, wrong, True, 100, 50, "sample", "eos", 10.0)])
        return out

    def score_candidates(self, prompts, responses, batch_size=8):
        return [-0.1 - 0.001 * len(r) + (0.05 if "newcall_" in r and i % 3 == 0 else 0.0) for i, r in enumerate(responses)]


@pytest.fixture
def cfg(tmp_path):
    import tomllib
    c = tomllib.load(open(Path(__file__).resolve().parents[1] / "configs/a4000.toml", "rb"))
    c["run"]["output_dir"] = str(tmp_path / "report")
    c["run"]["seeds"] = [1, 2]
    c["eval"]["bootstrap_replicates"] = 200
    c["eval"]["capacities"] = [4, 8]
    c["eval"]["update_checkpoints"] = [5, 10, 100]
    c["model"]["num_candidates"] = 3
    c["rvm"]["length_scale_grid"] = [0.6]
    c["rvm"]["min_support_grid"] = [0.3]
    c["rvm"]["posterior_z_grid"] = [1.0]
    c["rvm"]["margin_threshold_grid"] = [0.0]
    c["knn"]["k_grid"] = [3]
    c["knn"]["threshold_grid"] = [0.0]
    c["knn"]["min_support_grid"] = [0.3]
    return c


def test_suite_end_to_end_cpu(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_catalog", lambda: CHANGES)
    monkeypatch.setattr(pipeline, "SANDBOX_FACTORY", lambda images, vcfg: {k: RuleSandbox(k, b.spec.package_version) for k, b in images.items()})
    from sparce.suite import run_suite
    envs = {}
    for c in CHANGES:
        for pin in (c.old_pin, c.new_pin):
            spec = environment_for(c.package, pin, (), c.python_version)
            envs[spec.env_id] = BuiltImage(spec, "sha256:fake")
    work = tmp_path / "work"
    res = run_suite(cfg, work_dir=work, resume=False, no_mlflow=True, llm_factory=lambda c: FakeLLM(),
                    embedder_factory=lambda c: HashEmbedder(), sandbox_images=envs)
    split = json.loads((work / "data" / "split_manifest.json").read_text())
    assert split["leakage_audit"]["ok"]
    assert split["n_excluded_by_execution"] == 0
    e1, e2, e3 = res["e1"], res["e2"], res["e3"]
    assert e1["decision"]["status"] in ("GO", "NO-GO", "INCONCLUSIVE")
    assert "rvm" in e1["methods"] and "knn_full" in e1["methods"] and "rag" in e1["methods"]
    assert e2["rows"] and e2["pareto_front"]
    print("ROLLBACK", json.dumps(e3["rollback"]), [ (r["checkpoint"], r["promoted"]) for r in e3["stream"]])
    assert e3["rollback"]["performed"] and e3["rollback"]["exact"]
    assert e3["false_submissions"]["n_admitted"] == 0
    report = Path(res["report"])
    assert report.exists() and "Overall decision" in report.read_text()
    # resume must skip everything cached and reproduce identical decisions
    res2 = run_suite(cfg, work_dir=work, resume=True, no_mlflow=True, llm_factory=lambda c: FakeLLM(),
                     embedder_factory=lambda c: HashEmbedder(), sandbox_images=envs, stages=("analyze",))
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("latency")}  # noqa: E731
    assert strip(res2["e1"]["methods"]["rvm"][0]["repair"]) == strip(e1["methods"]["rvm"][0]["repair"])
