"""Isolated, network-disabled execution of UNTRUSTED candidate code.

Every candidate program is executed inside a Docker container built from a
version-pinned image (see :mod:`sparce.environments`), as an unprivileged user,
with ``--network none``, a read-only root filesystem, memory / pid / cpu quotas
and a per-job wall-clock timeout. Benchmark or generated code is NEVER executed
in the host Python process. Jobs for the same environment are batched into one
container invocation to amortise container start-up; each job still runs as its
own subprocess in a private scratch directory.

Statuses: ``pass`` (tests exit 0), ``fail`` (non-zero exit, i.e. exception or
assertion), ``timeout``, ``error`` (infrastructure failure: image missing,
docker unavailable). ``syntax_ok`` records whether the solution compiled.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

SANDBOX_UID = 65534  # 'nobody' in Debian-based python images

RUNNER_SOURCE = textwrap.dedent(
    '''
    """In-container job runner. Runs each job in its own subprocess with limits."""
    import json, os, resource, shutil, subprocess, sys, time, py_compile

    JOBS = "/jobs"; OUT = "/out/results.json"; WORK = "/work"
    timeout = float(os.environ.get("SPARCE_JOB_TIMEOUT", "30"))
    mem_bytes = int(os.environ.get("SPARCE_JOB_MEM_MB", "1536")) * 1024 * 1024

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))

    results = {}
    for job in sorted(os.listdir(JOBS)):
        src = os.path.join(JOBS, job)
        if not os.path.isdir(src):
            continue
        work = os.path.join(WORK, job)
        shutil.copytree(src, work)
        rec = {"status": "error", "syntax_ok": False, "returncode": None, "stderr_tail": "", "execution_ms": 0.0}
        try:
            py_compile.compile(os.path.join(work, "solution.py"), doraise=True)
            rec["syntax_ok"] = True
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "fail"; rec["stderr_tail"] = ("SyntaxError: " + str(exc))[-2000:]
            results[job] = rec; continue
        start = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-E", "-s", "-B", "-W", "ignore", "test_solution.py"], cwd=work,
                                  capture_output=True, text=True, timeout=timeout, preexec_fn=limits,
                                  env={"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", ""), "HOME": work,
                                       "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            rec["execution_ms"] = (time.perf_counter() - start) * 1000.0
            rec["returncode"] = proc.returncode
            rec["stderr_tail"] = (proc.stderr or "")[-2000:]
            rec["status"] = "pass" if proc.returncode == 0 else "fail"
        except subprocess.TimeoutExpired as exc:
            rec["execution_ms"] = (time.perf_counter() - start) * 1000.0
            rec["status"] = "timeout"; rec["stderr_tail"] = ((exc.stderr or b"")[-2000:]).decode("utf8", "replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")[-2000:]
        art = os.path.join(work, "artifact.json")
        if os.path.isfile(art) and os.path.getsize(art) <= 4 * 1024 * 1024:
            with open(art, "r", encoding="utf8", errors="replace") as fh:
                rec["artifact"] = fh.read()
        results[job] = rec
        shutil.rmtree(work, ignore_errors=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh)
    '''
)


@dataclass(frozen=True)
class SandboxJob:
    job_id: str
    solution_code: str
    test_code: str


@dataclass(frozen=True)
class SandboxResult:
    job_id: str
    status: str            # pass | fail | timeout | error
    syntax_ok: bool
    returncode: int | None
    stderr_tail: str
    execution_ms: float
    verifier_id: str
    evidence_hash: str     # sha256 over (image digest, solution, tests, status, stderr tail)
    artifact: str = ""     # optional job-produced artifact.json content (used for API inventories only)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SandboxLimits:
    walltime_seconds: int = 30
    memory_mb: int = 2048
    job_memory_mb: int = 1536
    pids: int = 128
    cpus: float = 2.0
    container_timeout_seconds: int = 900


def docker_available() -> bool:
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=30)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def evidence_hash(image_digest: str, solution: str, tests: str, status: str, stderr_tail: str) -> str:
    h = hashlib.sha256()
    for part in (image_digest, solution, tests, status, stderr_tail):
        h.update(part.encode("utf8"))
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()


class DockerSandbox:
    """Runs batches of jobs inside a pinned image. Thread-safe for batch calls."""

    def __init__(self, image: str, image_digest: str, limits: SandboxLimits | None = None,
                 verifier_id: str = "sparce-docker-verifier/v1"):
        self.image = image
        self.image_digest = image_digest
        self.limits = limits or SandboxLimits()
        self.verifier_id = verifier_id

    def _run_batch(self, jobs: Sequence[SandboxJob]) -> list[SandboxResult]:
        tmp_root = Path(tempfile.mkdtemp(prefix="sparce-sbx-"))
        jobs_dir = tmp_root / "jobs"
        out_dir = tmp_root / "out"
        jobs_dir.mkdir()
        out_dir.mkdir()
        os.chmod(out_dir, 0o777)
        for index, job in enumerate(jobs):
            d = jobs_dir / f"{index:05d}"
            d.mkdir()
            (d / "solution.py").write_text(job.solution_code, encoding="utf8")
            (d / "test_solution.py").write_text(job.test_code, encoding="utf8")
        (jobs_dir / "runner.py").write_text(RUNNER_SOURCE, encoding="utf8")
        os.chmod(jobs_dir, 0o755)
        for p in jobs_dir.rglob("*"):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        lim = self.limits
        cmd = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--user", f"{SANDBOX_UID}:{SANDBOX_UID}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", f"{lim.memory_mb}m", "--memory-swap", f"{lim.memory_mb}m",
            "--pids-limit", str(lim.pids), "--cpus", str(lim.cpus),
            "--tmpfs", f"/work:rw,size=512m,uid={SANDBOX_UID},gid={SANDBOX_UID},mode=1777",
            "--tmpfs", f"/tmp:rw,size=64m,uid={SANDBOX_UID},gid={SANDBOX_UID},mode=1777",
            "-v", f"{jobs_dir}:/jobs:ro", "-v", f"{out_dir}:/out:rw",
            "-e", f"SPARCE_JOB_TIMEOUT={lim.walltime_seconds}",
            "-e", f"SPARCE_JOB_MEM_MB={lim.job_memory_mb}",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            self.image, "python", "-I", "/jobs/runner.py",
        ]
        infra_error = ""
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=lim.container_timeout_seconds)
            if proc.returncode != 0:
                infra_error = (proc.stderr or "")[-2000:]
        except subprocess.TimeoutExpired:
            infra_error = "container wall-clock timeout"
        except FileNotFoundError:
            infra_error = "docker not found"
        results_path = out_dir / "results.json"
        raw: dict = {}
        if results_path.exists():
            try:
                raw = json.loads(results_path.read_text(encoding="utf8"))
            except json.JSONDecodeError:
                infra_error = infra_error or "unreadable results.json"
        shutil.rmtree(tmp_root, ignore_errors=True)
        out: list[SandboxResult] = []
        for index, job in enumerate(jobs):
            rec = raw.get(f"{index:05d}")
            if rec is None:
                rec = {"status": "error", "syntax_ok": False, "returncode": None,
                       "stderr_tail": infra_error or "no result", "execution_ms": 0.0}
            status = str(rec["status"])
            tail = str(rec.get("stderr_tail", ""))
            out.append(SandboxResult(
                job_id=job.job_id, status=status, syntax_ok=bool(rec.get("syntax_ok", False)),
                returncode=rec.get("returncode"), stderr_tail=tail,
                execution_ms=float(rec.get("execution_ms", 0.0)), verifier_id=self.verifier_id,
                evidence_hash=evidence_hash(self.image_digest, job.solution_code, job.test_code, status, tail),
                artifact=str(rec.get("artifact", "")),
            ))
        return out

    def run(self, jobs: Sequence[SandboxJob], *, batch_size: int = 24, workers: int = 4) -> list[SandboxResult]:
        if not jobs:
            return []
        batches = [list(jobs[i:i + batch_size]) for i in range(0, len(jobs), batch_size)]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(self._run_batch, batches))
        flat = [r for batch in results for r in batch]
        return flat


class FakeSandbox:
    """CPU/CI stand-in that executes NOTHING: returns 'unverified' for every job."""

    def __init__(self, verifier_id: str = "fake-no-execution"):
        self.verifier_id = verifier_id
        self.image_digest = "none"

    def run(self, jobs: Sequence[SandboxJob], **_: object) -> list[SandboxResult]:
        return [SandboxResult(j.job_id, "unverified", False, None, "", 0.0, self.verifier_id,
                              evidence_hash("none", j.solution_code, j.test_code, "unverified", ""))
                for j in jobs]


def runner_timing_sanity() -> float:
    """Cheap import-time check used by tests: runner source must compile."""
    start = time.perf_counter()
    compile(RUNNER_SOURCE, "runner.py", "exec")
    return time.perf_counter() - start
