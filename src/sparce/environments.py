"""Version-pinned Docker environments for the isolated verifier.

One image per distinct (python version, pinned requirement set). The image id
(sha256) is the ``environment_digest`` recorded for every verification.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentSpec:
    env_id: str
    python_version: str
    requirements: tuple[str, ...]     # sorted "pkg==ver" pins; empty for stdlib-only
    package: str
    package_version: str

    @property
    def image_tag(self) -> str:
        return f"sparce-env:{self.env_id}"

    def dockerfile(self) -> str:
        lines = [f"FROM python:{self.python_version}-slim",
                 "ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1"]
        if self.requirements:
            pins = " ".join(f'"{r}"' for r in self.requirements)
            lines.append(f"RUN pip install --no-cache-dir {pins}")
        lines.append("RUN python -c 'import sys; print(sys.version)'")
        lines.append('USER 65534:65534')
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        return {"env_id": self.env_id, "python_version": self.python_version,
                "requirements": list(self.requirements), "package": self.package,
                "package_version": self.package_version}


def make_env_id(python_version: str, requirements: tuple[str, ...]) -> str:
    key = f"py{python_version}|" + "|".join(sorted(requirements))
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    head = sorted(requirements)[0].replace("==", "-") if requirements else "stdlib"
    return f"py{python_version}-{head}-{digest}".replace("_", "-").lower()


def environment_for(package: str, version: str, extras: tuple[str, ...], python_version: str,
                    stdlib: bool = False) -> EnvironmentSpec:
    reqs: list[str] = [] if stdlib else [f"{package}=={version}"]
    for extra in extras:
        name = extra.split("==")[0].lower()
        if not any(r.split("==")[0].lower() == name for r in reqs):
            reqs.append(extra)
    reqs_t = tuple(sorted(reqs))
    py = version if stdlib else python_version
    return EnvironmentSpec(make_env_id(py, reqs_t), py, reqs_t, package, version)


@dataclass
class BuiltImage:
    spec: EnvironmentSpec
    image_id: str
    build_log_tail: str = ""
    ok: bool = True


def image_id(tag: str) -> str | None:
    proc = subprocess.run(["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def build_image(spec: EnvironmentSpec, *, rebuild: bool = False, timeout: int = 1200) -> BuiltImage:
    existing = image_id(spec.image_tag)
    if existing and not rebuild:
        return BuiltImage(spec, existing)
    with tempfile.TemporaryDirectory(prefix="sparce-img-") as d:
        (Path(d) / "Dockerfile").write_text(spec.dockerfile(), encoding="utf8")
        proc = subprocess.run(["docker", "build", "--network", "default", "-t", spec.image_tag, d],
                              capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        return BuiltImage(spec, "", (proc.stderr or proc.stdout)[-3000:], ok=False)
    return BuiltImage(spec, image_id(spec.image_tag) or "", (proc.stdout or "")[-500:], ok=True)


def write_manifest(images: list[BuiltImage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{**b.spec.to_dict(), "image_id": b.image_id, "ok": b.ok, "error": b.build_log_tail if not b.ok else ""}
               for b in images]
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")
