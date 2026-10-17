"""Offline, pinned-version API-reference retrieval (baseline B2).

The corpus for an environment is the *installed package's own public API
inventory* (name, signature, first docstring line, and NumPy's expired-attribute
messages) dumped INSIDE the isolated sandbox for the exact pinned version.
Retrieval is deterministic lexical matching on identifiers taken from the
request code and the error headline. No hidden tests, reference fixes or
release notes written by us enter the corpus.
"""

from __future__ import annotations

import difflib
import json
import re
import textwrap
from typing import Sequence

INVENTORY_SCRIPT = textwrap.dedent(
    '''
    import importlib, inspect, json, pkgutil, sys, warnings
    warnings.simplefilter("ignore")
    package = sys.argv[1]
    entries = []
    seen = set()
    def add(qual, obj):
        if qual in seen: return
        seen.add(qual)
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = ""
        doc = (inspect.getdoc(obj) or "").strip().splitlines()
        entries.append({"name": qual, "signature": sig[:200], "doc": (doc[0] if doc else "")[:200]})
    def walk(modname, depth):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            return
        for name in dir(mod):
            if name.startswith("_"): continue
            try: obj = getattr(mod, name)
            except Exception: continue
            if inspect.ismodule(obj):
                if depth < 1 and getattr(obj, "__name__", "").startswith(package.split(".")[0]):
                    walk(obj.__name__, depth + 1)
                continue
            if callable(obj) or inspect.isclass(obj):
                add(modname + "." + name, obj)
    walk(package, 0)
    try:
        exp = importlib.import_module(package + "._expired_attrs_2_0").__expired_attributes__
        for k, v in exp.items():
            entries.append({"name": package + "." + k, "signature": "", "doc": "REMOVED: " + str(v)[:200]})
    except Exception:
        pass
    print(json.dumps(entries))
    '''
)

IMPORT_NAMES = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "scikit-learn": "sklearn",
                "pillow": "PIL", "networkx": "networkx", "jinja2": "jinja2", "pydantic": "pydantic"}

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def inventory_test_code(package: str) -> str:
    """Test-code wrapper so the inventory dump can run through the standard sandbox runner."""
    mod = IMPORT_NAMES.get(package, package)
    subs = {"scipy": ["scipy.integrate", "scipy.stats", "scipy.interpolate", "scipy.signal", "scipy.sparse", "scipy.misc", "scipy.linalg"],
            "sklearn": ["sklearn.metrics", "sklearn.preprocessing", "sklearn.linear_model", "sklearn.ensemble", "sklearn.tree", "sklearn.model_selection", "sklearn.datasets"],
            "PIL": ["PIL.Image", "PIL.ImageDraw", "PIL.ImageFont"],
            "pandas": ["pandas"], "numpy": ["numpy", "numpy.linalg"], "networkx": ["networkx"],
            "jinja2": ["jinja2"], "pydantic": ["pydantic"]}
    targets = subs.get(mod, [mod])
    return ("import json, subprocess, sys\nout = []\n" + "for pkg in " + json.dumps(targets) + ":\n"
            "    p = subprocess.run([sys.executable, '-I', 'solution.py', pkg], capture_output=True, text=True, timeout=120)\n"
            "    out.extend(json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else [])\n"
            "open('artifact.json', 'w').write(json.dumps(out))\n")


def parse_inventory(artifact: str) -> list[dict]:
    try:
        data = json.loads(artifact) if artifact else []
    except json.JSONDecodeError:
        return []
    return [e for e in data if isinstance(e, dict) and "name" in e]


class ApiReferenceRetriever:
    def __init__(self, inventory: Sequence[dict], top_k: int = 6):
        self.entries = list(inventory)
        self.by_short: dict[str, list[dict]] = {}
        for e in self.entries:
            self.by_short.setdefault(e["name"].split(".")[-1].lower(), []).append(e)
        self.top_k = top_k

    def retrieve(self, request_code: str, error_headline: str) -> list[dict]:
        idents = {m.group(0) for m in _IDENT.finditer(error_headline)}
        quoted = set(re.findall(r"'([A-Za-z_][A-Za-z0-9_.]*)'", error_headline))
        idents |= {q.split(".")[-1] for q in quoted}
        code_idents = {m.group(0) for m in _IDENT.finditer(request_code)}
        hits: list[tuple[int, dict]] = []
        seen: set[str] = set()
        for name in sorted(idents, key=len, reverse=True):
            for e in self.by_short.get(name.lower(), []):
                if e["name"] not in seen:
                    seen.add(e["name"]); hits.append((3, e))
            for close in difflib.get_close_matches(name.lower(), list(self.by_short), n=3, cutoff=0.6):
                for e in self.by_short.get(close, []):
                    if e["name"] not in seen:
                        seen.add(e["name"]); hits.append((2, e))
        for name in sorted(code_idents):
            for e in self.by_short.get(name.lower(), []):
                if e["name"] not in seen and len(hits) < 3 * self.top_k:
                    seen.add(e["name"]); hits.append((1, e))
        hits.sort(key=lambda h: (-h[0], h[1]["name"]))
        return [h[1] for h in hits[: self.top_k]]

    @staticmethod
    def format(entries: Sequence[dict]) -> str:
        if not entries:
            return "(no matching entries in the installed API inventory)"
        return "\n".join(f"- {e['name']}{e['signature']}: {e['doc']}".rstrip(": ") for e in entries)
