"""Frozen causal-LM adapter (HF transformers). No weights are ever updated.

* ``generate_candidates``: candidate 0 is greedy; candidates 1..K-1 are sampled
  with a fixed, task-derived seed (reproducible candidate diversity).
* ``score_candidates``: **length-normalised** mean token log-likelihood of the
  generated response tokens under the frozen model (teacher-forced forward
  pass). This is the base reranking score shared by every method.
* Deterministic metadata (model id, revision, dtype, generation config hash).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Sequence

SYSTEM_PROMPT = ("You are an expert Python engineer. You repair Python modules so that they run "
                 "correctly on the exact library version specified by the user.")

PROMPT_WITH_ERROR = """The following Python module must run correctly with {package}=={version} on Python {python}.
It was written for an older version. Running it with {package}=={version} produced this error:

```
{error}
```

Rewrite the module so that it works with {package}=={version}. Preserve the module's behaviour, function names and signatures. Do not add new dependencies. Return only the complete corrected module in a single ```python code block.

```python
{code}
```"""

PROMPT_NO_ERROR = """The following Python module must run correctly with {package}=={version} on Python {python}.
Check it for API usage that is incompatible with {package}=={version}. If it is already compatible, return it unchanged; otherwise fix only what is necessary. Preserve the module's behaviour, function names and signatures. Do not add new dependencies. Return only the complete module in a single ```python code block.

```python
{code}
```"""

RAG_CONTEXT = """Reference for the installed {package}=={version} (official API inventory, retrieved automatically):
{context}

"""

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
ASSISTANT_PREFILL = "```python\n"


def extract_code(response: str) -> tuple[str, bool]:
    """Response text follows an assistant prefill of a code fence: code runs until the closing fence."""
    if not response.lstrip().startswith("```"):
        end = response.find("```")
        if end >= 0:
            return response[:end].rstrip() + "\n", True
        return response.rstrip() + "\n", False   # hit max_new_tokens before the closing fence
    m = _CODE_BLOCK.search(response)
    if m:
        return m.group(1).rstrip() + "\n", True
    # unterminated block (hit max_new_tokens) -> take everything after the fence
    idx = response.find("```")
    if idx >= 0:
        body = response[idx:].split("\n", 1)
        if len(body) == 2:
            return body[1].rstrip() + "\n", False
    return response.strip() + "\n", False


def build_user_prompt(package: str, version: str, python: str, code: str, error: str | None,
                      rag_context: str | None = None) -> str:
    err = (error or "").strip()
    if err:
        body = PROMPT_WITH_ERROR.format(package=package, version=version, python=python, error=err, code=code.rstrip())
    else:
        body = PROMPT_NO_ERROR.format(package=package, version=version, python=python, code=code.rstrip())
    if rag_context:
        body = RAG_CONTEXT.format(package=package, version=version, context=rag_context) + body
    return body


def _seed_for(task_id: str, base_seed: int) -> int:
    return (int(hashlib.sha256(f"{base_seed}:{task_id}".encode()).hexdigest(), 16) % (2**31 - 1))


@dataclass(frozen=True)
class Generation:
    text: str                 # raw assistant response
    code: str                 # extracted code
    extracted_ok: bool
    prompt_tokens: int
    completion_tokens: int
    mode: str                 # greedy | sample
    finish: str               # eos | length
    gen_ms: float             # amortised wall-clock per candidate (batch time / batch size)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationConfig:
    num_candidates: int = 2
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 8
    seed: int = 42

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


class FrozenCausalLM:
    def __init__(self, model_id: str, revision: str, dtype: str = "float16", max_input_tokens: int = 2048,
                 device: str | None = None, local_files_only: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.dtype_name = dtype
        self.max_input_tokens = int(max_input_tokens)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch_dtype,
                                                          local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)   # frozen: no gradient, no update
        self._torch = torch

    # ------------------------------------------------------------------ metadata
    def metadata(self) -> dict:
        t = self._torch
        info = {"model_id": self.model_id, "revision": self.revision, "dtype": self.dtype_name,
                "device": self.device, "max_input_tokens": self.max_input_tokens,
                "torch": t.__version__, "cuda": t.version.cuda,
                "n_parameters": int(sum(p.numel() for p in self.model.parameters()))}
        if self.device.startswith("cuda"):
            info["gpu_name"] = t.cuda.get_device_name(0)
        return info

    def peak_vram_bytes(self) -> int:
        if self.device.startswith("cuda"):
            return int(self._torch.cuda.max_memory_reserved())
        return 0

    def chat_prompt(self, user_prompt: str) -> str:
        """Chat template + assistant prefill of a code fence (code-only responses, shared by all methods)."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + ASSISTANT_PREFILL

    # ------------------------------------------------------------------ generation
    def _generate_batch(self, prompts: Sequence[str], *, do_sample: bool, n: int, seeds: Sequence[int],
                        cfg: GenerationConfig) -> list[list[Generation]]:
        torch = self._torch
        enc = self.tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_input_tokens).to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        kwargs = dict(max_new_tokens=cfg.max_new_tokens, do_sample=do_sample, num_return_sequences=n,
                      pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        if do_sample:
            kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
            torch.manual_seed(int(seeds[0]))
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(int(seeds[0]))
        else:
            kwargs.update(temperature=None, top_p=None, top_k=None)
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_cand_ms = elapsed_ms / max(1, out.shape[0])
        gens: list[list[Generation]] = []
        eos_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        ptoks = enc["attention_mask"].sum(dim=1).tolist()
        for i in range(len(prompts)):
            row: list[Generation] = []
            for j in range(n):
                seq = out[i * n + j, prompt_len:]
                ids = seq.tolist()
                finish = "length"
                cut = len(ids)
                for k, tok in enumerate(ids):
                    if tok in eos_ids:
                        cut = k
                        finish = "eos"
                        break
                ids = ids[:cut]
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                code, ok = extract_code(text)
                row.append(Generation(text, code, ok, int(ptoks[i]), len(ids), "sample" if do_sample else "greedy",
                                      finish, per_cand_ms))
            gens.append(row)
        return gens

    def generate_candidates(self, task_ids: Sequence[str], user_prompts: Sequence[str], cfg: GenerationConfig,
                            progress=None) -> list[list[Generation]]:
        """Returns, per task, K candidates: [greedy, sample_1, ..., sample_{K-1}]."""
        prompts = [self.chat_prompt(p) for p in user_prompts]
        results: list[list[Generation]] = [[] for _ in prompts]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # length-sorted batching
        bs = max(1, cfg.batch_size)
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            batch = [prompts[i] for i in idx]
            greedy = self._generate_batch(batch, do_sample=False, n=1, seeds=[0], cfg=cfg)
            for i, g in zip(idx, greedy):
                results[i].extend(g)
            if cfg.num_candidates > 1:
                seeds = [_seed_for(task_ids[idx[0]], cfg.seed)]
                sampled = self._generate_batch(batch, do_sample=True, n=cfg.num_candidates - 1, seeds=seeds, cfg=cfg)
                for i, g in zip(idx, sampled):
                    results[i].extend(g)
            if progress:
                progress(min(len(order), start + bs), len(order))
        return results

    # ------------------------------------------------------------------ scoring
    def score_candidates(self, user_prompts: Sequence[str], responses: Sequence[str], batch_size: int = 8) -> list[float]:
        """Mean per-token log-likelihood of each response given its prompt."""
        torch = self._torch
        prompts = [self.chat_prompt(p) for p in user_prompts]
        scores: list[float] = [float("nan")] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]) + len(responses[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            ids_list, resp_lens = [], []
            for i in idx:
                p_ids = self.tokenizer(prompts[i], add_special_tokens=False)["input_ids"][-self.max_input_tokens:]
                r_ids = self.tokenizer(responses[i], add_special_tokens=False)["input_ids"][:512]
                if self.tokenizer.eos_token_id is not None:
                    r_ids = r_ids + [self.tokenizer.eos_token_id]
                ids_list.append(p_ids + r_ids)
                resp_lens.append(len(r_ids))
            max_len = max(len(x) for x in ids_list)
            pad = self.tokenizer.pad_token_id
            input_ids = torch.full((len(idx), max_len), pad, dtype=torch.long)
            attn = torch.zeros((len(idx), max_len), dtype=torch.long)
            for r, ids in enumerate(ids_list):   # right padding for scoring
                input_ids[r, :len(ids)] = torch.tensor(ids)
                attn[r, :len(ids)] = 1
            input_ids, attn = input_ids.to(self.device), attn.to(self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
                for r, i in enumerate(idx):
                    total = len(ids_list[r])
                    n_resp = resp_lens[r]
                    if n_resp == 0:
                        scores[i] = float("nan")
                        continue
                    # positions predicting the response tokens: logits[t] predicts token t+1
                    span = logits[r, total - 1 - n_resp: total - 1].float()
                    target = input_ids[r, total - n_resp: total]
                    lp = torch.log_softmax(span, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    scores[i] = float(lp.mean().item())
                del logits
        return scores
