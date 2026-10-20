"""In-process vLLM rollout engine for the policy (Swiss vLLM fork), with direct weight push.

This replaces Unsloth's fast-inference path: vLLM runs in the trainer process
(VLLM_ENABLE_V1_MULTIPROCESSING=0) and after every optimizer step the merged LoRA weights are
loaded into the vLLM model with model.load_weights(), the same mechanism TRL's colocate mode
uses. Works for any architecture the pinned vLLM supports, without needing vLLM LoRA support.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

log = logging.getLogger("rlaif.rollout")


@dataclass
class Sample:
    completion_ids: list[int]
    finish_reason: str


class PolicyEngine:
    def __init__(self, mc, rc, revision: str, max_model_len: int, seed: int):
        from vllm import LLM

        self.rc = rc
        self.llm = LLM(
            model=mc.id, revision=revision, tokenizer_revision=revision, dtype="bfloat16",
            gpu_memory_utilization=rc.gpu_memory_utilization, max_model_len=max_model_len,
            max_num_seqs=rc.max_num_seqs, enable_prefix_caching=rc.enable_prefix_caching,
            enforce_eager=rc.enforce_eager, seed=seed, **rc.vllm_extra_kwargs,
        )

    def generate(self, prompts: list[list[int]], *, n: int, temperature: float, top_p: float,
                 top_k: int, max_tokens: int) -> list[list[Sample]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k,
                            max_tokens=max_tokens, skip_special_tokens=False)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        return [[Sample(list(o.token_ids), o.finish_reason or "stop") for o in r.outputs] for r in outs]

    def sequence_logprobs(self, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        """Raw (un-tempered) log p of completion tokens under the vLLM weights, via prompt_logprobs."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        full = prompt_ids + completion_ids
        sp = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, skip_special_tokens=False)
        out = self.llm.generate([TokensPrompt(prompt_token_ids=full)], sp, use_tqdm=False)[0]
        pl = out.prompt_logprobs
        return [pl[p][full[p]].logprob for p in range(len(prompt_ids), len(full))]

    # ------------------------------------------------------------------ weight sync
    def _load(self, batch: list) -> None:
        def fn(model):
            model.load_weights(batch)

        if hasattr(self.llm, "apply_model"):
            self.llm.apply_model(fn)
            return
        engine = self.llm.llm_engine
        for path in ("model_executor.driver_worker.worker.model_runner.model",
                     "model_executor.driver_worker.model_runner.model"):
            obj = engine
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                obj.load_weights(batch)
                return
        raise RuntimeError("cannot reach the in-process vLLM model for weight sync "
                           "(is VLLM_ENABLE_V1_MULTIPROCESSING=0?)")

    def push_weights(self, weights: Iterable) -> int:
        batch, size, count = [], 0, 0
        for name, t in weights:
            batch.append((name, t))
            size += t.numel() * t.element_size()
            count += 1
            if size >= self.rc.weight_push_chunk_bytes:
                self._load(batch)
                batch, size = [], 0
        if batch:
            self._load(batch)
        self.llm.reset_prefix_cache()  # cached KV was computed with the old weights
        return count
