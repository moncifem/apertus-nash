"""Apertus 1.5 policy: HF Transformers (Swiss fork) + PEFT LoRA in BF16. No Unsloth.

- LoRA targets are discovered from named_modules() and restricted to the language backbone.
- Selected-token log-probs are computed in chunks against the real output head (never a full
  [seq, vocab] tensor for the whole sequence); the path is calibrated against a reference full
  forward at startup.
- Merged weights (W + scale*B@A) are computed out-of-place for pushing into vLLM: no
  merge/unmerge drift in the trainable model.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

log = logging.getLogger("rlaif.policy")


def load_tokenizer(model_id: str, revision: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, revision=revision)
    except Exception as e:  # noqa: BLE001
        log.warning("AutoTokenizer failed (%s); trying AutoProcessor.tokenizer", e)
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model_id, revision=revision).tokenizer


def _hf_config(model_id: str, revision: str):
    """Config from Transformers' own Apertus 1.5 class.

    vLLM registers its own `Apertus1p5Config` under the same model_type, which makes AutoConfig
    return a class without `text_config` and breaks every Auto* model class.
    """
    try:
        from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5Config
    except ImportError:
        return None
    return Apertus1p5Config.from_pretrained(model_id, revision=revision)


def _load_base(mc, revision: str):
    import transformers

    errors = []
    cfg = _hf_config(mc.id, revision)
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            kw = {"config": cfg} if cfg is not None else {}
            attn = mc.attn_implementation
            if cfg is not None and getattr(cfg, "sub_configs", None):
                # Only the text backbone is used; non-text towers (vision/audio tokenizers) don't
                # implement SDPA, so they get eager attention.
                attn = {"": "eager", **{k: ("eager" if k != "text_config" else mc.attn_implementation)
                                        for k in cfg.sub_configs}}
            m = cls.from_pretrained(
                mc.id, revision=revision, dtype=torch.bfloat16,
                attn_implementation=attn, device_map={"": 0}, **kw,
            )
            log.info("loaded %s via %s (%s)", mc.id, cls_name, type(m).__name__)
            return m
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cls_name}: {e!r}")
    raise RuntimeError("could not load policy with the pinned Transformers:\n" + "\n".join(errors))


def select_lora_targets(model: nn.Module, mc) -> list[str]:
    names = []
    excl = [x.lower() for x in mc.lora_exclude_substrings]
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in mc.lora_target_suffixes:
            continue
        if any(x in name.lower() for x in excl):
            continue
        names.append(name)
    if not names:
        raise RuntimeError("no LoRA target modules found; inspect artifacts/module_tree.txt")
    return names


def build_policy(mc, revision: str):
    from peft import LoraConfig, get_peft_model

    base = _load_base(mc, revision)
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)
    targets = select_lora_targets(base, mc)
    lcfg = LoraConfig(r=mc.lora_rank, lora_alpha=mc.lora_alpha, lora_dropout=mc.lora_dropout,
                      target_modules=targets, bias="none")
    model = get_peft_model(base, lcfg)
    if mc.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("LoRA on %d modules, %.1fM trainable params", len(targets), n_train / 1e6)
    return model, targets


def module_tree(model: nn.Module) -> str:
    return "\n".join(f"{n}\t{type(m).__name__}" for n, m in model.named_modules())


def _chunk_logp(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, tgt: torch.Tensor) -> torch.Tensor:
    logits = F.linear(h, w, b).float()
    return logits.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


class LogprobComputer:
    def __init__(self, peft_model, chunk: int):
        self.peft = peft_model
        self.base = peft_model.get_base_model()
        self.head = self.base.get_output_embeddings()
        self.chunk = chunk
        self.backbone = None
        self.backbone_name = None

    @property
    def vocab_rows(self) -> int:
        return self.head.weight.shape[0]

    def _candidates(self):
        seen, out = set(), []

        def add(name, m):
            if m is not None and id(m) not in seen and m is not self.base:
                seen.add(id(m))
                out.append((name, m))

        try:
            add("get_decoder()", self.base.get_decoder())
        except Exception:  # noqa: BLE001
            pass
        inner = getattr(self.base, "model", None)
        if inner is not None:
            add("model.language_model", getattr(inner, "language_model", None))
            add("model", inner)
        add("language_model", getattr(self.base, "language_model", None))
        return out

    def _hidden(self, bb, input_ids):
        out = bb(input_ids=input_ids, use_cache=False)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def _select(self, h: torch.Tensor, input_ids: torch.Tensor, n_prompt: int, use_ckpt: bool) -> torch.Tensor:
        h = h[0, n_prompt - 1: -1]
        tgt = input_ids[0, n_prompt:]
        w, b = self.head.weight, getattr(self.head, "bias", None)
        pieces = []
        for s in range(0, h.size(0), self.chunk):
            hs, ts = h[s:s + self.chunk], tgt[s:s + self.chunk]
            if use_ckpt and torch.is_grad_enabled():
                pieces.append(checkpoint(_chunk_logp, hs, w, b, ts, use_reentrant=False))
            else:
                pieces.append(_chunk_logp(hs, w, b, ts))
        return torch.cat(pieces)

    @torch.no_grad()
    def calibrate(self, input_ids: torch.Tensor, n_prompt: int, tol: float = 2e-2) -> dict:
        """Find the backbone whose hidden states + output head reproduce the full forward."""
        ref_logits = self.base(input_ids=input_ids, use_cache=False).logits.float()
        report = {"reference_vocab": int(ref_logits.shape[-1]), "head_rows": self.vocab_rows, "tried": []}
        tgt = input_ids[0, n_prompt:]
        ref = ref_logits[0, n_prompt - 1:-1].log_softmax(-1).gather(-1, tgt[:, None]).squeeze(-1)
        del ref_logits
        for name, bb in self._candidates():
            try:
                got = self._select(self._hidden(bb, input_ids), input_ids, n_prompt, use_ckpt=False)
                err = (got - ref).abs().max().item()
            except Exception as e:  # noqa: BLE001
                report["tried"].append({"backbone": name, "error": repr(e)[:300]})
                continue
            report["tried"].append({"backbone": name, "max_abs_err": err})
            if err < tol:
                self.backbone, self.backbone_name = bb, name
                report.update(selected=name, max_abs_err=err, tolerance=tol, passed=True)
                return report
        report["passed"] = False
        raise RuntimeError(f"chunked log-prob path does not match the reference forward: {report}")

    def logprobs(self, input_ids: torch.Tensor, n_prompt: int) -> torch.Tensor:
        """Log p of tokens input_ids[n_prompt:], shape [L]. Differentiable w.r.t. LoRA."""
        assert self.backbone is not None, "call calibrate() first"
        return self._select(self._hidden(self.backbone, input_ids), input_ids, n_prompt, use_ckpt=True)


@torch.no_grad()
def iter_merged_weights(peft_model, rewrites: list[tuple[str, str]], adapter: str = "default") -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (hf_checkpoint_name, W + delta) for every LoRA-adapted Linear."""
    from peft.tuners.lora import LoraLayer

    pats = [(re.compile(a), b) for a, b in rewrites]
    for name, mod in peft_model.named_modules():
        if not isinstance(mod, LoraLayer) or not hasattr(mod, "base_layer"):
            continue
        if adapter not in getattr(mod, "lora_A", {}):
            continue
        w = mod.base_layer.weight
        merged = (w.float() + mod.get_delta_weight(adapter).float()).to(w.dtype)
        hf = name[len("base_model.model."):] if name.startswith("base_model.model.") else name
        hf += ".weight"
        for p, r in pats:
            hf = p.sub(r, hf)
        yield hf, merged
