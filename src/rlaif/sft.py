"""SFT stage: LoRA fine-tuning of Apertus on the ordered curriculum, then hand the adapter to RL.

    python -m rlaif.sft --config configs/sft.yaml --dry-run     # tokenise, stats, RL-parser check; no model
    python -m rlaif.sft --config configs/sft.yaml               # train (needs the GPU to itself)

Design (everything below mirrors the RL trainer so the adapter drops straight into it):
- Same model loader, LoRA targets and chunked log-prob path as RL (`rlaif.policy`): BF16 weights, gradient
  checkpointing, no [seq, vocab] tensor. The log-prob path is calibrated against a full forward first.
- Loss is token-level negative log-likelihood on the completion only (prompt tokens are never trained on),
  normalised by the number of completion tokens in the optimizer batch.
- Curriculum order is kept: examples are visited in `order`; every epoch repeats the same easy-to-hard
  pass (optionally shuffled inside small windows with `shuffle_window`).
- The adapter is saved in the RL format (`adapter_model.safetensors`); RL starts from it with
  `python -m rlaif.train --config ... --init-adapter /runs/<run>/best/adapter`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import yaml
from pydantic import Field

from .config import ModelCfg, Strict
from .sft_data import build_dataset, check_with_rl_parser, dataset_stats, load_rows, load_variants

log = logging.getLogger("rlaif.sft")


# ============================================================================ config
class SFTData(Strict):
    path: str = "/app/data/nash/sft.jsonl"
    train_splits: list[str] = ["train"]
    val_splits: list[str] = ["val"]
    prompt_path: str = "prompts/user_message.txt"   # default wrapper (rows without a prompt_variant)
    # concise / standard / detailed paraphrases; rows pick one by `prompt_variant` (raw rows get the bare question)
    variants_path: str | None = None
    # replaces the chat template's built-in system message ("Apertus 1.5 Omni ..."); None keeps the default.
    # RL must use the same text (data.policy_system_prompt) or the identity learned here does not transfer.
    system_prompt: str | None = None
    end_token: str = "<|assistant_end|>"
    max_seq_tokens: int = 4096
    limit: int = 0
    cache_tokenized: bool = True           # pickle the tokenised examples next to the data file; reused when nothing changed
    val_limit: int = 0                     # 0 = all validation rows; set it when examples are long (evals get slow)


class SFTTrain(Strict):
    seed: int = 42
    epochs: int = 3
    batch_examples: int = 16
    learning_rate: float = 1.0e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    shuffle_window: int = 0  # 0 = strict curriculum order; k > 1 shuffles inside consecutive windows of k
    eval_every_steps: int = 10
    checkpoint_every_steps: int = 20
    keep_checkpoints: int = 2
    output_dir: str = "/runs"
    run_name: str = "apertus-sft-v1"


class SFTConfig(Strict):
    schema_version: int = 1
    model: ModelCfg = Field(default_factory=ModelCfg)
    data: SFTData = Field(default_factory=SFTData)
    train: SFTTrain = Field(default_factory=SFTTrain)


def load_sft_config(path: str) -> SFTConfig:
    return SFTConfig(**(yaml.safe_load(Path(path).read_text()) or {}))


# ============================================================================ pure helpers (unit-tested)
def lr_at(step: int, total: int, warmup: int, base: float, min_ratio: float) -> float:
    """Linear warmup, then cosine decay to base * min_ratio."""
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))


def make_batches(n: int, batch_examples: int, epochs: int, shuffle_window: int, seed: int) -> list[tuple[int, list[int]]]:
    """One entry per optimizer step: (epoch, indices into the ordered example list)."""
    out: list[tuple[int, list[int]]] = []
    for ep in range(epochs):
        idx = list(range(n))
        if shuffle_window and shuffle_window > 1:
            rng = random.Random(seed * 1000 + ep)
            for s in range(0, n, shuffle_window):
                w = idx[s:s + shuffle_window]
                rng.shuffle(w)
                idx[s:s + shuffle_window] = w
        for s in range(0, n, batch_examples):
            out.append((ep, idx[s:s + batch_examples]))
    return out


# ============================================================================ plumbing
def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):  # idempotent: one set of handlers per run
        root.removeHandler(h)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log")):
        h.setFormatter(fmt)
        root.addHandler(h)


def resolve_sha(repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, revision=revision).sha


def load_adapter(policy, adapter_dir: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    res = set_peft_model_state_dict(policy, load_file(str(adapter_dir / "adapter_model.safetensors")))
    unexpected = getattr(res, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"adapter has unexpected keys: {unexpected[:5]}")


def save_adapter(policy, path: Path, info: dict | None = None) -> None:
    tmp = path.parent / f".tmp-{path.name}"
    shutil.rmtree(tmp, ignore_errors=True)
    policy.save_pretrained(str(tmp / "adapter"))
    if info is not None:
        (tmp / "info.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


_STOP = False


def _on_signal(signum, _frame):
    global _STOP
    _STOP = True
    log.warning("signal %s received: will checkpoint and stop after the current step", signum)


# ============================================================================ data
def prepare(cfg: SFTConfig, tok, output_dir: Path):
    d = cfg.data
    import pickle
    template = Path(d.prompt_path).read_text()
    end_id = tok.convert_tokens_to_ids(d.end_token)
    if end_id is None or end_id == getattr(tok, "unk_token_id", -1):
        raise SystemExit(f"end token {d.end_token!r} is not in the tokenizer")
    variants = load_variants(d.variants_path) if d.variants_path else None
    kw = dict(thinking=cfg.model.thinking, think_open=cfg.model.think_open, think_close=cfg.model.think_close,
              end_token=d.end_token, end_id=end_id, max_tokens=d.max_seq_tokens,
              variants=variants, system=d.system_prompt)
    # output head rows = base vocabulary; special ids above it (<|image|> ...) can be embedded but never predicted
    kw["max_target_id"] = int(getattr(tok, "vocab_size", 0)) or None
    cache = None
    if d.cache_tokenized:
        import hashlib
        import pickle
        h = hashlib.sha256()
        h.update(Path(d.path).read_bytes())
        h.update(json.dumps([template, d.train_splits, d.val_splits, d.limit, d.val_limit, d.system_prompt, d.variants_path,
                             {k: v for k, v in kw.items() if k != "variants"}, len(tok), "v2"], default=str).encode())
        cache = output_dir.parent / "_tokcache" / f"{Path(d.path).stem}-{h.hexdigest()[:16]}.pkl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            train, val, report = pickle.loads(cache.read_bytes())
            log.info("tokenised data loaded from cache %s", cache)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "sft_data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
            return train, val, report
    train_rows = load_rows(d.path, d.train_splits, d.limit)
    val_rows = load_rows(d.path, d.val_splits, d.val_limit) if d.val_splits else []
    train, skip_t = build_dataset(tok, train_rows, template, **kw)
    val, skip_v = build_dataset(tok, val_rows, template, **kw)
    report = {
        "rows_train": len(train_rows), "rows_val": len(val_rows),
        "examples_train": len(train), "examples_val": len(val),
        "skipped": [{"id": i, "reason": r} for i, r in skip_t + skip_v],
        "train_stats": dataset_stats(train), "val_stats": dataset_stats(val),
        "rl_parser_check_train": check_with_rl_parser(tok, train, cfg.model.think_open, cfg.model.think_close),
        "sample_completion": train[0].completion_text if train else None,
    }
    (output_dir).mkdir(parents=True, exist_ok=True)
    (output_dir / "sft_data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if cache is not None:
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps((train, val, report)))
        os.replace(tmp, cache)
        log.info("tokenised data cached at %s", cache)
    return train, val, report


# ============================================================================ training
def val_loss(policy, lp, examples, device) -> dict:
    import torch

    policy.eval()
    tot, n = 0.0, 0
    by_level, by_domain, by_style = {}, {}, {}
    with torch.no_grad():
        for e in examples:
            ids = torch.tensor([e.prompt_ids + e.completion_ids], device=device)
            nll = -lp.logprobs(ids, len(e.prompt_ids)).sum().item()
            tot += nll
            n += len(e.completion_ids)
            for bucket, key in ((by_level, str(e.level)), (by_domain, str(e.domain)), (by_style, e.answer_style)):
                a = bucket.setdefault(key, [0.0, 0])
                a[0] += nll
                a[1] += len(e.completion_ids)
    policy.train()
    fmt = lambda b: {k: round(v[0] / max(1, v[1]), 4) for k, v in sorted(b.items())}
    # by_domain shows whether identity / trap rows are learned next to math; by_style separates plain from boxed answers
    return {"val_nll": tot / max(1, n), "n": len(examples), "tokens": n,
            "by_level": fmt(by_level), "by_domain": fmt(by_domain), "by_style": fmt(by_style)}


def run(cfg: SFTConfig, args) -> None:
    import torch

    tcfg = cfg.train
    if args.output_dir:
        tcfg.output_dir = args.output_dir
    run_dir = Path(tcfg.output_dir) / tcfg.run_name
    setup_logging(run_dir)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    (run_dir / "resolved-config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))

    from .policy import LogprobComputer, build_policy, load_tokenizer, module_tree

    model_sha = resolve_sha(cfg.model.id, cfg.model.revision)
    tok = load_tokenizer(cfg.model.id, model_sha)
    train, val, report = prepare(cfg, tok, run_dir)
    log.info("model %s@%s | train %d val %d examples (skipped %d) | %s tokens trainable",
             cfg.model.id, model_sha, len(train), len(val), len(report["skipped"]),
             report["train_stats"].get("trainable_tokens"))
    log.info("RL-parser check on targets: %s", {k: v for k, v in report["rl_parser_check_train"].items()
                                                if k in ("n", "hard_valid", "format_valid", "steps_ok")})
    if not train:
        raise SystemExit("no trainable examples")
    if args.dry_run:
        log.info("dry run complete; see %s", run_dir / "sft_data_report.json")
        return
    if report["rl_parser_check_train"]["hard_valid"] < report["rl_parser_check_train"]["n"]:
        raise SystemExit("some SFT targets are not hard-valid under the RL parser; fix the data first "
                         f"({report['rl_parser_check_train']['first_hard_failures']})")

    policy, targets = build_policy(cfg.model, model_sha)
    device = next(policy.parameters()).device
    (run_dir / "module_tree.txt").write_text(module_tree(policy.get_base_model()))
    if args.init_adapter:
        load_adapter(policy, Path(args.init_adapter))
        log.info("initialised from adapter %s", args.init_adapter)
    lp = LogprobComputer(policy, cfg.model.logprob_chunk_tokens)
    s0 = train[0]
    policy.eval()
    calib = lp.calibrate(torch.tensor([s0.prompt_ids + s0.completion_ids], device=device), len(s0.prompt_ids))
    policy.train()
    (run_dir / "compatibility.json").write_text(json.dumps({
        "model": cfg.model.id, "model_sha": model_sha, "lora_targets": len(targets),
        "logprob_calibration": calib, "init_adapter": args.init_adapter}, indent=2))

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay, betas=(0.9, 0.99), eps=1e-8)
    batches = make_batches(len(train), tcfg.batch_examples, tcfg.epochs, tcfg.shuffle_window, tcfg.seed)
    total = len(batches)
    # read by the live dashboard (/sft): totals and the settings that decide which checkpoints exist
    (run_dir / "plan.json").write_text(json.dumps({
        "run": tcfg.run_name, "total_steps": total, "epochs": tcfg.epochs, "batch_examples": tcfg.batch_examples,
        "steps_per_epoch": -(-len(train) // tcfg.batch_examples), "examples_train": len(train), "examples_val": len(val),
        "eval_every_steps": tcfg.eval_every_steps, "checkpoint_every_steps": tcfg.checkpoint_every_steps,
        "keep_checkpoints": tcfg.keep_checkpoints, "learning_rate": tcfg.learning_rate, "warmup_steps": tcfg.warmup_steps,
        "init_adapter": args.init_adapter, "model": cfg.model.id, "system_prompt": cfg.data.system_prompt,
        "by_domain_train": report["train_stats"].get("by_domain"), "by_variant_train": report["train_stats"].get("by_prompt_variant"),
        "started": time.time()}, indent=2))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_at(s, total, tcfg.warmup_steps, tcfg.learning_rate, tcfg.min_lr_ratio) / tcfg.learning_rate)

    start, best = 0, {"val_nll": float("inf")}
    last = run_dir / "checkpoints" / "LAST"
    if last.exists() and not args.fresh:
        ck = run_dir / "checkpoints" / last.read_text().strip()
        st = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
        load_adapter(policy, ck / "adapter")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best = st["step"], st["best"]
        log.info("resumed from %s at step %d/%d", ck, start, total)

    metrics = open(run_dir / "metrics.jsonl", "a")

    def record(row: dict) -> None:
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()

    last_eval = -1

    def do_eval(step: int) -> None:
        nonlocal best, last_eval
        if not val or step == last_eval:
            return
        last_eval = step
        res = val_loss(policy, lp, val, device)
        res["update"] = step
        record({"eval": res})
        log.info("eval @%d: val_nll=%.4f by_level=%s", step, res["val_nll"], res["by_level"])
        if res["val_nll"] < best["val_nll"]:
            best = {"val_nll": res["val_nll"], "step": step}
            save_adapter(policy, run_dir / "best", best)

    if start == 0:
        do_eval(0)  # the starting model's loss on held-out rows
    log.info("training: %d optimizer steps (%d epochs x %d examples, batch %d), lr %.1e",
             total, tcfg.epochs, len(train), tcfg.batch_examples, tcfg.learning_rate)

    policy.train()
    for step in range(start, total):
        ep, idxs = batches[step]
        t0 = time.time()
        batch = [train[i] for i in idxs]
        n_tok = sum(len(e.completion_ids) for e in batch)
        nll_sum = 0.0
        for e in batch:
            ids = torch.tensor([e.prompt_ids + e.completion_ids], device=device)
            nll = -lp.logprobs(ids, len(e.prompt_ids)).sum()
            (nll / n_tok).backward()
            nll_sum += nll.item()
        gn = torch.nn.utils.clip_grad_norm_(params, tcfg.max_grad_norm).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        # field names match the RL dashboard (update, loss_out, grad_norm, lr)
        record({"update": step + 1, "epoch": ep + 1, "B": len(batch), "tokens": n_tok, "loss_out": nll_sum / n_tok,
                "grad_norm": gn, "lr": lr_now, "time_total": time.time() - t0})
        log.info("step %d/%d ep %d nll=%.4f gn=%.3f lr=%.2e %.0fs", step + 1, total, ep + 1, nll_sum / n_tok, gn,
                 lr_now, time.time() - t0)
        if tcfg.eval_every_steps and (step + 1) % tcfg.eval_every_steps == 0:
            do_eval(step + 1)
        stop = _STOP
        if stop or (tcfg.checkpoint_every_steps and (step + 1) % tcfg.checkpoint_every_steps == 0):
            ckd = run_dir / "checkpoints"
            ckd.mkdir(parents=True, exist_ok=True)
            tmp = ckd / f".tmp-step-{step + 1:06d}"
            shutil.rmtree(tmp, ignore_errors=True)
            policy.save_pretrained(str(tmp / "adapter"))
            torch.save({"step": step + 1, "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best}, tmp / "state.pt")
            final = ckd / f"step-{step + 1:06d}"
            shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
            (ckd / "LAST").write_text(final.name)
            for old in sorted(p for p in ckd.glob("step-*") if p.is_dir())[:-tcfg.keep_checkpoints]:
                shutil.rmtree(old, ignore_errors=True)
        if stop:
            log.warning("stopped early at step %d; rerun the same command to resume", step + 1)
            return

    do_eval(total)
    save_adapter(policy, run_dir / "final", {"step": total})
    if not (run_dir / "best").exists():
        save_adapter(policy, run_dir / "best", {"step": total, "note": "no validation split"})
    summary = {
        "steps": total, "epochs": tcfg.epochs, "examples": len(train), "best": best,
        "next": {
            "evaluate": f"./rlaif eval --adapter /runs/{tcfg.run_name}/best/adapter --split val",
            "start_rl": f"python3 -m rlaif.train --config <rl config> --init-adapter /runs/{tcfg.run_name}/best/adapter",
        },
    }
    (run_dir / "sft_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done: %s", summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="tokenise and report; do not load the model")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--init-adapter", default=None, help="start from this adapter directory")
    ap.add_argument("--output-dir", default=None, help="override train.output_dir")
    ap.add_argument("--data", default=None, help="override data.path")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training rows")
    args = ap.parse_args()
    cfg = load_sft_config(args.config)
    if args.data:
        cfg.data.path = args.data
    if args.limit is not None:
        cfg.data.limit = args.limit
    run(cfg, args)


if __name__ == "__main__":
    main()
