"""Step-aware Dr. GRPO training of Apertus 1.5 with a Qwen3-4B step judge. No Unsloth.

Loop per update (phases are serialized: generate -> verify -> judge -> backward -> sync):
  1. vLLM (in-process) samples G completions for each of P prompts with native thinking.
  2. Parser splits the thinking span into numbered steps (exact token spans) + one \\boxed answer.
  3. Math-Verify checks the final answer against the trusted gold.
  4. The judge scores each step, prefix-only (after the process warm-up).
  5. Outcome reward = correctness - format/truncation penalties + conciseness (length) reward,
     mean-centred per group (Dr. GRPO, no std scaling).
     Process reward = per-step q in {-1, 0, -0.5, +1} applied to that step's tokens only.
  6. Backward one trajectory at a time (exact global normalisation), AdamW step on LoRA.
  7. Merged weights pushed into vLLM; periodic trainer/vLLM log-prob parity check.

Run:  python -m rlaif.train --config configs/apertus_math.yaml [--fresh] [--eval-only ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import yaml

from .config import Config, load_config
from .data import Problem, prepare_data
from .deepseek_judge import DeepSeekScorer, split_completion
from .judge import JudgeUnavailable, StepJudge, verdict_to_q
from .loss import trajectory_loss
from .memory import MemoryMonitor
from .parsing import ParsedTrace, parse_completion
from .rewards import group_advantages, outcome_rewards, process_token_targets, ramp
from .reward_client import RewardClient, apply_rewards
from .reward_v2 import RewardV2, advantages as v2_advantages
from .verify import Verifier, is_operational_failure

log = logging.getLogger("rlaif")

EXIT_CODES = {
    "MODEL_ACCESS_DENIED": 10, "JUDGE_UNAVAILABLE": 11, "MEMORY_BUDGET_EXCEEDED": 12,
    "HOST_MEMORY_LOW": 13, "VERIFIER_FAILURE": 14, "POLICY_SYNC_FAILED": 15,
    "TRUNCATION_RATE_HIGH": 16, "COMPATIBILITY_FAILED": 17, "FINGERPRINT_MISMATCH": 18,
}


class Abort(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


STOP = {"flag": False}


def _on_signal(signum, _frame):
    log.warning("signal %s: finishing current update, then checkpoint + exit", signum)
    STOP["flag"] = True


# ============================================================================ data containers
@dataclass
class PromptItem:
    problem: Problem
    ids: list[int]


@dataclass
class Traj:
    item: PromptItem
    completion_ids: list[int]
    finish_reason: str
    parsed: ParsedTrace
    correct: bool = False
    verify_status: str = "no_answer"
    reward: float = 0.0
    length_term: float = 0.0
    reward_parts: dict | None = None
    advantage: float = 0.0
    step_q: list[float] | None = None
    verdicts: list | None = None


class Sampler:
    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch, self.pos = n, seed, 0, 0
        self.order = self._perm()

    def _perm(self):
        o = list(range(self.n))
        random.Random(self.seed + 1000 * self.epoch).shuffle(o)
        return o

    def take(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if self.pos >= self.n:
                self.epoch, self.pos = self.epoch + 1, 0
                self.order = self._perm()
            out.append(self.order[self.pos])
            self.pos += 1
        return out

    def state(self):
        return {"epoch": self.epoch, "pos": self.pos}

    def load(self, s):
        self.epoch, self.pos = s["epoch"], s["pos"]
        self.order = self._perm()


# ============================================================================ helpers
def render_prompt(tok, template: str, problem: str, thinking: bool, system: str | None = None) -> str:
    msgs = [{"role": "user", "content": template.replace("{problem}", problem)}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def build_items(tok, problems, template, thinking, max_prompt, system: str | None = None) -> list[PromptItem]:
    items = []
    for p in problems:
        text = render_prompt(tok, template, p.problem, thinking, system)
        ids = tok(text, add_special_tokens=False).input_ids  # template already has BOS
        if len(ids) <= max_prompt:
            items.append(PromptItem(p, ids))
    return items


def score_rollouts(items, samples, tok, cfg: Config, verifier, opened, verify: bool = True) -> list[list[Traj]]:
    groups, pairs, refs = [], [], []
    for it, ss in zip(items, samples):
        g = []
        for s in ss:
            parsed = parse_completion(
                s.completion_ids, tok, think_open=cfg.model.think_open, think_close=cfg.model.think_close,
                opened_in_prompt=opened, finish_reason=s.finish_reason, max_steps=cfg.reward.max_steps,
                answer_mode=cfg.data.answer_format,
                require_box=cfg.data.answer_format != "plain",
                reasoning_format=getattr(cfg.data, "reasoning_format", "numbered"))
            t = Traj(it, s.completion_ids, s.finish_reason, parsed)
            if verify and parsed.final_answer is not None and not parsed.truncated:
                pairs.append((it.problem.gold, parsed.final_answer))
                refs.append(t)
            g.append(t)
        groups.append(g)
    for t, (status, ok) in zip(refs, verifier.check(pairs)):
        t.verify_status, t.correct = status, ok
    return groups


def evaluate(engine, tok, items, cfg, verifier, opened, limit, baseline_per_id=None) -> dict:
    items = items[:limit]
    samples = engine.generate([it.ids for it in items], n=1, temperature=0.0, top_p=1.0, top_k=-1,
                              max_tokens=cfg.train.max_completion_tokens)
    trajs = [g[0] for g in score_rollouts(items, samples, tok, cfg, verifier, opened)]
    n = max(1, len(trajs))
    think_correct = [t.parsed.thinking_tokens for t in trajs if t.correct]
    per_id = {t.item.problem.id: [bool(t.correct), t.parsed.thinking_tokens] for t in trajs}
    res = {
        "n": len(trajs),
        "accuracy": sum(t.correct for t in trajs) / n,
        "format_valid": sum(t.parsed.format_valid for t in trajs) / n,
        "hard_valid": sum(t.parsed.hard_valid for t in trajs) / n,
        "steps_ok": sum(t.parsed.spans_ok for t in trajs) / n,
        "truncated": sum(t.parsed.truncated for t in trajs) / n,
        "operational_failures": sum(is_operational_failure(t.verify_status) for t in trajs),
        "completion_tokens_mean": statistics.fmean(len(t.completion_ids) for t in trajs) if trajs else 0,
        "thinking_tokens_correct_median": statistics.median(think_correct) if think_correct else None,
        "thinking_tokens_correct_mean": statistics.fmean(think_correct) if think_correct else None,
        "steps_mean": statistics.fmean(len(t.parsed.steps) for t in trajs) if trajs else 0,
    }
    if baseline_per_id:
        both = [(baseline_per_id[k][1], v[1]) for k, v in per_id.items()
                if k in baseline_per_id and baseline_per_id[k][0] and v[0]]
        if both:
            res["both_solved_n"] = len(both)
            res["both_solved_thinking_ratio"] = sum(b for _, b in both) / max(1, sum(a for a, _ in both))
    return res, per_id


# ============================================================================ checkpoints
def save_checkpoint(run_dir: Path, update: int, policy, opt, sched, sampler, state, keep: int) -> Path:
    ck = run_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    tmp = ck / f".tmp-step-{update:06d}"
    shutil.rmtree(tmp, ignore_errors=True)
    policy.save_pretrained(str(tmp / "adapter"))
    torch.save({
        "update": update, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "sampler": sampler.state(), "state": state,
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }, tmp / "trainer_state.pt")
    final = ck / f"step-{update:06d}"
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)  # atomic on the same filesystem
    (ck / "LAST").write_text(final.name)
    steps = sorted(p for p in ck.glob("step-*") if p.is_dir())
    for old in steps[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    log.info("checkpoint saved: %s", final)
    return final


def latest_checkpoint(run_dir: Path) -> Path | None:
    last = run_dir / "checkpoints" / "LAST"
    if last.exists():
        p = run_dir / "checkpoints" / last.read_text().strip()
        if (p / "trainer_state.pt").exists():
            return p
    return None


def load_adapter(policy, adapter_dir: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(policy, sd)
    unexpected = getattr(res, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(f"adapter has unexpected keys: {unexpected[:5]}")


def save_best(run_dir: Path, policy, info: dict) -> None:
    tmp, final = run_dir / ".tmp-best", run_dir / "best"
    shutil.rmtree(tmp, ignore_errors=True)
    policy.save_pretrained(str(tmp / "adapter"))
    (tmp / "info.json").write_text(json.dumps(info, indent=2))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)


# ============================================================================ main
def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log")):
        h.setFormatter(fmt)
        root.addHandler(h)


def resolve_sha(repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        return HfApi().model_info(repo_id, revision=revision).sha
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise Abort("MODEL_ACCESS_DENIED",
                    f"No access to {repo_id}. Accept its terms on huggingface.co and put a token with "
                    f"read access in secrets/hf_token. ({e.__class__.__name__})") from e


def versions() -> dict:
    out = {"python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for mod in ("transformers", "vllm", "peft", "math_verify", "datasets"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out[mod] = f"unavailable: {e!r}"
    try:
        import transformers

        out["transformers_path"] = transformers.__file__
    except Exception:  # noqa: BLE001
        pass
    lock = Path("/opt/rlaif/pip-freeze.txt")
    if lock.exists():
        out["pip_freeze_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    return out


def parity_check(engine, lp, policy, items, cfg) -> dict:
    """Mean |logp_trainer - logp_vllm| on a greedy sample. Detects broken weight sync."""
    it = items[0]
    s = engine.generate([it.ids], n=1, temperature=0.0, top_p=1.0, top_k=-1, max_tokens=256)[0][0]
    comp = s.completion_ids[:256]
    if not comp:
        return {"skipped": "empty completion"}
    v = engine.sequence_logprobs(it.ids, comp)
    ids = torch.tensor([it.ids + comp], device="cuda")
    policy.eval()
    with torch.no_grad():
        t = lp.logprobs(ids, len(it.ids)).float().cpu().tolist()
    policy.train()
    diffs = [abs(a - b) for a, b in zip(t, v)]
    res = {"tokens": len(diffs), "mean_abs": statistics.fmean(diffs), "max_abs": max(diffs),
           "tolerance": cfg.rollout.parity_tolerance}
    res["passed"] = res["mean_abs"] <= cfg.rollout.parity_tolerance
    return res


def run(cfg: Config, args) -> None:
    from .structured import validate_training_config
    validate_training_config(cfg)
    tcfg, rcfg = cfg.train, cfg.reward
    run_dir = Path(tcfg.output_dir) / tcfg.run_name
    setup_logging(run_dir)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)

    # Memory baseline BEFORE we allocate anything (judge is already up via compose).
    mem = MemoryMonitor(cfg.memory)
    mem.start()

    (run_dir / "resolved-config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    model_sha = resolve_sha(cfg.model.id, cfg.model.revision)
    ds = cfg.judge.backend == "deepseek"
    judge_sha = "deepseek:" + cfg.judge.deepseek_model if ds else resolve_sha(cfg.judge.hf_id, cfg.judge.revision)
    log.info("policy %s@%s, judge %s@%s", cfg.model.id, model_sha, cfg.judge.hf_id, judge_sha)

    from transformers import AutoTokenizer

    from .policy import LogprobComputer, build_policy, iter_merged_weights, load_tokenizer, module_tree
    from .rollout import PolicyEngine

    tok = load_tokenizer(cfg.model.id, model_sha)
    verifier = Verifier(rcfg.verifier_workers, rcfg.verifier_timeout_s)
    data = prepare_data(cfg.data, tcfg.seed, verifier)
    template = Path(cfg.data.policy_prompt_path).read_text()

    system = cfg.data.policy_system_prompt
    train_items = build_items(tok, data["train"], template, cfg.model.thinking, tcfg.max_prompt_tokens, system)
    val_items = build_items(tok, data["val"], template, cfg.model.thinking, tcfg.max_prompt_tokens, system)
    test_items = build_items(tok, data["test"], template, cfg.model.thinking, tcfg.max_prompt_tokens, system)
    if "nash" in cfg.data.sources and any(
        len(items) != len(data[split])
        for split, items in (("train", train_items), ("val", val_items), ("test", test_items))
    ):
        raise Abort("DATA_PROMPT_LIMIT", "Nash prompts exceeded the prompt limit; no rows may be silently dropped")
    sample_prompt = render_prompt(tok, template, "What is 2+2?", cfg.model.thinking, system)
    opened = sample_prompt.rstrip().endswith(cfg.model.think_open)
    log.info("items train=%d val=%d test=%d; thinking opened by template: %s",
             len(train_items), len(val_items), len(test_items), opened)

    # ---- engines: vLLM first (so its memory profiling sees the free pool), then trainer model
    engine = PolicyEngine(cfg.model, cfg.rollout, model_sha,
                          tcfg.max_prompt_tokens + tcfg.max_completion_tokens, tcfg.seed)
    policy, targets = build_policy(cfg.model, model_sha)
    (run_dir / "module_tree.txt").write_text(module_tree(policy.get_base_model()))
    lp = LogprobComputer(policy, cfg.model.logprob_chunk_tokens)

    compat = {"model": cfg.model.id, "model_sha": model_sha, "judge": cfg.judge.hf_id, "judge_sha": judge_sha,
              "versions": versions(), "lora_targets": len(targets), "lora_target_examples": targets[:8],
              "thinking_opened_by_template": opened, "prompt_tail": sample_prompt[-200:],
              "think_marker_ids": [tok.convert_tokens_to_ids(cfg.model.think_open),
                                   tok.convert_tokens_to_ids(cfg.model.think_close)],
              "unsloth": False, "nf4": False, "rollout_engine": "vllm in-process + merged-weight push"}
    try:
        s = engine.generate([val_items[0].ids], n=1, temperature=0.0, top_p=1.0, top_k=-1, max_tokens=64)[0][0]
        ids = torch.tensor([val_items[0].ids + s.completion_ids], device="cuda")
        policy.eval()
        compat["logprob_calibration"] = lp.calibrate(ids, len(val_items[0].ids))
        policy.train()
    except Exception as e:  # noqa: BLE001
        compat["logprob_calibration"] = {"passed": False, "error": repr(e)[:500]}
        (run_dir / "compatibility.json").write_text(json.dumps(compat, indent=2))
        raise Abort("COMPATIBILITY_FAILED", f"log-prob calibration failed: {e}") from e

    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay,
                            betas=(0.9, 0.99), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, tcfg.warmup_updates)))
    sampler = Sampler(len(train_items), tcfg.seed)
    state: dict = {"fingerprint": cfg.fingerprint(), "length_gate_open": True, "best": None}
    start = 0

    ck = None if args.fresh else latest_checkpoint(run_dir)
    if ck is not None:
        st = torch.load(ck / "trainer_state.pt", map_location="cpu", weights_only=False)
        if st["state"].get("fingerprint") != cfg.fingerprint() and not args.force_resume:
            raise Abort("FINGERPRINT_MISMATCH",
                        f"{ck} was produced with a different model/judge/reward/data config. "
                        "Use a new run_name (or --force-resume if you really mean it).")
        load_adapter(policy, ck / "adapter")
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        sampler.load(st["sampler"])
        state = st["state"]
        random.setstate(st["rng"]["python"])
        torch.set_rng_state(st["rng"]["torch"])
        if st["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        start = st["update"] + 1
        log.info("resumed from %s at update %d", ck, start)
    elif getattr(args, "init_adapter", None):
        load_adapter(policy, Path(args.init_adapter))
        log.info("policy initialised from adapter %s (the baseline eval will measure this policy)", args.init_adapter)

    # push current weights (also validates HF->vLLM parameter names) and check parity
    n_pushed = engine.push_weights(iter_merged_weights(policy, cfg.model.weight_name_rewrites))
    compat["weights_pushed_per_sync"] = n_pushed
    compat["parity_initial"] = parity_check(engine, lp, policy, val_items, cfg)
    (run_dir / "compatibility.json").write_text(json.dumps(compat, indent=2))
    if not compat["parity_initial"].get("passed", True):
        raise Abort("POLICY_SYNC_FAILED", f"trainer/vLLM log-prob parity failed: {compat['parity_initial']}")

    if ds:
        judge = DeepSeekScorer(cfg.judge)
        judge.rubric_sha = hashlib.sha256(judge.system.encode()).hexdigest()
    else:
        judge_tok = AutoTokenizer.from_pretrained(cfg.judge.hf_id, revision=judge_sha)
        rubric = Path(cfg.judge.rubric_path).read_text()
        judge = StepJudge(cfg.judge, rcfg, rubric, judge_tok, judge_sha)
        judge.wait_ready()

    v2 = None
    if ds and cfg.reward_v2.enabled:
        from .novelty import Archive
        from .verifier_client import RemoteVerifier
        rv = RemoteVerifier(cfg.reward_v2.verifier_url, cfg.reward_v2.verifier_timeout_s)
        hh = rv.health()
        if hh is None:
            raise Abort("VERIFIER_UNAVAILABLE", f"reward_v2 needs the verifier service at {cfg.reward_v2.verifier_url} (docker compose up -d verifier sandbox)")
        archive = Archive((cfg.reward_v2.archive_path or "").replace("{run}", tcfg.run_name) or None)
        v2 = RewardV2(cfg.reward_v2, rv, archive)
        log.info("agent service version %s", hh.get("version"))
        log.info("reward v2 enabled: exact fast path + ReAct verifier agent service %s, gated judge bonus, novelty bonus", cfg.reward_v2.verifier_url)

    v3 = None
    if ds and cfg.reward_v3.enabled:
        v3 = RewardClient(cfg.reward_v3.url, cfg.reward_v3.timeout_s)
        hh = v3.health()
        if hh is None or not hh.get("key_ok"):
            raise Abort("REWARD_SERVICE_UNAVAILABLE", f"reward_v3 needs the reward service at {cfg.reward_v3.url} with a readable DeepSeek key: {hh}")
        log.info("reward service v3 at %s version %s (the agent decides the reward, range %s..%s)", cfg.reward_v3.url, hh.get("version"), hh.get("reward_min"), hh.get("reward_max"))

    (run_dir / "manifest.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "fingerprint": cfg.fingerprint(),
        "model_sha": model_sha, "judge_sha": judge_sha, "rubric_sha256": judge.rubric_sha,
        "policy_prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "policy_system_prompt": cfg.data.policy_system_prompt,
        "versions": compat["versions"], "seed": tcfg.seed,
        "memory_limits": cfg.memory.model_dump(),
    }, indent=2))

    if args.eval_only:
        if args.adapter:
            load_adapter(policy, Path(args.adapter))
            engine.push_weights(iter_merged_weights(policy, cfg.model.weight_name_rewrites))
        split_items = {"val": val_items, "test": test_items}[args.split]
        res, _ = evaluate(engine, tok, split_items, cfg, verifier, opened, args.limit or len(split_items))
        print(json.dumps(res, indent=2))
        (run_dir / f"eval-{args.split}.json").write_text(json.dumps(res, indent=2))
        return

    if state.get("baseline") is None and not ds:
        log.info("baseline evaluation on %d val problems", min(tcfg.eval_problems, len(val_items)))
        res, per_id = evaluate(engine, tok, val_items, cfg, verifier, opened, tcfg.eval_problems)
        state["baseline"], state["baseline_per_id"] = res, per_id
        (run_dir / "eval").mkdir(exist_ok=True)
        (run_dir / "eval" / "baseline.json").write_text(json.dumps(res, indent=2))
        log.info("baseline: %s", res)

    metrics_f = open(run_dir / "metrics.jsonl", "a")
    (run_dir / "rollouts").mkdir(exist_ok=True)
    trunc_window: deque = deque(maxlen=100)
    P, G, L_cap = tcfg.prompts_per_update, tcfg.generations_per_prompt, tcfg.max_completion_tokens
    last_saved = start - 1
    update = start - 1

    try:
        t_launch = time.time()
        for update in range(start, tcfg.max_updates):
            if STOP["flag"]:
                break
            if v3 is not None and cfg.reward_v3.max_cost_usd and v3.cost_usd() >= cfg.reward_v3.max_cost_usd:
                log.info("reward service spend guard: estimated $%.2f >= max_cost_usd %.2f, stopping cleanly at update %d", v3.cost_usd(), cfg.reward_v3.max_cost_usd, update)
                break
            if ds and v3 is None and cfg.judge.max_cost_usd and judge.cost_usd() + (v2.agent.cost_usd() if v2 is not None else 0.0) >= cfg.judge.max_cost_usd:
                log.info("DeepSeek spend guard: estimated $%.2f >= max_cost_usd %.2f, stopping cleanly at update %d", judge.cost_usd(), cfg.judge.max_cost_usd, update)
                break
            if ds and update % 10 == 0:
                bal = judge.balance_usd()
                if bal is not None and bal < cfg.judge.min_balance_usd:
                    log.info("DeepSeek balance $%.2f < min_balance_usd %.2f, stopping cleanly at update %d", bal, cfg.judge.min_balance_usd, update)
                    break
            if tcfg.max_hours and time.time() - t_launch > tcfg.max_hours * 3600:
                log.info("max_hours=%.1f reached after %.1f h: stopping cleanly at update %d", tcfg.max_hours, (time.time() - t_launch) / 3600, update)
                break
            if mem.breach:
                raise Abort(mem.breach, f"memory watchdog: {mem.snapshot()}")
            t0 = time.time()
            lam_p = ramp(update, rcfg.process_start_update, rcfg.process_ramp_updates, rcfg.process_weight)
            lam_len = (ramp(update, rcfg.length_start_update, rcfg.length_ramp_updates, rcfg.length_weight)
                       if state.get("length_gate_open", True) else 0.0)

            if ds:
                lam_p = lam_len = 0.0       # reward = DeepSeek score only

            # ---- 1) rollouts
            items = [train_items[i] for i in sampler.take(P)]
            samples = engine.generate([it.ids for it in items], n=G, temperature=cfg.rollout.temperature,
                                      top_p=cfg.rollout.top_p, top_k=cfg.rollout.top_k, max_tokens=L_cap)
            t_gen = time.time()

            # ---- 2-3) parse + verify; drop groups with operational/data failures
            groups = score_rollouts(items, samples, tok, cfg, verifier, opened, verify=not ds)
            dropped = []
            kept_groups = []
            for g in groups:
                bad = [t.verify_status for t in g if is_operational_failure(t.verify_status)]
                if bad:
                    dropped.append({"problem": g[0].item.problem.id, "reasons": bad})
                else:
                    kept_groups.append(g)
            if dropped:
                log.warning("dropped %d groups (verifier/data failure): %s", len(dropped), dropped)
            if len(dropped) > P // 2:
                raise Abort("VERIFIER_FAILURE", f"{len(dropped)}/{P} groups failed verification")
            groups = kept_groups
            if not groups:
                continue
            for g in groups:
                for t in g:
                    trunc_window.append(t.parsed.truncated)
            if (tcfg.truncated == "mask" and len(trunc_window) == trunc_window.maxlen
                    and sum(trunc_window) / len(trunc_window) > tcfg.max_truncation_rate):
                raise Abort("TRUNCATION_RATE_HIGH",
                            f"{sum(trunc_window)}% of last 100 rollouts truncated; raise max_completion_tokens "
                            "(new run) or use truncated: penalize")

            if ds:
                # ---- 4-5) DeepSeek scores every rollout; a prompt group with any failed call is dropped, never guessed
                gjobs = []
                for g in groups:
                    cands = []
                    for t in g:
                        th, reply = split_completion(tok.decode(t.completion_ids, skip_special_tokens=False),
                                                     cfg.model.think_open, cfg.model.think_close)
                        cands.append({"thinking": th, "reply": reply, "finish": "length" if t.parsed.truncated else "stop"})
                    gjobs.append({"problem": g[0].item.problem.problem, "reference": g[0].item.problem.gold, "cands": cands})
                ds_items = Counter()
                if v3 is not None:
                    # the reward service owns scoring; the trainer only ships raw groups and applies the finished numbers
                    v3_jobs = [{"problem": jb["problem"], "gold": jb["reference"], "domain": g[0].item.problem.source,
                                "cands": [dict(c, truncated=t.parsed.truncated, hard_valid=t.parsed.hard_valid, ntok=len(t.completion_ids))
                                          for t, c in zip(g, jb["cands"])]} for g, jb in zip(groups, gjobs)]
                    apply_rewards(groups, v3.score_groups(v3_jobs))
                    g_rewards = g_items = None
                else:
                    g_rewards, g_items = judge.score_groups(gjobs)
                    for g, rs, its in zip(groups, g_rewards, g_items):
                        for t, sc, it in zip(g, rs, its):
                            t.reward = float("nan") if sc is None else sc
                            if it is not None:
                                for k, v in it.items():
                                    ds_items[k] += v
                if v2 is not None:
                    v2_groups = []
                    for g, its, jb in zip(groups, g_items, gjobs):
                        v2_groups.append({"problem": jb["problem"], "gold_reply": jb["reference"], "domain": g[0].item.problem.source,
                                          "cands": [dict(c, truncated=t.parsed.truncated, items=it, ntok=len(t.completion_ids)) for t, c, it in zip(g, jb["cands"], its)]})
                    v2_parts = v2.score(v2_groups, update)
                    for g, parts in zip(groups, v2_parts):
                        for k, t in enumerate(g):
                            t.reward = float("nan") if parts is None else parts[k].reward
                            if parts is not None:
                                t.correct = parts[k].v is not None and parts[k].v >= cfg.reward_v2.gate_v
                                t.verify_status = parts[k].src      # exact | code | text | think | skip | none (was a stale 'no_answer' in the logs)
                                t.reward_parts = dataclasses.asdict(parts[k])
                    if update % 10 == 0 and v2.archive is not None:
                        v2.archive.save()
                ok_groups = [g for g in groups if all(t.reward == t.reward for t in g)]
                if len(ok_groups) < len(groups):
                    log.warning("dropped %d/%d groups: no usable reward (judge call or verifier failed: %s)", len(groups) - len(ok_groups), len(groups), dict(v2.stats) if v2 is not None else dict(v3.stats) if v3 is not None else dict(judge.stats))
                if len(ok_groups) < len(groups) * (1 - cfg.judge.max_failed_groups_fraction):
                    raise Abort("JUDGE_UNAVAILABLE", f"DeepSeek failed for {len(groups) - len(ok_groups)}/{len(groups)} groups: {dict(judge.stats)}")
                groups = ok_groups
                if not groups:
                    continue
                if v2 is not None or v3 is not None:
                    flat_eps, neg_beta = ((cfg.reward_v3.flat_group_eps, cfg.reward_v3.negative_beta) if v3 is not None
                                          else (cfg.reward_v2.flat_group_eps, cfg.reward_v2.negative_beta))
                    n_flat = 0
                    keep = []
                    for g in groups:
                        rs = [t.reward for t in g]
                        if max(rs) - min(rs) < flat_eps:
                            n_flat += 1
                            continue
                        for t, a in zip(g, v2_advantages(rs, neg_beta)):
                            t.advantage = a
                        keep.append(g)
                    if n_flat:
                        log.info("skipped %d flat groups (no reward spread)", n_flat)
                    groups = keep
                    if not groups:
                        continue
                else:
                    for g in groups:
                        for t, a in zip(g, group_advantages([t.reward for t in g])):
                            t.advantage, t.correct = a, t.reward >= 0.5
            t_ds = time.time()

            # ---- 4) step judge (prefix-only), only for well-formed, complete traces
            judged = []
            if lam_p > 0:
                for g in groups:
                    for t in g:
                        if t.parsed.hard_valid and t.parsed.spans_ok and not t.parsed.truncated:
                            judged.append(t)
                verdicts = judge.judge_batch([(t.item.problem.problem, [s.text for s in t.parsed.steps])
                                              for t in judged])
                for t, vs in zip(judged, verdicts):
                    t.verdicts = vs
                    t.step_q = [verdict_to_q(v, rcfg) for v in vs]
            t_judge = time.time()

            # ---- 5) outcome rewards + Dr. GRPO advantages
            for g in ([] if ds else groups):
                rewards, lterm = outcome_rewards(
                    [t.correct for t in g], [t.parsed.hard_valid for t in g], [t.parsed.truncated for t in g],
                    [t.parsed.thinking_tokens for t in g], format_penalty=rcfg.format_penalty,
                    truncation_penalty=rcfg.truncation_penalty, lam_len=lam_len,
                    step_ok=[t.parsed.spans_ok for t in g], rep_frac=[t.parsed.rep_frac for t in g],
                    step_bonus=rcfg.step_format_bonus, rep_penalty=rcfg.repetition_penalty)
                for t, r, l, a in zip(g, rewards, lterm, group_advantages(rewards)):
                    t.reward, t.length_term, t.advantage = r, l, a

            # ---- 6) policy gradient, one trajectory at a time, exact global normalisation
            all_t = [t for g in groups for t in g]
            B = len(all_t)
            policy.train()
            stats = Counter()
            work = []
            for t in all_t:
                n = len(t.completion_ids)
                if n == 0 or max(t.completion_ids) >= lp.vocab_rows:
                    stats["skipped_bad_ids"] += 1
                    continue
                masked = t.parsed.truncated and tcfg.truncated == "mask"
                has_out = (not masked) and t.advantage != 0.0
                q_tok = w_tok = None
                if lam_p > 0 and t.step_q is not None:
                    q, w = process_token_targets([(s.tok_start, s.tok_end) for s in t.parsed.steps], t.step_q, n)
                    if any(q):
                        q_tok, w_tok = q, w
                if not has_out and q_tok is None:
                    continue  # contributes exactly zero gradient; still counted in B
                work.append((t, masked, q_tok, w_tok))

            old_cache: dict[int, torch.Tensor] = {}
            if tcfg.ppo_epochs > 1:
                with torch.no_grad():
                    for k, (t, *_r) in enumerate(work):
                        ids = torch.tensor([t.item.ids + t.completion_ids], device="cuda")
                        old_cache[k] = lp.logprobs(ids, len(t.item.ids)).detach()

            grad_norm = 0.0
            for epoch in range(tcfg.ppo_epochs):
                opt.zero_grad(set_to_none=True)
                for k, (t, masked, q_tok, w_tok) in enumerate(work):
                    ids = torch.tensor([t.item.ids + t.completion_ids], device="cuda")
                    logp = lp.logprobs(ids, len(t.item.ids))
                    old = old_cache.get(k, logp.detach())
                    n = logp.shape[0]
                    out_mask = torch.zeros(n, device=logp.device) if masked else torch.ones(n, device=logp.device)
                    q_t = torch.tensor(q_tok, device=logp.device) if q_tok is not None else None
                    w_t = torch.tensor(w_tok, device=logp.device) if w_tok is not None else None
                    loss, l_out, l_proc, clipped = trajectory_loss(
                        logp, old, t.advantage, out_mask, q_t, w_t, B=B, L_cap=L_cap, lam_proc=lam_p,
                        eps_low=tcfg.clip_eps_low, eps_high=tcfg.clip_eps_high)
                    loss.backward()
                    stats["loss_out"] += l_out.item()
                    stats["loss_proc"] += l_proc.item()
                    stats["clipped_sum"] += clipped.item()
                    stats["trained_traj"] += 1
                    del logp, loss
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, tcfg.max_grad_norm))
                if not torch.isfinite(torch.tensor(grad_norm)):
                    raise RuntimeError("non-finite gradient norm")
                opt.step()
            sched.step()
            t_train = time.time()

            # ---- 7) sync weights into vLLM
            engine.push_weights(iter_merged_weights(policy, cfg.model.weight_name_rewrites))
            t_sync = time.time()

            # ---- metrics + audit
            corr = [t for t in all_t if t.correct]
            jstats = Counter()
            for t in judged:
                for v in t.verdicts or []:
                    jstats["none" if v is None else v.validity] += 1
                    if v is not None and (v.redundant or v.relevance == "irrelevant"):
                        jstats["redundant_or_irrelevant"] += 1
            n_verd = max(1, sum(len(t.verdicts or []) for t in judged))
            group_div = sum(1 for g in groups if len({round(t.reward, 6) for t in g}) > 1) / len(groups)
            row = {
                "update": update, "lr": sched.get_last_lr()[0], "lam_process": lam_p, "lam_length": lam_len,
                "B": B, "trained_traj": stats["trained_traj"] // tcfg.ppo_epochs,
                "accuracy": len(corr) / B,
                "reward_mean": statistics.fmean(t.reward for t in all_t),
                "reward_std": statistics.pstdev(t.reward for t in all_t),
                "group_diversity": group_div,
                "format_valid": sum(t.parsed.format_valid for t in all_t) / B,
                "hard_valid": sum(t.parsed.hard_valid for t in all_t) / B,
                "steps_ok": sum(t.parsed.spans_ok for t in all_t) / B,
                "rep_frac_mean": statistics.fmean(t.parsed.rep_frac for t in all_t),
                "truncated": sum(t.parsed.truncated for t in all_t) / B,
                "completion_tokens_mean": statistics.fmean(len(t.completion_ids) for t in all_t),
                "thinking_tokens_mean": statistics.fmean(t.parsed.thinking_tokens for t in all_t),
                "thinking_tokens_correct_mean": statistics.fmean(t.parsed.thinking_tokens for t in corr) if corr else None,
                "steps_mean": statistics.fmean(len(t.parsed.steps) for t in all_t),
                "judged_traj": len(judged),
                "step_invalid_rate": jstats["invalid"] / n_verd,
                "step_uncertain_rate": (jstats["uncertain"] + jstats["none"]) / n_verd,
                "step_redundant_rate": jstats["redundant_or_irrelevant"] / n_verd,
                "loss_out": stats["loss_out"], "loss_proc": stats["loss_proc"],
                "clip_frac": stats["clipped_sum"] / max(1, stats["trained_traj"]),
                "grad_norm": grad_norm, "dropped_groups": len(dropped),
                "format_errors": dict(Counter(e for t in all_t for e in t.parsed.errors)),
                "judge_stats": dict(judge.stats),
                "reward_min": min(t.reward for t in all_t), "reward_max": max(t.reward for t in all_t),
                "reward_p50": statistics.median(t.reward for t in all_t),
                "ds_calls": judge.stats.get("calls") if ds else None, "ds_errors": judge.stats.get("errors") if ds else None,
                "ds_unparseable": judge.stats.get("unparseable") if ds else None,
                "ds_tokens_out": judge.stats.get("completion_tokens") if ds else None,
                "ds_items": ({k: v / B for k, v in ds_items.items()} if ds else None),
                "ds_cost_usd": (round(judge.cost_usd(), 4) if ds else None),
                "v3": ({**dict(v3.stats), "cost_usd": round(v3.cost_usd(), 4), "version": v3.version} if v3 is not None else None),
                "v2": ({**dict(v2.stats), "rho": round(v2.rho_tracker.rho(), 3), "agent_cost_usd": round(v2.agent.cost_usd(), 4),
                        "agent_stats": dict(v2.agent.stats), "agent_version": v2.agent.version} if v2 is not None else None),
                "time_gen": t_gen - t0, "time_judge": t_judge - t_gen, "time_train": t_train - t_judge,
                "time_sync": t_sync - t_train, "time_total": t_sync - t0,
                "memory": mem.snapshot(),
            }
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            log.info("upd %d acc=%.3f R=%.3f think=%.0f trunc=%.2f fmt=%.2f λp=%.2f λlen=%.2f gn=%.3f %.0fs",
                     update, row["accuracy"], row["reward_mean"], row["thinking_tokens_mean"], row["truncated"],
                     row["hard_valid"], lam_p, lam_len, grad_norm, row["time_total"])

            with gzip.open(run_dir / "rollouts" / f"update-{update:06d}.jsonl.gz", "wt") as f:
                for t in all_t:
                    f.write(json.dumps({
                        "update": update, "problem_id": t.item.problem.id, "gold": t.item.problem.gold,
                        "completion_ids": t.completion_ids, "completion": tok.decode(t.completion_ids, skip_special_tokens=False),
                        "finish_reason": t.finish_reason, "final_answer": t.parsed.final_answer,
                        "correct": t.correct, "verify_status": t.verify_status,
                        "format_valid": t.parsed.format_valid, "hard_valid": t.parsed.hard_valid, "rep_frac": t.parsed.rep_frac,
                        "format_errors": t.parsed.errors,
                        "truncated": t.parsed.truncated, "thinking_tokens": t.parsed.thinking_tokens,
                        "reward": t.reward, "length_term": t.length_term, "advantage": t.advantage, "reward_parts": t.reward_parts,
                        "steps": [{"id": s.id, "tok": [s.tok_start, s.tok_end],
                                   "q": (t.step_q[i] if t.step_q else None),
                                   "verdict": (t.verdicts[i].model_dump() if t.verdicts and t.verdicts[i] else None)}
                                  for i, s in enumerate(t.parsed.steps)],
                    }) + "\n")

            # ---- periodic: parity, eval, checkpoint
            if cfg.rollout.parity_check_every and (update + 1) % cfg.rollout.parity_check_every == 0:
                par = parity_check(engine, lp, policy, val_items, cfg)
                log.info("parity: %s", par)
                if not par.get("passed", True):
                    raise Abort("POLICY_SYNC_FAILED", f"parity drift: {par}")
            if tcfg.eval_every and (update + 1) % tcfg.eval_every == 0:
                res, _ = evaluate(engine, tok, val_items, cfg, verifier, opened, tcfg.eval_problems,
                                  state.get("baseline_per_id"))
                base_acc = state["baseline"]["accuracy"]
                state["length_gate_open"] = res["accuracy"] >= base_acc - rcfg.length_accuracy_tolerance
                res.update(update=update, length_gate_open=state["length_gate_open"])
                metrics_f.write(json.dumps({"eval": res}) + "\n")
                metrics_f.flush()
                log.info("eval @%d: %s", update, res)
                score = (res["accuracy"], -(res["thinking_tokens_correct_mean"] or 1e9))
                if state.get("best") is None or score > tuple(state["best"]["score"]):
                    state["best"] = {"update": update, "score": list(score), "eval": res}
                    save_best(run_dir, policy, state["best"])
            if tcfg.checkpoint_every and (update + 1) % tcfg.checkpoint_every == 0:
                save_checkpoint(run_dir, update, policy, opt, sched, sampler, state, tcfg.keep_checkpoints)
                last_saved = update
    except JudgeUnavailable as e:
        if update - 1 > last_saved:  # the failing update had not been applied yet
            save_checkpoint(run_dir, update - 1, policy, opt, sched, sampler, state, tcfg.keep_checkpoints)
        raise Abort("JUDGE_UNAVAILABLE", str(e)) from e
    except Abort as e:
        if e.code == "HOST_MEMORY_LOW" and update - 1 > last_saved:
            save_checkpoint(run_dir, update - 1, policy, opt, sched, sampler, state, tcfg.keep_checkpoints)
        raise
    finally:
        metrics_f.close()

    if update > last_saved:
        save_checkpoint(run_dir, update, policy, opt, sched, sampler, state, tcfg.keep_checkpoints)
    if STOP["flag"]:
        log.info("stopped cleanly at update %d", update)
        return

    if ds:
        log.info("done (DeepSeek-judged run: no verifier eval; evaluate the saved checkpoints separately)")
        return

    # ---- final report: val + held-out test (never used for selection)
    final_val, _ = evaluate(engine, tok, val_items, cfg, verifier, opened, tcfg.eval_problems,
                            state.get("baseline_per_id"))
    final_test, _ = evaluate(engine, tok, test_items, cfg, verifier, opened, len(test_items))
    (run_dir / "eval" / "final.json").write_text(json.dumps({"val": final_val, "test": final_test}, indent=2))
    b = state["baseline"]
    (run_dir / "report.md").write_text(
        f"# Apertus 1.5 RLAIF report\n\n"
        f"Updates: {update + 1}. Model {cfg.model.id}@{model_sha}. Judge {cfg.judge.hf_id}@{judge_sha}.\n\n"
        f"| split | accuracy | thinking tokens (correct, mean) | format valid | truncated |\n|---|---|---|---|---|\n"
        f"| val baseline | {b['accuracy']:.3f} | {b['thinking_tokens_correct_mean']} | {b['format_valid']:.3f} | {b['truncated']:.3f} |\n"
        f"| val final | {final_val['accuracy']:.3f} | {final_val['thinking_tokens_correct_mean']} | {final_val['format_valid']:.3f} | {final_val['truncated']:.3f} |\n"
        f"| GSM8K test final | {final_test['accuracy']:.3f} | {final_test['thinking_tokens_correct_mean']} | {final_test['format_valid']:.3f} | {final_test['truncated']:.3f} |\n\n"
        f"Thinking-token ratio on val problems solved by both baseline and final: "
        f"{final_val.get('both_solved_thinking_ratio')} (n={final_val.get('both_solved_n')}).\n\n"
        f"Peak training memory estimate: {mem.peak_training / 1e9:.1f} GB; min host available: "
        f"{mem.min_available / 1e9:.1f} GB (method: {mem.snapshot()['method']}).\n\n"
        f"Training reward and judge scores alone do not establish better reasoning; compare the "
        f"answer-only / +process / +process+length ablations at equal budget and seeds.\n")
    log.info("done. final val %s | test %s", final_val, final_test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints in the run dir")
    ap.add_argument("--force-resume", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adapter", default=None, help="adapter dir for --eval-only")
    ap.add_argument("--init-adapter", default=None, help="start training from this adapter (e.g. the SFT result)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default=None,
                    help="override data.policy_prompt_path (robustness eval: prompts/eval_heldout_wrapper.txt)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.prompt:
        cfg.data.policy_prompt_path = args.prompt
    try:
        run(cfg, args)
    except Abort as e:
        log.error("ABORT %s: %s", e.code, e)
        print(json.dumps({"error": e.code, "detail": str(e)[:2000]}), file=sys.stderr)
        sys.exit(EXIT_CODES.get(e.code, 1))


if __name__ == "__main__":
    main()
