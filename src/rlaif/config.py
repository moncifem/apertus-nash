"""Strict, validated configuration. Unknown fields are rejected."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelCfg(Strict):
    id: str = "swiss-ai/Apertus-v1.5-8B"
    revision: str = "main"  # resolved to an immutable commit SHA at startup and recorded
    attn_implementation: str = "sdpa"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # xIELU MLP has up_proj/down_proj and no gate_proj.
    lora_target_suffixes: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    # Any module whose full name contains one of these (case-insensitive) is never adapted.
    lora_exclude_substrings: list[str] = ["vision", "visual", "audio", "image", "projector", "multi_modal", "mm_"]
    thinking: bool = True
    think_open: str = "<|inner_prefix|>"
    think_close: str = "<|inner_suffix|>"
    gradient_checkpointing: bool = True
    logprob_chunk_tokens: int = 1024
    # (regex, replacement) pairs applied to HF parameter names before pushing them into vLLM.
    # Only needed if the Transformers module tree uses different names than the checkpoint.
    weight_name_rewrites: list[tuple[str, str]] = []


class RolloutCfg(Strict):
    gpu_memory_utilization: float = 0.30
    max_num_seqs: int = 64
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    vllm_extra_kwargs: dict[str, Any] = {}
    parity_tolerance: float = 0.05  # mean |logp_trainer - logp_vllm| on sampled tokens
    parity_check_every: int = 25
    weight_push_chunk_bytes: int = 2_000_000_000


class TrainCfg(Strict):
    seed: int = 42
    max_updates: int = 300
    prompts_per_update: int = 8
    generations_per_prompt: int = 8
    max_prompt_tokens: int = 1024
    max_completion_tokens: int = 4096
    learning_rate: float = 1.0e-5
    warmup_updates: int = 10
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    ppo_epochs: int = 1  # 1 = strictly on-policy (old logp == current logp detached)
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28
    truncated: Literal["penalize", "mask"] = "penalize"
    max_truncation_rate: float = 0.10  # over a 100-rollout window; enforced (pause) only in "mask" mode
    checkpoint_every: int = 25
    keep_checkpoints: int = 3
    eval_every: int = 50
    eval_problems: int = 200
    output_dir: str = "/runs"
    run_name: str = "apertus-math"
    max_hours: float = 0.0       # wall-clock limit for this launch (0 = none): stops cleanly after the update that crosses it and saves a checkpoint


class JudgeCfg(Strict):
    base_url: str = "http://judge:8000/v1"
    served_model_name: str = "judge"
    hf_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str = "main"
    max_context_tokens: int = 16384
    max_output_tokens: int = 256
    temperature: float = 0.0
    max_concurrency: int = 16
    timeout_s: float = 120.0
    retries: int = 3
    max_malformed_fraction: float = 0.05
    cache_path: str = "/runs/judge_cache.sqlite"
    rubric_path: str = "prompts/judge_system.txt"
    use_json_schema: bool = True
    ready_timeout_s: float = 1800.0
    # backend "deepseek": the reward is DeepSeek's score/100 (no Qwen judge, no verifier). The key comes from a file (docker secret).
    backend: Literal["qwen", "deepseek"] = "qwen"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_effort: Literal["none", "low", "high", "max"] = "none"      # "none" = no hidden reasoning tokens
    deepseek_max_tokens: int = 8000
    deepseek_key_path: str = "/run/secrets/deepseek_key"
    deepseek_prompt_path: str = "prompts/judge/deepseek_judge_items.md"
    max_failed_groups_fraction: float = 0.5
    # >0: the judge sees at most this many chars of each candidate's thinking (head 2/3 + tail 1/3). Needed with long completions (8k tokens x 8
    # candidates would be ~64k tokens per call). 0 = send everything.
    max_thinking_chars: int = 0
    # spend guard (USD per 1M tokens at PEAK prices, i.e. conservative): stop cleanly when this run's estimated spend reaches max_cost_usd (0 = off)
    # or when the live account balance drops below min_balance_usd
    price_hit: float = 0.006
    price_miss: float = 0.30
    price_out: float = 1.20
    max_cost_usd: float = 0.0
    min_balance_usd: float = 1.0


class RewardCfg(Strict):
    format_penalty: float = 0.2
    truncation_penalty: float = 0.5
    step_format_bonus: float = 0.1     # small bonus when thinking has well-formed numbered steps
    repetition_penalty: float = 0.2    # penalty for looping / repeated n-grams in thinking
    # Conciseness: group-relative length reward on thinking tokens (Kimi k1.5 style).
    length_weight: float = 0.2
    length_start_update: int = 30
    length_ramp_updates: int = 30
    length_accuracy_tolerance: float = 0.02  # gate: val acc must stay within this of baseline
    # Step-level process reward from the judge.
    process_weight: float = 0.2
    process_start_update: int = 20
    process_ramp_updates: int = 30
    max_steps: int = 48
    q_invalid: float = -1.0
    q_uncertain: float = 0.0
    q_redundant: float = -0.5
    q_valid: float = 1.0
    verifier_timeout_s: float = 10.0
    verifier_workers: int = 4


class DataCfg(Strict):
    sources: list[Literal["gsm8k", "numina", "curriculum", "rewritten", "nash"]] = ["gsm8k", "numina"]
    nash_path: str = "data/nash/rlaif.jsonl"
    gsm8k_revision: str = "main"
    numina_revision: str = "main"
    gsm8k_max_examples: int | None = None
    numina_max_examples: int = 12000
    val_size: int = 300
    # how the final answer is read from a reply: "boxed" = one \\boxed{} (RL v3, SFT v1); "anchor" = last "**Answer:** x" line (free-form replies)
    answer_format: Literal["boxed", "anchor", "plain"] = "boxed"
    reasoning_format: Literal["numbered", "prose"] = "numbered"
    policy_prompt_path: str = "prompts/user_message.txt"
    # Replaces the chat template's built-in system message. None keeps it (the v3 run). Use the SAME text as
    # sft.yaml `data.system_prompt` when starting RL from an SFT adapter.
    policy_system_prompt: str | None = None
    cache_dir: str = "/runs/data"

    # --- generated curriculum (dataset/SPEC.md) -------------------------------
    # out/curriculum.jsonl is the product of the SPEC.md pipeline. Only rows whose
    # answer is mechanically checkable are usable here, because the RLAIF reward
    # compares the \boxed{} answer to `gold` with Math-Verify.
    curriculum_path: str = "out/curriculum.jsonl"
    curriculum_min_level: int = 0
    curriculum_max_level: int = 5
    curriculum_answer_types: list[str] = ["scalar", "expression"]
    curriculum_check_types: list[str] = ["numeric", "exact", "symbolic", "multi"]
    # Use the row's own train/val/test split, or re-split deterministically.
    curriculum_use_row_split: bool = True
    # sources ["rewritten"]: prompts from the SFT rewrite rows (out/sft_rewritten_rows.jsonl); their reference reply is the judge's ground truth
    rewritten_path: str = "/app/out/sft_rewritten_rows.jsonl"
    rewritten_single_turn_only: bool = True


class RewardV2Cfg(Strict):
    """Reward v2 (src/rlaif/reward_v2.py): verifier-first, gated judge bonus, novelty bonus. Used with judge.backend: deepseek."""
    enabled: bool = False
    # --- trusted verification: exact fast path + the verifier agent, which runs as its own service (reward_service/, compose service `verifier`)
    # and is configured there (reward_service/config.json, prompt.md): limits, model, concurrency. Editing those files updates it on the fly.
    verifier_url: str = "http://verifier:8090"
    verifier_timeout_s: float = 1800.0
    agent_max_cost_usd: float = 5.0        # trainer-side cap on the agent's spend for this run (groups beyond it are dropped, never guessed)
    # --- reward weights (docstring of reward_v2.py)
    w_verified: float = 1.0
    w_text_equiv: float = 0.7
    w_near: float = 0.25
    w_track: float = 0.30
    w_fmt: float = 0.10
    unverified_cap: float = 0.5
    gate_v: float = 0.7
    gate_p: float = 0.5
    gate_new_p: float = 0.6
    judge_lambda0: float = 0.15
    judge_fade_updates: int = 600
    rho_window: int = 400
    rho_default: float = 0.5
    w_new: float = 0.25
    new_floor: float = 0.35
    w_pattern: float = 0.05
    reward_max: float = 2.0
    accept_boxed: bool = True            # read the answer from \\boxed{}/Final Answer too, not only from an **Answer:** line
    think_credit: float = 0.4            # v for a reply-less candidate whose thinking ends in the exact reference answer (the right idea, unclosed)
    w_short: float = 0.10                # among verified candidates of a group: bonus for the shortest, 0 for the longest
    trunc_penalty: float = 0.5
    negative_beta: float = 1.0
    flat_group_eps: float = 0.05
    archive_path: str | None = "/runs/archive_{run}.json"


class RewardV3Cfg(Strict):
    """Reward service v3 (reward_service/reward.py, weights in reward_service/reward_config.json): the trainer sends raw groups, the service returns rewards."""
    enabled: bool = False
    url: str = "http://verifier:8090"
    timeout_s: float = 900.0
    max_cost_usd: float = 10.0            # stop cleanly (checkpoint saved) when the service's estimated spend reaches this
    negative_beta: float = 1.0
    flat_group_eps: float = 0.05


class MemoryCfg(Strict):
    training_ceiling_bytes: int = 80_000_000_000
    min_host_available_bytes: int = 20_000_000_000
    sample_interval_s: float = 1.0


class Config(Strict):
    schema_version: Literal[1] = 1
    model: ModelCfg = Field(default_factory=ModelCfg)
    rollout: RolloutCfg = Field(default_factory=RolloutCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    judge: JudgeCfg = Field(default_factory=JudgeCfg)
    reward: RewardCfg = Field(default_factory=RewardCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    reward_v2: RewardV2Cfg = Field(default_factory=RewardV2Cfg)   # not part of the resume fingerprint
    reward_v3: RewardV3Cfg = Field(default_factory=RewardV3Cfg)   # not part of the resume fingerprint

    @model_validator(mode="after")
    def _check(self) -> "Config":
        t = self.train
        if t.generations_per_prompt < 2:
            raise ValueError("generations_per_prompt must be >= 2 for group-relative advantages")
        if t.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be >= 1")
        if not 0 < self.rollout.gpu_memory_utilization < 1:
            raise ValueError("rollout.gpu_memory_utilization must be in (0, 1)")
        return self

    def fingerprint(self) -> str:
        """Hash of everything that must not change across a resume."""
        keep = {
            "model": self.model.model_dump(),
            "judge": {k: v for k, v in self.judge.model_dump().items() if k in ("hf_id", "revision", "rubric_path")},
            "reward": self.reward.model_dump(),
            "data": self.data.model_dump(),
            "train": {k: v for k, v in self.train.model_dump().items()
                      if k in ("prompts_per_update", "generations_per_prompt", "max_completion_tokens", "seed")},
        }
        return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(raw)
