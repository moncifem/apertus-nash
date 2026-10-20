"""Step-aware Dr. GRPO loss for one trajectory (summing over trajectories == batch loss).

L_out  = -sum_i sum_t m_it * S(rho_it, A_i) / (B * L_cap)
L_proc = -(1/B) sum_i sum_t w_it * S(rho_it, q_it)       (w from rewards.process_token_targets)
L      = L_out + lambda_process * L_proc
S(rho, a) = min(rho * a, clip(rho, 1 - eps_low, 1 + eps_high) * a)
"""
from __future__ import annotations

import torch


def clipped_surrogate(ratio: torch.Tensor, adv: torch.Tensor, eps_low: float, eps_high: float) -> torch.Tensor:
    return torch.minimum(ratio * adv, ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * adv)


def trajectory_loss(
    logp: torch.Tensor,            # [L] current policy log p of sampled completion tokens (grad)
    old_logp: torch.Tensor,        # [L] detached
    advantage: float,              # outcome advantage A_i (scalar)
    out_mask: torch.Tensor,        # [L] 1 for tokens that receive the outcome term
    q_tok: torch.Tensor | None,    # [L] step reward per token
    w_tok: torch.Tensor | None,    # [L] process weight per token
    *,
    B: int,
    L_cap: int,
    lam_proc: float,
    eps_low: float,
    eps_high: float,
):
    ratio = torch.exp(logp - old_logp.detach())
    adv = torch.full_like(logp, float(advantage))
    s_out = clipped_surrogate(ratio, adv, eps_low, eps_high)
    l_out = -(s_out * out_mask).sum() / (B * L_cap)
    if lam_proc > 0 and q_tok is not None and w_tok is not None:
        s_proc = clipped_surrogate(ratio, q_tok, eps_low, eps_high)
        l_proc = -(s_proc * w_tok).sum() / B
    else:
        l_proc = logp.new_zeros(())
    with torch.no_grad():
        clipped = ((ratio < 1 - eps_low) | (ratio > 1 + eps_high)).float().mean()
    return l_out + lam_proc * l_proc, l_out.detach(), l_proc.detach(), clipped
