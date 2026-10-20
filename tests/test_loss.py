import pytest

torch = pytest.importorskip("torch")

from rlaif.loss import trajectory_loss  # noqa: E402
from rlaif.rewards import process_token_targets  # noqa: E402

KW = dict(L_cap=16, eps_low=0.2, eps_high=0.28)


def _grad(adv, q, w, lam, B=1, n=10):
    logp = torch.full((n,), -1.0, requires_grad=True)
    loss, *_ = trajectory_loss(logp, logp.detach(), adv, torch.ones(n),
                               None if q is None else torch.tensor(q), None if w is None else torch.tensor(w),
                               B=B, lam_proc=lam, **KW)
    loss.backward()
    return logp.grad


def test_process_gradient_signs_and_answer_tokens():
    # tokens 0-3: wrong step, 4-7: good step, 8-9: answer
    q, w = process_token_targets([(0, 4), (4, 8)], [-1.0, 1.0], 10)
    g = _grad(0.0, q, w, lam=0.2)
    assert (g[:4] > 0).all()   # descent lowers log p of the wrong step
    assert (g[4:8] < 0).all()  # descent raises log p of the good step
    assert (g[8:] == 0).all()  # answer tokens: no process term


def test_outcome_only_when_lambda_zero():
    q, w = process_token_targets([(0, 4)], [-1.0], 10)
    g = _grad(0.5, q, w, lam=0.0)
    assert torch.allclose(g, torch.full((10,), -0.5 / 16))


def test_accumulation_equals_batched():
    torch.manual_seed(0)
    lps = [torch.randn(n) for n in (5, 9, 7)]
    advs = [0.3, -0.7, 0.4]
    B = 3
    total = 0.0
    for lp, a in zip(lps, advs):
        x = lp.clone().requires_grad_(True)
        old = (lp + 0.05 * torch.randn_like(lp)).detach()
        loss, *_ = trajectory_loss(x, old, a, torch.ones_like(lp), None, None, B=B, lam_proc=0.0, **KW)
        total += loss.item()
        lp.old = old
    # batched reference: one padded tensor
    L = max(len(l) for l in lps)
    cur = torch.zeros(B, L); old = torch.zeros(B, L); m = torch.zeros(B, L)
    for i, lp in enumerate(lps):
        cur[i, :len(lp)] = lp; old[i, :len(lp)] = lp.old; m[i, :len(lp)] = 1
    ratio = torch.exp(cur - old)
    A = torch.tensor(advs)[:, None]
    s = torch.minimum(ratio * A, ratio.clamp(0.8, 1.28) * A)
    ref = -(s * m).sum() / (B * 16)
    assert total == pytest.approx(ref.item(), rel=1e-5)
