# tauon.py
# -----------------------------------------------------------------------------
# Tauon – a light‑weight optimiser for neural‑network weight matrices stored
#         as low‑rank factors (W = L @ R.T), derived from a nuclear‑norm
#         regularisation viewpoint.
#
# Derivation summary
# ------------------
#   *   Nuclear‑norm surrogate:  ||W||_*  ≈ ½ (||L||_F² + ||R||_F²)
#   *   Objective:   L(W) + λ/2 (||L||_F² + ||R||_F²)
#   *   Exact factor gradients:
#           ∇_L Φ =  G @ R + λ L
#           ∇_R Φ =  Gᵀ @ L + λ R
#   *   Update (SGD + optional momentum/Nesterov):
#           L ← L − η · update(∇_L Φ)
#           R ← R − η · update(∇_R Φ)
#
# The implementation below mirrors the structure of `muon.py`, minus the
# Newton–Schulz orthogonalisation / distributed buffering because Tauon
# requires neither.  Pass only the *factor* parameters to Tauon – 2‑D tensors.
#
# Example
# -------
#     factors = [module.L, module.R, ...]        # every factor you created
#     opt = Tauon(factors, lr=2e‑2, weight_decay=2e‑2,
#                 momentum=0.9, nesterov=True)
#     ...
#     opt.step()
#
# Notes
# -----
#   •  Do **not** use Tauon on embeddings, biases, or any ≤1‑D parameters.
#      Keep those on AdamW / SGD as usual.
#   •  If you split 4‑D conv kernels into (out·kh·kw, in) then factorise,
#      Tauon works just the same.
# -----------------------------------------------------------------------------

from __future__ import annotations
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class Tauon(Optimizer):
    r"""Tauon ‑ a nuclear‑norm motivated optimiser that updates low‑rank factors.

    Args:
        params (iterable):      the *factor* parameters (each a 2‑D tensor)
        lr (float, optional):   learning rate η                                    (default: 0.02)
        weight_decay (float):   λ – strength of the nuclear‑norm surrogate         (default: 0.02)
        momentum (float):       momentum factor β                                  (default: 0.0 = off)
        nesterov (bool):        whether to use Nesterov momentum                   (default: False)

    Warning:
        • Pass **only** 2‑D factor parameters to this optimiser.
        • All non‑factor parameters (vectors, scalars, …) should live in a
          separate optimiser such as AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.02,
        momentum: float = 0.0,
        nesterov: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum == 0.0:
            raise ValueError("Nesterov momentum requires a non‑zero momentum")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        r"""Performs a single optimisation step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.  Not used by Tauon, present for API compat.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            wd: float = group["weight_decay"]
            momentum: float = group["momentum"]
            nesterov: bool = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad: Tensor = p.grad

                # apply nuclear‑norm surrogate term:  ∇Φ = ∇L + λ L     (or R)
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                if momentum != 0.0:
                    state = self.state[p]

                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf: Tensor = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                # gradient descent step
                p.add_(grad, alpha=-lr)

        return loss
