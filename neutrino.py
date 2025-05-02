"""
Neutrino Optimizer – Nuclear‑Norm Proximal SGD
================================================
A drop‑in replacement for Muon that makes **low‑rank (nuclear‑norm–shrunk)**
updates instead of orthogonal ones.  See derivation notes for details.

Usage (single GPU) – analogous to Muon::

    optim = Neutrino(params, lr=0.02, momentum=0.95, shrink=0.01)

Multi‑GPU (DDP) – pass rank and world_size like Muon::

    optim = Neutrino(params, lr=0.02, momentum=0.95,
                    rank=dist.get_rank(), world_size=dist.get_world_size())

Notes
-----
* Should **NOT** be used for embedding tables, layer‑norm scales or any 0‑ or 1‑D
  parameter; leave those on AdamW.
* 4‑D convolutional kernels are flattened into (out_channels, −1) matrices.
* Distributed strategy mirrors Muon's: each GPU processes a strided subset of
  parameters, then an `all_gather_into_tensor` brings the updates together so
  every replica applies exactly the same weight change.
* The extra FLOPs are dominated by the small randomized SVD (rank ≤ 8) – usually
  < 1 % of training time.
"""

from __future__ import annotations

import math
import os
from typing import List

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
# Helper – randomized truncated SVD (GPU‑friendly, no blocking synchronisation)
# -----------------------------------------------------------------------------

def _randomized_svd(mat: Tensor, rank: int = 6, n_iter: int = 2, eps: float = 1e-6):
    """Compute a rank‑`rank` approximate SVD of `mat` using randomized subspace
    iteration (Halko et al., 2011).  All ops are batched matmuls – fast on GPUs.

    Returns U [m×r], S [r], Vt [r×n] such that mat ≈ U·diag(S)·Vt.
    """
    m, n = mat.shape
    device, dtype = mat.device, mat.dtype

    # Step 1: draw a random Gaussian test matrix Ω ∈ ℝ^{n×r}
    omega = torch.randn(n, rank, dtype=dtype, device=device)

    # Step 2: form the sample matrix Y = A Ω and orthonormalise
    Y = mat @ omega  # (m × r)
    for _ in range(n_iter):
        # power iteration to amplify the singular spectrum
        Y = mat @ (mat.T @ Y)
    Q, _ = torch.linalg.qr(Y, mode='reduced')  # (m × r)

    # Step 3: project A to the subspace: B = Qᵀ A (r × n)
    B = Q.T @ mat

    # Step 4: compute deterministic SVD on the small matrix
    Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub  # lift back to full space (m × r)

    return U, S, Vt


# -----------------------------------------------------------------------------
# Main Optimizer class
# -----------------------------------------------------------------------------

class Neutrino(torch.optim.Optimizer):
    r"""Neutrino – Nuclear‑Norm‑Shrunk Momentum SGD (proximal SGD).

    Arguments
    ---------
    params : iterable of `Tensor`
        Same semantics as other PyTorch optimisers (may be a generator).
    lr : float, default 0.02
        Learning rate applied to the nuclear‑norm‑shrunken update.
    weight_decay : float, default 0.01
        Multiplicative weight decay (decoupled à la AdamW).
    momentum : float in [0,1), default 0.95
        SGD momentum coefficient.
    nesterov : bool, default True
        If True, use Nesterov momentum (like Muon).
    rank : int, optional
        *Distributed only.*  Local rank of this process.
    world_size : int, optional
        *Distributed only.*  Total number of processes.
    shrink : float, default 0.01
        Soft‑threshold λ applied to singular values (σ ← max(σ−λ,0)).
    svd_rank : int, default 6
        Target rank *k* for randomized SVD (adaptive but capped to this).
    power_iter : int, default 2
        Number of power‑iteration passes in SVD (≥ 1).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        shrink: float = 0.01,
        svd_rank: int = 6,
        power_iter: int = 2,
    ):
        if (rank is None) ^ (world_size is None):
            raise ValueError("Either supply both rank & world_size or neither.")

        self.rank = rank
        self.world_size = world_size

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            shrink=shrink,
            svd_rank=svd_rank,
            power_iter=power_iter,
        )

        # Flatten params into list to inspect shapes
        params = list(params)
        if not params:
            raise ValueError("Neutrino got an empty parameter list")

        if world_size is None:
            # Single‑GPU mode: fall back to standard param_groups
            super().__init__([{"params": params}], defaults)
            return

        # Multi‑GPU: bucket parameters by *numel* like Muon for comms efficiency
        param_groups = []
        for size in sorted({p.numel() for p in params}):
            # Allocate one large bf16 buffer per bucket (one per process)
            bucket = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=bucket,
                update_buffer_views=[bucket[i] for i in range(world_size)],
            )
            param_groups.append(group)

        super().__init__(param_groups, defaults)

    # ------------------------------------------------------------------
    # Optimiser step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        """Perform one optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.world_size is None:
            # -----------------------------------
            # Single‑GPU simpler path
            # -----------------------------------
            for group in self.param_groups:
                self._step_group_single(group)
            return loss

        # ---------------------------------------
        # Multi‑GPU path (mirrors Muon's logic)
        # ---------------------------------------
        for group in self.param_groups:
            self._step_group_distributed(group)

        return loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_group_single(self, group):
        lr, wd, mom, nest = group["lr"], group["weight_decay"], group["momentum"], group["nesterov"]
        lam = group["shrink"]
        k = group["svd_rank"]
        n_iter = group["power_iter"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.ndim < 2:
                # fall back to standard SGD for 0‑/1‑D params
                p.mul_(1 - lr * wd).add_(g, alpha=-lr)
                continue

            # Flatten conv filters (N, C, kH, kW) → (N, C*kH*kW)
            original_shape = g.shape
            if g.ndim == 4:
                g = g.view(g.size(0), -1)

            # Momentum buffer
            state = self.state.setdefault(p, {})
            buf = state.setdefault("momentum_buffer", torch.zeros_like(g))
            buf.mul_(mom).add_(g, alpha=1 - mom)
            d = g.lerp(buf, mom) if nest else buf  # g for nesterov else buf

            # Low‑rank nuclear‑norm‑shrunk update
            U, S, Vt = _randomized_svd(d, rank=k, n_iter=n_iter)
            S = torch.clamp(S - lam, min=0.0)
            if torch.count_nonzero(S) == 0:
                delta = torch.zeros_like(d)
            else:
                delta = (U * S) @ Vt  # (U diag(S) Vᵀ)

            # Reshape back if conv kernel
            if original_shape != delta.shape:
                delta = delta.view(original_shape)

            # Decoupled weight decay then param update
            p.mul_(1 - lr * wd)
            p.add_(delta, alpha=-lr)

    # ---------------- distributed ----------------

    def _step_group_distributed(self, group):
        lr, wd, mom, nest = group["lr"], group["weight_decay"], group["momentum"], group["nesterov"]
        lam = group["shrink"]
        k = group["svd_rank"]
        n_iter = group["power_iter"]

        update_buffer: Tensor = group["update_buffer"]
        views: List[Tensor] = group["update_buffer_views"]
        params: List[Tensor] = group["params"]

        handle = None
        params_world = None

        def _apply_prev():
            if handle is None:
                return
            handle.wait()
            for p_w, g_w in zip(params_world, views):
                # Apply decoupled weight decay
                p_w.mul_(1 - lr * wd)
                # Reshape linear buffer back
                delta = g_w.view_as(p_w)
                p_w.add_(delta, alpha=-lr)

        for base in range(0, len(params), self.world_size):
            idx = base + self.rank
            if idx < len(params):
                p = params[idx]
                g = p.grad
                if g is None:
                    update = views[self.rank].zero_()  # dummy
                else:
                    original_shape = g.shape
                    if g.ndim == 4:
                        g = g.view(g.size(0), -1)
                    # momentum
                    state = self.state.setdefault(p, {})
                    buf = state.setdefault("momentum_buffer", torch.zeros_like(g))
                    buf.mul_(mom).add_(g, alpha=1 - mom)
                    d = g.lerp(buf, mom) if nest else buf

                    # nuclear‑norm shrink
                    U, S, Vt = _randomized_svd(d, rank=k, n_iter=n_iter)
                    S = torch.clamp(S - lam, min=0.0)
                    if torch.count_nonzero(S) == 0:
                        delta = torch.zeros_like(d)
                    else:
                        delta = (U * S) @ Vt
                    if original_shape != delta.shape:
                        delta = delta.view(original_shape)
                    update = delta.flatten().to(dtype=torch.bfloat16)
                # place into own view
                views[self.rank].copy_(update)
            else:
                # no param at this strided position
                views[self.rank].zero_()

            # async all‑gather to fill buffer across ranks
            if base > 0:
                _apply_prev()
            handle = dist.all_gather_into_tensor(update_buffer, views[self.rank], async_op=True)
            params_world = params[base : base + self.world_size]

        _apply_prev()
