"""SamAdams: Adam-inspired adaptive-stepsize Langevin sampler.

Reference: B. Leimkuhler, R. Lohmann, P. A. Whalley, "A Langevin Sampling
Algorithm Inspired by the Adam Optimizer", arXiv:2504.18911 (2025).

This is the overdamped variant from Appendix A (the generic
``Z·Phi·Z`` wrap), which lets us reuse a fixed-stepsize Euler-Maruyama
Langevin step inside two ``Z`` half-steps that update an auxiliary
scalar ``zeta`` per particle.

The pieces (paper eqs. 24-26, 34-38):

* ``zeta`` follows ``d zeta / d tau = -alpha * zeta + g(x)``, the
  Adam-style EMA of a monitor function ``g(x) = ||grad U(x)||^s / Omega``.
* The Sundman kernel ``psi(zeta)`` maps ``zeta`` to a stepsize
  multiplier; with the bounded kernels from eq. (25), the effective
  stepsize is constrained to ``dt in [m * dtau, M * dtau]``.
* Each iteration: half-step on ``zeta`` -> compute ``dt`` -> one
  fixed-stepsize Langevin step at that ``dt`` -> half-step on ``zeta``
  again. The two half-steps make the per-step weights ``mu_n =
  psi(zeta_n)`` accurate enough to use for ergodic-average
  reweighting (Theorem 1 in the paper).

The classic failure mode of constant-stepsize Langevin in high
dimensions is that one ``eta`` cannot satisfy two competing
constraints at once: small enough to avoid blowing up in steep
gradient regions, but large enough to actually move during the long
flat phases. SamAdams reads ``||grad U||`` and rescales ``dt``
accordingly -- shrinking near sharp features, dilating on plateaus.

Defaults follow Sec. 5 of the paper (s=2, Omega=1, m=0.1, M=10,
r=0.25, alpha=1, kernel=psi^(1)). Out of these, ``Omega`` is the only
hyperparameter that *must* be tuned to the problem -- it sets what
"large gradient" means. A reasonable starting point is
``Omega ~ median(||grad U(x_0)||^s)`` over the initial particles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------
# Sundman transform kernels (paper eq. 25)
# ----------------------------------------------------------------------

def psi1(zeta: torch.Tensor, m: float, M: float, r: float) -> torch.Tensor:
    """psi^(1)(zeta) = m * (zeta^r + M) / (zeta^r + m).

    Bounded in (m, M]: psi(0) = M, psi(infty) = m.
    """
    zr = zeta.clamp_min(0.0).pow(r)
    return m * (zr + M) / (zr + m)


def psi2(zeta: torch.Tensor, m: float, M: float, r: float) -> torch.Tensor:
    """psi^(2)(zeta) = m * (zeta^r + M/m) / (zeta^r + 1).

    Same bounds as psi^(1) but a sharper transition near ``zeta = 1``;
    asymptotically psi^(2)(zeta) ~ m + (M*m - m^2)/zeta^r as zeta->inf.
    """
    zr = zeta.clamp_min(0.0).pow(r)
    return m * (zr + M / m) / (zr + 1.0)


_KERNELS: dict[str, Callable[..., torch.Tensor]] = {
    "psi1": psi1,
    "psi2": psi2,
}


# ----------------------------------------------------------------------
# Config + sampler
# ----------------------------------------------------------------------

@dataclass
class SamAdamsConfig:
    """Hyperparameters for the SamAdams sampler.

    Args:
        dtau:      Base (rescaled) stepsize ``Delta tau``. The actual
                   stepsize used is ``dt_n = psi(zeta_n) * dtau``,
                   bounded in ``[m * dtau, M * dtau]``.
        alpha:     Relaxation rate for ``zeta``. The EMA decay over a
                   single step is ``rho = exp(-alpha * dtau)``.
        s:         Power on the gradient norm in the monitor
                   ``g(x) = ||grad U||^s / Omega``. Use ``s=2`` to mimic
                   Adam's accumulation of squared gradients;
                   ``s=1`` is globally Lipschitz and slightly more
                   stable, but less aggressive.
        Omega:     Normalization of the monitor function. The
                   stationary value of ``zeta`` is roughly
                   ``E[||grad U||^s] / (alpha * Omega)``, so picking
                   ``Omega ~ median(||grad U(x_0)||^s)`` keeps the
                   running ``zeta`` of order one and the kernel
                   ``psi(zeta)`` near its midpoint.
        m, M:      Lower / upper multiplier on the stepsize.
                   ``dt`` is restricted to ``(m * dtau, M * dtau]``.
                   Typical: ``m=0.1, M=10`` (a 100x dilation range).
        r:         Sundman kernel power. Larger ``r`` makes ``psi``
                   more sensitive to ``zeta``. Paper uses
                   ``r in {0.25, 0.5, 1}``.
        kernel:    Sundman kernel name: ``"psi1"`` or ``"psi2"``.
        grad_clip: Optional per-particle clip on ``||grad U||`` for the
                   Langevin step (does NOT enter the monitor function;
                   the unclipped norm controls ``zeta``).
    """

    dtau: float = 1e-2
    alpha: float = 1.0
    s: float = 2.0
    Omega: float = 1.0
    m: float = 0.1
    M: float = 10.0
    r: float = 0.25
    kernel: str = "psi1"
    grad_clip: Optional[float] = None


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _grad_energy(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(E(x), grad_x sum_i E_i(x))`` for batched ``x``.

    ``E`` is a length-B vector (one scalar per particle), and the
    gradient w.r.t. ``x`` is per-particle because the particles do not
    interact in the energy.
    """
    x_in = x.detach().requires_grad_(True)
    E = energy_fn(x_in)
    grad = torch.autograd.grad(E.sum(), x_in)[0]
    return E.detach(), grad.detach()


def _monitor(grad: torch.Tensor, s: float, Omega: float) -> torch.Tensor:
    """Compute g(x) = ||grad U||^s / Omega, returned per particle [B]."""
    norm2 = grad.flatten(1).pow(2).sum(dim=-1)  # ||grad||^2, shape [B]
    if s == 2.0:
        return norm2 / Omega
    return norm2.clamp_min(1e-30).pow(s / 2.0) / Omega


def _kernel_fn(name: str) -> Callable[..., torch.Tensor]:
    if name not in _KERNELS:
        raise ValueError(
            f"Unknown SamAdams kernel '{name}'. Options: {list(_KERNELS)}"
        )
    return _KERNELS[name]


# ----------------------------------------------------------------------
# Public sampler
# ----------------------------------------------------------------------

def samadams_sample(
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    n_steps: int,
    temperature=1.0,
    dtau_schedule=None,
    config: SamAdamsConfig | None = None,
    project: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    record_every: int = 1,
    zeta_init: str = "g",
    log_every: int | None = None,
) -> dict:
    """Run overdamped SamAdams (the ``Z * Phi * Z`` scheme from App. A).

    The inner fixed-stepsize step is Euler-Maruyama on overdamped
    Langevin: ``x <- x - dt * grad U + sqrt(2 * dt * T) * xi`` with
    ``xi ~ N(0, I)``. SamAdams wraps it with two half-steps on the
    auxiliary scalar ``zeta`` (one per particle), which adapts ``dt``
    to local gradient magnitude.

    Args:
        energy_fn:   ``E: [B, D] -> [B]`` (or any shape; flattened
                     internally). Treated as ``U(x)`` for sampling
                     ``pi propto exp(-U / T)``.
        x0:          ``[B, *input_shape]`` initial particles. The
                     ``B`` dimension is the parallel-chain index.
        n_steps:     Number of integration steps.
        temperature: Either a scalar ``T`` (constant) or a callable
                     ``t -> T(t)`` for an annealing schedule.
        config:      :class:`SamAdamsConfig`. Defaults if ``None``.
        project:     Optional ``[B, *] -> [B, *]`` projection applied
                     after each Langevin step (e.g. clamp to ``[0, 1]``
                     for image pixels).
        record_every: Snapshot ``x``, energy, ``dt``, ``mu`` every
                     ``record_every`` steps (plus the final state).
        zeta_init:   ``"zero"`` -> ``zeta_0 = 0`` (so ``dt_0 = M *
                     dtau``); ``"g"`` -> ``zeta_0 = g(x_0) / alpha``
                     (the steady-state value if g were constant; gives
                     a problem-appropriate initial stepsize).

    Returns:
        Dict with keys:

        * ``x_final``  : ``[B, *]`` final positions on CPU.
        * ``traj``     : ``[T_rec, B, *]`` recorded positions.
        * ``energies`` : ``[T_rec, B]`` energy at each snapshot.
        * ``dts``      : ``[T_rec, B]`` adaptive stepsize at each
                         snapshot.
        * ``weights``  : ``[T_rec, B]`` reweighting weights
                         ``mu_n = psi(zeta_n)``.
        * ``temps``    : ``[n_steps]`` temperature schedule used.
    """
    if config is None:
        config = SamAdamsConfig()

    # Temperature schedule -> callable.
    if callable(temperature):
        T_fn: Callable[[int], float] = temperature
    else:
        T_const = float(temperature)
        T_fn = lambda _t: T_const  # noqa: E731

    # dtau schedule -> callable. Default = constant config.dtau.
    if dtau_schedule is None:
        dtau_const = float(config.dtau)
        dtau_fn: Callable[[int], float] = lambda _t: dtau_const  # noqa: E731
    elif callable(dtau_schedule):
        dtau_fn = dtau_schedule
    else:
        dtau_v = float(dtau_schedule)
        dtau_fn = lambda _t: dtau_v  # noqa: E731

    kernel = _kernel_fn(config.kernel)
    inv_alpha = 1.0 / config.alpha

    x = x0.detach().clone()
    device = x.device
    B = x.shape[0]
    rest_shape = x.shape[1:]

    # Initial Z half-step needs g(x_0).
    E0, grad0 = _grad_energy(energy_fn, x)
    g0 = _monitor(grad0, config.s, config.Omega)  # [B]
    if zeta_init == "zero":
        zeta = torch.zeros(B, device=device, dtype=x.dtype)
    elif zeta_init == "g":
        # Steady-state if g were held constant: zeta_* = g / alpha.
        zeta = g0 * inv_alpha
    else:
        raise ValueError(f"zeta_init must be 'zero' or 'g', got {zeta_init!r}")

    traj_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []
    dt_list: list[np.ndarray] = []
    mu_list: list[np.ndarray] = []
    temp_list: list[float] = []

    # We'll reuse the gradient computed at the *end* of one iteration
    # as the gradient for the *start* of the next (the paper's
    # observation in Appendix A: the post-step force evaluation
    # serves the next iteration's first Z half-step).
    grad_carry = grad0
    g_carry = g0
    E_carry = E0

    for t in range(n_steps):
        T_t = max(float(T_fn(t)), 0.0)
        temp_list.append(T_t)
        dtau_t = max(float(dtau_fn(t)), 0.0)
        rho_half_t = math.exp(-0.5 * config.alpha * dtau_t)

        # --- Z half-step using g(x_n).
        zeta = rho_half_t * zeta + inv_alpha * (1.0 - rho_half_t) * g_carry

        # --- Adaptive stepsize. Per-particle in [m * dtau, M * dtau].
        psi_z = kernel(zeta, config.m, config.M, config.r)  # [B]
        dt = (psi_z * dtau_t).clamp_min(0.0)

        # Snapshot pre-step (records the *upcoming* dt).
        if (t % record_every) == 0:
            traj_list.append(x.detach().cpu().numpy().copy())
            e_list.append(E_carry.cpu().numpy().copy())
            dt_list.append(dt.detach().cpu().numpy().copy())
            mu_list.append(psi_z.detach().cpu().numpy().copy())

        # --- Inner Phi step: Euler-Maruyama Langevin with stepsize dt.
        grad_step = grad_carry
        if config.grad_clip is not None:
            gn = grad_step.flatten(1).norm(dim=-1).clamp_min(1e-12)  # [B]
            scale = (gn.clamp_max(config.grad_clip) / gn).view(
                B, *([1] * len(rest_shape))
            )
            grad_step = grad_step * scale

        # Broadcast dt and noise scale over the rest dims.
        dt_b = dt.view(B, *([1] * len(rest_shape)))
        noise_scale = torch.sqrt(2.0 * dt_b * T_t)
        noise = noise_scale * torch.randn_like(x)
        x = x - dt_b * grad_step + noise
        if project is not None:
            x = project(x)

        # --- Z half-step using g(x_{n+1}). Reuse this gradient next iter.
        E_carry, grad_carry = _grad_energy(energy_fn, x)
        g_carry = _monitor(grad_carry, config.s, config.Omega)
        zeta = rho_half_t * zeta + inv_alpha * (1.0 - rho_half_t) * g_carry

        if log_every is not None and (t + 1) % log_every == 0:
            with torch.no_grad():
                e_med = float(E_carry.median().item())
                dt_med = float(dt.median().item())
                drift_med = float(
                    (x.detach() - x0).flatten(1).norm(dim=-1).median().item()
                ) / math.sqrt(int(np.prod(rest_shape)))
            print(
                f"  step {t+1:>6d}  E.med={e_med:.3e}  dt.med={dt_med:.3e}  "
                f"per-pixel drift.med={drift_med:.3e}",
                flush=True,
            )

    # Final snapshot.
    psi_z_final = kernel(zeta, config.m, config.M, config.r)
    dtau_final = max(float(dtau_fn(max(n_steps - 1, 0))), 0.0)
    traj_list.append(x.detach().cpu().numpy().copy())
    e_list.append(E_carry.cpu().numpy().copy())
    dt_list.append((psi_z_final * dtau_final).detach().cpu().numpy().copy())
    mu_list.append(psi_z_final.detach().cpu().numpy().copy())

    return {
        "x_final": x.detach().cpu(),
        "traj": np.stack(traj_list, axis=0),
        "energies": np.stack(e_list, axis=0),
        "dts": np.stack(dt_list, axis=0),
        "weights": np.stack(mu_list, axis=0),
        "temps": np.asarray(temp_list, dtype=np.float32),
    }
