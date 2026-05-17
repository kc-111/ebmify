"""Planner adapters used by ``eval_wm.py``.

Wraps swm's built-in solvers (``CEMSolver`` / ``iCEMSolver`` / ``MPPI``)
and ``WorldModelPolicy`` behind a single ``Planner`` protocol so the eval
loop stays planner-agnostic. The ``EbmifyGradientPlanner`` stub marks
the future hook for plugging in
``src/ebmify/sampler/samadams.py`` over actions; it's not implemented in
this migration PR per plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Planner(Protocol):
    """Adapter shape used by the eval loop."""

    name: str

    def build_policy(self, *, model: Any, plan_cfg: Any,
                      transform: dict[str, Any], process: dict[str, Any]) -> Any:
        """Return an ``swm.policy.*`` instance wired up against ``model``."""
        ...


@dataclass
class SwmCEMPlanner:
    name: str = "cem"
    num_samples: int = 300
    n_steps: int = 30
    topk: int = 30
    var_scale: float = 1.0
    seed: int = 42
    device: str = "cuda"

    def build_policy(self, *, model, plan_cfg, transform, process):
        import stable_worldmodel as swm

        solver = swm.solver.CEMSolver(
            model=model,
            num_samples=self.num_samples,
            n_steps=self.n_steps,
            topk=self.topk,
            var_scale=self.var_scale,
            seed=self.seed,
            device=self.device,
        )
        return swm.policy.WorldModelPolicy(
            solver=solver, config=plan_cfg, process=process, transform=transform,
        )


@dataclass
class SwmICEMPlanner:
    name: str = "icem"
    num_samples: int = 300
    n_steps: int = 30
    topk: int = 30
    var_scale: float = 1.0
    seed: int = 42
    device: str = "cuda"

    def build_policy(self, *, model, plan_cfg, transform, process):
        import stable_worldmodel as swm

        solver = swm.solver.iCEMSolver(
            model=model,
            num_samples=self.num_samples,
            n_steps=self.n_steps,
            topk=self.topk,
            var_scale=self.var_scale,
            seed=self.seed,
            device=self.device,
        )
        return swm.policy.WorldModelPolicy(
            solver=solver, config=plan_cfg, process=process, transform=transform,
        )


@dataclass
class SwmMPPIPlanner:
    name: str = "mppi"
    num_samples: int = 300
    n_steps: int = 30
    lambda_: float = 1.0
    var_scale: float = 1.0
    seed: int = 42
    device: str = "cuda"

    def build_policy(self, *, model, plan_cfg, transform, process):
        import stable_worldmodel as swm

        solver = swm.solver.MPPI(
            model=model,
            num_samples=self.num_samples,
            n_steps=self.n_steps,
            lambda_=self.lambda_,
            var_scale=self.var_scale,
            seed=self.seed,
            device=self.device,
        )
        return swm.policy.WorldModelPolicy(
            solver=solver, config=plan_cfg, process=process, transform=transform,
        )


@dataclass
class EbmifyGradientPlanner:
    """Stub: Langevin sampler over actions with leverage-as-energy.

    Hook for the follow-up PR that plugs ``src/ebmify/sampler/samadams.py``
    in as a planning solver, using ``feature_leverage(...)`` of a world
    model's predictor output as the energy. Intentionally raises so the
    eval CLI lists the option but signals it's not implemented yet.
    """
    name: str = "ebmify_gradient"

    def build_policy(self, *, model, plan_cfg, transform, process):
        raise NotImplementedError(
            "EbmifyGradientPlanner is not implemented yet — this is the "
            "future hook for leverage-as-EBM planning over actions. See "
            "src/ebmify/sampler/samadams.py for the sampler we'll wrap."
        )


def get(name: str, **overrides) -> Planner:
    """Return a planner adapter by short name (cem|icem|mppi|ebmify)."""
    table = {
        "cem": SwmCEMPlanner,
        "icem": SwmICEMPlanner,
        "mppi": SwmMPPIPlanner,
        "ebmify": EbmifyGradientPlanner,
    }
    if name not in table:
        raise ValueError(f"unknown planner {name!r}; choose from {sorted(table)}")
    return table[name](**overrides)


__all__ = [
    "Planner",
    "SwmCEMPlanner",
    "SwmICEMPlanner",
    "SwmMPPIPlanner",
    "EbmifyGradientPlanner",
    "get",
]
