"""Adaptive-stepsize samplers for non-trivial energy landscapes.

The flagship method here is :func:`samadams_sample`, an Adam-inspired
adaptive-stepsize Langevin sampler from Leimkuhler, Lohmann & Whalley
(2025) "A Langevin Sampling Algorithm Inspired by the Adam Optimizer"
(arXiv:2504.18911). Use it whenever a fixed-stepsize Langevin run gets
stuck choosing between "too small to descend the landscape" and "too
large to remain stable in steep regions" -- e.g. high-D leverage-energy
descent from random-noise initial conditions.
"""

from .samadams import (
    SamAdamsConfig,
    samadams_sample,
    psi1,
    psi2,
)

__all__ = [
    "SamAdamsConfig",
    "samadams_sample",
    "psi1",
    "psi2",
]
