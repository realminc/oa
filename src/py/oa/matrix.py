"""Stateless Matrix operations matching Rust :mod:`oa::matrix`."""

from __future__ import annotations

from ._gen.matrix import *
from ._gen.matrix import __all__ as _GEN_ALL
from ._native import (
	Engine,
	Matrix,
	MoeExpertPlan,
	matrix_full as full,
	matrix_ones as ones,
	matrix_set_rng_seed as set_rng_seed,
)

__all__ = _GEN_ALL + [
	"Engine",
	"Matrix",
	"MoeExpertPlan",
	"full",
	"ones",
	"set_rng_seed",
]
