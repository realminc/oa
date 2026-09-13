"""Stateless Matrix operations matching Rust ``oa::matrix``."""

from __future__ import annotations

from collections.abc import Sequence

from ._native import Engine, Matrix

__all__ = ["add", "full", "mat_mul_nt", "ones"]


def ones(engine: Engine, shape: Sequence[int]) -> Matrix:
	"""Create an FP32 Matrix filled with ones."""
	return engine._ones(shape)


def full(engine: Engine, shape: Sequence[int], value: float) -> Matrix:
	"""Create an FP32 Matrix filled with ``value``."""
	return engine._full(shape, value)


def add(left: Matrix, right: Matrix) -> Matrix:
	"""Add equal-shaped Matrices elementwise."""
	return left._add(right)


def mat_mul_nt(left: Matrix, right: Matrix) -> Matrix:
	"""Multiply ``[M,K]`` by donor-layout ``[N,K]`` into ``[M,N]``."""
	return left._mat_mul_nt(right)
