"""Optimizer binding and GPU execution contracts."""

from __future__ import annotations

import math

import pytest

import oa_python_test  # noqa: F401 - bootstraps source builds

import oa


@pytest.fixture(scope="module")
def engine():
	if not oa.initComputeEngine():
		pytest.fail("GPU profile requires an initialized OA Vulkan engine")
	yield
	oa.shutdownComputeEngine()


def test_muon_is_one_gpu_optimizer(engine) -> None:
	assert oa.ml.Muon is oa.Muon
	assert not hasattr(oa, "MuonAdamWConfig")
	assert not hasattr(oa, "OptimizerComposite")
	assert not hasattr(oa, "makeMuonAdamWOptimizer")

	linear = oa.Linear(4, 8, False)
	parameters = linear.parameters()
	assert len(parameters) == 1
	parameter = parameters[0]
	parameter.grad = oa.FnMatrix.ones(parameter.data.shape())

	optimizer = oa.Muon(parameters, lr=0.01, ns5Iters=2)
	optimizer.step()

	assert optimizer.getStep() == 1
	assert math.isfinite(parameter.data.at(0))
