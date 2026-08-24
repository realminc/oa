#!/usr/bin/env python3
"""Tests for OA's reinforcement-learning Python surface."""

from __future__ import annotations

import sys

import pytest


import oa_python_test  # noqa: F401 - bootstraps source builds

import oa
core = oa.FnMatrix
ml = oa
runtime = oa


@pytest.fixture(scope="session")
def engine():
	if not runtime.initComputeEngine():
		pytest.fail("GPU profile requires an initialized OA Vulkan engine")
	yield
	shutdown = getattr(runtime, "shutdownComputeEngine", None)
	if shutdown is not None:
		shutdown()


def _u8(values: list[int], shape: list[int]):
	return core.fromBytes(values, shape, oa.ScalarType.UInt8)


def test_environment_contract_is_available():
	observation = ml.EnvironmentSpace.box("observation", [4])
	action = ml.EnvironmentSpace.discrete("action", 2)
	reward = ml.EnvironmentSpace.box("reward", [], minimum=0.0, maximum=1.0)
	terminated = ml.EnvironmentSpace.binary("terminated")
	truncated = ml.EnvironmentSpace.binary("truncated")

	spec = ml.EnvironmentSpec()
	spec.observation = observation
	spec.action = action
	spec.reward = reward
	spec.terminated = terminated
	spec.truncated = truncated
	spec.validateDefinition()

	assert observation.shape == [4]
	assert action.kind == ml.EnvironmentSpaceKind.Discrete
	assert action.cardinality == 2


def test_gae_policy_ppo_and_rollout(engine):
	reward = core.fromFloats([1.0, 10.0, 2.0, 20.0], [2, 2])
	value = core.fromFloats([0.0, 0.0, 0.0, 0.0], [2, 2])
	nextValue = core.fromFloats([0.0, 0.0, 0.0, 5.0], [2, 2])
	terminated = _u8([0, 0, 1, 0], [2, 2])
	truncated = _u8([0, 0, 0, 1], [2, 2])

	gaeConfig = ml.GaeConfig()
	gaeConfig.gamma = 1.0
	setattr(gaeConfig, "lambda", 1.0)
	gae = oa.FnAdvantage.gae(
		reward, value, nextValue, terminated, truncated, gaeConfig
	)
	assert gae.isValid()

	logits = core.fromFloats([1.0, 2.0, -1.0, 0.5, 0.5, 0.5], [2, 3])
	action = core.fromInt32([1, 0], [2])
	policyValue = core.fromFloats([0.25, -0.75], [2])
	policy = oa.FnPolicy.evaluateCategorical(logits, action, policyValue)
	assert policy.isValid()

	ppo = oa.FnLoss.ppo(
		policy.logProbability,
		policy.logProbability,
		core.fromFloats([1.0, -1.0], [2]),
		policy.value,
		core.fromFloats([0.5, -0.5], [2]),
		policy.entropy,
	)
	assert ppo.isValid()

	rolloutConfig = ml.RolloutConfig()
	rolloutConfig.time = 1
	rolloutConfig.environments = 2
	rolloutConfig.observationShape = [4]
	rollout = ml.RolloutBuffer.create(rolloutConfig)
	rollout.append(
		core.fromFloats([0.0] * 8, [2, 4]),
		action,
		core.fromFloats([1.0, 1.0], [2]),
		policy.value,
		core.fromFloats([0.0, 0.0], [2]),
		policy.logProbability,
		_u8([0, 0], [2]),
		_u8([0, 0], [2]),
	)
	rollout.finalize()

	assert core.copyToHost(gae.advantage) == pytest.approx([3.0, 35.0, 2.0, 25.0])
	assert ppo.totalLoss.numElements() == 1
	assert rollout.isFull()
	assert rollout.isFinalized()
	assert rollout.batch.advantage.shape() == [1, 2]

	optimizer = ml.OptimizerNoOp()
	trainingConfig = ml.ItRolloutTrainingConfig()
	trainingConfig.rollouts = 1
	trainingConfig.horizon = 1
	trainingConfig.environments = 2
	trainingConfig.updateEpochs = 1
	training = ml.ItRolloutTraining(optimizer, trainingConfig)

	training.beginRollout(rollout)
	rollout.append(
		core.fromFloats([0.0] * 8, [2, 4]),
		action,
		core.fromFloats([1.0, 1.0], [2]),
		policy.value,
		core.fromFloats([0.0, 0.0], [2]),
		policy.logProbability,
		_u8([0, 0], [2]),
		_u8([0, 0], [2]),
	)
	training.finalizeRollout(rollout)
	assert training.phase() == ml.RolloutTrainingPhase.Update

	# RL composes the ordinary optimizer-step iterator, so it consumes the exact
	# same typed live-control session as supervised training.
	session = ml.TrainingSession(training.updateLoop())
	pauseSequence = session.pause()
	assert not training.beginUpdate()
	assert session.state() == ml.TrainingState.Paused
	results = session.takeResults()
	assert results[-1].sequence == pauseSequence
	assert results[-1].success
	session.resume()
	assert training.beginUpdate()
	training.nextUpdate(ppo.totalLoss)
	assert training.isDone()
	training.finish()
	assert session.latestSnapshot().state == ml.TrainingState.Completed


def test_continuous_replay_off_policy_and_transforms(engine):
	mean = core.fromFloats([0.0, 0.5, -0.5, 1.0], [2, 2])
	logStddev = core.fromFloats([-1.0] * 4, [2, 2])
	value = core.fromFloats([0.0, 0.0], [2])
	continuous = oa.FnPolicy.sampleTanhNormal(
		mean, logStddev, value, minimum=-2.0, maximum=2.0, seed=717
	)
	assert continuous.isValid()

	normalized = oa.FnEnvironment.normalizeObservation(
		core.fromFloats([1.0, 4.0, 5.0, -8.0], [2, 2]),
		core.fromFloats([1.0, 0.0], [2]),
		core.fromFloats([2.0, 2.0], [2]),
		clip=3.0,
	)
	scaled = oa.FnEnvironment.scaleAction(
		core.fromFloats([-2.0, 0.0, 2.0], [3]), -1.0, 1.0, 0.0, 10.0
	)
	clipped = oa.FnEnvironment.clipReward(
		core.fromFloats([-3.0, 0.25, 4.0], [3])
	)

	replayConfig = ml.ReplayConfig()
	replayConfig.capacity = 4
	replayConfig.observationShape = [2]
	replayConfig.actionShape = []
	replayConfig.actionDtype = oa.ScalarType.Int32
	replay = ml.ReplayBuffer.create(replayConfig)
	terminated = _u8([0, 1], [2])
	truncated = _u8([1, 0], [2])
	replay.append(
		core.fromFloats([0.0, 1.0, 2.0, 3.0], [2, 2]),
		core.fromInt32([0, 1], [2]),
		core.fromFloats([1.0, 2.0, 3.0, 4.0], [2, 2]),
		core.fromFloats([1.0, 2.0], [2]),
		terminated,
		truncated,
	)
	sampled = replay.sample(3, 9917)
	assert sampled.isValid()

	dqn = oa.FnLoss.dqn(
		core.fromFloats([1.0, 2.0, 4.0, 3.0], [2, 2]),
		core.fromInt32([1, 0], [2]),
		core.fromFloats([1.0, 2.0], [2]),
		core.fromFloats([5.0, 6.0, 7.0, 8.0], [2, 2]),
		terminated,
		truncated,
	)
	sac = oa.FnLoss.sacCritic(
		core.fromFloats([1.0, 2.0], [2]),
		core.fromFloats([1.5, 2.5], [2]),
		core.fromFloats([1.0, 2.0], [2]),
		core.fromFloats([5.0, 6.0], [2]),
		core.fromFloats([4.0, 7.0], [2]),
		core.fromFloats([-0.5, -0.25], [2]),
		terminated,
		truncated,
	)

	assert dqn.isValid()
	assert sac.isValid()
	assert replay.size() == 2
	assert sampled.observation.shape() == [3, 2]
	assert core.copyToHost(normalized) == pytest.approx([0.0, 2.0, 2.0, -3.0])
	assert core.copyToHost(scaled) == pytest.approx([0.0, 5.0, 10.0])
	assert core.copyToHost(clipped) == pytest.approx([-1.0, 0.25, 1.0])


@pytest.mark.oa_external
def test_optional_gymnasium_adapter_preserves_step_boundaries(engine):
	import gymnasium as gym
	from oa.gymnasium import GymnasiumAdapter

	adapter = GymnasiumAdapter(gym.make("CartPole-v1"))
	observation, info = adapter.reset(seed=717)
	assert observation.shape() == [1, 4]
	assert isinstance(info, dict)
	transition = adapter.step(core.fromInt32([1], [1]))
	assert transition.observation.shape() == [1, 4]
	assert transition.nextObservation.shape() == [1, 4]
	assert transition.reward.shape() == [1]
	assert transition.terminated.shape() == [1]
	assert transition.truncated.shape() == [1]
	adapter.environment.close()


if __name__ == "__main__":
	raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
