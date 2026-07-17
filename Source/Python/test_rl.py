#!/usr/bin/env python3
"""Focused smoke coverage for OA's minimal categorical-PPO Python surface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _add_dev_paths() -> None:
    candidates: list[Path] = []
    if build_dir := os.getenv("OA_PYTHON_BUILD_DIR"):
        candidates.append(Path(build_dir).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "Build" / "Release",
            REPO_ROOT / "Build" / "Debug",
            REPO_ROOT / "build",
            REPO_ROOT / "Source" / "Python",
        ]
    )
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_add_dev_paths()

oa = pytest.importorskip("oa", reason="oa Python package is not importable")
core = oa.core
ml = oa.ml
runtime = oa.runtime


@pytest.fixture(scope="session")
def engine():
    if not runtime.OaInitComputeEngine():
        pytest.skip("OA compute engine could not initialize, likely no Vulkan device")
    yield
    shutdown = getattr(runtime, "OaShutdownComputeEngine", None)
    if shutdown is not None:
        shutdown()


def _u8(values: list[int], shape: list[int]):
    return core.FromBytes(values, shape, core.OaScalarType.UInt8)


def test_environment_contract_is_available():
    observation = ml.OaRlFieldSpec.Box("observation", [4])
    action = ml.OaRlFieldSpec.Discrete("action", 2)
    reward = ml.OaRlFieldSpec.Box("reward", [], minimum=0.0, maximum=1.0)
    terminated = ml.OaRlFieldSpec.Binary("terminated")
    truncated = ml.OaRlFieldSpec.Binary("truncated")

    spec = ml.OaRlEnvironmentSpec()
    spec.Observation = observation
    spec.Action = action
    spec.Reward = reward
    spec.Terminated = terminated
    spec.Truncated = truncated
    spec.ValidateDefinition()

    assert observation.Shape == [4]
    assert action.Kind == ml.OaRlSpaceKind.Discrete
    assert action.Cardinality == 2


def test_gae_policy_ppo_and_rollout(engine):
    reward = core.FromFloats([1.0, 10.0, 2.0, 20.0], [2, 2])
    value = core.FromFloats([0.0, 0.0, 0.0, 0.0], [2, 2])
    next_value = core.FromFloats([0.0, 0.0, 0.0, 5.0], [2, 2])
    terminated = _u8([0, 0, 1, 0], [2, 2])
    truncated = _u8([0, 0, 0, 1], [2, 2])

    gae_config = ml.OaGaeConfig()
    gae_config.Gamma = 1.0
    gae_config.Lambda = 1.0
    gae = ml.Gae(reward, value, next_value, terminated, truncated, gae_config)
    assert gae.IsValid()

    logits = core.FromFloats([1.0, 2.0, -1.0, 0.5, 0.5, 0.5], [2, 3])
    action = core.FromInt32([1, 0], [2])
    policy_value = core.FromFloats([0.25, -0.75], [2])
    policy = ml.EvaluateCategoricalPolicy(logits, action, policy_value)
    assert policy.IsValid()

    ppo = ml.Ppo(
        policy.LogProbability,
        policy.LogProbability,
        core.FromFloats([1.0, -1.0], [2]),
        policy.Value,
        core.FromFloats([0.5, -0.5], [2]),
        policy.Entropy,
    )
    assert ppo.IsValid()

    rollout_config = ml.OaRlRolloutConfig()
    rollout_config.Time = 1
    rollout_config.Environments = 2
    rollout_config.ObservationShape = [4]
    rollout = ml.OaRlRolloutBuffer.Create(rollout_config)
    rollout.Append(
        core.FromFloats([0.0] * 8, [2, 4]),
        action,
        core.FromFloats([1.0, 1.0], [2]),
        policy.Value,
        core.FromFloats([0.0, 0.0], [2]),
        policy.LogProbability,
        _u8([0, 0], [2]),
        _u8([0, 0], [2]),
    )
    rollout.Finalize()

    assert core.CopyToHost(gae.Advantage) == pytest.approx([3.0, 35.0, 2.0, 25.0])
    assert ppo.TotalLoss.NumElements() == 1
    assert rollout.IsFull()
    assert rollout.IsFinalized()
    assert rollout.Batch.Advantage.Shape() == [1, 2]

    optimizer = ml.OaOptimizerNoOp()
    training_config = ml.OaItRlTrainingConfig()
    training_config.Rollouts = 1
    training_config.Horizon = 1
    training_config.Environments = 2
    training_config.UpdateEpochs = 1
    training = ml.OaItRlTraining(optimizer, training_config)

    training.BeginRollout(rollout)
    rollout.Append(
        core.FromFloats([0.0] * 8, [2, 4]),
        action,
        core.FromFloats([1.0, 1.0], [2]),
        policy.Value,
        core.FromFloats([0.0, 0.0], [2]),
        policy.LogProbability,
        _u8([0, 0], [2]),
        _u8([0, 0], [2]),
    )
    training.FinalizeRollout(rollout)
    assert training.Phase() == ml.OaRlTrainingPhase.Update

    # RL composes the ordinary optimizer-step iterator, so it consumes the exact
    # same typed live-control session as supervised training.
    session = ml.OaTrainingSession(training.UpdateLoop())
    pause_sequence = session.Pause()
    assert not training.BeginUpdate()
    assert session.State() == ml.OaTrainingState.Paused
    results = session.TakeResults()
    assert results[-1].Sequence == pause_sequence
    assert results[-1].Success
    session.Resume()
    assert training.BeginUpdate()
    training.NextUpdate(ppo.TotalLoss)
    assert training.IsDone()
    training.Finish()
    assert session.LatestSnapshot().State == ml.OaTrainingState.Completed


def test_continuous_replay_off_policy_and_transforms(engine):
    mean = core.FromFloats([0.0, 0.5, -0.5, 1.0], [2, 2])
    log_stddev = core.FromFloats([-1.0] * 4, [2, 2])
    value = core.FromFloats([0.0, 0.0], [2])
    continuous = ml.SampleTanhNormalPolicy(
        mean, log_stddev, value, minimum=-2.0, maximum=2.0, seed=717
    )
    assert continuous.IsValid()

    normalized = ml.NormalizeObservation(
        core.FromFloats([1.0, 4.0, 5.0, -8.0], [2, 2]),
        core.FromFloats([1.0, 0.0], [2]),
        core.FromFloats([2.0, 2.0], [2]),
        clip=3.0,
    )
    scaled = ml.ScaleAction(
        core.FromFloats([-2.0, 0.0, 2.0], [3]), -1.0, 1.0, 0.0, 10.0
    )
    clipped = ml.ClipReward(core.FromFloats([-3.0, 0.25, 4.0], [3]))

    replay_config = ml.OaRlReplayConfig()
    replay_config.Capacity = 4
    replay_config.ObservationShape = [2]
    replay_config.ActionShape = []
    replay_config.ActionDtype = core.OaScalarType.Int32
    replay = ml.OaRlReplayBuffer.Create(replay_config)
    terminated = _u8([0, 1], [2])
    truncated = _u8([1, 0], [2])
    replay.Append(
        core.FromFloats([0.0, 1.0, 2.0, 3.0], [2, 2]),
        core.FromInt32([0, 1], [2]),
        core.FromFloats([1.0, 2.0, 3.0, 4.0], [2, 2]),
        core.FromFloats([1.0, 2.0], [2]),
        terminated,
        truncated,
    )
    sampled = replay.Sample(3, 9917)
    assert sampled.IsValid()

    dqn = ml.Dqn(
        core.FromFloats([1.0, 2.0, 4.0, 3.0], [2, 2]),
        core.FromInt32([1, 0], [2]),
        core.FromFloats([1.0, 2.0], [2]),
        core.FromFloats([5.0, 6.0, 7.0, 8.0], [2, 2]),
        terminated,
        truncated,
    )
    sac = ml.SacCritic(
        core.FromFloats([1.0, 2.0], [2]),
        core.FromFloats([1.5, 2.5], [2]),
        core.FromFloats([1.0, 2.0], [2]),
        core.FromFloats([5.0, 6.0], [2]),
        core.FromFloats([4.0, 7.0], [2]),
        core.FromFloats([-0.5, -0.25], [2]),
        terminated,
        truncated,
    )

    assert dqn.IsValid()
    assert sac.IsValid()
    assert replay.Size() == 2
    assert sampled.Observation.Shape() == [3, 2]
    assert core.CopyToHost(normalized) == pytest.approx([0.0, 2.0, 2.0, -3.0])
    assert core.CopyToHost(scaled) == pytest.approx([0.0, 5.0, 10.0])
    assert core.CopyToHost(clipped) == pytest.approx([-1.0, 0.25, 1.0])


def test_optional_gymnasium_adapter_preserves_step_boundaries(engine):
    gym = pytest.importorskip("gymnasium")
    from oa.gymnasium import GymnasiumAdapter

    adapter = GymnasiumAdapter(gym.make("CartPole-v1"))
    observation, info = adapter.reset(seed=717)
    assert observation.Shape() == [1, 4]
    assert isinstance(info, dict)
    transition = adapter.step(core.FromInt32([1], [1]))
    assert transition.observation.Shape() == [1, 4]
    assert transition.next_observation.Shape() == [1, 4]
    assert transition.reward.Shape() == [1]
    assert transition.terminated.Shape() == [1]
    assert transition.truncated.Shape() == [1]
    adapter.environment.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
