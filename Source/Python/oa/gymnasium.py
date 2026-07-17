"""Optional Gymnasium interoperability for OA reinforcement learning.

Gymnasium is intentionally not an OA runtime dependency. Importing :mod:`oa`
does not import Gymnasium or NumPy; constructing this adapter does. The adapter
is a correctness-oriented scalar-environment boundary. Native vectorized OA
environments remain the primary high-throughput path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import core, ml


@dataclass(slots=True)
class GymnasiumTransition:
    observation: Any
    action: Any
    next_observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: dict[str, Any]


class GymnasiumAdapter:
    """Adapt one scalar Gymnasium ``Env`` to OA matrices and RL specs.

    ``terminated`` and ``truncated`` remain separate. When an episode ends, the
    terminal observation is returned in the transition and the wrapped scalar
    environment is reset immediately for the next call. The reset observation
    is available through :attr:`observation`.
    """

    def __init__(self, environment: Any):
        try:
            import gymnasium as gym
            import numpy as np
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise ImportError(
                "GymnasiumAdapter requires the optional gymnasium and numpy packages"
            ) from error

        if getattr(environment, "num_envs", 1) != 1:
            raise ValueError(
                "GymnasiumAdapter currently accepts one scalar Env; use a native "
                "OaRlEnvironment for high-throughput vector execution"
            )
        self._gym = gym
        self._np = np
        self.environment = environment
        self.spec = ml.OaRlEnvironmentSpec()
        self.spec.Observation = self._field_spec(
            "observation", environment.observation_space, observation=True
        )
        self.spec.Action = self._field_spec(
            "action", environment.action_space, observation=False
        )
        self.spec.Reward = ml.OaRlFieldSpec.Box("reward", [])
        self.spec.Terminated = ml.OaRlFieldSpec.Binary("terminated")
        self.spec.Truncated = ml.OaRlFieldSpec.Binary("truncated")
        self.spec.ValidateDefinition()
        self.observation = None
        self.info: dict[str, Any] = {}

    def _field_spec(self, name: str, space: Any, *, observation: bool):
        gym = self._gym
        np = self._np
        if isinstance(space, gym.spaces.Box):
            if not np.issubdtype(space.dtype, np.floating):
                raise TypeError(f"OA {name} Box currently requires a floating dtype")
            minimum = float(np.min(space.low))
            maximum = float(np.max(space.high))
            return ml.OaRlFieldSpec.Box(
                name, list(space.shape), core.OaScalarType.Float32,
                minimum=minimum, maximum=maximum,
            )
        if isinstance(space, gym.spaces.Discrete):
            if observation:
                raise TypeError("Discrete observations are not supported by this adapter yet")
            if int(space.start) != 0:
                raise ValueError("OA discrete actions currently require start=0")
            return ml.OaRlFieldSpec.Discrete(name, int(space.n))
        if isinstance(space, gym.spaces.MultiBinary):
            shape = list(space.shape) if space.shape else [int(space.n)]
            return ml.OaRlFieldSpec.Binary(name, shape)
        raise TypeError(f"Unsupported Gymnasium space for {name}: {type(space).__name__}")

    def _observation_matrix(self, value: Any):
        array = self._np.asarray(value, dtype=self._np.float32)
        expected = tuple(self.spec.Observation.Shape)
        if array.shape != expected:
            raise ValueError(
                f"observation shape {array.shape} does not match declared {expected}"
            )
        return core.FromFloats(array.reshape(-1).tolist(), [1, *expected])

    @staticmethod
    def _boundary(value: bool):
        return core.FromBytes(
            [1 if value else 0], [1], core.OaScalarType.UInt8
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.environment.reset(seed=seed, options=options)
        self.observation = self._observation_matrix(observation)
        self.info = dict(info)
        return self.observation, self.info

    def step(self, action: Any) -> GymnasiumTransition:
        if self.observation is None:
            raise RuntimeError("reset must be called before step")
        prior = self.observation
        if hasattr(action, "Shape"):
            self.spec.ValidateAction(action, 1)
            host_action = core.CopyToHost(action)
            if self.spec.Action.Kind == ml.OaRlSpaceKind.Discrete:
                gym_action: Any = int(host_action[0])
            else:
                gym_action = self._np.asarray(
                    host_action, dtype=self._np.float32
                ).reshape(tuple(self.spec.Action.Shape))
        else:
            gym_action = action

        next_observation, reward, terminated, truncated, info = (
            self.environment.step(gym_action)
        )
        terminal_observation = self._observation_matrix(next_observation)
        transition = GymnasiumTransition(
            observation=prior,
            action=action,
            next_observation=terminal_observation,
            reward=core.FromFloats([float(reward)], [1]),
            terminated=self._boundary(bool(terminated)),
            truncated=self._boundary(bool(truncated)),
            info=dict(info),
        )
        if terminated or truncated:
            reset_observation, reset_info = self.environment.reset()
            self.observation = self._observation_matrix(reset_observation)
            self.info = dict(reset_info)
        else:
            self.observation = terminal_observation
            self.info = dict(info)
        return transition


__all__ = ["GymnasiumAdapter", "GymnasiumTransition"]
