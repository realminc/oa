"""Optional Gymnasium interoperability for OA reinforcement learning.

Gymnasium is intentionally not an OA runtime dependency. Importing :mod:`oa`
does not import Gymnasium or NumPy; constructing this adapter does. The adapter
is a correctness-oriented scalar-environment boundary. Native vectorized OA
environments remain the primary high-throughput path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import (
	FnMatrix,
	EnvironmentSpec,
	EnvironmentSpace,
	EnvironmentSpaceKind,
	Matrix,
	ScalarType,
)


@dataclass(slots=True)
class GymnasiumTransition:
	observation: Any
	action: Any
	nextObservation: Any
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
				"Environment for high-throughput vector execution"
			)
		self._gym = gym
		self._np = np
		self.environment = environment
		self.spec = EnvironmentSpec()
		self.spec.observation = self._fieldSpec(
			"observation", environment.observation_space, observation=True
		)
		self.spec.action = self._fieldSpec(
			"action", environment.action_space, observation=False
		)
		self.spec.reward = EnvironmentSpace.box("reward", [])
		self.spec.terminated = EnvironmentSpace.binary("terminated")
		self.spec.truncated = EnvironmentSpace.binary("truncated")
		self.spec.validateDefinition()
		self.observation = None
		self.info: dict[str, Any] = {}

	def _fieldSpec(self, name: str, space: Any, *, observation: bool):
		gym = self._gym
		np = self._np
		if isinstance(space, gym.spaces.Box):
			if not np.issubdtype(space.dtype, np.floating):
				raise TypeError(f"OA {name} Box currently requires a floating dtype")
			minimum = float(np.min(space.low))
			maximum = float(np.max(space.high))
			return EnvironmentSpace.box(
				name, list(space.shape), ScalarType.Float32,
				minimum=minimum, maximum=maximum,
			)
		if isinstance(space, gym.spaces.Discrete):
			if observation:
				raise TypeError("Discrete observations are not supported by this adapter yet")
			if int(space.start) != 0:
				raise ValueError("OA discrete actions currently require start=0")
			return EnvironmentSpace.discrete(name, int(space.n))
		if isinstance(space, gym.spaces.MultiBinary):
			shape = list(space.shape) if space.shape else [int(space.n)]
			return EnvironmentSpace.binary(name, shape)
		raise TypeError(f"Unsupported Gymnasium space for {name}: {type(space).__name__}")

	def _observationMatrix(self, value: Any):
		array = self._np.asarray(value, dtype=self._np.float32)
		expected = tuple(self.spec.observation.shape)
		if array.shape != expected:
			raise ValueError(
				f"observation shape {array.shape} does not match declared {expected}"
			)
		return FnMatrix.fromFloats(array.reshape(-1).tolist(), [1, *expected])

	@staticmethod
	def _boundary(value: bool):
		return FnMatrix.fromBytes(
			[1 if value else 0], [1], ScalarType.UInt8
		)

	def reset(self, *, seed: int | None = None, options: dict | None = None):
		observation, info = self.environment.reset(seed=seed, options=options)
		self.observation = self._observationMatrix(observation)
		self.info = dict(info)
		return self.observation, self.info

	def step(self, action: Any) -> GymnasiumTransition:
		if self.observation is None:
			raise RuntimeError("reset must be called before step")
		prior = self.observation
		if isinstance(action, Matrix):
			self.spec.validateAction(action, 1)
			hostAction = FnMatrix.copyToHost(action)
			if self.spec.action.kind == EnvironmentSpaceKind.Discrete:
				gymAction: Any = int(hostAction[0])
			else:
				gymAction = self._np.asarray(
					hostAction, dtype=self._np.float32
				).reshape(tuple(self.spec.action.shape))
		else:
			gymAction = action

		nextObservation, reward, terminated, truncated, info = (
			self.environment.step(gymAction)
		)
		terminalObservation = self._observationMatrix(nextObservation)
		transition = GymnasiumTransition(
			observation=prior,
			action=action,
			nextObservation=terminalObservation,
			reward=FnMatrix.fromFloats([float(reward)], [1]),
			terminated=self._boundary(bool(terminated)),
			truncated=self._boundary(bool(truncated)),
			info=dict(info),
		)
		if terminated or truncated:
			resetObservation, resetInfo = self.environment.reset()
			self.observation = self._observationMatrix(resetObservation)
			self.info = dict(resetInfo)
		else:
			self.observation = terminalObservation
			self.info = dict(info)
		return transition


__all__ = ["GymnasiumAdapter", "GymnasiumTransition"]
