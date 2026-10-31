"""Reinforcement-learning primitives matching Rust :mod:`oa::ml` RL surface.

Re-exports every RL type and function from :mod:`oa._native` under their
public names, grouping them by sub-domain.
"""

from .._native import (
	# Environment schema
	EnvironmentSpace,
	EnvironmentSpaceKind,
	EnvironmentSpec,
	EnvironmentTransition,
	ml_normalize_observation as normalize_observation,
	ml_scale_action as scale_action,
	ml_clip_reward as clip_reward,
	# Replay buffer
	ReplayBatch,
	ReplayBuffer,
	ReplayConfig,
	ReplayTransition,
	# Rollout buffer
	RolloutBuffer,
	RolloutConfig,
	RolloutTransition,
	# Actor-critic
	CategoricalActorCritic,
	CategoricalActorCriticConfig,
	CategoricalActorCriticOutput,
	# Loss configs/results
	DqnLossConfig,
	DqnLossResult,
	PpoLossConfig,
	PpoLossResult,
	SacCriticLossResult,
	SacLossConfig,
	# Advantage
	GaeConfig,
	GaeResult,
	ml_advantage_normalize as advantage_normalize,
	ml_advantage_gae as advantage_gae,
	# Policy results
	ContinuousPolicyResult,
	PolicyResult,
	# Loss functions
	ml_loss_ppo_clipped_policy as loss_ppo_clipped_policy,
	ml_loss_ppo as loss_ppo,
	ml_loss_dqn as loss_dqn,
	ml_loss_sac_critic as loss_sac_critic,
	ml_loss_sac_actor as loss_sac_actor,
	# Policy functions
	ml_policy_evaluate_categorical as policy_evaluate_categorical,
	ml_policy_sample_categorical as policy_sample_categorical,
	ml_policy_evaluate_tanh_normal as policy_evaluate_tanh_normal,
	ml_policy_sample_tanh_normal as policy_sample_tanh_normal,
)

__all__ = [
	# Environment schema
	"EnvironmentSpace",
	"EnvironmentSpaceKind",
	"EnvironmentSpec",
	"EnvironmentTransition",
	"normalize_observation",
	"scale_action",
	"clip_reward",
	# Replay buffer
	"ReplayBatch",
	"ReplayBuffer",
	"ReplayConfig",
	"ReplayTransition",
	# Rollout buffer
	"RolloutBuffer",
	"RolloutConfig",
	"RolloutTransition",
	# Actor-critic
	"CategoricalActorCritic",
	"CategoricalActorCriticConfig",
	"CategoricalActorCriticOutput",
	# Loss configs/results
	"DqnLossConfig",
	"DqnLossResult",
	"PpoLossConfig",
	"PpoLossResult",
	"SacCriticLossResult",
	"SacLossConfig",
	# Advantage
	"GaeConfig",
	"GaeResult",
	"advantage_normalize",
	"advantage_gae",
	# Policy results
	"ContinuousPolicyResult",
	"PolicyResult",
	# Loss functions
	"loss_ppo_clipped_policy",
	"loss_ppo",
	"loss_dqn",
	"loss_sac_critic",
	"loss_sac_actor",
	# Policy functions
	"policy_evaluate_categorical",
	"policy_sample_categorical",
	"policy_evaluate_tanh_normal",
	"policy_sample_tanh_normal",
]
