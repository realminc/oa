//! Concrete reinforcement-learning workloads built on OA ML contracts.

mod cart_pole;
pub mod lunar_lander;

pub use cart_pole::{CartPole, CartPoleConfig};
pub use lunar_lander::{
	LUNAR_ENVIRONMENT_VERSION, LUNAR_OBSERVATION_SIZE, LUNAR_OBSERVATION_VERSION,
	LUNAR_PHYSICS_VERSION, LUNAR_RANDOM_VERSION, LUNAR_REWARD_VERSION, LUNAR_TERRAIN_VERSION,
	LunarAction, LunarContactDiagnostics, LunarEndReason, LunarEpisodeManifest,
	LunarFirstEpisodeEvaluation, LunarFirstEpisodeEvaluationConfig, LunarLander3dConfig,
	LunarLander3dEpisodeTelemetry, LunarLander3dState, LunarLander3dVector,
	LunarLander3dVectorConfig, LunarLander3dVectorStep, LunarPhysicsResult, LunarRandomPurpose,
	LunarRewardTerms, LunarScalarEnvironment, LunarSupportSphere, LunarTeacherConfig,
	LunarTeacherMetrics, LunarTerrain, LunarTerrainConfig, LunarTerrainSample,
	LunarTerrainTriangle, LunarTransition, collect_ppo_rollout, evaluate_first_episodes,
	pretrain_scripted_teacher,
};
