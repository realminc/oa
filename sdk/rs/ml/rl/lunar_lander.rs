//! Versioned deterministic foundations for the SDK Lunar Lander 3D workload.

use crate::{Error, Result};

mod environment;
mod physics;
mod terrain;
mod training;
mod vector;

pub use environment::{
	LunarRewardTerms, LunarScalarEnvironment, LunarTransition, scripted_landing_action,
};
pub use physics::{
	LunarContactDiagnostics, LunarLander3dConfig, LunarLander3dState, LunarPhysicsResult,
	LunarSupportSphere, integrate, observe, potential, support_world_center,
};
pub use terrain::{LunarTerrain, LunarTerrainConfig, LunarTerrainSample, LunarTerrainTriangle};
pub use training::{
	LunarFirstEpisodeEvaluation, LunarFirstEpisodeEvaluationConfig, LunarTeacherConfig,
	LunarTeacherMetrics, collect_ppo_rollout, evaluate_first_episodes, pretrain_scripted_teacher,
};
pub use vector::{
	LUNAR_VECTOR_CONFIG_LAYOUT_VERSION, LUNAR_VECTOR_STATE_LAYOUT_VERSION,
	LunarLander3dEpisodeTelemetry, LunarLander3dVector, LunarLander3dVectorConfig,
	LunarLander3dVectorStep,
};

/// Current Lunar Lander environment contract version.
pub const LUNAR_ENVIRONMENT_VERSION: u32 = 1;
/// Current random-derivation contract version.
pub const LUNAR_RANDOM_VERSION: u32 = 1;
/// Current terrain contract version.
pub const LUNAR_TERRAIN_VERSION: u32 = 1;
/// Current scalar-physics contract version.
pub const LUNAR_PHYSICS_VERSION: u32 = 1;
/// Current observation-layout contract version.
pub const LUNAR_OBSERVATION_VERSION: u32 = 1;
/// Current reward contract version.
pub const LUNAR_REWARD_VERSION: u32 = 1;
/// Number of scalar values in one Lunar Lander observation.
pub const LUNAR_OBSERVATION_SIZE: usize = 33;

/// Independent deterministic random domain within one episode.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LunarRandomPurpose {
	/// Terrain generation.
	Terrain = 0x5445_5252_4149_4e31,
	/// Initial state generation.
	Spawn = 0x5350_4157_4e30_3031,
	/// Episode domain randomization.
	Domain = 0x444f_4d41_494e_3031,
}

/// Discrete control selected for one simulation step.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LunarAction {
	/// Apply no thrust or attitude torque.
	#[default]
	Coast = 0,
	/// Apply the main engine.
	MainEngine = 1,
	/// Apply positive pitch torque.
	PitchPositive = 2,
	/// Apply negative pitch torque.
	PitchNegative = 3,
	/// Apply positive roll torque.
	RollPositive = 4,
	/// Apply negative roll torque.
	RollNegative = 5,
	/// Apply positive yaw torque.
	YawPositive = 6,
	/// Apply negative yaw torque.
	YawNegative = 7,
}

impl TryFrom<u32> for LunarAction {
	type Error = Error;

	fn try_from(value: u32) -> Result<Self> {
		match value {
			0 => Ok(Self::Coast),
			1 => Ok(Self::MainEngine),
			2 => Ok(Self::PitchPositive),
			3 => Ok(Self::PitchNegative),
			4 => Ok(Self::RollPositive),
			5 => Ok(Self::RollNegative),
			6 => Ok(Self::YawPositive),
			7 => Ok(Self::YawNegative),
			_ => Err(Error::invalid_argument("invalid Lunar Lander action")),
		}
	}
}

/// Terminal or truncation cause for one episode.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LunarEndReason {
	/// Episode remains active.
	#[default]
	None = 0,
	/// Stable safe landing completed.
	SafeLanding = 1,
	/// A body support struck terrain.
	BodyImpact = 2,
	/// A foot struck terrain above the safe closing speed.
	HardFootImpact = 3,
	/// The lander left the task domain.
	OutOfBounds = 4,
	/// Simulation state became non-finite.
	NumericalFailure = 5,
	/// Episode reached its configured step limit.
	TimeLimit = 6,
	/// An external controller stopped the episode.
	ExternalStop = 7,
	/// The action value was outside the discrete action space.
	InvalidAction = 8,
}

/// Complete reproducibility identity for one vector-environment episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LunarEpisodeManifest {
	environment_version: u32,
	random_version: u32,
	terrain_version: u32,
	physics_version: u32,
	observation_version: u32,
	reward_version: u32,
	config_fingerprint: u64,
	base_seed: u64,
	environment_lane: u32,
	episode_index: u64,
	terrain_seed: u64,
	spawn_seed: u64,
	domain_seed: u64,
}

impl LunarEpisodeManifest {
	/// Derive a manifest using every current contract version.
	pub fn derive(
		base_seed: u64,
		environment_lane: u32,
		episode_index: u64,
		config_fingerprint: u64,
	) -> Self {
		Self::derive_versioned(
			base_seed,
			environment_lane,
			episode_index,
			LUNAR_ENVIRONMENT_VERSION,
			LUNAR_TERRAIN_VERSION,
			LUNAR_PHYSICS_VERSION,
			LUNAR_OBSERVATION_VERSION,
			LUNAR_REWARD_VERSION,
			config_fingerprint,
		)
	}

	/// Derive a manifest carrying explicit persisted contract versions.
	#[allow(clippy::too_many_arguments)]
	pub fn derive_versioned(
		base_seed: u64,
		environment_lane: u32,
		episode_index: u64,
		environment_version: u32,
		terrain_version: u32,
		physics_version: u32,
		observation_version: u32,
		reward_version: u32,
		config_fingerprint: u64,
	) -> Self {
		let mut root = 0x4f41_4c55_4e41_5231_u64;
		for value in [
			u64::from(environment_version),
			u64::from(LUNAR_RANDOM_VERSION),
			u64::from(terrain_version),
			u64::from(physics_version),
			u64::from(observation_version),
			u64::from(reward_version),
			base_seed,
			u64::from(environment_lane),
			episode_index,
		] {
			root = combine_seed(root, value);
		}
		Self {
			environment_version,
			random_version: LUNAR_RANDOM_VERSION,
			terrain_version,
			physics_version,
			observation_version,
			reward_version,
			config_fingerprint,
			base_seed,
			environment_lane,
			episode_index,
			terrain_seed: combine_seed(root, LunarRandomPurpose::Terrain as u64),
			spawn_seed: combine_seed(root, LunarRandomPurpose::Spawn as u64),
			domain_seed: combine_seed(root, LunarRandomPurpose::Domain as u64),
		}
	}

	/// Validate supported versions and all deterministically derived seeds.
	pub fn validate(self) -> Result<()> {
		if self.environment_version != LUNAR_ENVIRONMENT_VERSION
			|| self.random_version != LUNAR_RANDOM_VERSION
			|| self.terrain_version != LUNAR_TERRAIN_VERSION
			|| self.physics_version != LUNAR_PHYSICS_VERSION
			|| self.observation_version != LUNAR_OBSERVATION_VERSION
			|| self.reward_version != LUNAR_REWARD_VERSION
		{
			return Err(Error::invalid_argument(
				"unsupported Lunar Lander contract version",
			));
		}
		let expected = Self::derive_versioned(
			self.base_seed,
			self.environment_lane,
			self.episode_index,
			self.environment_version,
			self.terrain_version,
			self.physics_version,
			self.observation_version,
			self.reward_version,
			self.config_fingerprint,
		);
		if self.terrain_seed != expected.terrain_seed
			|| self.spawn_seed != expected.spawn_seed
			|| self.domain_seed != expected.domain_seed
		{
			return Err(Error::invalid_argument(
				"Lunar Lander manifest derived seeds do not match its versioned inputs",
			));
		}
		Ok(())
	}

	/// Return the seed for one independent random purpose.
	pub const fn seed_for(self, purpose: LunarRandomPurpose) -> u64 {
		match purpose {
			LunarRandomPurpose::Terrain => self.terrain_seed,
			LunarRandomPurpose::Spawn => self.spawn_seed,
			LunarRandomPurpose::Domain => self.domain_seed,
		}
	}

	/// Sample the donor's exact deterministic half-open unit interval.
	pub fn sample_01(self, purpose: LunarRandomPurpose, counter: u64) -> f64 {
		let mantissa = combine_seed(self.seed_for(purpose), counter) >> 11;
		mantissa as f64 * (1.0 / 9_007_199_254_740_992.0)
	}

	/// Return the configuration fingerprint bound to this episode.
	pub const fn config_fingerprint(self) -> u64 {
		self.config_fingerprint
	}
	/// Return the environment contract version.
	pub const fn environment_version(self) -> u32 {
		self.environment_version
	}
	/// Return the random-derivation contract version.
	pub const fn random_version(self) -> u32 {
		self.random_version
	}
	/// Return the terrain contract version.
	pub const fn terrain_version(self) -> u32 {
		self.terrain_version
	}
	/// Return the scalar-physics contract version.
	pub const fn physics_version(self) -> u32 {
		self.physics_version
	}
	/// Return the observation-layout contract version.
	pub const fn observation_version(self) -> u32 {
		self.observation_version
	}
	/// Return the reward contract version.
	pub const fn reward_version(self) -> u32 {
		self.reward_version
	}
	/// Return the root caller seed.
	pub const fn base_seed(self) -> u64 {
		self.base_seed
	}
	/// Return the vector lane identity.
	pub const fn environment_lane(self) -> u32 {
		self.environment_lane
	}
	/// Return the per-lane episode index.
	pub const fn episode_index(self) -> u64 {
		self.episode_index
	}
	/// Return the terrain seed.
	pub const fn terrain_seed(self) -> u64 {
		self.terrain_seed
	}
	/// Return the spawn seed.
	pub const fn spawn_seed(self) -> u64 {
		self.spawn_seed
	}
	/// Return the domain-randomization seed.
	pub const fn domain_seed(self) -> u64 {
		self.domain_seed
	}
}

fn mix64(mut value: u64) -> u64 {
	value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
	value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
	value ^ (value >> 31)
}

fn combine_seed(seed: u64, value: u64) -> u64 {
	mix64(seed ^ mix64(value))
}
