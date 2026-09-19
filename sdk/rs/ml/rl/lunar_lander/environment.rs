//! Deterministic scalar Lunar Lander episode owner and reward oracle.

use crate::core::vlm::{DQuat, DVec3};
use crate::{Error, Result};

use super::{
	LUNAR_OBSERVATION_SIZE, LUNAR_TERRAIN_VERSION, LunarAction, LunarContactDiagnostics,
	LunarEndReason, LunarEpisodeManifest, LunarLander3dConfig, LunarLander3dState,
	LunarPhysicsResult, LunarRandomPurpose, LunarTerrain, integrate, observe, potential,
};

/// Decomposed reward evidence for one completed environment transition.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LunarRewardTerms {
	pub potential_before: f64,
	pub potential_after: f64,
	pub shaping: f64,
	pub main_fuel_cost: f64,
	pub attitude_fuel_cost: f64,
	pub soft_foot_contact: f64,
	pub stable_dwell: f64,
	pub terminal: f64,
	pub total: f64,
}

impl LunarRewardTerms {
	/// Return whether every term is finite.
	pub fn is_finite(self) -> bool {
		self.potential_before.is_finite()
			&& self.potential_after.is_finite()
			&& self.shaping.is_finite()
			&& self.main_fuel_cost.is_finite()
			&& self.attitude_fuel_cost.is_finite()
			&& self.soft_foot_contact.is_finite()
			&& self.stable_dwell.is_finite()
			&& self.terminal.is_finite()
			&& self.total.is_finite()
	}

	/// Sum the independently reported reward contributions.
	pub fn sum(self) -> f64 {
		self.shaping
			+ self.main_fuel_cost
			+ self.attitude_fuel_cost
			+ self.soft_foot_contact
			+ self.stable_dwell
			+ self.terminal
	}
}

/// One scalar environment transition and its diagnostic evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct LunarTransition {
	pub observation: [f32; LUNAR_OBSERVATION_SIZE],
	pub reward: f64,
	pub terminated: bool,
	pub truncated: bool,
	pub end_reason: LunarEndReason,
	pub reward_terms: LunarRewardTerms,
	pub contact: LunarContactDiagnostics,
	pub integration_error: Option<String>,
}

impl Default for LunarTransition {
	fn default() -> Self {
		Self {
			observation: [0.0; LUNAR_OBSERVATION_SIZE],
			reward: 0.0,
			terminated: false,
			truncated: false,
			end_reason: LunarEndReason::None,
			reward_terms: LunarRewardTerms::default(),
			contact: LunarContactDiagnostics::default(),
			integration_error: None,
		}
	}
}

/// Deterministic scalar reference environment used to qualify vector kernels.
#[derive(Clone, Debug, PartialEq)]
pub struct LunarScalarEnvironment {
	config: LunarLander3dConfig,
	manifest: LunarEpisodeManifest,
	terrain: LunarTerrain,
	state: LunarLander3dState,
}

impl LunarScalarEnvironment {
	/// Construct an environment with level terrain and reset it deterministically.
	pub fn flat(config: LunarLander3dConfig, manifest: LunarEpisodeManifest) -> Result<Self> {
		let terrain = LunarTerrain::flat(config.terrain)?;
		Self::with_terrain(config, manifest, terrain)
	}

	/// Construct an environment with manifest-seeded bounded terrain.
	pub fn seeded(config: LunarLander3dConfig, manifest: LunarEpisodeManifest) -> Result<Self> {
		let terrain = LunarTerrain::seeded(config.terrain, manifest)?;
		Self::with_terrain(config, manifest, terrain)
	}

	/// Construct an environment with caller-provided checked terrain.
	pub fn with_terrain(
		config: LunarLander3dConfig,
		manifest: LunarEpisodeManifest,
		terrain: LunarTerrain,
	) -> Result<Self> {
		config.validate()?;
		manifest.validate()?;
		if manifest.environment_version() != config.environment_version
			|| manifest.physics_version() != config.physics_version
			|| manifest.observation_version() != config.observation_version
			|| manifest.reward_version() != config.reward_version
			|| manifest.terrain_version() != LUNAR_TERRAIN_VERSION
		{
			return Err(invalid(
				"lunar manifest versions do not match the environment configuration",
			));
		}
		if manifest.config_fingerprint() != config.contract_fingerprint() {
			return Err(invalid(
				"lunar manifest configuration fingerprint does not match",
			));
		}
		if terrain.config() != config.terrain {
			return Err(invalid(
				"lunar terrain configuration does not match the environment",
			));
		}
		let mut result = Self {
			config,
			manifest,
			terrain,
			state: LunarLander3dState::default(),
		};
		result.reset()?;
		Ok(result)
	}

	pub const fn config(&self) -> LunarLander3dConfig {
		self.config
	}

	pub const fn manifest(&self) -> LunarEpisodeManifest {
		self.manifest
	}

	pub const fn terrain(&self) -> &LunarTerrain {
		&self.terrain
	}

	pub const fn state(&self) -> &LunarLander3dState {
		&self.state
	}

	pub fn observation(&self) -> [f32; LUNAR_OBSERVATION_SIZE] {
		observe(&self.config, &self.terrain, &self.state)
	}

	/// Reset to the deterministic manifest-derived spawn state.
	pub fn reset(&mut self) -> Result<()> {
		let state = self.spawn_state()?;
		if !state.is_finite() {
			return Err(invalid(
				"lunar deterministic reset produced an invalid state",
			));
		}
		self.state = state;
		Ok(())
	}

	/// Replace the current state after validating lifecycle and numerical fields.
	pub fn set_state(&mut self, mut state: LunarLander3dState) -> Result<()> {
		if !state.is_finite()
			|| !(0.0..=self.config.fuel_capacity).contains(&state.fuel)
			|| (state.terminated && state.truncated)
			|| ((state.terminated || state.truncated) && state.end_reason == LunarEndReason::None)
			|| (!state.terminated && !state.truncated && state.end_reason != LunarEndReason::None)
		{
			return Err(invalid("invalid Lunar Lander scalar state"));
		}
		state.orientation = state
			.orientation
			.try_normalized()
			.ok_or_else(|| invalid("invalid Lunar Lander scalar orientation"))?;
		self.state = state;
		Ok(())
	}

	/// Advance one policy transition.
	///
	/// Invalid discrete actions deliberately terminate this lane with the failure
	/// penalty so a vector implementation need not reject a whole batch.
	pub fn step(&mut self, action: u32, external_stop: bool) -> Result<LunarTransition> {
		if self.state.terminated || self.state.truncated {
			return Err(Error::failed_precondition(
				"lunar episode has ended; reset is required",
			));
		}
		let Ok(action) = LunarAction::try_from(action) else {
			self.state.episode_step += 1;
			self.state.terminated = true;
			self.state.end_reason = LunarEndReason::InvalidAction;
			self.state.episode_return += self.config.failure_penalty;
			let reward_terms = LunarRewardTerms {
				terminal: self.config.failure_penalty,
				total: self.config.failure_penalty,
				..LunarRewardTerms::default()
			};
			return Ok(LunarTransition {
				observation: self.observation(),
				reward: self.config.failure_penalty,
				terminated: true,
				end_reason: LunarEndReason::InvalidAction,
				reward_terms,
				..LunarTransition::default()
			});
		};
		if external_stop {
			self.state.truncated = true;
			self.state.end_reason = LunarEndReason::ExternalStop;
			return Ok(LunarTransition {
				observation: self.observation(),
				truncated: true,
				end_reason: LunarEndReason::ExternalStop,
				..LunarTransition::default()
			});
		}
		self.step_valid(action)
	}

	fn step_valid(&mut self, action: LunarAction) -> Result<LunarTransition> {
		let potential_before = potential(&self.config, &self.state);
		let (physics, integration_error) =
			match integrate(&self.config, &self.terrain, action, &mut self.state) {
				Ok(physics) => (physics, None),
				Err(error) => {
					self.state.terminated = true;
					self.state.end_reason = LunarEndReason::NumericalFailure;
					(LunarPhysicsResult::default(), Some(error.to_string()))
				}
			};
		self.state.episode_step += 1;
		let instantaneous_safe = self.resolve_end_reason(&physics, integration_error.is_some());
		let mut reward = self.reward_terms(potential_before, &physics, instantaneous_safe);
		if physics.contact.maximum_foot_closing_speed <= self.config.hard_foot_impact_speed {
			for index in 0..self.state.foot_contacts.len() {
				if self.state.foot_contacts[index] && !self.state.foot_contact_rewarded[index] {
					reward.soft_foot_contact += self.config.soft_foot_contact_reward;
					self.state.foot_contact_rewarded[index] = true;
				}
			}
		}
		reward.terminal = if self.state.end_reason == LunarEndReason::SafeLanding {
			self.config.success_reward
		} else if self.state.terminated {
			self.config.failure_penalty
		} else {
			0.0
		};
		reward.total = reward.sum();
		self.state.episode_return += reward.total;
		Ok(LunarTransition {
			observation: self.observation(),
			reward: reward.total,
			terminated: self.state.terminated,
			truncated: self.state.truncated,
			end_reason: self.state.end_reason,
			reward_terms: reward,
			contact: physics.contact,
			integration_error,
		})
	}

	fn resolve_end_reason(&mut self, physics: &LunarPhysicsResult, failed: bool) -> bool {
		let mut safe = false;
		if failed || !self.state.is_finite() {
			self.state.terminated = true;
			self.state.end_reason = LunarEndReason::NumericalFailure;
		} else if !self
			.terrain
			.contains(self.state.position.x, self.state.position.z)
			|| self.state.position.y < self.config.task_minimum_y
			|| self.state.position.y > self.config.task_maximum_y
		{
			self.state.terminated = true;
			self.state.end_reason = LunarEndReason::OutOfBounds;
		} else if physics.contact.body_contact_occurred || self.state.body_contacts.contains(&true) {
			self.state.terminated = true;
			self.state.end_reason = LunarEndReason::BodyImpact;
		} else if physics.contact.maximum_foot_closing_speed > self.config.hard_foot_impact_speed {
			self.state.terminated = true;
			self.state.end_reason = LunarEndReason::HardFootImpact;
		} else {
			safe = instantaneously_safe(&self.config, &self.state);
			self.state.stable_dwell = if safe { self.state.stable_dwell + 1 } else { 0 };
			if self.state.stable_dwell >= self.config.safe_dwell_steps {
				self.state.terminated = true;
				self.state.end_reason = LunarEndReason::SafeLanding;
			}
		}
		if !self.state.terminated && self.state.episode_step >= self.config.max_episode_steps {
			self.state.truncated = true;
			self.state.end_reason = LunarEndReason::TimeLimit;
		}
		safe
	}

	fn reward_terms(
		&self,
		potential_before: f64,
		physics: &LunarPhysicsResult,
		instantaneous_safe: bool,
	) -> LunarRewardTerms {
		let potential_after = potential(&self.config, &self.state);
		LunarRewardTerms {
			potential_before,
			potential_after,
			shaping: self.config.reward_gamma
				* if self.state.terminated {
					0.0
				} else {
					potential_after
				} - potential_before,
			main_fuel_cost: -self.config.main_fuel_cost_weight * physics.main_fuel_used,
			attitude_fuel_cost: -self.config.attitude_fuel_cost_weight * physics.attitude_fuel_used,
			stable_dwell: if instantaneous_safe {
				self.config.stable_dwell_reward
			} else {
				0.0
			},
			..LunarRewardTerms::default()
		}
	}

	fn spawn_state(&self) -> Result<LunarLander3dState> {
		let sample = |counter| self.manifest.sample_01(LunarRandomPurpose::Spawn, counter);
		let pad_range = self.config.terrain.pad_half_extent * 0.35;
		let pitch = (sample(6) * 2.0 - 1.0) * 0.03;
		let roll = (sample(7) * 2.0 - 1.0) * 0.03;
		let yaw = (sample(8) * 2.0 - 1.0) * 0.08;
		let axis_angle = |axis, angle| {
			DQuat::try_from_axis_angle(axis, angle)
				.ok_or_else(|| invalid("lunar deterministic spawn rotation is invalid"))
		};
		let orientation = (axis_angle(
			DVec3 {
				x: 0.0,
				y: 1.0,
				z: 0.0,
			},
			yaw,
		)? * axis_angle(
			DVec3 {
				x: 1.0,
				y: 0.0,
				z: 0.0,
			},
			pitch,
		)? * axis_angle(
			DVec3 {
				x: 0.0,
				y: 0.0,
				z: 1.0,
			},
			roll,
		)?)
		.try_normalized()
		.ok_or_else(|| invalid("lunar deterministic spawn rotation is invalid"))?;
		Ok(LunarLander3dState {
			position: DVec3 {
				x: (sample(0) * 2.0 - 1.0) * pad_range,
				y: 5.0 + sample(1) * 2.0,
				z: (sample(2) * 2.0 - 1.0) * pad_range,
			},
			linear_velocity: DVec3 {
				x: (sample(3) * 2.0 - 1.0) * 0.12,
				y: -0.1 - sample(4) * 0.2,
				z: (sample(5) * 2.0 - 1.0) * 0.12,
			},
			orientation,
			fuel: self.config.fuel_capacity,
			..LunarLander3dState::default()
		})
	}
}

/// Deterministic scalar controller used by the donor solvability oracle.
pub fn scripted_landing_action(
	config: &LunarLander3dConfig,
	state: &LunarLander3dState,
) -> LunarAction {
	let upright_foot_height = config
		.foot_supports
		.iter()
		.map(|support| support.radius - support.body_offset.y)
		.fold(0.0_f64, f64::max);
	let foot_clearance = (state.position.y - upright_foot_height).max(0.0);
	let desired_descent = -(0.48 * foot_clearance.sqrt()).clamp(0.24, 0.72);
	if state.linear_velocity.y < desired_descent {
		return LunarAction::MainEngine;
	}
	let up = state
		.orientation
		.try_rotate(DVec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		})
		.unwrap_or_default();
	let guidance = foot_clearance.clamp(0.0, 1.0);
	let target_x =
		guidance * (-0.055 * state.position.x - 0.30 * state.linear_velocity.x).clamp(-0.12, 0.12);
	let target_z =
		guidance * (-0.055 * state.position.z - 0.30 * state.linear_velocity.z).clamp(-0.12, 0.12);
	let pitch = -8.0 * (up.z - target_z) - 3.0 * state.angular_velocity_body.x;
	let roll = 8.0 * (up.x - target_x) - 3.0 * state.angular_velocity_body.z;
	if pitch.abs().max(roll.abs()) <= 0.025 {
		LunarAction::Coast
	} else if pitch.abs() >= roll.abs() {
		if pitch > 0.0 {
			LunarAction::PitchPositive
		} else {
			LunarAction::PitchNegative
		}
	} else if roll > 0.0 {
		LunarAction::RollPositive
	} else {
		LunarAction::RollNegative
	}
}

fn instantaneously_safe(config: &LunarLander3dConfig, state: &LunarLander3dState) -> bool {
	let up = state
		.orientation
		.try_rotate(DVec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		})
		.unwrap_or_default();
	let tilt = up.y.clamp(-1.0, 1.0).acos();
	state.foot_contacts.iter().all(|value| *value)
		&& state.feet_on_pad.iter().all(|value| *value)
		&& !state.body_contacts.contains(&true)
		&& state.linear_velocity.length() <= config.safe_linear_speed
		&& state.angular_velocity_body.length() <= config.safe_angular_speed
		&& tilt <= config.safe_tilt_radians
}

fn invalid(message: &'static str) -> Error {
	Error::invalid_argument(message)
}
