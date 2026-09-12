//! Scalar double-precision Lunar Lander dynamics and observation oracle.

use crate::core::vlm::{DQuat, DVec3};
use crate::{Error, Result};

use super::{
	LUNAR_ENVIRONMENT_VERSION, LUNAR_OBSERVATION_SIZE, LUNAR_OBSERVATION_VERSION,
	LUNAR_PHYSICS_VERSION, LUNAR_REWARD_VERSION, LUNAR_TERRAIN_VERSION, LunarAction,
	LunarEndReason, LunarTerrain, LunarTerrainConfig,
};

/// One spherical collision support fixed in lander body space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LunarSupportSphere {
	pub body_offset: DVec3,
	pub radius: f64,
}

impl LunarSupportSphere {
	pub const fn new(body_offset: DVec3, radius: f64) -> Self {
		Self {
			body_offset,
			radius,
		}
	}

	fn is_valid(self) -> bool {
		self.body_offset.is_finite() && finite_positive(self.radius)
	}
}

/// Versioned scalar dynamics, contact, observation, and reward contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LunarLander3dConfig {
	pub environment_version: u32,
	pub physics_version: u32,
	pub observation_version: u32,
	pub reward_version: u32,
	pub terrain: LunarTerrainConfig,
	pub policy_time_step: f64,
	pub physics_substeps: u32,
	pub contact_iterations: u32,
	pub gravity: f64,
	pub mass: f64,
	pub diagonal_inertia: DVec3,
	pub main_thrust: f64,
	pub attitude_torque: f64,
	pub fuel_capacity: f64,
	pub main_fuel_rate: f64,
	pub attitude_fuel_rate: f64,
	pub restitution: f64,
	pub friction: f64,
	pub contact_slop: f64,
	pub penetration_correction_fraction: f64,
	pub max_position_correction_per_contact: f64,
	pub max_contact_impulse: f64,
	pub max_bias_speed: f64,
	pub task_minimum_y: f64,
	pub task_maximum_y: f64,
	pub safe_linear_speed: f64,
	pub safe_angular_speed: f64,
	pub safe_tilt_radians: f64,
	pub hard_foot_impact_speed: f64,
	pub safe_dwell_steps: u32,
	pub max_episode_steps: u32,
	pub position_observation_scale: f64,
	pub velocity_observation_scale: f64,
	pub angular_velocity_observation_scale: f64,
	pub terrain_clearance_observation_scale: f64,
	pub foot_clearance_observation_scale: f64,
	pub terrain_probe_spacing: f64,
	pub reward_gamma: f64,
	pub position_potential_weight: f64,
	pub velocity_potential_weight: f64,
	pub tilt_potential_weight: f64,
	pub angular_potential_weight: f64,
	pub main_fuel_cost_weight: f64,
	pub attitude_fuel_cost_weight: f64,
	pub soft_foot_contact_reward: f64,
	pub stable_dwell_reward: f64,
	pub success_reward: f64,
	pub failure_penalty: f64,
	pub body_supports: [LunarSupportSphere; 3],
	pub foot_supports: [LunarSupportSphere; 4],
}

impl Default for LunarLander3dConfig {
	fn default() -> Self {
		Self {
			environment_version: LUNAR_ENVIRONMENT_VERSION,
			physics_version: LUNAR_PHYSICS_VERSION,
			observation_version: LUNAR_OBSERVATION_VERSION,
			reward_version: LUNAR_REWARD_VERSION,
			terrain: LunarTerrainConfig::default(),
			policy_time_step: 1.0 / 60.0,
			physics_substeps: 4,
			contact_iterations: 4,
			gravity: 1.62,
			mass: 1200.0,
			diagonal_inertia: DVec3 {
				x: 900.0,
				y: 800.0,
				z: 900.0,
			},
			main_thrust: 4200.0,
			attitude_torque: 900.0,
			fuel_capacity: 100.0,
			main_fuel_rate: 2.0,
			attitude_fuel_rate: 0.35,
			restitution: 0.05,
			friction: 0.65,
			contact_slop: 0.01,
			penetration_correction_fraction: 0.6,
			max_position_correction_per_contact: 0.03,
			max_contact_impulse: 6000.0,
			max_bias_speed: 1.0,
			task_minimum_y: -2.0,
			task_maximum_y: 40.0,
			safe_linear_speed: 0.55,
			safe_angular_speed: 0.25,
			safe_tilt_radians: 0.18,
			hard_foot_impact_speed: 1.2,
			safe_dwell_steps: 20,
			max_episode_steps: 1200,
			position_observation_scale: 16.0,
			velocity_observation_scale: 4.0,
			angular_velocity_observation_scale: 2.0,
			terrain_clearance_observation_scale: 8.0,
			foot_clearance_observation_scale: 2.0,
			terrain_probe_spacing: 1.5,
			reward_gamma: 0.99,
			position_potential_weight: 1.0,
			velocity_potential_weight: 0.6,
			tilt_potential_weight: 0.5,
			angular_potential_weight: 0.25,
			main_fuel_cost_weight: 0.04,
			attitude_fuel_cost_weight: 0.02,
			soft_foot_contact_reward: 0.02,
			stable_dwell_reward: 0.05,
			success_reward: 100.0,
			failure_penalty: -100.0,
			body_supports: [
				LunarSupportSphere::new(
					DVec3 {
						x: 0.0,
						y: 0.25,
						z: 0.0,
					},
					0.50,
				),
				LunarSupportSphere::new(
					DVec3 {
						x: 0.0,
						y: -0.15,
						z: 0.0,
					},
					0.50,
				),
				LunarSupportSphere::new(
					DVec3 {
						x: 0.0,
						y: 0.65,
						z: 0.0,
					},
					0.38,
				),
			],
			foot_supports: [
				LunarSupportSphere::new(
					DVec3 {
						x: -0.85,
						y: -1.0,
						z: -0.85,
					},
					0.15,
				),
				LunarSupportSphere::new(
					DVec3 {
						x: 0.85,
						y: -1.0,
						z: -0.85,
					},
					0.15,
				),
				LunarSupportSphere::new(
					DVec3 {
						x: 0.85,
						y: -1.0,
						z: 0.85,
					},
					0.15,
				),
				LunarSupportSphere::new(
					DVec3 {
						x: -0.85,
						y: -1.0,
						z: 0.85,
					},
					0.15,
				),
			],
		}
	}
}

impl LunarLander3dConfig {
	/// Validate every behavior-changing field.
	///
	/// # Errors
	///
	/// Returns an error when a version or numerical contract is invalid.
	pub fn validate(self) -> Result<()> {
		if self.environment_version != LUNAR_ENVIRONMENT_VERSION
			|| self.physics_version != LUNAR_PHYSICS_VERSION
			|| self.observation_version != LUNAR_OBSERVATION_VERSION
			|| self.reward_version != LUNAR_REWARD_VERSION
		{
			return Err(invalid("unsupported lunar lander contract version"));
		}
		self.terrain.validate()?;
		if !finite_positive(self.policy_time_step) || self.policy_time_step > 0.1 {
			return Err(invalid(
				"lunar policy time step must be finite and in (0, 0.1]",
			));
		}
		if !(1..=64).contains(&self.physics_substeps) {
			return Err(invalid("lunar physics substeps must be in [1, 64]"));
		}
		if !(1..=16).contains(&self.contact_iterations) {
			return Err(invalid("lunar contact iterations must be in [1, 16]"));
		}
		if !finite_non_negative(self.gravity)
			|| !finite_positive(self.mass)
			|| !finite_positive(self.diagonal_inertia.x)
			|| !finite_positive(self.diagonal_inertia.y)
			|| !finite_positive(self.diagonal_inertia.z)
		{
			return Err(invalid(
				"lunar gravity, mass, and diagonal inertia are invalid",
			));
		}
		if !finite_non_negative(self.main_thrust)
			|| !finite_non_negative(self.attitude_torque)
			|| !finite_positive(self.fuel_capacity)
			|| !finite_non_negative(self.main_fuel_rate)
			|| !finite_non_negative(self.attitude_fuel_rate)
		{
			return Err(invalid("lunar actuator or fuel configuration is invalid"));
		}
		if (self.main_thrust > 0.0 && self.main_fuel_rate <= 0.0)
			|| (self.attitude_torque > 0.0 && self.attitude_fuel_rate <= 0.0)
		{
			return Err(invalid(
				"lunar enabled actuators require a positive fuel rate",
			));
		}
		if !finite_non_negative(self.restitution)
			|| self.restitution > 1.0
			|| !finite_non_negative(self.friction)
			|| !finite_non_negative(self.contact_slop)
			|| !finite_non_negative(self.penetration_correction_fraction)
			|| self.penetration_correction_fraction > 1.0
			|| !finite_non_negative(self.max_position_correction_per_contact)
			|| !finite_positive(self.max_contact_impulse)
			|| !finite_non_negative(self.max_bias_speed)
		{
			return Err(invalid("lunar contact parameters are invalid"));
		}
		if !self.task_minimum_y.is_finite()
			|| !self.task_maximum_y.is_finite()
			|| self.task_minimum_y >= self.task_maximum_y
		{
			return Err(invalid("lunar task vertical bounds are invalid"));
		}
		if !finite_non_negative(self.safe_linear_speed)
			|| !finite_non_negative(self.safe_angular_speed)
			|| !finite_non_negative(self.safe_tilt_radians)
			|| self.safe_tilt_radians > std::f64::consts::FRAC_PI_2
			|| !finite_non_negative(self.hard_foot_impact_speed)
			|| self.safe_dwell_steps == 0
			|| self.max_episode_steps == 0
		{
			return Err(invalid("lunar terminal thresholds are invalid"));
		}
		if !finite_positive(self.position_observation_scale)
			|| !finite_positive(self.velocity_observation_scale)
			|| !finite_positive(self.angular_velocity_observation_scale)
			|| !finite_positive(self.terrain_clearance_observation_scale)
			|| !finite_positive(self.foot_clearance_observation_scale)
			|| !finite_positive(self.terrain_probe_spacing)
		{
			return Err(invalid("lunar observation scales are invalid"));
		}
		if !self.reward_gamma.is_finite()
			|| !(0.0..=1.0).contains(&self.reward_gamma)
			|| !finite_non_negative(self.position_potential_weight)
			|| !finite_non_negative(self.velocity_potential_weight)
			|| !finite_non_negative(self.tilt_potential_weight)
			|| !finite_non_negative(self.angular_potential_weight)
			|| !finite_non_negative(self.main_fuel_cost_weight)
			|| !finite_non_negative(self.attitude_fuel_cost_weight)
			|| !finite_non_negative(self.soft_foot_contact_reward)
			|| !finite_non_negative(self.stable_dwell_reward)
			|| !finite_non_negative(self.success_reward)
			|| !self.failure_penalty.is_finite()
			|| self.failure_penalty > 0.0
		{
			return Err(invalid("lunar reward parameters are invalid"));
		}
		if self.body_supports.iter().any(|support| !support.is_valid()) {
			return Err(invalid("lunar body support sphere is invalid"));
		}
		if self.foot_supports.iter().any(|support| !support.is_valid()) {
			return Err(invalid("lunar foot support sphere is invalid"));
		}
		Ok(())
	}

	/// Return the donor-compatible fingerprint over every contract field.
	pub fn contract_fingerprint(self) -> u64 {
		let mut fingerprint = 0x4f41_4c55_4e43_4631_u64;
		macro_rules! add {
			($value:expr) => {
				fingerprint = fingerprint_add(fingerprint, $value);
			};
		}
		macro_rules! add_f64 {
			($value:expr) => {
				add!(canonical_f64_bits($value));
			};
		}
		for value in [
			u64::from(self.environment_version),
			u64::from(self.physics_version),
			u64::from(self.observation_version),
			u64::from(self.reward_version),
			u64::from(LUNAR_TERRAIN_VERSION),
			u64::from(self.terrain.cells_x),
			u64::from(self.terrain.cells_z),
		] {
			add!(value);
		}
		for value in [
			self.terrain.cell_size,
			self.terrain.max_abs_height,
			self.terrain.max_slope,
			self.terrain.pad_half_extent,
			self.terrain.pad_transition_width,
			self.policy_time_step,
		] {
			add_f64!(value);
		}
		add!(u64::from(self.physics_substeps));
		add!(u64::from(self.contact_iterations));
		for value in [
			self.gravity,
			self.mass,
			self.diagonal_inertia.x,
			self.diagonal_inertia.y,
			self.diagonal_inertia.z,
			self.main_thrust,
			self.attitude_torque,
			self.fuel_capacity,
			self.main_fuel_rate,
			self.attitude_fuel_rate,
			self.restitution,
			self.friction,
			self.contact_slop,
			self.penetration_correction_fraction,
			self.max_position_correction_per_contact,
			self.max_contact_impulse,
			self.max_bias_speed,
			self.task_minimum_y,
			self.task_maximum_y,
			self.safe_linear_speed,
			self.safe_angular_speed,
			self.safe_tilt_radians,
			self.hard_foot_impact_speed,
		] {
			add_f64!(value);
		}
		add!(u64::from(self.safe_dwell_steps));
		add!(u64::from(self.max_episode_steps));
		for value in [
			self.position_observation_scale,
			self.velocity_observation_scale,
			self.angular_velocity_observation_scale,
			self.terrain_clearance_observation_scale,
			self.foot_clearance_observation_scale,
			self.terrain_probe_spacing,
			self.reward_gamma,
			self.position_potential_weight,
			self.velocity_potential_weight,
			self.tilt_potential_weight,
			self.angular_potential_weight,
			self.main_fuel_cost_weight,
			self.attitude_fuel_cost_weight,
			self.soft_foot_contact_reward,
			self.stable_dwell_reward,
			self.success_reward,
			self.failure_penalty,
		] {
			add_f64!(value);
		}
		for support in self.body_supports.into_iter().chain(self.foot_supports) {
			add_f64!(support.body_offset.x);
			add_f64!(support.body_offset.y);
			add_f64!(support.body_offset.z);
			add_f64!(support.radius);
		}
		fingerprint
	}
}

/// Complete mutable state of one scalar Lunar Lander episode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LunarLander3dState {
	pub position: DVec3,
	pub linear_velocity: DVec3,
	pub orientation: DQuat,
	pub angular_velocity_body: DVec3,
	pub fuel: f64,
	pub last_action: LunarAction,
	pub main_throttle: f64,
	pub attitude_command_body: DVec3,
	pub body_contacts: [bool; 3],
	pub body_contact_impulses: [f64; 3],
	pub foot_contacts: [bool; 4],
	pub feet_on_pad: [bool; 4],
	pub foot_contact_impulses: [f64; 4],
	pub foot_contact_rewarded: [bool; 4],
	pub episode_step: u32,
	pub stable_dwell: u32,
	pub terminated: bool,
	pub truncated: bool,
	pub end_reason: LunarEndReason,
	pub episode_return: f64,
}

impl Default for LunarLander3dState {
	fn default() -> Self {
		Self {
			position: DVec3 {
				x: 0.0,
				y: 6.0,
				z: 0.0,
			},
			linear_velocity: DVec3::default(),
			orientation: DQuat::identity(),
			angular_velocity_body: DVec3::default(),
			fuel: 100.0,
			last_action: LunarAction::Coast,
			main_throttle: 0.0,
			attitude_command_body: DVec3::default(),
			body_contacts: [false; 3],
			body_contact_impulses: [0.0; 3],
			foot_contacts: [false; 4],
			feet_on_pad: [false; 4],
			foot_contact_impulses: [0.0; 4],
			foot_contact_rewarded: [false; 4],
			episode_step: 0,
			stable_dwell: 0,
			terminated: false,
			truncated: false,
			end_reason: LunarEndReason::None,
			episode_return: 0.0,
		}
	}
}

impl LunarLander3dState {
	pub fn is_finite(self) -> bool {
		let norm_squared = self.orientation.norm_squared();
		self.position.is_finite()
			&& self.linear_velocity.is_finite()
			&& self.orientation.is_finite()
			&& norm_squared.is_finite()
			&& norm_squared > 1.0e-20
			&& self.angular_velocity_body.is_finite()
			&& self.fuel.is_finite()
			&& self.main_throttle.is_finite()
			&& self.attitude_command_body.is_finite()
			&& self.episode_return.is_finite()
			&& self
				.body_contact_impulses
				.iter()
				.all(|value| value.is_finite())
			&& self
				.foot_contact_impulses
				.iter()
				.all(|value| value.is_finite())
	}
}

/// Bounded contact-solver evidence for one policy step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LunarContactDiagnostics {
	pub maximum_penetration: f64,
	pub maximum_normal_impulse: f64,
	pub maximum_friction_impulse: f64,
	pub maximum_foot_closing_speed: f64,
	pub total_position_correction: f64,
	pub contact_count: u32,
	pub body_contact_occurred: bool,
	pub foot_contact_occurred: bool,
	pub bounded: bool,
}

impl Default for LunarContactDiagnostics {
	fn default() -> Self {
		Self {
			maximum_penetration: 0.0,
			maximum_normal_impulse: 0.0,
			maximum_friction_impulse: 0.0,
			maximum_foot_closing_speed: 0.0,
			total_position_correction: 0.0,
			contact_count: 0,
			body_contact_occurred: false,
			foot_contact_occurred: false,
			bounded: true,
		}
	}
}

impl LunarContactDiagnostics {
	pub fn is_finite(self) -> bool {
		self.maximum_penetration.is_finite()
			&& self.maximum_normal_impulse.is_finite()
			&& self.maximum_friction_impulse.is_finite()
			&& self.maximum_foot_closing_speed.is_finite()
			&& self.total_position_correction.is_finite()
	}
}

/// Result and actuator accounting from one scalar integration step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LunarPhysicsResult {
	pub main_fuel_used: f64,
	pub attitude_fuel_used: f64,
	pub contact: LunarContactDiagnostics,
}

/// Return a support's world-space center.
pub fn support_world_center(state: &LunarLander3dState, support: LunarSupportSphere) -> DVec3 {
	state.position + rotate(state.orientation, support.body_offset)
}

/// Advance one scalar state with the donor's fixed-order semi-implicit solver.
///
/// # Errors
///
/// Returns an error for invalid configuration, mismatched terrain, invalid
/// input state, or non-finite integration/contact output. The state may have
/// been partially advanced when a numerical output failure is reported.
pub fn integrate(
	config: &LunarLander3dConfig,
	terrain: &LunarTerrain,
	action: LunarAction,
	state: &mut LunarLander3dState,
) -> Result<LunarPhysicsResult> {
	config.validate()?;
	if terrain.config() != config.terrain {
		return Err(invalid(
			"lunar integration terrain does not match the versioned configuration",
		));
	}
	if !state.is_finite() || !(0.0..=config.fuel_capacity).contains(&state.fuel) {
		return Err(invalid("lunar integration received an invalid state"));
	}
	state.last_action = action;
	state.main_throttle = 0.0;
	state.attitude_command_body = DVec3::default();
	state.body_contacts = [false; 3];
	state.body_contact_impulses = [0.0; 3];
	state.foot_contacts = [false; 4];
	state.feet_on_pad = [false; 4];
	state.foot_contact_impulses = [0.0; 4];
	let mut result = LunarPhysicsResult::default();
	let time_step = config.policy_time_step / f64::from(config.physics_substeps);
	for _ in 0..config.physics_substeps {
		let mut main_activation = 0.0;
		let mut attitude_torque = DVec3::default();
		if action == LunarAction::MainEngine {
			let requested = config.main_fuel_rate * time_step;
			main_activation = if requested > 0.0 {
				(state.fuel / requested).min(1.0)
			} else {
				1.0
			};
			let used = requested * main_activation;
			state.fuel -= used;
			result.main_fuel_used += used;
		} else if action != LunarAction::Coast {
			let requested = config.attitude_fuel_rate * time_step;
			let activation = if requested > 0.0 {
				(state.fuel / requested).min(1.0)
			} else {
				1.0
			};
			let used = requested * activation;
			state.fuel -= used;
			result.attitude_fuel_used += used;
			attitude_torque = action_torque(action, config.attitude_torque * activation);
		}
		state.fuel = state.fuel.max(0.0);
		state.main_throttle = main_activation;
		state.attitude_command_body = attitude_torque;
		let thrust = rotate(
			state.orientation,
			DVec3 {
				x: 0.0,
				y: config.main_thrust * main_activation,
				z: 0.0,
			},
		);
		let acceleration = thrust / config.mass
			+ DVec3 {
				x: 0.0,
				y: -config.gravity,
				z: 0.0,
			};
		state.linear_velocity += acceleration * time_step;
		state.position += state.linear_velocity * time_step;
		let momentum = state
			.angular_velocity_body
			.component_mul(config.diagonal_inertia);
		let gyroscopic = state.angular_velocity_body.cross(momentum);
		let angular_acceleration =
			(attitude_torque - gyroscopic).component_div(config.diagonal_inertia);
		state.angular_velocity_body += angular_acceleration * time_step;
		let angular = DQuat {
			x: state.angular_velocity_body.x,
			y: state.angular_velocity_body.y,
			z: state.angular_velocity_body.z,
			w: 0.0,
		};
		let derivative = (state.orientation * angular) * 0.5;
		state.orientation = (state.orientation + derivative * time_step)
			.try_normalized()
			.ok_or_else(|| invalid("lunar integration produced a non-finite state"))?;
		if !state.is_finite() {
			return Err(invalid("lunar integration produced a non-finite state"));
		}
		for _ in 0..config.contact_iterations {
			for (index, support) in config.body_supports.iter().copied().enumerate() {
				resolve_support(
					config,
					terrain,
					time_step,
					support,
					SupportIndex::Body(index),
					state,
					&mut result.contact,
				);
			}
			for (index, support) in config.foot_supports.iter().copied().enumerate() {
				resolve_support(
					config,
					terrain,
					time_step,
					support,
					SupportIndex::Foot(index),
					state,
					&mut result.contact,
				);
			}
		}
		if !state.is_finite() || !result.contact.is_finite() {
			return Err(invalid(
				"lunar contact resolution produced a non-finite state",
			));
		}
	}
	refresh_contacts(config, terrain, state);
	let maximum_correction = f64::from(config.physics_substeps * config.contact_iterations)
		* (config.body_supports.len() + config.foot_supports.len()) as f64
		* config.max_position_correction_per_contact;
	result.contact.bounded = result.contact.maximum_normal_impulse <= config.max_contact_impulse
		&& result.contact.maximum_friction_impulse <= config.max_contact_impulse
		&& result.contact.total_position_correction <= maximum_correction + 1.0e-12;
	Ok(result)
}

/// Produce the frozen 33-value normalized observation layout.
pub fn observe(
	config: &LunarLander3dConfig,
	terrain: &LunarTerrain,
	state: &LunarLander3dState,
) -> [f32; LUNAR_OBSERVATION_SIZE] {
	let mut output = [0.0; LUNAR_OBSERVATION_SIZE];
	if !state.is_finite() {
		return output;
	}
	let mut offset = 0;
	for value in [state.position.x, state.position.y, state.position.z] {
		output[offset] = normalized(value, config.position_observation_scale);
		offset += 1;
	}
	for value in [
		state.linear_velocity.x,
		state.linear_velocity.y,
		state.linear_velocity.z,
	] {
		output[offset] = normalized(value, config.velocity_observation_scale);
		offset += 1;
	}
	let up = rotate(
		state.orientation,
		DVec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		},
	);
	let forward = rotate(
		state.orientation,
		DVec3 {
			x: 0.0,
			y: 0.0,
			z: -1.0,
		},
	);
	for value in [up.x, up.y, up.z, forward.x, forward.y, forward.z] {
		output[offset] = clamp_unit(value) as f32;
		offset += 1;
	}
	for value in [
		state.angular_velocity_body.x,
		state.angular_velocity_body.y,
		state.angular_velocity_body.z,
	] {
		output[offset] = normalized(value, config.angular_velocity_observation_scale);
		offset += 1;
	}
	for probe_z in -1..=1 {
		for probe_x in -1..=1 {
			let sample = terrain.query(
				state.position.x + f64::from(probe_x) * config.terrain_probe_spacing,
				state.position.z + f64::from(probe_z) * config.terrain_probe_spacing,
			);
			let clearance = if sample.in_bounds {
				state.position.y - sample.height
			} else {
				config.terrain_clearance_observation_scale
			};
			output[offset] = normalized(clearance, config.terrain_clearance_observation_scale);
			offset += 1;
		}
	}
	for support in config.foot_supports {
		let center = support_world_center(state, support);
		let sample = terrain.query(center.x, center.z);
		let clearance = if sample.in_bounds {
			center.y - support.radius - sample.height
		} else {
			config.foot_clearance_observation_scale
		};
		output[offset] = normalized(clearance, config.foot_clearance_observation_scale);
		offset += 1;
	}
	for contact in state.foot_contacts {
		output[offset] = u8::from(contact) as f32;
		offset += 1;
	}
	output[offset] = (state.fuel / config.fuel_capacity).clamp(0.0, 1.0) as f32;
	output
}

/// Return the bounded donor potential used by reward shaping.
pub fn potential(config: &LunarLander3dConfig, state: &LunarLander3dState) -> f64 {
	if !state.is_finite() {
		return 0.0;
	}
	let position = (state.position.length() / config.position_observation_scale).min(1.0);
	let velocity = (state.linear_velocity.length() / config.velocity_observation_scale).min(1.0);
	let up = rotate(
		state.orientation,
		DVec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		},
	);
	let tilt = ((1.0 - up.y) * 0.5).clamp(0.0, 1.0);
	let angular =
		(state.angular_velocity_body.length() / config.angular_velocity_observation_scale).min(1.0);
	-(config.position_potential_weight * position
		+ config.velocity_potential_weight * velocity
		+ config.tilt_potential_weight * tilt
		+ config.angular_potential_weight * angular)
}

#[derive(Clone, Copy)]
enum SupportIndex {
	Body(usize),
	Foot(usize),
}

fn resolve_support(
	config: &LunarLander3dConfig,
	terrain: &LunarTerrain,
	time_step: f64,
	support: LunarSupportSphere,
	support_index: SupportIndex,
	state: &mut LunarLander3dState,
	diagnostics: &mut LunarContactDiagnostics,
) {
	let center = support_world_center(state, support);
	let sample = terrain.query(center.x, center.z);
	if !sample.in_bounds {
		return;
	}
	let separation = center.y - support.radius - sample.height;
	if separation >= 0.0 {
		return;
	}
	let penetration = -separation;
	diagnostics.maximum_penetration = diagnostics.maximum_penetration.max(penetration);
	diagnostics.contact_count += 1;
	let lever = center - state.position;
	let angular_world = rotate(state.orientation, state.angular_velocity_body);
	let mut velocity = state.linear_velocity + angular_world.cross(lever);
	let normal_velocity = velocity.dot(sample.normal);
	let closing_speed = (-normal_velocity).max(0.0);
	if matches!(support_index, SupportIndex::Foot(_)) {
		diagnostics.foot_contact_occurred = true;
		diagnostics.maximum_foot_closing_speed =
			diagnostics.maximum_foot_closing_speed.max(closing_speed);
	}
	let bias_speed = config
		.max_bias_speed
		.min(penetration * config.penetration_correction_fraction / time_step);
	let target_delta_speed = (-(1.0 + config.restitution) * normal_velocity + bias_speed).max(0.0);
	let inverse_mass = effective_inverse_mass(config, state, lever, sample.normal);
	let normal_impulse = (target_delta_speed / inverse_mass).clamp(0.0, config.max_contact_impulse);
	apply_impulse(config, state, lever, sample.normal * normal_impulse);
	diagnostics.maximum_normal_impulse = diagnostics.maximum_normal_impulse.max(normal_impulse);
	let angular_world = rotate(state.orientation, state.angular_velocity_body);
	velocity = state.linear_velocity + angular_world.cross(lever);
	let tangent_velocity = velocity - sample.normal * velocity.dot(sample.normal);
	let tangent_speed = tangent_velocity.length();
	let mut friction_impulse = 0.0;
	if tangent_speed > 1.0e-12 {
		let direction = -tangent_velocity / tangent_speed;
		let inverse_mass = effective_inverse_mass(config, state, lever, direction);
		friction_impulse = (tangent_speed / inverse_mass)
			.min(config.friction * normal_impulse)
			.clamp(0.0, config.max_contact_impulse);
		apply_impulse(config, state, lever, direction * friction_impulse);
	}
	diagnostics.maximum_friction_impulse =
		diagnostics.maximum_friction_impulse.max(friction_impulse);
	let correction = config
		.max_position_correction_per_contact
		.min(penetration * config.penetration_correction_fraction);
	state.position += sample.normal * correction;
	diagnostics.total_position_correction += correction;
	match support_index {
		SupportIndex::Foot(index) => {
			state.foot_contacts[index] = true;
			state.foot_contact_impulses[index] += normal_impulse;
		}
		SupportIndex::Body(index) => {
			diagnostics.body_contact_occurred = true;
			state.body_contacts[index] = true;
			state.body_contact_impulses[index] += normal_impulse;
		}
	}
}

fn refresh_contacts(
	config: &LunarLander3dConfig,
	terrain: &LunarTerrain,
	state: &mut LunarLander3dState,
) {
	for (index, support) in config.body_supports.iter().copied().enumerate() {
		let center = support_world_center(state, support);
		let sample = terrain.query(center.x, center.z);
		state.body_contacts[index] =
			sample.in_bounds && center.y - support.radius - sample.height <= config.contact_slop;
	}
	for (index, support) in config.foot_supports.iter().copied().enumerate() {
		let center = support_world_center(state, support);
		let sample = terrain.query(center.x, center.z);
		state.foot_contacts[index] =
			sample.in_bounds && center.y - support.radius - sample.height <= config.contact_slop;
		state.feet_on_pad[index] = sample.in_bounds && terrain.is_on_pad(center.x, center.z);
	}
}

fn effective_inverse_mass(
	config: &LunarLander3dConfig,
	state: &LunarLander3dState,
	lever: DVec3,
	direction: DVec3,
) -> f64 {
	let rotational_world = lever.cross(direction);
	let rotational_body = inverse_rotate(state.orientation, rotational_world);
	1.0 / config.mass + rotational_body.dot(rotational_body.component_div(config.diagonal_inertia))
}

fn apply_impulse(
	config: &LunarLander3dConfig,
	state: &mut LunarLander3dState,
	lever: DVec3,
	impulse: DVec3,
) {
	state.linear_velocity += impulse / config.mass;
	let angular_body = inverse_rotate(state.orientation, lever.cross(impulse));
	state.angular_velocity_body += angular_body.component_div(config.diagonal_inertia);
}

fn action_torque(action: LunarAction, magnitude: f64) -> DVec3 {
	match action {
		LunarAction::PitchPositive => DVec3 {
			x: magnitude,
			y: 0.0,
			z: 0.0,
		},
		LunarAction::PitchNegative => DVec3 {
			x: -magnitude,
			y: 0.0,
			z: 0.0,
		},
		LunarAction::RollPositive => DVec3 {
			x: 0.0,
			y: 0.0,
			z: magnitude,
		},
		LunarAction::RollNegative => DVec3 {
			x: 0.0,
			y: 0.0,
			z: -magnitude,
		},
		LunarAction::YawPositive => DVec3 {
			x: 0.0,
			y: magnitude,
			z: 0.0,
		},
		LunarAction::YawNegative => DVec3 {
			x: 0.0,
			y: -magnitude,
			z: 0.0,
		},
		LunarAction::Coast | LunarAction::MainEngine => DVec3::default(),
	}
}

fn rotate(rotation: DQuat, vector: DVec3) -> DVec3 {
	rotation.try_rotate(vector).unwrap_or_default()
}

fn inverse_rotate(rotation: DQuat, vector: DVec3) -> DVec3 {
	rotation.try_inverse_rotate(vector).unwrap_or_default()
}

fn finite_positive(value: f64) -> bool {
	value.is_finite() && value > 0.0
}

fn finite_non_negative(value: f64) -> bool {
	value.is_finite() && value >= 0.0
}

fn clamp_unit(value: f64) -> f64 {
	value.clamp(-1.0, 1.0)
}

fn normalized(value: f64, scale: f64) -> f32 {
	if !value.is_finite() || !finite_positive(scale) {
		return 0.0;
	}
	clamp_unit(value / scale) as f32
}

fn fingerprint_add(mut fingerprint: u64, value: u64) -> u64 {
	fingerprint ^= value
		.wrapping_add(0x9e37_79b9_7f4a_7c15)
		.wrapping_add(fingerprint << 6)
		.wrapping_add(fingerprint >> 2);
	fingerprint ^= fingerprint >> 30;
	fingerprint = fingerprint.wrapping_mul(0xbf58_476d_1ce4_e5b9);
	fingerprint ^= fingerprint >> 27;
	fingerprint = fingerprint.wrapping_mul(0x94d0_49bb_1331_11eb);
	fingerprint ^ (fingerprint >> 31)
}

fn canonical_f64_bits(value: f64) -> u64 {
	if value == 0.0 {
		0.0_f64.to_bits()
	} else {
		value.to_bits()
	}
}

fn invalid(message: &'static str) -> Error {
	Error::invalid_argument(message)
}
