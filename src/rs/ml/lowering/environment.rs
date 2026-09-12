//! Private lowering for SDK-owned native environment kernels.

use crate::runtime::{BufferBinding, KernelId, PushConstant};
use crate::{Matrix, OpAttribute, Result};

use super::common::record_semantic;

#[allow(
	clippy::too_many_arguments,
	reason = "the Lunar reset ABI keeps every persistent state resource explicit"
)]
pub(crate) fn lunar_lander_reset(
	config_f32: &Matrix,
	config_u32: &Matrix,
	terrain_f32: &Matrix,
	state_f32: &Matrix,
	state_u32: &Matrix,
	observation: &Matrix,
	end_reason: &Matrix,
	environments: u32,
	seed: u64,
	only_completed: bool,
	environment_version: u32,
	state_layout_version: u32,
) -> Result<()> {
	let buffers = [
		BufferBinding::read(config_f32.storage()),
		BufferBinding::read(config_u32.storage()),
		BufferBinding::read(terrain_f32.storage()),
		BufferBinding::write(state_f32.storage()),
		BufferBinding::read_write(state_u32.storage()),
		BufferBinding::write(observation.storage()),
		BufferBinding::write(end_reason.storage()),
	];
	let push_constants = [
		PushConstant::U32(environments),
		PushConstant::U32(seed as u32),
		PushConstant::U32((seed >> 32) as u32),
		PushConstant::U32(u32::from(only_completed)),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "environment_version".into(),
			value: u64::from(environment_version),
		},
		OpAttribute::UnsignedInteger {
			name: "state_layout_version".into(),
			value: u64::from(state_layout_version),
		},
		OpAttribute::UnsignedInteger {
			name: "seed".into(),
			value: seed,
		},
		OpAttribute::Boolean {
			name: "only_completed".into(),
			value: only_completed,
		},
	];
	let inputs = [
		config_f32,
		config_u32,
		terrain_f32,
		state_f32,
		state_u32,
		observation,
		end_reason,
	];
	let outputs = [state_f32, state_u32, observation, end_reason];
	let kernel = KernelId::MlLunarLanderResetF32;
	record_semantic(
		&inputs,
		&outputs,
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(environments),
	)
}

#[allow(
	clippy::too_many_arguments,
	reason = "the Lunar step ABI keeps every persistent and transition resource explicit"
)]
pub(crate) fn lunar_lander_step(
	action: &Matrix,
	external_stop: &Matrix,
	config_f32: &Matrix,
	config_u32: &Matrix,
	terrain_f32: &Matrix,
	state_f32: &Matrix,
	state_u32: &Matrix,
	transition_observation: &Matrix,
	observation: &Matrix,
	reward: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	end_reason: &Matrix,
	environments: u32,
	environment_version: u32,
	physics_version: u32,
	observation_version: u32,
	reward_version: u32,
	state_layout_version: u32,
	config_identity: u64,
	max_episode_steps: u32,
	failure_penalty: f64,
) -> Result<()> {
	let buffers = [
		BufferBinding::read(action.storage()),
		BufferBinding::read(external_stop.storage()),
		BufferBinding::read(config_f32.storage()),
		BufferBinding::read(config_u32.storage()),
		BufferBinding::read(terrain_f32.storage()),
		BufferBinding::read_write(state_f32.storage()),
		BufferBinding::read_write(state_u32.storage()),
		BufferBinding::write(transition_observation.storage()),
		BufferBinding::read_write(observation.storage()),
		BufferBinding::write(reward.storage()),
		BufferBinding::write(terminated.storage()),
		BufferBinding::write(truncated.storage()),
		BufferBinding::write(end_reason.storage()),
	];
	let push_constants = [PushConstant::U32(environments)];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "environment_version".into(),
			value: u64::from(environment_version),
		},
		OpAttribute::UnsignedInteger {
			name: "physics_version".into(),
			value: u64::from(physics_version),
		},
		OpAttribute::UnsignedInteger {
			name: "observation_version".into(),
			value: u64::from(observation_version),
		},
		OpAttribute::UnsignedInteger {
			name: "reward_version".into(),
			value: u64::from(reward_version),
		},
		OpAttribute::UnsignedInteger {
			name: "state_layout_version".into(),
			value: u64::from(state_layout_version),
		},
		OpAttribute::UnsignedInteger {
			name: "config_identity".into(),
			value: config_identity,
		},
		OpAttribute::UnsignedInteger {
			name: "max_episode_steps".into(),
			value: u64::from(max_episode_steps),
		},
		OpAttribute::Float {
			name: "failure_penalty".into(),
			value: failure_penalty,
		},
	];
	let inputs = [
		action,
		external_stop,
		config_f32,
		config_u32,
		terrain_f32,
		state_f32,
		state_u32,
		transition_observation,
		observation,
		reward,
		terminated,
		truncated,
		end_reason,
	];
	let outputs = [
		state_f32,
		state_u32,
		transition_observation,
		observation,
		reward,
		terminated,
		truncated,
		end_reason,
	];
	let kernel = KernelId::MlLunarLanderStepF32;
	record_semantic(
		&inputs,
		&outputs,
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(environments),
	)
}

pub(crate) fn cart_pole_reset(
	done: &Matrix,
	state: &Matrix,
	episode_steps: &Matrix,
	episode_index: &Matrix,
	environments: u32,
	seed: u64,
	only_done: bool,
) -> Result<()> {
	let buffers = [
		BufferBinding::read_write(done.storage()),
		BufferBinding::write(state.storage()),
		BufferBinding::write(episode_steps.storage()),
		BufferBinding::read_write(episode_index.storage()),
	];
	let push_constants = [
		PushConstant::U32(environments),
		PushConstant::U32(seed as u32),
		PushConstant::U32((seed >> 32) as u32),
		PushConstant::U32(u32::from(only_done)),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "seed".into(),
			value: seed,
		},
		OpAttribute::Boolean {
			name: "only_completed".into(),
			value: only_done,
		},
	];
	let values = [done, state, episode_steps, episode_index];
	let kernel = KernelId::MlCartPoleResetF32;
	record_semantic(
		&values,
		&values,
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(environments),
	)
}

#[allow(
	clippy::too_many_arguments,
	reason = "CartPole dynamics and every persistent transition field remain explicit"
)]
pub(crate) fn cart_pole_step(
	action: &Matrix,
	state: &Matrix,
	transition_observation: &Matrix,
	reward: &Matrix,
	terminated: &Matrix,
	truncated: &Matrix,
	done: &Matrix,
	episode_steps: &Matrix,
	environments: u32,
	max_episode_steps: u32,
	gravity: f32,
	cart_mass: f32,
	pole_mass: f32,
	half_pole_length: f32,
	force_magnitude: f32,
	time_step: f32,
	position_threshold: f32,
	angle_threshold_radians: f32,
	dynamics_identity: u64,
) -> Result<()> {
	let buffers = [
		BufferBinding::read(action.storage()),
		BufferBinding::read_write(state.storage()),
		BufferBinding::write(transition_observation.storage()),
		BufferBinding::write(reward.storage()),
		BufferBinding::write(terminated.storage()),
		BufferBinding::write(truncated.storage()),
		BufferBinding::read_write(done.storage()),
		BufferBinding::read_write(episode_steps.storage()),
	];
	let push_constants = [
		PushConstant::U32(environments),
		PushConstant::U32(max_episode_steps),
		PushConstant::F32(gravity),
		PushConstant::F32(cart_mass),
		PushConstant::F32(pole_mass),
		PushConstant::F32(half_pole_length),
		PushConstant::F32(force_magnitude),
		PushConstant::F32(time_step),
		PushConstant::F32(position_threshold),
		PushConstant::F32(angle_threshold_radians),
	];
	let attributes = [
		OpAttribute::UnsignedInteger {
			name: "dynamics_version".into(),
			value: 1,
		},
		OpAttribute::UnsignedInteger {
			name: "dynamics_identity".into(),
			value: dynamics_identity,
		},
		OpAttribute::UnsignedInteger {
			name: "max_episode_steps".into(),
			value: u64::from(max_episode_steps),
		},
		OpAttribute::Float {
			name: "gravity".into(),
			value: f64::from(gravity),
		},
		OpAttribute::Float {
			name: "force_magnitude".into(),
			value: f64::from(force_magnitude),
		},
		OpAttribute::Float {
			name: "time_step".into(),
			value: f64::from(time_step),
		},
		OpAttribute::Float {
			name: "position_threshold".into(),
			value: f64::from(position_threshold),
		},
		OpAttribute::Float {
			name: "angle_threshold_radians".into(),
			value: f64::from(angle_threshold_radians),
		},
	];
	let inputs = [
		action,
		state,
		transition_observation,
		reward,
		terminated,
		truncated,
		done,
		episode_steps,
	];
	let outputs = [
		transition_observation,
		state,
		reward,
		terminated,
		truncated,
		done,
		episode_steps,
	];
	let kernel = KernelId::MlCartPoleStepF32;
	record_semantic(
		&inputs,
		&outputs,
		&attributes,
		kernel,
		&buffers,
		&push_constants,
		kernel.linear_workgroups(environments),
	)
}
