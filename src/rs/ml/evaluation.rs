//! Explicit policy-evaluation telemetry boundaries.

use crate::{Error, Result, matrix};

use super::{
	ActorCritic, EnvironmentSpaceKind, RolloutBuffer, RolloutConfig, RolloutTransition,
	environment::Environment, policy,
};

/// Fixed deterministic categorical evaluation horizon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyEvaluationConfig {
	/// Number of vector-environment steps to evaluate.
	pub horizon: usize,
	/// Reset seed and deterministic greedy operation seed.
	pub seed: u64,
}

impl Default for PolicyEvaluationConfig {
	fn default() -> Self {
		Self {
			horizon: 1000,
			seed: 1,
		}
	}
}

/// Completed categorical evaluation telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PolicyEvaluationMetrics {
	/// Vector-environment steps evaluated.
	pub environment_steps: u64,
	/// Individual lane transitions evaluated.
	pub transitions: u64,
	/// Terminated or truncated episodes completed inside the horizon.
	pub completed_episodes: u64,
	/// Mean return across completed episodes, or zero when none completed.
	pub mean_completed_return: f32,
	/// Minimum completed return, or zero when none completed.
	pub minimum_completed_return: f32,
	/// Maximum completed return, or zero when none completed.
	pub maximum_completed_return: f32,
}

/// Deterministically evaluate a categorical policy and return compact telemetry.
///
/// The complete horizon is recorded and submitted once, followed by one exact
/// wait and three compact rollout readbacks. Greedy action selection uses the
/// donor temperature-zero route. The environment remains open and idle.
///
/// # Errors
///
/// Returns an error for zero horizon/lanes, a non-discrete environment, model
/// ownership mismatch, environment/model/policy/rollout recording failure,
/// submission/wait failure, readback failure, or counter overflow.
pub fn evaluate_categorical(
	environment: &mut dyn Environment,
	model: &dyn ActorCritic,
	config: PolicyEvaluationConfig,
) -> Result<PolicyEvaluationMetrics> {
	if config.horizon == 0 || environment.environments() == 0 {
		return Err(Error::invalid_argument(
			"categorical evaluation requires a nonempty horizon and vector environment",
		));
	}
	if environment.spec().action().kind() != EnvironmentSpaceKind::Discrete {
		return Err(Error::invalid_argument(
			"categorical evaluation requires a discrete action space",
		));
	}
	for parameter in model.all_parameters()? {
		if !environment.engine().owns_matrix(&parameter.data()) {
			return Err(Error::invalid_argument(
				"evaluation model must belong to the environment Engine",
			));
		}
	}
	let environments = environment.environments() as usize;
	let observation_shape = environment.spec().observation().shape().to_vec();
	let observation_elements = environment.spec().observation().elements_per_environment();
	let mut rollout = RolloutBuffer::new(
		environment.engine(),
		RolloutConfig {
			time: config.horizon,
			environments,
			observation_shape,
		},
	)?;

	if let Err(error) = record_categorical_evaluation(
		environment,
		model,
		config,
		environments,
		observation_elements,
		&mut rollout,
	) {
		let _ = environment.cancel();
		rollout.abort_unsubmitted();
		return Err(error);
	}
	let event = environment.submit()?;
	environment.wait(&event)?;
	let reward = rollout.batch().reward().read_f32()?;
	let terminated = rollout.batch().terminated().read::<u8>()?;
	let truncated = rollout.batch().truncated().read::<u8>()?;
	summarize(
		config.horizon,
		environments,
		&reward,
		&terminated,
		&truncated,
	)
}

fn record_categorical_evaluation(
	environment: &mut dyn Environment,
	model: &dyn ActorCritic,
	config: PolicyEvaluationConfig,
	environments: usize,
	observation_elements: usize,
	rollout: &mut RolloutBuffer,
) -> Result<()> {
	environment.reset(config.seed)?;
	rollout.reset()?;
	for _ in 0..config.horizon {
		let observation = environment.observation().clone();
		let flat = matrix::reshape(&observation, [environments, observation_elements])?;
		let network = model.evaluate(&flat)?;
		let action = matrix::sample_logits(&network.logits, 0.0, 0, 1.0, config.seed)?;
		let selected = policy::evaluate_categorical(&network.logits, &action, &network.value)?;
		let transition = environment.step(&action)?;
		rollout.append(&RolloutTransition::new(
			transition.observation().clone(),
			action,
			transition.reward().clone(),
			network.value.clone(),
			network.value,
			selected.log_probability,
			transition.terminated().clone(),
			transition.truncated().clone(),
		))?;
		environment.reset_completed()?;
	}
	Ok(())
}

fn summarize(
	horizon: usize,
	environments: usize,
	reward: &[f32],
	terminated: &[u8],
	truncated: &[u8],
) -> Result<PolicyEvaluationMetrics> {
	let transitions = horizon
		.checked_mul(environments)
		.ok_or_else(|| Error::resource_exhausted("evaluation transition count overflows"))?;
	if reward.len() != transitions
		|| terminated.len() != transitions
		|| truncated.len() != transitions
	{
		return Err(Error::failed_precondition(
			"evaluation rollout readback length is inconsistent",
		));
	}
	let mut episode_return = vec![0.0_f32; environments];
	let mut sum = 0.0_f64;
	let mut minimum = f32::INFINITY;
	let mut maximum = f32::NEG_INFINITY;
	let mut completed = 0_u64;
	for step in 0..horizon {
		for (lane, lane_return) in episode_return.iter_mut().enumerate() {
			let index = step * environments + lane;
			*lane_return += reward[index];
			if terminated[index] != 0 || truncated[index] != 0 {
				sum += f64::from(*lane_return);
				minimum = minimum.min(*lane_return);
				maximum = maximum.max(*lane_return);
				*lane_return = 0.0;
				completed = completed
					.checked_add(1)
					.ok_or_else(|| Error::resource_exhausted("completed episode count exhausted"))?;
			}
		}
	}
	Ok(PolicyEvaluationMetrics {
		environment_steps: u64::try_from(horizon)
			.map_err(|_| Error::resource_exhausted("evaluation horizon exceeds u64"))?,
		transitions: u64::try_from(transitions)
			.map_err(|_| Error::resource_exhausted("evaluation transitions exceed u64"))?,
		completed_episodes: completed,
		mean_completed_return: if completed == 0 {
			0.0
		} else {
			(sum / completed as f64) as f32
		},
		minimum_completed_return: if completed == 0 { 0.0 } else { minimum },
		maximum_completed_return: if completed == 0 { 0.0 } else { maximum },
	})
}

#[cfg(test)]
mod tests {
	use super::summarize;

	#[test]
	fn episode_summary_separates_lanes_and_both_boundary_kinds() -> crate::Result<()> {
		let metrics = summarize(
			3,
			2,
			&[1.0, 10.0, 2.0, 20.0, 4.0, 40.0],
			&[0, 0, 1, 0, 0, 0],
			&[0, 1, 0, 0, 1, 0],
		)?;
		assert_eq!(metrics.environment_steps, 3);
		assert_eq!(metrics.transitions, 6);
		assert_eq!(metrics.completed_episodes, 3);
		assert_eq!(metrics.minimum_completed_return, 3.0);
		assert_eq!(metrics.maximum_completed_return, 10.0);
		assert!((metrics.mean_completed_return - 17.0 / 3.0).abs() < f32::EPSILON);
		Ok(())
	}
}
