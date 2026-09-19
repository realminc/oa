//! Lunar-specific policy training and held-out evaluation workflow.
//!
//! Port provenance: OA C++ test evidence
//! `test/cpp/ml/rl/lunarLander3dPpo.{h,cpp}`. Reusable PPO, autograd,
//! optimizer, rollout, event, and model-file behavior remains owned by
//! `oa::ml`; this module only owns the concrete Lunar curriculum and telemetry.

use crate::ml::{
	ActorCritic, AdamW, CategoricalActorCritic, GradientTape, ItTraining, ItTrainingConfig, Module,
	PpoTrainer, RolloutBuffer, RolloutConfig, RolloutTransition, environment::Environment, loss,
	policy,
};
use crate::{Engine, Error, Matrix, Result, matrix};

use super::{
	LUNAR_OBSERVATION_SIZE, LunarEndReason, LunarEpisodeManifest, LunarLander3dConfig,
	LunarLander3dVector, LunarLander3dVectorConfig, LunarScalarEnvironment, scripted_landing_action,
};

/// Deterministic flat-terrain scripted-teacher curriculum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LunarTeacherConfig {
	/// Number of independent scalar episodes used to build the dataset.
	pub episodes: u32,
	/// Complete shuffled passes over the retained dataset.
	pub epochs: u32,
	/// Maximum examples in one optimizer step.
	pub batch_size: usize,
	/// Hard cap on retained observation/action examples.
	pub maximum_samples: usize,
	/// Base seed used only for teacher episodes.
	pub environment_seed: u64,
	/// Seed used by deterministic epoch shuffling.
	pub shuffle_seed: u64,
	/// Learning rate for the temporary policy-only AdamW optimizer.
	pub learning_rate: f32,
}

impl Default for LunarTeacherConfig {
	fn default() -> Self {
		Self {
			episodes: 512,
			epochs: 24,
			batch_size: 2048,
			maximum_samples: 524_288,
			environment_seed: 0x5445_4143_4845_525f,
			shuffle_seed: 0x494d_4954_4154_455f,
			learning_rate: 1.0e-3,
		}
	}
}

/// Completed scripted-teacher dataset and optimization evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LunarTeacherMetrics {
	/// Scalar episodes that reached a boundary while collecting the dataset.
	pub episodes: u32,
	/// Episodes ending in a safe landing.
	pub safe_landings: u32,
	/// Episodes ending in a body impact.
	pub body_impacts: u32,
	/// Episodes ending in a hard foot impact.
	pub hard_foot_impacts: u32,
	/// Episodes ending outside the task bounds.
	pub out_of_bounds: u32,
	/// Episodes ending at the configured time limit.
	pub time_limits: u32,
	/// Episodes ending for any other reason.
	pub other_failures: u32,
	/// Retained observation/action examples.
	pub samples: usize,
	/// Policy-only optimizer steps completed.
	pub optimizer_steps: u64,
	/// Scripted action counts over the retained dataset.
	pub action_counts: [u64; 8],
	/// FNV-1a digest of the ordered, unshuffled dataset.
	pub dataset_digest: u64,
	/// Cross-entropy over the fixed dataset prefix before training.
	pub initial_loss: f32,
	/// Cross-entropy over the same prefix after training.
	pub final_loss: f32,
}

/// Fixed flat-terrain first-episode evaluation policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LunarFirstEpisodeEvaluationConfig {
	/// Number of independent vector lanes and expected episodes.
	pub environments: u32,
	/// Maximum steps retained for episode zero of every lane.
	pub horizon: usize,
	/// Maximum environment steps in one submission.
	pub submission_chunk_steps: usize,
	/// Held-out reset seed, which must remain disjoint from teacher data.
	pub environment_seed: u64,
}

impl Default for LunarFirstEpisodeEvaluationConfig {
	fn default() -> Self {
		Self {
			environments: 512,
			horizon: 1200,
			submission_chunk_steps: 16,
			environment_seed: 0x5049_4c4f_545f_4556,
		}
	}
}

/// Complete first-episode Lunar policy evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct LunarFirstEpisodeEvaluation {
	/// Number of episodes required by the evaluation.
	pub expected_episodes: u32,
	/// Number of lanes that reached a terminal or truncation boundary.
	pub completed_episodes: u32,
	/// Individual lane transitions recorded, including terminal padding.
	pub recorded_environment_steps: u64,
	/// Exact number of Vulkan submissions.
	pub submissions: u32,
	/// Episodes ending in a safe landing.
	pub safe_landings: u32,
	/// Episodes ending in a body impact.
	pub body_impacts: u32,
	/// Episodes ending in a hard foot impact.
	pub hard_foot_impacts: u32,
	/// Episodes ending outside the task bounds.
	pub out_of_bounds: u32,
	/// Episodes ending after a numerical failure.
	pub numerical_failures: u32,
	/// Episodes ending at the time limit.
	pub time_limits: u32,
	/// Episodes stopped by an external controller.
	pub external_stops: u32,
	/// Episodes ended by an invalid action.
	pub invalid_actions: u32,
	/// Lanes that did not finish within the bounded horizon.
	pub incomplete_episodes: u32,
	/// Safe landings divided by all expected episodes.
	pub safe_landing_rate: f64,
	/// Wilson-score 95% lower confidence bound for safe landing rate.
	pub wilson_lower_95: f64,
	/// Mean return over all expected episodes.
	pub mean_return: f64,
	/// Minimum return over all expected episodes.
	pub minimum_return: f64,
	/// Maximum return over all expected episodes.
	pub maximum_return: f64,
	/// Mean simulated episode length.
	pub mean_episode_steps: f64,
	/// Mean terminal or final fuel.
	pub mean_fuel_remaining: f64,
	/// Mean terminal or final linear speed.
	pub mean_terminal_linear_speed: f64,
	/// Mean terminal or final angular speed.
	pub mean_terminal_angular_speed: f64,
	/// Mean maximum foot-contact impulse.
	pub mean_maximum_foot_impulse: f64,
	/// Greedy action counts before each lane's first boundary.
	pub action_counts: [u64; 8],
	/// FNV-1a digest of every recorded greedy action.
	pub action_trace_digest: u64,
	/// FNV-1a digest of every recorded critic value.
	pub value_trace_digest: u64,
}

impl Default for LunarFirstEpisodeEvaluation {
	fn default() -> Self {
		Self {
			expected_episodes: 0,
			completed_episodes: 0,
			recorded_environment_steps: 0,
			submissions: 0,
			safe_landings: 0,
			body_impacts: 0,
			hard_foot_impacts: 0,
			out_of_bounds: 0,
			numerical_failures: 0,
			time_limits: 0,
			external_stops: 0,
			invalid_actions: 0,
			incomplete_episodes: 0,
			safe_landing_rate: 0.0,
			wilson_lower_95: 0.0,
			mean_return: 0.0,
			minimum_return: 0.0,
			maximum_return: 0.0,
			mean_episode_steps: 0.0,
			mean_fuel_remaining: 0.0,
			mean_terminal_linear_speed: 0.0,
			mean_terminal_angular_speed: 0.0,
			mean_maximum_foot_impulse: 0.0,
			action_counts: [0; 8],
			action_trace_digest: 0,
			value_trace_digest: 0,
		}
	}
}

#[derive(Clone, Copy)]
struct TeacherSample {
	observation: [f32; LUNAR_OBSERVATION_SIZE],
	action: i32,
}

/// Record, submit, and wait for one complete Lunar PPO rollout.
///
/// # Errors
///
/// Returns an error from environment reset/recording/submission or PPO
/// collection. A rejected recording rewinds both owners transactionally.
pub fn collect_ppo_rollout(
	environment: &mut LunarLander3dVector<'_>,
	trainer: &mut PpoTrainer<'_, '_>,
) -> Result<()> {
	if environment.execution().submission_count() == 0
		&& !environment.execution().has_active_recording()
	{
		environment.reset(environment.config().seed)?;
	}
	environment.begin()?;
	trainer.begin_collection()?;
	let horizon = trainer.config().horizon;
	let recorded = (|| {
		for _ in 0..horizon {
			let policy = trainer.act(environment.observation())?;
			let transition = Environment::step(environment, &policy.action)?;
			trainer.observe(
				transition.observation(),
				transition.next_observation(),
				transition.reward(),
				transition.terminated(),
				transition.truncated(),
				&policy,
			)?;
			environment.reset_completed()?;
		}
		trainer.end_collection()
	})();
	if let Err(error) = recorded {
		let _ = environment.cancel();
		let _ = trainer.abort_collection();
		return Err(error);
	}
	let event = match environment.submit() {
		Ok(event) => event,
		Err(error) => {
			let _ = trainer.abort_collection();
			return Err(error);
		}
	};
	environment.wait(&event)
}

/// Pretrain only the categorical policy tower from the scripted scalar oracle.
///
/// The temporary optimizer has independent moment state. A caller-created PPO
/// optimizer therefore remains at step zero for a clean critic warm-up.
///
/// # Errors
///
/// Returns an error for an invalid curriculum, teacher/evaluator seed overlap,
/// scalar simulation failure, upload/model/loss/autograd failure, or incomplete
/// optimizer accounting.
pub fn pretrain_scripted_teacher(
	engine: &Engine,
	model: &CategoricalActorCritic,
	config: LunarTeacherConfig,
) -> Result<LunarTeacherMetrics> {
	validate_teacher_config(config)?;
	if config.environment_seed == LunarFirstEpisodeEvaluationConfig::default().environment_seed {
		return Err(Error::invalid_argument(
			"Lunar teacher and held-out evaluator seeds must be disjoint",
		));
	}
	let environment_config = LunarLander3dConfig::default();
	let mut samples = Vec::with_capacity(config.maximum_samples);
	let mut metrics = LunarTeacherMetrics::default();
	let mut digest = Digest64::new();
	digest.add_u64(environment_config.contract_fingerprint());
	digest.add_u64(config.environment_seed);
	for lane in 0..config.episodes {
		if samples.len() == config.maximum_samples {
			break;
		}
		let manifest = LunarEpisodeManifest::derive(
			config.environment_seed,
			lane,
			0,
			environment_config.contract_fingerprint(),
		);
		let mut environment = LunarScalarEnvironment::flat(environment_config, manifest)?;
		while !environment.state().terminated
			&& !environment.state().truncated
			&& samples.len() < config.maximum_samples
		{
			let observation = environment.observation();
			let action = scripted_landing_action(&environment_config, environment.state());
			for value in observation {
				if !value.is_finite() {
					return Err(Error::data_loss(
						"Lunar teacher produced a non-finite observation",
					));
				}
				digest.add_f32(value);
			}
			digest.add_i32(action as i32);
			metrics.action_counts[action as usize] += 1;
			samples.push(TeacherSample {
				observation,
				action: action as i32,
			});
			environment.step(action as u32, false)?;
		}
		if environment.state().terminated || environment.state().truncated {
			metrics.episodes += 1;
			match environment.state().end_reason {
				LunarEndReason::SafeLanding => metrics.safe_landings += 1,
				LunarEndReason::BodyImpact => metrics.body_impacts += 1,
				LunarEndReason::HardFootImpact => metrics.hard_foot_impacts += 1,
				LunarEndReason::OutOfBounds => metrics.out_of_bounds += 1,
				LunarEndReason::TimeLimit => metrics.time_limits += 1,
				_ => metrics.other_failures += 1,
			}
		}
	}
	if samples.is_empty() {
		return Err(Error::failed_precondition(
			"Lunar teacher produced no training samples",
		));
	}
	metrics.samples = samples.len();
	metrics.dataset_digest = digest.value();
	let probe_samples = samples.len().min(config.batch_size);
	metrics.initial_loss = evaluate_teacher_probe(engine, model, &samples[..probe_samples])?;

	let policy_parameters = model
		.all_named_parameters()?
		.into_iter()
		.filter(|named| named.path().starts_with("policy"))
		.map(|named| named.parameter())
		.collect::<Vec<_>>();
	if policy_parameters.is_empty() {
		return Err(Error::internal(
			"Lunar teacher could not resolve policy parameters",
		));
	}
	let mut optimizer = AdamW::with_hyperparameters(
		policy_parameters,
		config.learning_rate,
		0.9,
		0.999,
		1.0e-8,
		0.0,
	)?;
	let steps_per_epoch = samples.len().div_ceil(config.batch_size);
	let total_steps = u64::from(config.epochs)
		.checked_mul(steps_per_epoch as u64)
		.ok_or_else(|| Error::resource_exhausted("Lunar teacher step count overflows u64"))?;
	let mut order = (0..samples.len()).collect::<Vec<_>>();
	let mut random = Shuffle64::new(config.shuffle_seed);
	{
		let mut training = ItTraining::new_eager(
			engine,
			&mut optimizer,
			ItTrainingConfig {
				total_steps,
				steps_per_epoch: steps_per_epoch as u64,
				batch_size: config.batch_size as u64,
				timer_name: "lunar_teacher_imitation".into(),
				..ItTrainingConfig::default()
			},
		)?;
		let mut completed_steps = 0_u64;
		while training.begin_step()? {
			let step_in_epoch = completed_steps as usize % steps_per_epoch;
			if step_in_epoch == 0 {
				random.shuffle(&mut order);
			}
			let begin = step_in_epoch * config.batch_size;
			let end = (begin + config.batch_size).min(samples.len());
			let (observation, action) = upload_teacher_batch(engine, &samples, &order[begin..end])?;
			training.seal_replay_inputs()?;
			training.zero_grad();
			let tape = GradientTape::new();
			let logits = model.evaluate(&observation)?.logits;
			let batch_loss = loss::cross_entropy(&logits, &action)?;
			tape.backward(&batch_loss)?;
			training.complete_step(&batch_loss)?;
			completed_steps += 1;
			if !training.snapshot().last_loss().is_some_and(f32::is_finite) {
				return Err(Error::data_loss("Lunar teacher loss became non-finite"));
			}
		}
		training.finish()?;
		metrics.optimizer_steps = completed_steps;
	}
	if u64::from(optimizer.step_count()) != metrics.optimizer_steps {
		return Err(Error::internal(
			"Lunar teacher optimizer accounting diverged",
		));
	}
	metrics.final_loss = evaluate_teacher_probe(engine, model, &samples[..probe_samples])?;
	Ok(metrics)
}

/// Evaluate episode zero of a fresh flat vector environment with greedy TopK.
///
/// Completed lanes are not reset. Bounded rollout chunks retain action, value,
/// reward, and boundary history without a per-step host synchronization.
///
/// # Errors
///
/// Returns an error for invalid dimensions, ownership/model/environment/runtime
/// failure, malformed device results, or inconsistent final telemetry.
pub fn evaluate_first_episodes(
	engine: &Engine,
	model: &dyn ActorCritic,
	config: LunarFirstEpisodeEvaluationConfig,
) -> Result<LunarFirstEpisodeEvaluation> {
	if config.environments == 0 || config.horizon == 0 || config.submission_chunk_steps == 0 {
		return Err(Error::invalid_argument(
			"Lunar first-episode evaluation dimensions must be nonzero",
		));
	}
	for parameter in model.all_parameters()? {
		if !engine.owns_matrix(&parameter.data()) {
			return Err(Error::invalid_argument(
				"Lunar evaluation model must belong to the selected Engine",
			));
		}
	}
	let mut environment = LunarLander3dVector::flat(
		engine,
		LunarLander3dVectorConfig {
			environments: config.environments,
			seed: config.environment_seed,
			environment: LunarLander3dConfig::default(),
		},
	)?;
	let evaluation = evaluate_first_episodes_impl(&mut environment, model, config);
	let closed = environment.close();
	match (evaluation, closed) {
		(Err(error), _) => Err(error),
		(Ok(_), Err(error)) => Err(error),
		(Ok(evaluation), Ok(())) => Ok(evaluation),
	}
}

fn evaluate_first_episodes_impl(
	environment: &mut LunarLander3dVector<'_>,
	model: &dyn ActorCritic,
	config: LunarFirstEpisodeEvaluationConfig,
) -> Result<LunarFirstEpisodeEvaluation> {
	let lanes = config.environments as usize;
	let mut result = LunarFirstEpisodeEvaluation {
		expected_episodes: config.environments,
		minimum_return: f64::INFINITY,
		maximum_return: f64::NEG_INFINITY,
		..LunarFirstEpisodeEvaluation::default()
	};
	let mut action_digest = Digest64::new();
	let mut value_digest = Digest64::new();
	let mut completed = vec![false; lanes];
	let mut terminal_reasons = vec![LunarEndReason::None; lanes];
	let mut accumulated_returns = vec![0.0_f32; lanes];
	let mut chunk_start = 0_usize;
	while chunk_start < config.horizon {
		let chunk_steps = config
			.submission_chunk_steps
			.min(config.horizon - chunk_start);
		environment.begin()?;
		let mut history = RolloutBuffer::new(
			environment.engine(),
			RolloutConfig {
				time: chunk_steps,
				environments: lanes,
				observation_shape: vec![LUNAR_OBSERVATION_SIZE],
			},
		)?;
		for _ in 0..chunk_steps {
			let observation = environment.observation().clone();
			let network = model.evaluate(&observation)?;
			let best = matrix::top_k(&network.logits, 1, 1)?;
			let action = matrix::reshape(&best.indices, [lanes])?;
			let selected = policy::evaluate_categorical(&network.logits, &action, &network.value)?;
			let transition = Environment::step(environment, &action)?;
			history.append(&RolloutTransition::new(
				transition.observation().clone(),
				action,
				transition.reward().clone(),
				network.value.clone(),
				network.value,
				selected.log_probability,
				transition.terminated().clone(),
				transition.truncated().clone(),
			))?;
		}
		let event = environment.submit()?;
		environment.wait(&event)?;
		result.submissions = result
			.submissions
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("Lunar evaluation submission count overflows"))?;
		let actions = history.batch().action().read::<i32>()?;
		let values = history.batch().value().read_f32()?;
		let rewards = history.batch().reward().read_f32()?;
		let terminated = history.batch().terminated().read::<u8>()?;
		let truncated = history.batch().truncated().read::<u8>()?;
		let reasons = environment.end_reason().read::<u32>()?;
		let expected = chunk_steps
			.checked_mul(lanes)
			.ok_or_else(|| Error::resource_exhausted("Lunar evaluation history size overflows"))?;
		if actions.len() != expected
			|| values.len() != expected
			|| rewards.len() != expected
			|| terminated.len() != expected
			|| truncated.len() != expected
			|| reasons.len() != lanes
		{
			return Err(Error::data_loss(
				"Lunar evaluation history has an unexpected extent",
			));
		}
		for step in 0..chunk_steps {
			for lane in 0..lanes {
				let index = step * lanes + lane;
				let action = actions[index];
				let value = values[index];
				let reward = rewards[index];
				if !(0..8).contains(&action) {
					return Err(Error::data_loss(
						"Lunar evaluation produced an invalid greedy action",
					));
				}
				if !value.is_finite() || !reward.is_finite() {
					return Err(Error::data_loss(
						"Lunar evaluation produced a non-finite value or reward",
					));
				}
				action_digest.add_i32(action);
				value_digest.add_f32(value);
				let is_completed = terminated[index] != 0 || truncated[index] != 0;
				if completed[lane] {
					if !is_completed || reward != 0.0 {
						return Err(Error::data_loss(
							"Lunar completed lane did not remain terminal and reward-free",
						));
					}
				} else {
					result.action_counts[action as usize] += 1;
					accumulated_returns[lane] += reward;
					if !accumulated_returns[lane].is_finite() {
						return Err(Error::data_loss(
							"Lunar evaluation return became non-finite",
						));
					}
					if is_completed {
						completed[lane] = true;
					}
				}
			}
		}
		for lane in 0..lanes {
			if completed[lane] {
				let reason = end_reason_from_u32(reasons[lane])?;
				if reason == LunarEndReason::None
					|| (terminal_reasons[lane] != LunarEndReason::None && terminal_reasons[lane] != reason)
				{
					return Err(Error::data_loss(
						"Lunar evaluation terminal reason changed after completion",
					));
				}
				terminal_reasons[lane] = reason;
			}
		}
		result.recorded_environment_steps = result
			.recorded_environment_steps
			.checked_add(expected as u64)
			.ok_or_else(|| Error::resource_exhausted("Lunar evaluation step count overflows"))?;
		chunk_start += chunk_steps;
		if completed.iter().all(|value| *value) {
			break;
		}
	}
	if u64::from(result.submissions) != environment.execution().submission_count() {
		return Err(Error::internal(
			"Lunar evaluation submission accounting diverged",
		));
	}
	let telemetry = environment.copy_episode_telemetry()?;
	if telemetry.len() != lanes {
		return Err(Error::data_loss(
			"Lunar evaluation telemetry has an unexpected lane count",
		));
	}
	let mut return_sum = 0.0_f64;
	let mut step_sum = 0.0_f64;
	let mut fuel_sum = 0.0_f64;
	let mut linear_speed_sum = 0.0_f64;
	let mut angular_speed_sum = 0.0_f64;
	let mut foot_impulse_sum = 0.0_f64;
	for (lane, episode) in telemetry.into_iter().enumerate() {
		let is_completed = episode.terminated || episode.truncated;
		if !episode.is_valid()
			|| episode.episode_step as usize > config.horizon
			|| completed[lane] != is_completed
			|| (is_completed && terminal_reasons[lane] != episode.end_reason)
		{
			return Err(Error::data_loss(
				"Lunar final telemetry disagrees with the recorded first episode",
			));
		}
		let difference =
			(f64::from(episode.episode_return) - f64::from(accumulated_returns[lane])).abs();
		let tolerance = 1.0e-3 + 2.0e-5 * f64::from(episode.episode_return).abs();
		if difference > tolerance {
			return Err(Error::data_loss(
				"Lunar final return disagrees with transition history",
			));
		}
		if is_completed {
			result.completed_episodes += 1;
		}
		match episode.end_reason {
			LunarEndReason::None => result.incomplete_episodes += 1,
			LunarEndReason::SafeLanding => result.safe_landings += 1,
			LunarEndReason::BodyImpact => result.body_impacts += 1,
			LunarEndReason::HardFootImpact => result.hard_foot_impacts += 1,
			LunarEndReason::OutOfBounds => result.out_of_bounds += 1,
			LunarEndReason::NumericalFailure => result.numerical_failures += 1,
			LunarEndReason::TimeLimit => result.time_limits += 1,
			LunarEndReason::ExternalStop => result.external_stops += 1,
			LunarEndReason::InvalidAction => result.invalid_actions += 1,
		}
		let episode_return = f64::from(episode.episode_return);
		return_sum += episode_return;
		step_sum += f64::from(episode.episode_step);
		fuel_sum += f64::from(episode.fuel_remaining);
		linear_speed_sum += f64::from(episode.terminal_linear_speed);
		angular_speed_sum += f64::from(episode.terminal_angular_speed);
		foot_impulse_sum += f64::from(episode.maximum_foot_impulse);
		result.minimum_return = result.minimum_return.min(episode_return);
		result.maximum_return = result.maximum_return.max(episode_return);
	}
	let reason_count = result.safe_landings
		+ result.body_impacts
		+ result.hard_foot_impacts
		+ result.out_of_bounds
		+ result.numerical_failures
		+ result.time_limits
		+ result.external_stops
		+ result.invalid_actions;
	if reason_count != result.completed_episodes
		|| result.completed_episodes + result.incomplete_episodes != result.expected_episodes
	{
		return Err(Error::data_loss(
			"Lunar evaluation terminal-reason accounting diverged",
		));
	}
	let episodes = f64::from(result.expected_episodes);
	result.safe_landing_rate = f64::from(result.safe_landings) / episodes;
	result.wilson_lower_95 = wilson_lower_95(result.safe_landings, result.expected_episodes);
	result.mean_return = return_sum / episodes;
	result.mean_episode_steps = step_sum / episodes;
	result.mean_fuel_remaining = fuel_sum / episodes;
	result.mean_terminal_linear_speed = linear_speed_sum / episodes;
	result.mean_terminal_angular_speed = angular_speed_sum / episodes;
	result.mean_maximum_foot_impulse = foot_impulse_sum / episodes;
	result.action_trace_digest = action_digest.value();
	result.value_trace_digest = value_digest.value();
	for value in [
		result.safe_landing_rate,
		result.wilson_lower_95,
		result.mean_return,
		result.minimum_return,
		result.maximum_return,
		result.mean_episode_steps,
		result.mean_fuel_remaining,
		result.mean_terminal_linear_speed,
		result.mean_terminal_angular_speed,
		result.mean_maximum_foot_impulse,
	] {
		if !value.is_finite() {
			return Err(Error::data_loss(
				"Lunar evaluation aggregate became non-finite",
			));
		}
	}
	Ok(result)
}

fn validate_teacher_config(config: LunarTeacherConfig) -> Result<()> {
	if config.episodes == 0
		|| config.epochs == 0
		|| config.batch_size == 0
		|| config.maximum_samples == 0
		|| !config.learning_rate.is_finite()
		|| config.learning_rate <= 0.0
	{
		return Err(Error::invalid_argument(
			"Lunar teacher requires nonzero dimensions and a positive finite learning rate",
		));
	}
	Ok(())
}

fn upload_teacher_batch(
	engine: &Engine,
	samples: &[TeacherSample],
	order: &[usize],
) -> Result<(Matrix, Matrix)> {
	let capacity = order
		.len()
		.checked_mul(LUNAR_OBSERVATION_SIZE)
		.ok_or_else(|| Error::resource_exhausted("Lunar teacher batch size overflows"))?;
	let mut observations = Vec::with_capacity(capacity);
	let mut actions = Vec::with_capacity(order.len());
	for index in order {
		let sample = samples
			.get(*index)
			.ok_or_else(|| Error::internal("Lunar teacher shuffle index is invalid"))?;
		observations.extend_from_slice(&sample.observation);
		actions.push(sample.action);
	}
	Ok((
		Matrix::from_f32(engine, [order.len(), LUNAR_OBSERVATION_SIZE], &observations)?,
		Matrix::from_slice(engine, [order.len()], &actions)?,
	))
}

fn evaluate_teacher_probe(
	engine: &Engine,
	model: &CategoricalActorCritic,
	samples: &[TeacherSample],
) -> Result<f32> {
	let order = (0..samples.len()).collect::<Vec<_>>();
	let (observation, action) = upload_teacher_batch(engine, samples, &order)?;
	let logits = model.evaluate(&observation)?.logits;
	let value = loss::cross_entropy(&logits, &action)?.read_f32()?[0];
	if !value.is_finite() {
		return Err(Error::data_loss(
			"Lunar teacher loss probe became non-finite",
		));
	}
	Ok(value)
}

fn end_reason_from_u32(value: u32) -> Result<LunarEndReason> {
	match value {
		0 => Ok(LunarEndReason::None),
		1 => Ok(LunarEndReason::SafeLanding),
		2 => Ok(LunarEndReason::BodyImpact),
		3 => Ok(LunarEndReason::HardFootImpact),
		4 => Ok(LunarEndReason::OutOfBounds),
		5 => Ok(LunarEndReason::NumericalFailure),
		6 => Ok(LunarEndReason::TimeLimit),
		7 => Ok(LunarEndReason::ExternalStop),
		8 => Ok(LunarEndReason::InvalidAction),
		_ => Err(Error::data_loss(
			"Lunar evaluation produced an unknown terminal reason",
		)),
	}
}

fn wilson_lower_95(successes: u32, trials: u32) -> f64 {
	if trials == 0 {
		return 0.0;
	}
	const Z: f64 = 1.959_963_984_540_054;
	let trials = f64::from(trials);
	let proportion = f64::from(successes) / trials;
	let z_squared = Z * Z;
	let denominator = 1.0 + z_squared / trials;
	let center = proportion + z_squared / (2.0 * trials);
	let radius = Z * ((proportion * (1.0 - proportion) + z_squared / (4.0 * trials)) / trials).sqrt();
	((center - radius) / denominator).max(0.0)
}

struct Digest64(u64);

impl Digest64 {
	const fn new() -> Self {
		Self(14_695_981_039_346_656_037)
	}

	fn add_u32(&mut self, value: u32) {
		for byte in value.to_le_bytes() {
			self.0 ^= u64::from(byte);
			self.0 = self.0.wrapping_mul(1_099_511_628_211);
		}
	}

	fn add_i32(&mut self, value: i32) {
		self.add_u32(value as u32);
	}

	fn add_f32(&mut self, value: f32) {
		self.add_u32(value.to_bits());
	}

	fn add_u64(&mut self, value: u64) {
		self.add_u32(value as u32);
		self.add_u32((value >> 32) as u32);
	}

	const fn value(&self) -> u64 {
		self.0
	}
}

struct Shuffle64(u64);

impl Shuffle64 {
	const fn new(seed: u64) -> Self {
		Self(seed)
	}

	fn next(&mut self) -> u64 {
		self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
		let mut value = self.0;
		value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
		value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
		value ^ (value >> 31)
	}

	fn shuffle<T>(&mut self, values: &mut [T]) {
		for index in (1..values.len()).rev() {
			let bound = index as u64 + 1;
			let threshold = bound.wrapping_neg() % bound;
			let selected = loop {
				let random = self.next();
				if random >= threshold {
					break (random % bound) as usize;
				}
			};
			values.swap(index, selected);
		}
	}
}
