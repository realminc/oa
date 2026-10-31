//! Bounded PPO update replay. Donor objective/schedule: oa/ml/ppoTrainer.cpp.
//! Reuses OA capture, GradientTape and TrainingProgram; never owns execution.
use super::{ItRolloutTraining, PpoTrainerConfig, PpoTrainerMetrics, TrainingProgram};
use crate::ml::{ActorCritic, GradientTape, Parameter, RolloutBatch, advantage, loss, policy};
use crate::{Engine, Error, ExecutionPlan, Matrix, Result, matrix};
use std::time::Duration;

/// Reusable fixed-geometry PPO backward and AdamW programs.
///
/// Retain this value across trainers when checkpointing between rollouts. Each
/// minibatch reuses one bounded tape's storage, accumulating sample-weighted
/// gradients before exactly one AdamW step per epoch. Tail shapes compile once.
/// Drop never submits or waits. Different models/configurations are rejected.
#[derive(Default)]
pub struct PpoUpdateProgram {
	state: Option<State>,
	pub(super) rollout: Option<crate::ml::RolloutBuffer>,
}
struct State {
	config: PpoTrainerConfig,
	parameters: Vec<Parameter>,
	ids: Vec<u64>,
	accumulators: Vec<Matrix>,
	inputs: [Matrix; 5],
	normalize: ExecutionPlan,
	cursor: Matrix,
	metrics: [Matrix; 4],
	batches: Vec<(usize, ExecutionPlan)>,
	optimizer: TrainingProgram,
	epochs: u64,
}
impl PpoUpdateProgram {
	/// Number of complete epochs submitted through the compiled program.
	pub fn replay_count(&self) -> u64 {
		self.state.as_ref().map_or(0, |s| s.epochs)
	}
	/// Full graph-command recordings, excluding timestamp bracket commands.
	pub fn command_recording_count(&self) -> u64 {
		self.state.as_ref().map_or(0, |s| {
			s.normalize.diagnostics().command_recording_count()
				+ s.optimizer.diagnostics().command_recording_count()
				+ s
					.batches
					.iter()
					.map(|(_, p)| p.diagnostics().command_recording_count())
					.sum::<u64>()
		})
	}

	/// Whether the fixed update programs have been compiled.
	pub fn is_compiled(&self) -> bool {
		self.state.is_some()
	}
	pub(super) fn update(
		&mut self,
		engine: &Engine,
		model: &dyn ActorCritic,
		config: &PpoTrainerConfig,
		rollout: &RolloutBatch,
		training: &mut ItRolloutTraining<'_, '_>,
	) -> Result<PpoTrainerMetrics> {
		let batch = config
			.horizon
			.checked_mul(config.environments)
			.ok_or_else(|| Error::resource_exhausted("PPO batch overflow"))?;
		let width = config.observation_shape.iter().product::<usize>();
		let inputs = [
			matrix::reshape(rollout.observation(), [batch, width])?,
			matrix::reshape(rollout.action(), [batch])?,
			matrix::reshape(rollout.old_log_probability(), [batch])?,
			matrix::reshape(rollout.advantage(), [batch])?,
			matrix::reshape(rollout.returns(), [batch])?,
		];
		engine.checkpoint()?.wait()?;
		if self.state.is_none() {
			self.state = Some(State::capture(
				engine,
				model,
				config,
				inputs.clone(),
				training,
			)?);
		}
		let s = self.state.as_mut().expect("captured state");
		let ids = model
			.all_parameters()?
			.iter()
			.map(|p| p.data().value_id())
			.collect::<Vec<_>>();
		if ids != s.ids
			|| config.horizon != s.config.horizon
			|| config.environments != s.config.environments
			|| config.observation_shape != s.config.observation_shape
			|| config.update_minibatch_size != s.config.update_minibatch_size
			|| config.loss != s.config.loss
		{
			return Err(Error::failed_precondition(
				"compiled PPO model/geometry/loss changed",
			));
		}
		for (old, new) in s.inputs.iter().zip(&inputs) {
			if old.same_value_as(new) {
				continue;
			}
			if old.same_value_as(&s.inputs[3]) {
				s.normalize.bind_matrix_input(old, new)?;
			} else {
				for (_, plan) in &mut s.batches {
					plan.bind_matrix_input(old, new)?;
				}
			}
		}
		// Whole-rollout advantages are normalized once, before any microbatch.
		let mut elapsed = submit(engine, &s.normalize, config.enable_gpu_timing)?;
		let mini = if config.update_minibatch_size == 0 {
			batch
		} else {
			config.update_minibatch_size
		};
		for start in (0..batch).step_by(mini) {
			let count = mini.min(batch - start);
			let (_, plan) = s
				.batches
				.iter_mut()
				.find(|(n, _)| *n == count)
				.expect("compiled tail");
			// Completion precedes overwriting the small stable cursor slot. No observations
			// or gradients cross the host boundary.
			plan.upload_matrix_input(
				&s.cursor,
				&[
					u32::try_from(start).map_err(|_| Error::resource_exhausted("PPO cursor overflow"))?,
					u32::from(start == 0),
				],
			)?;
			elapsed = elapsed
				.checked_add(submit(engine, plan, config.enable_gpu_timing)?)
				.ok_or_else(|| Error::resource_exhausted("PPO timing overflow"))?;
		}
		let mut metrics = [0.0; 4];
		for (out, value) in metrics.iter_mut().zip(&s.metrics) {
			*out = value.read_f32()?[0];
			if !out.is_finite() {
				return Err(Error::failed_precondition("compiled PPO metric not finite"));
			}
		}
		for (p, g) in s.parameters.iter().zip(&s.accumulators) {
			p.restore_gradient(g.clone());
		}
		training.complete_compiled_update(&mut s.optimizer, elapsed)?;
		s.epochs = s
			.epochs
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("PPO replay count overflow"))?;
		Ok(PpoTrainerMetrics {
			rollout: training.rollout_index(),
			update_epoch: training.update_epoch(),
			total_loss: metrics[0],
			policy_loss: metrics[1],
			value_loss: metrics[2],
			entropy: metrics[3],
		})
	}
}
impl State {
	fn capture(
		engine: &Engine,
		model: &dyn ActorCritic,
		config: &PpoTrainerConfig,
		inputs: [Matrix; 5],
		training: &mut ItRolloutTraining<'_, '_>,
	) -> Result<Self> {
		let parameters = model.all_parameters()?;
		let ids = parameters.iter().map(|p| p.data().value_id()).collect();
		let accumulators = parameters
			.iter()
			.map(|p| {
				let d = p.data();
				Matrix::new_from_f32(engine, d.shape().to_vec(), &vec![0.; d.num_elements()])
			})
			.collect::<Result<Vec<_>>>()?;
		let metrics: [Result<Matrix>; 4] =
			std::array::from_fn(|_| Matrix::new_from_f32(engine, [], &[0.]));
		let metrics: [Matrix; 4] = metrics
			.into_iter()
			.collect::<Result<Vec<_>>>()?
			.try_into()
			.map_err(|_| Error::internal("PPO metric count"))?;
		let cursor = Matrix::new_from_slice(engine, [2], &[0_u32, 1])?;
		engine.checkpoint()?.wait()?;
		let normalized = Matrix::new_from_f32(
			engine,
			inputs[3].shape().to_vec(),
			&vec![0.; inputs[3].num_elements()],
		)?;
		let copy_cursor = Matrix::new_from_slice(engine, [2], &[0_u32, 1])?;
		let (normalize, ()) = engine.capture_with_initialized_state(&[&normalized], || {
			crate::ml::matrix::training_accumulate(
				&normalized,
				&advantage::normalize(&inputs[3], 1e-8)?,
				&copy_cursor,
				1.,
			)
		})?;
		let batch = config.horizon * config.environments;
		let mini = if config.update_minibatch_size == 0 {
			batch
		} else {
			config.update_minibatch_size
		};
		let mut counts = vec![mini];
		if !batch.is_multiple_of(mini) {
			counts.push(batch % mini);
		}
		let mut batches = Vec::new();
		for count in counts {
			for p in &parameters {
				p.clear_gradient();
			}
			let mutable = accumulators
				.iter()
				.chain(metrics.iter())
				.collect::<Vec<_>>();
			let (plan, ()) = engine.capture_with_initialized_state(&mutable, || {
				let [obs, actions, logs, adv, ret] = crate::ml::rollout::minibatch_cursor(
					[&inputs[0], &inputs[1], &inputs[2], &normalized, &inputs[4]],
					&cursor,
					count,
				)?;
				let tape = GradientTape::new();
				let net = model.evaluate(&obs)?;
				let p = policy::evaluate_categorical(&net.logits, &actions, &net.value)?;
				let l = loss::ppo(
					&p.log_probability,
					&logs,
					&adv,
					&p.value,
					&ret,
					&p.entropy,
					config.loss,
				)?;
				let total = model.auxiliary_loss(&l.total_loss)?;
				let weight = count as f32 / batch as f32;
				tape.backward(&matrix::scale(&total, weight)?)?;
				for (p, target) in parameters.iter().zip(&accumulators) {
					let g = p
						.gradient()
						.ok_or_else(|| Error::failed_precondition("compiled PPO missing gradient"))?;
					crate::ml::matrix::training_accumulate(target, &g, &cursor, 1.)?;
				}
				for (target, value) in
					metrics
						.iter()
						.zip([&total, &l.policy_loss, &l.value_loss, &l.entropy])
				{
					crate::ml::matrix::training_accumulate(target, value, &cursor, weight)?;
				}
				Ok(())
			})?;
			batches.push((count, plan));
		}
		let optimizer =
			TrainingProgram::capture(engine, training.update_loop_mut().adamw_mut()?, || {
				for (p, g) in parameters.iter().zip(&accumulators) {
					p.restore_gradient(g.clone());
				}
				matrix::scale(&metrics[0], 1.)
			})?;
		Ok(Self {
			config: config.clone(),
			parameters,
			ids,
			accumulators,
			inputs,
			normalize,
			cursor,
			metrics,
			batches,
			optimizer,
			epochs: 0,
		})
	}
}
fn submit(engine: &Engine, plan: &ExecutionPlan, timed: bool) -> Result<Duration> {
	if timed {
		engine.submit_timed(plan)?.device_duration()
	} else {
		engine.submit(plan)?.wait()?;
		Ok(Duration::ZERO)
	}
}
