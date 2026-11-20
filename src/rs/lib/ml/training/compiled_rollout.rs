//! Generic compiled PPO collection over [`CompiledEnvironment`] (OA-CC-20).
//!
//! Seven cursor words are uploaded once per chunk and advance on device;
//! reset, step and GAE plans are captured once and replayed for the whole
//! horizon, so collection submits recorded plans instead of recording host
//! commands per step. See `oaCompiledEnvironment.md`.

use super::PpoTrainerConfig;
use crate::ml::environment::CompiledEnvironment;
use crate::ml::{ActorCritic, RolloutBuffer, RolloutTransition, policy};
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{
	DType, Engine, Error, ExecutionPlan, ExecutionPlanDiagnostics, Matrix, OpAttribute, Result,
	matrix,
};
use std::time::Duration;

/// Per-collection recipe for [`CompiledRollout::collect`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompiledCollectionOptions {
	/// Reset-seed words uploaded into cursor words 3,4.
	pub seed: u64,
	/// Optional lane-status check cadence in steps; zero disables checks.
	pub check_every: u32,
	/// Reward shaping offset applied to every collected step.
	pub shaping: f32,
	/// Continue a committed environment instead of a full reset.
	pub continuing: bool,
}

/// Reusable compiled collection owner: captured reset/step/GAE plans plus
/// replay accounting.
pub struct CompiledRollout {
	state: Option<State>,
}

/// Diagnostics for the plans a [`CompiledRollout`] captured at collection.
///
/// Each field reports the same node/barrier/descriptor/push/peak-memory and
/// alias census as [`ExecutionPlan::diagnostics`]; `resume` is present only
/// once a continuing collection has captured the resume plan.
#[derive(Clone, Copy, Debug)]
pub struct CompiledRolloutDiagnostics {
	reset: ExecutionPlanDiagnostics,
	resume: Option<ExecutionPlanDiagnostics>,
	step: ExecutionPlanDiagnostics,
	gae: ExecutionPlanDiagnostics,
}

impl CompiledRolloutDiagnostics {
	/// Diagnostics for the reset plan.
	pub const fn reset(self) -> ExecutionPlanDiagnostics {
		self.reset
	}

	/// Diagnostics for the continuing-collection resume plan, when captured.
	pub const fn resume(self) -> Option<ExecutionPlanDiagnostics> {
		self.resume
	}

	/// Diagnostics for the per-step replay plan.
	pub const fn step(self) -> ExecutionPlanDiagnostics {
		self.step
	}

	/// Diagnostics for the batched GAE successor-evaluation plan.
	pub const fn gae(self) -> ExecutionPlanDiagnostics {
		self.gae
	}
}

struct State {
	env: Vec<u64>,
	rollout: Vec<u64>,
	parameters: Vec<u64>,
	ppo: PpoTrainerConfig,
	shaping: f32,
	observation: Matrix,
	cursor: Matrix,
	reset: ExecutionPlan,
	resume: Option<ExecutionPlan>,
	step: ExecutionPlan,
	gae: ExecutionPlan,
	replays: u64,
}

impl CompiledRollout {
	/// Construct an empty collection owner; plans capture on first collect.
	pub const fn new() -> Self {
		Self { state: None }
	}

	/// Number of replayed step-plan submissions since capture.
	pub fn replay_count(&self) -> u64 {
		self.state.as_ref().map_or(0, |state| state.replays)
	}

	/// Number of full graph-command recordings, excluding timestamp brackets.
	pub fn command_recording_count(&self) -> u64 {
		self.state.as_ref().map_or(0, |s| {
			s.reset.diagnostics().command_recording_count()
				+ s.step.diagnostics().command_recording_count()
				+ s.gae.diagnostics().command_recording_count()
				+ s
					.resume
					.as_ref()
					.map_or(0, |p| p.diagnostics().command_recording_count())
		})
	}

	/// Per-plan diagnostics for the captured collection, `None` before the
	/// first [`collect`](Self::collect) captures plans.
	pub fn diagnostics(&self) -> Option<CompiledRolloutDiagnostics> {
		self.state.as_ref().map(|state| CompiledRolloutDiagnostics {
			reset: state.reset.diagnostics(),
			resume: state.resume.as_ref().map(|plan| plan.diagnostics()),
			step: state.step.diagnostics(),
			gae: state.gae.diagnostics(),
		})
	}

	/// Collect one PPO horizon by replaying captured plans.
	///
	/// Captures reset/step/GAE plans on first use, then replays them per step.
	/// Any change to environment identity, rollout storage, model parameters
	/// or PPO geometry is rejected rather than silently re-captured.
	///
	/// # Errors
	///
	/// Returns precondition/identity mismatch, capture, submission, readback
	/// or status failures.
	pub fn collect<E: CompiledEnvironment>(
		&mut self,
		engine: &Engine,
		model: &dyn ActorCritic,
		env: &mut E,
		rollout: &mut RolloutBuffer,
		ppo: &PpoTrainerConfig,
		options: CompiledCollectionOptions,
	) -> Result<Duration> {
		let CompiledCollectionOptions {
			seed,
			check_every,
			shaping,
			continuing,
		} = options;
		if !env.replay_valid()
			|| !shaping.is_finite()
			|| !engine.owns_matrix(env.observation())
			|| env.environments() as usize != ppo.environments
		{
			return Err(Error::invalid_argument(
				"compiled collection geometry/Engine/shaping mismatch",
			));
		}
		engine.checkpoint()?.wait()?;
		let reset_environment = !continuing || !env.replay_initialized();
		if self.state.is_none() {
			self.state = Some(State::capture(
				engine, model, env, rollout, ppo, shaping, continuing,
			)?);
		}
		let s = self.state.as_mut().expect("compiled collection state");
		let parameters = model
			.all_parameters()?
			.iter()
			.map(|p| p.data().value_id())
			.collect::<Vec<_>>();
		if rollout
			.replay_slots()
			.iter()
			.map(|m| m.value_id())
			.collect::<Vec<_>>()
			!= s.rollout
			|| env.replay_identity() != s.env
			|| parameters != s.parameters
			|| s.shaping != shaping
			|| s.resume.is_some() != continuing
			|| ppo.horizon != s.ppo.horizon
			|| ppo.environments != s.ppo.environments
			|| ppo.observation_shape != s.ppo.observation_shape
			|| ppo.gae != s.ppo.gae
		{
			return Err(Error::failed_precondition(
				"compiled collection owner/recipe changed",
			));
		}
		ppo
			.seed
			.checked_add(ppo.horizon as u64)
			.ok_or_else(|| Error::resource_exhausted("collection seed exhausted"))?;
		let action_seed = ppo
			.seed
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("collection seed exhausted"))?;
		s.cursor.write_values(&[
			0_u32,
			action_seed as u32,
			(action_seed >> 32) as u32,
			seed as u32,
			(seed >> 32) as u32,
			u32::from(!continuing && ppo.horizon == 1),
			0,
		])?;
		let reset = if reset_environment {
			&s.reset
		} else {
			s.resume.as_ref().expect("continuation reset plan")
		};
		let mut elapsed = submit(engine, reset, ppo.enable_gpu_timing)?;
		env.replay_observation(s.observation.clone());
		let mut completed = 0_usize;
		while completed < ppo.horizon {
			let until_check = if check_every == 0 {
				16
			} else {
				check_every as usize - completed % (check_every as usize)
			};
			let count = (ppo.horizon - completed).min(16).min(until_check);
			let mut events = Vec::with_capacity(count);
			for _ in 0..count {
				events.push(if ppo.enable_gpu_timing {
					engine.submit_timed(&s.step)?
				} else {
					engine.submit(&s.step)?
				});
			}
			// Engine timeline dependencies preserve visibility and exact order.
			events.last().expect("nonempty replay batch").wait()?;
			if ppo.enable_gpu_timing {
				for event in events {
					elapsed = elapsed
						.checked_add(event.device_duration()?)
						.ok_or_else(|| Error::resource_exhausted("collection GPU time overflow"))?;
				}
			}
			completed += count;
			s.replays = s
				.replays
				.checked_add(count as u64)
				.ok_or_else(|| Error::resource_exhausted("collection replay exhausted"))?;
			if check_every != 0 && completed.is_multiple_of(check_every as usize) {
				env.check_status()?;
			}
		}
		elapsed = elapsed
			.checked_add(submit(engine, &s.gae, ppo.enable_gpu_timing)?)
			.ok_or_else(|| Error::resource_exhausted("collection GPU time overflow"))?;
		env.check_status()?;
		if s.cursor.read::<u32>()?[6] != ppo.horizon as u32 {
			return Err(Error::failed_precondition(
				"compiled rollout append cursor did not advance",
			));
		}
		rollout.finish_replayed_collection();
		Ok(elapsed)
	}
}

impl Default for CompiledRollout {
	fn default() -> Self {
		Self::new()
	}
}

impl State {
	fn capture<E: CompiledEnvironment>(
		engine: &Engine,
		model: &dyn ActorCritic,
		env: &mut E,
		rollout: &mut RolloutBuffer,
		ppo: &PpoTrainerConfig,
		shaping: f32,
		continuing: bool,
	) -> Result<Self> {
		rollout.initialize_replay_storage()?;
		let cursor = Matrix::new_from_slice(engine, [7], &[0_u32, 1, 0, 1, 0, 0, 0])?;
		let copy_cursor = Matrix::new_from_slice(engine, [2], &[0_u32, 1])?;
		let observation = env.observation().clone();
		let observation_elements = ppo
			.observation_shape
			.iter()
			.try_fold(1_usize, |count, extent| count.checked_mul(*extent))
			.ok_or_else(|| Error::resource_exhausted("rollout observation size overflows usize"))?;
		let observation_count = ppo
			.environments
			.checked_mul(observation_elements)
			.ok_or_else(|| Error::resource_exhausted("rollout observation count overflows usize"))?;
		// Successor observations are retained pre-reset because done lanes lose
		// the terminal/truncation state to the in-place environment reset before
		// the next step's carried observation is produced.
		let next_observation_rows = ppo
			.horizon
			.checked_mul(ppo.environments)
			.ok_or_else(|| Error::resource_exhausted("rollout rows overflow usize"))?;
		let next_observations = Matrix::allocate(
			&engine.handle(),
			vec![next_observation_rows, observation_elements],
			next_observation_rows
				.checked_mul(observation_elements)
				.ok_or_else(|| {
					Error::resource_exhausted("rollout next_observation storage overflows usize")
				})?,
			DType::F32,
		)?;
		let initialized = env
			.replay_slots()
			.into_iter()
			.chain(rollout.replay_slots())
			.chain(std::iter::once(&cursor))
			.chain(std::iter::once(&next_observations))
			.cloned()
			.collect::<Vec<_>>();
		let state = initialized.iter().collect::<Vec<_>>();
		let (reset, ()) = engine.capture_with_initialized_state(&state, || {
			rollout.reset_replay_commands()?;
			env.record_reset_replay(false, &cursor)
		})?;
		env.replay_observation(observation.clone());
		let resume = if continuing {
			Some(
				engine
					.capture_with_initialized_state(&state, || rollout.reset_replay_commands())?
					.0,
			)
		} else {
			None
		};
		let (step, ()) = engine.capture_with_initialized_state(&state, || {
			let network = model.evaluate(&observation)?;
			let action = crate::ml::matrix::sample_categorical_cursor(&network.logits, &cursor)?;
			let policy = policy::evaluate_categorical(&network.logits, &action, &network.value)?;
			let step = env.record_step_replay(&action, &cursor)?;
			let reward = if shaping != 0. {
				matrix::add_scalar(step.reward(), shaping)?
			} else {
				step.reward().clone()
			};
			// The appended next_value is a placeholder completed once by the GAE
			// plan's single batched evaluation over the retained successor
			// observations instead of a second evaluation per step.
			rollout.append_replay(
				&RolloutTransition::new(
					step.observation().clone(),
					action,
					reward,
					policy.value.clone(),
					policy.value,
					policy.log_probability,
					step.terminated().clone(),
					step.truncated().clone(),
				),
				&cursor,
			)?;
			crate::ml::rollout::next_observation_append(
				step.next_observation(),
				&next_observations,
				&cursor,
				0,
				observation_count,
			)?;
			env.record_reset_replay(true, &cursor)?;
			crate::ml::matrix::training_accumulate(&observation, env.observation(), &copy_cursor, 1.)?;
			advance_cursor(engine, &cursor, ppo.horizon as u32, continuing)
		})?;
		env.replay_observation(observation.clone());
		let (gae, ()) = engine.capture_with_initialized_state(&state, || {
			// One batched evaluation over all retained successor observations
			// replaces the removed second per-step evaluation: every slot gets
			// V(next_obs_t) exactly, including pre-reset terminal and truncation
			// states the carried observation no longer holds.
			let network = model.evaluate(&next_observations)?;
			crate::ml::matrix::training_accumulate(
				rollout.batch().next_value(),
				&network
					.value
					.reshape_view(vec![ppo.horizon, ppo.environments])?,
				&copy_cursor,
				1.,
			)?;
			rollout.finalize_replay_commands(ppo.gae)
		})?;
		Ok(Self {
			env: env.replay_identity(),
			rollout: rollout
				.replay_slots()
				.iter()
				.map(|m| m.value_id())
				.collect(),
			parameters: model
				.all_parameters()?
				.iter()
				.map(|p| p.data().value_id())
				.collect(),
			ppo: ppo.clone(),
			shaping,
			observation,
			cursor,
			reset,
			resume,
			step,
			gae,
			replays: 0,
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

fn advance_cursor(engine: &Engine, cursor: &Matrix, horizon: u32, continuing: bool) -> Result<()> {
	// The seven-word rollout cursor and its advance kernel are shared
	// machinery despite the trading donor name.
	engine.handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlTradingCursorAdvanceU32,
			buffers: &[BufferBinding::read_write(cursor.storage())],
			push_constants: &[
				PushConstant::U32(horizon),
				PushConstant::U32(u32::from(continuing)),
			],
			workgroups: [1, 1, 1],
			indirect: None,
		},
		SemanticDispatch {
			contract: crate::core::operation::ml::TRADING_CURSOR_ADVANCE,
			inputs: &[cursor],
			outputs: &[cursor],
			attributes: &[
				OpAttribute::UnsignedInteger {
					name: "horizon".into(),
					value: u64::from(horizon),
				},
				OpAttribute::Boolean {
					name: "continuing".into(),
					value: continuing,
				},
			],
		},
	)
}
