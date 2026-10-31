//! Compiled trading collection over OA's canonical environment and rollout.
//! No exchange authority; exact existing Philox, terminal costs and GAE semantics.
use super::PpoTrainerConfig;
use crate::ml::environment::{Environment, TradingConfig, TradingEnvironment};
use crate::ml::{ActorCritic, RolloutBuffer, RolloutTransition, policy};
use crate::runtime::{BufferBinding, ComputeDispatch, KernelId, PushConstant, SemanticDispatch};
use crate::{Engine, Error, ExecutionPlan, Matrix, OpAttribute, Result, matrix};
use std::time::Duration;

/// Fixed-geometry trading reset, policy/step/append and GAE replay programs.
/// Retain alongside the same environment and `PpoUpdateProgram` across rollouts.
/// Seven cursor words are uploaded once per chunk and advance on device; tables refresh at
/// explicit completed rollout boundaries. Drop never executes or waits.
#[derive(Default)]
pub struct TradingRolloutProgram {
	state: Option<State>,
}
pub(super) struct CollectionOptions {
	pub seed: u64,
	pub check_every: u32,
	pub shaping: f32,
	pub continuing: bool,
}
struct State {
	env: Vec<u64>,
	rollout: Vec<u64>,
	parameters: Vec<u64>,
	config: TradingConfig,
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
impl TradingRolloutProgram {
	/// Complete vector steps executed by the cached collection program.
	pub fn replay_count(&self) -> u64 {
		self.state.as_ref().map_or(0, |s| s.replays)
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
	pub(super) fn collect(
		&mut self,
		engine: &Engine,
		model: &dyn ActorCritic,
		env: &mut TradingEnvironment<'_>,
		rollout: &mut RolloutBuffer,
		ppo: &PpoTrainerConfig,
		options: CollectionOptions,
	) -> Result<Duration> {
		let CollectionOptions {
			seed,
			check_every,
			shaping,
			continuing,
		} = options;
		if !env.replay_tables_valid()
			|| !shaping.is_finite()
			|| !engine.owns_matrix(env.observation())
			|| env.environments() as usize != ppo.environments
		{
			return Err(Error::invalid_argument(
				"compiled trading collection geometry/Engine/shaping mismatch",
			));
		}
		engine.checkpoint()?.wait()?;
		let reset_environment = !continuing || !env.replay_initialized();
		if self.state.is_none() {
			self.state = Some(State::capture(
				engine, model, env, rollout, ppo, shaping, continuing,
			)?);
		}
		let s = self.state.as_mut().expect("compiled trading collection");
		let mut config = env.replay_config();
		config.seed = 0;
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
			|| config != s.config
			|| s.shaping != shaping
			|| s.resume.is_some() != continuing
			|| ppo.horizon != s.ppo.horizon
			|| ppo.environments != s.ppo.environments
			|| ppo.observation_shape != s.ppo.observation_shape
			|| ppo.gae != s.ppo.gae
		{
			return Err(Error::failed_precondition(
				"compiled trading collection owner/recipe changed",
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
impl State {
	fn capture(
		engine: &Engine,
		model: &dyn ActorCritic,
		env: &mut TradingEnvironment<'_>,
		rollout: &mut RolloutBuffer,
		ppo: &PpoTrainerConfig,
		shaping: f32,
		continuing: bool,
	) -> Result<Self> {
		rollout.initialize_replay_storage()?;
		let cursor = Matrix::new_from_slice(engine, [7], &[0_u32, 1, 0, 1, 0, 0, 0])?;
		let copy_cursor = Matrix::new_from_slice(engine, [2], &[0_u32, 1])?;
		let observation = env.observation().clone();
		let initialized = env
			.replay_slots()
			.into_iter()
			.chain(rollout.replay_slots())
			.chain(std::iter::once(&cursor))
			.cloned()
			.collect::<Vec<_>>();
		let state = initialized.iter().collect::<Vec<_>>();
		let (reset, ()) = engine.capture_with_initialized_state(&state, || {
			rollout.reset_replay_commands()?;
			env.reset_replay_commands(false, &cursor)
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
			let step = env.step_replay_commands(&action, &cursor)?;
			let next = model.evaluate(step.next_observation())?;
			let reward = if shaping != 0. {
				matrix::add_scalar(step.reward(), shaping)?
			} else {
				step.reward().clone()
			};
			rollout.append_replay(
				&RolloutTransition::new(
					step.observation().clone(),
					action,
					reward,
					policy.value,
					next.value,
					policy.log_probability,
					step.terminated().clone(),
					step.truncated().clone(),
				),
				&cursor,
			)?;
			env.reset_replay_commands(true, &cursor)?;
			crate::ml::matrix::training_accumulate(&observation, env.observation(), &copy_cursor, 1.)?;
			advance_cursor(engine, &cursor, ppo.horizon as u32, continuing)
		})?;
		env.replay_observation(observation.clone());
		let (gae, ()) = engine
			.capture_with_initialized_state(&state, || rollout.finalize_replay_commands(ppo.gae))?;
		let mut config = env.replay_config();
		config.seed = 0;
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
			config,
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
	engine.handle().record_semantic(
		ComputeDispatch {
			kernel: KernelId::MlTradingCursorAdvanceU32,
			buffers: &[BufferBinding::read_write(cursor.storage())],
			push_constants: &[
				PushConstant::U32(horizon),
				PushConstant::U32(u32::from(continuing)),
			],
			workgroups: [1, 1, 1],
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
