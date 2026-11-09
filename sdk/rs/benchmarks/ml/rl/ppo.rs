//! PPO training benchmark for the opt-in JIT specialization route.
//!
//! Runs the same seeded CartPole PPO workload as `tu_rl_cart_pole_ppo`
//! and reports per-rollout losses plus wall and device update timings.
//! Compare a baseline run against `OA_JIT_FUSION=1` on identical seeds:
//! every rollout's reported losses must match within tolerance, and only
//! then does the warmed throughput difference mean anything.
//!
//! Workload knobs: `OA_PPO_BENCH_ROLLOUTS`, `OA_PPO_BENCH_ENVS`,
//! `OA_PPO_BENCH_HORIZON`, `OA_PPO_BENCH_HIDDEN`, `OA_PPO_BENCH_EPOCHS`.
//!
//! Phase lines split host wall time. Collection records policy, environment
//! and transition work eagerly, then `execute` submits and waits for the
//! recorded collection including GAE. `update` covers every epoch's recording
//! and its synchronized completion. Device update time stays in
//! `OA_PPO_RESULT`; the fusion census line reports JIT compile and pipeline
//! time separately.

use std::time::{Duration, Instant};

use oa::ml::{
	AdamW, CategoricalActorCritic, CategoricalActorCriticConfig, Module,
	environment::Environment,
	training::{PpoTrainer, PpoTrainerConfig, RolloutTrainingPhase},
};
use oa::sdk::ml::rl::{CartPole, CartPoleConfig};

const TRAINING_SEED: u64 = 0x0a11_ce55;
const LEARNING_RATE: f32 = 2.5e-4;

fn env_usize(name: &str, default: usize) -> usize {
	std::env::var(name)
		.ok()
		.and_then(|value| value.parse().ok())
		.unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
	std::env::var(name)
		.ok()
		.and_then(|value| value.parse().ok())
		.unwrap_or(default)
}

/// Host wall time of one collection, by phase.
#[derive(Default)]
struct CollectionTiming {
	act: Duration,
	step: Duration,
	observe: Duration,
	finalize: Duration,
	execute: Duration,
}

fn milliseconds(duration: Duration) -> f64 {
	duration.as_secs_f64() * 1000.0
}

fn collect(
	environment: &mut CartPole<'_>,
	trainer: &mut PpoTrainer<'_, '_>,
) -> oa::Result<CollectionTiming> {
	let mut timing = CollectionTiming::default();
	if environment.execution().submission_count() == 0
		&& !environment.execution().has_active_recording()
	{
		environment.reset(TRAINING_SEED)?;
	}
	environment.begin()?;
	trainer.begin_collection()?;
	let recorded = (|| {
		for _ in 0..trainer.config().horizon {
			let started = Instant::now();
			let policy = trainer.act(environment.observation())?;
			timing.act += started.elapsed();
			let started = Instant::now();
			let transition = environment.step(&policy.action)?;
			timing.step += started.elapsed();
			let started = Instant::now();
			trainer.observe(
				transition.observation(),
				transition.next_observation(),
				transition.reward(),
				transition.terminated(),
				transition.truncated(),
				&policy,
			)?;
			timing.observe += started.elapsed();
			let started = Instant::now();
			environment.reset_completed()?;
			timing.step += started.elapsed();
		}
		let started = Instant::now();
		let finalized = trainer.end_collection();
		timing.finalize += started.elapsed();
		finalized
	})();
	if let Err(error) = recorded {
		let _ = environment.cancel();
		let _ = trainer.abort_collection();
		return Err(error);
	}
	let started = Instant::now();
	let event = match environment.submit() {
		Ok(event) => event,
		Err(error) => {
			let _ = trainer.abort_collection();
			return Err(error);
		}
	};
	environment.wait(&event)?;
	timing.execute = started.elapsed();
	Ok(timing)
}

fn main() -> oa::Result<()> {
	let rollouts = env_u64("OA_PPO_BENCH_ROLLOUTS", 12);
	let environments = env_usize("OA_PPO_BENCH_ENVS", 64) as u32;
	let horizon = env_usize("OA_PPO_BENCH_HORIZON", 128);
	let epochs = env_u64("OA_PPO_BENCH_EPOCHS", 4);
	let hidden = env_usize("OA_PPO_BENCH_HIDDEN", 64);
	let jit = oa::EnvFlag::is_set("OA_JIT_FUSION");

	let process_started = Instant::now();
	let engine = oa::Engine::new()?;
	let model = CategoricalActorCritic::new(
		&engine,
		CategoricalActorCriticConfig {
			observation_size: 4,
			action_count: 2,
			hidden_size: hidden,
			seed: TRAINING_SEED,
		},
	)?;
	let mut optimizer = AdamW::new_with_hyperparameters(
		model.all_parameters()?,
		LEARNING_RATE,
		0.9,
		0.999,
		1.0e-8,
		0.0,
	)?;
	let mut environment = CartPole::new(
		&engine,
		CartPoleConfig {
			environments,
			max_episode_steps: 500,
			seed: TRAINING_SEED,
			..CartPoleConfig::default()
		},
	)?;

	println!(
		"OA_PPO_BENCH workload=cartpole_ppo route={} rollouts={rollouts} envs={environments} horizon={horizon} epochs={epochs} hidden={hidden} seed={TRAINING_SEED:#x}",
		if jit { "jit" } else { "baseline" }
	);

	let setup = process_started.elapsed();
	let started = Instant::now();
	let mut collection_total = CollectionTiming::default();
	let mut update_total = Duration::ZERO;
	let snapshot = {
		let mut trainer = PpoTrainer::new(
			&engine,
			&model,
			&mut optimizer,
			PpoTrainerConfig {
				rollouts,
				horizon,
				environments: environments as usize,
				update_epochs: epochs,
				observation_shape: vec![4],
				seed: TRAINING_SEED,
				enable_gpu_timing: true,
				..PpoTrainerConfig::default()
			},
		)?;
		let mut collection = CollectionTiming::default();
		let mut rollout_update = Duration::ZERO;
		while !trainer.is_done() {
			if trainer.needs_collection() {
				collection = collect(&mut environment, &mut trainer)?;
				rollout_update = Duration::ZERO;
			}
			let update_started = Instant::now();
			trainer.update()?;
			rollout_update += update_started.elapsed();
			let metrics = trainer.metrics();
			if trainer.phase() != RolloutTrainingPhase::Update {
				println!(
					"OA_PPO_PHASE index={} act_ms={:.3} step_ms={:.3} observe_ms={:.3} finalize_ms={:.3} execute_ms={:.3} update_ms={:.3}",
					metrics.rollout,
					milliseconds(collection.act),
					milliseconds(collection.step),
					milliseconds(collection.observe),
					milliseconds(collection.finalize),
					milliseconds(collection.execute),
					milliseconds(rollout_update)
				);
				collection_total.act += collection.act;
				collection_total.step += collection.step;
				collection_total.observe += collection.observe;
				collection_total.finalize += collection.finalize;
				collection_total.execute += collection.execute;
				update_total += rollout_update;
				println!(
					"OA_PPO_ROLLOUT index={} wall_ms={:.3} total_loss={:.9e} policy_loss={:.9e} value_loss={:.9e} entropy={:.9e}",
					metrics.rollout,
					update_started.elapsed().as_secs_f64() * 1000.0,
					metrics.total_loss,
					metrics.policy_loss,
					metrics.value_loss,
					metrics.entropy
				);
			}
		}
		trainer.finish()?
	};
	environment.close()?;
	let wall = started.elapsed();
	let gpu = snapshot.gpu_timing_stats();

	println!(
		"OA_PPO_RESULT route={} steps={} wall_s={:.3} wall_ms_per_step={:.3} gpu_mean_ms={:.3} gpu_median_ms={:.3} gpu_p95_ms={:.3} gpu_min_ms={:.3} transitions={}",
		if jit { "jit" } else { "baseline" },
		snapshot.step_count(),
		wall.as_secs_f64(),
		snapshot.wall_ms_per_step(),
		gpu.mean_ms,
		gpu.median_ms,
		gpu.p95_ms,
		gpu.min_ms,
		u64::from(environments) * horizon as u64 * rollouts
	);
	println!(
		"OA_PPO_PHASES route={} setup_ms={:.3} act_ms={:.3} step_ms={:.3} observe_ms={:.3} finalize_ms={:.3} execute_ms={:.3} update_ms={:.3} other_ms={:.3}",
		if jit { "jit" } else { "baseline" },
		milliseconds(setup),
		milliseconds(collection_total.act),
		milliseconds(collection_total.step),
		milliseconds(collection_total.observe),
		milliseconds(collection_total.finalize),
		milliseconds(collection_total.execute),
		milliseconds(update_total),
		milliseconds(wall.saturating_sub(
			collection_total.act
				+ collection_total.step
				+ collection_total.observe
				+ collection_total.finalize
				+ collection_total.execute
				+ update_total
		))
	);
	println!(
		"OA_PPO_FUSION route={} {}",
		if jit { "jit" } else { "baseline" },
		engine.fusion_stats()
	);
	Ok(())
}
