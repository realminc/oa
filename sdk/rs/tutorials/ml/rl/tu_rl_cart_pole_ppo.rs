use std::time::Instant;

use oa::ml::{
	AdamW, CategoricalActorCritic, CategoricalActorCriticConfig, Module,
	environment::Environment,
	evaluation::{PolicyEvaluationConfig, PolicyEvaluationMetrics, evaluate_categorical},
	training::{PpoTrainer, PpoTrainerConfig, RolloutTrainingPhase},
};
use oa::sdk::ml::rl::{CartPole, CartPoleConfig};

const ENVIRONMENTS: u32 = 64;
const HORIZON: usize = 128;
const ROLLOUTS: u64 = 40;
const UPDATE_EPOCHS: u64 = 4;
const TRAINING_SEED: u64 = 0x0a11_ce55;
const EVALUATION_SEED: u64 = 0x0e7a1;
const LEARNING_RATE: f32 = 2.5e-4;
const EVALUATION_HORIZON: usize = 500;

fn evaluate(
	engine: &oa::Engine,
	model: &CategoricalActorCritic,
	seed: u64,
) -> oa::Result<PolicyEvaluationMetrics> {
	let mut environment = CartPole::new(
		engine,
		CartPoleConfig {
			environments: ENVIRONMENTS,
			max_episode_steps: 500,
			seed,
			..CartPoleConfig::default()
		},
	)?;
	let initialized = environment.submit()?;
	environment.wait(&initialized)?;
	let result = evaluate_categorical(
		&mut environment,
		model,
		PolicyEvaluationConfig {
			horizon: EVALUATION_HORIZON,
			seed,
		},
	)?;
	environment.close()?;
	Ok(result)
}

fn collect(environment: &mut CartPole<'_>, trainer: &mut PpoTrainer<'_, '_>) -> oa::Result<()> {
	if environment.execution().submission_count() == 0
		&& !environment.execution().has_active_recording()
	{
		environment.reset(TRAINING_SEED)?;
	}
	environment.begin()?;
	trainer.begin_collection()?;
	let recorded = (|| {
		for _ in 0..HORIZON {
			let policy = trainer.act(environment.observation())?;
			let transition = environment.step(&policy.action)?;
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

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let model = CategoricalActorCritic::new(
		&engine,
		CategoricalActorCriticConfig {
			observation_size: 4,
			action_count: 2,
			hidden_size: 64,
			seed: TRAINING_SEED,
		},
	)?;
	let mut optimizer = AdamW::with_hyperparameters(
		model.all_parameters()?,
		LEARNING_RATE,
		0.9,
		0.999,
		1.0e-8,
		0.0,
	)?;

	println!("\nOA reinforcement learning — CartPole PPO");
	println!("  separate actor/critic: each 4 -> 64 -> 64");
	println!("  heads: categorical policy 2 · scalar value 1");
	println!(
		"  rollout: {ENVIRONMENTS} env x {HORIZON} steps · {UPDATE_EPOCHS} PPO epochs · {ROLLOUTS} rollouts"
	);

	let before = evaluate(&engine, &model, EVALUATION_SEED)?;
	println!(
		"  before: {:.2} mean completed return ({} episodes)",
		before.mean_completed_return, before.completed_episodes
	);

	let mut environment = CartPole::new(
		&engine,
		CartPoleConfig {
			environments: ENVIRONMENTS,
			max_episode_steps: 500,
			seed: TRAINING_SEED,
			..CartPoleConfig::default()
		},
	)?;
	let started = Instant::now();
	let snapshot = {
		let mut trainer = PpoTrainer::new(
			&engine,
			&model,
			&mut optimizer,
			PpoTrainerConfig {
				rollouts: ROLLOUTS,
				horizon: HORIZON,
				environments: ENVIRONMENTS as usize,
				update_epochs: UPDATE_EPOCHS,
				observation_shape: vec![4],
				seed: TRAINING_SEED,
				enable_gpu_timing: true,
				..PpoTrainerConfig::default()
			},
		)?;
		let mut last_printed = 0;
		while !trainer.is_done() {
			if trainer.needs_collection() {
				collect(&mut environment, &mut trainer)?;
			}
			trainer.update()?;
			let metrics = trainer.metrics();
			if trainer.phase() != RolloutTrainingPhase::Update
				&& metrics.rollout != last_printed
				&& metrics.rollout % 10 == 0
			{
				last_printed = metrics.rollout;
				println!(
					"  rollout {}/{} · loss {:.5}",
					metrics.rollout, ROLLOUTS, metrics.total_loss
				);
			}
		}
		trainer.finish()?
	};
	environment.close()?;
	let wall = started.elapsed();

	let after = evaluate(&engine, &model, EVALUATION_SEED)?;
	let checkpoint = std::env::temp_dir().join("oars_cart_pole_ppo.oam");
	oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	let restored_model = CategoricalActorCritic::new(
		&engine,
		CategoricalActorCriticConfig {
			observation_size: 4,
			action_count: 2,
			hidden_size: 64,
			seed: TRAINING_SEED,
		},
	)?;
	let mut restored_optimizer = AdamW::with_hyperparameters(
		restored_model.all_parameters()?,
		LEARNING_RATE,
		0.9,
		0.999,
		1.0e-8,
		0.0,
	)?;
	oa::ml::load_checkpoint(
		&engine,
		&checkpoint,
		&restored_model,
		&mut restored_optimizer,
	)?;
	let restored = evaluate(&engine, &restored_model, EVALUATION_SEED)?;
	std::fs::remove_file(&checkpoint).expect("remove completed CartPole checkpoint");

	println!(
		"  after:  {:.2} mean completed return ({} episodes)",
		after.mean_completed_return, after.completed_episodes
	);
	println!(
		"  improvement: {:+.2}",
		after.mean_completed_return - before.mean_completed_return
	);
	println!(
		"  checkpoint: {:.2} restored return · AdamW step {}",
		restored.mean_completed_return,
		restored_optimizer.step_count()
	);
	println!(
		"  training: {:.2}s wall · {:.2} ms/update wall · {:.2} ms/update GPU · {} transitions",
		wall.as_secs_f64(),
		snapshot.wall_ms_per_step(),
		snapshot.gpu_timing_stats().mean_ms,
		u64::from(ENVIRONMENTS) * HORIZON as u64 * ROLLOUTS
	);

	assert!(after.mean_completed_return >= before.mean_completed_return + 25.0);
	assert!(after.mean_completed_return >= 75.0);
	assert_eq!(restored.mean_completed_return, after.mean_completed_return);
	assert_eq!(restored.completed_episodes, after.completed_episodes);
	assert_eq!(restored_optimizer.step_count(), optimizer.step_count());
	Ok(())
}
