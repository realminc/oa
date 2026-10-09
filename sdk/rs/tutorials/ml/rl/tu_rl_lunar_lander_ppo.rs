use std::time::Instant;

use oa::ml::{
	AdamW, CategoricalActorCritic, CategoricalActorCriticConfig, Module,
	environment::Environment,
	training::{PpoTrainer, PpoTrainerConfig, RolloutTrainingPhase},
};
use oa::sdk::ml::rl::{
	LUNAR_OBSERVATION_SIZE, LunarFirstEpisodeEvaluation, LunarFirstEpisodeEvaluationConfig,
	LunarLander3dConfig, LunarLander3dVector, LunarLander3dVectorConfig, LunarTeacherConfig,
	collect_ppo_rollout, evaluate_first_episodes, pretrain_scripted_teacher,
};

const ENVIRONMENTS: u32 = 64;
const HORIZON: usize = 128;
const ROLLOUTS: u64 = 40;
const UPDATE_EPOCHS: u64 = 4;
const HIDDEN_SIZE: usize = 64;
const TRAINING_SEED: u64 = 0x1a2b_3c4d;
const LEARNING_RATE: f32 = 2.5e-4;

fn print_evaluation(label: &str, value: &LunarFirstEpisodeEvaluation) {
	println!(
		"  {label}: return {:.3} [{:.3}, {:.3}] · safe {}/{} ({:.2}%, Wilson95 {:.2}%)",
		value.mean_return,
		value.minimum_return,
		value.maximum_return,
		value.safe_landings,
		value.expected_episodes,
		value.safe_landing_rate * 100.0,
		value.wilson_lower_95 * 100.0,
	);
	println!(
		"           body {} · hard-foot {} · bounds {} · numerical {} · timeout {} · incomplete {}",
		value.body_impacts,
		value.hard_foot_impacts,
		value.out_of_bounds,
		value.numerical_failures,
		value.time_limits,
		value.incomplete_episodes,
	);
}

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let model = CategoricalActorCritic::new(
		&engine,
		CategoricalActorCriticConfig {
			observation_size: LUNAR_OBSERVATION_SIZE,
			action_count: 8,
			hidden_size: HIDDEN_SIZE,
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
	let evaluation_config = LunarFirstEpisodeEvaluationConfig::default();

	println!("\nOA reinforcement learning — Lunar Lander 3D flat-v0");
	println!(
		"  separate actor/critic: each {} -> {} -> {} · categorical actions 8",
		LUNAR_OBSERVATION_SIZE, HIDDEN_SIZE, HIDDEN_SIZE
	);

	let before = evaluate_first_episodes(&engine, &model, evaluation_config)?;
	print_evaluation("untrained", &before);

	if std::env::var_os("OA_LUNAR_RUN_TEACHER_GATE").is_some() {
		let started = Instant::now();
		let teacher = pretrain_scripted_teacher(&engine, &model, LunarTeacherConfig::default())?;
		let after = evaluate_first_episodes(&engine, &model, evaluation_config)?;
		println!(
			"  teacher: {} samples · {} updates · loss {:.6} -> {:.6} · {:.2}s wall",
			teacher.samples,
			teacher.optimizer_steps,
			teacher.initial_loss,
			teacher.final_loss,
			started.elapsed().as_secs_f64(),
		);
		print_evaluation("learned", &after);
		assert!(teacher.final_loss < teacher.initial_loss);
		assert_eq!(optimizer.step_count(), 0);
		assert_eq!(after.completed_episodes, after.expected_episodes);
		assert!(after.safe_landing_rate >= 0.80);
		assert!(after.wilson_lower_95 >= 0.75);
		return Ok(());
	}

	println!(
		"  rollout: {} env x {} steps · {} PPO epochs · {} rollouts",
		ENVIRONMENTS, HORIZON, UPDATE_EPOCHS, ROLLOUTS
	);
	let mut environment = LunarLander3dVector::flat(
		&engine,
		LunarLander3dVectorConfig {
			environments: ENVIRONMENTS,
			seed: TRAINING_SEED,
			environment: LunarLander3dConfig::default(),
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
				observation_shape: vec![LUNAR_OBSERVATION_SIZE],
				seed: TRAINING_SEED,
				enable_gpu_timing: true,
				..PpoTrainerConfig::default()
			},
		)?;
		let mut last_printed = 0;
		while !trainer.is_done() {
			if trainer.needs_collection() {
				collect_ppo_rollout(&mut environment, &mut trainer)?;
			}
			trainer.update()?;
			let metrics = trainer.metrics();
			if trainer.phase() != RolloutTrainingPhase::Update
				&& metrics.rollout != last_printed
				&& (metrics.rollout % 10 == 0 || trainer.is_done())
			{
				last_printed = metrics.rollout;
				println!(
					"  rollout {}/{} · loss {:.6} policy {:.6} value {:.6} entropy {:.6}",
					metrics.rollout,
					ROLLOUTS,
					metrics.total_loss,
					metrics.policy_loss,
					metrics.value_loss,
					metrics.entropy,
				);
			}
		}
		trainer.finish()?
	};
	environment.close()?;
	let wall = started.elapsed();
	let after = evaluate_first_episodes(&engine, &model, evaluation_config)?;
	print_evaluation("trained", &after);

	let checkpoint = std::env::temp_dir().join("oars_lunar_lander_3d_flat_ppo.oam");
	oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	let restored_model = CategoricalActorCritic::new(
		&engine,
		CategoricalActorCriticConfig {
			observation_size: LUNAR_OBSERVATION_SIZE,
			action_count: 8,
			hidden_size: HIDDEN_SIZE,
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
	let restored = evaluate_first_episodes(&engine, &restored_model, evaluation_config)?;
	std::fs::remove_file(&checkpoint).expect("remove completed Lunar checkpoint");

	println!(
		"  training: {:.2}s wall · {:.2} ms/update wall · {:.2} ms/update GPU · {} transitions",
		wall.as_secs_f64(),
		snapshot.wall_ms_per_step(),
		snapshot.gpu_timing_stats().mean_ms,
		u64::from(ENVIRONMENTS) * HORIZON as u64 * ROLLOUTS,
	);
	println!(
		"  checkpoint: action {:016x} · value {:016x} · AdamW step {}",
		restored.action_trace_digest,
		restored.value_trace_digest,
		restored_optimizer.step_count(),
	);
	assert_eq!(after.completed_episodes, after.expected_episodes);
	assert_eq!(restored, after);
	assert_eq!(restored_optimizer.step_count(), optimizer.step_count());
	if std::env::var_os("OA_LUNAR_REQUIRE_FLAT_GATE").is_some() {
		assert!(after.mean_return > before.mean_return);
		assert!(after.safe_landing_rate >= 0.80);
		assert!(after.wilson_lower_95 >= 0.75);
	}
	Ok(())
}
