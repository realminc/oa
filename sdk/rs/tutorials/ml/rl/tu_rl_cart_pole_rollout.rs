use oa::ml::{
	RolloutBuffer, RolloutConfig, RolloutTransition, advantage, environment::Environment, policy,
};
use oa::sdk::ml::rl::{CartPole, CartPoleConfig};

const ENVIRONMENTS: usize = 32;
const HORIZON: usize = 64;
const SEED: u64 = 20_260_716;

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let mut environment = CartPole::new(
		&engine,
		CartPoleConfig {
			environments: ENVIRONMENTS as u32,
			max_episode_steps: 500,
			seed: SEED,
			..CartPoleConfig::default()
		},
	)?;
	let mut rollout = RolloutBuffer::new(
		&engine,
		RolloutConfig {
			time: HORIZON,
			environments: ENVIRONMENTS,
			observation_shape: vec![4],
		},
	)?;
	let policy_weight = oa::Matrix::from_f32(
		&engine,
		[2, 4],
		&[0.0, -0.1, -4.0, -1.0, 0.0, 0.1, 4.0, 1.0],
	)?;
	let value = oa::Matrix::from_f32(&engine, [ENVIRONMENTS], &[0.0; ENVIRONMENTS])?;

	println!("\nOA reinforcement learning — vectorized GPU CartPole rollout");
	println!(
		"  environments: {ENVIRONMENTS} · horizon: {HORIZON} · transitions: {}",
		ENVIRONMENTS * HORIZON
	);

	rollout.reset()?;
	for step in 0..HORIZON {
		let logits = oa::matrix::mat_mul_nt(environment.observation(), &policy_weight)?;
		let selected = policy::sample_categorical(&logits, &value, SEED + step as u64 + 1)?;
		let transition = environment.step(&selected.action)?;
		rollout.append(&RolloutTransition::new(
			transition.observation().clone(),
			selected.action,
			transition.reward().clone(),
			selected.value,
			value.clone(),
			selected.log_probability,
			transition.terminated().clone(),
			transition.truncated().clone(),
		))?;
		environment.reset_completed()?;
	}
	rollout.finalize(advantage::GaeConfig::default())?;
	let completion = environment.submit()?;
	environment.wait(&completion)?;

	let reward = rollout.batch().reward().read_f32()?;
	let terminated = rollout.batch().terminated().read::<u8>()?;
	let truncated = rollout.batch().truncated().read::<u8>()?;
	let valid = rollout.batch().valid().read::<u8>()?;
	let advantage = rollout.batch().advantage().read_f32()?;
	let reward_sum = reward.iter().copied().map(f64::from).sum::<f64>();
	let episodes = terminated
		.iter()
		.zip(&truncated)
		.filter(|(terminated, truncated)| **terminated != 0 || **truncated != 0)
		.count();
	assert!(valid.iter().all(|value| *value == 1));
	assert!(advantage.iter().all(|value| value.is_finite()));
	assert_eq!(rollout.len(), HORIZON);
	assert!(reward_sum > 0.0);

	println!(
		"  result: {reward_sum:.0} reward · {episodes} completed episodes · {:.2} reward/env",
		reward_sum / ENVIRONMENTS as f64
	);
	println!("  path: policy -> sample -> step -> append -> reset-done -> GAE");
	println!("  host tensor reads during collection: 0");
	environment.close()?;
	Ok(())
}
