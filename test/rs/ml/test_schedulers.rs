use oa::ml::{
	CosineScheduler, CosineWarmRestartsScheduler, CyclicMode, CyclicScheduler,
	LinearWarmupCosineScheduler, LrScheduler, OneCycleScheduler, PlateauMode,
	ReduceOnPlateauScheduler, SequentialScheduler, WarmupScheduler,
};

struct Constant(f32);

impl LrScheduler for Constant {
	fn learning_rate(&self, _step: u64) -> f32 {
		self.0
	}
}

fn assert_close(actual: f32, expected: f32) {
	assert!(
		(actual - expected).abs() <= 1.0e-6,
		"expected {expected}, found {actual}"
	);
}

#[test]
fn donor_learning_rate_schedules_preserve_boundaries_and_step_origins() -> oa::Result<()> {
	let cosine = CosineScheduler::new(1.0, 0.0, 4)?;
	assert_close(cosine.learning_rate(0), 1.0);
	assert_close(cosine.learning_rate(2), 0.5);
	assert_close(cosine.learning_rate(4), 0.0);
	assert_close(cosine.learning_rate(u64::MAX), 0.0);

	let warmup = WarmupScheduler::new(1.0, 2)?;
	assert_close(warmup.learning_rate(0), 0.5);
	assert_close(warmup.learning_rate(1), 1.0);
	assert_close(warmup.learning_rate(2), 1.0);

	let warmup_then_cosine =
		WarmupScheduler::with_after(1.0, 2, Box::new(CosineScheduler::new(1.0, 0.0, 2)?))?;
	assert_close(warmup_then_cosine.learning_rate(1), 1.0);
	assert_close(warmup_then_cosine.learning_rate(2), 1.0);
	assert_close(warmup_then_cosine.learning_rate(3), 0.5);
	assert_close(warmup_then_cosine.learning_rate(4), 0.0);

	let one_cycle = OneCycleScheduler::new(1.0, 10, 0.3, 25.0, 10_000.0)?;
	let one_cycle_default = OneCycleScheduler::with_defaults(1.0, 10)?;
	assert_close(one_cycle.learning_rate(0), 0.000_004);
	assert_close(one_cycle.learning_rate(3), 1.0);
	assert_close(one_cycle.learning_rate(10), 0.000_004);
	assert_close(
		one_cycle_default.learning_rate(3),
		one_cycle.learning_rate(3),
	);

	let cyclic = CyclicScheduler::new(0.1, 0.5, 2, CyclicMode::Triangular, 1.0)?;
	let triangular = CyclicScheduler::triangular(0.1, 0.5, 2)?;
	for (step, expected) in [(0, 0.1), (1, 0.3), (2, 0.5), (3, 0.3), (4, 0.1)] {
		assert_close(cyclic.learning_rate(step), expected);
		assert_close(triangular.learning_rate(step), expected);
	}

	let fixed_restart = CosineWarmRestartsScheduler::new(1.0, 2, 1, 0.0)?;
	let fixed_restart_default = CosineWarmRestartsScheduler::fixed(1.0, 2)?;
	assert_close(fixed_restart.learning_rate(0), 1.0);
	assert_close(fixed_restart.learning_rate(1), 0.5);
	assert_close(fixed_restart.learning_rate(2), 1.0);
	assert_close(fixed_restart_default.learning_rate(1), 0.5);

	let growing_restart = CosineWarmRestartsScheduler::new(1.0, 2, 2, 0.0)?;
	assert_close(growing_restart.learning_rate(2), 1.0);
	assert_close(growing_restart.learning_rate(4), 0.5);
	assert_close(growing_restart.learning_rate(6), 1.0);

	let sequential = SequentialScheduler::new(
		vec![Box::new(Constant(0.25)), Box::new(Constant(0.5))],
		vec![3],
	)?;
	assert_close(sequential.learning_rate(2), 0.25);
	assert_close(sequential.learning_rate(3), 0.5);

	let composed = LinearWarmupCosineScheduler::new(2, 6, 1.0, 0.0)?;
	assert_close(composed.learning_rate(1), 0.5);
	assert_close(composed.learning_rate(2), 1.0);
	assert_close(composed.learning_rate(4), 0.853_553_4);
	assert_close(composed.learning_rate(5), 0.5);
	assert_close(composed.learning_rate(7), 0.0);
	Ok(())
}

#[test]
fn plateau_schedule_matches_donor_patience_and_floor_contract() -> oa::Result<()> {
	let mut schedule = ReduceOnPlateauScheduler::new(1.0, 0.1, 1, 0.01, 0.05, PlateauMode::Min)?;
	assert_close(
		ReduceOnPlateauScheduler::with_defaults(1.0)?.learning_rate(0),
		1.0,
	);
	schedule.step(2.0)?;
	assert_close(schedule.learning_rate(0), 1.0);
	schedule.step(2.0)?;
	assert_close(schedule.learning_rate(0), 1.0);
	schedule.step(2.0)?;
	assert_close(schedule.learning_rate(0), 0.1);
	schedule.step(2.0)?;
	schedule.step(2.0)?;
	assert_close(schedule.learning_rate(0), 0.05);

	let mut maximize = ReduceOnPlateauScheduler::new(0.5, 0.5, 0, 0.0, 0.0, PlateauMode::Max)?;
	maximize.step(1.0)?;
	maximize.step(0.5)?;
	assert_close(maximize.learning_rate(0), 0.25);
	Ok(())
}

#[test]
fn invalid_scheduler_domains_fail_before_step_evaluation() {
	assert!(CosineScheduler::new(1.0, 0.0, 0).is_err());
	assert!(CosineScheduler::new(0.1, 0.2, 10).is_err());
	assert!(OneCycleScheduler::new(1.0, 10, 1.5, 25.0, 10_000.0).is_err());
	assert!(CyclicScheduler::new(0.1, 1.0, 0, CyclicMode::Triangular, 1.0).is_err());
	assert!(CosineWarmRestartsScheduler::new(1.0, 0, 1, 0.0).is_err());
	assert!(LinearWarmupCosineScheduler::new(4, 4, 1.0, 0.0).is_err());
	assert!(SequentialScheduler::new(vec![Box::new(Constant(1.0))], vec![1]).is_err());
	assert!(ReduceOnPlateauScheduler::new(1.0, 0.0, 1, 0.0, 0.0, PlateauMode::Min).is_err());
}
