use super::*;

#[test]
#[ignore = "requires an admitted Vulkan compute device with timestamps"]
fn composed_device_prefix_reaches_snapshot_and_callbacks() -> Result<()> {
	struct Observe(Option<Duration>);
	impl TrainingCallback for Observe {
		fn on_step_end(
			&mut self,
			context: &mut TrainingCallbackContext<'_>,
		) -> Result<TrainingControl> {
			self.0 = context.snapshot().last_gpu_time();
			Ok(TrainingControl::Continue)
		}
	}
	let engine = Engine::builder()
		.devices(crate::DeviceSelection::Index(0))
		.build()?;
	let mut optimizer = crate::ml::NoOpOptimizer::default();
	let mut observed = Observe(None);
	let mut training = ItTraining::new_eager(
		&engine,
		&mut optimizer,
		ItTrainingConfig {
			total_steps: 1,
			enable_gpu_timing: true,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_callback(&mut observed);
	assert!(training.begin_step()?);
	let input = Matrix::new_from_f32(&engine, [], &[1.0])?;
	let loss = crate::matrix::add_scalar(&input, 0.25)?;
	let prefix = Duration::from_secs(1);
	training.complete_step_with_device_prefix(&loss, prefix)?;
	let snapshot = training.finish()?;
	assert!(snapshot.last_gpu_time().is_some_and(|time| time > prefix));
	assert_eq!(observed.0, snapshot.last_gpu_time());
	assert_eq!(snapshot.gpu_timing_stats().count, 1);
	Ok(())
}
