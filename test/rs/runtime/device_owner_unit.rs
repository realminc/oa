/// A resource must retain the complete device domain after its Engine is gone,
/// and the final resource release must not leave an ownership cycle.
#[test]
fn buffer_retains_device_after_engine_drop() -> crate::Result<()> {
	let engine = match crate::Engine::new() {
		Ok(engine) => engine,
		Err(error)
			if matches!(
				error.kind(),
				crate::ErrorKind::NoSuitableDevice | crate::ErrorKind::BackendUnavailable
			) =>
		{
			return Ok(());
		}
		Err(error) => return Err(error),
	};
	let device = engine.handle().state.borrow().device.clone();
	let weak = std::sync::Arc::downgrade(&device);
	let buffer = super::Buffer::host_visible_storage(&device, 4)?;
	buffer.write(0, &[3, 1, 4, 1])?;
	drop(device);
	drop(engine);
	assert!(weak.upgrade().is_some(), "buffer must retain its Device");
	let mut observed = [0; 4];
	buffer.read(0, &mut observed)?;
	assert_eq!(observed, [3, 1, 4, 1]);
	drop(buffer);
	assert!(
		weak.upgrade().is_none(),
		"Device must have no ownership cycle"
	);
	Ok(())
}

/// Engine startup admits metadata only; first use publishes one retained pipeline.
#[test]
#[ignore = "requires an explicitly selected Vulkan device; first-use pipeline proof"]
fn compute_pipelines_are_created_once_on_first_use() -> crate::Result<()> {
	use crate::{DeviceSelection, Engine, matrix};
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit OA_TEST_DEVICE_INDEX")
		.parse::<usize>()
		.expect("device index");
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle().state.borrow().device.clone();
	assert_eq!(device.initialized_pipeline_count(), 0);
	let a = matrix::from_f32_on(&engine, [1, 3], &[1.0, 2.0, 4.0])?;
	assert_eq!(device.initialized_pipeline_count(), 0);
	for _ in 0..2 {
		assert_eq!(matrix::add(&a, &a)?.read_f32()?, vec![2.0, 4.0, 8.0]);
		assert_eq!(device.initialized_pipeline_count(), 1);
	}
	Ok(())
}

impl super::EngineHandle {
	/// Record one measured winner. Callers must prove the variant is legal
	/// for the route key's complete problem and device contract first.
	///
	/// Only the device-gated tuning driver produces measurements; production
	/// code paths read but never write.
	pub(in crate::runtime) fn record_matmul_route(
		&self,
		key: crate::runtime::dna::provider::blas::MatmulRouteKey,
		value: crate::runtime::dna::provider::blas::MatmulRouteValue,
	) {
		self.state.borrow_mut().matmul_route_cache.publish(
			key,
			value.winner_variant,
			value.median_gpu_time_ms,
			value.p95_gpu_time_ms,
			value.sample_count,
		);
	}

	/// Persist measured routes immediately — the donor's explicit
	/// `GemmTuner::saveCache` after a tuning sweep. Engine teardown also
	/// saves best-effort; an explicit save keeps a qualification result
	/// durable even if the process does not reach a clean drop.
	pub(in crate::runtime) fn save_matmul_route_cache(&self) -> crate::Result<()> {
		self
			.state
			.borrow()
			.matmul_route_cache
			.save(&crate::runtime::dna::provider::blas::route_cache_path())
	}
}
