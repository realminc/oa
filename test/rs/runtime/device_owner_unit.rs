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

/// Engine startup admits metadata only; the first submission publishes
/// retained pipelines lazily, and repeated submissions create none.
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
	// Submit-time fusion probes its candidate kernels on the first
	// submission, so the primed count covers the epilogue/fusion probes and
	// the dispatched add — the contract is that repeats add nothing.
	assert_eq!(matrix::add(&a, &a)?.read_f32()?, vec![2.0, 4.0, 8.0]);
	let primed = device.initialized_pipeline_count();
	assert!(primed > 0, "first submission must initialize pipelines");
	for _ in 0..4 {
		assert_eq!(matrix::add(&a, &a)?.read_f32()?, vec![2.0, 4.0, 8.0]);
		assert_eq!(
			device.initialized_pipeline_count(),
			primed,
			"repeated submission must not create pipelines"
		);
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
		use crate::runtime::device::vkcache;
		let (scope, payload) = {
			let state = self.state.borrow();
			let scope = vkcache::VkCacheScope::new(state.device.physical());
			let (vendor_id, device_id, driver_id, driver_version) = scope.device_route_scope();
			(
				scope.clone(),
				state
					.matmul_route_cache
					.serialize_scope(vendor_id, device_id, driver_id, driver_version),
			)
		};
		let update = match payload {
			Some(bytes) => {
				vkcache::SectionUpdate::replace(vkcache::SECTION_GEMM_MEASURED_ROUTES, 1, bytes)
			}
			None => vkcache::SectionUpdate::remove(vkcache::SECTION_GEMM_MEASURED_ROUTES),
		};
		vkcache::update(&scope.path(), &scope, &[update])
	}
}
