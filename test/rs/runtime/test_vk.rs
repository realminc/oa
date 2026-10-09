fn test_log_directory() -> std::path::PathBuf {
	std::env::temp_dir().join(format!("oars-engine-log-{}", std::process::id()))
}

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn creates_and_drops_engine_on_hardware_vulkan() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let info = engine.device_info();
	assert!(!info.name().is_empty());
	assert!(!info.api_version().is_empty());
	assert_ne!(info.capability_identity(), 0);
	assert!(info.max_compute_workgroup_invocations() > 0);
	assert!(info.storage_buffer_descriptors() >= 3);
	assert!(info.storage_image_descriptors() >= 1);
	Ok(())
}

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn engine_owns_flushes_and_closes_its_logging_session() -> Result<(), Box<dyn std::error::Error>> {
	let directory = test_log_directory();
	let _ = std::fs::remove_dir_all(&directory);
	let engine = oa::Engine::builder()
		.logging(
			oa::LogOptions::new()
				.directory(&directory)
				.prefix("runtime-test")
				.minimum_level(oa::LogLevel::Info)
				.console_output(false)
				.file_output(true),
		)
		.build()?;
	oa::log_info!(oa::LogComponent::RUNTIME, "selected engine logger");
	engine.flush_log()?;
	let path = engine
		.log_path()
		.ok_or_else(|| std::io::Error::other("engine did not expose its log path"))?;
	let contents = std::fs::read_to_string(&path)?;
	assert!(contents.contains("[ENGN] oa engine v"));
	assert!(contents.contains("Vulkan · 1 compute device"));
	assert!(contents.contains("] ComputeDevice ·"));
	assert!(contents.contains("Driver ·"));
	assert!(contents.contains("Hardware · PCI"));
	assert!(contents.contains("Capability identity · oa.device.v1."));
	assert!(contents.contains("Compute · "));
	assert!(contents.contains("queried f16"));
	assert!(contents.contains("[RT  ] selected engine logger"));
	engine.close()?;
	std::fs::remove_dir_all(directory)?;
	Ok(())
}

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn builder_selects_automatic_and_exact_devices() -> oa::Result<()> {
	let automatic = oa::Engine::builder()
		.devices(oa::DeviceSelection::Automatic)
		.build()?;
	drop(automatic);

	let exact = oa::Engine::builder()
		.devices(oa::DeviceSelection::Index(0))
		.build()?;
	drop(exact);
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan loader"]
fn builder_rejects_an_out_of_range_device_index() {
	let result = oa::Engine::builder()
		.devices(oa::DeviceSelection::Index(usize::MAX))
		.build();
	let error = match result {
		Ok(_) => panic!("out-of-range Vulkan device index was accepted"),
		Err(error) => error,
	};

	assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	assert!(error.message().contains("out of range"));
}

#[test]
#[ignore = "requires a Vulkan 1.3 loader"]
fn presentation_requirement_rejects_missing_surface_extensions() {
	let result = oa::Engine::builder()
		.requirements(oa::EngineRequirements::default().presentation())
		.build();
	let error = match result {
		Ok(_) => panic!("presentation requirement accepted an instance without surface extensions"),
		Err(error) => error,
	};
	assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	assert!(error.message().contains("surface instance extensions"));
}

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn default_matrix_source_reuses_its_thread_engine_and_rejects_foreign_values() -> oa::Result<()> {
	oa::Engine::init_default()?;
	let first = oa::matrix::from_f32([2], &[1.0, 2.0])?;
	let second = oa::Engine::with_default(|default| {
		// Reentrant use of the default selector must not hold an exclusive borrow.
		let nested = oa::matrix::from_f32([2], &[3.0, 4.0])?;
		let explicit = oa::matrix::from_f32_on(default, [2], &[5.0, 6.0])?;
		let sum = oa::matrix::add(&nested, &explicit)?;
		assert_eq!(sum.read_f32()?, [8.0, 10.0]);
		Ok(nested)
	})?;
	assert_eq!(oa::matrix::add(&first, &second)?.read_f32()?, [4.0, 6.0]);

	let foreign_engine = oa::Engine::new()?;
	let foreign = oa::matrix::from_f32_on(&foreign_engine, [2], &[1.0, 1.0])?;
	let error = match oa::matrix::add(&first, &foreign) {
		Ok(_) => panic!("mixed-engine operands were accepted"),
		Err(error) => error,
	};
	assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	Ok(())
}

test_vk!(checkpoint_event_waits_for_its_exact_epoch, engine, {
	let event = engine.checkpoint()?;
	event.wait()?;
	assert!(event.is_complete()?);
	Ok(())
});

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn event_remains_valid_after_engine_drop() -> oa::Result<()> {
	let event = {
		let engine = oa::Engine::new()?;
		engine.checkpoint()?
	};

	event.wait()?;
	assert!(event.is_complete()?);
	Ok(())
}

test_vk!(dropped_event_does_not_abandon_retirement, engine, {
	let abandoned_handle = engine.checkpoint()?;
	drop(abandoned_handle);

	let following = engine.checkpoint()?;
	following.wait()?;
	assert!(following.is_complete()?);
	Ok(())
});

test_vk!(fp32_storage_round_trips_through_vma, engine, {
	let expected = [1.25_f32, -2.5, 0.0, 128.75, f32::INFINITY, -0.0];
	let matrix = oa::Matrix::from_f32(&engine, [2, 3], &expected)?;

	assert_eq!(matrix.shape(), [2, 3]);
	assert_eq!(matrix.dtype(), oa::DType::F32);
	let actual = matrix.read_f32()?;
	assert_eq!(
		actual
			.iter()
			.map(|value| value.to_bits())
			.collect::<Vec<_>>(),
		expected
			.iter()
			.map(|value| value.to_bits())
			.collect::<Vec<_>>()
	);
	Ok(())
});

test_vk!(zero_extent_matrix_needs_no_vulkan_buffer, engine, {
	let matrix = oa::Matrix::from_f32(&engine, [3, 0, usize::MAX], &[])?;

	assert_eq!(matrix.shape(), [3, 0, usize::MAX]);
	assert!(matrix.read_f32()?.is_empty());
	Ok(())
});

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn matrix_keeps_its_destruction_services_alive() -> oa::Result<()> {
	let matrix = {
		let engine = oa::Engine::new()?;
		oa::Matrix::from_f32(&engine, [2], &[3.5, -7.0])?
	};

	assert_eq!(matrix.read_f32()?, [3.5, -7.0]);
	Ok(())
}

test_vk!(
	matrix_add_matches_an_independent_host_oracle_for_boundary_sizes,
	engine,
	{
		for element_count in [1_usize, 255, 256, 257, 513] {
			let left_values = (0..element_count)
				.map(|index| (index % 29) as f32 - 14.0)
				.collect::<Vec<_>>();
			let right_values = (0..element_count)
				.map(|index| (index % 17) as f32 * 0.5)
				.collect::<Vec<_>>();
			let expected = left_values
				.iter()
				.zip(&right_values)
				.map(|(left, right)| left + right)
				.collect::<Vec<_>>();
			let left = oa::Matrix::from_f32(&engine, [element_count], &left_values)?;
			let right = oa::Matrix::from_f32(&engine, [element_count], &right_values)?;
			let output = oa::matrix::add(&left, &right)?;
			drop(left);
			drop(right);
			let actual = output.read_f32()?;
			assert_eq!(
				actual
					.iter()
					.map(|value| value.to_bits())
					.collect::<Vec<_>>(),
				expected
					.iter()
					.map(|value| value.to_bits())
					.collect::<Vec<_>>()
			);
		}
		Ok(())
	}
);

test_vk!(
	matrix_add_handles_zero_extent_and_rejects_invalid_ownership,
	engine,
	{
		let left = oa::Matrix::from_f32(&engine, [2, 0, usize::MAX], &[])?;
		let right = oa::Matrix::from_f32(&engine, [2, 0, usize::MAX], &[])?;
		let output = oa::matrix::add(&left, &right)?;
		assert_eq!(output.shape(), [2, 0, usize::MAX]);
		assert!(output.read_f32()?.is_empty());

		let other_engine = oa::Engine::new()?;
		let foreign = oa::Matrix::from_f32(&other_engine, [1], &[1.0])?;
		let local = oa::Matrix::from_f32(&engine, [1], &[2.0])?;
		let error = match oa::matrix::add(&local, &foreign) {
			Ok(_) => panic!("cross-engine matrix add was accepted"),
			Err(error) => error,
		};
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		Ok(())
	}
);

test_vk!(
	matrix_add_rejects_shape_mismatch_without_submission,
	engine,
	{
		let left = oa::Matrix::from_f32(&engine, [2], &[1.0, 2.0])?;
		let right = oa::Matrix::from_f32(&engine, [1, 2], &[3.0, 4.0])?;
		let error = match oa::matrix::add(&left, &right) {
			Ok(_) => panic!("shape-mismatched matrix add was accepted"),
			Err(error) => error,
		};
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
		Ok(())
	}
);

test_vk!(
	matrix_add_supports_input_aliasing_and_batched_outputs,
	engine,
	{
		let input = oa::Matrix::from_f32(&engine, [5], &[1.0, -2.0, 3.5, 0.0, 8.0])?;

		let first = oa::matrix::add(&input, &input)?;
		let second = oa::matrix::add(&input, &input)?;
		let error = first
			.try_read_f32()
			.expect_err("non-blocking read submitted or waited for recorded eager work");
		assert_eq!(error.kind(), oa::ErrorKind::NotReady);

		let expected = [2.0_f32, -4.0, 7.0, 0.0, 16.0];
		assert_eq!(first.read_f32()?, expected);
		assert_eq!(second.read_f32()?, expected);
		Ok(())
	}
);

test_vk!(checkpoint_submits_the_pending_eager_batch, engine, {
	let one = oa::matrix::ones(&engine, [4])?;
	let two = oa::matrix::full(&engine, [4], 2.0)?;
	let three = oa::matrix::add(&one, &two)?;
	let six = oa::matrix::add(&three, &three)?;

	assert_eq!(
		six.try_read_f32().unwrap_err().kind(),
		oa::ErrorKind::NotReady
	);
	let submitted = engine.checkpoint()?;
	submitted.wait()?;
	assert_eq!(three.read_f32()?, [3.0; 4]);
	assert_eq!(six.read_f32()?, [6.0; 4]);
	Ok(())
});

test_vk!(
	empty_and_invalid_observation_do_not_flush_unrelated_eager_work,
	engine,
	{
		let left = oa::Matrix::from_slice(&engine, [2], &[1_i32, 2])?;
		let right = oa::Matrix::from_slice(&engine, [2], &[3_i32, 4])?;
		let pending = oa::matrix::add(&left, &right)?;

		let empty_left = oa::Matrix::from_f32(&engine, [0], &[])?;
		let empty_right = oa::Matrix::from_f32(&engine, [0], &[])?;
		let empty = oa::matrix::add(&empty_left, &empty_right)?;
		assert!(empty.read_f32()?.is_empty());
		assert_eq!(
			pending.try_read::<i32>().unwrap_err().kind(),
			oa::ErrorKind::NotReady
		);

		assert_eq!(
			pending.read_f32().unwrap_err().kind(),
			oa::ErrorKind::InvalidArgument
		);
		assert_eq!(
			pending.try_read::<i32>().unwrap_err().kind(),
			oa::ErrorKind::NotReady
		);
		assert_eq!(pending.read::<i32>()?, [4, 6]);
		Ok(())
	}
);

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn matrix_add_resources_survive_public_engine_drop() -> oa::Result<()> {
	let output = {
		let engine = oa::Engine::new()?;
		let left = oa::Matrix::from_f32(&engine, [3], &[1.0, 2.0, 3.0])?;
		let right = oa::Matrix::from_f32(&engine, [3], &[4.0, 5.0, 6.0])?;
		oa::matrix::add(&left, &right)?
	};

	assert_eq!(output.read_f32()?, [5.0, 7.0, 9.0]);
	Ok(())
}

test_vk!(
	matrix_constructors_and_chained_add_need_no_manual_submission,
	engine,
	{
		let one = oa::matrix::ones(&engine, [2, 3])?;
		let two = oa::matrix::full(&engine, [2, 3], 2.0)?;
		let three = oa::matrix::add(&one, &two)?;
		let six = oa::matrix::add(&three, &three)?;

		assert_eq!(three.read_f32()?, [3.0; 6]);
		assert_eq!(six.read_f32()?, [6.0; 6]);
		Ok(())
	}
);

#[test]
#[ignore = "requires a software Vulkan CPU ICD"]
fn software_vulkan_index_executes_compute_and_crypto() -> oa::Result<()> {
	let devices = oa::Engine::available_devices()?;
	let cpu = devices
		.iter()
		.find(|device| device.kind() == oa::DeviceKind::Cpu)
		.expect("software Vulkan CPU device was not enumerated");
	let engine = oa::Engine::builder()
		.devices(oa::DeviceSelection::Index(cpu.index()))
		.build()?;
	assert_eq!(engine.device_info().index(), cpu.index());
	assert_eq!(engine.device_info().name(), cpu.name());
	assert_eq!(engine.device_info().kind(), oa::DeviceKind::Cpu);
	let a = oa::Matrix::from_f32(&engine, [3], &[1.0, -2.0, 0.5])?;
	let sum = oa::matrix::add(&a, &a)?;
	assert_eq!(sum.read_f32()?, [2.0, -4.0, 1.0]);
	let input = oa::Matrix::from_slice(
		&engine,
		[3, 3],
		&[11_u8, 48, 85, 122, 159, 196, 233, 14, 51],
	)?;
	let digest = oa::cryptography::hash::shake256(&input, 32)?;
	let mut expected = Vec::new();
	for row in [11_u8, 48, 85, 122, 159, 196, 233, 14, 51].chunks(3) {
		let mut bytes = [0_u8; 32];
		oa::cryptography::shake256(row, &mut bytes);
		expected.extend_from_slice(&bytes);
	}
	assert_eq!(digest.read::<u8>()?, expected);
	// Pure software enumeration must never become an implicit CPU fallback.
	if devices
		.iter()
		.all(|device| device.kind() == oa::DeviceKind::Cpu)
	{
		let error = match oa::Engine::new() {
			Ok(_) => panic!("automatic election silently selected software Vulkan"),
			Err(error) => error,
		};
		assert_eq!(error.kind(), oa::ErrorKind::NoSuitableDevice);
	}
	Ok(())
}
