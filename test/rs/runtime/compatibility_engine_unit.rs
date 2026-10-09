/// Public Engine proof for the bounded-binding profile. On machines without a
/// Vulkan 1.2 compute device, device admission is covered by synthetic tests.
#[test]
fn bounded_engine_records_eager_and_reusable_graphs() -> crate::Result<()> {
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
	let left = crate::matrix::from_f32_on(
		&engine,
		[17],
		&(0..17).map(|i| i as f32).collect::<Vec<_>>(),
	)?;
	let right = crate::matrix::full(&engine, [17], 2.0)?;
	let first = crate::matrix::add(&left, &right)?;
	let second = crate::matrix::add(&first, &right)?;
	if engine.device_info().api_version().starts_with("1.2.") {
		let difference = crate::matrix::sub(&left, &right)?;
		assert_eq!(
			difference.read_f32()?,
			(0..17).map(|i| i as f32 - 2.0).collect::<Vec<_>>()
		);
		let ui_error = match crate::ui::Ui::init(&engine, 8, 8) {
			Ok(_) => panic!("unqualified UI compositor was admitted"),
			Err(error) => error,
		};
		assert_eq!(ui_error.kind(), crate::ErrorKind::MissingCapability);
	}
	assert_eq!(
		second.read_f32()?,
		(0..17).map(|i| i as f32 + 4.0).collect::<Vec<_>>()
	);

	let (plan, output) = engine.capture(|| {
		let sum = crate::matrix::add(&left, &right)?;
		crate::matrix::add(&sum, &sum)
	})?;
	let first_event = engine.submit(&plan)?;
	let second_event = engine.submit(&plan)?;
	first_event.wait()?;
	second_event.wait()?;
	assert_eq!(
		output.read_f32()?,
		(0..17).map(|i| (i as f32 + 2.0) * 2.0).collect::<Vec<_>>()
	);
	Ok(())
}
