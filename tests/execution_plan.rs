#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn captures_and_replays_an_immutable_plan() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let one = oa::matrix::ones(&engine, [2, 3])?;
	let two = oa::matrix::full(&engine, [2, 3], 2.0)?;
	let (plan, six) = engine.capture(|| {
		let three = oa::matrix::add(&one, &two)?;
		oa::matrix::add(&three, &three)
	})?;

	assert_eq!(
		six.try_read_f32().unwrap_err().kind(),
		oa::ErrorKind::NotReady
	);
	assert_eq!(
		six.read_f32().unwrap_err().kind(),
		oa::ErrorKind::FailedPrecondition
	);
	let captured_input_error = match oa::matrix::add(&six, &six) {
		Ok(_) => panic!("captured output was used before plan submission"),
		Err(error) => error,
	};
	assert_eq!(
		captured_input_error.kind(),
		oa::ErrorKind::FailedPrecondition
	);

	let first = engine.submit(&plan)?;
	first.wait()?;
	assert_eq!(six.read_f32()?, [6.0; 6]);

	let second = engine.submit(&plan)?;
	second.wait()?;
	assert_eq!(six.read_f32()?, [6.0; 6]);
	Ok(())
}

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device with timestamp support"]
fn timed_replays_report_exact_device_durations() -> oa::Result<()> {
	use std::time::Duration;

	let engine = oa::Engine::new()?;
	let one = oa::matrix::ones(&engine, [262_147])?;
	let two = oa::matrix::full(&engine, [262_147], 2.0)?;
	let (plan, output) = engine.capture(|| {
		let three = oa::matrix::add(&one, &two)?;
		oa::matrix::add(&three, &three)
	})?;

	let untimed = engine.submit(&plan)?;
	assert_eq!(
		untimed.device_duration().unwrap_err().kind(),
		oa::ErrorKind::FailedPrecondition
	);

	let first = engine.submit_timed(&plan)?;
	let second = engine.submit_timed(&plan)?;
	let first_duration = first.device_duration()?;
	let second_duration = second.device_duration()?;
	assert!(first_duration > Duration::ZERO);
	assert!(second_duration > Duration::ZERO);
	assert_eq!(first.try_device_duration()?, Some(first_duration));
	assert_eq!(second.try_device_duration()?, Some(second_duration));
	assert_eq!(output.read_f32()?, vec![6.0; 262_147]);

	let surviving_event = {
		let engine = oa::Engine::new()?;
		let input = oa::matrix::ones(&engine, [262_147])?;
		let (plan, _output) = engine.capture(|| oa::matrix::add(&input, &input))?;
		engine.submit_timed(&plan)?
	};
	assert!(surviving_event.device_duration()? > Duration::ZERO);
	Ok(())
}

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn capture_rejects_dirty_empty_nested_and_failed_scopes() -> oa::Result<()> {
	use std::panic::{AssertUnwindSafe, catch_unwind};

	let engine = oa::Engine::new()?;
	let one = oa::matrix::ones(&engine, [2])?;
	let two = oa::matrix::full(&engine, [2], 2.0)?;
	let pending = oa::matrix::add(&one, &two)?;

	let dirty = engine.capture(|| Ok(()));
	assert_eq!(dirty.unwrap_err().kind(), oa::ErrorKind::FailedPrecondition);
	assert_eq!(pending.read_f32()?, [3.0; 2]);

	let empty = engine.capture(|| Ok(()));
	assert_eq!(empty.unwrap_err().kind(), oa::ErrorKind::FailedPrecondition);

	let nested = engine.capture(|| {
		let (_plan, _output) = engine.capture(|| Ok(()))?;
		Ok(())
	});
	assert_eq!(
		nested.unwrap_err().kind(),
		oa::ErrorKind::FailedPrecondition
	);

	let mismatched = oa::matrix::ones(&engine, [1, 2])?;
	let mut escaped = None;
	let failed = engine.capture(|| {
		escaped = Some(oa::matrix::add(&one, &two)?);
		oa::matrix::add(&one, &mismatched)?;
		Ok(())
	});
	assert_eq!(failed.unwrap_err().kind(), oa::ErrorKind::InvalidArgument);
	let Some(escaped) = escaped else {
		panic!("captured output did not escape");
	};
	assert_eq!(
		escaped.read_f32().unwrap_err().kind(),
		oa::ErrorKind::FailedPrecondition
	);

	let mut panic_escaped = None;
	let unwound = catch_unwind(AssertUnwindSafe(|| {
		let _capture = engine.capture(|| -> oa::Result<()> {
			panic_escaped = Some(oa::matrix::add(&one, &two)?);
			panic!("capture unwind probe");
		});
	}));
	assert!(unwound.is_err());
	let Some(panic_escaped) = panic_escaped else {
		panic!("unwound captured output did not escape");
	};
	assert_eq!(
		panic_escaped.read_f32().unwrap_err().kind(),
		oa::ErrorKind::FailedPrecondition
	);
	assert_eq!(oa::matrix::add(&one, &two)?.read_f32()?, [3.0; 2]);
	Ok(())
}

#[test]
#[ignore = "requires a hardware Vulkan 1.3 compute device"]
fn rejects_foreign_plan_without_poisoning_its_origin() -> oa::Result<()> {
	let origin = oa::Engine::new()?;
	let input = oa::matrix::ones(&origin, [3])?;
	let (plan, output) = origin.capture(|| oa::matrix::add(&input, &input))?;
	let foreign = oa::Engine::new()?;

	let error = foreign.submit(&plan).unwrap_err();
	assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	origin.submit(&plan)?.wait()?;
	assert_eq!(output.read_f32()?, [2.0; 3]);
	Ok(())
}
