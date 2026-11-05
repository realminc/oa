//! Named host/device timer tests.
//!
//! Donor: `oacpp/test/cpp/runtime/profiling/testTimer.cpp` — host region
//! lifecycle, device timer initialization against an engine, timer-tagged
//! submission, pending/committed state transitions, and throughput.

use super::*;

#[test]
fn timer_host_region_commits_elapsed_and_throughput() {
	let mut timer = Timer::new(TimerDomain::Host, "host.region");
	assert_eq!(timer.name(), "host.region");
	assert!(timer.is_host());
	assert!(!timer.is_device());
	assert!(timer.is_initialized());
	assert!(!timer.is_pending());
	assert_eq!(timer.elapsed_milliseconds(), 0.0);
	assert_eq!(timer.throughput(), 0.0);

	timer.begin().expect("begin");
	assert!(timer.is_pending());
	std::thread::sleep(std::time::Duration::from_millis(1));
	timer.end().expect("end");
	assert!(!timer.is_pending());

	let milliseconds = timer.commit(10.0).expect("commit");
	assert!(milliseconds > 0.0);
	assert_eq!(timer.elapsed_milliseconds(), milliseconds);
	assert!((timer.throughput() - 10_000.0 / milliseconds).abs() < 1e-9);
}

#[test]
fn timer_host_rejects_unpaired_regions() {
	let mut timer = Timer::new(TimerDomain::Host, "host.misuse");
	assert_eq!(
		timer.end().expect_err("end without begin").kind(),
		crate::ErrorKind::FailedPrecondition
	);
	assert_eq!(
		timer.commit(1.0).expect_err("commit without region").kind(),
		crate::ErrorKind::FailedPrecondition
	);
	timer.begin().expect("begin");
	assert_eq!(
		timer.begin().expect_err("nested begin").kind(),
		crate::ErrorKind::FailedPrecondition
	);
	timer.end().expect("end");
}

#[test]
fn timer_auto_resolves_to_host_on_begin() {
	let mut timer = Timer::new(TimerDomain::Auto, "auto.host");
	assert_eq!(timer.domain(), TimerDomain::Auto);
	assert!(!timer.is_initialized());
	timer.begin().expect("begin");
	assert_eq!(timer.domain(), TimerDomain::Host);
	assert!(timer.is_host());
	timer.end().expect("end");
	assert!(timer.commit(1.0).expect("commit") > 0.0);
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn timer_device_commits_measured_submissions() -> crate::Result<()> {
	let engine = match crate::Engine::new() {
		Ok(engine) => engine,
		Err(_) => return Ok(()),
	};

	// A host timer and an uninitialized device timer cannot tag submissions.
	let mut host = Timer::new(TimerDomain::Host, "host.submit");
	assert_eq!(
		engine
			.checkpoint_with_timer(&mut host)
			.expect_err("host timer cannot tag submissions")
			.kind(),
		crate::ErrorKind::FailedPrecondition
	);
	let mut uninitialized = Timer::new(TimerDomain::Device, "device.uninitialized");
	assert_eq!(
		engine
			.checkpoint_with_timer(&mut uninitialized)
			.expect_err("device timer requires init")
			.kind(),
		crate::ErrorKind::FailedPrecondition
	);

	let mut timer = Timer::new(TimerDomain::Device, "device.region");
	assert!(!timer.is_initialized());
	timer.init(&engine, "device.region")?;
	assert!(timer.is_device());
	assert!(timer.is_initialized());
	assert!(!timer.is_pending());

	// Repeated timer-tagged checkpoints measure each pending batch; the
	// returned events carry no timing, which belongs to the timer instead.
	for _ in 0..2 {
		let one = crate::matrix::ones(&engine, [262_147])?;
		let _two = crate::matrix::add(&one, &one)?;
		let event = engine.checkpoint_with_timer(&mut timer)?;
		assert_eq!(
			event
				.device_duration()
				.expect_err("timer-tagged submission carries no event timing")
				.kind(),
			crate::ErrorKind::FailedPrecondition
		);
		event.wait()?;
		let milliseconds = timer.commit(1_024.0)?;
		assert!(milliseconds > 0.0);
		assert_eq!(timer.elapsed_milliseconds(), milliseconds);
		assert!(timer.throughput() > 0.0);
	}
	Ok(())
}
