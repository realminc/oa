//! Clock-calibration contract tests.
//!
//! Donor: oacpp/test/cpp/runtime/profiling/testTimer.cpp
//! (ClockCalibration.MapsWrappedDeviceTimestamps and
//! DeviceTimer.CalibratesDeviceAndHostClocks).

use super::*;

#[test]
fn clock_calibration_maps_wrapped_device_timestamps() {
	let calibration = ClockCalibration {
		device_timestamp_ticks: (1_u64 << 36) - 4,
		host_timestamp_nanoseconds: 1000,
		maximum_deviation_nanoseconds: 0,
		device_nanoseconds_per_tick: 2.0,
		device_timestamp_valid_bits: 36,
		host_clock_domain: HostClockDomain::Monotonic,
	};
	assert_eq!(calibration.host_nanoseconds_for_device_timestamp(3), 1014.0);
	assert_eq!(
		calibration.host_nanoseconds_for_device_timestamp((1_u64 << 36) - 9),
		990.0
	);
}

#[test]
fn clock_calibration_rejects_unusable_samples() {
	let mut calibration = ClockCalibration {
		device_timestamp_ticks: 10,
		host_timestamp_nanoseconds: 1000,
		maximum_deviation_nanoseconds: 0,
		device_nanoseconds_per_tick: 2.0,
		device_timestamp_valid_bits: 64,
		host_clock_domain: HostClockDomain::Monotonic,
	};
	assert_eq!(
		calibration.host_nanoseconds_for_device_timestamp(14),
		1008.0
	);
	calibration.device_timestamp_valid_bits = 0;
	assert_eq!(calibration.host_nanoseconds_for_device_timestamp(14), 0.0);
	calibration.device_timestamp_valid_bits = 65;
	assert_eq!(calibration.host_nanoseconds_for_device_timestamp(14), 0.0);
	calibration.device_timestamp_valid_bits = 64;
	calibration.device_nanoseconds_per_tick = 0.0;
	assert_eq!(calibration.host_nanoseconds_for_device_timestamp(14), 0.0);
}

#[test]
#[ignore = "requires an admitted Vulkan compute device"]
fn clock_calibration_calibrates_device_and_host_clocks() -> crate::Result<()> {
	let engine = match crate::Engine::new() {
		Ok(engine) => engine,
		Err(_) => return Ok(()),
	};
	let calibration = engine.calibrate_clock(16);
	if !engine.supports_clock_calibration() {
		let error = calibration.expect_err("unsupported device must reject calibration");
		assert_eq!(error.kind(), crate::ErrorKind::Unavailable);
		return Ok(());
	}
	let calibration = calibration?;
	assert!(calibration.device_nanoseconds_per_tick > 0.0);
	assert!(calibration.device_timestamp_valid_bits >= 36);
	assert!(calibration.device_timestamp_valid_bits <= 64);
	assert!(calibration.maximum_deviation_nanoseconds > 0);
	assert_eq!(
		calibration.host_nanoseconds_for_device_timestamp(calibration.device_timestamp_ticks),
		calibration.host_timestamp_nanoseconds as f64
	);

	let clock = match calibration.host_clock_domain {
		HostClockDomain::MonotonicRaw => libc::CLOCK_MONOTONIC_RAW,
		HostClockDomain::Monotonic => libc::CLOCK_MONOTONIC,
	};
	let mut now = libc::timespec {
		tv_sec: 0,
		tv_nsec: 0,
	};
	// SAFETY: `now` is live writable storage for the duration of the call.
	assert_eq!(unsafe { libc::clock_gettime(clock, &mut now) }, 0);
	let now_nanoseconds = now.tv_sec as u64 * 1_000_000_000 + now.tv_nsec as u64;
	assert!(now_nanoseconds >= calibration.host_timestamp_nanoseconds);
	assert!(now_nanoseconds - calibration.host_timestamp_nanoseconds < 1_000_000_000);
	Ok(())
}
