use super::*;

#[test]
fn audible_clock_respects_output_latency_and_seek_origin() {
	let now = Instant::now();
	let start = now + Duration::from_millis(20);
	let clock = OutputClock {
		origin_frame: 48_000,
		end_frame: 48_480,
		ends_at: Some(start + Duration::from_millis(10)),
	};
	assert_eq!(clock.audible_frame(now, 48_000), 48_000);
	assert_eq!(clock.audible_frame(start, 48_000), 48_000);
	assert_eq!(
		clock.audible_frame(start + Duration::from_millis(5), 48_000),
		48_240
	);
	assert_eq!(
		clock.audible_frame(start + Duration::from_millis(20), 48_000),
		48_480
	);
}

#[test]
fn output_device_notifications_preserve_playback_and_first_terminal_failure() {
	let errors = OutputErrors::default();
	for kind in [
		cpal::ErrorKind::Xrun,
		cpal::ErrorKind::RealtimeDenied,
		cpal::ErrorKind::DeviceChanged,
	] {
		errors.record(kind.into());
	}
	assert_eq!(errors.count.load(Ordering::Relaxed), 3);
	assert!(errors.failure.get().is_none());
	errors.record(cpal::ErrorKind::DeviceNotAvailable.into());
	errors.record(cpal::ErrorKind::BackendError.into());
	assert_eq!(errors.count.load(Ordering::Relaxed), 5);
	assert_eq!(
		errors.failure.get().unwrap().kind(),
		cpal::ErrorKind::DeviceNotAvailable
	);
}
