//! Camera/screen capture contract tests.

use oa::vision::{
	CameraCapture, CameraCaptureConfig, ScreenCapture, ScreenCaptureConfig, ScreenCaptureCursor,
	ScreenCaptureTarget,
};

#[test]
fn camera_config_default_matches_donor() {
	let config = CameraCaptureConfig::default();
	assert_eq!(config.device_index, 0);
	assert_eq!(config.width, 1280);
	assert_eq!(config.height, 720);
	assert_eq!(config.fps, 30);
}

#[test]
fn screen_config_default_matches_donor() {
	let config = ScreenCaptureConfig::default();
	assert_eq!(config.target, ScreenCaptureTarget::MonitorOrWindow);
	assert_eq!(config.cursor, ScreenCaptureCursor::Embedded);
	assert_eq!(config.preferred_width, 1920);
	assert_eq!(config.preferred_height, 1080);
	assert_eq!(config.preferred_fps, 30);
	assert_eq!(config.ring_frames, 4);
}

test_vk!(camera_open_rejects_zero_extent, engine, {
	for (width, height) in [(0, 720), (1280, 0)] {
		let config = CameraCaptureConfig {
			width,
			height,
			..Default::default()
		};
		match CameraCapture::open(&engine, &config) {
			Err(error) => assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument),
			Ok(_) => panic!("camera open accepted a zero extent {width}x{height}"),
		}
	}
	Ok(())
});

test_vk!(camera_open_rejects_zero_fps, engine, {
	let config = CameraCaptureConfig {
		fps: 0,
		..Default::default()
	};
	match CameraCapture::open(&engine, &config) {
		Err(error) => assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument),
		Ok(_) => panic!("camera open accepted a zero frame rate"),
	}
	Ok(())
});

test_vk!(camera_open_rejects_missing_device_index, engine, {
	let config = CameraCaptureConfig {
		device_index: u32::MAX,
		..Default::default()
	};
	match CameraCapture::open(&engine, &config) {
		Err(error) => assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument),
		Ok(_) => panic!("camera open accepted an out-of-range device index"),
	}
	Ok(())
});

test_vk!(
	camera_poll_close_contract,
	engine,
	"requires a Vulkan device and an OS-visible camera",
	{
		let mut capture = match CameraCapture::open(&engine, &CameraCaptureConfig::default()) {
			Ok(capture) => capture,
			Err(error) => {
				assert_eq!(error.kind(), oa::ErrorKind::MissingCapability);
				eprintln!("no camera present; contract path verified at open");
				return Ok(());
			}
		};
		assert!(capture.is_streaming());
		assert!(capture.width() > 0);
		assert!(capture.height() > 0);
		assert!(capture.frame_rate() > 0.0);
		// Polling never blocks: drain until a frame lands or give up after a
		// bounded number of attempts covering several negotiated frame periods.
		let mut frame = None;
		for _ in 0..300 {
			match capture.poll() {
				Ok(Some(found)) => {
					frame = Some(found);
					break;
				}
				Ok(None) => std::thread::sleep(std::time::Duration::from_millis(10)),
				Err(error) => panic!("camera poll failed: {error}"),
			}
		}
		let frame = frame.expect("camera produced no frame within 3 s of polling");
		assert_eq!(frame.width(), capture.width() as usize);
		assert_eq!(frame.height(), capture.height() as usize);
		assert!(frame.as_texture().is_some());
		// A second immediate poll must not block or corrupt state.
		match capture.poll() {
			Ok(_) | Err(_) => {}
		}
		capture.close();
		assert!(!capture.is_streaming());
		match capture.poll() {
			Err(error) => assert_eq!(error.kind(), oa::ErrorKind::FailedPrecondition),
			Ok(_) => panic!("poll after close did not fail"),
		}
		Ok(())
	}
);

test_vk!(screen_capture_reports_capability_gate, engine, {
	assert!(!ScreenCapture::is_supported());
	match ScreenCapture::open(&engine, &ScreenCaptureConfig::default()) {
		Err(error) => assert_eq!(error.kind(), oa::ErrorKind::MissingCapability),
		Ok(_) => panic!("screen capture opened without a linked backend"),
	}
	Ok(())
});

test_vk!(screen_capture_rejects_zero_hints, engine, {
	for (key, config) in [
		(
			"extent",
			ScreenCaptureConfig {
				preferred_width: 0,
				..Default::default()
			},
		),
		(
			"frame rate",
			ScreenCaptureConfig {
				preferred_fps: 0,
				..Default::default()
			},
		),
		(
			"ring depth",
			ScreenCaptureConfig {
				ring_frames: 0,
				..Default::default()
			},
		),
	] {
		match ScreenCapture::open(&engine, &config) {
			Err(error) => assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument, "{key}"),
			Ok(_) => panic!("screen capture accepted a zero {key}"),
		}
	}
	Ok(())
});
