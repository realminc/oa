use super::*;

#[test]
fn recording_controls_reserve_gesture_starts_at_every_scale() {
	let source = RecordingSource::new(0);
	for scale in [1.0, 1.5, 2.0] {
		let width = (640.0 * scale) as u32;
		let height = (480.0 * scale) as u32;
		let controls = source.canvas_controls_rect(width, height, scale).unwrap();
		assert!(controls.contains(width as f32 * 0.5, height as f32 - 76.0 * scale));
		assert!(!controls.contains(width as f32 * 0.5, height as f32 - 140.0 * scale));
	}
}
#[test]
#[ignore = "requires camera, hardware H.264 encode and a live SDL display"]
fn viewer_records_camera_and_finalizes_on_close()
-> std::result::Result<(), Box<dyn std::error::Error>> {
	preview_records_source(RecordingSource::new(0))
}

#[test]
#[ignore = "requires interactive desktop portal consent, hardware H.264 encode and a live display"]
fn viewer_records_screen_and_finalizes_on_close()
-> std::result::Result<(), Box<dyn std::error::Error>> {
	let mut source = RecordingSource::new(0);
	source.screen_config = Some(ScreenCaptureConfig::default());
	preview_records_source(source)
}

fn preview_records_source(
	source: RecordingSource,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
	struct Preview {
		source: RecordingSource,
		path: PathBuf,
		frames: u64,
		sender: sdl3::event::EventSender,
		opened: Option<Instant>,
		started: bool,
		quitting: bool,
	}
	impl ViewerLiveSource for Preview {
		fn capabilities(&self) -> ViewerLiveCapabilities {
			self.source.capabilities()
		}
		fn open(&mut self, runtime: &Runtime) -> Result<()> {
			self.source.open(runtime)?;
			self.opened = Some(Instant::now());
			Ok(())
		}
		fn update(&mut self, delta: f32) -> Result<()> {
			self.source.update(delta)
		}
		fn canvas_texture(&self) -> Option<&Texture> {
			self.source.canvas_texture()
		}
		fn canvas_controls_rect(&self, width: u32, height: u32, scale: f32) -> Option<PixelRect> {
			self.source.canvas_controls_rect(width, height, scale)
		}
		fn render(&mut self, ui: &mut Ui<'_>, width: u32, height: u32) -> Result<()> {
			if !self.started && self.source.texture.is_some() {
				self.source.start_to(ui.runtime(), self.path.clone())?;
				self.started = true;
			}
			self.source.render(ui, width, height)?;
			if !self.quitting
				&& self
					.source
					.started
					.is_some_and(|start| start.elapsed().as_secs_f64() >= 2.0)
			{
				self
					.sender
					.push_event(sdl3::event::Event::Quit { timestamp: 0 })
					.map_err(|error| Error::callback(error.to_string()))?;
				self.quitting = true;
			}
			if self
				.opened
				.is_some_and(|open| open.elapsed().as_secs() >= 15)
			{
				return Err(Error::unavailable("capture preview timed out"));
			}
			Ok(())
		}
		fn close(&mut self) -> Result<()> {
			self.frames = self
				.source
				.recorder
				.as_ref()
				.map_or(0, VideoRecorder::frame_count);
			self.source.close()
		}
	}
	let path = std::env::temp_dir().join(format!("oa-capture-viewer-{}.mp4", std::process::id()));
	let sdl = sdl3::init()?;
	let events = sdl.event()?;
	let mut preview = Preview {
		source,
		path: path.clone(),
		frames: 0,
		sender: events.event_sender(),
		opened: None,
		started: false,
		quitting: false,
	};
	let mut viewer = Viewer::new(ViewerConfig {
		title: "OA capture qualification".into(),
		width: 640,
		height: 480,
		show_help: false,
		..Default::default()
	});
	viewer.set_live_source(&mut preview);
	let result = viewer.run_standalone().and_then(|()| {
		if let Some(texture) = &preview.source.texture {
			eprintln!(
				"capture qualification: {}x{} · {} encoded frames",
				texture.width(),
				texture.height(),
				preview.frames
			);
		}
		// The fixed 30 Hz screen-recording cadence is an application policy.
		// Camera acquisition follows hardware timestamps and may vary with
		// device exposure; retain its existing bounded 20-frame correctness gate.
		let minimum_frames = if preview.source.screen_config.is_some() && !cfg!(debug_assertions) {
			50
		} else {
			20
		};
		if preview.frames < minimum_frames {
			return Err(Error::data_loss(format!(
				"capture recorded only {} frames",
				preview.frames
			)));
		}
		let probe = std::process::Command::new("ffprobe")
			.args([
				"-v",
				"error",
				"-select_streams",
				"v:0",
				"-count_frames",
				"-show_entries",
				"stream=nb_read_frames:format=duration",
				"-of",
				"json",
			])
			.arg(&path)
			.output()
			.map_err(|error| Error::io("probe viewer recording", error))?;
		assert!(
			probe.status.success(),
			"{}",
			String::from_utf8_lossy(&probe.stderr)
		);
		let report: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
		assert_eq!(
			report["streams"][0]["nb_read_frames"]
				.as_str()
				.unwrap()
				.parse::<u64>()
				.unwrap(),
			preview.frames
		);
		assert!(
			report["format"]["duration"]
				.as_str()
				.unwrap()
				.parse::<f64>()
				.unwrap()
				>= 1.5,
			"{report}"
		);
		Ok(())
	});
	let _ = std::fs::remove_file(path);
	Ok(result?)
}

#[test]
#[ignore = "requires camera access and hardware H.264 encode"]
fn camera_recording_finalizes_playable_mp4() -> Result<()> {
	let runtime = Runtime::builder()
		.requirements(
			oa::RuntimeRequirements::default().video_encode(VideoEncodeProfile::h264_high_420_8bit()),
		)
		.build()?;
	Runtime::set_default(runtime.clone());
	let mut source = RecordingSource::new(0);
	source.open(&runtime)?;
	let path = std::env::temp_dir().join(format!("oa-record-hardware-{}.mp4", std::process::id()));
	source.start_to(&runtime, path.clone())?;
	let result = (|| -> Result<()> {
		let deadline = Instant::now() + std::time::Duration::from_secs(10);
		while source
			.recorder
			.as_ref()
			.map_or(0, VideoRecorder::frame_count)
			< 30
			&& Instant::now() < deadline
		{
			source.update(0.0)?;
			source.write_pending_frame()?;
			std::thread::sleep(std::time::Duration::from_millis(5));
		}
		assert!(
			source.recorder.as_ref().unwrap().frame_count() >= 30,
			"camera produced fewer than 30 frames"
		);
		source.stop()?;
		let probe = std::process::Command::new("ffprobe")
			.args([
				"-v",
				"error",
				"-select_streams",
				"v:0",
				"-count_frames",
				"-show_entries",
				"stream=codec_name,nb_read_frames",
				"-of",
				"csv=p=0",
			])
			.arg(&path)
			.output()
			.map_err(|error| Error::io("probe recording", error))?;
		assert!(
			probe.status.success(),
			"{}",
			String::from_utf8_lossy(&probe.stderr)
		);
		let report = String::from_utf8_lossy(&probe.stdout);
		assert!(report.trim().starts_with("h264,30"), "{report}");
		Ok(())
	})();
	let close = source.close();
	let _ = std::fs::remove_file(path);
	result.and(close)
}
#[test]
fn save_preserves_recording_and_replaces_destination() {
	let dir = std::env::temp_dir().join(format!("oa-record-save-{}", std::process::id()));
	std::fs::create_dir_all(&dir).unwrap();
	let source = dir.join("original.mp4");
	let destination = dir.join("saved.mp4");
	std::fs::write(&source, b"recorded movie").unwrap();
	std::fs::write(&destination, b"old").unwrap();
	save_copy(&source, &destination).unwrap();
	assert_eq!(std::fs::read(&source).unwrap(), b"recorded movie");
	assert_eq!(std::fs::read(&destination).unwrap(), b"recorded movie");
	std::fs::remove_file(&destination).unwrap();
	std::fs::hard_link(&source, &destination).unwrap();
	save_copy(&source, &destination).unwrap();
	assert_eq!(std::fs::read(&source).unwrap(), b"recorded movie");
	assert!(save_copy(&source, &dir.join("missing/movie.mp4")).is_err());
	assert_eq!(std::fs::read(&source).unwrap(), b"recorded movie");
	std::fs::remove_dir_all(dir).unwrap();
}
