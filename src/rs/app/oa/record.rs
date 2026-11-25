//! Camera/screen recording policy over capture sessions, Viewer and VideoRecorder.
//! The recorder and camera retain their existing donor-backed implementations.

use std::{path::PathBuf, sync::mpsc, time::Instant};

use oa::input::{InputEvent, InputEventKind, Key};
use oa::ui::{PixelRect, TextLayoutConfig, Ui};
use oa::video::{
	VideoColorMatrix, VideoEncodeConfig, VideoEncodeProfile, VideoRecorder, VideoRecorderConfig,
};
use oa::viewer::{ViewerLiveCapabilities, ViewerLiveSource};
use oa::vision::{CameraCapture, CameraCaptureConfig};
use oa::vision::{ScreenCapture, ScreenCaptureConfig, ScreenCaptureCursor, ScreenCaptureTarget};
use oa::{Color, Error, Result, Runtime, Texture, Viewer, ViewerConfig};

type SaveResult = std::result::Result<Vec<PathBuf>, sdl3::dialog::DialogError>;

pub(super) fn run(device_index: u32) -> Result<()> {
	run_source(RecordingSource::new(device_index), "oa record")
}

pub(super) fn capture(window: bool, hide_cursor: bool) -> Result<()> {
	let mut source = RecordingSource::new(0);
	source.screen_config = Some(ScreenCaptureConfig {
		target: if window {
			ScreenCaptureTarget::Window
		} else {
			ScreenCaptureTarget::Monitor
		},
		cursor: if hide_cursor {
			ScreenCaptureCursor::Hidden
		} else {
			ScreenCaptureCursor::Embedded
		},
		..Default::default()
	});
	source.status = "Waiting for screen selection".into();
	run_source(source, "oa capture")
}

fn run_source(mut source: RecordingSource, title: &str) -> Result<()> {
	let mut viewer = Viewer::new(ViewerConfig {
		title: title.into(),
		show_stats: false,
		show_help: false,
		..Default::default()
	});
	viewer.set_live_source(&mut source);
	viewer.run_standalone()
}

struct RecordingSource {
	config: CameraCaptureConfig,
	camera: Option<CameraCapture>,
	screen_config: Option<ScreenCaptureConfig>,
	screen: Option<ScreenCapture>,
	texture: Option<Texture>,
	pending_pts: Option<u64>,
	pts_origin: Option<u64>,
	recorder: Option<VideoRecorder>,
	path: Option<PathBuf>,
	started: Option<Instant>,
	next_screen_encode: Option<Instant>,
	toggle: bool,
	save_requested: bool,
	dialog_active: bool,
	tx: mpsc::Sender<SaveResult>,
	rx: mpsc::Receiver<SaveResult>,
	status: String,
}

impl RecordingSource {
	fn new(device_index: u32) -> Self {
		let (tx, rx) = mpsc::channel();
		Self {
			config: CameraCaptureConfig {
				device_index,
				..Default::default()
			},
			camera: None,
			screen_config: None,
			screen: None,
			texture: None,
			pending_pts: None,
			pts_origin: None,
			recorder: None,
			path: None,
			started: None,
			next_screen_encode: None,
			toggle: false,
			save_requested: false,
			dialog_active: false,
			tx,
			rx,
			status: "Waiting for camera · Space to record".into(),
		}
	}

	fn start(&mut self, runtime: &Runtime) -> Result<()> {
		self.start_to(runtime, recording_path()?)
	}

	fn start_to(&mut self, runtime: &Runtime, path: PathBuf) -> Result<()> {
		let (width, height) = if let Some(texture) = &self.texture {
			(texture.width() as u32, texture.height() as u32)
		} else if let Some(camera) = &self.camera {
			(camera.width(), camera.height())
		} else {
			return Err(Error::failed_precondition(
				"capture has not produced its first frame",
			));
		};
		let mut encode =
			VideoEncodeConfig::new(VideoEncodeProfile::h264_high_420_8bit(), width, height);
		encode.frame_rate = self.camera.as_ref().map_or_else(
			|| self.screen_config.as_ref().map_or(30, |c| c.preferred_fps),
			|c| c.frame_rate().round().max(1.0) as u32,
		);
		let recorder = VideoRecorder::new(
			runtime,
			VideoRecorderConfig {
				output_path: path
					.to_str()
					.ok_or_else(|| Error::invalid_argument("recording path must be UTF-8"))?
					.to_owned(),
				encode,
				color_space: VideoColorMatrix::Bt709,
				full_range: false,
				audio: None,
			},
		)?;
		self.path = Some(path);
		self.recorder = Some(recorder);
		self.started = Some(Instant::now());
		self.pts_origin = None;
		self.next_screen_encode = None;
		self.status = "Recording · Space to stop".into();
		Ok(())
	}

	fn write_pending_frame(&mut self) -> Result<()> {
		// Static desktop sources only publish damage. Repeat the latest frame
		// at the requested cadence so a motionless screen still records time.
		if self.screen.is_some() && self.recorder.is_some() {
			let now = Instant::now();
			let period = std::time::Duration::from_secs_f64(
				1.0
					/ f64::from(
						self
							.recorder
							.as_ref()
							.map_or(30, |r| r.config().encode.frame_rate),
					),
			);
			let due = self.next_screen_encode.unwrap_or(now);
			if now < due {
				return Ok(());
			}
			if let Some(started) = self.started {
				self.pending_pts = Some(
					u64::try_from(started.elapsed().as_micros())
						.map_err(|_| Error::out_of_range("recording timestamp overflow"))?,
				);
				// Keep the cadence anchored across display-frame jitter; avoid a
				// 30 Hz recorder slipping to 20 Hz on a nominal 60 Hz viewer.
				self.next_screen_encode = Some(if now.duration_since(due) > period {
					now + period
				} else {
					due + period
				});
			}
		}
		if let (Some(recorder), Some(texture), Some(pts)) =
			(&mut self.recorder, &self.texture, self.pending_pts.take())
		{
			let origin = *self.pts_origin.get_or_insert(pts);
			let relative = pts
				.checked_sub(origin)
				.ok_or_else(|| Error::data_loss("capture timestamp moved backwards"))?;
			recorder.write_rgba(&texture.read_rgba8()?, relative)?;
		}
		Ok(())
	}

	fn stop(&mut self) -> Result<()> {
		if let Some(mut recorder) = self.recorder.take() {
			let result = if recorder.frame_count() == 0 {
				Err(Error::failed_precondition(
					"no capture frames were recorded",
				))
			} else {
				recorder.finalize()
			};
			recorder.close();
			self.started = None;
			if let Err(error) = result {
				self.path = None;
				return Err(error);
			}
			if let Some(path) = &self.path {
				self.status = format!("Saved {}", path.display());
			}
			self.save_requested = true;
		}
		Ok(())
	}

	fn save_dialog(&mut self) -> Result<()> {
		if self.dialog_active || self.path.is_none() {
			return Ok(());
		}
		let tx = self.tx.clone();
		sdl3::dialog::show_save_file_dialog(
			&[sdl3::dialog::DialogFileFilter {
				name: "MP4 video",
				pattern: "mp4",
			}],
			self.path.as_deref(),
			None::<&sdl3::video::Window>,
			Box::new(move |result, _| {
				let _ = tx.send(result);
			}),
		)
		.map_err(|e| {
			Error::io(
				"recording save dialog",
				std::io::Error::other(e.to_string()),
			)
		})?;
		self.dialog_active = true;
		Ok(())
	}
}

impl ViewerLiveSource for RecordingSource {
	fn capabilities(&self) -> ViewerLiveCapabilities {
		ViewerLiveCapabilities {
			receives_events: true,
			..Default::default()
		}
	}
	fn open(&mut self, runtime: &Runtime) -> Result<()> {
		if let Some(config) = &self.screen_config {
			self.screen = Some(ScreenCapture::open(runtime, config)?);
		} else {
			self.camera = Some(CameraCapture::open(runtime, &self.config)?);
		}
		Ok(())
	}
	fn update(&mut self, _: f32) -> Result<()> {
		let frame = if let Some(screen) = &mut self.screen {
			screen.poll()?
		} else {
			self
				.camera
				.as_mut()
				.map(CameraCapture::poll)
				.transpose()?
				.flatten()
		};
		if let Some(frame) = frame {
			if self.recorder.as_ref().is_some_and(|r| {
				r.config().encode.width as usize != frame.width()
					|| r.config().encode.height as usize != frame.height()
			}) {
				self.stop()?;
				self.status = "Source resized · recording saved · Space starts a new recording".into();
			}
			if self.texture.is_none() {
				self.status = "Ready · Space to record".into();
			}
			self.pending_pts = Some(
				u64::try_from(frame.timing().presentation_timestamp().as_micros())
					.map_err(|_| Error::out_of_range("capture timestamp exceeds u64"))?,
			);
			self.texture = frame.into_texture();
		}
		if let Ok(result) = self.rx.try_recv() {
			self.dialog_active = false;
			match result {
				Ok(paths) => {
					if let (Some(source), Some(destination)) = (&self.path, paths.first())
						&& source != destination
					{
						// Keep the original recording on any copy failure, including cross-filesystem saves.
						match save_copy(source, destination) {
							Ok(_) => {
								self.status = format!("Saved {}", destination.display());
							}
							Err(error) => {
								self.status = format!("Save failed: {error}. Original: {}", source.display());
							}
						}
					}
				}
				Err(sdl3::dialog::DialogError::Canceled) => {}
				Err(error) => self.status = format!("Save dialog: {error}"),
			}
		}
		Ok(())
	}
	fn canvas_texture(&self) -> Option<&Texture> {
		self.texture.as_ref()
	}
	fn canvas_controls_rect(&self, width: u32, height: u32, scale: f32) -> Option<PixelRect> {
		let controls_height = (120.0 * scale).round() as u32;
		Some(PixelRect::new(
			0,
			height.saturating_sub(controls_height) as i32,
			width,
			controls_height,
		))
	}
	fn render(&mut self, ui: &mut Ui<'_>, width: u32, height: u32) -> Result<()> {
		let scale = ui.content_scale();
		let size = (64.0 * scale).round() as u32;
		let bottom = (108.0 * scale).round() as u32;
		let rect = PixelRect::new(
			(width.saturating_sub(size) / 2) as i32,
			height.saturating_sub(bottom) as i32,
			size,
			size,
		);
		if ui.text_button(
			"record-toggle",
			rect,
			"",
			Color::new(1.0, 1.0, 1.0, 1.0),
			self.texture.is_some() && !self.dialog_active,
		) {
			self.toggle = true;
		}
		if self.toggle {
			self.toggle = false;
			let result = if self.recorder.is_some() {
				self.stop()
			} else {
				self.start(ui.runtime())
			};
			if let Err(error) = result {
				self.status = error.to_string();
			}
		}
		let inset = if self.recorder.is_some() {
			size / 3
		} else {
			size / 5
		};
		let inner = PixelRect::new(
			rect.x + inset as i32,
			rect.y + inset as i32,
			size - inset * 2,
			size - inset * 2,
		);
		ui.rect_outline_rounded_at(
			rect,
			Color::new(0.9, 0.9, 0.9, 1.0),
			(2.0 * scale).max(1.0) as u32,
			rect,
			size as f32 / 2.0,
		);
		ui.rect(
			inner,
			Color::new(0.9, 0.15, 0.18, 1.0),
			if self.recorder.is_some() {
				3.0 * scale
			} else {
				size as f32 / 2.0
			},
		)?;
		self.write_pending_frame()?;
		if self.save_requested {
			self.save_requested = false;
			if let Err(error) = self.save_dialog() {
				self.status = error.to_string();
			}
		}
		let label = if let Some(start) = self.started {
			let seconds = start.elapsed().as_secs();
			format!("{:02}:{:02} · Space to stop", seconds / 60, seconds % 60)
		} else {
			self.status.clone()
		};
		ui.text_at(
			&label,
			[16.0 * scale, height as f32 - 24.0 * scale],
			TextLayoutConfig {
				size: 14.0 * scale,
				..Default::default()
			},
			Color::new(0.9, 0.9, 0.9, 1.0),
			PixelRect::new(0, 0, width, height),
		)?;
		Ok(())
	}
	fn on_event(&mut self, event: &InputEvent) -> Result<()> {
		if event.kind == InputEventKind::KeyDown && !event.key_repeat && !self.dialog_active {
			if event.key == Key::Space && self.texture.is_some() {
				self.toggle = true;
			}
			if event.key == Key::S && event.ctrl() {
				if self.recorder.is_some() {
					self.toggle = true;
				} else {
					self.save_requested = true;
				}
			}
		}
		Ok(())
	}
	fn close(&mut self) -> Result<()> {
		let result = self.stop();
		if let Some(camera) = &mut self.camera {
			camera.close();
		}
		let screen_result = self.screen.as_mut().map_or(Ok(()), ScreenCapture::close);
		if let Some(path) = &self.path {
			eprintln!(
				"oa {}: {}",
				if self.screen_config.is_some() {
					"capture"
				} else {
					"record"
				},
				path.display()
			);
		}
		result.and(screen_result)
	}
}

fn recording_path() -> Result<PathBuf> {
	use sdl3::filesystem::{Folder, get_user_folder};

	let videos = match get_user_folder(Folder::VIDEOS) {
		Ok(path) => path.to_owned(),
		Err(_) => {
			let home = std::env::var_os("HOME")
				.ok_or_else(|| Error::failed_precondition("Videos folder and HOME are unavailable"))?;
			PathBuf::from(home).join("Videos")
		}
	};
	let root = videos.join("OA");
	std::fs::create_dir_all(&root).map_err(|error| Error::io("create recording directory", error))?;
	let stamp = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_err(|e| Error::failed_precondition(e.to_string()))?
		.as_nanos();
	let directory = root.join(format!("recording-{stamp}-{}", std::process::id()));
	std::fs::create_dir(&directory)
		.map_err(|error| Error::io("reserve recording directory", error))?;
	Ok(directory.join("recording.mp4"))
}

/// Stage the copy beside its destination, then replace it atomically. In
/// particular, saving onto a symlink/hardlink to the original cannot truncate it.
fn save_copy(source: &std::path::Path, destination: &std::path::Path) -> std::io::Result<()> {
	use std::io::Write;
	let parent = destination
		.parent()
		.filter(|p| !p.as_os_str().is_empty())
		.unwrap_or_else(|| std::path::Path::new("."));
	let stamp = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_err(std::io::Error::other)?
		.as_nanos();
	let temporary = parent.join(format!(".oa-record-{stamp}-{}.tmp", std::process::id()));
	let mut output = std::fs::OpenOptions::new()
		.write(true)
		.create_new(true)
		.open(&temporary)?;
	let result = (|| {
		std::io::copy(&mut std::fs::File::open(source)?, &mut output)?;
		output.flush()?;
		output.sync_all()?;
		drop(output);
		std::fs::rename(&temporary, destination)
	})();
	if result.is_err() {
		let _ = std::fs::remove_file(&temporary);
	}
	result
}

#[cfg(test)]
#[path = "../../../../test/rs/app/record_unit.rs"]
mod tests;
