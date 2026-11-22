//! Audio-clocked transport for independently owned audio and video sessions.

use std::{
	path::{Path, PathBuf},
	time::{Duration, Instant},
};

pub use crate::media::types::MediaPlayerConfig;
use crate::{
	AudioPlayer, Engine, Error, LogComponent, LogLevel, Result, VideoFrame, VideoPlayer,
	audio::AudioPlayerConfig,
	video::{
		Mp4TrackEdit, VideoContainerInfo, VideoContainerKind, VideoPlayerConfig, mp4_track_edits,
	},
};

use super::clock::TrackTimeline;

impl MediaPlayerConfig {
	pub fn new(video_path: impl Into<PathBuf>, audio_path: impl Into<PathBuf>) -> Self {
		Self {
			video_path: video_path.into(),
			audio_path: audio_path.into(),
			audio_ring_milliseconds: 500,
		}
	}

	/// Decode the video and default audio tracks of the same local container.
	pub fn embedded(path: impl Into<PathBuf>) -> Self {
		let path = path.into();
		Self::new(path.clone(), path)
	}
}

/// Synchronized playback session over independent audio and video owners.
///
/// The caller ticks video on the Engine-owning thread. During the audio segment,
/// the device output clock paces decoded video. The host clock covers silent
/// lead-in and the tail after audio drains. A presented native frame retains its
/// decoder backing until it is replaced and any external GPU consumers retire.
pub struct MediaPlayer {
	audio: Option<AudioPlayer>,
	video: VideoPlayer,
	video_timeline: TrackTimeline,
	audio_timeline: TrackTimeline,
	duration: Duration,
	position: Duration,
	last_tick: Option<Instant>,
	audio_active: bool,
	audio_needs_seek: bool,
	looping: bool,
	done: bool,
}

impl MediaPlayer {
	/// Open both tracks paused, with the first video frame decoded.
	///
	/// # Errors
	///
	/// Requires a supported video stream, both track sources, and an audio
	/// output configuration that can play the selected PCM.
	pub fn open(engine: &Engine, config: MediaPlayerConfig) -> Result<Self> {
		if config.video_path.as_os_str().is_empty() || config.audio_path.as_os_str().is_empty() {
			return Err(Error::invalid_argument(
				"MediaPlayer requires both track paths",
			));
		}
		let video = open_video_track(engine, &config.video_path)?;
		let audio_config = AudioPlayerConfig {
			uri: config.audio_path,
			loop_playback: false,
			ring_milliseconds: config.audio_ring_milliseconds,
		};
		let embedded = audio_config.uri == config.video_path;
		let audio = if embedded {
			AudioPlayer::open_embedded(engine, audio_config)?
				.ok_or_else(|| Error::not_found("MediaPlayer source has no audio track"))?
		} else {
			AudioPlayer::open(engine, audio_config)?
		};
		Self::compose(&config.video_path, video, Some(audio), embedded)
	}

	pub(crate) fn open_viewer(engine: &Engine, path: &Path) -> Result<Self> {
		let mut video = open_video_track(engine, path)?;
		if crate::EnvFlag::is_set("OA_VIEWER_YUV_CANVAS") {
			video.enable_canvas_yuv()?;
		}
		// MPEG-TS is an unindexed sequential video path. The current Audio
		// probe supports embedded PCM sources through MP4 and Matroska/WebM.
		let info = video
			.info()
			.ok_or_else(|| Error::internal("video track failed to open"))?;
		let audio = if info.kind() == VideoContainerKind::MpegTs {
			None
		} else {
			match AudioPlayer::open_embedded(engine, AudioPlayerConfig::new(path)) {
				Ok(audio) => audio,
				Err(error) => {
					engine.log(
						LogLevel::Warn,
						LogComponent::AUDIO,
						format!("Viewer audio unavailable: {error}"),
					)?;
					None
				}
			}
		};
		Self::compose(path, video, audio, true)
	}

	fn compose(
		path: &Path,
		mut video: VideoPlayer,
		audio: Option<AudioPlayer>,
		embedded: bool,
	) -> Result<Self> {
		let info = video
			.info()
			.ok_or_else(|| Error::internal("video track failed to open"))?;
		let [video_edit, audio_edit] = if embedded && info.kind() == VideoContainerKind::Mp4 {
			mp4_track_edits(
				path,
				[
					Some(info.track_id()),
					audio.as_ref().map(AudioPlayer::track_id),
				],
			)?
		} else {
			[Mp4TrackEdit::default(); 2]
		};
		let video_edit = if info.kind() == VideoContainerKind::MpegTs {
			Mp4TrackEdit {
				media_start: video.current_frame()?.timing().presentation_timestamp(),
				..video_edit
			}
		} else {
			video_edit
		};
		let video_timeline = TrackTimeline::new(video_edit, video.duration()?)?;
		let audio_duration = audio.as_ref().map_or(Duration::ZERO, |audio| {
			Duration::from_micros(audio.duration_us())
		});
		let audio_timeline = TrackTimeline::new(audio_edit, audio_duration)?;
		if let (Some(audio), Some(length)) = (&audio, audio_edit.length) {
			audio.set_playback_end_us(micros(
				audio_edit
					.media_start
					.checked_add(length)
					.ok_or_else(|| Error::out_of_range("audio edit end exceeds Duration"))?,
			));
		}
		if info.kind() != VideoContainerKind::MpegTs && !video_edit.media_start.is_zero() {
			video.seek_time_at_or_before(video_edit.media_start)?;
		}
		Ok(Self {
			audio,
			video,
			video_timeline,
			audio_timeline,
			duration: video_timeline.end.max(audio_timeline.end),
			position: Duration::ZERO,
			last_tick: None,
			audio_active: false,
			audio_needs_seek: true,
			looping: false,
			done: false,
		})
	}

	pub fn play(&mut self) -> Result<()> {
		self.require_open()?;
		if self.is_playing() {
			return Ok(());
		}
		if self.done {
			self.seek_us(0)?;
		}
		self.start_audio()?;
		self.video.play()?;
		self.last_tick = Some(Instant::now());
		Ok(())
	}

	pub fn pause(&mut self) -> Result<()> {
		self.require_open()?;
		if let Some(audio) = &self.audio {
			audio.pause();
		}
		self.audio_active = false;
		self.last_tick = None;
		self.video.pause()
	}

	/// Advance up to `max_frames` display frames reached by the playback clock.
	///
	/// # Errors
	///
	/// Returns an error after close, for a zero budget, or on decoder or device
	/// failure. The transport pauses on a playback failure.
	pub fn tick(&mut self, max_frames: u32) -> Result<u32> {
		self.require_open()?;
		if max_frames == 0 {
			return Err(Error::invalid_argument(
				"MediaPlayer tick budget must be positive",
			));
		}
		let Some(previous) = self.last_tick else {
			return Ok(0);
		};
		let now = Instant::now();
		self.last_tick = Some(now);
		let elapsed = now.saturating_duration_since(previous);
		let result = self.advance_to_clock(elapsed, max_frames);
		if result.is_err() {
			let _ = self.pause();
		}
		result
	}

	fn advance_to_clock(&mut self, elapsed: Duration, budget: u32) -> Result<u32> {
		if let Some(audio) = &self.audio {
			audio.check_playback()?;
		}
		let previous = self.position;
		if self.audio_active {
			let audio = self
				.audio
				.as_ref()
				.ok_or_else(|| Error::internal("active media clock has no audio source"))?;
			let audible = audio.audible_position_us();
			self.position = self
				.position
				.max(self.audio_timeline.movie_at(Duration::from_micros(audible)));
			if audio.is_eos() && audible >= audio.position_us() {
				self.audio_active = false;
			}
		} else {
			self.position = self.position.saturating_add(elapsed);
			if self.audio_needs_seek
				&& self.audio.is_some()
				&& previous < self.audio_timeline.start
				&& self.position >= self.audio_timeline.start
			{
				self.position = self.audio_timeline.start;
				self.start_audio()?;
			}
		}
		if !self.is_seekable() {
			// TS has no indexed end. Pace from the audible clock while sound
			// plays, then let the host clock carry any remaining video.
			let advanced = self
				.video
				.tick_bounded(self.position.saturating_sub(previous), budget)?;
			if self.video.is_done() && !self.audio_active {
				self.finish()?;
			}
			return Ok(advanced);
		}
		if !self.duration.is_zero() {
			self.position = self.position.min(self.duration);
		}
		let mut advanced = 0;
		let video_position = self.position.min(self.video_timeline.end);
		while advanced < budget && !self.video.is_done() {
			let Some(next) = self.video.next_presentation_boundary()? else {
				break;
			};
			if self.video_timeline.movie_at(next) > video_position
				|| self.video_timeline.is_past_end(next)
				|| !self.video.advance_for_playback()?
			{
				break;
			}
			advanced += 1;
		}
		let audio_drained = !self.audio_active;
		if advanced < budget && self.position >= self.duration && audio_drained {
			self.finish()?;
		}
		Ok(advanced)
	}

	fn start_audio(&mut self) -> Result<()> {
		if let Some(audio) = &mut self.audio
			&& self.audio_timeline.contains(self.position)
		{
			if self.audio_needs_seek {
				let position = micros(self.audio_timeline.media_at(self.position));
				if position != 0 || audio.position_us() != 0 || audio.is_eos() {
					audio.seek(position)?;
				}
				self.audio_needs_seek = false;
			}
			if !audio.is_eos() {
				audio.play()?;
				self.audio_active = true;
			}
		}
		Ok(())
	}

	fn finish(&mut self) -> Result<()> {
		if self.looping {
			self.seek_us(0)?;
		} else {
			self.pause()?;
			self.done = true;
		}
		Ok(())
	}

	/// Seek to a movie timestamp, selecting the video frame at or before it.
	/// While paused, the audio decoder is repositioned once on resume.
	///
	/// # Errors
	///
	/// Returns an error when video seeking fails or active audio cannot resume
	/// at the new position. The transport stays paused on failure.
	pub fn seek_us(&mut self, timestamp_us: u64) -> Result<()> {
		self.require_open()?;
		if timestamp_us != 0 && !self.is_seekable() {
			return Err(Error::missing_capability(
				"seeking requires an indexed media source",
			));
		}
		let resume = self.is_playing();
		self.pause()?;
		let target = Duration::from_micros(timestamp_us).min(self.duration);
		let raw = self.video_timeline.media_at(target);
		if !self.is_seekable() {
			self.video.reset()?;
		} else if let Err(error) = self.video.seek_time_at_or_before(raw) {
			let previous = self.video_timeline.media_at(self.position);
			if let Err(restore) = self.video.seek_time_at_or_before(previous) {
				self.video.close();
				return Err(Error::with_kind(
					error.kind(),
					format!(
						"media seek failed ({error}) and the previous frame could not be restored ({restore})"
					),
				));
			}
			self.audio_needs_seek = true;
			return Err(error);
		}
		self.position = target;
		self.audio_needs_seek = true;
		self.done = false;
		if resume {
			self.play()?;
		}
		Ok(())
	}

	pub(crate) fn step_frames(&mut self, steps: i32) -> Result<()> {
		self.pause()?;
		if steps < 0 && !self.is_seekable() {
			return Ok(());
		}
		self.video.step_frames(steps)?;
		let pts = self.current_frame()?.timing().presentation_timestamp();
		self.position = self.video_timeline.movie_at(pts);
		self.audio_needs_seek = true;
		self.done = self.video.is_done() && !self.is_seekable();
		Ok(())
	}

	pub fn current_frame(&self) -> Result<&VideoFrame> {
		self.video.current_frame()
	}

	pub(crate) fn current_viewer_frame(&mut self) -> Result<Option<VideoFrame>> {
		let pts = self.current_frame()?.timing().presentation_timestamp();
		if self.is_seekable() && self.video_timeline.movie_at(pts) > self.position {
			return Ok(None);
		}
		self.video.prepare_viewer_frame().map(Some)
	}

	pub fn position_us(&self) -> Result<u64> {
		self.require_open()?;
		Ok(micros(self.position))
	}

	pub(crate) fn duration_us(&self) -> u64 {
		if self.is_seekable() {
			micros(self.duration)
		} else {
			0
		}
	}

	pub(crate) fn info(&self) -> Option<VideoContainerInfo> {
		self.video.info()
	}

	pub(crate) fn audio(&self) -> Option<&AudioPlayer> {
		self.audio.as_ref()
	}

	pub(crate) fn set_looping(&mut self, looping: bool) {
		self.looping = looping;
	}

	pub fn is_playing(&self) -> bool {
		self.last_tick.is_some()
	}

	pub fn is_done(&self) -> bool {
		self.done || !self.video.is_open()
	}

	pub fn close(&mut self) -> Result<()> {
		self.last_tick = None;
		self.video.close();
		self.audio.take().map_or(Ok(()), |mut audio| audio.close())
	}

	fn require_open(&self) -> Result<()> {
		if self.video.is_open() {
			Ok(())
		} else {
			Err(Error::failed_precondition("MediaPlayer is closed"))
		}
	}

	fn is_seekable(&self) -> bool {
		self
			.info()
			.is_some_and(|info| info.kind() != VideoContainerKind::MpegTs)
	}
}

fn open_video_track(engine: &Engine, path: &Path) -> Result<VideoPlayer> {
	VideoPlayer::new_native(
		engine,
		path,
		VideoPlayerConfig {
			loop_playback: false,
			start_playing: false,
			..VideoPlayerConfig::default()
		},
	)
}

fn micros(duration: Duration) -> u64 {
	u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}
