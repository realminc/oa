//! Audio-clocked playback of one video track and one independent audio source.

use std::path::PathBuf;

use crate::{
	AudioPlayer, Engine, Error, Result, VideoFrame, VideoPlayer, audio::AudioPlayerConfig,
	video::VideoPlayerConfig,
};

use super::MediaTimestamp;

/// Two local tracks whose presentation timestamps share a zero-based epoch.
///
/// Audio currently uses the incremental WAV/FLAC/MP3 source boundary of
/// [`AudioPlayer`]. An audio track embedded in MP4 or Matroska is not yet
/// extracted by OA; callers must provide a separately decodable audio file.
#[derive(Clone, Debug)]
pub struct MediaPlayerConfig {
	pub video_path: PathBuf,
	pub audio_path: PathBuf,
	pub audio_ring_milliseconds: u32,
}

impl MediaPlayerConfig {
	pub fn new(video_path: impl Into<PathBuf>, audio_path: impl Into<PathBuf>) -> Self {
		Self {
			video_path: video_path.into(),
			audio_path: audio_path.into(),
			audio_ring_milliseconds: 500,
		}
	}
}

/// Synchronized playback session over independent audio and video owners.
///
/// Audio output frames consumed by the device callback are the master clock.
/// Call [`tick`](Self::tick) from the engine-owning thread to advance video.
/// Decode and audio callback workers never borrow or submit through `Engine`.
/// The current frame retains its decoder backing and must obey
/// [`VideoFrame::mark_consumed`] before its storage can be recycled.
pub struct MediaPlayer {
	audio: AudioPlayer,
	video: VideoPlayer,
	playing: bool,
}

impl MediaPlayer {
	/// Open both tracks paused, with the first video frame decoded.
	///
	/// Native video output preserves its GPU backing. Opening requires a
	/// compatible Vulkan Video device and an audio output device.
	pub fn open(engine: &Engine, config: MediaPlayerConfig) -> Result<Self> {
		if config.video_path.as_os_str().is_empty() || config.audio_path.as_os_str().is_empty() {
			return Err(Error::invalid_argument(
				"MediaPlayer requires both track paths",
			));
		}
		let video = VideoPlayer::open_native(
			engine,
			&config.video_path,
			VideoPlayerConfig {
				loop_playback: false,
				start_playing: false,
				..VideoPlayerConfig::default()
			},
		)?;
		let audio = AudioPlayer::open(
			engine,
			AudioPlayerConfig {
				uri: config.audio_path,
				loop_playback: false,
				ring_milliseconds: config.audio_ring_milliseconds,
			},
		)?;
		Ok(Self {
			audio,
			video,
			playing: false,
		})
	}

	pub fn play(&mut self) -> Result<()> {
		self.require_open()?;
		if self.audio.is_eos() {
			self.seek_us(0)?;
		}
		self.audio.play()?;
		self.playing = true;
		Ok(())
	}

	pub fn pause(&mut self) -> Result<()> {
		self.require_open()?;
		self.audio.pause();
		self.playing = false;
		Ok(())
	}

	/// Advance video up to the next frame boundary reached by audio output.
	///
	/// The caller controls cadence and may limit work per tick. A zero limit is
	/// rejected so that a forgotten budget cannot silently stall presentation.
	/// Returns the number of frames advanced. It never reads pixels to the host.
	pub fn tick(&mut self, max_frames: u32) -> Result<u32> {
		self.require_open()?;
		if max_frames == 0 {
			return Err(Error::invalid_argument(
				"MediaPlayer tick budget must be positive",
			));
		}
		if !self.playing {
			return Ok(0);
		}
		let audio_eos = self.audio.is_eos();
		let result = (|| {
			let clock = MediaTimestamp::from_microseconds(self.audio.position_us());
			let mut advanced = 0;
			while advanced < max_frames && !self.video.is_done() {
				let Some(boundary) = self.video.next_presentation_boundary()? else {
					break;
				};
				if !clock.has_reached(boundary) || !self.video.advance()? {
					break;
				}
				advanced += 1;
			}
			Ok(advanced)
		})();
		match result {
			Ok(advanced) => {
				if audio_eos && (advanced < max_frames || self.video.is_done()) {
					// A full budget may leave due frames for the next tick.
					self.playing = false;
				}
				Ok(advanced)
			}
			Err(error) => {
				self.audio.pause();
				self.playing = false;
				Err(error)
			}
		}
	}

	/// Seek both tracks to one timestamp in microseconds.
	///
	/// On a partial seek failure the session closes: it must not present tracks
	/// from different epochs. Reopening is the recovery boundary.
	pub fn seek_us(&mut self, timestamp_us: u64) -> Result<()> {
		self.require_open()?;
		let was_playing = self.playing;
		self.audio.pause();
		self.playing = false;
		let result = (|| {
			let info = self
				.video
				.info()
				.ok_or_else(|| Error::failed_precondition("video closed"))?;
			let ticks =
				MediaTimestamp::from_microseconds(timestamp_us).to_video_ticks(info.time_base())?;
			self.video.seek_at_or_before(ticks)?;
			self.audio.seek(timestamp_us)
		})();
		if let Err(error) = result {
			let _ = self.close();
			return Err(error);
		}
		if was_playing {
			self.play()?;
		}
		Ok(())
	}

	pub fn current_frame(&self) -> Result<&VideoFrame> {
		self.require_open()?;
		self.video.current_frame()
	}

	pub fn position_us(&self) -> Result<u64> {
		self.require_open()?;
		Ok(self.audio.position_us())
	}

	/// Whether transport remains active, including bounded video catch-up at audio EOS.
	pub fn is_playing(&self) -> bool {
		self.playing
	}

	/// True after audio EOS and any bounded video catch-up has finished.
	pub fn is_done(&self) -> bool {
		self.audio.is_eos() && !self.playing
	}

	pub fn close(&mut self) -> Result<()> {
		self.playing = false;
		self.video.close();
		self.audio.close()
	}

	fn require_open(&self) -> Result<()> {
		if self.video.is_open() && self.audio.is_open() {
			Ok(())
		} else {
			Err(Error::failed_precondition("MediaPlayer is closed"))
		}
	}
}
