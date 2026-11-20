//! Composed demux, decode, and presentation-order playback sessions.

use std::{collections::VecDeque, path::Path, time::Duration};

use crate::{Engine, Error, Result, Texture};

use super::{VideoContainerInfo, VideoContainerKind, VideoDecoder, VideoDemuxer, VideoFrame};
pub use crate::video::types::VideoPlayerConfig;

impl Default for VideoPlayerConfig {
	fn default() -> Self {
		Self {
			loop_playback: true,
			frame_rate_override: None,
			start_playing: true,
			presentation_cache_frames: 32,
			presentation_cache_bytes: 256 * 1024 * 1024,
		}
	}
}

/// Observable counters for one player lifetime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VideoPlayerStats {
	/// Frames presented by open, stepping, ticking, or seeking.
	pub presented_frames: u64,
	/// Packets submitted to the decoder.
	pub decoded_packets: u64,
	/// Explicit or looping seek resets.
	pub seek_resets: u64,
	/// End-of-stream wraps performed by looping playback.
	pub loop_restarts: u64,
	/// Frames decoded again while rebuilding history after a cache miss.
	pub seek_replay_frames: u64,
	/// Exact display steps served from retained history.
	pub presentation_cache_hits: u64,
	/// Backward/forward history requests that required deterministic replay.
	pub presentation_cache_misses: u64,
	/// Number of frames currently retained in presentation history.
	pub presentation_cache_resident: usize,
	/// Configured maximum number of retained presentation frames.
	pub presentation_cache_capacity: usize,
}

/// Stateful local video player composed from [`VideoDemuxer`] and [`VideoDecoder`].
///
/// `open` decodes through the first display-order frame, so [`current_frame`](Self::current_frame)
/// is immediately available on success. Timing controls do not create a second
/// decoder or runtime owner; all hardware work remains inside the supplied
/// Engine-owned decoder session.
pub struct VideoPlayer {
	state: Option<PlayerState>,
}

struct PlayerState {
	demuxer: VideoDemuxer,
	decoder: VideoDecoder,
	output: PlayerOutput,
	info: VideoContainerInfo,
	config: VideoPlayerConfig,
	current: Option<VideoFrame>,
	queued: VecDeque<VideoFrame>,
	decoder_drained: bool,
	presentation_cache: VecDeque<CachedPresentation>,
	presentation_cache_bytes: usize,
	canvas_yuv: bool,
	frame_interval: Duration,
	duration: Duration,
	accumulator: Duration,
	playing: bool,
	done: bool,
	display_index: u64,
	next_display_index: u64,
	display_timestamps: Vec<DisplayTimestamp>,
	indexed: bool,
	stats: VideoPlayerStats,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlayerOutput {
	HostPlanar,
	Native,
}

struct CachedPresentation {
	index: u64,
	frame: VideoFrame,
	bytes: usize,
}

#[derive(Clone, Copy)]
struct DisplayTimestamp {
	ticks: u64,
	duration: Duration,
}

impl VideoPlayer {
	/// Open a local source and present its first display-order frame.
	///
	/// # Errors
	///
	/// Returns an error for invalid pacing, demux/decode failure, unsupported
	/// codec/device capabilities, or a source with no displayable frame.
	pub fn open(engine: &Engine, path: impl AsRef<Path>, config: VideoPlayerConfig) -> Result<Self> {
		Self::open_with_output(engine, path, config, PlayerOutput::HostPlanar)
	}

	/// Open a player whose frames retain the decoder's native Vulkan images.
	///
	/// This route does not read decoded pixels back to the host. Frames returned
	/// by [`current_frame`](Self::current_frame) retain their decoder slots;
	/// GPU consumers must register completion with [`VideoFrame::mark_consumed`].
	/// Converted presentation textures use the configured bounded history;
	/// decoder slots themselves are never cached. A backward cache miss rebuilds
	/// a preceding window once, so subsequent reverse steps reuse GPU textures.
	///
	/// A native frame is not a render [`crate::Texture`]. A renderer must
	/// explicitly convert or sample its planes before displaying it.
	///
	/// # Errors
	///
	/// Returns the same stream and device errors as [`open`](Self::open).
	pub fn new_native(
		engine: &Engine,
		path: impl AsRef<Path>,
		config: VideoPlayerConfig,
	) -> Result<Self> {
		Self::open_with_output(engine, path, config, PlayerOutput::Native)
	}

	fn open_with_output(
		engine: &Engine,
		path: impl AsRef<Path>,
		config: VideoPlayerConfig,
		output: PlayerOutput,
	) -> Result<Self> {
		if config
			.frame_rate_override
			.is_some_and(|rate| !rate.is_finite() || rate <= 0.0)
		{
			return Err(Error::invalid_argument(
				"video player frame-rate override must be finite and positive",
			));
		}
		let demuxer = VideoDemuxer::open(path)?;
		let info = demuxer.info();
		let indexed = info.kind() != VideoContainerKind::MpegTs;
		let (timestamp_ticks, presentation_end) = if indexed {
			demuxer.presentation_timestamps()?
		} else {
			(Vec::new(), 0)
		};
		let mut display_timestamps = Vec::new();
		display_timestamps
			.try_reserve_exact(timestamp_ticks.len())
			.map_err(|_| Error::resource_exhausted("video display index allocation failed"))?;
		for ticks in timestamp_ticks {
			display_timestamps.push(DisplayTimestamp {
				ticks,
				duration: ticks_to_duration(ticks, info)?,
			});
		}
		let rate = config
			.frame_rate_override
			.unwrap_or_else(|| info.frame_rate());
		let rate = if rate.is_finite() && rate > 0.0 {
			rate
		} else {
			30.0
		};
		let frame_interval = Duration::try_from_secs_f64(1.0 / rate)
			.ok()
			.filter(|interval| !interval.is_zero())
			.ok_or_else(|| {
				Error::out_of_range("video player frame interval is outside Duration precision")
			})?;
		let duration = ticks_to_duration(presentation_end, info)?;
		let decoder = VideoDecoder::new(engine, info)?;
		let presentation_cache_capacity = if !indexed {
			0
		} else {
			let frame_bytes = usize::try_from(info.width())
				.ok()
				.and_then(|width| {
					usize::try_from(info.height())
						.ok()
						.and_then(|height| width.checked_mul(height))
				})
				.and_then(|luma| match output {
					// Current Texture storage retains both its packed buffer and
					// sampled image. Charge both copies to the history budget.
					PlayerOutput::Native => luma.checked_mul(8),
					PlayerOutput::HostPlanar => luma.checked_mul(3)?.checked_div(2),
				})
				.ok_or_else(|| Error::out_of_range("video player frame size overflows usize"))?;
			config.presentation_cache_frames.min(
				config
					.presentation_cache_bytes
					.checked_div(frame_bytes)
					.unwrap_or(0),
			)
		};
		let mut player = Self {
			state: Some(PlayerState {
				demuxer,
				decoder,
				output,
				info,
				config,
				current: None,
				queued: VecDeque::new(),
				decoder_drained: false,
				presentation_cache: VecDeque::new(),
				presentation_cache_bytes: 0,
				canvas_yuv: false,
				frame_interval,
				duration,
				accumulator: Duration::ZERO,
				playing: config.start_playing,
				done: false,
				display_index: 0,
				next_display_index: 0,
				display_timestamps,
				indexed,
				stats: VideoPlayerStats {
					presentation_cache_capacity,
					..VideoPlayerStats::default()
				},
			}),
		};
		if !player.advance_one()? {
			return Err(Error::data_loss(
				"video source contains no displayable frame",
			));
		}
		Ok(player)
	}

	/// Present exactly one next display-order frame, independent of play/pause state.
	///
	/// Returns `false` only after a non-looping source is exhausted.
	pub fn advance(&mut self) -> Result<bool> {
		self.advance_one()
	}

	/// Present the preceding display-order frame.
	///
	/// Retained history is presented without decoding. If the requested frame
	/// has already been evicted, the player seeks through a bounded preceding
	/// presentation window and deterministically rebuilds through the target.
	pub fn step_backward(&mut self) -> Result<()> {
		self.step_frames(-1)
	}

	/// Move by a signed number of display-order frames.
	///
	/// Positive movement stops successfully at the end of a non-looping stream.
	/// Negative movement saturates at the first indexed display frame. A zero
	/// delta is a no-op.
	pub fn step_frames(&mut self, delta: i32) -> Result<()> {
		if delta >= 0 {
			for _ in 0..delta.unsigned_abs() {
				if !self.advance_one()? {
					break;
				}
			}
			return Ok(());
		}

		let current = self.state_ref()?.display_index;
		let target = current.saturating_sub(u64::from(delta.unsigned_abs()));
		if target == current {
			return Ok(());
		}
		if !self.state_ref()?.indexed {
			return Err(Error::missing_capability(
				"backward stepping requires an indexed video source",
			));
		}
		if self.present_cached(target)? {
			return Ok(());
		}
		self.record_cache_miss()?;
		self.replay_to_index(target)
	}

	/// Advance according to elapsed wall-clock time while playing.
	///
	/// Returns the number of frames presented. Paused players do not accumulate
	/// elapsed time, preventing a burst of implicit skipping after resume.
	pub fn tick(&mut self, elapsed: Duration) -> Result<u32> {
		self.tick_bounded(elapsed, u32::MAX)
	}

	pub(crate) fn tick_bounded(&mut self, elapsed: Duration, budget: u32) -> Result<u32> {
		let state = self.state_mut()?;
		if !state.playing || state.done {
			return Ok(0);
		}
		state.accumulator = state
			.accumulator
			.checked_add(elapsed)
			.ok_or_else(|| Error::out_of_range("video player clock accumulator overflowed"))?;
		let mut advanced = 0_u32;
		while advanced < budget {
			let state = self.state_mut()?;
			// Sequential containers lack an index and packet durations. Decode
			// at most one display frame ahead to pace from its real PTS. Keep
			// decoder reorder output in the same queue used by manual stepping.
			let mut interval = state.frame_interval;
			if !state.indexed && state.config.frame_rate_override.is_none() {
				Self::fill_display_queue(state)?;
				if let (Some(current), Some(next)) = (&state.current, state.queued.front()) {
					// PES packets without a timestamp carry zero. Repeated or
					// regressing timestamps retain the last usable cadence.
					if let Some(delta) = next
						.timing()
						.presentation_timestamp()
						.checked_sub(current.timing().presentation_timestamp())
						.filter(|delta| !delta.is_zero())
					{
						interval = delta;
						state.frame_interval = delta;
					}
				}
			}
			if state.accumulator < interval {
				break;
			}
			state.accumulator -= interval;
			if !self.advance_one()? {
				break;
			}
			advanced = advanced
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("video tick frame count overflowed"))?;
		}
		Ok(advanced)
	}

	/// Seek to a presentation timestamp in the selected track time base.
	///
	/// The demuxer rewinds to the preceding keyframe; the player then decodes
	/// forward until it reaches the first frame at or after `timestamp`.
	pub fn seek(&mut self, timestamp: u64) -> Result<()> {
		if !self.state_ref()?.indexed {
			if timestamp != 0 {
				return Err(Error::missing_capability(
					"unindexed video sources support rewind to zero only",
				));
			}
			let state = self.state_mut()?;
			state.decoder.flush()?;
			state.demuxer.seek(0)?;
			state.queued.clear();
			state.decoder_drained = false;
			Self::clear_presentation_cache(state);
			state.current = None;
			state.accumulator = Duration::ZERO;
			state.done = false;
			state.display_index = 0;
			state.next_display_index = 0;
			state.stats.seek_resets = state
				.stats
				.seek_resets
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("video seek counter overflowed"))?;
			if !self.advance_decoded_one()? {
				return Err(Error::data_loss(
					"video source contains no displayable frame",
				));
			}
			return Ok(());
		}
		let target = {
			let state = self.state_ref()?;
			if timestamp >= state.info.duration() && state.info.duration() != 0 {
				return Err(Error::out_of_range(
					"video player seek timestamp is outside the stream duration",
				));
			}
			state
				.display_timestamps
				.partition_point(|point| point.ticks < timestamp)
		};
		if target == self.state_ref()?.display_timestamps.len() {
			return Err(Error::out_of_range(
				"video seek found no display frame at the requested timestamp",
			));
		}
		self.rebuild_to_index(target, false)
	}

	/// Present the most recent indexed frame at or before a track timestamp.
	///
	/// This is the seek policy for an external audio clock: choosing the first
	/// frame after the requested time would display future video early. A
	/// timestamp before the first frame selects that first frame. Unindexed
	/// streams can only be rewound to zero.
	pub fn seek_at_or_before(&mut self, timestamp: u64) -> Result<()> {
		let state = self.state_ref()?;
		if !state.indexed {
			return if timestamp == 0 {
				self.seek(0)
			} else {
				Err(Error::missing_capability(
					"timestamp seeking requires an indexed video source",
				))
			};
		}
		let target = display_index_at_or_before(&state.display_timestamps, timestamp)
			.ok_or_else(|| Error::data_loss("video source has no indexed display frames"))?;
		self.seek_frame(
			u64::try_from(target).map_err(|_| Error::out_of_range("video display index exceeds u64"))?,
		)
	}

	pub(crate) fn seek_time_at_or_before(&mut self, target: Duration) -> Result<()> {
		let state = self.state_ref()?;
		let points = &state.display_timestamps;
		let last = points
			.get(
				points
					.partition_point(|point| point.duration <= target)
					.saturating_sub(1),
			)
			.ok_or_else(|| Error::data_loss("video source has no indexed display frames"))?;
		self.seek_at_or_before(last.ticks)
	}

	/// Seek to an exact zero-based display-order frame.
	///
	/// A retained target is presented without decoding. Otherwise the player
	/// seeks to a bounded preceding presentation window and rebuilds decoder
	/// state through the requested frame.
	pub fn seek_frame(&mut self, index: u64) -> Result<()> {
		if !self.state_ref()?.indexed {
			return Err(Error::missing_capability(
				"frame seeking requires an indexed video source",
			));
		}
		let target =
			usize::try_from(index).map_err(|_| Error::out_of_range("video frame index exceeds usize"))?;
		if target >= self.state_ref()?.display_timestamps.len() {
			return Err(Error::out_of_range(
				"video frame index is outside the display sequence",
			));
		}
		let state = self.state_ref()?;
		if (state.current.is_some() && state.display_index == index) || self.present_cached(index)? {
			return Ok(());
		}
		self.rebuild_to_index(target, false)
	}

	/// Rewind and present the first frame.
	pub fn reset(&mut self) -> Result<()> {
		self.seek(0)
	}

	/// End the current decode sequence and discard delayed frames.
	///
	/// Playback becomes done until [`seek`](Self::seek) or [`reset`](Self::reset)
	/// establishes a new random-access boundary.
	pub fn flush(&mut self) -> Result<usize> {
		let state = self.state_mut()?;
		let discarded = state.decoder.flush()?.len() + state.queued.len();
		state.queued.clear();
		state.decoder_drained = false;
		Self::clear_presentation_cache(state);
		state.done = true;
		state.accumulator = Duration::ZERO;
		Ok(discarded)
	}

	/// Explicitly release decoder and demuxer state. A second close is harmless.
	pub fn close(&mut self) {
		if let Some(mut state) = self.state.take() {
			state.decoder.close();
			state.demuxer.close();
		}
	}

	/// Resume time-paced playback.
	pub fn play(&mut self) -> Result<()> {
		self.state_mut()?.playing = true;
		Ok(())
	}

	/// Pause time-paced playback and discard fractional accumulated time.
	pub fn pause(&mut self) -> Result<()> {
		let state = self.state_mut()?;
		state.playing = false;
		state.accumulator = Duration::ZERO;
		Ok(())
	}

	/// Toggle time-paced playback.
	pub fn toggle_play(&mut self) -> Result<()> {
		let state = self.state_mut()?;
		state.playing = !state.playing;
		if !state.playing {
			state.accumulator = Duration::ZERO;
		}
		Ok(())
	}

	/// Change end-of-stream looping policy.
	pub fn set_looping(&mut self, looping: bool) -> Result<()> {
		self.state_mut()?.config.loop_playback = looping;
		Ok(())
	}

	/// Borrow the currently presented frame.
	pub fn current_frame(&self) -> Result<&VideoFrame> {
		self
			.state_ref()?
			.current
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("VideoPlayer has no current frame"))
	}

	/// Convert the currently presented frame into a `Texture` for rendering.
	///
	/// Native decoded frames convert on-device through the decoder's
	/// native-plane-to-RGBA path. The presented frame is then re-seated on the
	/// texture: its timing, extent, and color metadata remain observable
	/// through [`current_frame`](Self::current_frame) while the decoder slot
	/// is released for the next decode. A texture-backed frame returns its
	/// backing directly, so repeated calls for the same presented frame are
	/// free. Host-planar playback has no texture path.
	///
	/// # Errors
	///
	/// Returns an error after close, when no frame is presented, for a
	/// non-native frame without a texture backing, or when the conversion
	/// submission fails.
	pub fn current_frame_texture(&mut self) -> Result<Texture> {
		let state = self.state_mut()?;
		let frame = state
			.current
			.clone()
			.ok_or_else(|| Error::failed_precondition("VideoPlayer has no current frame"))?;
		if let Some(texture) = frame.as_texture() {
			return Ok(texture.clone());
		}
		let texture = state.decoder.native_frame_texture(&frame)?;
		if frame.native_backing().is_some() {
			let presented =
				VideoFrame::from_texture(texture.clone(), frame.timing(), frame.color_info(), None);
			state.current = Some(presented.clone());
			Self::retain_presentation(state, state.display_index, presented)?;
		}
		Ok(texture)
	}

	pub(crate) fn enable_canvas_yuv(&mut self) -> Result<()> {
		self.state_mut()?.canvas_yuv = true;
		Ok(())
	}

	pub(crate) fn prepare_viewer_frame(&mut self) -> Result<VideoFrame> {
		let state = self.state_mut()?;
		let frame = state
			.current
			.clone()
			.ok_or_else(|| Error::failed_precondition("VideoPlayer has no current frame"))?;
		if frame.canvas_planes().is_some() || frame.as_texture().is_some() {
			return Ok(frame);
		}
		if state.canvas_yuv
			&& matches!(
				frame.native_format(),
				Some(
					crate::video::VideoPixelFormat::P010
						| crate::video::VideoPixelFormat::Yuv420Planar10
						| crate::video::VideoPixelFormat::P012
				)
			) {
			let staged = state.decoder.stage_native_canvas_frame(&frame)?;
			state.current = Some(staged.clone());
			Self::retain_presentation(state, state.display_index, staged.clone())?;
			return Ok(staged);
		}
		self.current_frame_texture()?;
		Ok(self.current_frame()?.clone())
	}

	/// Timestamp at which the next display frame may replace the current one.
	///
	/// Indexed containers use the next frame's PTS, which respects variable
	/// frame rate and gaps. Sequential sources use the current frame's declared
	/// end; an unknown duration cannot be paced by an external clock.
	/// `None` means the indexed stream has no further display frame.
	pub fn next_presentation_boundary(&self) -> Result<Option<Duration>> {
		let state = self.state_ref()?;
		if state.indexed {
			let next = state
				.display_index
				.checked_add(1)
				.ok_or_else(|| Error::out_of_range("video display index overflowed"))?;
			let next = usize::try_from(next)
				.map_err(|_| Error::out_of_range("video display index exceeds usize"))?;
			return Ok(
				state
					.display_timestamps
					.get(next)
					.map(|point| point.duration),
			);
		}
		let timing = self.current_frame()?.timing();
		let duration = timing.duration().ok_or_else(|| {
			Error::missing_capability(
				"unindexed video requires known frame durations for clocked playback",
			)
		})?;
		let boundary = timing
			.presentation_timestamp()
			.checked_add(duration)
			.ok_or_else(|| Error::out_of_range("video presentation time overflowed"))?;
		Ok(Some(boundary))
	}

	/// Return the absolute zero-based display index in the selected track.
	pub fn current_frame_index(&self) -> Result<u64> {
		Ok(self.state_ref()?.display_index)
	}

	/// Return selected stream metadata while open.
	pub fn info(&self) -> Option<VideoContainerInfo> {
		self.state.as_ref().map(|state| state.info)
	}

	pub(crate) fn duration(&self) -> Result<Duration> {
		Ok(self.state_ref()?.duration)
	}

	pub(crate) fn advance_for_playback(&mut self) -> Result<bool> {
		let state = self.state_mut()?;
		if state.indexed
			&& state
				.current
				.as_ref()
				.is_some_and(|frame| frame.native_backing().is_some())
		{
			// Catch-up can skip rendering several successive frames. Release the
			// superseded lease just as the replay path does before decoding again.
			state.current = None;
		}
		self.advance_one()
	}

	/// Return a snapshot of playback counters while open.
	pub fn stats(&self) -> Option<VideoPlayerStats> {
		self.state.as_ref().map(|state| state.stats)
	}

	/// Return whether time-paced playback is enabled.
	pub fn is_playing(&self) -> bool {
		self.state.as_ref().is_some_and(|state| state.playing)
	}

	/// Return whether a non-looping stream has presented its final frame.
	pub fn is_done(&self) -> bool {
		self.state.as_ref().is_none_or(|state| state.done)
	}

	/// Return whether the player still owns its sessions.
	pub const fn is_open(&self) -> bool {
		self.state.is_some()
	}

	fn advance_one(&mut self) -> Result<bool> {
		let cached_target = {
			let state = self.state_ref()?;
			state
				.display_index
				.checked_add(1)
				.filter(|next| *next < state.next_display_index)
		};
		if let Some(target) = cached_target {
			if self.present_cached(target)? {
				return Ok(true);
			}
			self.record_cache_miss()?;
			self.replay_to_index(target)?;
			return Ok(true);
		}
		self.advance_decoded_one()
	}

	fn fill_display_queue(state: &mut PlayerState) -> Result<()> {
		while state.queued.is_empty() && !state.decoder_drained {
			let Some(packet) = state.demuxer.read_next_packet()? else {
				state.queued.extend(state.decoder.flush()?);
				state.decoder_drained = true;
				break;
			};
			state.stats.decoded_packets = state
				.stats
				.decoded_packets
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("video decoded-packet counter overflowed"))?;
			let frame = match state.output {
				PlayerOutput::HostPlanar => state.decoder.decode(&packet)?,
				PlayerOutput::Native => state.decoder.decode_native(&packet)?,
			};
			state.queued.extend(frame);
		}
		Ok(())
	}

	fn advance_decoded_one(&mut self) -> Result<bool> {
		let state = self.state_mut()?;
		if state.done {
			return Ok(false);
		}
		Self::fill_display_queue(state)?;
		if state.queued.is_empty() && state.config.loop_playback && state.next_display_index != 0 {
			state.demuxer.seek(0)?;
			state.stats.loop_restarts = state
				.stats
				.loop_restarts
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("video loop counter overflowed"))?;
			state.stats.seek_resets = state
				.stats
				.seek_resets
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("video seek counter overflowed"))?;
			state.display_index = 0;
			state.next_display_index = 0;
			state.decoder_drained = false;
			Self::clear_presentation_cache(state);
			Self::fill_display_queue(state)?;
		}
		let Some(frame) = state.queued.pop_front() else {
			state.done = true;
			return Ok(false);
		};
		Self::present_new(state, frame)?;
		if state.indexed
			&& state.decoder_drained
			&& state.queued.is_empty()
			&& !state.config.loop_playback
		{
			state.done = true;
		}
		Ok(true)
	}

	fn present_new(state: &mut PlayerState, frame: VideoFrame) -> Result<()> {
		if !state.indexed {
			let index = state.next_display_index;
			state.next_display_index = index
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("video display index overflowed"))?;
			state.current = Some(frame);
			state.display_index = index;
			state.stats.presented_frames = state
				.stats
				.presented_frames
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("video presented-frame counter overflowed"))?;
			return Ok(());
		}
		let minimum = usize::try_from(state.next_display_index)
			.map_err(|_| Error::out_of_range("video display index exceeds usize"))?;
		let timestamp = frame.timing().presentation_timestamp();
		let remaining = state.display_timestamps.get(minimum..).ok_or_else(|| {
			Error::data_loss("video decoder advanced beyond the container display index")
		})?;
		let relative = remaining.partition_point(|point| point.duration < timestamp);
		let resolved = minimum
			.checked_add(relative)
			.ok_or_else(|| Error::out_of_range("video display index overflowed"))?;
		let Some(point) = state.display_timestamps.get(resolved) else {
			return Err(Error::data_loss(
				"decoded frame timestamp is absent from the container display index",
			));
		};
		if point.duration != timestamp {
			return Err(Error::data_loss(
				"decoded frame timestamp does not match the container display index",
			));
		}
		let index = u64::try_from(resolved)
			.map_err(|_| Error::out_of_range("video display index exceeds u64"))?;
		state.current = Some(frame.clone());
		state.display_index = index;
		state.next_display_index = index
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("video display index overflowed"))?;
		state.stats.presented_frames = state
			.stats
			.presented_frames
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("video presented-frame counter overflowed"))?;
		Self::retain_presentation(state, index, frame)?;
		Ok(())
	}

	fn present_cached(&mut self, index: u64) -> Result<bool> {
		let state = self.state_mut()?;
		let Some(frame) = state
			.presentation_cache
			.iter()
			.find(|entry| entry.index == index)
			.map(|entry| entry.frame.clone())
		else {
			return Ok(false);
		};
		state.current = Some(frame);
		state.display_index = index;
		state.done = false;
		state.stats.presented_frames = state
			.stats
			.presented_frames
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("video presented-frame counter overflowed"))?;
		state.stats.presentation_cache_hits = state
			.stats
			.presentation_cache_hits
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("video cache-hit counter overflowed"))?;
		Ok(true)
	}

	fn record_cache_miss(&mut self) -> Result<()> {
		let state = self.state_mut()?;
		state.stats.presentation_cache_misses = state
			.stats
			.presentation_cache_misses
			.checked_add(1)
			.ok_or_else(|| Error::resource_exhausted("video cache-miss counter overflowed"))?;
		Ok(())
	}

	fn replay_to_index(&mut self, target: u64) -> Result<()> {
		let target = usize::try_from(target)
			.map_err(|_| Error::out_of_range("video replay index exceeds usize"))?;
		self.rebuild_to_index(target, true)
	}

	fn rebuild_to_index(&mut self, target: usize, replay: bool) -> Result<()> {
		let (timestamp, history_start) = {
			let state = self.state_ref()?;
			let history = state.stats.presentation_cache_capacity.max(1);
			let seek_start = target.saturating_add(1).saturating_sub(history);
			let timestamp = state
				.display_timestamps
				.get(seek_start)
				.ok_or_else(|| Error::out_of_range("video rebuild index is outside the stream"))?
				.ticks;
			(timestamp, seek_start)
		};
		{
			let state = self.state_mut()?;
			state.decoder.flush()?;
			state.demuxer.seek(timestamp)?;
			state.queued.clear();
			state.decoder_drained = false;
			Self::clear_presentation_cache(state);
			state.current = None;
			state.accumulator = Duration::ZERO;
			state.done = false;
			state.display_index = 0;
			state.next_display_index = 0;
			state.stats.seek_resets = state
				.stats
				.seek_resets
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("video seek counter overflowed"))?;
		}
		loop {
			// Rebuild intermediates are private, superseded frames. Release our
			// previous native lease before reserving the next DPB output; holding
			// it until present_new can exhaust AV1's reference-heavy slot pool.
			self.state_mut()?.current = None;
			if !self.advance_decoded_one()? {
				return Err(Error::data_loss(
					"video rebuild ended before the requested display frame",
				));
			}
			let current = {
				let state = self.state_mut()?;
				if replay {
					state.stats.seek_replay_frames = state
						.stats
						.seek_replay_frames
						.checked_add(1)
						.ok_or_else(|| Error::resource_exhausted("video replay counter overflowed"))?;
				}
				usize::try_from(state.display_index)
					.map_err(|_| Error::out_of_range("video display index exceeds usize"))?
			};
			// Cache independent presentation resources, not DPB leases. Populate
			// the preceding window during this replay rather than repeating a
			// random-access decode for every held-left key repeat.
			if self.state_ref()?.output == PlayerOutput::Native
				&& self.state_ref()?.stats.presentation_cache_capacity > 0
				&& current >= history_start
			{
				self.prepare_viewer_frame()?;
			}
			if current == target {
				return Ok(());
			}
			if current > target {
				return Err(Error::data_loss(
					"video rebuild skipped the requested display frame",
				));
			}
		}
	}

	fn retain_presentation(state: &mut PlayerState, index: u64, frame: VideoFrame) -> Result<()> {
		let bytes = if let Some((planes, _)) = frame.canvas_planes() {
			Some(planes.byte_len())
		} else if let Some(texture) = frame.as_texture() {
			texture
				.width()
				.checked_mul(texture.height())
				.and_then(|pixels| pixels.checked_mul(8))
		} else if let Some(samples) = frame.as_yuv420p10() {
			samples.len().checked_mul(2)
		} else {
			frame.as_yuv420p().map(<[u8]>::len)
		};
		let Some(bytes) = bytes else {
			return Ok(());
		};
		if state.stats.presentation_cache_capacity == 0
			|| state.config.presentation_cache_bytes == 0
			|| bytes > state.config.presentation_cache_bytes
		{
			return Ok(());
		}
		if let Some(position) = state
			.presentation_cache
			.iter()
			.position(|entry| entry.index == index)
		{
			let previous = state
				.presentation_cache
				.remove(position)
				.ok_or_else(|| Error::internal("video cache entry disappeared"))?;
			state.presentation_cache_bytes -= previous.bytes;
		}
		state.presentation_cache_bytes = state
			.presentation_cache_bytes
			.checked_add(bytes)
			.ok_or_else(|| Error::resource_exhausted("video cache byte count overflowed"))?;
		state.presentation_cache.push_back(CachedPresentation {
			index,
			frame,
			bytes,
		});
		while state.presentation_cache.len() > state.stats.presentation_cache_capacity
			|| state.presentation_cache_bytes > state.config.presentation_cache_bytes
		{
			let evicted = state
				.presentation_cache
				.pop_front()
				.ok_or_else(|| Error::internal("video presentation cache accounting diverged"))?;
			state.presentation_cache_bytes -= evicted.bytes;
		}
		state.stats.presentation_cache_resident = state.presentation_cache.len();
		Ok(())
	}

	fn clear_presentation_cache(state: &mut PlayerState) {
		state.presentation_cache.clear();
		state.presentation_cache_bytes = 0;
		state.stats.presentation_cache_resident = 0;
	}

	fn state_ref(&self) -> Result<&PlayerState> {
		self
			.state
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("VideoPlayer is closed"))
	}

	fn state_mut(&mut self) -> Result<&mut PlayerState> {
		self
			.state
			.as_mut()
			.ok_or_else(|| Error::failed_precondition("VideoPlayer is closed"))
	}
}

fn ticks_to_duration(ticks: u64, info: VideoContainerInfo) -> Result<Duration> {
	let time_base = info.time_base();
	let nanos = u128::from(ticks)
		.checked_mul(u128::from(time_base.numerator()))
		.and_then(|value| value.checked_mul(1_000_000_000))
		.ok_or_else(|| Error::out_of_range("video player timestamp overflows nanoseconds"))?
		/ u128::from(time_base.denominator());
	let seconds = u64::try_from(nanos / 1_000_000_000)
		.map_err(|_| Error::out_of_range("video player timestamp exceeds Duration"))?;
	let subsec_nanos = u32::try_from(nanos % 1_000_000_000)
		.map_err(|_| Error::out_of_range("video player timestamp remainder exceeds u32"))?;
	Ok(Duration::new(seconds, subsec_nanos))
}

fn display_index_at_or_before(points: &[DisplayTimestamp], ticks: u64) -> Option<usize> {
	let last = points.get(
		points
			.partition_point(|point| point.ticks <= ticks)
			.saturating_sub(1),
	)?;
	Some(points.partition_point(|point| point.ticks < last.ticks))
}
