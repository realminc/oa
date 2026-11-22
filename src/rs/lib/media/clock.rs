//! Checked microsecond timeline shared by audio output and video presentation.

use std::time::Duration;

use crate::{
	Error, Result,
	video::{VideoFrameTiming, VideoTimeBase},
};

/// Non-negative position on a media session's zero-based microsecond timeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct MediaTimestamp(u64);

impl MediaTimestamp {
	pub const fn new_from_microseconds(value: u64) -> Self {
		Self(value)
	}

	pub const fn as_microseconds(self) -> u64 {
		self.0
	}

	/// Whether an absolute presentation boundary has been reached.
	pub fn has_reached(self, boundary: Duration) -> bool {
		boundary <= Duration::from_micros(self.0)
	}

	/// Whether a frame's declared end has been reached by this clock.
	///
	/// Sequential sources require a known duration; indexed sources can use
	/// their next presentation timestamp through [`has_reached`](Self::has_reached).
	pub fn frame_is_due(self, timing: VideoFrameTiming) -> Result<bool> {
		let duration = timing.duration().ok_or_else(|| {
			Error::missing_capability("MediaPlayer requires known video frame durations")
		})?;
		let next_boundary = timing
			.presentation_timestamp()
			.checked_add(duration)
			.ok_or_else(|| Error::out_of_range("video presentation time overflowed"))?;
		Ok(self.has_reached(next_boundary))
	}

	/// Convert to an encoded track's rational timestamp ticks, rounding down.
	pub fn to_video_ticks(self, base: VideoTimeBase) -> Result<u64> {
		let ticks = u128::from(self.0) * u128::from(base.denominator())
			/ (u128::from(base.numerator()) * 1_000_000);
		u64::try_from(ticks).map_err(|_| Error::out_of_range("media timestamp exceeds video time base"))
	}
}

/// Map a continuous track segment between its raw and movie timestamp epochs.
#[derive(Clone, Copy)]
pub(super) struct TrackTimeline {
	pub start: Duration,
	pub end: Duration,
	media_start: Duration,
	media_end: Option<Duration>,
}

impl TrackTimeline {
	pub fn new(edit: crate::video::Mp4TrackEdit, raw_duration: Duration) -> Result<Self> {
		let length = edit
			.length
			.unwrap_or_else(|| raw_duration.saturating_sub(edit.media_start));
		let media_end = edit
			.length
			.map(|length| {
				edit
					.media_start
					.checked_add(length)
					.ok_or_else(|| Error::out_of_range("media segment end exceeds Duration"))
			})
			.transpose()?;
		let end = edit
			.movie_start
			.checked_add(length)
			.ok_or_else(|| Error::out_of_range("movie edit end exceeds Duration"))?;
		Ok(Self {
			start: edit.movie_start,
			end,
			media_start: edit.media_start,
			media_end,
		})
	}

	pub fn movie_at(&self, media: Duration) -> Duration {
		self
			.start
			.saturating_add(media.saturating_sub(self.media_start))
	}

	pub fn media_at(&self, movie: Duration) -> Duration {
		let elapsed = movie.saturating_sub(self.start);
		let time = self.media_start.saturating_add(elapsed);
		self.media_end.map_or(time, |end| {
			time.min(end.saturating_sub(Duration::from_nanos(1)))
		})
	}

	pub fn contains(&self, movie: Duration) -> bool {
		movie >= self.start && (self.end == self.start || movie < self.end)
	}

	pub fn is_past_end(&self, media: Duration) -> bool {
		self.media_end.is_some_and(|end| media >= end)
	}
}
