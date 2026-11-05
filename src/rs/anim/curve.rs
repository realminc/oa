//! Immutable sparse animation curve.
//!
//! Donor: `oa/animation/animCurve.h`.

use crate::{Result, core::Duration};

use super::{AnimInterpolation, AnimWrapMode, Keyframe, keyframe::AnimCurveValue};

/// Immutable sparse animation curve over keyframe times.
///
/// Construction is validated through [`super::create_curve`]; keyframe times
/// are strictly increasing and every value and tangent is finite.
///
/// Donor: `oa::AnimCurve<T>`.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimCurve<T: AnimCurveValue> {
	pub(crate) interpolation: AnimInterpolation,
	pub(crate) keyframes: Vec<Keyframe<T>>,
}

impl<T: AnimCurveValue> AnimCurve<T> {
	/// Interpolation mode between adjacent keys.
	pub fn interpolation(&self) -> AnimInterpolation {
		self.interpolation
	}

	/// Number of stored keyframes.
	pub fn keyframe_count(&self) -> u32 {
		self.keyframes.len() as u32
	}

	/// Borrowed keyframes in strictly increasing time order.
	pub fn keyframes(&self) -> &[Keyframe<T>] {
		&self.keyframes
	}

	/// Time of the first keyframe.
	///
	/// # Panics
	///
	/// Validated curves always hold at least one keyframe.
	pub fn start_time(&self) -> Duration {
		self.keyframes[0].time
	}

	/// Time of the last keyframe.
	///
	/// # Panics
	///
	/// Validated curves always hold at least one keyframe.
	pub fn end_time(&self) -> Duration {
		self.keyframes[self.keyframes.len() - 1].time
	}

	/// Span between the last and first keyframe times.
	///
	/// The span is guaranteed representable by the construction-time
	/// validation of [`super::create_curve`].
	pub fn duration(&self) -> Duration {
		let first = self.start_time().nanoseconds();
		let last = self.end_time().nanoseconds();
		Duration::from_nanoseconds(last - first)
	}

	/// Sample the curve at `time` under `wrap_mode`.
	///
	/// Convenience delegate of [`super::sample_curve`].
	///
	/// # Errors
	///
	/// Returns an error when interpolation produces a non-finite or
	/// non-normalizable value.
	pub fn sample(&self, time: Duration, wrap_mode: AnimWrapMode) -> Result<T> {
		super::sample_curve(self, time, wrap_mode)
	}
}
