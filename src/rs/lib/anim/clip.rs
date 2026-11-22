//! Uniform sampled skeletal-animation clip.
//!
//! Donor: `oa/animation/animClip.h`.

use crate::{Result, core::Transform};

/// Wrap behavior applied when sampling outside the clip's frame range.
///
/// Donor: `oa::AnimWrapMode`. The Rust enum cannot hold invalid values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnimWrapMode {
	/// Clamp to the first or last frame.
	#[default]
	Clamp,
	/// Wrap cyclically across the clip period.
	Loop,
}

/// Uniform sampled skeletal-animation clip.
///
/// Stores `frame_count * joint_count` validated local [`Transform`] samples in
/// row-major frame order. Construction is validated through
/// [`super::create_clip`].
///
/// Donor: `oa::AnimClip`.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimClip {
	pub(crate) skeleton_id: u32,
	pub(crate) joint_count: u32,
	pub(crate) frame_count: u32,
	pub(crate) frames_per_second: f32,
	pub(crate) samples: Vec<Transform>,
}

impl AnimClip {
	/// Skeleton identity carried through sampling.
	pub fn skeleton_id(&self) -> u32 {
		self.skeleton_id
	}

	/// Joints per frame.
	pub fn joint_count(&self) -> u32 {
		self.joint_count
	}

	/// Complete frames stored.
	pub fn frame_count(&self) -> u32 {
		self.frame_count
	}

	/// Uniform sampling rate.
	pub fn frames_per_second(&self) -> f32 {
		self.frames_per_second
	}

	/// Clip length in seconds: `(frame_count - 1) / fps`, or zero for a
	/// single-frame clip.
	pub fn duration_seconds(&self) -> f64 {
		if self.frame_count <= 1 || self.frames_per_second <= 0.0 {
			return 0.0;
		}
		f64::from(self.frame_count - 1) / f64::from(self.frames_per_second)
	}

	/// All stored samples in frame-major order.
	pub fn samples(&self) -> &[Transform] {
		&self.samples
	}

	/// Sample the clip at `time_seconds` under `wrap_mode`.
	///
	/// Convenience delegate of [`super::sample_clip`].
	///
	/// # Errors
	///
	/// Returns an error when `time_seconds` is not finite.
	pub fn sample(&self, time_seconds: f64, wrap_mode: AnimWrapMode) -> Result<super::SkelPose> {
		super::sample_clip(self, time_seconds, wrap_mode)
	}
}
