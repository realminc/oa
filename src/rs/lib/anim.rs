//! Host-side skeletal animation values and operations.
//!
//! Sparse keyframe curves, uniform sampled clips, validated skeleton poses,
//! and Maya-semantic joints. This domain is host-only: it composes
//! [`crate::core::Transform`] and [`crate::vlm`] math without depending on
//! the Vulkan runtime.
//!
//! Donor: `oa/animation/*` (`oacpp/source/cpp/include/oa/animation/` and
//! `oacpp/source/cpp/lib/oa/animation/`).
//!
//! | C++ | Rust |
//! | --- | --- |
//! | `oa::Duration` (core) | [`crate::core::Duration`] |
//! | `oa::AnimInterpolation` | [`AnimInterpolation`] |
//! | `oa::AnimWrapMode` | [`AnimWrapMode`] |
//! | `oa::Keyframe<T>` | [`Keyframe<T>`] |
//! | `oa::AnimCurve<T>` | [`AnimCurve<T>`] |
//! | `oa::AnimClip` | [`AnimClip`] |
//! | `oa::SkelPose` | [`SkelPose`] |
//! | `oa::Joint` | [`Joint`] |
//! | `oa::JointDegreesOfFreedom` | [`JointDegreesOfFreedom`] |
//! | `oa::JointRotationLimits` | [`JointRotationLimits`] |
//! | `oa::FnAnim::*` | free functions in this module |
//! | `oa::FnJoint::*` | [`Joint`] methods |
//!
//! The C++ overloaded `FnAnim` operation names are spelled out per value
//! type (`create_clip`, `sample_clip`, `create_curve`, `sample_curve`, …)
//! because Rust does not overload on argument type.

mod clip;
mod curve;
mod joint;
mod keyframe;
mod pose;

pub use clip::{AnimClip, AnimWrapMode};
pub use curve::AnimCurve;
pub use joint::{Joint, JointDegreesOfFreedom, JointRotationLimits};
pub use keyframe::{AnimCurveValue, AnimInterpolation, Keyframe};
pub use pose::SkelPose;

use crate::{
	Error, Result,
	core::{Duration, Transform},
};

fn checked_sample_count(frame_count: u32, joint_count: u32) -> Option<usize> {
	if frame_count == 0 || joint_count == 0 {
		return None;
	}
	(frame_count as usize).checked_mul(joint_count as usize)
}

fn validate_transforms(transforms: &[Transform]) -> Result<()> {
	if transforms.is_empty() {
		return Err(Error::invalid_argument(
			"Animation transform array is empty",
		));
	}
	for transform in transforms {
		transform.validate()?;
	}
	Ok(())
}

fn validate_clip_metadata(clip: &AnimClip) -> Result<()> {
	if clip.joint_count == 0
		|| clip.frame_count == 0
		|| !clip.frames_per_second.is_finite()
		|| clip.frames_per_second <= 0.0
	{
		return Err(Error::invalid_argument("AnimClip metadata is invalid"));
	}
	let expected = checked_sample_count(clip.frame_count, clip.joint_count)
		.ok_or_else(|| Error::invalid_argument("AnimClip sample count is invalid"))?;
	if expected != clip.samples.len() {
		return Err(Error::invalid_argument("AnimClip sample count is invalid"));
	}
	Ok(())
}

fn interpolate_transform(a: &Transform, b: &Transform, weight: f32) -> Transform {
	let mut result = Transform::default();
	result.set_position(a.position().lerp(b.position(), weight));
	result.set_rotation(
		a.rotation()
			.try_slerp(b.rotation(), weight)
			.unwrap_or_else(|| a.rotation()),
	);
	result.set_scale(a.scale().lerp(b.scale(), weight));
	result.set_shear(a.shear().lerp(b.shear(), weight));
	result
}

fn curve_span_is_representable(first: i64, last: i64) -> bool {
	first >= 0 || last <= i64::MAX + first
}

fn validate_curve_keyframes<T: AnimCurveValue>(keyframes: &[Keyframe<T>]) -> Result<()> {
	if keyframes.is_empty() {
		return Err(Error::invalid_argument(
			"AnimCurve requires at least one keyframe",
		));
	}
	if keyframes.len() > u32::MAX as usize {
		return Err(Error::out_of_range(
			"AnimCurve keyframe count exceeds the supported range",
		));
	}
	for (index, keyframe) in keyframes.iter().enumerate() {
		if !keyframe.value.curve_is_valid()
			|| !keyframe.in_tangent.curve_is_finite()
			|| !keyframe.out_tangent.curve_is_finite()
		{
			return Err(Error::invalid_argument(
				"AnimCurve values and tangents must be finite; quaternion values must be non-zero",
			));
		}
		if index > 0 && keyframe.time <= keyframes[index - 1].time {
			return Err(Error::invalid_argument(
				"AnimCurve keyframe times must be strictly increasing",
			));
		}
	}
	if !curve_span_is_representable(
		keyframes[0].time.nanoseconds(),
		keyframes[keyframes.len() - 1].time.nanoseconds(),
	) {
		return Err(Error::out_of_range(
			"AnimCurve time span exceeds the supported duration range",
		));
	}
	Ok(())
}

fn validate_curve_metadata<T: AnimCurveValue>(curve: &AnimCurve<T>) -> Result<()> {
	let keyframes = &curve.keyframes;
	if keyframes.is_empty() || keyframes.len() > u32::MAX as usize {
		return Err(Error::invalid_argument("AnimCurve metadata is invalid"));
	}
	if !curve_span_is_representable(
		keyframes[0].time.nanoseconds(),
		keyframes[keyframes.len() - 1].time.nanoseconds(),
	) {
		return Err(Error::out_of_range(
			"AnimCurve time span exceeds the supported duration range",
		));
	}
	Ok(())
}

fn interpolate_curve_cubic<T: AnimCurveValue>(
	a: &Keyframe<T>,
	b: &Keyframe<T>,
	weight: f32,
	segment_seconds: f32,
) -> Result<T> {
	let squared = weight * weight;
	let cubed = squared * weight;
	let value_a_weight = 2.0 * cubed - 3.0 * squared + 1.0;
	let tangent_a_weight = cubed - 2.0 * squared + weight;
	let value_b_weight = -2.0 * cubed + 3.0 * squared;
	let tangent_b_weight = cubed - squared;
	let value = a.value * value_a_weight
		+ a.out_tangent * (segment_seconds * tangent_a_weight)
		+ b.value * value_b_weight
		+ b.in_tangent * (segment_seconds * tangent_b_weight);
	if !value.curve_is_valid() {
		return Err(Error::out_of_range(
			"AnimCurve cubic interpolation produced a non-finite or zero quaternion value",
		));
	}
	Ok(value.curve_normalize())
}

fn wrap_curve_time(time: i64, first: i64, last: i64) -> i64 {
	let span = (last - first) as u64;
	if time >= first {
		let distance = time.wrapping_sub(first) as u64;
		return first + (distance % span) as i64;
	}
	let distance = first.wrapping_sub(time) as u64;
	let remainder = distance % span;
	let offset = if remainder == 0 { 0 } else { span - remainder };
	first + offset as i64
}

fn sample_curve_validated<T: AnimCurveValue>(
	curve: &AnimCurve<T>,
	time: Duration,
	wrap_mode: AnimWrapMode,
) -> Result<T> {
	let keyframes = &curve.keyframes;
	if keyframes.len() == 1 {
		return Ok(keyframes[0].value);
	}
	let first = keyframes[0].time.nanoseconds();
	let last = keyframes[keyframes.len() - 1].time.nanoseconds();
	let mut time = time.nanoseconds();
	match wrap_mode {
		AnimWrapMode::Loop => {
			time = wrap_curve_time(time, first, last);
		}
		AnimWrapMode::Clamp => {
			if time <= first {
				return Ok(keyframes[0].value);
			}
			if time >= last {
				return Ok(keyframes[keyframes.len() - 1].value);
			}
		}
	}

	let mut lower = 1usize;
	let mut upper = keyframes.len();
	while lower < upper {
		let middle = lower + (upper - lower) / 2;
		if keyframes[middle].time.nanoseconds() <= time {
			lower = middle + 1;
		} else {
			upper = middle;
		}
	}
	let index_b = lower;
	let index_a = index_b - 1;
	let key_a = &keyframes[index_a];
	let key_b = &keyframes[index_b];
	if curve.interpolation == AnimInterpolation::Step {
		return Ok(key_a.value);
	}
	let segment_nanoseconds = key_b.time.nanoseconds() - key_a.time.nanoseconds();
	let weight = ((time - key_a.time.nanoseconds()) as f64 / segment_nanoseconds as f64) as f32;
	if curve.interpolation == AnimInterpolation::Linear {
		let value = key_a.value.curve_lerp(key_b.value, weight);
		if !value.curve_is_valid() {
			return Err(Error::out_of_range(
				"AnimCurve linear interpolation produced an invalid value",
			));
		}
		return Ok(value.curve_normalize());
	}
	interpolate_curve_cubic(
		key_a,
		key_b,
		weight,
		segment_nanoseconds as f32 / 1_000_000_000.0,
	)
}

fn checked_bake_time(start_nanoseconds: i64, frame: u32, frames_per_second: f32) -> Option<i64> {
	let seconds = f64::from(frame) / f64::from(frames_per_second);
	let nanoseconds = seconds * 1.0e9;
	if !nanoseconds.is_finite() || nanoseconds < 0.0 || nanoseconds >= 9.223372036854776e18 {
		return None;
	}
	let offset = nanoseconds as i64;
	if start_nanoseconds > 0 && offset > i64::MAX - start_nanoseconds {
		return None;
	}
	Some(start_nanoseconds + offset)
}

/// Create a validated [`SkelPose`] from local joint transforms.
///
/// # Errors
///
/// Returns an error when `local_transforms` is empty, contains a non-finite
/// transform, or exceeds the supported joint count.
///
/// Donor: `oa::FnAnim::createPose`.
pub fn create_pose(skeleton_id: u32, local_transforms: &[Transform]) -> Result<SkelPose> {
	validate_transforms(local_transforms)?;
	if local_transforms.len() > u32::MAX as usize {
		return Err(Error::out_of_range(
			"SkelPose joint count exceeds the supported range",
		));
	}
	Ok(SkelPose {
		skeleton_id,
		local_transforms: local_transforms.to_vec(),
	})
}

/// Re-validate a [`SkelPose`].
///
/// # Errors
///
/// Returns an error under the same contract as [`create_pose`].
///
/// Donor: `oa::FnAnim::validate(SkelPose)`.
pub fn validate_pose(pose: &SkelPose) -> Result<()> {
	if pose.local_transforms.len() > u32::MAX as usize {
		return Err(Error::out_of_range(
			"SkelPose joint count exceeds the supported range",
		));
	}
	validate_transforms(&pose.local_transforms)
}

/// Create a validated [`AnimClip`] from frame-major uniform samples.
///
/// `samples` must hold a whole number of `joint_count` frames.
///
/// # Errors
///
/// Returns an error for a zero joint count, non-finite or non-positive
/// `frames_per_second`, empty or partial-frame samples, a non-finite
/// transform, or a frame count beyond the supported range.
///
/// Donor: `oa::FnAnim::createClip`.
pub fn create_clip(
	skeleton_id: u32,
	joint_count: u32,
	frames_per_second: f32,
	samples: &[Transform],
) -> Result<AnimClip> {
	if joint_count == 0
		|| !frames_per_second.is_finite()
		|| frames_per_second <= 0.0
		|| samples.is_empty()
		|| !samples.len().is_multiple_of(joint_count as usize)
	{
		return Err(Error::invalid_argument(
			"AnimClip requires positive joint/frame counts, finite positive FPS, and complete frames",
		));
	}
	let frame_count = samples.len() / joint_count as usize;
	if frame_count > u32::MAX as usize {
		return Err(Error::out_of_range(
			"AnimClip frame count exceeds the supported range",
		));
	}
	validate_transforms(samples)?;
	Ok(AnimClip {
		skeleton_id,
		joint_count,
		frame_count: frame_count as u32,
		frames_per_second,
		samples: samples.to_vec(),
	})
}

/// Re-validate an [`AnimClip`].
///
/// # Errors
///
/// Returns an error under the same contract as [`create_clip`].
///
/// Donor: `oa::FnAnim::validate(AnimClip)`.
pub fn validate_clip(clip: &AnimClip) -> Result<()> {
	validate_clip_metadata(clip)?;
	validate_transforms(&clip.samples)
}

/// Extract the pose stored at `frame`.
///
/// # Errors
///
/// Returns an error when the clip is invalid or `frame` is out of range.
///
/// Donor: `oa::FnAnim::getFrame`.
pub fn frame(clip: &AnimClip, frame: u32) -> Result<SkelPose> {
	validate_clip_metadata(clip)?;
	if frame >= clip.frame_count {
		return Err(Error::out_of_range("AnimClip frame is out of range"));
	}
	let offset = frame as usize * clip.joint_count as usize;
	create_pose(
		clip.skeleton_id,
		&clip.samples[offset..offset + clip.joint_count as usize],
	)
}

/// Sample a clip at `time_seconds`, interpolating translation (lerp),
/// rotation (slerp), scale, and shear per joint.
///
/// # Errors
///
/// Returns an error when the clip is invalid or `time_seconds` is not finite.
///
/// Donor: `oa::FnAnim::sample(AnimClip)`.
pub fn sample_clip(
	clip: &AnimClip,
	time_seconds: f64,
	wrap_mode: AnimWrapMode,
) -> Result<SkelPose> {
	validate_clip_metadata(clip)?;
	if !time_seconds.is_finite() {
		return Err(Error::invalid_argument(
			"AnimClip sample time must be finite",
		));
	}
	if clip.frame_count == 1 {
		return frame(clip, 0);
	}
	let fps = f64::from(clip.frames_per_second);
	let (frame_a, frame_b, frame_position) = match wrap_mode {
		AnimWrapMode::Loop => {
			let period = f64::from(clip.frame_count) / fps;
			let mut wrapped = time_seconds % period;
			if wrapped < 0.0 {
				wrapped += period;
			}
			let frame_position = wrapped * fps;
			let mut a = frame_position.floor() as u32;
			if a >= clip.frame_count {
				a = clip.frame_count - 1;
			}
			(a, (a + 1) % clip.frame_count, frame_position)
		}
		AnimWrapMode::Clamp => {
			let maximum_frame = f64::from(clip.frame_count - 1);
			let duration = maximum_frame / fps;
			let frame_position = if time_seconds <= 0.0 {
				0.0
			} else if time_seconds >= duration {
				maximum_frame
			} else {
				time_seconds * fps
			};
			let a = frame_position.floor() as u32;
			(a, (a + 1).min(clip.frame_count - 1), frame_position)
		}
	};
	let weight = (frame_position - frame_position.floor()) as f32;
	let offset_a = frame_a as usize * clip.joint_count as usize;
	let offset_b = frame_b as usize * clip.joint_count as usize;
	let mut local_transforms = Vec::with_capacity(clip.joint_count as usize);
	for joint in 0..clip.joint_count as usize {
		local_transforms.push(interpolate_transform(
			&clip.samples[offset_a + joint],
			&clip.samples[offset_b + joint],
			weight,
		));
	}
	Ok(SkelPose {
		skeleton_id: clip.skeleton_id,
		local_transforms,
	})
}

/// Create a validated [`AnimCurve`] from strictly increasing keyframes.
///
/// Quaternion curves normalize every keyframe value on construction.
///
/// # Errors
///
/// Returns an error for empty or non-increasing keyframe times, non-finite
/// or invalid values/tangents, an excessive count, or a time span that
/// cannot be represented.
///
/// Donor: `oa::FnAnim::createCurve`.
pub fn create_curve<T: AnimCurveValue>(
	interpolation: AnimInterpolation,
	keyframes: &[Keyframe<T>],
) -> Result<AnimCurve<T>> {
	validate_curve_keyframes(keyframes)?;
	let mut stored = keyframes.to_vec();
	// Quaternions are re-normalized on construction (donor behavior); the
	// trait keeps this generic over the three curve value types.
	for keyframe in &mut stored {
		keyframe.value = keyframe.value.curve_normalize();
	}
	Ok(AnimCurve {
		interpolation,
		keyframes: stored,
	})
}

/// Re-validate an [`AnimCurve`] and every stored keyframe.
///
/// # Errors
///
/// Returns an error under the same contract as [`create_curve`].
///
/// Donor: `oa::FnAnim::validate(AnimCurve)`.
pub fn validate_curve<T: AnimCurveValue>(curve: &AnimCurve<T>) -> Result<()> {
	validate_curve_keyframes(&curve.keyframes)
}

/// Sample a curve at `time` under `wrap_mode`.
///
/// # Errors
///
/// Returns an error when interpolation produces a non-finite or
/// non-normalizable value.
///
/// Donor: `oa::FnAnim::sample(AnimCurve)`.
pub fn sample_curve<T: AnimCurveValue>(
	curve: &AnimCurve<T>,
	time: Duration,
	wrap_mode: AnimWrapMode,
) -> Result<T> {
	validate_curve_metadata(curve)?;
	sample_curve_validated(curve, time, wrap_mode)
}

/// Sample a curve uniformly into `frame_count` values beginning at
/// `start_time`.
///
/// # Errors
///
/// Returns an error for non-finite or non-positive `frames_per_second`, a
/// zero `frame_count`, a sample time beyond the representable range, or a
/// sampling failure.
///
/// Donor: `oa::FnAnim::bake`.
pub fn bake<T: AnimCurveValue>(
	curve: &AnimCurve<T>,
	start_time: Duration,
	frames_per_second: f32,
	frame_count: u32,
	wrap_mode: AnimWrapMode,
) -> Result<Vec<T>> {
	validate_curve_metadata(curve)?;
	if !frames_per_second.is_finite() || frames_per_second <= 0.0 || frame_count == 0 {
		return Err(Error::invalid_argument(
			"AnimCurve baking requires finite positive FPS and frame count",
		));
	}
	let mut values = Vec::with_capacity(frame_count as usize);
	for frame in 0..frame_count {
		let nanoseconds = checked_bake_time(start_time.nanoseconds(), frame, frames_per_second)
			.ok_or_else(|| {
				Error::out_of_range("AnimCurve bake time exceeds the supported duration range")
			})?;
		values.push(sample_curve_validated(
			curve,
			Duration::from_nanoseconds(nanoseconds),
			wrap_mode,
		)?);
	}
	Ok(values)
}
