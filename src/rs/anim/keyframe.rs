//! Sparse animation-curve sample values and interpolation modes.
//!
//! Donor: `oa/animation/keyframe.h`.

use std::ops::{Add, Mul};

use crate::{
	core::Duration,
	vlm::{Quat, Vec3},
};

/// Interpolation mode applied between two adjacent keyframes.
///
/// Donor: `oa::AnimInterpolation`. The Rust enum cannot hold the donor's
/// castable invalid values, so no separate validity predicate exists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnimInterpolation {
	/// Hold the left keyframe value until the next key.
	Step,
	/// Component-wise linear interpolation; quaternions use `slerp`.
	#[default]
	Linear,
	/// Hermite spline using per-second in/out tangents.
	CubicSpline,
}

mod sealed {
	pub trait Sealed {}
	impl Sealed for f32 {}
	impl Sealed for crate::vlm::Vec3 {}
	impl Sealed for crate::vlm::Quat {}
}

/// Value type admissible in [`Keyframe`] and [`AnimCurve`](super::AnimCurve).
///
/// Sealed: implemented by `f32`, [`Vec3`], and [`Quat`]. The trait spells the
/// donor's `isAnimCurveValueV` bound plus the value behaviors curve sampling
/// requires; it is not a public extension point.
pub trait AnimCurveValue:
	Copy + Default + Add<Output = Self> + Mul<f32, Output = Self> + sealed::Sealed
{
	/// Zero value used for absent tangents (`{0,0,0,0}` for quaternions).
	fn curve_zero() -> Self;

	/// All components finite.
	fn curve_is_finite(self) -> bool;

	/// Finite and, for quaternions, non-zero in at least one component.
	fn curve_is_valid(self) -> bool;

	/// Normalize where the type admits it (quaternions); otherwise identity.
	fn curve_normalize(self) -> Self;

	/// Component interpolation (`slerp` for quaternions).
	fn curve_lerp(self, other: Self, weight: f32) -> Self;
}

impl AnimCurveValue for f32 {
	fn curve_zero() -> Self {
		0.0
	}
	fn curve_is_finite(self) -> bool {
		self.is_finite()
	}
	fn curve_is_valid(self) -> bool {
		self.is_finite()
	}
	fn curve_normalize(self) -> Self {
		self
	}
	fn curve_lerp(self, other: Self, weight: f32) -> Self {
		self * (1.0 - weight) + other * weight
	}
}

impl AnimCurveValue for Vec3 {
	fn curve_zero() -> Self {
		Self::default()
	}
	fn curve_is_finite(self) -> bool {
		self.is_finite()
	}
	fn curve_is_valid(self) -> bool {
		self.is_finite()
	}
	fn curve_normalize(self) -> Self {
		self
	}
	fn curve_lerp(self, other: Self, weight: f32) -> Self {
		self.lerp(other, weight)
	}
}

impl AnimCurveValue for Quat {
	fn curve_zero() -> Self {
		Self {
			x: 0.0,
			y: 0.0,
			z: 0.0,
			w: 0.0,
		}
	}
	fn curve_is_finite(self) -> bool {
		self.is_finite()
	}
	fn curve_is_valid(self) -> bool {
		self.is_finite()
			&& self
				.x
				.abs()
				.max(self.y.abs())
				.max(self.z.abs())
				.max(self.w.abs())
				> 0.0
	}
	fn curve_normalize(self) -> Self {
		self.try_normalized().unwrap_or(self)
	}
	fn curve_lerp(self, other: Self, weight: f32) -> Self {
		// Keyframe values are validated finite and non-zero before sampling,
		// so the checked path always succeeds here.
		self.try_slerp(other, weight).unwrap_or(self)
	}
}

/// One sparse animation-curve sample.
///
/// Tangents are expressed per second and default to the value type's zero.
/// For quaternion curves the zero tangent is the all-zero quaternion, while
/// `value` defaults to identity.
///
/// Donor: `oa::Keyframe<T>`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keyframe<T: AnimCurveValue> {
	pub time: Duration,
	pub value: T,
	pub in_tangent: T,
	pub out_tangent: T,
}

impl<T: AnimCurveValue> Default for Keyframe<T> {
	fn default() -> Self {
		Self {
			time: Duration::ZERO,
			value: T::default(),
			in_tangent: T::curve_zero(),
			out_tangent: T::curve_zero(),
		}
	}
}
