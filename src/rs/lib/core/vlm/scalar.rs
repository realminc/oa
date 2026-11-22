use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

mod sealed {
	pub trait Sealed {}

	impl Sealed for f32 {}
	impl Sealed for f64 {}
}

/// A floating-point scalar admitted by VLM.
///
/// This sealed trait limits packed VLM values to `f32` and `f64` while allowing
/// their shared implementation to stay generic and dependency-free.
pub trait Float:
	sealed::Sealed
	+ Copy
	+ Debug
	+ Default
	+ PartialEq
	+ PartialOrd
	+ Add<Output = Self>
	+ Sub<Output = Self>
	+ Mul<Output = Self>
	+ Div<Output = Self>
	+ Neg<Output = Self>
{
	const ZERO: Self;
	const ONE: Self;
	const TWO: Self;
	const HALF: Self;
	const PI: Self;
	const EPSILON: Self;
	const TOLERANCE: Self;
	const INVERSE_TOLERANCE: Self;

	fn from_f64(value: f64) -> Self;
	fn abs(self) -> Self;
	fn sqrt(self) -> Self;
	fn sin(self) -> Self;
	fn cos(self) -> Self;
	fn tan(self) -> Self;
	fn asin(self) -> Self;
	fn acos(self) -> Self;
	fn atan2(self, other: Self) -> Self;
	fn is_finite(self) -> bool;
	fn max(self, other: Self) -> Self;
	fn min(self, other: Self) -> Self;
}

macro_rules! impl_float {
	($scalar:ty, $pi:expr, $tolerance:expr) => {
		impl Float for $scalar {
			const ZERO: Self = 0.0;
			const ONE: Self = 1.0;
			const TWO: Self = 2.0;
			const HALF: Self = 0.5;
			const PI: Self = $pi;
			const EPSILON: Self = <$scalar>::EPSILON;
			const TOLERANCE: Self = $tolerance;
			const INVERSE_TOLERANCE: Self = <$scalar>::EPSILON * 32.0;

			fn from_f64(value: f64) -> Self {
				value as Self
			}

			fn abs(self) -> Self {
				self.abs()
			}

			fn sqrt(self) -> Self {
				self.sqrt()
			}

			fn sin(self) -> Self {
				self.sin()
			}

			fn cos(self) -> Self {
				self.cos()
			}

			fn tan(self) -> Self {
				self.tan()
			}

			fn asin(self) -> Self {
				self.asin()
			}

			fn acos(self) -> Self {
				self.acos()
			}

			fn atan2(self, other: Self) -> Self {
				self.atan2(other)
			}

			fn is_finite(self) -> bool {
				self.is_finite()
			}

			fn max(self, other: Self) -> Self {
				self.max(other)
			}

			fn min(self, other: Self) -> Self {
				self.min(other)
			}
		}
	};
}

impl_float!(f32, std::f32::consts::PI, 1.0e-5);
impl_float!(f64, std::f64::consts::PI, 1.0e-12);

pub fn radians<T: Float>(degrees: T) -> T {
	degrees * T::PI / T::from_f64(180.0)
}

pub fn degrees<T: Float>(radians: T) -> T {
	radians * T::from_f64(180.0) / T::PI
}

pub(crate) fn valid_tolerance<T: Float>(tolerance: T) -> bool {
	tolerance.is_finite() && tolerance >= T::ZERO
}

pub(crate) fn approx<T: Float>(a: T, b: T, absolute: T, relative: T) -> bool {
	if !valid_tolerance(absolute) || !valid_tolerance(relative) {
		return false;
	}
	let difference = (a - b).abs();
	let scale = a.abs().max(b.abs());
	difference <= absolute || difference <= relative * scale
}
