use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::scalar::{Float, approx, degrees, radians, valid_tolerance};
use super::vector::Vec3;

/// Explicit Euler-axis application order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationOrder {
	Xyz,
	Xzy,
	Yxz,
	Yzx,
	Zxy,
	Zyx,
}

/// A packed vector-first quaternion `(x, y, z, w)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat<T: Float = f32> {
	pub x: T,
	pub y: T,
	pub z: T,
	pub w: T,
}

pub type DQuat = Quat<f64>;

impl<T: Float> Default for Quat<T> {
	fn default() -> Self {
		Self::identity()
	}
}

impl<T: Float> Quat<T> {
	pub fn identity() -> Self {
		Self {
			x: T::ZERO,
			y: T::ZERO,
			z: T::ZERO,
			w: T::ONE,
		}
	}

	/// Constructs an axis-angle rotation, returning `None` for invalid input.
	pub fn try_from_axis_angle(axis: Vec3<T>, angle_radians: T) -> Option<Self> {
		if !angle_radians.is_finite() {
			return None;
		}
		let axis = axis.try_normalized()?;
		let half_angle = angle_radians * T::HALF;
		let sine = half_angle.sin();
		let result = Self {
			x: axis.x * sine,
			y: axis.y * sine,
			z: axis.z * sine,
			w: half_angle.cos(),
		};
		result.is_finite().then_some(result)
	}

	pub fn is_finite(self) -> bool {
		self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
	}

	pub fn norm_squared(self) -> T {
		(self.x * self.x + self.y * self.y) + (self.z * self.z + self.w * self.w)
	}

	pub fn norm(self) -> T {
		let squared = self.norm_squared();
		if squared.is_finite() && squared > T::ZERO {
			return squared.sqrt();
		}
		if !self.is_finite() {
			return squared.sqrt();
		}
		let magnitude = self
			.x
			.abs()
			.max(self.y.abs())
			.max(self.z.abs())
			.max(self.w.abs());
		if magnitude == T::ZERO {
			return T::ZERO;
		}
		let scaled = self / magnitude;
		magnitude * scaled.norm_squared().sqrt()
	}

	/// Returns a unit quaternion, or `None` for a zero or non-finite value.
	pub fn try_normalized(self) -> Option<Self> {
		let squared = self.norm_squared();
		if squared.is_finite() && squared > T::ZERO {
			return Some(self / squared.sqrt());
		}
		if !self.is_finite() {
			return None;
		}
		let magnitude = self
			.x
			.abs()
			.max(self.y.abs())
			.max(self.z.abs())
			.max(self.w.abs());
		if magnitude <= T::ZERO {
			return None;
		}
		let scaled = self / magnitude;
		let norm = scaled.norm();
		if !norm.is_finite() || norm <= T::ZERO {
			return None;
		}
		let result = scaled / norm;
		result.is_finite().then_some(result)
	}

	pub fn conjugate(self) -> Self {
		Self {
			x: -self.x,
			y: -self.y,
			z: -self.z,
			w: self.w,
		}
	}

	pub fn dot(self, other: Self) -> T {
		self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
	}

	pub fn try_inverse(self, tolerance: T) -> Option<Self> {
		if !valid_tolerance(tolerance) || !self.is_finite() {
			return None;
		}
		let magnitude = self
			.x
			.abs()
			.max(self.y.abs())
			.max(self.z.abs())
			.max(self.w.abs());
		if magnitude <= tolerance {
			return None;
		}
		let scaled = self / magnitude;
		let norm_squared = scaled.norm_squared();
		if !norm_squared.is_finite() || norm_squared <= T::ZERO {
			return None;
		}
		let result = (scaled.conjugate() / norm_squared) / magnitude;
		result.is_finite().then_some(result)
	}

	#[inline]
	pub fn try_rotate(self, vector: Vec3<T>) -> Option<Vec3<T>> {
		let squared = self.norm_squared();
		let unit_tolerance = T::EPSILON * T::from_f64(8.0);
		let unit = if squared.is_finite() && (squared - T::ONE).abs() <= unit_tolerance {
			self
		} else {
			self.try_normalized()?
		};
		let imaginary = Vec3 {
			x: unit.x,
			y: unit.y,
			z: unit.z,
		};
		let twice_cross = imaginary.cross(vector) * T::TWO;
		Some(vector + twice_cross * unit.w + imaginary.cross(twice_cross))
	}

	pub fn try_inverse_rotate(self, vector: Vec3<T>) -> Option<Vec3<T>> {
		self.conjugate().try_rotate(vector)
	}

	/// Creates a rotation from finite Euler angles in radians.
	pub fn try_from_euler_radians(angles: Vec3<T>, order: RotationOrder) -> Option<Self> {
		if !angles.is_finite() {
			return None;
		}
		let x = Self::try_from_axis_angle(
			Vec3 {
				x: T::ONE,
				y: T::ZERO,
				z: T::ZERO,
			},
			angles.x,
		)?;
		let y = Self::try_from_axis_angle(
			Vec3 {
				x: T::ZERO,
				y: T::ONE,
				z: T::ZERO,
			},
			angles.y,
		)?;
		let z = Self::try_from_axis_angle(
			Vec3 {
				x: T::ZERO,
				y: T::ZERO,
				z: T::ONE,
			},
			angles.z,
		)?;
		Some(match order {
			RotationOrder::Xyz => z * (y * x),
			RotationOrder::Xzy => y * (z * x),
			RotationOrder::Yxz => z * (x * y),
			RotationOrder::Yzx => x * (z * y),
			RotationOrder::Zxy => y * (x * z),
			RotationOrder::Zyx => x * (y * z),
		})
	}

	pub fn try_from_euler_degrees(angles: Vec3<T>, order: RotationOrder) -> Option<Self> {
		Self::try_from_euler_radians(
			Vec3 {
				x: radians(angles.x),
				y: radians(angles.y),
				z: radians(angles.z),
			},
			order,
		)
	}

	pub fn try_to_euler_radians(self, order: RotationOrder) -> Option<Vec3<T>> {
		let q = self.try_normalized()?;
		let xx = q.x * q.x;
		let yy = q.y * q.y;
		let zz = q.z * q.z;
		let xy = q.x * q.y;
		let xz = q.x * q.z;
		let yz = q.y * q.z;
		let xw = q.x * q.w;
		let yw = q.y * q.w;
		let zw = q.z * q.w;
		let m11 = T::ONE - T::TWO * (yy + zz);
		let m12 = T::TWO * (xy - zw);
		let m13 = T::TWO * (xz + yw);
		let m21 = T::TWO * (xy + zw);
		let m22 = T::ONE - T::TWO * (xx + zz);
		let m23 = T::TWO * (yz - xw);
		let m31 = T::TWO * (xz - yw);
		let m32 = T::TWO * (yz + xw);
		let m33 = T::ONE - T::TWO * (xx + yy);
		let lock = T::ONE - T::EPSILON * T::from_f64(64.0);
		let mut result = Vec3::default();
		match order {
			RotationOrder::Zyx => {
				result.y = m13.max(-T::ONE).min(T::ONE).asin();
				if m13.abs() < lock {
					result.x = (-m23).atan2(m33);
					result.z = (-m12).atan2(m11);
				} else {
					result.x = m32.atan2(m22);
				}
			}
			RotationOrder::Zxy => {
				result.x = -m23.max(-T::ONE).min(T::ONE).asin();
				if m23.abs() < lock {
					result.y = m13.atan2(m33);
					result.z = m21.atan2(m22);
				} else {
					result.y = (-m31).atan2(m11);
				}
			}
			RotationOrder::Yxz => {
				result.x = m32.max(-T::ONE).min(T::ONE).asin();
				if m32.abs() < lock {
					result.y = (-m31).atan2(m33);
					result.z = (-m12).atan2(m22);
				} else {
					result.z = m21.atan2(m11);
				}
			}
			RotationOrder::Xyz => {
				result.y = -m31.max(-T::ONE).min(T::ONE).asin();
				if m31.abs() < lock {
					result.x = m32.atan2(m33);
					result.z = m21.atan2(m11);
				} else {
					result.z = (-m12).atan2(m22);
				}
			}
			RotationOrder::Xzy => {
				result.z = m21.max(-T::ONE).min(T::ONE).asin();
				if m21.abs() < lock {
					result.x = (-m23).atan2(m22);
					result.y = (-m31).atan2(m11);
				} else {
					result.y = m13.atan2(m33);
				}
			}
			RotationOrder::Yzx => {
				result.z = -m12.max(-T::ONE).min(T::ONE).asin();
				if m12.abs() < lock {
					result.x = m32.atan2(m22);
					result.y = m13.atan2(m11);
				} else {
					result.x = (-m23).atan2(m33);
				}
			}
		}
		Some(result)
	}

	pub fn try_to_euler_degrees(self, order: RotationOrder) -> Option<Vec3<T>> {
		let value = self.try_to_euler_radians(order)?;
		Some(Vec3 {
			x: degrees(value.x),
			y: degrees(value.y),
			z: degrees(value.z),
		})
	}

	pub fn try_nlerp(self, other: Self, amount: T) -> Option<Self> {
		let a = self.try_normalized()?;
		let mut b = other.try_normalized()?;
		if a.dot(b) < T::ZERO {
			b = -b;
		}
		(a + (b - a) * amount).try_normalized()
	}

	pub fn try_slerp(self, other: Self, amount: T) -> Option<Self> {
		let a = self.try_normalized()?;
		let mut b = other.try_normalized()?;
		let mut cosine = a.dot(b);
		if cosine < T::ZERO {
			b = -b;
			cosine = -cosine;
		}
		cosine = cosine.max(-T::ONE).min(T::ONE);
		if cosine > T::ONE - T::TOLERANCE {
			return a.try_nlerp(b, amount);
		}
		let angle = cosine.acos();
		let sine = angle.sin();
		let weight_a = ((T::ONE - amount) * angle).sin() / sine;
		let weight_b = (amount * angle).sin() / sine;
		(a * weight_a + b * weight_b).try_normalized()
	}

	pub fn angle(self) -> Option<T> {
		let q = self.try_normalized()?;
		Some(T::TWO * q.w.abs().max(-T::ONE).min(T::ONE).acos())
	}

	pub fn axis(self) -> Option<Vec3<T>> {
		let mut q = self.try_normalized()?;
		if q.w < T::ZERO {
			q = -q;
		}
		let sine_half = (T::ONE - q.w * q.w).max(T::ZERO).sqrt();
		if sine_half <= T::TOLERANCE {
			return Some(Vec3 {
				x: T::ONE,
				y: T::ZERO,
				z: T::ZERO,
			});
		}
		Some(Vec3 {
			x: q.x / sine_half,
			y: q.y / sine_half,
			z: q.z / sine_half,
		})
	}

	pub fn approximately_equal(self, other: Self, absolute: T, relative: T) -> bool {
		approx(self.x, other.x, absolute, relative)
			&& approx(self.y, other.y, absolute, relative)
			&& approx(self.z, other.z, absolute, relative)
			&& approx(self.w, other.w, absolute, relative)
	}
}

impl<T: Float> Add for Quat<T> {
	type Output = Self;

	fn add(self, other: Self) -> Self {
		Self {
			x: self.x + other.x,
			y: self.y + other.y,
			z: self.z + other.z,
			w: self.w + other.w,
		}
	}
}

impl<T: Float> Sub for Quat<T> {
	type Output = Self;

	fn sub(self, other: Self) -> Self {
		Self {
			x: self.x - other.x,
			y: self.y - other.y,
			z: self.z - other.z,
			w: self.w - other.w,
		}
	}
}

impl<T: Float> Mul<T> for Quat<T> {
	type Output = Self;

	fn mul(self, scalar: T) -> Self {
		Self {
			x: self.x * scalar,
			y: self.y * scalar,
			z: self.z * scalar,
			w: self.w * scalar,
		}
	}
}

impl<T: Float> Mul for Quat<T> {
	type Output = Self;

	fn mul(self, other: Self) -> Self {
		Self {
			x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
			y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
			z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
			w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
		}
	}
}

impl<T: Float> Div<T> for Quat<T> {
	type Output = Self;

	fn div(self, scalar: T) -> Self {
		Self {
			x: self.x / scalar,
			y: self.y / scalar,
			z: self.z / scalar,
			w: self.w / scalar,
		}
	}
}

impl<T: Float> Neg for Quat<T> {
	type Output = Self;

	fn neg(self) -> Self {
		Self {
			x: -self.x,
			y: -self.y,
			z: -self.z,
			w: -self.w,
		}
	}
}

impl<T: Float> AddAssign for Quat<T> {
	fn add_assign(&mut self, other: Self) {
		*self = *self + other;
	}
}

impl<T: Float> SubAssign for Quat<T> {
	fn sub_assign(&mut self, other: Self) {
		*self = *self - other;
	}
}

impl<T: Float> MulAssign<T> for Quat<T> {
	fn mul_assign(&mut self, scalar: T) {
		*self = *self * scalar;
	}
}

impl<T: Float> MulAssign for Quat<T> {
	fn mul_assign(&mut self, other: Self) {
		*self = *self * other;
	}
}

impl<T: Float> DivAssign<T> for Quat<T> {
	fn div_assign(&mut self, scalar: T) {
		*self = *self / scalar;
	}
}

impl Mul<Quat<f32>> for f32 {
	type Output = Quat<f32>;

	fn mul(self, quaternion: Quat<f32>) -> Self::Output {
		quaternion * self
	}
}

impl Mul<Quat<f64>> for f64 {
	type Output = Quat<f64>;

	fn mul(self, quaternion: Quat<f64>) -> Self::Output {
		quaternion * self
	}
}
