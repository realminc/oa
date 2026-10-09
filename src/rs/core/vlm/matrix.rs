use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::quaternion::Quat;
use super::scalar::{Float, approx, valid_tolerance};
use super::vector::{Vec3, Vec4};

/// A packed 3×3 row-major matrix.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat3<T: Float = f32> {
	pub m: [[T; 3]; 3],
}

/// A packed 4×4 row-major matrix.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat4<T: Float = f32> {
	pub m: [[T; 4]; 4],
}

pub type DMat3 = Mat3<f64>;
pub type DMat4 = Mat4<f64>;

impl<T: Float> Mat3<T> {
	pub fn identity() -> Self {
		Self {
			m: [
				[T::ONE, T::ZERO, T::ZERO],
				[T::ZERO, T::ONE, T::ZERO],
				[T::ZERO, T::ZERO, T::ONE],
			],
		}
	}

	pub fn is_finite(self) -> bool {
		self.m.iter().flatten().all(|value| value.is_finite())
	}

	pub fn transpose(self) -> Self {
		let mut result = Self::default();
		for (row, values) in result.m.iter_mut().enumerate() {
			for (column, value) in values.iter_mut().enumerate() {
				*value = self.m[column][row];
			}
		}
		result
	}

	pub fn determinant(self) -> T {
		self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1])
			- self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0])
			+ self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0])
	}

	pub fn try_inverse(self, tolerance: T) -> Option<Self> {
		if !valid_tolerance(tolerance) || !self.is_finite() {
			return None;
		}
		let magnitude = self
			.m
			.iter()
			.flatten()
			.fold(T::ZERO, |largest, value| largest.max(value.abs()));
		if magnitude <= T::ZERO {
			return None;
		}
		let scaled = self / magnitude;
		let determinant = scaled.determinant();
		if !determinant.is_finite() || determinant.abs() <= tolerance {
			return None;
		}
		let inverse_determinant = T::ONE / determinant;
		let m = scaled.m;
		let inverse = Self {
			m: [
				[
					(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inverse_determinant / magnitude,
					(m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inverse_determinant / magnitude,
					(m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inverse_determinant / magnitude,
				],
				[
					(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inverse_determinant / magnitude,
					(m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inverse_determinant / magnitude,
					(m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inverse_determinant / magnitude,
				],
				[
					(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inverse_determinant / magnitude,
					(m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inverse_determinant / magnitude,
					(m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inverse_determinant / magnitude,
				],
			],
		};
		if !inverse.is_finite() {
			return None;
		}
		let matrix_norm = row_norm3(self);
		let inverse_norm = row_norm3(inverse);
		let condition = matrix_norm * inverse_norm;
		if !condition.is_finite() || condition <= T::ZERO || T::ONE / condition <= tolerance {
			return None;
		}
		Some(inverse)
	}

	pub fn transform(self, value: Vec3<T>) -> Vec3<T> {
		value * self
	}

	pub fn approximately_equal(self, other: Self, absolute: T, relative: T) -> bool {
		valid_tolerance(absolute)
			&& valid_tolerance(relative)
			&& self.m.iter().enumerate().all(|(row, values)| {
				values
					.iter()
					.enumerate()
					.all(|(column, value)| approx(*value, other.m[row][column], absolute, relative))
			})
	}
}

fn row_norm3<T: Float>(matrix: Mat3<T>) -> T {
	matrix.m.iter().fold(T::ZERO, |largest, row| {
		largest.max(row.iter().fold(T::ZERO, |sum, value| sum + value.abs()))
	})
}

impl<T: Float> Mat4<T> {
	pub fn identity() -> Self {
		Self {
			m: [
				[T::ONE, T::ZERO, T::ZERO, T::ZERO],
				[T::ZERO, T::ONE, T::ZERO, T::ZERO],
				[T::ZERO, T::ZERO, T::ONE, T::ZERO],
				[T::ZERO, T::ZERO, T::ZERO, T::ONE],
			],
		}
	}

	pub fn is_finite(self) -> bool {
		self.m.iter().flatten().all(|value| value.is_finite())
	}

	pub fn transpose(self) -> Self {
		let mut result = Self::default();
		for (row, values) in result.m.iter_mut().enumerate() {
			for (column, value) in values.iter_mut().enumerate() {
				*value = self.m[column][row];
			}
		}
		result
	}

	pub fn determinant(self) -> T {
		if !self.is_finite() {
			return T::from_f64(f64::NAN);
		}
		let mut reduced = self.m;
		let mut result = T::ONE;
		let mut negative = false;
		for pivot_column in 0..4 {
			let mut pivot_row = pivot_column;
			let mut pivot_magnitude = reduced[pivot_row][pivot_column].abs();
			for (row, values) in reduced.iter().enumerate().skip(pivot_column + 1) {
				let magnitude = values[pivot_column].abs();
				if magnitude > pivot_magnitude {
					pivot_magnitude = magnitude;
					pivot_row = row;
				}
			}
			if pivot_magnitude == T::ZERO {
				return T::ZERO;
			}
			if pivot_row != pivot_column {
				reduced.swap(pivot_row, pivot_column);
				negative = !negative;
			}
			let pivot = reduced[pivot_column][pivot_column];
			result = result * pivot;
			let pivot_values = reduced[pivot_column];
			for row_values in reduced.iter_mut().skip(pivot_column + 1) {
				let factor = row_values[pivot_column] / pivot;
				for (column, value) in row_values.iter_mut().enumerate().skip(pivot_column + 1) {
					*value = *value - factor * pivot_values[column];
				}
			}
		}
		if negative { -result } else { result }
	}

	pub fn try_inverse(self, tolerance: T) -> Option<Self> {
		if !valid_tolerance(tolerance) || !self.is_finite() {
			return None;
		}
		if self.m[0][3] == T::ZERO
			&& self.m[1][3] == T::ZERO
			&& self.m[2][3] == T::ZERO
			&& self.m[3][3] == T::ONE
		{
			return self.try_affine_inverse(tolerance);
		}
		let mut augmented = [[T::ZERO; 8]; 4];
		let mut row_scale = [T::ZERO; 4];
		for row in 0..4 {
			for (column, value) in self.m[row].iter().copied().enumerate() {
				augmented[row][column] = value;
				row_scale[row] = row_scale[row].max(value.abs());
			}
			if row_scale[row] <= T::ZERO {
				return None;
			}
			augmented[row][row + 4] = T::ONE;
		}
		for pivot_column in 0..4 {
			let mut pivot_row = pivot_column;
			let mut pivot_ratio = augmented[pivot_row][pivot_column].abs() / row_scale[pivot_row];
			for row in (pivot_column + 1)..4 {
				let ratio = augmented[row][pivot_column].abs() / row_scale[row];
				if ratio > pivot_ratio {
					pivot_ratio = ratio;
					pivot_row = row;
				}
			}
			if !pivot_ratio.is_finite() || pivot_ratio <= tolerance {
				return None;
			}
			if pivot_row != pivot_column {
				augmented.swap(pivot_row, pivot_column);
				row_scale.swap(pivot_row, pivot_column);
			}
			let inverse_pivot = T::ONE / augmented[pivot_column][pivot_column];
			for value in &mut augmented[pivot_column] {
				*value = *value * inverse_pivot;
			}
			for row in 0..4 {
				if row == pivot_column {
					continue;
				}
				let factor = augmented[row][pivot_column];
				let pivot_values = augmented[pivot_column];
				for (column, value) in augmented[row].iter_mut().enumerate() {
					*value = *value - factor * pivot_values[column];
				}
			}
		}
		let mut inverse = Self::default();
		for (row, values) in inverse.m.iter_mut().enumerate() {
			for (column, value) in values.iter_mut().enumerate() {
				*value = augmented[row][column + 4];
			}
		}
		if !inverse.is_finite() {
			return None;
		}
		let residual = self * inverse;
		let factor = if T::TOLERANCE == T::from_f64(1.0e-5) {
			T::from_f64(256.0)
		} else {
			T::from_f64(65536.0)
		};
		let residual_tolerance = (T::TOLERANCE * factor).max(tolerance * factor);
		for row in 0..4 {
			for column in 0..4 {
				let expected = if row == column { T::ONE } else { T::ZERO };
				if (residual.m[row][column] - expected).abs() > residual_tolerance {
					return None;
				}
			}
		}
		Some(inverse)
	}

	pub fn linear_part(self) -> Mat3<T> {
		Mat3 {
			m: [
				[self.m[0][0], self.m[0][1], self.m[0][2]],
				[self.m[1][0], self.m[1][1], self.m[1][2]],
				[self.m[2][0], self.m[2][1], self.m[2][2]],
			],
		}
	}

	pub fn translation_part(self) -> Vec3<T> {
		Vec3 {
			x: self.m[3][0],
			y: self.m[3][1],
			z: self.m[3][2],
		}
	}

	pub fn from_affine(linear: Mat3<T>, translation: Vec3<T>) -> Self {
		Self {
			m: [
				[linear.m[0][0], linear.m[0][1], linear.m[0][2], T::ZERO],
				[linear.m[1][0], linear.m[1][1], linear.m[1][2], T::ZERO],
				[linear.m[2][0], linear.m[2][1], linear.m[2][2], T::ZERO],
				[translation.x, translation.y, translation.z, T::ONE],
			],
		}
	}

	pub fn is_affine(self, tolerance: T) -> bool {
		valid_tolerance(tolerance)
			&& self.is_finite()
			&& approx(self.m[0][3], T::ZERO, tolerance, T::ZERO)
			&& approx(self.m[1][3], T::ZERO, tolerance, T::ZERO)
			&& approx(self.m[2][3], T::ZERO, tolerance, T::ZERO)
			&& approx(self.m[3][3], T::ONE, tolerance, T::ZERO)
	}

	pub fn try_affine_inverse(self, tolerance: T) -> Option<Self> {
		if !self.is_affine(tolerance) {
			return None;
		}
		let inverse_linear = self.linear_part().try_inverse(tolerance)?;
		let inverse_translation = (-self.translation_part()) * inverse_linear;
		inverse_translation
			.is_finite()
			.then(|| Self::from_affine(inverse_linear, inverse_translation))
	}

	pub fn try_normal_matrix(self, tolerance: T) -> Option<Mat3<T>> {
		if !self.is_affine(tolerance) {
			return None;
		}
		Some(self.linear_part().try_inverse(tolerance)?.transpose())
	}

	pub fn transform(self, value: Vec4<T>) -> Vec4<T> {
		value * self
	}

	pub fn transform_point(self, point: Vec3<T>) -> Vec3<T> {
		let value = Vec4 {
			x: point.x,
			y: point.y,
			z: point.z,
			w: T::ONE,
		} * self;
		Vec3 {
			x: value.x,
			y: value.y,
			z: value.z,
		}
	}

	pub fn transform_direction(self, direction: Vec3<T>) -> Vec3<T> {
		let value = Vec4 {
			x: direction.x,
			y: direction.y,
			z: direction.z,
			w: T::ZERO,
		} * self;
		Vec3 {
			x: value.x,
			y: value.y,
			z: value.z,
		}
	}

	pub fn try_project_point(self, point: Vec3<T>, tolerance: T) -> Option<Vec3<T>> {
		if !valid_tolerance(tolerance) {
			return None;
		}
		let homogeneous = Vec4 {
			x: point.x,
			y: point.y,
			z: point.z,
			w: T::ONE,
		} * self;
		if !homogeneous.is_finite() || homogeneous.w.abs() <= tolerance {
			return None;
		}
		let projected = Vec3 {
			x: homogeneous.x / homogeneous.w,
			y: homogeneous.y / homogeneous.w,
			z: homogeneous.z / homogeneous.w,
		};
		projected.is_finite().then_some(projected)
	}

	pub fn try_transform_normal(self, normal: Vec3<T>, tolerance: T) -> Option<Vec3<T>> {
		let transformed = self
			.try_inverse(tolerance)?
			.transpose()
			.transform_direction(normal);
		if transformed.length() <= tolerance {
			return None;
		}
		transformed.try_normalized()
	}

	pub fn translation(value: Vec3<T>) -> Self {
		let mut result = Self::identity();
		result.m[3][0] = value.x;
		result.m[3][1] = value.y;
		result.m[3][2] = value.z;
		result
	}

	pub fn scaling(value: Vec3<T>) -> Self {
		let mut result = Self::identity();
		result.m[0][0] = value.x;
		result.m[1][1] = value.y;
		result.m[2][2] = value.z;
		result
	}

	pub fn try_from_quaternion(rotation: Quat<T>) -> Option<Self> {
		let q = rotation.try_normalized()?;
		let xx = q.x * q.x;
		let yy = q.y * q.y;
		let zz = q.z * q.z;
		let xy = q.x * q.y;
		let xz = q.x * q.z;
		let yz = q.y * q.z;
		let xw = q.x * q.w;
		let yw = q.y * q.w;
		let zw = q.z * q.w;
		let mut result = Self::identity();
		result.m[0][0] = T::ONE - T::TWO * (yy + zz);
		result.m[0][1] = T::TWO * (xy + zw);
		result.m[0][2] = T::TWO * (xz - yw);
		result.m[1][0] = T::TWO * (xy - zw);
		result.m[1][1] = T::ONE - T::TWO * (xx + zz);
		result.m[1][2] = T::TWO * (yz + xw);
		result.m[2][0] = T::TWO * (xz + yw);
		result.m[2][1] = T::TWO * (yz - xw);
		result.m[2][2] = T::ONE - T::TWO * (xx + yy);
		Some(result)
	}

	pub fn try_compose_trs(translation: Vec3<T>, rotation: Quat<T>, scale: Vec3<T>) -> Option<Self> {
		if !translation.is_finite() || !scale.is_finite() {
			return None;
		}
		let mut result = Self::try_from_quaternion(rotation)?;
		for column in 0..3 {
			result.m[0][column] = result.m[0][column] * scale.x;
			result.m[1][column] = result.m[1][column] * scale.y;
			result.m[2][column] = result.m[2][column] * scale.z;
		}
		result.m[3][0] = translation.x;
		result.m[3][1] = translation.y;
		result.m[3][2] = translation.z;
		Some(result)
	}

	pub fn approximately_equal(self, other: Self, absolute: T, relative: T) -> bool {
		valid_tolerance(absolute)
			&& valid_tolerance(relative)
			&& self.m.iter().enumerate().all(|(row, values)| {
				values
					.iter()
					.enumerate()
					.all(|(column, value)| approx(*value, other.m[row][column], absolute, relative))
			})
	}
}

macro_rules! matrix_ops {
	($matrix:ident, $size:expr $(, $product_inline:meta)?) => {
		impl<T: Float> Add for $matrix<T> {
			type Output = Self;

			fn add(self, other: Self) -> Self {
				let mut result = Self::default();
				for row in 0..$size {
					for column in 0..$size {
						result.m[row][column] = self.m[row][column] + other.m[row][column];
					}
				}
				result
			}
		}

		impl<T: Float> Sub for $matrix<T> {
			type Output = Self;

			fn sub(self, other: Self) -> Self {
				let mut result = Self::default();
				for row in 0..$size {
					for column in 0..$size {
						result.m[row][column] = self.m[row][column] - other.m[row][column];
					}
				}
				result
			}
		}

		impl<T: Float> Mul<T> for $matrix<T> {
			type Output = Self;

			fn mul(self, scalar: T) -> Self {
				let mut result = Self::default();
				for row in 0..$size {
					for column in 0..$size {
						result.m[row][column] = self.m[row][column] * scalar;
					}
				}
				result
			}
		}

		impl<T: Float> Div<T> for $matrix<T> {
			type Output = Self;

			fn div(self, scalar: T) -> Self {
				self * (T::ONE / scalar)
			}
		}

		impl<T: Float> Neg for $matrix<T> {
			type Output = Self;

			fn neg(self) -> Self {
				self * -T::ONE
			}
		}

		impl<T: Float> Mul for $matrix<T> {
			type Output = Self;

			$(#[$product_inline])?
			fn mul(self, other: Self) -> Self {
				let mut result = Self::default();
				for row in 0..$size {
					for column in 0..$size {
						for inner in 0..$size {
							result.m[row][column] =
								result.m[row][column] + self.m[row][inner] * other.m[inner][column];
						}
					}
				}
				result
			}
		}

		impl<T: Float> AddAssign for $matrix<T> {
			fn add_assign(&mut self, other: Self) {
				*self = *self + other;
			}
		}

		impl<T: Float> SubAssign for $matrix<T> {
			fn sub_assign(&mut self, other: Self) {
				*self = *self - other;
			}
		}

		impl<T: Float> MulAssign<T> for $matrix<T> {
			fn mul_assign(&mut self, scalar: T) {
				*self = *self * scalar;
			}
		}

		impl<T: Float> MulAssign for $matrix<T> {
			fn mul_assign(&mut self, other: Self) {
				*self = *self * other;
			}
		}

		impl<T: Float> DivAssign<T> for $matrix<T> {
			fn div_assign(&mut self, scalar: T) {
				*self = *self / scalar;
			}
		}
	};
}

matrix_ops!(Mat3, 3, inline);
matrix_ops!(Mat4, 4);

impl<T: Float> Mul<Mat3<T>> for Vec3<T> {
	type Output = Self;

	fn mul(self, matrix: Mat3<T>) -> Self {
		Self {
			x: self.x * matrix.m[0][0] + self.y * matrix.m[1][0] + self.z * matrix.m[2][0],
			y: self.x * matrix.m[0][1] + self.y * matrix.m[1][1] + self.z * matrix.m[2][1],
			z: self.x * matrix.m[0][2] + self.y * matrix.m[1][2] + self.z * matrix.m[2][2],
		}
	}
}

impl<T: Float> Mul<Mat4<T>> for Vec4<T> {
	type Output = Self;

	fn mul(self, matrix: Mat4<T>) -> Self {
		Self {
			x: self.x * matrix.m[0][0]
				+ self.y * matrix.m[1][0]
				+ self.z * matrix.m[2][0]
				+ self.w * matrix.m[3][0],
			y: self.x * matrix.m[0][1]
				+ self.y * matrix.m[1][1]
				+ self.z * matrix.m[2][1]
				+ self.w * matrix.m[3][1],
			z: self.x * matrix.m[0][2]
				+ self.y * matrix.m[1][2]
				+ self.z * matrix.m[2][2]
				+ self.w * matrix.m[3][2],
			w: self.x * matrix.m[0][3]
				+ self.y * matrix.m[1][3]
				+ self.z * matrix.m[2][3]
				+ self.w * matrix.m[3][3],
		}
	}
}

macro_rules! scalar_left_matrix {
	($scalar:ty, $matrix:ident) => {
		impl Mul<$matrix<$scalar>> for $scalar {
			type Output = $matrix<$scalar>;

			fn mul(self, matrix: $matrix<$scalar>) -> Self::Output {
				matrix * self
			}
		}
	};
}

scalar_left_matrix!(f32, Mat3);
scalar_left_matrix!(f32, Mat4);
scalar_left_matrix!(f64, Mat3);
scalar_left_matrix!(f64, Mat4);
