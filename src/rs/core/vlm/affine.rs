use super::matrix::{Mat3, Mat4};
use super::quaternion::Quat;
use super::scalar::{Float, approx, valid_tolerance};
use super::vector::Vec3;

/// Translation, rotation, signed scale, and reflection extracted from a strict TRS matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrsDecomposition<T: Float = f32> {
	pub translation: Vec3<T>,
	pub rotation: Quat<T>,
	pub scale: Vec3<T>,
	pub reflected: bool,
}

/// Translation, rotation, signed scale, row-basis shear, and reflection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineDecomposition<T: Float = f32> {
	pub translation: Vec3<T>,
	pub rotation: Quat<T>,
	pub scale: Vec3<T>,
	/// Dimensionless XY, XZ, and YZ row-basis shear factors.
	pub shear: Vec3<T>,
	pub reflected: bool,
}

pub type DTrsDecomposition = TrsDecomposition<f64>;
pub type DAffineDecomposition = AffineDecomposition<f64>;

impl<T: Float> Default for TrsDecomposition<T> {
	fn default() -> Self {
		Self {
			translation: Vec3::default(),
			rotation: Quat::identity(),
			scale: Vec3 {
				x: T::ONE,
				y: T::ONE,
				z: T::ONE,
			},
			reflected: false,
		}
	}
}

impl<T: Float> Default for AffineDecomposition<T> {
	fn default() -> Self {
		Self {
			translation: Vec3::default(),
			rotation: Quat::identity(),
			scale: Vec3 {
				x: T::ONE,
				y: T::ONE,
				z: T::ONE,
			},
			shear: Vec3::default(),
			reflected: false,
		}
	}
}

impl<T: Float> Mat3<T> {
	pub fn is_identity(self, tolerance: T) -> bool {
		self.approximately_equal(Self::identity(), tolerance, tolerance)
	}

	pub fn is_orthonormal(self, tolerance: T) -> bool {
		if !valid_tolerance(tolerance) {
			return false;
		}
		let x = Vec3 {
			x: self.m[0][0],
			y: self.m[0][1],
			z: self.m[0][2],
		};
		let y = Vec3 {
			x: self.m[1][0],
			y: self.m[1][1],
			z: self.m[1][2],
		};
		let z = Vec3 {
			x: self.m[2][0],
			y: self.m[2][1],
			z: self.m[2][2],
		};
		x.is_finite()
			&& y.is_finite()
			&& z.is_finite()
			&& approx(x.length_squared(), T::ONE, tolerance, tolerance)
			&& approx(y.length_squared(), T::ONE, tolerance, tolerance)
			&& approx(z.length_squared(), T::ONE, tolerance, tolerance)
			&& approx(x.dot(y), T::ZERO, tolerance, T::ZERO)
			&& approx(x.dot(z), T::ZERO, tolerance, T::ZERO)
			&& approx(y.dot(z), T::ZERO, tolerance, T::ZERO)
	}

	pub fn is_proper_rotation(self, tolerance: T) -> bool {
		self.is_orthonormal(tolerance) && approx(self.determinant(), T::ONE, tolerance, tolerance)
	}

	pub fn try_orthonormal_basis(first: Vec3<T>, second: Vec3<T>, tolerance: T) -> Option<Self> {
		if !valid_tolerance(tolerance) || first.length() <= tolerance {
			return None;
		}
		let first = first.try_normalized()?;
		let rejected = second.try_reject_from(first, tolerance)?;
		if rejected.length() <= tolerance {
			return None;
		}
		let second = rejected.try_normalized()?;
		let third = first.cross(second);
		let basis = Self {
			m: [
				[first.x, first.y, first.z],
				[second.x, second.y, second.z],
				[third.x, third.y, third.z],
			],
		};
		basis
			.is_proper_rotation(tolerance * T::from_f64(8.0))
			.then_some(basis)
	}

	pub fn try_transform_normal(self, normal: Vec3<T>, tolerance: T) -> Option<Vec3<T>> {
		let transformed = normal * self;
		if !valid_tolerance(tolerance) || transformed.length() <= tolerance {
			return None;
		}
		transformed.try_normalized()
	}
}

impl<T: Float> Mat4<T> {
	pub fn is_identity(self, tolerance: T) -> bool {
		self.approximately_equal(Self::identity(), tolerance, tolerance)
	}

	pub fn try_view_from_pose(position: Vec3<T>, rotation: Quat<T>, tolerance: T) -> Option<Self> {
		if !valid_tolerance(tolerance) || !position.is_finite() {
			return None;
		}
		let inverse_rotation = rotation.conjugate().try_normalized()?;
		let view = Self::translation(-position) * Self::try_from_quaternion(inverse_rotation)?;
		view.is_finite().then_some(view)
	}

	#[inline]
	pub fn try_look_at(
		eye: Vec3<T>,
		target: Vec3<T>,
		world_up: Vec3<T>,
		tolerance: T,
	) -> Option<Self> {
		if !valid_tolerance(tolerance)
			|| !eye.is_finite()
			|| !target.is_finite()
			|| !world_up.is_finite()
		{
			return None;
		}
		let forward_value = target - eye;
		if forward_value.length() <= tolerance {
			return None;
		}
		let forward = forward_value.try_normalized()?;
		let right_value = forward.cross(world_up);
		if right_value.length() <= tolerance {
			return None;
		}
		let right = right_value.try_normalized()?;
		let up = right.cross(forward);
		let basis = Mat3 {
			m: [
				[right.x, up.x, -forward.x],
				[right.y, up.y, -forward.y],
				[right.z, up.z, -forward.z],
			],
		};
		let translation = Vec3 {
			x: -right.dot(eye),
			y: -up.dot(eye),
			z: forward.dot(eye),
		};
		let view = Self::from_affine(basis, translation);
		view.is_finite().then_some(view)
	}

	pub fn compose_affine(decomposition: AffineDecomposition<T>) -> Option<Self> {
		if !decomposition.translation.is_finite()
			|| !decomposition.scale.is_finite()
			|| !decomposition.shear.is_finite()
		{
			return None;
		}
		let rotation = Self::try_from_quaternion(decomposition.rotation)?.linear_part();
		let x = Vec3 {
			x: rotation.m[0][0],
			y: rotation.m[0][1],
			z: rotation.m[0][2],
		};
		let y = Vec3 {
			x: rotation.m[1][0],
			y: rotation.m[1][1],
			z: rotation.m[1][2],
		};
		let z = Vec3 {
			x: rotation.m[2][0],
			y: rotation.m[2][1],
			z: rotation.m[2][2],
		};
		let row_x = x * decomposition.scale.x;
		let row_y = (y + x * decomposition.shear.x) * decomposition.scale.y;
		let row_z =
			(z + x * decomposition.shear.y + y * decomposition.shear.z) * decomposition.scale.z;
		Some(Self::from_affine(
			Mat3 {
				m: [
					[row_x.x, row_x.y, row_x.z],
					[row_y.x, row_y.y, row_y.z],
					[row_z.x, row_z.y, row_z.z],
				],
			},
			decomposition.translation,
		))
	}

	pub fn try_decompose_affine(self, tolerance: T) -> Option<AffineDecomposition<T>> {
		if !valid_tolerance(tolerance) || !self.is_affine(tolerance) {
			return None;
		}
		let linear = self.linear_part();
		let mut x = Vec3 {
			x: linear.m[0][0],
			y: linear.m[0][1],
			z: linear.m[0][2],
		};
		let mut y = Vec3 {
			x: linear.m[1][0],
			y: linear.m[1][1],
			z: linear.m[1][2],
		};
		let mut z = Vec3 {
			x: linear.m[2][0],
			y: linear.m[2][1],
			z: linear.m[2][2],
		};
		let mut scale_x = x.length();
		if !scale_x.is_finite() || scale_x <= tolerance {
			return None;
		}
		x /= scale_x;
		let mut shear_xy = x.dot(y);
		y -= x * shear_xy;
		let scale_y = y.length();
		if !scale_y.is_finite() || scale_y <= tolerance {
			return None;
		}
		y /= scale_y;
		shear_xy = shear_xy / scale_y;

		let mut shear_xz = x.dot(z);
		z -= x * shear_xz;
		let mut shear_yz = y.dot(z);
		z -= y * shear_yz;
		let scale_z = z.length();
		if !scale_z.is_finite() || scale_z <= tolerance {
			return None;
		}
		z /= scale_z;
		shear_xz = shear_xz / scale_z;
		shear_yz = shear_yz / scale_z;
		if !shear_xy.is_finite() || !shear_xz.is_finite() || !shear_yz.is_finite() {
			return None;
		}
		let orientation = x.cross(y).dot(z);
		let orientation_tolerance = tolerance * T::from_f64(8.0);
		if !orientation.is_finite()
			|| !approx(
				orientation.abs(),
				T::ONE,
				orientation_tolerance,
				orientation_tolerance,
			) {
			return None;
		}
		let reflected = orientation < T::ZERO;
		if reflected {
			x = -x;
			scale_x = -scale_x;
			shear_xy = -shear_xy;
			shear_xz = -shear_xz;
		}
		let rotation = Quat::try_from_rotation_matrix(
			Mat3 {
				m: [[x.x, x.y, x.z], [y.x, y.y, y.z], [z.x, z.y, z.z]],
			},
			orientation_tolerance,
		)?;
		Some(AffineDecomposition {
			translation: self.translation_part(),
			rotation,
			scale: Vec3 {
				x: scale_x,
				y: scale_y,
				z: scale_z,
			},
			shear: Vec3 {
				x: shear_xy,
				y: shear_xz,
				z: shear_yz,
			},
			reflected,
		})
	}

	pub fn try_decompose_trs(self, tolerance: T) -> Option<TrsDecomposition<T>> {
		let affine = self.try_decompose_affine(tolerance)?;
		let shear_tolerance = tolerance * T::from_f64(8.0);
		if affine.shear.x.abs() > shear_tolerance
			|| affine.shear.y.abs() > shear_tolerance
			|| affine.shear.z.abs() > shear_tolerance
		{
			return None;
		}
		Some(TrsDecomposition {
			translation: affine.translation,
			rotation: affine.rotation,
			scale: affine.scale,
			reflected: affine.reflected,
		})
	}
}

impl<T: Float> Quat<T> {
	pub fn try_from_rotation_matrix(matrix: Mat3<T>, tolerance: T) -> Option<Self> {
		if !valid_tolerance(tolerance) || !matrix.is_proper_rotation(tolerance) {
			return None;
		}
		quaternion_from_rotation_matrix_unchecked(Mat4::from_affine(matrix, Vec3::default()))
	}

	pub fn try_from_rotation_matrix4(matrix: Mat4<T>, tolerance: T) -> Option<Self> {
		if !matrix.is_affine(tolerance) {
			return None;
		}
		Self::try_from_rotation_matrix(matrix.linear_part(), tolerance)
	}

	pub fn try_from_to(from: Vec3<T>, to: Vec3<T>, tolerance: T) -> Option<Self> {
		if !valid_tolerance(tolerance) || from.length() <= tolerance || to.length() <= tolerance {
			return None;
		}
		let from = from.try_normalized()?;
		let to = to.try_normalized()?;
		let cosine = from.dot(to).max(-T::ONE).min(T::ONE);
		if cosine >= T::ONE - tolerance {
			return Some(Self::identity());
		}
		if cosine <= -T::ONE + tolerance {
			return Self::try_from_axis_angle(from.try_perpendicular(tolerance)?, T::PI);
		}
		let axis = from.cross(to);
		Self {
			x: axis.x,
			y: axis.y,
			z: axis.z,
			w: T::ONE + cosine,
		}
		.try_normalized()
	}

	pub fn try_look_rotation(forward: Vec3<T>, world_up: Vec3<T>, tolerance: T) -> Option<Self> {
		let view = Mat4::try_look_at(Vec3::default(), forward, world_up, tolerance)?;
		Self::try_from_rotation_matrix(view.linear_part().transpose(), tolerance * T::from_f64(8.0))
	}
}

fn quaternion_from_rotation_matrix_unchecked<T: Float>(matrix: Mat4<T>) -> Option<Quat<T>> {
	let at = |row: usize, column: usize| matrix.m[column][row];
	let m00 = at(0, 0);
	let m11 = at(1, 1);
	let m22 = at(2, 2);
	let trace = m00 + m11 + m22;
	let mut result = Quat::identity();
	if trace > T::ZERO {
		let scale = (trace + T::ONE).sqrt() * T::TWO;
		result.w = T::from_f64(0.25) * scale;
		result.x = (at(2, 1) - at(1, 2)) / scale;
		result.y = (at(0, 2) - at(2, 0)) / scale;
		result.z = (at(1, 0) - at(0, 1)) / scale;
	} else if m00 > m11 && m00 > m22 {
		let scale = (T::ONE + m00 - m11 - m22).sqrt() * T::TWO;
		result.w = (at(2, 1) - at(1, 2)) / scale;
		result.x = T::from_f64(0.25) * scale;
		result.y = (at(0, 1) + at(1, 0)) / scale;
		result.z = (at(0, 2) + at(2, 0)) / scale;
	} else if m11 > m22 {
		let scale = (T::ONE + m11 - m00 - m22).sqrt() * T::TWO;
		result.w = (at(0, 2) - at(2, 0)) / scale;
		result.x = (at(0, 1) + at(1, 0)) / scale;
		result.y = T::from_f64(0.25) * scale;
		result.z = (at(1, 2) + at(2, 1)) / scale;
	} else {
		let scale = (T::ONE + m22 - m00 - m11).sqrt() * T::TWO;
		result.w = (at(1, 0) - at(0, 1)) / scale;
		result.x = (at(0, 2) + at(2, 0)) / scale;
		result.y = (at(1, 2) + at(2, 1)) / scale;
		result.z = T::from_f64(0.25) * scale;
	}
	result.try_normalized()
}
