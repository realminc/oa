//! Compact host-side Vulkan linear math.
//!
//! VLM is independent of the GPU `Matrix` value. It owns packed scalar vectors,
//! quaternions, matrices, affine transforms, camera projections, and viewport
//! conversion under one fixed convention: right-handed coordinates, `+X`
//! right, `+Y` up, camera-forward `-Z`, row-major storage, row-vector
//! multiplication, and Vulkan `[0, 1]` clip depth.

mod affine;
mod matrix;
mod projection;
mod quaternion;
mod scalar;
mod vector;

pub use affine::{AffineDecomposition, DAffineDecomposition, DTrsDecomposition, TrsDecomposition};
pub use matrix::{DMat3, DMat4, Mat3, Mat4};
pub use projection::{DViewport, Viewport};
pub use quaternion::{DQuat, Quat, RotationOrder};
pub use scalar::{Float, degrees, radians};
pub use vector::{DVec2, DVec3, DVec4, Vec2, Vec3, Vec4};

pub const PI: f32 = std::f32::consts::PI;
pub const TOLERANCE: f32 = 1.0e-5;
pub const INVERSE_TOLERANCE: f32 = f32::EPSILON * 32.0;

/// Converts spherical `(yaw, pitch, radius)` coordinates to Cartesian space.
pub fn spherical_to_cartesian<T: Float>(yaw_radians: T, pitch_radians: T, radius: T) -> Vec3<T> {
	let cosine_yaw = yaw_radians.cos();
	let sine_yaw = yaw_radians.sin();
	let cosine_pitch = pitch_radians.cos();
	let sine_pitch = pitch_radians.sin();
	Vec3 {
		x: radius * sine_yaw * cosine_pitch,
		y: radius * sine_pitch,
		z: radius * cosine_yaw * cosine_pitch,
	}
}

/// Converts Cartesian coordinates to `(yaw, pitch, radius)`.
pub fn cartesian_to_spherical<T: Float>(value: Vec3<T>) -> Vec3<T> {
	let radius = value.length();
	if radius < T::TOLERANCE {
		return Vec3::default();
	}
	Vec3 {
		x: value.x.atan2(value.z),
		y: (value.y / radius).asin(),
		z: radius,
	}
}
