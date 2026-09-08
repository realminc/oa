use super::matrix::Mat4;
use super::scalar::{Float, radians, valid_tolerance};
use super::vector::{Vec2, Vec3};

/// A top-origin viewport with Vulkan `[0, 1]` depth.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport<T: Float = f32> {
	pub x: T,
	pub y: T,
	pub width: T,
	pub height: T,
	pub min_depth: T,
	pub max_depth: T,
}

pub type DViewport = Viewport<f64>;

impl<T: Float> Default for Viewport<T> {
	fn default() -> Self {
		Self {
			x: T::ZERO,
			y: T::ZERO,
			width: T::ONE,
			height: T::ONE,
			min_depth: T::ZERO,
			max_depth: T::ONE,
		}
	}
}

impl<T: Float> Viewport<T> {
	pub fn is_valid(self) -> bool {
		self.x.is_finite()
			&& self.y.is_finite()
			&& self.width.is_finite()
			&& self.height.is_finite()
			&& self.min_depth.is_finite()
			&& self.max_depth.is_finite()
			&& self.width > T::ZERO
			&& self.height > T::ZERO
			&& self.min_depth >= T::ZERO
			&& self.max_depth <= T::ONE
			&& self.min_depth < self.max_depth
	}

	pub fn try_project(
		self,
		point: Vec3<T>,
		world_view_projection: Mat4<T>,
		tolerance: T,
	) -> Option<Vec3<T>> {
		if !self.is_valid() || !valid_tolerance(tolerance) {
			return None;
		}
		let ndc = world_view_projection.try_project_point(point, tolerance)?;
		let result = Vec3 {
			x: self.x + (ndc.x + T::ONE) * T::HALF * self.width,
			y: self.y + (T::ONE - ndc.y) * T::HALF * self.height,
			z: self.min_depth + ndc.z * (self.max_depth - self.min_depth),
		};
		result.is_finite().then_some(result)
	}

	pub fn try_unproject(
		self,
		point: Vec3<T>,
		world_view_projection: Mat4<T>,
		tolerance: T,
	) -> Option<Vec3<T>> {
		if !self.is_valid() || !point.is_finite() || !valid_tolerance(tolerance) {
			return None;
		}
		let ndc = Vec3 {
			x: T::TWO * (point.x - self.x) / self.width - T::ONE,
			y: T::ONE - T::TWO * (point.y - self.y) / self.height,
			z: (point.z - self.min_depth) / (self.max_depth - self.min_depth),
		};
		world_view_projection
			.try_inverse(tolerance)?
			.try_project_point(ndc, tolerance)
	}
}

impl<T: Float> Mat4<T> {
	pub fn try_perspective_off_center(
		left: T,
		right: T,
		bottom: T,
		top: T,
		near: T,
		far: T,
	) -> Option<Self> {
		if !left.is_finite()
			|| !right.is_finite()
			|| !bottom.is_finite()
			|| !top.is_finite()
			|| !near.is_finite()
			|| !far.is_finite()
			|| right <= left
			|| top <= bottom
			|| near <= T::ZERO
			|| far <= near
		{
			return None;
		}
		let width = right - left;
		let height = top - bottom;
		let depth = near - far;
		if !width.is_finite() || !height.is_finite() || !depth.is_finite() {
			return None;
		}
		let mut result = Self::default();
		result.m[0][0] = T::TWO * near / width;
		result.m[1][1] = T::TWO * near / height;
		result.m[2][0] = (right + left) / width;
		result.m[2][1] = (top + bottom) / height;
		result.m[2][2] = far / depth;
		result.m[2][3] = -T::ONE;
		result.m[3][2] = far * near / depth;
		result.is_finite().then_some(result)
	}

	pub fn try_perspective(fov_y_degrees: T, aspect: T, near: T, far: T) -> Option<Self> {
		if !fov_y_degrees.is_finite()
			|| !aspect.is_finite()
			|| !near.is_finite()
			|| !far.is_finite()
			|| fov_y_degrees <= T::ZERO
			|| fov_y_degrees >= T::from_f64(180.0)
			|| aspect <= T::ZERO
			|| near <= T::ZERO
			|| far <= near
		{
			return None;
		}
		let half_height = near * (radians(fov_y_degrees) * T::HALF).tan();
		if !half_height.is_finite() || half_height <= T::ZERO {
			return None;
		}
		let half_width = half_height * aspect;
		Self::try_perspective_off_center(
			-half_width,
			half_width,
			-half_height,
			half_height,
			near,
			far,
		)
	}

	pub fn try_perspective_shifted(
		fov_y_degrees: T,
		aspect: T,
		near: T,
		far: T,
		ndc_offset: Vec2<T>,
	) -> Option<Self> {
		if !ndc_offset.is_finite() {
			return None;
		}
		let mut result = Self::try_perspective(fov_y_degrees, aspect, near, far)?;
		result.m[2][0] = -ndc_offset.x;
		result.m[2][1] = -ndc_offset.y;
		Some(result)
	}

	pub fn try_orthographic_off_center(
		left: T,
		right: T,
		bottom: T,
		top: T,
		near: T,
		far: T,
	) -> Option<Self> {
		if !left.is_finite()
			|| !right.is_finite()
			|| !bottom.is_finite()
			|| !top.is_finite()
			|| !near.is_finite()
			|| !far.is_finite()
			|| right <= left
			|| top <= bottom
			|| far <= near
		{
			return None;
		}
		let width = right - left;
		let height = top - bottom;
		let depth = far - near;
		if !width.is_finite() || !height.is_finite() || !depth.is_finite() {
			return None;
		}
		let mut result = Self::identity();
		result.m[0][0] = T::TWO / width;
		result.m[1][1] = T::TWO / height;
		result.m[2][2] = -T::ONE / depth;
		result.m[3][0] = -(right + left) / width;
		result.m[3][1] = -(top + bottom) / height;
		result.m[3][2] = -near / depth;
		result.is_finite().then_some(result)
	}

	pub fn try_orthographic(width: T, height: T, near: T, far: T, zoom: T) -> Option<Self> {
		if !width.is_finite()
			|| !height.is_finite()
			|| !zoom.is_finite()
			|| width <= T::ZERO
			|| height <= T::ZERO
			|| zoom <= T::ZERO
		{
			return None;
		}
		let half_width = width * T::HALF / zoom;
		let half_height = height * T::HALF / zoom;
		Self::try_orthographic_off_center(
			-half_width,
			half_width,
			-half_height,
			half_height,
			near,
			far,
		)
	}

	pub fn try_orthographic_shifted(
		width: T,
		height: T,
		near: T,
		far: T,
		zoom: T,
		ndc_offset: Vec2<T>,
	) -> Option<Self> {
		if !ndc_offset.is_finite() {
			return None;
		}
		let mut result = Self::try_orthographic(width, height, near, far, zoom)?;
		result.m[3][0] = -ndc_offset.x;
		result.m[3][1] = -ndc_offset.y;
		Some(result)
	}

	pub fn try_perspective_reverse_z(fov_y_degrees: T, aspect: T, near: T, far: T) -> Option<Self> {
		if !fov_y_degrees.is_finite()
			|| !aspect.is_finite()
			|| !near.is_finite()
			|| !far.is_finite()
			|| fov_y_degrees <= T::ZERO
			|| fov_y_degrees >= T::from_f64(180.0)
			|| aspect <= T::ZERO
			|| near <= T::ZERO
			|| far <= near
		{
			return None;
		}
		let tangent = (radians(fov_y_degrees) * T::HALF).tan();
		if !tangent.is_finite() || tangent <= T::ZERO {
			return None;
		}
		let mut result = Self::default();
		result.m[0][0] = T::ONE / (aspect * tangent);
		result.m[1][1] = T::ONE / tangent;
		result.m[2][2] = near / (far - near);
		result.m[2][3] = -T::ONE;
		result.m[3][2] = far * near / (far - near);
		result.is_finite().then_some(result)
	}

	pub fn try_perspective_reverse_z_infinite(
		fov_y_degrees: T,
		aspect: T,
		near: T,
	) -> Option<Self> {
		if !fov_y_degrees.is_finite()
			|| !aspect.is_finite()
			|| !near.is_finite()
			|| fov_y_degrees <= T::ZERO
			|| fov_y_degrees >= T::from_f64(180.0)
			|| aspect <= T::ZERO
			|| near <= T::ZERO
		{
			return None;
		}
		let tangent = (radians(fov_y_degrees) * T::HALF).tan();
		if !tangent.is_finite() || tangent <= T::ZERO {
			return None;
		}
		let mut result = Self::default();
		result.m[0][0] = T::ONE / (aspect * tangent);
		result.m[1][1] = T::ONE / tangent;
		result.m[2][3] = -T::ONE;
		result.m[3][2] = near;
		result.is_finite().then_some(result)
	}
}
