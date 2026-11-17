//! Semantic camera value.
//!
//! Donor: `oacpp/source/cpp/include/oa/render/camera.h` and
//! `oacpp/source/cpp/include/oa/render/fnCamera.h`.
//! Adaptation: placement is now a composed [`Transform`] value (matching donor
//! `oa::Transform localTransform`); lens parameters (fov, near, far, projection)
//! are the camera-shape fields. Orbit helpers delegate to the transform.

use crate::{
	Error, Result,
	core::transform::Transform,
	vlm::{Mat4, Vec3},
};

/// Projection kind for a [`Camera`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraProjection {
	Perspective,
	Orthographic,
}

/// Semantic camera value.
///
/// Composes a [`Transform`] (placement) with lens parameters (projection shape).
/// `get_view_matrix` derives from the transform's position and forward direction.
/// `get_projection_matrix` uses the lens parameters with Vulkan clip-space
/// conventions (reversed-Y, depth `[0, 1]`).
///
/// Donor: `oa::Camera` / `oa::FnCamera`.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
	transform: Transform,
	projection: CameraProjection,
	fov_y: f32,
	aspect: f32,
	near: f32,
	far: f32,
	ortho_width: f32,
	ortho_height: f32,
	zoom: f32,
	offset_x: f32,
	offset_y: f32,
	/// Pivot used by the orbit helpers.
	orbit_target: Vec3,
}

impl Default for Camera {
	/// Perspective camera at `(0, 0, 5)`, looking toward the origin.
	fn default() -> Self {
		Self::new_perspective(
			Vec3 {
				x: 0.0,
				y: 0.0,
				z: 5.0,
			},
			Vec3 {
				x: 0.0,
				y: 0.0,
				z: 0.0,
			},
			60.0,
			1280.0 / 720.0,
			0.1,
			10000.0,
		)
	}
}

impl Camera {
	/// Create a perspective camera.
	///
	/// `fov_y` is in degrees. `near` and `far` must be finite, positive, and
	/// `near < far`. `aspect` must be finite and positive.
	pub fn new_perspective(
		position: Vec3,
		target: Vec3,
		fov_y: f32,
		aspect: f32,
		near: f32,
		far: f32,
	) -> Self {
		let mut transform = Transform::default();
		transform.set_position(position);
		transform.look_at(
			target,
			Vec3 {
				x: 0.0,
				y: 1.0,
				z: 0.0,
			},
		);
		Self {
			transform,
			projection: CameraProjection::Perspective,
			fov_y,
			aspect,
			near,
			far,
			ortho_width: 1.0,
			ortho_height: 1.0,
			zoom: 1.0,
			offset_x: 0.0,
			offset_y: 0.0,
			orbit_target: Vec3 {
				x: 0.0,
				y: 0.0,
				z: 0.0,
			},
		}
	}

	/// Create an orthographic camera.
	pub fn new_orthographic(width: f32, height: f32, near: f32, far: f32) -> Self {
		let position = Vec3 {
			x: 0.0,
			y: 0.0,
			z: 1.0,
		};
		let target = Vec3 {
			x: 0.0,
			y: 0.0,
			z: 0.0,
		};
		let mut transform = Transform::default();
		transform.set_position(position);
		transform.look_at(
			target,
			Vec3 {
				x: 0.0,
				y: 1.0,
				z: 0.0,
			},
		);
		Self {
			transform,
			projection: CameraProjection::Orthographic,
			fov_y: 60.0,
			aspect: if height != 0.0 { width / height } else { 1.0 },
			near,
			far,
			ortho_width: width,
			ortho_height: height,
			zoom: 1.0,
			offset_x: 0.0,
			offset_y: 0.0,
			orbit_target: Vec3 {
				x: 0.0,
				y: 0.0,
				z: 0.0,
			},
		}
	}

	// ── Placement ─────────────────────────────────────────────────────────────

	/// Return a reference to the placement transform.
	pub const fn transform(&self) -> &Transform {
		&self.transform
	}

	/// Return a mutable reference to the placement transform.
	pub fn transform_mut(&mut self) -> &mut Transform {
		&mut self.transform
	}

	/// Return the camera position (from the placement transform).
	pub fn position(&self) -> Vec3 {
		self.transform.position()
	}

	/// Set the camera position.
	pub fn set_position(&mut self, position: Vec3) {
		self.transform.set_position(position);
	}

	/// Orient the camera to look at `target` (world up = +Y).
	pub fn look_at(&mut self, target: Vec3) {
		self.transform.look_at(
			target,
			Vec3 {
				x: 0.0,
				y: 1.0,
				z: 0.0,
			},
		);
	}

	// ── Lens parameters ───────────────────────────────────────────────────────

	/// Return the current projection kind.
	pub const fn projection(&self) -> CameraProjection {
		self.projection
	}

	/// Return the vertical field of view in degrees.
	pub const fn fov_y(&self) -> f32 {
		self.fov_y
	}

	/// Return the effective vertical field of view in degrees, applying zoom.
	///
	/// Matches donor `getEffectiveFovY`: `fov_y / zoom` clamped to (0°, 180°).
	pub fn effective_fov_y(&self) -> f32 {
		(self.fov_y / self.zoom.max(f32::EPSILON)).clamp(f32::EPSILON, 179.999)
	}

	/// Return the aspect ratio.
	pub const fn aspect(&self) -> f32 {
		self.aspect
	}

	/// Return the near plane distance.
	pub const fn near(&self) -> f32 {
		self.near
	}

	/// Return the far plane distance.
	pub const fn far(&self) -> f32 {
		self.far
	}

	/// Set perspective projection parameters.
	pub fn set_perspective(&mut self, fov_y: f32, aspect: f32, near: f32, far: f32) {
		self.projection = CameraProjection::Perspective;
		self.fov_y = fov_y;
		self.aspect = aspect;
		self.near = near;
		self.far = far;
	}

	/// Set orthographic projection parameters.
	pub fn set_orthographic(&mut self, width: f32, height: f32, near: f32, far: f32) {
		self.projection = CameraProjection::Orthographic;
		self.ortho_width = width;
		self.ortho_height = height;
		self.near = near;
		self.far = far;
	}

	/// Set aspect ratio without changing other perspective parameters.
	pub fn set_aspect(&mut self, aspect: f32) {
		self.aspect = aspect;
	}

	/// Adjust aspect ratio and orthographic extent to match a window size.
	pub fn fit_to_window(&mut self, window_width: f32, window_height: f32) {
		if window_height > 0.0 {
			self.aspect = window_width / window_height;
		}
		if matches!(self.projection, CameraProjection::Orthographic) {
			self.ortho_width = window_width;
			self.ortho_height = window_height;
		}
	}

	/// Set zoom factor (must be > 0).
	pub fn set_zoom(&mut self, zoom: f32) {
		self.zoom = zoom.max(f32::EPSILON);
	}

	/// Return zoom factor.
	pub const fn zoom(&self) -> f32 {
		self.zoom
	}

	// ── Matrix derivation ─────────────────────────────────────────────────────

	/// Compute the row-vector view matrix from the placement transform.
	///
	/// The camera faces its local -Z axis (OA convention). Returns an identity
	/// view when the forward vector is degenerate.
	pub fn get_view_matrix(&self) -> Mat4 {
		let eye = self.transform.position();
		let fwd = self.transform.forward();
		let up = Vec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		};
		let center = Vec3 {
			x: eye.x + fwd.x,
			y: eye.y + fwd.y,
			z: eye.z + fwd.z,
		};
		look_at(eye, center, up)
	}

	/// Compute the row-vector projection matrix in Vulkan clip space.
	///
	/// Perspective: reversed-Y (NDC Y points down) and Vulkan depth [0, 1].
	/// Orthographic: likewise.
	pub fn get_projection_matrix(&self) -> Mat4 {
		match self.projection {
			CameraProjection::Perspective => {
				perspective_vlm(self.effective_fov_y(), self.aspect, self.near, self.far)
			}
			CameraProjection::Orthographic => {
				let hw = (self.ortho_width * 0.5) / self.zoom.max(f32::EPSILON);
				let hh = (self.ortho_height * 0.5) / self.zoom.max(f32::EPSILON);
				ortho_vlm(
					-hw + self.offset_x,
					hw + self.offset_x,
					-hh + self.offset_y,
					hh + self.offset_y,
					self.near,
					self.far,
				)
			}
		}
	}

	/// Compute the combined view-projection matrix.
	pub fn get_view_projection_matrix(&self) -> Mat4 {
		// row-vector: v * V * P
		self.get_view_matrix() * self.get_projection_matrix()
	}

	// ── Orbit helpers ─────────────────────────────────────────────────────────

	/// Set the orbit pivot.
	pub fn set_orbit_target(&mut self, target: Vec3) {
		self.orbit_target = target;
	}

	/// Set the orbit radius without changing the pivot or angles.
	pub fn set_orbit_distance(&mut self, distance: f32) {
		let pos = self.transform.position();
		let dir = pos - self.orbit_target;
		let len = dir.length();
		if len > f32::EPSILON {
			let normalized = Vec3 {
				x: dir.x / len,
				y: dir.y / len,
				z: dir.z / len,
			};
			let new_pos = self.orbit_target + normalized * distance;
			self.transform.set_position(new_pos);
			self.transform.look_at(
				self.orbit_target,
				Vec3 {
					x: 0.0,
					y: 1.0,
					z: 0.0,
				},
			);
		}
	}

	/// Apply incremental yaw (around world Y) and pitch (around local right) deltas.
	pub fn orbit_yaw_pitch(&mut self, yaw_delta: f32, pitch_delta: f32) {
		let offset = self.transform.position() - self.orbit_target;
		let offset = rotate_y(offset, yaw_delta.to_radians());
		let right = Vec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		}
		.cross(offset)
		.try_normalized()
		.unwrap_or(Vec3 {
			x: 1.0,
			y: 0.0,
			z: 0.0,
		});
		let offset = rotate_axis(offset, right, pitch_delta.to_radians());
		let new_pos = self.orbit_target + offset;
		self.transform.set_position(new_pos);
		self.transform.look_at(
			self.orbit_target,
			Vec3 {
				x: 0.0,
				y: 1.0,
				z: 0.0,
			},
		);
	}

	/// Set absolute yaw and pitch around the orbit pivot.
	pub fn orbit_set_yaw_pitch(&mut self, yaw: f32, pitch: f32) {
		let radius = (self.transform.position() - self.orbit_target).length();
		let (sy, cy) = yaw.to_radians().sin_cos();
		let (sp, cp) = pitch.to_radians().sin_cos();
		let new_pos = Vec3 {
			x: self.orbit_target.x + radius * cy * cp,
			y: self.orbit_target.y + radius * sp,
			z: self.orbit_target.z + radius * sy * cp,
		};
		self.transform.set_position(new_pos);
		self.transform.look_at(
			self.orbit_target,
			Vec3 {
				x: 0.0,
				y: 1.0,
				z: 0.0,
			},
		);
	}

	/// Validate camera parameters for use in a frame.
	///
	/// # Errors
	///
	/// Returns an error for non-finite parameters, zero/negative near/far,
	/// `near >= far`, or a non-finite view-projection matrix.
	pub fn validate(&self) -> Result<()> {
		if !self.fov_y.is_finite()
			|| !self.aspect.is_finite()
			|| !self.near.is_finite()
			|| !self.far.is_finite()
			|| self.near <= 0.0
			|| self.far <= 0.0
			|| self.near >= self.far
			|| self.aspect <= 0.0
		{
			return Err(Error::invalid_argument(
				"render::Camera requires finite positive near/far with near < far and a positive aspect ratio",
			));
		}
		let vp = self.get_view_projection_matrix();
		if !vp.is_finite() {
			return Err(Error::invalid_argument(
				"render::Camera view-projection matrix must be finite",
			));
		}
		Ok(())
	}
}

// ── Math helpers ──────────────────────────────────────────────────────────────

/// Row-vector look-at view matrix (OA convention: camera-forward is -Z).
fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
	let f = (center - eye).try_normalized().unwrap_or(Vec3 {
		x: 0.0,
		y: 0.0,
		z: -1.0,
	});
	let r = f.cross(up).try_normalized().unwrap_or(Vec3 {
		x: 1.0,
		y: 0.0,
		z: 0.0,
	});
	let u = r.cross(f);
	// Row-vector: rows are the three basis vectors; translation in last row.
	Mat4 {
		m: [
			[r.x, u.x, -f.x, 0.0],
			[r.y, u.y, -f.y, 0.0],
			[r.z, u.z, -f.z, 0.0],
			[-eye.dot(r), -eye.dot(u), eye.dot(f), 1.0],
		],
	}
}

/// Row-vector perspective matrix in Vulkan clip space (depth [0, 1], Y flipped).
fn perspective_vlm(fov_y_deg: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
	let half = (fov_y_deg * 0.5).to_radians().tan();
	let sy = 1.0 / half;
	let sx = sy / aspect;
	let range = far / (near - far);
	Mat4 {
		m: [
			[sx, 0.0, 0.0, 0.0],
			[0.0, -sy, 0.0, 0.0],
			[0.0, 0.0, range, -1.0],
			[0.0, 0.0, near * range, 0.0],
		],
	}
}

/// Row-vector orthographic matrix in Vulkan clip space (depth [0, 1], Y flipped).
#[allow(clippy::too_many_arguments)]
fn ortho_vlm(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
	let rml = right - left;
	let tmb = top - bottom;
	let fmn = far - near;
	Mat4 {
		m: [
			[2.0 / rml, 0.0, 0.0, 0.0],
			[0.0, -2.0 / tmb, 0.0, 0.0],
			[0.0, 0.0, -1.0 / fmn, 0.0],
			[
				-(right + left) / rml,
				(top + bottom) / tmb,
				-near / fmn,
				1.0,
			],
		],
	}
}

/// Rotate a vector around the world Y axis.
fn rotate_y(v: Vec3, radians: f32) -> Vec3 {
	let (s, c) = radians.sin_cos();
	Vec3 {
		x: v.x * c + v.z * s,
		y: v.y,
		z: -v.x * s + v.z * c,
	}
}

/// Rotate a vector around an arbitrary axis (Rodrigues).
fn rotate_axis(v: Vec3, axis: Vec3, radians: f32) -> Vec3 {
	let (s, c) = radians.sin_cos();
	let k = axis;
	v * c + k.cross(v) * s + k * k.dot(v) * (1.0 - c)
}
