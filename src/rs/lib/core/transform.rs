//! Semantic local spatial transform value.
//!
//! [`Transform`] owns affine placement — translation, rotation, scale, and
//! optional row-basis shear. It is the universal placement primitive composed
//! by cameras, scene nodes, lights, joints, and any other type that exists in
//! 3D space.
//!
//! All stateless operations live as free functions in this module (`FnTransform`
//! in the C++ donor collapses into module functions per the Rust porting rule).
//!
//! Donor: `oacpp/source/cpp/include/oa/core/transform.h` and
//! `oacpp/source/cpp/include/oa/core/fnTransform.h`.
//! Adaptation: C++ static-method namespace → module functions; `setMatrix` /
//! `getMatrix` become `set_matrix` / `get_matrix`; six-D rotation helpers are
//! module-level free functions.

use crate::{
	Error, Result,
	vlm::{INVERSE_TOLERANCE, Mat4, Quat, RotationOrder, Vec3},
};

// ── Transform ─────────────────────────────────────────────────────────────────

/// Semantic local spatial transform value.
///
/// Stores translation + unit-quaternion rotation + per-axis scale + optional
/// dimensionless row-basis shear (XY, XZ, YZ). All constructed values are
/// finite; operations that would produce a non-finite result leave the value
/// unchanged and return an error.
///
/// Donor: `oa::Transform`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
	translation: Vec3,
	rotation: Quat,
	scale: Vec3,
	/// Dimensionless row-basis shear factors: XY, XZ, and YZ.
	shear: Vec3,
}

impl Default for Transform {
	/// Identity transform: zero translation, identity rotation, unit scale, no shear.
	fn default() -> Self {
		Self {
			translation: Vec3::default(),
			rotation: Quat::identity(),
			scale: Vec3 {
				x: 1.0,
				y: 1.0,
				z: 1.0,
			},
			shear: Vec3::default(),
		}
	}
}

impl Transform {
	/// Construct from explicit components.
	///
	/// `rotation` is normalized on construction. `scale` must be non-zero on
	/// each axis for the transform to be invertible, but construction does not
	/// enforce this.
	pub fn new(translation: Vec3, rotation: Quat, scale: Vec3, shear: Vec3) -> Self {
		Self {
			translation,
			rotation: rotation.try_normalized().unwrap_or(Quat::identity()),
			scale,
			shear,
		}
	}

	// ── Setters ───────────────────────────────────────────────────────────────

	/// Set the translation component.
	pub fn set_position(&mut self, position: Vec3) {
		self.translation = position;
	}

	/// Set the rotation component. The quaternion is normalized.
	pub fn set_rotation(&mut self, rotation: Quat) {
		self.rotation = rotation.try_normalized().unwrap_or(Quat::identity());
	}

	/// Set the per-axis scale.
	pub fn set_scale(&mut self, scale: Vec3) {
		self.scale = scale;
	}

	/// Set the row-basis shear (XY, XZ, YZ).
	pub fn set_shear(&mut self, shear: Vec3) {
		self.shear = shear;
	}

	// ── Getters ───────────────────────────────────────────────────────────────

	/// Return the translation.
	pub const fn position(&self) -> Vec3 {
		self.translation
	}

	/// Return the rotation quaternion (always normalized).
	pub const fn rotation(&self) -> Quat {
		self.rotation
	}

	/// Return the per-axis scale.
	pub const fn scale(&self) -> Vec3 {
		self.scale
	}

	/// Return the row-basis shear (XY, XZ, YZ).
	pub const fn shear(&self) -> Vec3 {
		self.shear
	}

	// ── Derived directions (OA: camera-forward = -Z) ──────────────────────────

	/// Return the local forward vector (-Z in world space).
	pub fn forward(&self) -> Vec3 {
		let neg_z = Vec3 {
			x: 0.0,
			y: 0.0,
			z: -1.0,
		};
		self.rotation.try_rotate(neg_z).unwrap_or(neg_z)
	}

	/// Return the local right vector (+X in world space).
	pub fn right(&self) -> Vec3 {
		let pos_x = Vec3 {
			x: 1.0,
			y: 0.0,
			z: 0.0,
		};
		self.rotation.try_rotate(pos_x).unwrap_or(pos_x)
	}

	/// Return the local up vector (+Y in world space).
	pub fn up(&self) -> Vec3 {
		let pos_y = Vec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		};
		self.rotation.try_rotate(pos_y).unwrap_or(pos_y)
	}

	// ── Rotation helpers ──────────────────────────────────────────────────────

	/// Orient the transform to look toward `target` with the given `world_up`.
	///
	/// When `target` equals the current position or `world_up` is degenerate the
	/// rotation is left unchanged.
	pub fn look_at(&mut self, target: Vec3, world_up: Vec3) {
		if let Some(q) = quat_look_at(self.translation, target, world_up) {
			self.rotation = q;
		}
	}

	/// Set rotation from Euler angles in degrees using `order`.
	///
	/// Leaves the rotation unchanged on non-finite or degenerate input.
	pub fn set_rotation_degrees(&mut self, degrees: Vec3, order: RotationOrder) {
		if let Some(q) = Quat::try_from_euler_degrees(degrees, order) {
			self.rotation = q;
		}
	}

	/// Return the current rotation as Euler angles in degrees using `order`.
	///
	/// Returns `Vec3::default()` when the quaternion is degenerate.
	pub fn rotation_degrees(&self, order: RotationOrder) -> Vec3 {
		self
			.rotation
			.try_to_euler_degrees(order)
			.unwrap_or_default()
	}

	/// Translate in local space: `right` × local-right + `up` × local-up +
	/// `forward` × local-forward.
	pub fn pan_local(&mut self, right: f32, up: f32, forward: f32) {
		let r = self.right();
		let u = self.up();
		let f = self.forward();
		self.translation.x += r.x * right + u.x * up + f.x * forward;
		self.translation.y += r.y * right + u.y * up + f.y * forward;
		self.translation.z += r.z * right + u.z * up + f.z * forward;
	}

	// ── Matrix conversion ─────────────────────────────────────────────────────

	/// Compute the row-vector TRS+shear matrix.
	///
	/// The matrix is: S × Shear × R × T (row-major, row-vector convention).
	/// Returns [`Mat4::identity`] for a default transform.
	pub fn get_matrix(&self) -> Mat4 {
		trs_shear_to_mat4(self.translation, self.rotation, self.scale, self.shear)
	}

	/// Decompose a finite non-singular affine matrix into this transform.
	///
	/// Leaves the transform unchanged on failure.
	///
	/// # Errors
	///
	/// Returns an error when `matrix` is non-finite, singular, or the
	/// decomposed quaternion is degenerate.
	pub fn set_matrix(&mut self, matrix: Mat4) -> Result<()> {
		if !matrix.is_finite() {
			return Err(Error::invalid_argument(
				"core::Transform::set_matrix requires a finite matrix",
			));
		}
		let d = matrix
			.try_decompose_affine(INVERSE_TOLERANCE)
			.ok_or_else(|| {
				Error::invalid_argument("core::Transform::set_matrix requires a non-singular affine matrix")
			})?;
		let rotation = d.rotation.try_normalized().ok_or_else(|| {
			Error::invalid_argument(
				"core::Transform::set_matrix produced a degenerate rotation quaternion",
			)
		})?;
		self.translation = d.translation;
		self.rotation = rotation;
		self.scale = d.scale;
		self.shear = d.shear;
		Ok(())
	}

	// ── Validation ────────────────────────────────────────────────────────────

	/// Return `true` when all components are finite and the rotation is
	/// normalizable.
	pub fn is_finite(&self) -> bool {
		self.translation.is_finite()
			&& self.rotation.is_finite()
			&& self.scale.is_finite()
			&& self.shear.is_finite()
			&& self.rotation.try_normalized().is_some()
	}

	/// Validate that all components are finite and well-formed.
	///
	/// # Errors
	///
	/// Returns an error for non-finite components or a degenerate rotation.
	pub fn validate(&self) -> Result<()> {
		if !self.is_finite() {
			return Err(Error::invalid_argument(
				"core::Transform components must be finite and the rotation must be normalizable",
			));
		}
		Ok(())
	}
}

// ── Six-D rotation helpers ─────────────────────────────────────────────────────

/// Pack a unit quaternion into a 6-component continuous rotation representation.
///
/// The first two columns of the rotation matrix (rows in row-vector convention)
/// are stored as `[r0x, r0y, r0z, r1x, r1y, r1z]`. The third column is
/// recovered by the cross product at decode time.
///
/// Donor: `oa::FnTransform::quaternionToSixD`.
pub fn quaternion_to_six_d(q: Quat) -> [f32; 6] {
	// Derive the rotation matrix's first two basis vectors from the quaternion.
	let r = Vec3 {
		x: 1.0,
		y: 0.0,
		z: 0.0,
	};
	let u = Vec3 {
		x: 0.0,
		y: 1.0,
		z: 0.0,
	};
	let col0 = q.try_rotate(r).unwrap_or(r);
	let col1 = q.try_rotate(u).unwrap_or(u);
	[col0.x, col0.y, col0.z, col1.x, col1.y, col1.z]
}

/// Recover a unit quaternion from a 6-component rotation representation.
///
/// Donor: `oa::FnTransform::quaternionFromSixD`.
pub fn quaternion_from_six_d(d: [f32; 6]) -> Quat {
	// Reconstruct a3 orthonormal basis and extract the quaternion.
	let a0 = Vec3 {
		x: d[0],
		y: d[1],
		z: d[2],
	};
	let a1 = Vec3 {
		x: d[3],
		y: d[4],
		z: d[5],
	};
	let Some(b0) = a0.try_normalized() else {
		return Quat::identity();
	};
	// Gram-Schmidt: make b1 orthogonal to b0.
	let dot = b0.dot(a1);
	let a1_orth = Vec3 {
		x: a1.x - b0.x * dot,
		y: a1.y - b0.y * dot,
		z: a1.z - b0.z * dot,
	};
	let Some(b1) = a1_orth.try_normalized() else {
		return Quat::identity();
	};
	let b2 = b0.cross(b1);

	// Build a row-vector rotation matrix (rows = basis vectors) and extract
	// the quaternion via the Shepperd method.
	quat_from_rotation_rows(b0, b1, b2)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Build the row-major TRS+shear matrix from components.
///
/// Row-vector convention (same as OA VLM): a point is multiplied on the left:
/// `p' = p * M`.
///
/// The construction matches OA's column ordering:
/// - Row 0: right (scaled + sheared)
/// - Row 1: up    (scaled + sheared)
/// - Row 2: forward (scaled)
/// - Row 3: translation (homogeneous W=1)
fn trs_shear_to_mat4(t: Vec3, q: Quat, s: Vec3, shear: Vec3) -> Mat4 {
	// Rotation matrix rows from quaternion.
	let unit = q.try_normalized().unwrap_or(Quat::identity());
	let r0 = unit
		.try_rotate(Vec3 {
			x: 1.0,
			y: 0.0,
			z: 0.0,
		})
		.unwrap_or(Vec3 {
			x: 1.0,
			y: 0.0,
			z: 0.0,
		});
	let r1 = unit
		.try_rotate(Vec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		})
		.unwrap_or(Vec3 {
			x: 0.0,
			y: 1.0,
			z: 0.0,
		});
	let r2 = unit
		.try_rotate(Vec3 {
			x: 0.0,
			y: 0.0,
			z: 1.0,
		})
		.unwrap_or(Vec3 {
			x: 0.0,
			y: 0.0,
			z: 1.0,
		});

	// Apply scale then shear to the rotation rows.
	// Shear (XY, XZ, YZ): row-0 += shear.x * row-1 + shear.y * row-2,
	//                      row-1 += shear.z * row-2.
	let sx = s.x;
	let sy = s.y;
	let sz = s.z;
	let sxy = shear.x;
	let sxz = shear.y;
	let syz = shear.z;

	let row0 = Vec3 {
		x: r0.x * sx + r1.x * sy * sxy + r2.x * sz * sxz,
		y: r0.y * sx + r1.y * sy * sxy + r2.y * sz * sxz,
		z: r0.z * sx + r1.z * sy * sxy + r2.z * sz * sxz,
	};
	let row1 = Vec3 {
		x: r1.x * sy + r2.x * sz * syz,
		y: r1.y * sy + r2.y * sz * syz,
		z: r1.z * sy + r2.z * sz * syz,
	};
	let row2 = Vec3 {
		x: r2.x * sz,
		y: r2.y * sz,
		z: r2.z * sz,
	};

	Mat4 {
		m: [
			[row0.x, row0.y, row0.z, 0.0],
			[row1.x, row1.y, row1.z, 0.0],
			[row2.x, row2.y, row2.z, 0.0],
			[t.x, t.y, t.z, 1.0],
		],
	}
}

/// Compute a look-at rotation quaternion from `eye` toward `target` with
/// `world_up`. Returns `None` when degenerate.
fn quat_look_at(eye: Vec3, target: Vec3, world_up: Vec3) -> Option<Quat> {
	let f = (target - eye).try_normalized()?;
	let r = f.cross(world_up).try_normalized()?;
	let backward = -f;
	let u = backward.cross(r);
	quat_from_rotation_rows(r, u, backward).try_normalized()
}

/// Extract a quaternion from three orthonormal row-basis vectors (Shepperd).
fn quat_from_rotation_rows(r: Vec3, u: Vec3, f: Vec3) -> Quat {
	// The 3×3 rotation matrix in column-major (OpenGL) is:
	//   [ r.x  u.x  f.x ]
	//   [ r.y  u.y  f.y ]
	//   [ r.z  u.z  f.z ]
	// but OA row-vector means rows ARE the basis vectors:
	//   m[0] = r, m[1] = u, m[2] = f
	// Trace = r.x + u.y + f.z
	let trace = r.x + u.y + f.z;
	if trace > 0.0 {
		let s = (trace + 1.0).sqrt() * 2.0; // s = 4w
		Quat {
			w: 0.25 * s,
			x: (f.y - u.z) / s,
			y: (r.z - f.x) / s,
			z: (u.x - r.y) / s,
		}
	} else if r.x > u.y && r.x > f.z {
		let s = (1.0 + r.x - u.y - f.z).sqrt() * 2.0; // s = 4x
		Quat {
			w: (f.y - u.z) / s,
			x: 0.25 * s,
			y: (r.y + u.x) / s,
			z: (r.z + f.x) / s,
		}
	} else if u.y > f.z {
		let s = (1.0 + u.y - r.x - f.z).sqrt() * 2.0; // s = 4y
		Quat {
			w: (r.z - f.x) / s,
			x: (r.y + u.x) / s,
			y: 0.25 * s,
			z: (u.z + f.y) / s,
		}
	} else {
		let s = (1.0 + f.z - r.x - u.y).sqrt() * 2.0; // s = 4z
		Quat {
			w: (u.x - r.y) / s,
			x: (r.z + f.x) / s,
			y: (u.z + f.y) / s,
			z: 0.25 * s,
		}
	}
}
