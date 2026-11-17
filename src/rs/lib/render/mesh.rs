//! Stateless mesh math and geometry generation.
//!
//! Operates on [`MeshData`] and [`MeshVertex`]. All construction functions
//! return new values; mutation operations take `&mut MeshData` and leave the
//! mesh unchanged on failure.
//!
//! Donor: `oacpp/source/cpp/include/oa/render/fnMesh.h` (`oa::FnMesh`).
//! Adaptation: C++ static namespace members become module functions;
//! trivial size/count queries (`getVertexCount` etc.) are omitted because
//! callers use `.vertices.len()` and `.indices.len()` directly.

use crate::{
	Error, Result,
	vlm::{Mat4, Vec3},
};

use super::scene::{MeshData, MeshVertex};

// ── Aabb ──────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box.
///
/// Donor: `oa::Aabb` in `oacpp/source/cpp/include/oa/render/fnMesh.h`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Aabb {
	/// Minimum corner (smallest X, Y, Z).
	pub min: Vec3,
	/// Maximum corner (largest X, Y, Z).
	pub max: Vec3,
}

impl Aabb {
	/// Return the center of the box.
	pub fn center(self) -> Vec3 {
		Vec3 {
			x: (self.min.x + self.max.x) * 0.5,
			y: (self.min.y + self.max.y) * 0.5,
			z: (self.min.z + self.max.z) * 0.5,
		}
	}

	/// Return the per-axis extent (max − min).
	pub fn extent(self) -> Vec3 {
		Vec3 {
			x: self.max.x - self.min.x,
			y: self.max.y - self.min.y,
			z: self.max.z - self.min.z,
		}
	}

	/// Return the signed volume of the box.
	pub fn volume(self) -> f32 {
		let e = self.extent();
		e.x * e.y * e.z
	}

	/// Return `true` when `point` is within or on the surface of the box.
	pub fn contains(self, point: Vec3) -> bool {
		point.x >= self.min.x
			&& point.x <= self.max.x
			&& point.y >= self.min.y
			&& point.y <= self.max.y
			&& point.z >= self.min.z
			&& point.z <= self.max.z
	}
}

// ── Geometry generation ───────────────────────────────────────────────────────

/// Create a centered quad (2 triangles, 4 vertices) with top-origin image UVs.
///
/// UV V=0 is the source-image top and world +Y is the quad top.
/// For image/video viewing, pass the pixel dimensions (e.g. 1920, 1080).
///
/// `width` and `height` must be finite and positive.
///
/// Donor: `oa::FnMesh::createQuad`.
///
/// # Errors
///
/// Returns an error for non-finite or non-positive dimensions.
pub fn create_quad(width: f32, height: f32) -> Result<MeshData> {
	if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
		return Err(Error::invalid_argument(
			"render::create_quad requires finite positive width and height",
		));
	}
	let hx = width * 0.5;
	let hy = height * 0.5;
	let n = Vec3 {
		x: 0.0,
		y: 0.0,
		z: 1.0,
	};
	let c = crate::vlm::Vec4 {
		x: 1.0,
		y: 1.0,
		z: 1.0,
		w: 1.0,
	};
	let mut mesh = MeshData {
		vertices: vec![
			MeshVertex {
				position: Vec3 {
					x: -hx,
					y: hy,
					z: 0.0,
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 0.0, y: 0.0 },
				color: c,
			},
			MeshVertex {
				position: Vec3 {
					x: hx,
					y: hy,
					z: 0.0,
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 1.0, y: 0.0 },
				color: c,
			},
			MeshVertex {
				position: Vec3 {
					x: hx,
					y: -hy,
					z: 0.0,
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 1.0, y: 1.0 },
				color: c,
			},
			MeshVertex {
				position: Vec3 {
					x: -hx,
					y: -hy,
					z: 0.0,
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 0.0, y: 1.0 },
				color: c,
			},
		],
		indices: vec![0, 1, 2, 0, 2, 3],
		bounds: Aabb::default(),
		bounds_dirty: true,
	};
	compute_bounds(&mut mesh);
	Ok(mesh)
}

/// Create a fullscreen NDC quad (2 triangles, 4 vertices) in `[-1, 1]` XY
/// with top-origin image UVs. Used for post-processing passes.
///
/// Donor: `oa::FnMesh::createFullscreenQuad`.
pub fn create_fullscreen_quad() -> MeshData {
	let n = Vec3 {
		x: 0.0,
		y: 0.0,
		z: 1.0,
	};
	let c = crate::vlm::Vec4 {
		x: 1.0,
		y: 1.0,
		z: 1.0,
		w: 1.0,
	};
	let mut mesh = MeshData {
		vertices: vec![
			MeshVertex {
				position: Vec3 {
					x: -1.0,
					y: 1.0,
					z: 0.0,
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 0.0, y: 0.0 },
				color: c,
			},
			MeshVertex {
				position: Vec3 {
					x: 1.0,
					y: 1.0,
					z: 0.0,
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 1.0, y: 0.0 },
				color: c,
			},
			MeshVertex {
				position: Vec3 {
					x: 1.0,
					y: -1.0,
					z: 0.0,
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 1.0, y: 1.0 },
				color: c,
			},
			MeshVertex {
				position: Vec3 {
					x: -1.0,
					y: -1.0,
					z: 0.0,
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 0.0, y: 1.0 },
				color: c,
			},
		],
		indices: vec![0, 1, 2, 0, 2, 3],
		bounds: Aabb {
			min: Vec3 {
				x: -1.0,
				y: -1.0,
				z: 0.0,
			},
			max: Vec3 {
				x: 1.0,
				y: 1.0,
				z: 0.0,
			},
		},
		bounds_dirty: false,
	};
	compute_bounds(&mut mesh);
	mesh
}

/// Create a 3D axis-aligned cube (12 triangles, 24 vertices, size × size × size).
///
/// Each face has its own four vertices with an outward normal and per-face UVs.
/// `size` must be finite and positive.
///
/// Donor: `oa::FnMesh::createCube`.
///
/// # Errors
///
/// Returns an error for non-finite or non-positive size.
pub fn create_cube(size: f32) -> Result<MeshData> {
	if !size.is_finite() || size <= 0.0 {
		return Err(Error::invalid_argument(
			"render::create_cube requires a finite positive size",
		));
	}
	let h = size * 0.5;
	let c = crate::vlm::Vec4 {
		x: 1.0,
		y: 1.0,
		z: 1.0,
		w: 1.0,
	};
	// 6 faces × 4 vertices each, with explicit normals per face.
	// Winding: CCW when viewed from outside (front-face default).
	let face = |nx: f32, ny: f32, nz: f32, p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]| {
		let n = Vec3 {
			x: nx,
			y: ny,
			z: nz,
		};
		[
			MeshVertex {
				position: Vec3 {
					x: p0[0],
					y: p0[1],
					z: p0[2],
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 0.0, y: 0.0 },
				color: c,
			},
			MeshVertex {
				position: Vec3 {
					x: p1[0],
					y: p1[1],
					z: p1[2],
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 1.0, y: 0.0 },
				color: c,
			},
			MeshVertex {
				position: Vec3 {
					x: p2[0],
					y: p2[1],
					z: p2[2],
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 1.0, y: 1.0 },
				color: c,
			},
			MeshVertex {
				position: Vec3 {
					x: p3[0],
					y: p3[1],
					z: p3[2],
				},
				normal: n,
				uv: crate::vlm::Vec2 { x: 0.0, y: 1.0 },
				color: c,
			},
		]
	};
	let vertices: Vec<MeshVertex> = [
		// +Z front
		face(
			0.0,
			0.0,
			1.0,
			[-h, h, h],
			[h, h, h],
			[h, -h, h],
			[-h, -h, h],
		),
		// -Z back
		face(
			0.0,
			0.0,
			-1.0,
			[h, h, -h],
			[-h, h, -h],
			[-h, -h, -h],
			[h, -h, -h],
		),
		// +Y top
		face(
			0.0,
			1.0,
			0.0,
			[-h, h, -h],
			[h, h, -h],
			[h, h, h],
			[-h, h, h],
		),
		// -Y bottom
		face(
			0.0,
			-1.0,
			0.0,
			[-h, -h, h],
			[h, -h, h],
			[h, -h, -h],
			[-h, -h, -h],
		),
		// +X right
		face(
			1.0,
			0.0,
			0.0,
			[h, h, h],
			[h, h, -h],
			[h, -h, -h],
			[h, -h, h],
		),
		// -X left
		face(
			-1.0,
			0.0,
			0.0,
			[-h, h, -h],
			[-h, h, h],
			[-h, -h, h],
			[-h, -h, -h],
		),
	]
	.into_iter()
	.flatten()
	.collect();

	// 6 faces × 2 triangles × 3 indices, base-offset per face.
	let indices: Vec<u32> = (0u32..6)
		.flat_map(|face| {
			let b = face * 4;
			[b, b + 1, b + 2, b, b + 2, b + 3]
		})
		.collect();

	let mut mesh = MeshData {
		vertices,
		indices,
		bounds: Aabb {
			min: Vec3 {
				x: -h,
				y: -h,
				z: -h,
			},
			max: Vec3 { x: h, y: h, z: h },
		},
		bounds_dirty: false,
	};
	compute_bounds(&mut mesh);
	Ok(mesh)
}

// ── Bounds ────────────────────────────────────────────────────────────────────

/// Recompute and store the tight axis-aligned bounding box for `mesh`.
///
/// Clears `bounds_dirty`. Empty meshes produce an all-zero `Aabb`.
///
/// Donor: `oa::FnMesh::computeBounds`.
pub fn compute_bounds(mesh: &mut MeshData) {
	mesh.bounds_dirty = false;
	if mesh.vertices.is_empty() {
		mesh.bounds = Aabb::default();
		return;
	}
	let first = mesh.vertices[0].position;
	let mut mn = first;
	let mut mx = first;
	for v in &mesh.vertices {
		let p = v.position;
		if p.x < mn.x {
			mn.x = p.x;
		}
		if p.y < mn.y {
			mn.y = p.y;
		}
		if p.z < mn.z {
			mn.z = p.z;
		}
		if p.x > mx.x {
			mx.x = p.x;
		}
		if p.y > mx.y {
			mx.y = p.y;
		}
		if p.z > mx.z {
			mx.z = p.z;
		}
	}
	mesh.bounds = Aabb { min: mn, max: mx };
}

// ── Transforms ────────────────────────────────────────────────────────────────

/// Apply an affine matrix to all vertex positions and normals.
///
/// The mesh is left unchanged when `matrix` is non-finite, singular, or
/// produces non-finite vertex positions. On success, `bounds_dirty` is set.
///
/// Donor: `oa::FnMesh::transform`.
///
/// # Errors
///
/// Returns an error when `matrix` is non-finite, singular (no normal matrix),
/// or produces non-finite vertex positions.
pub fn transform(mesh: &mut MeshData, matrix: Mat4) -> Result<()> {
	if !matrix.is_finite() {
		return Err(Error::invalid_argument(
			"render::transform requires a finite transformation matrix",
		));
	}
	let normal_matrix = matrix
		.try_normal_matrix(crate::vlm::INVERSE_TOLERANCE)
		.ok_or_else(|| {
			Error::invalid_argument("render::transform requires a non-singular transformation matrix")
		})?;

	// Compute transformed positions first; verify before writing.
	let new_positions: Vec<Vec3> = mesh
		.vertices
		.iter()
		.map(|v| matrix.transform_point(v.position))
		.collect();

	if new_positions.iter().any(|p| !p.is_finite()) {
		return Err(Error::invalid_argument(
			"render::transform produced non-finite vertex positions",
		));
	}

	for (v, pos) in mesh.vertices.iter_mut().zip(new_positions) {
		v.position = pos;
		v.normal = v.normal * normal_matrix;
	}
	mesh.bounds_dirty = true;
	Ok(())
}

/// Translate all vertex positions in-place.
///
/// Always succeeds; bounds are marked dirty.
///
/// Donor: `oa::FnMesh::translate`.
pub fn translate(mesh: &mut MeshData, offset: Vec3) {
	for v in &mut mesh.vertices {
		v.position.x += offset.x;
		v.position.y += offset.y;
		v.position.z += offset.z;
	}
	// Shift the cached bounds when valid.
	if !mesh.bounds_dirty {
		mesh.bounds.min.x += offset.x;
		mesh.bounds.min.y += offset.y;
		mesh.bounds.min.z += offset.z;
		mesh.bounds.max.x += offset.x;
		mesh.bounds.max.y += offset.y;
		mesh.bounds.max.z += offset.z;
	}
}

/// Scale all vertex positions and normals.
///
/// Zero or non-finite scale components fail without modifying the mesh.
///
/// Donor: `oa::FnMesh::scale`.
///
/// # Errors
///
/// Returns an error when any scale component is zero or non-finite.
pub fn scale(mesh: &mut MeshData, s: Vec3) -> Result<()> {
	if !s.is_finite() || s.x == 0.0 || s.y == 0.0 || s.z == 0.0 {
		return Err(Error::invalid_argument(
			"render::scale requires finite non-zero scale components",
		));
	}
	// Inverse scale for normals (transpose-inverse of a diagonal = 1/s).
	let inv = Vec3 {
		x: 1.0 / s.x,
		y: 1.0 / s.y,
		z: 1.0 / s.z,
	};
	for v in &mut mesh.vertices {
		v.position.x *= s.x;
		v.position.y *= s.y;
		v.position.z *= s.z;
		// Renormalize after applying the inverse-transpose.
		let nx = v.normal.x * inv.x;
		let ny = v.normal.y * inv.y;
		let nz = v.normal.z * inv.z;
		let len = (nx * nx + ny * ny + nz * nz).sqrt();
		if len > f32::EPSILON {
			v.normal = Vec3 {
				x: nx / len,
				y: ny / len,
				z: nz / len,
			};
		}
	}
	mesh.bounds_dirty = true;
	Ok(())
}

/// Flip all vertex UVs vertically (V ← 1 − V).
///
/// Used to convert explicitly bottom-origin source content to top-origin.
/// Never fails.
///
/// Donor: `oa::FnMesh::flipUvsY`.
pub fn flip_uvs_y(mesh: &mut MeshData) {
	for v in &mut mesh.vertices {
		v.uv.y = 1.0 - v.uv.y;
	}
}
