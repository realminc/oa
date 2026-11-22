//! External contracts for render mesh operations and the Aabb value.
//!
//! Covers: Aabb, create_quad, create_fullscreen_quad, create_cube,
//! compute_bounds, transform, translate, scale, flip_uvs_y.

use oa::{
	ErrorKind,
	render::{
		Aabb, MeshData, MeshVertex, compute_bounds, create_cube, create_fullscreen_quad, create_quad,
		flip_uvs_y, scale, transform, translate,
	},
	vlm::{Mat4, Vec2, Vec3, Vec4},
};

// ── Aabb ──────────────────────────────────────────────────────────────────────

#[test]
fn aabb_center_is_midpoint() {
	let a = Aabb {
		min: Vec3 {
			x: -2.0,
			y: 0.0,
			z: -1.0,
		},
		max: Vec3 {
			x: 2.0,
			y: 4.0,
			z: 1.0,
		},
	};
	let c = a.center();
	assert!((c.x - 0.0).abs() < 1e-6);
	assert!((c.y - 2.0).abs() < 1e-6);
	assert!((c.z - 0.0).abs() < 1e-6);
}

#[test]
fn aabb_extent_and_volume() {
	let a = Aabb {
		min: Vec3 {
			x: 0.0,
			y: 0.0,
			z: 0.0,
		},
		max: Vec3 {
			x: 2.0,
			y: 3.0,
			z: 4.0,
		},
	};
	let e = a.extent();
	assert!((e.x - 2.0).abs() < 1e-6);
	assert!((e.y - 3.0).abs() < 1e-6);
	assert!((e.z - 4.0).abs() < 1e-6);
	assert!((a.volume() - 24.0).abs() < 1e-4);
}

#[test]
fn aabb_contains_boundary_and_outside() {
	let a = Aabb {
		min: Vec3 {
			x: -1.0,
			y: -1.0,
			z: -1.0,
		},
		max: Vec3 {
			x: 1.0,
			y: 1.0,
			z: 1.0,
		},
	};
	// corners are inside
	assert!(a.contains(Vec3 {
		x: -1.0,
		y: -1.0,
		z: -1.0
	}));
	assert!(a.contains(Vec3 {
		x: 1.0,
		y: 1.0,
		z: 1.0
	}));
	assert!(a.contains(Vec3 {
		x: 0.0,
		y: 0.0,
		z: 0.0
	}));
	// outside
	assert!(!a.contains(Vec3 {
		x: 1.1,
		y: 0.0,
		z: 0.0
	}));
	assert!(!a.contains(Vec3 {
		x: 0.0,
		y: -1.1,
		z: 0.0
	}));
}

#[test]
fn aabb_default_is_zero() {
	let a = Aabb::default();
	assert_eq!(
		a.min,
		Vec3 {
			x: 0.0,
			y: 0.0,
			z: 0.0
		}
	);
	assert_eq!(
		a.max,
		Vec3 {
			x: 0.0,
			y: 0.0,
			z: 0.0
		}
	);
	assert!((a.volume() - 0.0).abs() < 1e-6);
}

// ── create_quad ───────────────────────────────────────────────────────────────

#[test]
fn create_quad_has_donor_winding_and_top_origin_uvs() {
	let quad = create_quad(2.0, 4.0).expect("valid quad");
	assert_eq!(quad.indices, vec![0, 1, 2, 0, 2, 3]);
	// vertex 0: top-left
	assert_eq!(
		quad.vertices[0].position,
		Vec3 {
			x: -1.0,
			y: 2.0,
			z: 0.0
		}
	);
	assert_eq!(quad.vertices[0].uv, Vec2 { x: 0.0, y: 0.0 });
	// vertex 2: bottom-right (UV bottom-right)
	assert_eq!(quad.vertices[2].uv, Vec2 { x: 1.0, y: 1.0 });
}

#[test]
fn create_quad_rejects_invalid_dimensions() {
	for (w, h) in [
		(0.0, 1.0),
		(1.0, 0.0),
		(-1.0, 1.0),
		(f32::NAN, 1.0),
		(1.0, f32::INFINITY),
	] {
		assert_eq!(
			create_quad(w, h).unwrap_err().kind(),
			ErrorKind::InvalidArgument,
			"accepted w={w} h={h}"
		);
	}
}

#[test]
fn create_quad_has_up_to_date_bounds() {
	let quad = create_quad(4.0, 2.0).expect("valid quad");
	assert!(!quad.bounds_dirty);
	// bounds are tight: ±2 in X, ±1 in Y, 0 in Z
	assert!((quad.bounds.min.x - -2.0).abs() < 1e-6);
	assert!((quad.bounds.max.x - 2.0).abs() < 1e-6);
	assert!((quad.bounds.min.y - -1.0).abs() < 1e-6);
	assert!((quad.bounds.max.y - 1.0).abs() < 1e-6);
	assert!((quad.bounds.min.z - 0.0).abs() < 1e-6);
	assert!((quad.bounds.max.z - 0.0).abs() < 1e-6);
}

// ── create_fullscreen_quad ────────────────────────────────────────────────────

#[test]
fn create_fullscreen_quad_spans_ndc() {
	let q = create_fullscreen_quad();
	assert_eq!(q.indices, vec![0, 1, 2, 0, 2, 3]);
	assert_eq!(q.vertices.len(), 4);
	// min/max positions must be ±1 in XY
	let xs: Vec<f32> = q.vertices.iter().map(|v| v.position.x).collect();
	let ys: Vec<f32> = q.vertices.iter().map(|v| v.position.y).collect();
	assert!(xs.iter().any(|&x| (x - -1.0).abs() < 1e-6));
	assert!(xs.iter().any(|&x| (x - 1.0).abs() < 1e-6));
	assert!(ys.iter().any(|&y| (y - -1.0).abs() < 1e-6));
	assert!(ys.iter().any(|&y| (y - 1.0).abs() < 1e-6));
	assert!(!q.bounds_dirty);
}

// ── create_cube ───────────────────────────────────────────────────────────────

#[test]
fn create_cube_has_correct_topology() {
	let cube = create_cube(2.0).expect("valid cube");
	assert_eq!(cube.vertices.len(), 24, "6 faces × 4 vertices");
	assert_eq!(cube.indices.len(), 36, "6 faces × 2 triangles × 3 indices");
	assert!(!cube.bounds_dirty);
}

#[test]
fn create_cube_bounds_are_half_size() {
	let cube = create_cube(2.0).expect("valid cube");
	assert!((cube.bounds.min.x - -1.0).abs() < 1e-6);
	assert!((cube.bounds.max.x - 1.0).abs() < 1e-6);
	assert!((cube.bounds.min.y - -1.0).abs() < 1e-6);
	assert!((cube.bounds.max.y - 1.0).abs() < 1e-6);
	assert!((cube.bounds.min.z - -1.0).abs() < 1e-6);
	assert!((cube.bounds.max.z - 1.0).abs() < 1e-6);
}

#[test]
fn create_cube_rejects_invalid_size() {
	assert_eq!(
		create_cube(0.0).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
	assert_eq!(
		create_cube(-1.0).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
	assert_eq!(
		create_cube(f32::NAN).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
}

#[test]
fn create_cube_face_normals_are_unit() {
	let cube = create_cube(1.0).expect("valid cube");
	for v in &cube.vertices {
		let len = (v.normal.x * v.normal.x + v.normal.y * v.normal.y + v.normal.z * v.normal.z).sqrt();
		assert!((len - 1.0).abs() < 1e-5, "normal not unit: {:?}", v.normal);
	}
}

// ── compute_bounds ────────────────────────────────────────────────────────────

#[test]
fn compute_bounds_updates_dirty_flag_and_aabb() {
	let mut mesh = MeshData {
		vertices: vec![
			MeshVertex {
				position: Vec3 {
					x: -3.0,
					y: 1.0,
					z: 0.0,
				},
				normal: Vec3 {
					x: 0.0,
					y: 0.0,
					z: 1.0,
				},
				uv: Vec2::default(),
				color: Vec4 {
					x: 1.0,
					y: 1.0,
					z: 1.0,
					w: 1.0,
				},
			},
			MeshVertex {
				position: Vec3 {
					x: 5.0,
					y: -2.0,
					z: 4.0,
				},
				normal: Vec3 {
					x: 0.0,
					y: 0.0,
					z: 1.0,
				},
				uv: Vec2::default(),
				color: Vec4 {
					x: 1.0,
					y: 1.0,
					z: 1.0,
					w: 1.0,
				},
			},
		],
		indices: vec![],
		..Default::default()
	};
	assert!(mesh.bounds_dirty);
	compute_bounds(&mut mesh);
	assert!(!mesh.bounds_dirty);
	assert!((mesh.bounds.min.x - -3.0).abs() < 1e-6);
	assert!((mesh.bounds.max.x - 5.0).abs() < 1e-6);
	assert!((mesh.bounds.min.y - -2.0).abs() < 1e-6);
	assert!((mesh.bounds.max.y - 1.0).abs() < 1e-6);
	assert!((mesh.bounds.min.z - 0.0).abs() < 1e-6);
	assert!((mesh.bounds.max.z - 4.0).abs() < 1e-6);
}

#[test]
fn compute_bounds_empty_mesh_produces_zero_aabb() {
	let mut mesh = MeshData::default();
	compute_bounds(&mut mesh);
	assert!(!mesh.bounds_dirty);
	assert_eq!(mesh.bounds, Aabb::default());
}

// ── translate ─────────────────────────────────────────────────────────────────

#[test]
fn translate_shifts_positions_and_updates_cached_bounds() {
	let mut mesh = create_quad(2.0, 2.0).expect("quad");
	let offset = Vec3 {
		x: 10.0,
		y: -5.0,
		z: 3.0,
	};
	translate(&mut mesh, offset);
	assert!(
		!mesh.bounds_dirty,
		"bounds should remain valid after translate"
	);
	assert!((mesh.bounds.min.x - 9.0).abs() < 1e-5);
	assert!((mesh.bounds.max.x - 11.0).abs() < 1e-5);
	for v in &mesh.vertices {
		assert!(v.position.z == 3.0);
	}
}

// ── scale ─────────────────────────────────────────────────────────────────────

#[test]
fn scale_doubles_extents_and_normalizes_normals() {
	let mut mesh = create_cube(1.0).expect("cube");
	scale(
		&mut mesh,
		Vec3 {
			x: 2.0,
			y: 2.0,
			z: 2.0,
		},
	)
	.expect("uniform scale");
	assert!(mesh.bounds_dirty);
	compute_bounds(&mut mesh);
	// cube was ±0.5, now ±1.0
	assert!((mesh.bounds.min.x - -1.0).abs() < 1e-5);
	assert!((mesh.bounds.max.x - 1.0).abs() < 1e-5);
	// normals must remain unit
	for v in &mesh.vertices {
		let len = (v.normal.x * v.normal.x + v.normal.y * v.normal.y + v.normal.z * v.normal.z).sqrt();
		assert!((len - 1.0).abs() < 1e-5, "normal not unit after scale");
	}
}

#[test]
fn scale_rejects_zero_and_nan() {
	let mut mesh = create_quad(1.0, 1.0).expect("quad");
	assert_eq!(
		scale(
			&mut mesh,
			Vec3 {
				x: 0.0,
				y: 1.0,
				z: 1.0
			}
		)
		.unwrap_err()
		.kind(),
		ErrorKind::InvalidArgument
	);
	assert_eq!(
		scale(
			&mut mesh,
			Vec3 {
				x: f32::NAN,
				y: 1.0,
				z: 1.0
			}
		)
		.unwrap_err()
		.kind(),
		ErrorKind::InvalidArgument
	);
}

// ── transform ─────────────────────────────────────────────────────────────────

#[test]
fn transform_identity_is_a_no_op() {
	let original = create_quad(2.0, 2.0).expect("quad");
	let mut mesh = original.clone();
	transform(&mut mesh, Mat4::identity()).expect("identity transform");
	for (a, b) in mesh.vertices.iter().zip(original.vertices.iter()) {
		assert!((a.position.x - b.position.x).abs() < 1e-5);
		assert!((a.position.y - b.position.y).abs() < 1e-5);
	}
}

#[test]
fn transform_rejects_singular_matrix() {
	let mut mesh = create_quad(1.0, 1.0).expect("quad");
	let singular = Mat4 { m: [[0.0; 4]; 4] };
	assert_eq!(
		transform(&mut mesh, singular).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
}

#[test]
fn transform_rejects_non_finite_matrix() {
	let mut mesh = create_quad(1.0, 1.0).expect("quad");
	let mut bad = Mat4::identity();
	bad.m[0][0] = f32::NAN;
	assert_eq!(
		transform(&mut mesh, bad).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
}

// ── flip_uvs_y ────────────────────────────────────────────────────────────────

#[test]
fn flip_uvs_y_inverts_v_coordinate() {
	let mut mesh = create_quad(1.0, 1.0).expect("quad");
	let before: Vec<f32> = mesh.vertices.iter().map(|v| v.uv.y).collect();
	flip_uvs_y(&mut mesh);
	let after: Vec<f32> = mesh.vertices.iter().map(|v| v.uv.y).collect();
	for (b, a) in before.iter().zip(after.iter()) {
		assert!(
			(b + a - 1.0).abs() < 1e-6,
			"V={b} did not flip to {}",
			1.0 - b
		);
	}
	// Double flip is identity
	flip_uvs_y(&mut mesh);
	let restored: Vec<f32> = mesh.vertices.iter().map(|v| v.uv.y).collect();
	for (orig, rest) in before.iter().zip(restored.iter()) {
		assert!((orig - rest).abs() < 1e-6);
	}
}

// ── MeshData default has bounds_dirty ────────────────────────────────────────

#[test]
fn meshdata_default_is_dirty() {
	let mesh = MeshData::default();
	assert!(mesh.bounds_dirty);
	assert!(mesh.vertices.is_empty());
	assert!(mesh.indices.is_empty());
}
