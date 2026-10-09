//! External contracts for Material values: AlphaMode, FlatColorMaterial,
//! UnlitMaterial, StandardSurfaceMaterial, Material, and the scene
//! material integration (SceneMaterial, SceneNode::material, validate_scene).

use oa::{
	AlphaMode, ErrorKind, FlatColorMaterial, Material, MaterialId, SceneMaterial,
	StandardSurfaceMaterial, TextureHandle, Transform, UnlitMaterial,
	render::{Scene, SceneMeshId, SceneNode, SceneNodeId, validate_scene},
};

// ── AlphaMode ─────────────────────────────────────────────────────────────────

#[test]
fn alpha_mode_default_is_opaque() {
	assert_eq!(AlphaMode::default(), AlphaMode::Opaque);
}

// ── FlatColorMaterial ─────────────────────────────────────────────────────────

#[test]
fn flat_color_default_is_opaque_white() {
	let m = FlatColorMaterial::default();
	assert_eq!(m.color, oa::Color::new(1.0, 1.0, 1.0, 1.0));
	assert_eq!(m.opacity, 1.0);
	assert!(m.validate().is_ok());
}

#[test]
fn flat_color_validate_accepts_boundary_values() {
	let m = FlatColorMaterial {
		color: oa::Color::new(0.0, 0.0, 0.0, 0.0),
		opacity: 0.0,
	};
	assert!(m.validate().is_ok());
	let m = FlatColorMaterial {
		color: oa::Color::new(1.0, 1.0, 1.0, 1.0),
		opacity: 1.0,
	};
	assert!(m.validate().is_ok());
}

#[test]
fn flat_color_validate_rejects_out_of_range() {
	let m = FlatColorMaterial {
		color: oa::Color::new(1.1, 0.0, 0.0, 0.0),
		opacity: 1.0,
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
	let m = FlatColorMaterial {
		color: oa::Color::new(1.0, -0.1, 0.0, 0.0),
		opacity: 1.0,
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn flat_color_validate_rejects_non_finite() {
	let m = FlatColorMaterial {
		color: oa::Color::new(f32::NAN, 0.0, 0.0, 1.0),
		opacity: 1.0,
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
	let m = FlatColorMaterial {
		color: oa::Color::new(1.0, 0.0, 0.0, 1.0),
		opacity: f32::INFINITY,
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

// ── UnlitMaterial ─────────────────────────────────────────────────────────────

#[test]
fn unlit_default_is_opaque_white_no_texture() {
	let m = UnlitMaterial::default();
	assert_eq!(m.base_color_texture, None);
	assert_eq!(m.tint, oa::Color::new(1.0, 1.0, 1.0, 1.0));
	assert_eq!(m.uv_scale, [1.0, 1.0]);
	assert_eq!(m.uv_offset, [0.0, 0.0]);
	assert_eq!(m.alpha_mode, AlphaMode::Opaque);
	assert!(m.validate().is_ok());
}

#[test]
fn unlit_validate_accepts_texture_handle_and_blend() {
	let m = UnlitMaterial {
		base_color_texture: Some(TextureHandle(1)),
		tint: oa::Color::new(0.5, 0.5, 0.5, 0.8),
		uv_scale: [2.0, 2.0],
		uv_offset: [-0.5, 0.25],
		alpha_mode: AlphaMode::Blend,
	};
	assert!(m.validate().is_ok());
}

#[test]
fn unlit_validate_rejects_nan_tint() {
	let m = UnlitMaterial {
		tint: oa::Color::new(f32::NAN, 1.0, 1.0, 1.0),
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn unlit_validate_rejects_infinite_uv_scale() {
	let m = UnlitMaterial {
		uv_scale: [f32::INFINITY, 1.0],
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn unlit_validate_rejects_reserved_zero_texture_handle() {
	let m = UnlitMaterial {
		base_color_texture: Some(TextureHandle(0)),
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

// ── StandardSurfaceMaterial ───────────────────────────────────────────────────

#[test]
fn standard_default_is_neutral_gray_dielectric() {
	let m = StandardSurfaceMaterial::default();
	assert_eq!(m.base_color, [0.8, 0.8, 0.8]);
	assert_eq!(m.metalness, 0.0);
	assert_eq!(m.roughness, 0.5);
	assert_eq!(m.specular_ior, 1.5);
	assert_eq!(m.emission_scale, 0.0);
	assert_eq!(m.coat_weight, 0.0);
	assert_eq!(m.sheen_weight, 0.0);
	assert_eq!(m.subsurface_weight, 0.0);
	assert_eq!(m.opacity, 1.0);
	assert_eq!(m.alpha_mode, AlphaMode::Opaque);
	assert_eq!(m.uv_scale, [1.0, 1.0]);
	assert_eq!(m.uv_offset, [0.0, 0.0]);
	// All texture slots are None.
	assert!(m.base_color_texture.is_none());
	assert!(m.orm_texture.is_none());
	assert!(m.normal_texture.is_none());
	assert!(m.emission_texture.is_none());
	assert!(m.coat_normal_texture.is_none());
	assert!(m.opacity_texture.is_none());
	assert!(m.validate().is_ok());
}

#[test]
fn standard_validate_accepts_fully_populated_material() {
	let m = StandardSurfaceMaterial {
		base_color: [0.2, 0.4, 0.8],
		base_color_texture: Some(TextureHandle(1)),
		metalness: 1.0,
		roughness: 0.0,
		orm_texture: Some(TextureHandle(2)),
		normal_texture: Some(TextureHandle(3)),
		normal_scale: 2.0,
		specular_ior: 2.4,
		specular_tint: [0.9, 0.9, 1.0],
		emission_color: [1.0, 0.8, 0.0],
		emission_scale: 5.0,
		emission_texture: Some(TextureHandle(4)),
		coat_weight: 1.0,
		coat_roughness: 0.05,
		coat_ior: 1.5,
		coat_normal_texture: Some(TextureHandle(5)),
		sheen_weight: 0.5,
		sheen_color: [1.0, 1.0, 1.0],
		sheen_roughness: 0.3,
		subsurface_weight: 0.8,
		subsurface_color: [0.9, 0.7, 0.6],
		subsurface_radius: [0.5, 0.3, 0.2],
		opacity: 0.9,
		opacity_texture: Some(TextureHandle(6)),
		alpha_mode: AlphaMode::Blend,
		alpha_cutoff: 0.5,
		uv_scale: [2.0, 2.0],
		uv_offset: [0.1, 0.1],
	};
	assert!(m.validate().is_ok());
}

#[test]
fn standard_validate_rejects_metalness_above_one() {
	let m = StandardSurfaceMaterial {
		metalness: 1.001,
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn standard_validate_rejects_roughness_below_zero() {
	let m = StandardSurfaceMaterial {
		roughness: -0.01,
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn standard_validate_rejects_zero_ior() {
	let m = StandardSurfaceMaterial {
		specular_ior: 0.0,
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn standard_validate_rejects_negative_ior() {
	let m = StandardSurfaceMaterial {
		specular_ior: -1.5,
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn standard_validate_rejects_negative_emission_scale() {
	let m = StandardSurfaceMaterial {
		emission_scale: -0.1,
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn standard_validate_rejects_nan_base_color() {
	let m = StandardSurfaceMaterial {
		base_color: [f32::NAN, 0.0, 0.0],
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn standard_validate_rejects_negative_subsurface_radius() {
	let m = StandardSurfaceMaterial {
		subsurface_radius: [1.0, -0.1, 1.0],
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

#[test]
fn standard_validate_rejects_reserved_zero_texture_handle() {
	let m = StandardSurfaceMaterial {
		normal_texture: Some(TextureHandle(0)),
		..Default::default()
	};
	assert_eq!(m.validate().unwrap_err().kind(), ErrorKind::InvalidArgument);
}

// ── Material enum ─────────────────────────────────────────────────────────────

#[test]
fn material_default_is_standard_neutral_gray() {
	assert!(matches!(Material::default(), Material::Standard(_)));
}

#[test]
fn material_validate_delegates_to_inner() {
	let bad = Material::FlatColor(FlatColorMaterial {
		color: oa::Color::new(2.0, 0.0, 0.0, 1.0),
		opacity: 1.0,
	});
	assert_eq!(
		bad.validate().unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);

	let ok = Material::Unlit(UnlitMaterial::default());
	assert!(ok.validate().is_ok());
}

// ── Scene material integration ────────────────────────────────────────────────

fn minimal_scene_with_material(mat_id: MaterialId, mat: Material) -> Scene {
	Scene {
		meshes: vec![],
		materials: vec![SceneMaterial {
			id: mat_id,
			name: "m".into(),
			data: mat,
		}],
		nodes: vec![SceneNode {
			id: SceneNodeId(1),
			name: "n".into(),
			parent: SceneNodeId(0),
			mesh: SceneMeshId(0),
			material: Some(mat_id),
			local_transform: Transform::default(),
			visible: true,
		}],
	}
}

#[test]
fn scene_accepts_node_with_valid_material_ref() {
	let scene = minimal_scene_with_material(
		MaterialId(1),
		Material::Standard(StandardSurfaceMaterial::default()),
	);
	assert!(validate_scene(&scene).is_ok());
}

#[test]
fn scene_accepts_node_with_no_material() {
	let scene = Scene {
		meshes: vec![],
		materials: vec![],
		nodes: vec![SceneNode {
			id: SceneNodeId(1),
			name: "n".into(),
			parent: SceneNodeId(0),
			mesh: SceneMeshId(0),
			material: None,
			local_transform: Transform::default(),
			visible: true,
		}],
	};
	assert!(validate_scene(&scene).is_ok());
}

#[test]
fn scene_rejects_duplicate_material_ids() {
	let scene = Scene {
		meshes: vec![],
		materials: vec![
			SceneMaterial {
				id: MaterialId(1),
				name: "a".into(),
				data: Material::default(),
			},
			SceneMaterial {
				id: MaterialId(1),
				name: "b".into(),
				data: Material::default(),
			},
		],
		nodes: vec![],
	};
	assert_eq!(
		validate_scene(&scene).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
}

#[test]
fn scene_rejects_zero_material_id() {
	let scene = Scene {
		meshes: vec![],
		materials: vec![SceneMaterial {
			id: MaterialId(0),
			name: "zero".into(),
			data: Material::default(),
		}],
		nodes: vec![],
	};
	assert_eq!(
		validate_scene(&scene).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
}

#[test]
fn scene_rejects_node_referencing_missing_material() {
	let scene = Scene {
		meshes: vec![],
		materials: vec![],
		nodes: vec![SceneNode {
			id: SceneNodeId(1),
			name: "n".into(),
			parent: SceneNodeId(0),
			mesh: SceneMeshId(0),
			material: Some(MaterialId(99)),
			local_transform: Transform::default(),
			visible: true,
		}],
	};
	assert_eq!(
		validate_scene(&scene).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
}

#[test]
fn scene_rejects_invalid_material_parameters() {
	let bad_mat = Material::FlatColor(FlatColorMaterial {
		color: oa::Color::new(99.0, 0.0, 0.0, 1.0), // out of range
		opacity: 1.0,
	});
	let scene = minimal_scene_with_material(MaterialId(1), bad_mat);
	assert_eq!(
		validate_scene(&scene).unwrap_err().kind(),
		ErrorKind::InvalidArgument
	);
}

#[test]
fn scene_material_compile_is_unaffected_by_material_presence() {
	// compile_scene bakes mesh geometry; material assignment doesn't change the
	// compiled MeshData (materials are GPU-side binding policy).
	use oa::render::{SceneMesh, create_quad};
	let quad = create_quad(2.0, 2.0).expect("quad");
	let mesh_id = SceneMeshId(1);
	let mat_id = MaterialId(1);
	let scene = Scene {
		meshes: vec![SceneMesh {
			id: mesh_id,
			name: "q".into(),
			data: quad.clone(),
		}],
		materials: vec![SceneMaterial {
			id: mat_id,
			name: "flat".into(),
			data: Material::FlatColor(FlatColorMaterial::default()),
		}],
		nodes: vec![SceneNode {
			id: SceneNodeId(1),
			name: "n".into(),
			parent: SceneNodeId(0),
			mesh: mesh_id,
			material: Some(mat_id),
			local_transform: Transform::default(),
			visible: true,
		}],
	};
	let compiled = oa::render::compile_scene(&scene).expect("compile");
	assert_eq!(compiled.vertices.len(), quad.vertices.len());
	assert_eq!(compiled.indices, quad.indices);
}
