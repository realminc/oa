//! CPU semantic mesh, material, and scene snapshots, independent of renderer
//! storage.

use std::collections::{HashMap, HashSet};

use crate::{
	Error, Result,
	core::transform::Transform,
	vlm::{Mat4, Vec2, Vec3, Vec4},
};

use super::{
	material::{Material, MaterialId},
	mesh::Aabb,
};

/// Stable identity of one mesh in a [`Scene`]. Zero is invalid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SceneMeshId(pub u64);

/// Stable identity of one node in a [`Scene`]. Zero is invalid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SceneNodeId(pub u64);

/// CPU vertex contract used by the first scene compiler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshVertex {
	pub position: Vec3,
	pub normal: Vec3,
	pub uv: Vec2,
	pub color: Vec4,
}

/// Indexed triangle-list mesh snapshot.
///
/// `bounds` and `bounds_dirty` mirror the OA donor fields. Constructed values
/// from `mesh::create_*` have up-to-date bounds; directly assembled `MeshData`
/// values start with `bounds_dirty = true` and `bounds = Aabb::default()`.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshData {
	pub vertices: Vec<MeshVertex>,
	pub indices: Vec<u32>,
	/// Cached tight AABB; valid when `bounds_dirty` is `false`.
	pub bounds: Aabb,
	/// Set to `true` when vertices have changed since the last `compute_bounds`.
	pub bounds_dirty: bool,
}

impl Default for MeshData {
	fn default() -> Self {
		Self {
			vertices: Vec::new(),
			indices: Vec::new(),
			bounds: Aabb::default(),
			bounds_dirty: true,
		}
	}
}

/// Named immutable mesh value in a semantic scene.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneMesh {
	pub id: SceneMeshId,
	pub name: String,
	pub data: MeshData,
}

/// Named material entry in a semantic scene.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneMaterial {
	/// Unique non-zero material identity within this scene.
	pub id: MaterialId,
	pub name: String,
	pub data: Material,
}

/// One hierarchy node that optionally instances a mesh.
///
/// `local_transform` is the donor `oa::Transform localTransform` — a TRS
/// semantic value, not a raw matrix.
///
/// `material` overrides the renderer's default material for this node.
/// `None` falls back to `FlatColor { color: [0.8, 0.8, 0.8, 1.0] }` at draw
/// time.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneNode {
	pub id: SceneNodeId,
	pub name: String,
	pub parent: SceneNodeId,
	pub mesh: SceneMeshId,
	/// Optional material override. References a [`SceneMaterial`] in the same
	/// [`Scene`] by identity.
	pub material: Option<MaterialId>,
	pub local_transform: Transform,
	pub visible: bool,
}

/// Caller-owned CPU semantic scene snapshot.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
	pub meshes: Vec<SceneMesh>,
	/// Material library for this scene.
	pub materials: Vec<SceneMaterial>,
	pub nodes: Vec<SceneNode>,
}

/// Validate one scene without retaining or uploading it.
pub fn validate_scene(scene: &Scene) -> Result<()> {
	// ── Meshes ────────────────────────────────────────────────────────────────
	let mut meshes = HashSet::new();
	for mesh in &scene.meshes {
		if mesh.id.0 == 0 || !meshes.insert(mesh.id) {
			return Err(Error::invalid_argument(
				"render::Scene mesh IDs must be non-zero and unique",
			));
		}
		validate_mesh(&mesh.data)?;
	}

	// ── Materials ─────────────────────────────────────────────────────────────
	let mut materials = HashSet::new();
	for mat in &scene.materials {
		if mat.id.0 == 0 || !materials.insert(mat.id) {
			return Err(Error::invalid_argument(
				"render::Scene material IDs must be non-zero and unique",
			));
		}
		mat.data.validate().map_err(|e| {
			Error::invalid_argument(format!("render::Scene material '{}': {}", mat.name, e))
		})?;
	}

	// ── Nodes ─────────────────────────────────────────────────────────────────
	let mut nodes = HashSet::new();
	for node in &scene.nodes {
		// Transform is always finite by construction (validate catches
		// any degenerate rotation introduced through set_matrix).
		if node.id.0 == 0 || !nodes.insert(node.id) || !node.local_transform.is_finite() {
			return Err(Error::invalid_argument(
				"render::Scene nodes require unique non-zero IDs and finite transforms",
			));
		}
		if node.mesh.0 != 0 && !meshes.contains(&node.mesh) {
			return Err(Error::invalid_argument(
				"render::Scene node references a missing mesh",
			));
		}
		if node
			.material
			.is_some_and(|mat_id| !materials.contains(&mat_id))
		{
			return Err(Error::invalid_argument(
				"render::Scene node references a missing material",
			));
		}
	}

	// ── Hierarchy ─────────────────────────────────────────────────────────────
	for node in &scene.nodes {
		if node.parent.0 != 0 && (!nodes.contains(&node.parent) || node.parent == node.id) {
			return Err(Error::invalid_argument(
				"render::Scene node parent is missing or self-referential",
			));
		}
		let mut seen = HashSet::new();
		let mut parent = node.parent;
		while parent.0 != 0 {
			if !seen.insert(parent) {
				return Err(Error::invalid_argument(
					"render::Scene hierarchy contains a cycle",
				));
			}
			parent = scene
				.nodes
				.iter()
				.find(|candidate| candidate.id == parent)
				.expect("validated parent")
				.parent;
		}
	}
	Ok(())
}

/// Compile visible scene instances into one deterministic indexed mesh snapshot.
pub fn compile_scene(scene: &Scene) -> Result<MeshData> {
	let (mesh, _) = compile_scene_packets(scene)?;
	Ok(mesh)
}

/// Per-node draw range produced by [`compile_scene_packets`].
///
/// `index_start` and `index_count` are byte-aligned offsets into the merged
/// index buffer returned alongside this packet.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneDrawPacket {
	/// First index in the merged index buffer for this node.
	pub index_start: u32,
	/// Number of indices (must be a multiple of 3).
	pub index_count: u32,
	/// The material assigned to this node; `None` means use the renderer
	/// default.
	pub material_id: Option<MaterialId>,
}

/// Compile visible scene nodes into a merged `MeshData` and one
/// [`SceneDrawPacket`] per visible node that contributes geometry.
///
/// The returned mesh has world-space positions and normals; each packet
/// records the index range within the merged buffer that belongs to that
/// node. Draw order is deterministic (depth-first scene-node order).
pub fn compile_scene_packets(scene: &Scene) -> Result<(MeshData, Vec<SceneDrawPacket>)> {
	validate_scene(scene)?;
	let node_by_id: HashMap<_, _> = scene.nodes.iter().map(|node| (node.id, node)).collect();
	let mesh_by_id: HashMap<_, _> = scene.meshes.iter().map(|mesh| (mesh.id, mesh)).collect();
	let mut output = MeshData::default();
	let mut packets: Vec<SceneDrawPacket> = Vec::new();
	for node in &scene.nodes {
		if !is_visible(node, &node_by_id) || node.mesh.0 == 0 {
			continue;
		}
		let world = world_transform(node, &node_by_id);
		let normal_matrix = world
			.try_normal_matrix(crate::vlm::INVERSE_TOLERANCE)
			.ok_or_else(|| Error::invalid_argument("render::Scene world transform is singular"))?;
		let base = u32::try_from(output.vertices.len())
			.map_err(|_| Error::out_of_range("render::Scene vertex count exceeds u32"))?;
		let index_start = u32::try_from(output.indices.len())
			.map_err(|_| Error::out_of_range("render::Scene index count exceeds u32"))?;
		let mesh = mesh_by_id[&node.mesh];
		output
			.vertices
			.extend(mesh.data.vertices.iter().map(|vertex| MeshVertex {
				position: world.transform_point(vertex.position),
				normal: vertex.normal * normal_matrix,
				..*vertex
			}));
		let new_indices: Vec<u32> = mesh
			.data
			.indices
			.iter()
			.map(|index| {
				base
					.checked_add(*index)
					.ok_or_else(|| Error::out_of_range("render::Scene index overflows u32"))
			})
			.collect::<Result<Vec<_>>>()?;
		let index_count = u32::try_from(new_indices.len())
			.map_err(|_| Error::out_of_range("render::Scene index count exceeds u32"))?;
		output.indices.extend(new_indices);
		packets.push(SceneDrawPacket {
			index_start,
			index_count,
			material_id: node.material,
		});
	}
	Ok((output, packets))
}

fn validate_mesh(mesh: &MeshData) -> Result<()> {
	if !mesh.indices.len().is_multiple_of(3)
		|| mesh
			.indices
			.iter()
			.any(|index| *index as usize >= mesh.vertices.len())
		|| mesh.vertices.iter().any(|v| {
			!v.position.is_finite() || !v.normal.is_finite() || !v.uv.is_finite() || !v.color.is_finite()
		}) {
		return Err(Error::invalid_argument(
			"render::MeshData requires finite vertices and in-range triangle indices",
		));
	}
	Ok(())
}

/// Accumulate local transforms up to the root, returning a world-space Mat4.
fn world_transform(node: &SceneNode, nodes: &HashMap<SceneNodeId, &SceneNode>) -> Mat4 {
	let mut result = node.local_transform.get_matrix();
	let mut parent = node.parent;
	while parent.0 != 0 {
		let value = nodes[&parent];
		result *= value.local_transform.get_matrix();
		parent = value.parent;
	}
	result
}

fn is_visible(node: &SceneNode, nodes: &HashMap<SceneNodeId, &SceneNode>) -> bool {
	if !node.visible {
		return false;
	}
	let mut parent = node.parent;
	while parent.0 != 0 {
		let value = nodes[&parent];
		if !value.visible {
			return false;
		}
		parent = value.parent;
	}
	true
}
