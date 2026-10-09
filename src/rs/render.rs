//! Rendering, texture, mesh, material, scene, camera, and presentation
//! contracts.
//!
//! This namespace intentionally exposes no placeholder renderer or presenter.
//! Stateful rendering and presentation types are admitted only with explicit
//! engine ownership, synchronization, failure, and lifecycle behavior.

mod camera;
mod material;
mod mesh;
mod renderer;
mod scene;
mod texture;

pub use camera::{Camera, CameraProjection};
pub use material::{
	AlphaMode, FlatColorMaterial, Material, MaterialId, StandardSurfaceMaterial, TextureHandle,
	UnlitMaterial,
};
pub use mesh::{
	Aabb, compute_bounds, create_cube, create_fullscreen_quad, create_quad, flip_uvs_y, scale,
	transform, translate,
};
pub use renderer::{RenderFrame, Renderer, RendererConfig};
pub use scene::{
	MeshData, MeshVertex, Scene, SceneDrawPacket, SceneMaterial, SceneMesh, SceneMeshId, SceneNode,
	SceneNodeId, compile_scene, compile_scene_packets, validate_scene,
};
pub use texture::{
	Texture, blit, clear, save_texture_file, texture_from_image, texture_from_rgba8,
};
