use std::{
	any::TypeId,
	time::{SystemTime, UNIX_EPOCH},
};

use oa::{ErrorKind, Texture};

#[test]
fn root_texture_export_is_the_render_value_identity() {
	assert_eq!(TypeId::of::<Texture>(), TypeId::of::<oa::render::Texture>());
}

test_vk!(
	packed_texture_upload_readback_and_file_sink_are_exact,
	engine,
	{
		let rgba = [
			255, 0, 0, 255, 0, 255, 0, 128, // row zero
			0, 0, 255, 64, 255, 255, 255, 0, // row one
		];
		let texture = oa::render::texture_from_rgba8(&engine, &rgba, 2, 2)?;
		assert_eq!((texture.width(), texture.height()), (2, 2));
		assert_eq!(texture.dtype(), oa::DType::U8);
		assert_eq!(texture.read_rgba8()?, rgba);
		assert_eq!(texture.clone().read_rgba8()?, rgba);

		let path = temporary_path();
		oa::render::save_texture_file(&texture, &path, 90)?;
		let decoded = image_codec::open(&path)
			.expect("saved texture PNG must decode")
			.to_rgba8();
		assert_eq!(decoded.into_raw(), rgba);
		std::fs::remove_file(path).expect("temporary texture image removal failed");
		Ok(())
	}
);

test_vk!(packed_texture_rejects_invalid_host_ranges, engine, {
	for error in [
		oa::render::texture_from_rgba8(&engine, &[], 0, 1)
			.err()
			.expect("zero width was accepted"),
		oa::render::texture_from_rgba8(&engine, &[0; 3], 1, 1)
			.err()
			.expect("short RGBA input was accepted"),
	] {
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
	Ok(())
});

test_vk!(
	packed_texture_clear_and_whole_surface_blit_are_exact,
	engine,
	{
		let source = oa::render::texture_from_rgba8(&engine, &[1, 2, 3, 4, 5, 6, 7, 8], 2, 1)?;
		let copy = oa::render::blit(&source)?;
		assert_eq!(copy.read_rgba8()?, source.read_rgba8()?);
		let cleared = oa::render::clear(
			&source,
			oa::Color {
				r: 1.0,
				g: 0.5,
				b: 0.0,
				a: 0.25,
			},
		)?;
		assert_eq!(
			cleared.read_rgba8()?,
			vec![255, 128, 0, 64, 255, 128, 0, 64]
		);
		assert_eq!(source.read_rgba8()?, vec![1, 2, 3, 4, 5, 6, 7, 8]);
		let saturated = oa::render::clear(
			&source,
			oa::Color {
				r: -0.1,
				g: 2.0,
				b: 0.0,
				a: 1.0,
			},
		)?;
		assert_eq!(
			saturated.read_rgba8()?,
			vec![0, 255, 0, 255, 0, 255, 0, 255]
		);
		Ok(())
	}
);

test_vk!(
	packed_rgba_image_conversion_preserves_texture_semantics,
	engine,
	{
		let image = oa::Image::new(
			oa::Matrix::from_slice(&engine, [1, 2, 4], &[9_u8, 8, 7, 6, 5, 4, 3, 2])?,
			oa::ImageLayout::Hwc,
			oa::ImageFormat::Rgba,
		)?;
		let texture = oa::render::texture_from_image(&image)?;
		assert_eq!((texture.width(), texture.height()), (2, 1));
		assert_eq!(texture.read_rgba8()?, vec![9, 8, 7, 6, 5, 4, 3, 2]);
		Ok(())
	}
);

test_vk!(
	headless_renderer_uses_explicit_target_ring_leases,
	engine,
	{
		let renderer = oa::render::Renderer::new(
			&engine,
			oa::render::RendererConfig {
				width: 2,
				height: 1,
				target_slot_count: 1,
				max_vertex_count: 0,
				max_index_count: 0,
				..Default::default()
			},
		)?;
		let source = oa::render::texture_from_rgba8(&engine, &[1, 2, 3, 4, 5, 6, 7, 8], 2, 1)?;
		renderer.begin_frame(&source)?;
		let frame = renderer.submit_frame()?;
		assert_eq!((frame.width(), frame.height()), (2, 1));
		let timing = oa::video::VideoFrameTiming::from_microseconds(17, 33);
		let color = oa::video::VideoColorInfo::unspecified();
		let video_frame = oa::video::from_texture(frame.color(), timing, color)?;
		assert!(video_frame.ready_event().is_none());
		drop(video_frame);
		assert_eq!(frame.consume_readback()?, vec![1, 2, 3, 4, 5, 6, 7, 8]);
		renderer.begin_frame(&source)?;
		renderer.cancel_frame()?;
		renderer.close()?;
		Ok(())
	}
);

test_vk!(
	headless_renderer_draws_flat_color_geometry,
	engine,
	"requires a hardware Vulkan graphics device",
	{
		let renderer = oa::render::Renderer::new(
			&engine,
			oa::render::RendererConfig {
				width: 32,
				height: 32,
				target_slot_count: 1,
				max_vertex_count: 4,
				max_index_count: 6,
				clear_color: oa::Color::new(0.0, 0.0, 0.0, 1.0),
				..Default::default()
			},
		)?;
		let mesh = oa::render::create_quad(1.0, 1.0)?;
		let camera = oa::render::Camera::new_orthographic(2.0, 2.0, 0.1, 10.0);
		let projected = camera
			.get_view_projection_matrix()
			.try_project_point(mesh.vertices[0].position, f32::EPSILON);
		let material = oa::render::FlatColorMaterial {
			color: oa::Color::new(1.0, 0.0, 0.0, 1.0),
			opacity: 1.0,
		};
		renderer.begin_frame_flat_color(&mesh, &camera, &material)?;
		let frame = renderer.submit_frame()?;
		let timing = oa::video::VideoFrameTiming::from_microseconds(17, 33);
		let color = oa::video::VideoColorInfo::unspecified();
		let error = oa::video::from_texture(frame.color(), timing, color)
			.err()
			.expect("render target must retain producer completion");
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
		let video_frame =
			oa::video::from_texture_with_ready(frame.color(), timing, color, frame.producer())?;
		assert!(video_frame.ready_event().is_some());
		let consumer = engine.checkpoint()?;
		video_frame.mark_consumed(&consumer)?;
		let stale = video_frame
			.mark_consumed(frame.producer())
			.expect_err("an earlier event must not replace consumer completion");
		assert_eq!(stale.kind(), ErrorKind::InvalidArgument);
		consumer.wait()?;
		drop(video_frame);
		let retained_texture = frame.color().clone();
		let pixels = frame.consume_readback()?;
		let maxima = pixels
			.as_chunks::<4>()
			.0
			.iter()
			.fold([0_u8; 4], |mut maxima, pixel| {
				for channel in 0..4 {
					maxima[channel] = maxima[channel].max(pixel[channel]);
				}
				maxima
			});
		assert!(
			pixels
				.as_chunks::<4>()
				.0
				.iter()
				.any(|pixel| pixel[0] > 200 && pixel[1] < 10 && pixel[2] < 10),
			"flat-color draw produced no red pixels; channel maxima were {maxima:?}, first vertex projected to {projected:?}"
		);
		assert_eq!(
			renderer
				.begin_frame_flat_color(&mesh, &camera, &material)
				.expect_err("live render-target Texture alias allowed slot reuse")
				.kind(),
			ErrorKind::ResourceExhausted
		);
		drop(retained_texture);
		assert_eq!(renderer.collect()?, 1);
		renderer.begin_frame_flat_color(&mesh, &camera, &material)?;
		renderer.cancel_frame()?;
		renderer.close()?;
		Ok(())
	}
);

test_vk!(
	headless_renderer_samples_registered_unlit_texture,
	engine,
	"requires a hardware Vulkan graphics device",
	{
		let renderer = oa::render::Renderer::new(
			&engine,
			oa::render::RendererConfig {
				width: 32,
				height: 32,
				target_slot_count: 1,
				max_vertex_count: 4,
				max_index_count: 6,
				clear_color: oa::Color::new(0.0, 0.0, 0.0, 1.0),
				..Default::default()
			},
		)?;
		let source = oa::render::texture_from_rgba8(
			&engine,
			&[
				255, 0, 0, 255, 0, 255, 0, 255, // row zero
				0, 0, 255, 255, 255, 255, 255, 255, // row one
			],
			2,
			2,
		)?;
		let handle = oa::render::TextureHandle(1);
		renderer.register_texture(handle, &source)?;
		let material = oa::render::UnlitMaterial {
			base_color_texture: Some(handle),
			..Default::default()
		};
		let mesh = oa::render::create_quad(2.0, 2.0)?;
		let camera = oa::render::Camera::new_orthographic(2.0, 2.0, 0.1, 10.0);
		renderer.begin_frame_unlit(&mesh, &camera, &material)?;
		let pixels = renderer.submit_frame()?.consume_readback()?;
		assert!(
			pixels.as_chunks::<4>().0.iter().any(|pixel| pixel[0] > 180),
			"unlit draw did not sample the registered texture"
		);
		renderer.close()?;
		Ok(())
	}
);

test_vk!(
	headless_renderer_draws_standard_surface_scene,
	engine,
	"requires a hardware Vulkan graphics device",
	{
		let renderer = oa::render::Renderer::new(
			&engine,
			oa::render::RendererConfig {
				width: 32,
				height: 32,
				target_slot_count: 1,
				max_vertex_count: 4,
				max_index_count: 6,
				clear_color: oa::Color::new(0.0, 0.0, 0.0, 1.0),
				..Default::default()
			},
		)?;
		let material_id = oa::render::MaterialId(1);
		let mesh_id = oa::render::SceneMeshId(1);
		let material = oa::render::StandardSurfaceMaterial {
			base_color: [0.4, 0.0, 0.0],
			emission_color: [1.0, 0.0, 0.0],
			emission_scale: 1.0,
			..Default::default()
		};
		let scene = oa::render::Scene {
			meshes: vec![oa::render::SceneMesh {
				id: mesh_id,
				name: "quad".into(),
				data: oa::render::create_quad(1.0, 1.0)?,
			}],
			materials: vec![oa::render::SceneMaterial {
				id: material_id,
				name: "emissive red".into(),
				data: oa::render::Material::Standard(material),
			}],
			nodes: vec![oa::render::SceneNode {
				id: oa::render::SceneNodeId(1),
				name: "quad".into(),
				parent: oa::render::SceneNodeId(0),
				mesh: mesh_id,
				material: Some(material_id),
				local_transform: oa::Transform::default(),
				visible: true,
			}],
		};
		let camera = oa::render::Camera::new_orthographic(2.0, 2.0, 0.1, 10.0);
		renderer.begin_frame_scene(&scene, &camera)?;
		let pixels = renderer.submit_frame()?.consume_readback()?;
		assert!(
			pixels
				.as_chunks::<4>()
				.0
				.iter()
				.any(|pixel| pixel[0] > 180 && pixel[1] < 20 && pixel[2] < 20),
			"standard-surface scene draw produced no emissive red pixels"
		);
		renderer.close()?;
		Ok(())
	}
);

test_vk!(
	headless_renderer_draws_heterogeneous_two_material_scene,
	engine,
	"requires a hardware Vulkan graphics device",
	{
		// Two quads with two distinct materials: one emissive red, one emissive blue.
		// The renderer must submit both draws in one render pass and produce pixels
		// from both materials.
		let renderer = oa::render::Renderer::new(
			&engine,
			oa::render::RendererConfig {
				width: 64,
				height: 32,
				target_slot_count: 1,
				max_vertex_count: 8,
				max_index_count: 12,
				clear_color: oa::Color::new(0.0, 0.0, 0.0, 1.0),
				max_materials_per_scene: 2,
				..Default::default()
			},
		)?;
		let red_id = oa::render::MaterialId(1);
		let blue_id = oa::render::MaterialId(2);
		let mesh_a = oa::render::SceneMeshId(1);
		let mesh_b = oa::render::SceneMeshId(2);
		// Left quad (-2..0 in X) — red emissive.
		let left_quad = oa::render::MeshData {
			vertices: vec![
				oa::render::MeshVertex {
					position: oa::vlm::Vec3 {
						x: -2.0,
						y: -1.0,
						z: 0.0,
					},
					normal: oa::vlm::Vec3 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
					},
					uv: oa::vlm::Vec2 { x: 0.0, y: 1.0 },
					color: oa::vlm::Vec4 {
						x: 1.0,
						y: 0.0,
						z: 0.0,
						w: 1.0,
					},
				},
				oa::render::MeshVertex {
					position: oa::vlm::Vec3 {
						x: 0.0,
						y: -1.0,
						z: 0.0,
					},
					normal: oa::vlm::Vec3 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
					},
					uv: oa::vlm::Vec2 { x: 1.0, y: 1.0 },
					color: oa::vlm::Vec4 {
						x: 1.0,
						y: 0.0,
						z: 0.0,
						w: 1.0,
					},
				},
				oa::render::MeshVertex {
					position: oa::vlm::Vec3 {
						x: 0.0,
						y: 1.0,
						z: 0.0,
					},
					normal: oa::vlm::Vec3 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
					},
					uv: oa::vlm::Vec2 { x: 1.0, y: 0.0 },
					color: oa::vlm::Vec4 {
						x: 1.0,
						y: 0.0,
						z: 0.0,
						w: 1.0,
					},
				},
				oa::render::MeshVertex {
					position: oa::vlm::Vec3 {
						x: -2.0,
						y: 1.0,
						z: 0.0,
					},
					normal: oa::vlm::Vec3 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
					},
					uv: oa::vlm::Vec2 { x: 0.0, y: 0.0 },
					color: oa::vlm::Vec4 {
						x: 1.0,
						y: 0.0,
						z: 0.0,
						w: 1.0,
					},
				},
			],
			indices: vec![0, 1, 2, 0, 2, 3],
			..Default::default()
		};
		// Right quad (0..2 in X) — blue emissive.
		let right_quad = oa::render::MeshData {
			vertices: vec![
				oa::render::MeshVertex {
					position: oa::vlm::Vec3 {
						x: 0.0,
						y: -1.0,
						z: 0.0,
					},
					normal: oa::vlm::Vec3 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
					},
					uv: oa::vlm::Vec2 { x: 0.0, y: 1.0 },
					color: oa::vlm::Vec4 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
						w: 1.0,
					},
				},
				oa::render::MeshVertex {
					position: oa::vlm::Vec3 {
						x: 2.0,
						y: -1.0,
						z: 0.0,
					},
					normal: oa::vlm::Vec3 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
					},
					uv: oa::vlm::Vec2 { x: 1.0, y: 1.0 },
					color: oa::vlm::Vec4 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
						w: 1.0,
					},
				},
				oa::render::MeshVertex {
					position: oa::vlm::Vec3 {
						x: 2.0,
						y: 1.0,
						z: 0.0,
					},
					normal: oa::vlm::Vec3 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
					},
					uv: oa::vlm::Vec2 { x: 1.0, y: 0.0 },
					color: oa::vlm::Vec4 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
						w: 1.0,
					},
				},
				oa::render::MeshVertex {
					position: oa::vlm::Vec3 {
						x: 0.0,
						y: 1.0,
						z: 0.0,
					},
					normal: oa::vlm::Vec3 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
					},
					uv: oa::vlm::Vec2 { x: 0.0, y: 0.0 },
					color: oa::vlm::Vec4 {
						x: 0.0,
						y: 0.0,
						z: 1.0,
						w: 1.0,
					},
				},
			],
			indices: vec![0, 1, 2, 0, 2, 3],
			..Default::default()
		};
		let scene = oa::render::Scene {
			meshes: vec![
				oa::render::SceneMesh {
					id: mesh_a,
					name: "left".into(),
					data: left_quad,
				},
				oa::render::SceneMesh {
					id: mesh_b,
					name: "right".into(),
					data: right_quad,
				},
			],
			materials: vec![
				oa::render::SceneMaterial {
					id: red_id,
					name: "red".into(),
					data: oa::render::Material::Standard(oa::render::StandardSurfaceMaterial {
						emission_color: [1.0, 0.0, 0.0],
						emission_scale: 1.0,
						..Default::default()
					}),
				},
				oa::render::SceneMaterial {
					id: blue_id,
					name: "blue".into(),
					data: oa::render::Material::Standard(oa::render::StandardSurfaceMaterial {
						emission_color: [0.0, 0.0, 1.0],
						emission_scale: 1.0,
						..Default::default()
					}),
				},
			],
			nodes: vec![
				oa::render::SceneNode {
					id: oa::render::SceneNodeId(1),
					name: "left node".into(),
					parent: oa::render::SceneNodeId(0),
					mesh: mesh_a,
					material: Some(red_id),
					local_transform: oa::Transform::default(),
					visible: true,
				},
				oa::render::SceneNode {
					id: oa::render::SceneNodeId(2),
					name: "right node".into(),
					parent: oa::render::SceneNodeId(0),
					mesh: mesh_b,
					material: Some(blue_id),
					local_transform: oa::Transform::default(),
					visible: true,
				},
			],
		};
		let camera = oa::render::Camera::new_orthographic(4.0, 2.0, 0.1, 10.0);
		renderer.begin_frame_scene(&scene, &camera)?;
		let pixels = renderer.submit_frame()?.consume_readback()?;
		let has_red = pixels
			.as_chunks::<4>()
			.0
			.iter()
			.any(|p| p[0] > 150 && p[1] < 20 && p[2] < 20);
		let has_blue = pixels
			.as_chunks::<4>()
			.0
			.iter()
			.any(|p| p[2] > 150 && p[0] < 20 && p[1] < 20);
		assert!(
			has_red,
			"heterogeneous scene: red emissive quad not found in output"
		);
		assert!(
			has_blue,
			"heterogeneous scene: blue emissive quad not found in output"
		);
		renderer.close()?;
		Ok(())
	}
);

#[test]
fn scene_validation_rejects_a_parent_cycle() {
	let scene = oa::render::Scene {
		meshes: vec![],
		materials: vec![],
		nodes: vec![
			oa::render::SceneNode {
				id: oa::render::SceneNodeId(1),
				name: "a".into(),
				parent: oa::render::SceneNodeId(2),
				mesh: oa::render::SceneMeshId(0),
				material: None,
				local_transform: oa::Transform::default(),
				visible: true,
			},
			oa::render::SceneNode {
				id: oa::render::SceneNodeId(2),
				name: "b".into(),
				parent: oa::render::SceneNodeId(1),
				mesh: oa::render::SceneMeshId(0),
				material: None,
				local_transform: oa::Transform::default(),
				visible: true,
			},
		],
	};
	assert_eq!(
		oa::render::validate_scene(&scene)
			.expect_err("cycle accepted")
			.kind(),
		ErrorKind::InvalidArgument
	);
}

#[test]
fn quad_geometry_has_donor_winding_and_top_origin_uvs() {
	let quad = oa::render::create_quad(2.0, 4.0).expect("valid quad");
	assert_eq!(quad.indices, vec![0, 1, 2, 0, 2, 3]);
	assert_eq!(
		quad.vertices[0].position,
		oa::vlm::Vec3 {
			x: -1.0,
			y: 2.0,
			z: 0.0
		}
	);
	assert_eq!(quad.vertices[0].uv, oa::vlm::Vec2 { x: 0.0, y: 0.0 });
	assert_eq!(quad.vertices[2].uv, oa::vlm::Vec2 { x: 1.0, y: 1.0 });
	assert_eq!(
		oa::render::create_quad(0.0, 1.0)
			.expect_err("zero width accepted")
			.kind(),
		ErrorKind::InvalidArgument
	);
}

fn temporary_path() -> std::path::PathBuf {
	let nonce = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.expect("system time before Unix epoch")
		.as_nanos();
	std::env::temp_dir().join(format!(
		"oa-render-texture-{}-{nonce}.png",
		std::process::id()
	))
}
