// ═══════════════════════════════════════════════════════════════════════════
// OA Tutorial: Ui composition → Viewer display
// Level-1 API — oa::Ui, oa::Viewer::show
// ═══════════════════════════════════════════════════════════════════════════
//
// Demonstrates the reusable editor/tooling shape:
//   1. Build one oa::Ui frame without a window (headless GPU composition).
//   2. Submit the composed Texture through a one-slot Renderer.
//   3. Display its generation-safe RenderFrame in a blocking Viewer window.
//
// This is the renderer/sink split (architecture/oaArchitecture.md §10).
// The same Ui compositor can feed any downstream sink — swapchain, PNG,
// video encoder — without changing the draw commands.
//
// Widgets exercised:
//   - ui.rect_raw / rect_outline_raw   (background and borders)
//   - ui.tab_bar / UiTabItem           (document tabs with dirty indicator)
//   - ui.node_canvas_grid / NodeCanvas (graph editor background)
//   - ui.line_at                        (node graph connector wires)
//   - ui.tree_row / UiTreeRowConfig     (outliner hierarchy)
//   - ui.label                          (section headers)
//   - ui.input_text                     (name field)
//   - ui.checkbox                       (enabled toggle)
//   - ui.slider_f32                     (exposure)
//   - ui.slider_i32                     (sample count)
//   - ui.separator
//   - ui.color_swatch                   (base color preview)
//   - ui.progress_bar                   (compile progress)
//   - ui.button                         (render frame action)
//
// usage:  cargo run --example tu_renderer_ui
//
// Donor: sdk/cpp/tutorials/ui/tuRendererUi.cpp
// ═══════════════════════════════════════════════════════════════════════════

use oa::render::{Renderer, RendererConfig, save_texture_file, texture_from_rgba8};
use oa::runtime::SdlWindow;
use oa::{
	Color, Engine, NodeCanvas, PixelRect, Result, Ui, UiStyle, UiTabBarState, UiTabItem,
	UiTreeRowConfig, Viewer, ViewerConfig,
};

fn main() -> Result<()> {
	const W: u32 = 960;
	const H: u32 = 600;

	// ── Engine ────────────────────────────────────────────────────────────────
	// A presentation-capable engine is required so Viewer can display the
	// resulting frame in a swapchain window.
	let window = SdlWindow::new("OA Renderer · UI Composition", W, H)?;
	let extensions = window.required_instance_extensions()?;
	let engine = Engine::builder().instance_extensions(extensions).build()?;

	// ── Headless Ui compositor ────────────────────────────────────────────────
	let mut ui = Ui::init_with_style(&engine, W, H, UiStyle::default())?;

	// ── Build UI frame ────────────────────────────────────────────────────────
	ui.begin_frame(0.0);

	let full = PixelRect::new(0, 0, W, H);

	// Full-frame background.
	ui.rect_raw(full, Color::new(0.082, 0.082, 0.090, 1.0), full, 0.0);

	// Tab bar (document tabs).
	let tabs = vec![
		UiTabItem {
			id: "scene".into(),
			label: "Scene.oa".into(),
			dirty: true,
			closable: true,
			enabled: true,
		},
		UiTabItem {
			id: "material".into(),
			label: "Material".into(),
			dirty: false,
			closable: true,
			enabled: true,
		},
		UiTabItem {
			id: "output".into(),
			label: "output".into(),
			dirty: false,
			closable: true,
			enabled: true,
		},
	];
	let mut tab_state = UiTabBarState {
		selected: 0,
		first_visible: 0,
	};
	ui.tab_bar(
		"documents",
		PixelRect::new(0, 0, W, 34),
		&tabs,
		&mut tab_state,
	);

	// Node-graph canvas (center panel).
	let canvas_rect = PixelRect::new(190, 42, 570, 520);
	let mut canvas = NodeCanvas::new();
	canvas.set_view_size(570.0, 520.0)?;
	ui.node_canvas_grid(&canvas, canvas_rect);

	// Connector wires.
	let accent = Color::new(0.388, 0.400, 0.945, 1.0);
	ui.line_at([304.0, 183.0], [448.0, 287.0], accent, 2.0, canvas_rect)?;
	ui.line_at([536.0, 287.0], [650.0, 399.0], accent, 2.0, canvas_rect)?;

	// Node boxes.
	let surface_c = Color::new(0.141, 0.141, 0.161, 1.0);
	let border_c = Color::new(0.250, 0.250, 0.280, 1.0);
	ui.rect_raw(
		PixelRect::new(236, 142, 132, 82),
		surface_c,
		canvas_rect,
		4.0,
	);
	ui.rect_outline_raw(PixelRect::new(236, 142, 132, 82), accent, 2, canvas_rect);
	ui.rect_raw(
		PixelRect::new(432, 246, 120, 82),
		surface_c,
		canvas_rect,
		4.0,
	);
	ui.rect_outline_raw(PixelRect::new(432, 246, 120, 82), border_c, 1, canvas_rect);
	ui.rect_raw(
		PixelRect::new(594, 358, 116, 82),
		surface_c,
		canvas_rect,
		4.0,
	);
	ui.rect_outline_raw(PixelRect::new(594, 358, 116, 82), border_c, 1, canvas_rect);

	// Left outliner panel.
	ui.begin_panel("outliner", PixelRect::new(8, 42, 174, 520));
	ui.label("OUTLINER");
	ui.tree_row(
		"world",
		PixelRect::new(16, 82, 158, 26),
		"World",
		UiTreeRowConfig {
			has_children: true,
			open: true,
			selected: true,
			..UiTreeRowConfig::new()
		},
	);
	ui.tree_row(
		"camera",
		PixelRect::new(16, 110, 158, 26),
		"camera",
		UiTreeRowConfig {
			depth: 1,
			..UiTreeRowConfig::new()
		},
	);
	ui.tree_row(
		"character",
		PixelRect::new(16, 138, 158, 26),
		"character",
		UiTreeRowConfig {
			depth: 1,
			..UiTreeRowConfig::new()
		},
	);
	ui.tree_row(
		"lights",
		PixelRect::new(16, 166, 158, 26),
		"Lights",
		UiTreeRowConfig {
			depth: 1,
			..UiTreeRowConfig::new()
		},
	);
	ui.end_panel();

	// Right inspector panel.
	let mut enabled = true;
	let mut exposure = 0.65_f32;
	let mut samples = 64_i32;
	let mut name = String::from("HeroSurface");
	ui.begin_panel("inspector", PixelRect::new(768, 42, 184, 520));
	ui.label("INSPECTOR");
	ui.input_text("Name", &mut name);
	ui.checkbox("enabled", &mut enabled);
	ui.slider_f32("Exposure", &mut exposure, 0.0, 2.0);
	ui.slider_i32("samples", &mut samples, 1, 256);
	ui.separator();
	ui.label("base color");
	ui.color_swatch(Color::new(0.16, 0.48, 0.92, 1.0), [16.0, 16.0]);
	ui.progress_bar(0.72);
	ui.button("Render frame");
	ui.end_panel();

	ui.end_frame();

	// ── Sink: convert composed image to a Texture for Viewer display ──────────
	// render_rgba8 flushes and waits. texture_from_rgba8 re-uploads for blit.
	let rgba = ui.render_rgba8()?;
	let tex = texture_from_rgba8(&engine, &rgba, W as usize, H as usize)?;
	ui.close()?;

	// Optional: also write to disk for headless CI.
	if let Ok(out) = std::env::var("TU_RENDERER_UI_OUTPUT") {
		save_texture_file(&tex, &out, 90)?;
		println!("OK: wrote {} × {} to {}", W, H, out);
		return Ok(());
	}

	// ── Interactive: submit and display a generation-safe RenderFrame ─────────
	let renderer = Renderer::new(
		&engine,
		RendererConfig {
			width: W,
			height: H,
			target_slot_count: 1,
			max_vertex_count: 0,
			max_index_count: 0,
			..RendererConfig::default()
		},
	)?;
	renderer.begin_frame(&tex)?;
	let frame = renderer.submit_frame()?;

	// One target slot proves the frame lease was returned: a new frame is
	// immediately recordable after the blocking Viewer exits.
	let viewer_config = ViewerConfig {
		title: "OA renderer · UI Composition".to_owned(),
		width: W,
		height: H,
		show_help: false,
		show_stats: false,
		show_timeline: false,
		..ViewerConfig::default()
	};
	Viewer::show_render_frame(&engine, &renderer, frame, viewer_config)?;
	renderer.begin_frame(&tex)?;
	renderer.cancel_frame()?;
	renderer.close()
}
