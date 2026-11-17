//! Canvas background and grid overlays. Donor: `oa::Viewer::renderVisualWorkspace`.

use super::config::{ViewerCanvasBackground, ViewerConfig};
use crate::input::Navigation;
use crate::ui::{TextLayoutConfig, Ui, types::PixelRect};
use crate::{Color, Result};

/// Preserve the donor's image-space guide geometry and background composition.
/// Donor: `oa::Viewer::renderVisualWorkspace` (mechanical Rust adaptation).
pub fn draw_canvas_background(
	ui: &mut Ui<'_>,
	rect: PixelRect,
	destination: PixelRect,
	navigation: &Navigation,
	content_scale: f32,
	allow_grid: bool,
	config: &ViewerConfig,
) -> Result<()> {
	let gradient = config.canvas_background == ViewerCanvasBackground::Gradient;
	let draw_grid = allow_grid && config.show_canvas_grid;
	if !gradient && !draw_grid {
		return Ok(());
	}
	let mut minor_spacing = 100.0 * navigation.zoom();
	for _ in 0..8 {
		if minor_spacing >= 8.0 {
			break;
		}
		minor_spacing *= 10.0;
	}
	let origin = [
		destination.x as f32 + destination.w as f32 * 0.5,
		destination.y as f32 + destination.h as f32 * 0.5,
	];
	let radius = (0.5 * (destination.w as f32).hypot(destination.h as f32)).max(1.0);
	ui.grid(
		rect,
		crate::ui::UiGridConfig {
			origin,
			minor_spacing: [minor_spacing; 2],
			minor_thickness: 0.65 * content_scale,
			major_thickness: 0.90 * content_scale,
			super_major_thickness: content_scale,
			axis_thickness: 1.25 * content_scale,
			guide_fade_center: origin,
			guide_fade_inner_radius: radius,
			guide_fade_outer_radius: radius * 3.5,
			background_top: Color::new_from_u32(0x1F1F1FA6),
			background_bottom: Color::new_from_u32(0x181818A6),
			fill_background: gradient,
			draw_grid,
			draw_axes: draw_grid,
			radial_guide_fade: true,
			dither_background: gradient,
			..Default::default()
		},
	);
	Ok(())
}

/// Draw optional status in a bottom-right overlay box.
///
/// `bottom_inset` reserves the transport block height so the box clears the
/// timeline in temporal modes. Startup help belongs in the log, as in the
/// donor.
pub fn draw_hud(
	ui: &mut Ui<'_>,
	viewport: PixelRect,
	navigation: &Navigation,
	mode: &str,
	bottom_inset: f32,
	config: &ViewerConfig,
) -> Result<()> {
	if !config.show_stats {
		return Ok(());
	}
	let scale = ui.content_scale();
	let text = format!("{mode}  {:.0}%", navigation.zoom() * 100.0);
	let layout = TextLayoutConfig {
		font: crate::ui::FontId::SansSemibold,
		size: 24.0 * scale,
		..Default::default()
	};
	let pad_x = 12.0 * scale;
	let pad_y = 8.0 * scale;
	let box_w = ui.measure_text(&text, layout)? + 2.0 * pad_x;
	let box_h = layout.size + 2.0 * pad_y;
	let margin = 16.0 * scale;
	let box_x = (viewport.x + viewport.w as i32) as f32 - margin - box_w;
	let box_y = (viewport.y + viewport.h as i32) as f32 - bottom_inset - margin - box_h;
	ui.rect(
		PixelRect::new(
			box_x.round() as i32,
			box_y.round() as i32,
			box_w.ceil().max(1.0) as u32,
			box_h.ceil().max(1.0) as u32,
		),
		Color::new_from_u32(0x00000080),
		6.0 * scale,
	)?;
	ui.text_at(
		&text,
		[box_x + pad_x, box_y + pad_y + layout.size * 0.85],
		layout,
		Color::new_from_u32(0xFFFFFFFF),
		viewport,
	)?;
	Ok(())
}
