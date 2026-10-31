//! Canvas background and grid overlays. Donor: `oa::Viewer::renderVisualWorkspace`.

use super::config::{ViewerCanvasBackground, ViewerConfig};
use crate::ui::{TextLayoutConfig, Ui, navigation::Navigation, types::PixelRect};
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

/// Draw optional status. Startup help belongs in the log, as in the donor.
pub fn draw_hud(
	ui: &mut Ui<'_>,
	viewport: PixelRect,
	navigation: &Navigation,
	mode: &str,
	config: &ViewerConfig,
) -> Result<()> {
	if !config.show_stats {
		return Ok(());
	}
	ui.text_at(
		&format!("{mode}  {:+.0}%", navigation.zoom() * 100.0),
		[12.0, 60.0],
		TextLayoutConfig {
			size: 12.0,
			..Default::default()
		},
		config.style.text,
		viewport,
	)?;
	Ok(())
}
