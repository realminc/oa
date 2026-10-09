//! Timeline scrubber bar. Donor: `oa::Viewer::renderTimeline`.

use super::{MediaState, config::ViewerConfig};
use crate::ui::{Ui, types::PixelRect};
use crate::{Color, Result};

const TIMELINE_HEIGHT: u32 = 32;

/// Draw the playback timeline at the bottom of the viewport.
pub fn render(
	ui: &mut Ui<'_>,
	media: &mut MediaState,
	_config: &ViewerConfig,
	viewport: PixelRect,
) -> Result<()> {
	let bar_y = (viewport.y + viewport.h as i32) - TIMELINE_HEIGHT as i32;
	if bar_y < viewport.y {
		return Ok(());
	}
	let bar = PixelRect::new(viewport.x, bar_y, viewport.w, TIMELINE_HEIGHT);
	// Background.
	ui.rect_raw(bar, Color::new(0.08, 0.08, 0.10, 0.85), bar, 0.0);
	// Progress fill.
	let dur = media.duration_us();
	let pos = media.position_us();
	let fraction = if dur > 0 {
		(pos as f64 / dur as f64) as f32
	} else {
		0.0
	};
	let fill_w = ((bar.w as f32) * fraction.clamp(0.0, 1.0)) as u32;
	if fill_w > 0 {
		let fill = PixelRect::new(bar.x, bar.y, fill_w, bar.h);
		ui.rect_raw(fill, Color::new(0.388, 0.400, 0.945, 1.0), bar, 0.0);
	}
	// Detect scrub click.
	let input = ui.input().clone();
	if input.l_pressed && bar.contains(input.pointer_x, input.pointer_y) {
		let fx = (input.pointer_x - bar.x as f32) / bar.w.max(1) as f32;
		media.seek_fraction(fx)?;
	}
	Ok(())
}
