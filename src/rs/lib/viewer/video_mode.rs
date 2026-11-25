//! Video mode: retain native decoded frames for GPU presentation.
//!
//! Donor: `oa::Viewer::openVideo`, `oa::Viewer::renderVideo`.

use super::config::ViewerConfig;
use crate::input::Navigation;
use crate::ui::{Ui, types::PixelRect};
use crate::{MediaPlayer, Result, Runtime};

/// Open video with its default embedded audio track when available.
pub(super) fn open(runtime: &Runtime, config: &ViewerConfig) -> Result<MediaPlayer> {
	let mut player = MediaPlayer::open_viewer(runtime, std::path::Path::new(&config.path))?;
	player.set_looping(config.loop_media);
	if config.start_playing {
		player.play()?;
	}
	Ok(player)
}

/// Render the current decoded video frame on the shared image canvas.
pub(super) fn render(
	ui: &mut Ui<'_>,
	player: Option<&mut MediaPlayer>,
	nav: &Navigation,
) -> Result<()> {
	let Some(player) = player else {
		return Ok(());
	};
	let Some(frame) = player.current_viewer_frame()? else {
		return Ok(());
	};
	let (width, height) = (frame.width(), frame.height());
	let dst = PixelRect {
		x: nav.pan_x().round() as i32,
		y: nav.pan_y().round() as i32,
		w: ((width as f32) * nav.zoom()).round().max(1.0) as u32,
		h: ((height as f32) * nav.zoom()).round().max(1.0) as u32,
	};
	if let Some(texture) = frame.as_texture() {
		ui.image_at(texture, dst)?;
	} else {
		ui.video_planes_at(&frame, dst)?;
	}
	Ok(())
}
