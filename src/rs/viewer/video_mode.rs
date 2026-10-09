//! Video mode: retain native decoded frames for GPU presentation.
//!
//! Donor: `oa::Viewer::openVideo`, `oa::Viewer::renderVideo`.

use super::config::ViewerConfig;
use crate::ui::{Ui, navigation::Navigation, types::PixelRect};
use crate::video::VideoPlayerConfig;
use crate::{Engine, Error, Result, VideoPlayer};

/// Open a VideoPlayer from the config path.
pub fn open(engine: &Engine, config: &ViewerConfig) -> Result<VideoPlayer> {
	let player_config = VideoPlayerConfig {
		loop_playback: config.loop_media,
		start_playing: config.start_playing,
		..VideoPlayerConfig::default()
	};
	VideoPlayer::open_native(engine, &config.path, player_config)
}

/// Render the current decoded video frame when a Texture bridge is available.
pub fn render(
	ui: &mut Ui<'_>,
	player: Option<&mut VideoPlayer>,
	nav: &Navigation,
	_viewport: PixelRect,
	_config: &ViewerConfig,
) -> Result<()> {
	let Some(player) = player else {
		return Ok(());
	};
	let frame = match player.current_frame() {
		Ok(f) => f,
		Err(_) => return Ok(()),
	};
	let dst = PixelRect {
		x: nav.pan_x() as i32,
		y: nav.pan_y() as i32,
		w: ((frame.width() as f32) * nav.zoom()).max(1.0) as u32,
		h: ((frame.height() as f32) * nav.zoom()).max(1.0) as u32,
	};
	if dst.w == 0 || dst.h == 0 {
		return Ok(());
	}
	let texture = frame.as_texture().ok_or_else(|| {
		Error::missing_capability("Viewer video requires a GPU native-plane-to-Texture conversion path")
	})?;
	ui.image_at(texture, dst)?;
	Ok(())
}
