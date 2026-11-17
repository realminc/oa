//! Video mode: retain native decoded frames for GPU presentation.
//!
//! Donor: `oa::Viewer::openVideo`, `oa::Viewer::renderVideo`.

use super::config::ViewerConfig;
use crate::input::Navigation;
use crate::ui::{Ui, types::PixelRect};
use crate::video::VideoPlayerConfig;
use crate::{Engine, Result, VideoPlayer};

/// Open a VideoPlayer from the config path.
pub fn open(engine: &Engine, config: &ViewerConfig) -> Result<VideoPlayer> {
	let player_config = VideoPlayerConfig {
		loop_playback: config.loop_media,
		start_playing: config.start_playing,
		..VideoPlayerConfig::default()
	};
	VideoPlayer::new_native(engine, &config.path, player_config)
}

/// Render the current decoded video frame through the native-plane Texture path.
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
	let (width, height) = match player.current_frame() {
		Ok(f) => (f.width(), f.height()),
		Err(_) => return Ok(()),
	};
	let dst = PixelRect {
		x: nav.pan_x().round() as i32,
		y: nav.pan_y().round() as i32,
		w: ((width as f32) * nav.zoom()).round().max(1.0) as u32,
		h: ((height as f32) * nav.zoom()).round().max(1.0) as u32,
	};
	if dst.w == 0 || dst.h == 0 {
		return Ok(());
	}
	let texture = player.current_frame_texture()?;
	ui.image_at(&texture, dst)?;
	Ok(())
}
