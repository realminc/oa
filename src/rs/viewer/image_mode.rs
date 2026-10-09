//! Image mode: load and display a GPU texture with channel view modes.
//!
//! Donor: `oa::Viewer::openImage`, `oa::Viewer::renderImage`.

use super::config::ViewerConfig;
use crate::render::texture_from_image;
use crate::ui::{
	Ui,
	navigation::Navigation,
	types::{PixelRect, UiEvent, UiEventKind},
};
use crate::{Engine, Result, Texture};

/// Channel display mode — mirrors `oa::Viewer::ImageViewMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChannelMode {
	#[default]
	Rgb,
	Red,
	Green,
	Blue,
	Alpha,
}

/// Load and GPU-upload a texture from a file path.
pub fn load_texture(engine: &Engine, path: &str) -> Result<Texture> {
	let image = crate::image::decode_file(engine, path, crate::ImageFormat::Rgba)?;
	texture_from_image(&image)
}

/// Draw the image or its channel view into the viewport.
pub fn render(
	ui: &mut Ui<'_>,
	texture: Option<&Texture>,
	nav: &Navigation,
	channel_mode: &ChannelMode,
	viewport: PixelRect,
	_config: &ViewerConfig,
) -> Result<()> {
	let _ = viewport;
	let Some(texture) = texture else {
		return Ok(());
	};
	let dst = nav_dst_rect(nav, texture.width() as f32, texture.height() as f32);
	if dst.w == 0 || dst.h == 0 {
		return Ok(());
	}
	let channel = match channel_mode {
		ChannelMode::Rgb => 0,
		ChannelMode::Red => 1,
		ChannelMode::Green => 2,
		ChannelMode::Blue => 3,
		ChannelMode::Alpha => 4,
	};
	ui.image_at_channel(texture, dst, channel)?;
	Ok(())
}

/// Map navigation pan/zoom to a destination pixel rect for the given content size.
fn nav_dst_rect(nav: &Navigation, content_w: f32, content_h: f32) -> PixelRect {
	PixelRect {
		x: nav.pan_x() as i32,
		y: nav.pan_y() as i32,
		w: (content_w * nav.zoom()).max(1.0) as u32,
		h: (content_h * nav.zoom()).max(1.0) as u32,
	}
}

/// Route keyboard events to channel mode.
pub fn handle_event(ev: &UiEvent, mode: &mut ChannelMode, config: &ViewerConfig) {
	if ev.kind != UiEventKind::KeyDown {
		return;
	}
	if ev.key == config.key_red {
		*mode = ChannelMode::Red;
	}
	if ev.key == config.key_green {
		*mode = ChannelMode::Green;
	}
	if ev.key == config.key_blue {
		*mode = ChannelMode::Blue;
	}
	if ev.key == config.key_alpha {
		*mode = ChannelMode::Alpha;
	}
	if ev.key == config.key_rgb {
		*mode = ChannelMode::Rgb;
	}
}
