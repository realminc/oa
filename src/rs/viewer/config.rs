//! Viewer configuration types. Donor: `ViewerConfig`, `ViewerMode`, etc.

use crate::ui::types::{UiKey, UiStyle};

/// Media source mode for the Viewer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewerMode {
	/// Probe path as image → video → audio.
	#[default]
	Auto,
	Image,
	Video,
	Audio,
	/// Attach a `ViewerLiveSource` and render its output each frame.
	Live,
}

/// Audio visualization mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewerAudioView {
	#[default]
	Waveform,
	Spectrum,
	Mel,
}

/// Canvas background style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewerCanvasBackground {
	#[default]
	Dark,
	Gradient,
}

/// Full configuration block. Donor: `oa::ViewerConfig`.
#[derive(Clone, Debug)]
pub struct ViewerConfig {
	pub mode: ViewerMode,
	pub path: String,
	pub title: String,
	pub width: u32,
	pub height: u32,
	pub style: UiStyle,
	pub show_help: bool,
	pub show_stats: bool,
	pub show_timeline: bool,
	pub vsync: bool,
	pub loop_media: bool,
	pub start_playing: bool,
	pub auto_hide_controls: bool,
	pub controls_hide_delay_ms: u32,
	pub canvas_background: ViewerCanvasBackground,
	pub show_canvas_grid: bool,
	pub audio_view: ViewerAudioView,
	pub audio_waveform_bins: u32,
	pub audio_analysis_frames: u32,
	pub audio_fft_size: u32,
	pub audio_hop_size: u32,
	pub audio_mel_bins: u32,
	pub audio_ring_ms: u32,
	// Keyboard bindings.
	pub key_quit: UiKey,
	pub key_quit_q: UiKey,
	pub key_red: UiKey,
	pub key_green: UiKey,
	pub key_blue: UiKey,
	pub key_alpha: UiKey,
	pub key_rgb: UiKey,
	pub key_zoom_in: UiKey,
	pub key_zoom_out: UiKey,
	pub key_zoom_fit: UiKey,
	pub key_zoom_100: UiKey,
	pub key_canvas_background: UiKey,
	pub key_canvas_grid: UiKey,
}

impl Default for ViewerConfig {
	fn default() -> Self {
		Self {
			mode: ViewerMode::Auto,
			path: String::new(),
			title: String::from("Viewer"),
			width: 1280,
			height: 720,
			style: UiStyle::viewer_dark(),
			show_help: true,
			show_stats: false,
			show_timeline: true,
			vsync: true,
			loop_media: true,
			start_playing: true,
			auto_hide_controls: true,
			controls_hide_delay_ms: 2_000,
			canvas_background: ViewerCanvasBackground::Gradient,
			show_canvas_grid: true,
			audio_view: ViewerAudioView::Waveform,
			audio_waveform_bins: 2048,
			audio_analysis_frames: 2048,
			audio_fft_size: 1024,
			audio_hop_size: 256,
			audio_mel_bins: 80,
			audio_ring_ms: 500,
			key_quit: UiKey::Escape,
			key_quit_q: UiKey::Q,
			key_red: UiKey::Num1,
			key_green: UiKey::Num2,
			key_blue: UiKey::Num3,
			key_alpha: UiKey::Num4,
			key_rgb: UiKey::Num5,
			key_zoom_in: UiKey::Equals,
			key_zoom_out: UiKey::Minus,
			key_zoom_fit: UiKey::Num0,
			key_zoom_100: UiKey::Num9,
			key_canvas_background: UiKey::B,
			key_canvas_grid: UiKey::G,
		}
	}
}
