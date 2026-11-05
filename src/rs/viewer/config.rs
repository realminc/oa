//! Viewer configuration types. Donor: `ViewerConfig`, `ViewerMode`, etc.

use crate::input::{Key, MotionSpeed};
use crate::ui::types::UiStyle;
pub use crate::viewer::types::ViewerConfig;

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

impl Default for ViewerConfig {
	fn default() -> Self {
		Self {
			mode: ViewerMode::Auto,
			path: String::new(),
			title: String::from("Viewer"),
			width: 1280,
			height: 720,
			style: UiStyle::viewer_dark(),
			motion_speed: MotionSpeed::Fast,
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
			key_quit: Key::Escape,
			key_quit_q: Key::Q,
			key_red: Key::Num1,
			key_green: Key::Num2,
			key_blue: Key::Num3,
			key_alpha: Key::Num4,
			key_rgb: Key::Num5,
			key_zoom_in: Key::Equals,
			key_zoom_out: Key::Minus,
			key_zoom_fit: Key::Num0,
			key_zoom_100: Key::Num9,
			key_canvas_background: Key::B,
			key_canvas_grid: Key::G,
		}
	}
}
