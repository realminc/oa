//! Audio mode: open AudioPlayer and render waveform / spectrum / mel.
//!
//! Donor: `oa::Viewer::openAudio`, `oa::Viewer::renderAudio`.

use super::{
	audio_view::AudioAnalysis,
	config::{ViewerAudioView, ViewerConfig},
};
use crate::audio::AudioPlayerConfig;
use crate::ui::{Ui, types::PixelRect};
use crate::{AudioPlayer, Engine, Result};

/// Open an AudioPlayer for the path in config.
pub fn open(engine: &Engine, config: &ViewerConfig) -> Result<AudioPlayer> {
	let cfg = AudioPlayerConfig {
		uri: config.path.as_str().into(),
		loop_playback: config.loop_media,
		ring_milliseconds: config.audio_ring_ms,
	};
	AudioPlayer::open(engine, cfg)
}

/// Render the audio visualization.
pub fn render(
	ui: &mut Ui<'_>,
	_player: Option<&mut AudioPlayer>,
	analysis: Option<&AudioAnalysis>,
	viewport: PixelRect,
	config: &ViewerConfig,
) -> Result<()> {
	let Some(analysis) = analysis else {
		return Ok(());
	};
	let vis_rect = PixelRect::new(
		viewport.x,
		viewport.y,
		viewport.w,
		(viewport.h as f32 * 0.6) as u32,
	);
	if vis_rect.w == 0 || vis_rect.h == 0 {
		return Ok(());
	}
	match config.audio_view {
		ViewerAudioView::Waveform => {
			if let Some(env) = &analysis.envelope
				&& let Ok(values) = env.read_f32()
			{
				let y_range = values.iter().copied().fold(0.0_f32, f32::max).max(1e-6);
				ui.plot_line_at(
					&values,
					vis_rect,
					-y_range,
					y_range,
					crate::Color::new(0.388, 0.600, 0.945, 1.0),
					1.5,
				)?;
			}
		}
		ViewerAudioView::Spectrum => {
			if let Some(spec) = &analysis.spectrum
				&& let Ok(values) = spec.read_f32()
			{
				let y_max = values
					.iter()
					.copied()
					.fold(f32::NEG_INFINITY, f32::max)
					.max(1e-6);
				let y_min = values
					.iter()
					.copied()
					.fold(f32::INFINITY, f32::min)
					.min(-1e-6);
				ui.plot_line_at(
					&values,
					vis_rect,
					y_min,
					y_max,
					crate::Color::new(0.545, 0.388, 0.945, 1.0),
					1.5,
				)?;
			}
		}
		ViewerAudioView::Mel => {
			if let Some(mel) = &analysis.mel
				&& let Ok(values) = mel.read_f32()
			{
				let y_max = values
					.iter()
					.copied()
					.fold(f32::NEG_INFINITY, f32::max)
					.max(1e-6);
				let y_min = values
					.iter()
					.copied()
					.fold(f32::INFINITY, f32::min)
					.min(-1e-6);
				ui.plot_line_at(
					&values,
					vis_rect,
					y_min,
					y_max,
					crate::Color::new(0.388, 0.945, 0.600, 1.0),
					1.5,
				)?;
			}
		}
	}
	Ok(())
}
