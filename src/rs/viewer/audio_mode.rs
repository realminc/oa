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
	let mut player = AudioPlayer::open(engine, cfg)?;
	// Donor openAudio honors the viewer's initial transport state.
	if config.start_playing
		&& let Err(error) = player.play()
	{
		let _ = player.close();
		return Err(error);
	}
	Ok(player)
}

/// Render the audio visualization.
pub fn render(
	ui: &mut Ui<'_>,
	player: Option<&mut AudioPlayer>,
	analysis: Option<&AudioAnalysis>,
	viewport: PixelRect,
	config: &ViewerConfig,
) -> Result<()> {
	let Some(analysis) = analysis else {
		return Ok(());
	};
	let scale = ui.content_scale();
	let timeline = super::timeline::geometry(viewport, scale);
	let top = ((43.0 + 24.0) * scale).round() as i32;
	let bottom = if config.show_timeline {
		timeline.y - (46.0 * scale).round() as i32
	} else {
		viewport.h as i32 - (24.0 * scale).round() as i32
	};
	let vis_rect = PixelRect::new(
		timeline.x,
		top,
		timeline.w,
		bottom.saturating_sub(top).max(1) as u32,
	);
	let mut fraction = player.as_ref().map_or(0.0, |player| {
		player.position_us() as f64 / player.duration_us().max(1) as f64
	}) as f32;
	match config.audio_view {
		ViewerAudioView::Waveform => {
			if let Some(envelope) = &analysis.envelope
				&& ui.waveform_matrix_timeline("viewer-waveform", vis_rect, envelope, &mut fraction)?
				&& let Some(player) = player
			{
				player.seek((fraction as f64 * player.duration_us() as f64) as u64)?;
			}
		}
		ViewerAudioView::Spectrum => {
			if let Some(spectrum) = &analysis.spectrum {
				ui.heatmap_matrix_at(spectrum, vis_rect, [-12.0, 6.5], 1)?;
			}
		}
		ViewerAudioView::Mel => {
			if let Some(mel) = &analysis.mel {
				ui.heatmap_matrix_at(mel, vis_rect, [-12.0, 6.5], 0)?;
			}
		}
	}
	ui.mask_rounded_rect(vis_rect, 15.0 * scale, config.style.background)?;
	ui.rect_outline_rounded_at(
		vis_rect,
		config.style.border_strong,
		scale.round().max(1.0) as u32,
		vis_rect,
		15.0 * scale,
	);
	Ok(())
}
