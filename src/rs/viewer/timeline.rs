//! Donor viewer.cpp transport layout and playback controls.

use super::{MediaState, config::ViewerConfig};
use crate::Result;
use crate::ui::{PixelRect, TextLayoutConfig, Ui, UiIcon, UiIconButtonStyle};

/// Timeline scrubber row. The title row sits directly above it and the
/// control row directly below it within the bottom transport block.
pub(super) fn geometry(viewport: PixelRect, scale: f32) -> PixelRect {
	let px = |value: f32| (value * scale).round().max(1.0) as i32;
	let margin = px(if viewport.w as f32 / scale >= 550.0 {
		24.0
	} else {
		12.0
	});
	let inset = margin + px(12.0);
	PixelRect::new(
		inset,
		(viewport.h as i32 - margin - px(4.0) - px(34.0) * 2).max(0),
		(viewport.w as i32 - inset * 2).max(1) as u32,
		px(34.0) as u32,
	)
}

fn timestamp(microseconds: u64) -> String {
	let seconds = microseconds / 1_000_000;
	if seconds >= 3600 {
		format!(
			"{}:{:02}:{:02}",
			seconds / 3600,
			seconds / 60 % 60,
			seconds % 60
		)
	} else {
		format!("{}:{:02}", seconds / 60, seconds % 60)
	}
}

#[derive(Default)]
pub(super) struct TransportState {
	pub show_remaining: bool,
}

pub(super) fn render(
	ui: &mut Ui<'_>,
	media: &mut MediaState,
	config: &ViewerConfig,
	viewport: PixelRect,
	state: &mut TransportState,
) -> Result<()> {
	let duration = media.duration_us();
	if duration == 0 {
		return Ok(());
	}
	let scale = ui.content_scale();
	let bar = geometry(viewport, scale);
	let size = (34.0 * scale).round().max(1.0) as u32;
	let gap = (4.0 * scale).round().max(1.0) as i32;
	let controls_y = bar.y + bar.h as i32 + gap;

	// Title row.
	ui.text_at(
		&config.title,
		[bar.x as f32, (bar.y - size as i32) as f32 + 24.0 * scale],
		TextLayoutConfig {
			font: crate::ui::FontId::SansSemibold,
			size: 18.0 * scale,
			..Default::default()
		},
		config.style.text,
		PixelRect::new(bar.x, (bar.y - size as i32).max(0), bar.w, size),
	)?;

	// Timeline scrubber row.
	let position = media.position_us();
	let mut fraction = (position as f64 / duration as f64).clamp(0.0, 1.0) as f32;
	if ui.timeline("viewer-transport", bar, &mut fraction, None) {
		media.seek_fraction(fraction)?;
	}

	// Control row: play, step, volume, elapsed/total, then settings at the
	// right edge.
	let play = PixelRect::new(bar.x, controls_y, size, size);
	if ui.icon_button(
		"Play/Pause",
		play,
		if media.is_playing() {
			UiIcon::Pause
		} else {
			UiIcon::Play
		},
		true,
		UiIconButtonStyle::Header,
	) {
		media.toggle_play()?;
	}
	let previous = PixelRect::new(play.x + size as i32 + gap, controls_y, size, size);
	if ui.icon_button(
		"Step backward",
		previous,
		UiIcon::Previous,
		true,
		UiIconButtonStyle::Header,
	) {
		media.step_temporal(-1)?;
	}
	let next = PixelRect::new(previous.x + size as i32 + gap, controls_y, size, size);
	if ui.icon_button(
		"Step forward",
		next,
		UiIcon::Next,
		true,
		UiIconButtonStyle::Header,
	) {
		media.step_temporal(1)?;
	}
	let volume = PixelRect::new(next.x + size as i32 + gap, controls_y, size, size);
	let muted = matches!(media, MediaState::Audio { player, .. } if player.as_ref().as_ref().is_some_and(|player| player.is_muted()));
	let audible =
		matches!(media, MediaState::Audio { player, .. } if player.as_ref().as_ref().is_some());
	if ui.icon_button(
		if muted { "Unmute" } else { "Mute" },
		volume,
		if muted { UiIcon::Muted } else { UiIcon::Volume },
		audible,
		UiIconButtonStyle::Header,
	) && let MediaState::Audio { player, .. } = media
		&& let Some(player) = player.as_ref().as_ref()
	{
		player.set_muted(!muted)?;
	}
	let end = if state.show_remaining {
		format!("-{}", timestamp(duration.saturating_sub(position)))
	} else {
		timestamp(duration)
	};
	let elapsed = format!("{} / {}", timestamp(position), end);
	let text_x = volume.x + size as i32 + (10.0 * scale).round() as i32;
	let text_width = (elapsed.len() as f32 * 8.0 * scale).round().max(1.0) as u32;
	if ui.text_button(
		"Toggle duration or remaining",
		PixelRect::new(text_x, controls_y, text_width, size),
		"",
		config.style.text,
		true,
	) {
		state.show_remaining = !state.show_remaining;
	}
	ui.text_at(
		&elapsed,
		[text_x as f32, controls_y as f32 + 22.0 * scale],
		TextLayoutConfig {
			font: crate::ui::FontId::SansSemibold,
			size: 12.0 * scale,
			..Default::default()
		},
		config.style.text,
		viewport,
	)?;
	let settings = PixelRect::new(bar.x + bar.w as i32 - size as i32, controls_y, size, size);
	if ui.icon_button(
		"Playback settings",
		settings,
		UiIcon::Settings,
		true,
		UiIconButtonStyle::Header,
	) {
		ui.open_popup_at("viewer-window-menu", settings);
	}
	Ok(())
}
