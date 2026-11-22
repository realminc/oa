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

/// Only the bottom transport block owns timeline pointer gestures. The rest
/// of a video window remains the same navigable canvas as an image window.
pub(super) fn owns_pointer(viewport: PixelRect, scale: f32, y: f32) -> bool {
	let bar = geometry(viewport, scale);
	let title_top = bar.y - (34.0 * scale).round().max(1.0) as i32;
	y >= title_top as f32 && y < viewport.h as f32
}

#[derive(Default)]
pub(super) struct TransportState {
	pub show_remaining: bool,
	scrubbing: bool,
	resume_after_scrub: bool,
}

#[cfg(test)]
#[path = "../../../../test/rs/render/viewer_timeline_unit.rs"]
mod tests;

pub(super) fn render(
	ui: &mut Ui<'_>,
	media: &mut MediaState,
	config: &ViewerConfig,
	viewport: PixelRect,
	state: &mut TransportState,
) -> Result<()> {
	let duration = media.duration_us();
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
	if duration != 0 {
		let mut fraction = (position as f64 / duration as f64).clamp(0.0, 1.0) as f32;
		let mut active = false;
		let seek = ui.timeline("viewer-transport", bar, &mut fraction, Some(&mut active));
		if active && !state.scrubbing {
			state.resume_after_scrub = media.is_playing();
			media.pause_playback()?;
		}
		if seek {
			media.seek_fraction(fraction)?;
		}
		if !active
			&& state.scrubbing
			&& std::mem::take(&mut state.resume_after_scrub)
			&& !media.is_playing()
		{
			media.toggle_play()?;
		}
		state.scrubbing = active;
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
	let audible = media.audio_player();
	let muted = audible.is_some_and(|player| player.is_muted());
	if ui.icon_button(
		if muted { "Unmute" } else { "Mute" },
		volume,
		if muted { UiIcon::Muted } else { UiIcon::Volume },
		audible.is_some(),
		UiIconButtonStyle::Header,
	) && let Some(player) = audible
	{
		player.set_muted(!muted)?;
	}
	let end = if duration == 0 {
		"–".to_owned()
	} else if state.show_remaining {
		format!("-{}", timestamp(duration.saturating_sub(position)))
	} else {
		timestamp(duration)
	};
	let elapsed = format!("{} / {}", timestamp(position), end);
	let space = (10.0 * scale).round() as i32;
	let mut text_x = volume.x + size as i32 + space;
	let text_width = (elapsed.len() as f32 * 8.0 * scale).round().max(1.0) as u32;
	let settings = PixelRect::new(bar.x + bar.w as i32 - size as i32, controls_y, size, size);
	if let Some(player) = audible {
		let available = settings.x - text_x - text_width as i32 - 2 * space;
		if available >= (64.0 * scale) as i32 {
			let slider_width = available.min((96.0 * scale).round() as i32) as u32;
			let slider = PixelRect::new(text_x, controls_y, slider_width, size);
			let mut gain = player.volume();
			if ui.timeline("Volume", slider, &mut gain, None) {
				player.set_volume(gain)?;
				player.set_muted(false)?;
			}
			text_x += slider_width as i32 + space;
			let meter_width = (34.0 * scale).round().max(1.0) as u32;
			if settings.x - text_x >= (text_width + meter_width) as i32 + 2 * space {
				let meter_height = (4.0 * scale).round().max(1.0) as u32;
				let meter = PixelRect::new(
					text_x,
					controls_y + (size - meter_height) as i32 / 2,
					meter_width,
					meter_height,
				);
				ui.rect_raw(
					meter,
					config.style.surface,
					viewport,
					meter_height as f32 / 2.0,
				);
				let w = (meter_width as f32 * player.output_peak().sqrt()).round() as u32;
				if w > 0 {
					ui.rect_raw(
						PixelRect { w, ..meter },
						config.style.accent,
						viewport,
						meter_height as f32 / 2.0,
					);
				}
				text_x += meter_width as i32 + space;
			}
		}
	}
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
