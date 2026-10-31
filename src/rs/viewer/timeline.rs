//! Donor viewer.cpp transport layout and playback controls.

use super::{MediaState, config::ViewerConfig};
use crate::Result;
use crate::ui::{PixelRect, TextLayoutConfig, Ui, UiIcon, UiIconButtonStyle};

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
		(viewport.h as i32 - margin - px(3.0) - px(34.0) * 2).max(0),
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
	let margin = (if viewport.w as f32 / scale >= 550.0 {
		24.0
	} else {
		12.0
	} * scale)
		.round() as i32;
	let settings = PixelRect::new(
		viewport.w as i32 - margin - size as i32,
		(bar.y - size as i32).max(0),
		size,
		size,
	);
	let volume = PixelRect::new(settings.x - size as i32, settings.y, size, size);
	let muted = matches!(media, MediaState::Audio { player, .. } if player.as_ref().as_ref().is_some_and(|player| player.is_muted()));
	if ui.icon_button(
		"Adjust volume",
		volume,
		if muted { UiIcon::Muted } else { UiIcon::Volume },
		true,
		UiIconButtonStyle::Header,
	) {
		ui.open_popup_at("viewer-volume-menu", volume);
	}
	if ui.icon_button(
		"Playback settings",
		settings,
		UiIcon::Settings,
		true,
		UiIconButtonStyle::Header,
	) {
		ui.open_popup_at("viewer-window-menu", settings);
	}
	ui.text_at(
		&config.title,
		[bar.x as f32, settings.y as f32 + 24.0 * scale],
		TextLayoutConfig {
			font: crate::ui::FontId::SansSemibold,
			size: 18.0 * scale,
			..Default::default()
		},
		config.style.text,
		PixelRect::new(bar.x, settings.y, (volume.x - bar.x).max(1) as u32, size),
	)?;
	let position = media.position_us();
	let text_config = TextLayoutConfig {
		font: crate::ui::FontId::SansSemibold,
		size: 12.0 * scale,
		..Default::default()
	};
	let y = (bar.y + bar.h as i32) as f32 + 22.0 * scale;
	ui.text_at(
		&timestamp(position),
		[bar.x as f32, y],
		text_config,
		config.style.text,
		viewport,
	)?;
	let end = if state.show_remaining {
		format!("-{}", timestamp(duration.saturating_sub(position)))
	} else {
		timestamp(duration)
	};
	let end_rect = PixelRect::new(
		(bar.x + bar.w as i32 - (80.0 * scale).round() as i32).max(0),
		bar.y + bar.h as i32,
		(80.0 * scale).round().max(1.0) as u32,
		size,
	);
	if ui.text_button(
		"Toggle duration or remaining",
		end_rect,
		"",
		config.style.text,
		true,
	) {
		state.show_remaining = !state.show_remaining;
	}
	ui.text_at(
		&end,
		[
			(bar.x as f32 + bar.w as f32 - end.len() as f32 * 8.0 * scale).max(0.0),
			y,
		],
		text_config,
		config.style.text,
		viewport,
	)?;
	let mut fraction = (position as f64 / duration as f64).clamp(0.0, 1.0) as f32;
	if ui.timeline("viewer-transport", bar, &mut fraction, None) {
		media.seek_fraction(fraction)?;
	}
	let large = (56.0 * scale).round().max(1.0) as u32;
	let small = size;
	let gap = (12.0 * scale).round().max(4.0) as i32;
	let row_width = small * 2 + large + gap as u32 * 2;
	let left = (viewport.w as i32 - row_width as i32).max(0) / 2;
	let top = (viewport.h as i32 - large as i32).max(0) / 2;
	if ui.icon_button(
		"Previous frame",
		PixelRect::new(left, top + (large - small) as i32 / 2, small, small),
		UiIcon::Previous,
		true,
		UiIconButtonStyle::Overlay,
	) {
		media.step_temporal(-1)?;
	}
	let play_x = left + small as i32 + gap;
	if ui.icon_button(
		"Play/Pause",
		PixelRect::new(play_x, top, large, large),
		if media.is_playing() {
			UiIcon::Pause
		} else {
			UiIcon::Play
		},
		true,
		UiIconButtonStyle::Overlay,
	) {
		media.toggle_play()?;
	}
	if ui.icon_button(
		"Next frame",
		PixelRect::new(
			play_x + large as i32 + gap,
			top + (large - small) as i32 / 2,
			small,
			small,
		),
		UiIcon::Next,
		true,
		UiIconButtonStyle::Overlay,
	) {
		media.step_temporal(1)?;
	}
	Ok(())
}
