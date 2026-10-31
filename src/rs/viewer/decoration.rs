//! Donor viewerApplication.cpp header geometry and popup actions.
//! Mechanical port; SDL requests borrow the window retained by Presenter.

use super::{MediaState, ViewerAudioView, ViewerCanvasBackground, ViewerConfig};
use crate::Result;
use crate::runtime::{SdlWindow, window_decoration as metrics};
use crate::ui::{PixelRect, Ui, UiIcon, UiIconButtonStyle, UiPopupConfig, navigation::Navigation};

#[allow(clippy::too_many_arguments)]
pub(super) fn render(
	ui: &mut Ui<'_>,
	window: &SdlWindow,
	media: &mut MediaState,
	nav: &mut Navigation,
	config: &mut ViewerConfig,
	extent: [u32; 2],
	scale: [f32; 2],
) -> Result<bool> {
	let [w, _] = extent;
	let px = |value: i32| (value as f32 * scale[0]).round().max(1.0) as i32;
	let size = px(metrics::BUTTON_SIZE) as u32;
	let y = (metrics::OUTER_MARGIN as f32 * scale[1]).round().max(1.0) as i32;
	let inset = px(metrics::OUTER_MARGIN);
	let close_x = w as i32 - inset - size as i32;
	let maximize_x = close_x - px(metrics::CONTROL_SPACING) - size as i32;
	let minimize_x = maximize_x - px(metrics::CONTROL_SPACING) - size as i32;
	let menu_x = minimize_x - px(metrics::MENU_SPACING) - size as i32;
	let fullscreen = window.window.fullscreen_state() != sdl3::video::FullscreenType::Off;
	let maximized = window.window.is_maximized();
	let mut running = true;
	if window.client_frame {
		if ui.icon_button(
			"Fullscreen",
			PixelRect::new(inset, y, size, size),
			if fullscreen {
				UiIcon::ExitFullscreen
			} else {
				UiIcon::Fullscreen
			},
			true,
			UiIconButtonStyle::Header,
		) {
			window.toggle_fullscreen()?;
		}
		let menu = PixelRect::new(menu_x, y, size, size);
		if ui.icon_button(
			"Main menu",
			menu,
			UiIcon::Menu,
			true,
			UiIconButtonStyle::Header,
		) {
			ui.open_popup_at("viewer-window-menu", menu);
		}
		if ui.icon_button(
			"Minimize",
			PixelRect::new(minimize_x, y, size, size),
			UiIcon::Minimize,
			true,
			UiIconButtonStyle::WindowControl,
		) {
			window.minimize()?;
		}
		if ui.icon_button(
			"Maximize",
			PixelRect::new(maximize_x, y, size, size),
			if maximized {
				UiIcon::Restore
			} else {
				UiIcon::Maximize
			},
			true,
			UiIconButtonStyle::WindowControl,
		) {
			window.toggle_maximized()?;
		}
		running = !ui.icon_button(
			"Close",
			PixelRect::new(close_x, y, size, size),
			UiIcon::Close,
			true,
			UiIconButtonStyle::WindowControl,
		);
	}
	let visual = media.has_visual_content();
	let audio = matches!(media, MediaState::Audio { .. });
	let count = 7 + usize::from(visual) * 3 + usize::from(audio) * 3;
	let scale = (scale[0] + scale[1]) * 0.5;
	if ui.begin_popup(
		"viewer-window-menu",
		UiPopupConfig {
			width: 224.0 * scale,
			height: (12.0
				+ count as f32 * (config.style.font_size + 8.0)
				+ count.saturating_sub(1) as f32 * 4.0)
				* scale,
			gap: 6.0 * scale,
			padding: 6.0 * scale,
		},
	) {
		if ui.menu_item("Fit to Window", false, visual) {
			nav.fit_to_window_animated(true)?;
		}
		if ui.menu_item("Actual Size", false, visual) {
			nav.zoom_to_100();
		}
		if ui.menu_item("Loop", config.loop_media, media.has_timeline()) {
			config.loop_media = !config.loop_media;
			match media {
				MediaState::Video { player } => {
					if let Some(player) = player.as_mut() {
						player.set_looping(config.loop_media)?;
					}
				}
				MediaState::Audio { player, .. } => {
					if let Some(player) = player.as_ref() {
						player.set_loop(config.loop_media);
					}
				}
				_ => {}
			}
		}
		if ui.menu_item("Timeline", config.show_timeline, media.has_timeline()) {
			config.show_timeline = !config.show_timeline;
		}
		if ui.menu_item("Statistics", config.show_stats, true) {
			config.show_stats = !config.show_stats;
		}
		if ui.menu_item(
			&format!("Animation: {:?}", config.motion_speed),
			false,
			true,
		) {
			config.motion_speed = config.motion_speed.next();
		}
		if visual {
			if ui.menu_item(
				"Dark Canvas (B)",
				config.canvas_background == ViewerCanvasBackground::Dark,
				true,
			) {
				config.canvas_background = ViewerCanvasBackground::Dark;
			}
			if ui.menu_item(
				"Gradient Canvas (B)",
				config.canvas_background == ViewerCanvasBackground::Gradient,
				true,
			) {
				config.canvas_background = ViewerCanvasBackground::Gradient;
			}
			if ui.menu_item("Grid (G)", config.show_canvas_grid, true) {
				config.show_canvas_grid = !config.show_canvas_grid;
			}
		}
		if let MediaState::Audio {
			analysis: Some(analysis),
			..
		} = media
		{
			for (label, view, enabled) in [
				(
					"Waveform",
					ViewerAudioView::Waveform,
					analysis.envelope.is_some(),
				),
				(
					"Spectrum",
					ViewerAudioView::Spectrum,
					analysis.spectrum.is_some(),
				),
				(
					"Mel Spectrogram",
					ViewerAudioView::Mel,
					analysis.mel.is_some(),
				),
			] {
				if ui.menu_item(label, config.audio_view == view, enabled) {
					config.audio_view = view;
				}
			}
		}
		if ui.menu_item("Close", false, true) {
			running = false;
		}
		ui.end_popup();
	}
	if ui.begin_popup(
		"viewer-volume-menu",
		UiPopupConfig {
			width: 176.0 * scale,
			height: (12.0 + config.style.font_size + 8.0) * scale,
			gap: 6.0 * scale,
			padding: 6.0 * scale,
		},
	) {
		if let MediaState::Audio { player, .. } = media {
			if let Some(player) = player.as_ref() {
				let muted = player.is_muted();
				if ui.menu_item("Mute", muted, true) {
					player.set_muted(!muted)?;
				}
			}
		} else {
			ui.menu_item("No audio stream", false, false);
		}
		ui.end_popup();
	}
	Ok(running)
}
