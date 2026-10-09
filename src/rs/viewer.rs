//! `oa::Viewer` — windowed and headless media inspection application.
//!
//! Donor: `oa/ui/viewer.h`, `oa/ui/viewer.cpp`,
//!        `oa/ui/viewerApplication.cpp`, `oa/ui/viewerPresentation.cpp`
//!
//! The Viewer owns an SDL3 window, swapchain `Presenter`, `Ui` compositor,
//! `Navigation`, and optional media sources.  It borrows (or creates) one
//! `Engine` and drives a per-frame compose → direct-present loop until the
//! user closes the window or calls `quit()`.
//!
//! Entry points:
//! ```rust,ignore
//! // Static file path (creates its own engine):
//! Viewer::preview("path/to/image.png")?;
//!
//! // Borrowed engine:
//! Viewer::preview_with_engine(&engine, "path/to/video.mp4")?;
//!
//! // Existing GPU texture:
//! Viewer::show(&engine, &texture)?;
//!
//! // Custom live source:
//! let mut viewer = Viewer::new(ViewerConfig { mode: ViewerMode::Live, .. });
//! viewer.set_live_source(&mut my_source);
//! viewer.run(&engine)?;
//! ```

mod audio_mode;
mod audio_view;
mod image_mode;
mod live_mode;
mod overlay;
mod timeline;
mod video_mode;

pub use config::{ViewerAudioView, ViewerCanvasBackground, ViewerConfig, ViewerMode};
pub use live_source::{ViewerLiveCapabilities, ViewerLiveSource};

mod config;
mod live_source;

use std::time::Instant;

use crate::runtime::{Presenter, SdlWindow};
use crate::ui::{
	Ui,
	navigation::Navigation,
	sdl_events::{convert_sdl_event, window_pixel_scale},
	types::{PixelRect, UiEventKind, UiKey},
};
use crate::{Engine, Error, Result, Texture, render::Renderer};

// ── Viewer ────────────────────────────────────────────────────────────────────

/// Windowed / headless media inspection session. Donor: `oa::Viewer`.
pub struct Viewer<'src> {
	config: ViewerConfig,
	live_source: Option<&'src mut dyn ViewerLiveSource>,
}

impl<'src> Viewer<'src> {
	pub fn new(config: ViewerConfig) -> Self {
		Self {
			config,
			live_source: None,
		}
	}

	pub fn with_path(path: impl Into<String>) -> Self {
		Self::new(ViewerConfig {
			path: path.into(),
			..ViewerConfig::default()
		})
	}

	pub fn set_live_source(&mut self, source: &'src mut dyn ViewerLiveSource) {
		self.live_source = Some(source);
		self.config.mode = ViewerMode::Live;
	}

	// ── Blocking entry points ─────────────────────────────────────────────────

	/// Open `path` in a new window, creating a presentation-capable engine.
	pub fn preview(path: &str) -> Result<()> {
		Self::preview_with_config(path, ViewerConfig::default())
	}

	pub fn preview_with_config(path: &str, config: ViewerConfig) -> Result<()> {
		if path.is_empty() {
			return Err(Error::invalid_argument(
				"Viewer::preview requires a non-empty path",
			));
		}
		let full_config = ViewerConfig {
			path: path.to_owned(),
			..config
		};
		Viewer::new(full_config).run_owned()
	}

	/// Open `path` borrowing `engine`.
	pub fn preview_with_engine(engine: &Engine, path: &str) -> Result<()> {
		Self::preview_with_engine_and_config(engine, path, ViewerConfig::default())
	}

	pub fn preview_with_engine_and_config(
		engine: &Engine,
		path: &str,
		config: ViewerConfig,
	) -> Result<()> {
		if path.is_empty() {
			return Err(Error::invalid_argument(
				"Viewer::preview requires a non-empty path",
			));
		}
		let full_config = ViewerConfig {
			path: path.to_owned(),
			..config
		};
		Viewer::new(full_config).run_borrowed(engine)
	}

	/// Display `texture` in a blocking window using `engine`.
	pub fn show(engine: &Engine, texture: &Texture) -> Result<()> {
		Self::show_with_config(engine, texture, ViewerConfig::default())
	}

	pub fn show_with_config(engine: &Engine, texture: &Texture, config: ViewerConfig) -> Result<()> {
		if !texture.engine_handle().same_as(&engine.handle()) {
			return Err(Error::invalid_argument(
				"Viewer::show: Texture does not belong to this Engine",
			));
		}
		let app_config = ViewerConfig {
			mode: ViewerMode::Image,
			path: String::new(),
			..config
		};
		Viewer::new(app_config).run_with_texture(engine, texture)
	}

	/// Display a headless `RenderFrame` in a blocking window.
	///
	/// Waits for the frame's producer event before presenting, then returns
	/// once the window is closed.  A second `begin_frame` / `cancel_frame`
	/// cycle is recordable immediately after this call returns, proving that
	/// the ring slot was released.
	///
	/// Donor: `oa::Viewer::show(*engine, *renderer, *frame, cfg)`.
	pub fn show_render_frame(
		engine: &Engine,
		renderer: &Renderer<'_>,
		frame: crate::render::RenderFrame,
		config: ViewerConfig,
	) -> Result<()> {
		if !renderer.owns_frame(&frame) {
			return Err(Error::invalid_argument(
				"Viewer::show_render_frame requires a frame produced by the supplied Renderer and Engine",
			));
		}
		// Wait for the GPU producer before any readback or blit.
		frame.producer().wait()?;
		let texture = frame.color().clone();
		let app_config = ViewerConfig {
			mode: ViewerMode::Image,
			path: String::new(),
			..config
		};
		// `frame` must stay alive until the Viewer exits so the ring slot
		// is not recycled before the present completes.
		let result = Viewer::new(app_config).run_with_texture(engine, &texture);
		drop(texture);
		drop(frame);
		let collect_result = renderer.collect().map(|_| ());
		match (result, collect_result) {
			(Err(error), _) | (Ok(()), Err(error)) => Err(error),
			(Ok(()), Ok(())) => Ok(()),
		}
	}

	/// Run the viewer session borrowing `engine`.
	pub fn run(self, engine: &Engine) -> Result<()> {
		self.run_borrowed(engine)
	}

	// ── Private run ───────────────────────────────────────────────────────────

	/// Create a presentation-capable engine and run.
	///
	/// Use this when `Viewer::new(config).run_standalone()` — the viewer
	/// owns its own engine lifecycle.  Equivalent to the C++ `viewer.run()`
	/// one-liner when no engine is passed.
	pub fn run_standalone(self) -> Result<()> {
		self.run_owned()
	}

	fn run_owned(mut self) -> Result<()> {
		// Create the SDL3 window first — its extensions are required before
		// the Vulkan instance is created inside Engine::builder().build().
		let window = SdlWindow::new(&self.config.title, self.config.width, self.config.height)?;
		let extensions = window.required_instance_extensions()?;
		let engine = Engine::builder()
			.requirements(crate::EngineRequirements::default().presentation())
			.instance_extensions(extensions)
			.build()?;
		self.application_loop(&engine, window, None)
	}

	/// Run on a borrowed engine (creates its own window).
	fn run_borrowed(mut self, engine: &Engine) -> Result<()> {
		let window = SdlWindow::new(&self.config.title, self.config.width, self.config.height)?;
		self.application_loop(engine, window, None)
	}

	/// Run displaying a borrowed GPU texture.
	fn run_with_texture(mut self, engine: &Engine, texture: &Texture) -> Result<()> {
		let window = SdlWindow::new(&self.config.title, self.config.width, self.config.height)?;
		self.application_loop(engine, window, Some(texture))
	}

	/// Inner application loop. Donor: `oa::Viewer::runApplication`.
	fn application_loop(
		&mut self,
		engine: &Engine,
		window: SdlWindow,
		borrowed_texture: Option<&Texture>,
	) -> Result<()> {
		// Resolve initial drawable extent.
		let [w, h] = window.drawable_extent();
		if w == 0 || h == 0 {
			return Err(Error::invalid_argument(
				"Viewer window has zero drawable extent",
			));
		}

		// Build Presenter + swapchain.
		let mut presenter = Presenter::new(engine, &window)?;
		if !presenter.supports_swapchain() {
			return Err(Error::missing_capability(
				"Viewer requires a swapchain-capable device (VK_KHR_swapchain)",
			));
		}
		if !presenter.has_present() {
			return Err(Error::missing_capability(
				"Viewer requires a queue family that can present to the window surface",
			));
		}
		presenter.init_swapchain(&window)?;

		// Open the UI compositor.
		let style = self.config.style.clone();
		let mut ui = Ui::init_with_style(engine, w, h, style)?;

		// Open the media source.
		let mut media = MediaState::open(engine, &self.config, borrowed_texture)?;

		// Navigation — fitted to content.
		let mut nav = Navigation::new();
		if let Some((cw, ch)) = media.content_size() {
			nav.set_content_size(cw as f32, ch as f32)?;
		}
		nav.set_viewport_size(w as f32, h as f32)?;
		nav.fit_to_window()?;

		let mut source_opened = false;
		let mut run_result = (|| -> Result<()> {
			if let Some(source) = self.live_source.as_deref_mut() {
				source.open(engine)?;
				source_opened = true;
			}

			// Event pump.
			let mut event_pump = window
				.event_pump()
				.map_err(|e| Error::backend_failure("SDL3", "Viewer event pump", e))?;

			let mut running = true;
			let mut last_instant = Instant::now();
			let (mut sx, mut sy) = window_pixel_scale(&window.window);

			while running {
				let now = Instant::now();
				let delta_ms = now.duration_since(last_instant).as_secs_f32() * 1000.0;
				last_instant = now;

				// Handle SDL events.
				for sdl_ev in event_pump.poll_iter() {
					match &sdl_ev {
						sdl3::event::Event::Quit { .. } => {
							running = false;
						}
						sdl3::event::Event::Window {
							win_event: sdl3::event::WindowEvent::PixelSizeChanged(pw, ph),
							..
						} => {
							let pw = *pw as u32;
							let ph = *ph as u32;
							if pw > 0 && ph > 0 {
								presenter.notify_pixel_size_changed(pw, ph);
								ui.resize(pw, ph)?;
								nav.set_viewport_size(pw as f32, ph as f32)?;
								(sx, sy) = window_pixel_scale(&window.window);
							}
						}
						_ => {}
					}
					if let Some(ui_ev) = convert_sdl_event(&sdl_ev, sx, sy) {
						// Quit shortcut.
						if ui_ev.kind == UiEventKind::KeyDown
							&& (ui_ev.key == self.config.key_quit || ui_ev.key == self.config.key_quit_q)
						{
							running = false;
							continue;
						}

						// Live source raw events (before navigation).
						if let Some(source) = self.live_source.as_deref_mut()
							&& source.capabilities().receives_events
						{
							source.on_event(&ui_ev)?;
						}

						// Navigation consumes scroll/drag/pinch.
						let nav_consumed = nav.handle_event(&ui_ev);

						// Image view shortcuts.
						if !nav_consumed {
							media.handle_event(&ui_ev, &self.config)?;
						}

						// Route to UI widgets.
						ui.route_event(&ui_ev);
					}
				}

				// Recreate the swapchain after a resize.
				if presenter.needs_swapchain_recreate() {
					presenter.wait_presentation_idle()?;
					presenter.recreate_swapchain(&window)?;
				}

				// Per-frame update.
				nav.update(delta_ms);
				media.update(delta_ms, engine)?;
				if let Some(source) = self.live_source.as_deref_mut() {
					source.update(delta_ms)?;
				}

				// Build frame.
				let [compose_w, compose_h] = ui.compose_extent().unwrap_or([w, h]);
				ui.begin_frame(delta_ms);
				render_frame(
					&mut ui,
					&mut media,
					&mut nav,
					&self.config,
					compose_w,
					compose_h,
				)?;
				if matches!(media, MediaState::Live)
					&& let Some(source) = self.live_source.as_deref_mut()
				{
					source.render(&mut ui, compose_w, compose_h)?;
				}
				ui.end_frame();

				// Renderer targets already contain the complete frame. Present them
				// directly so the image-backed target never takes a host detour.
				if presenter.has_swapchain() {
					let suboptimal = if let Some(texture) = borrowed_texture
						&& texture.render_target_image().is_some()
					{
						presenter.present_texture(texture)?
					} else {
						ui.present_to(&mut presenter)?
					};
					if suboptimal {
						let [dw, dh] = window.drawable_extent();
						presenter.notify_pixel_size_changed(dw, dh);
					}
				}
			}
			Ok(())
		})();

		// Close every initialized session even when rendering or event handling
		// fails. Preserve the first error while still attempting later cleanup.
		preserve_first(&mut run_result, presenter.wait_presentation_idle());
		if source_opened && let Some(source) = self.live_source.as_deref_mut() {
			preserve_first(&mut run_result, source.close());
		}
		preserve_first(&mut run_result, media.close());
		preserve_first(&mut run_result, ui.close());
		run_result
	}
}

fn preserve_first(result: &mut Result<()>, next: Result<()>) {
	if result.is_ok() {
		*result = next;
	}
}

// ── Frame render ──────────────────────────────────────────────────────────────

fn render_frame(
	ui: &mut Ui<'_>,
	media: &mut MediaState,
	nav: &mut Navigation,
	config: &ViewerConfig,
	w: u32,
	h: u32,
) -> Result<()> {
	let full_rect = PixelRect::new(0, 0, w, h);

	// Canvas background.
	if media.has_visual_content() {
		overlay::draw_canvas_background(ui, full_rect, config)?;
	}

	// Media content area.
	match media {
		MediaState::Image {
			texture,
			channel_mode,
		} => {
			let tex_ref: Option<&Texture> = texture.as_ref().as_ref();
			image_mode::render(ui, tex_ref, nav, channel_mode, full_rect, config)?;
		}
		MediaState::Video { player } => {
			let p: Option<&mut crate::VideoPlayer> = player.as_mut().as_mut();
			video_mode::render(ui, p, nav, full_rect, config)?;
		}
		MediaState::Audio { player, analysis } => {
			let p: Option<&mut crate::AudioPlayer> = player.as_mut().as_mut();
			let a: Option<&audio_view::AudioAnalysis> = analysis.as_ref();
			audio_mode::render(ui, p, a, full_rect, config)?;
		}
		MediaState::Live => {}
		MediaState::Empty => {}
	}

	// Timeline (temporal modes).
	if config.show_timeline && media.has_timeline() {
		timeline::render(ui, media, config, full_rect)?;
	}
	overlay::draw_hud(ui, full_rect, nav, media.mode_name(), config)?;

	Ok(())
}

// ── Media state ───────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub(crate) enum MediaState {
	Image {
		texture: Box<Option<Texture>>,
		channel_mode: image_mode::ChannelMode,
	},
	Video {
		player: Box<Option<crate::VideoPlayer>>,
	},
	Audio {
		player: Box<Option<crate::AudioPlayer>>,
		analysis: Option<audio_view::AudioAnalysis>,
	},
	Live,
	Empty,
}

impl MediaState {
	fn open(engine: &Engine, config: &ViewerConfig, borrowed: Option<&Texture>) -> Result<Self> {
		match config.mode {
			ViewerMode::Image => {
				let texture = if let Some(t) = borrowed {
					Some(t.clone())
				} else if !config.path.is_empty() {
					Some(image_mode::load_texture(engine, &config.path)?)
				} else {
					None
				};
				Ok(Self::Image {
					texture: Box::new(texture),
					channel_mode: image_mode::ChannelMode::Rgb,
				})
			}
			ViewerMode::Video => {
				let player = if !config.path.is_empty() {
					Some(video_mode::open(engine, config)?)
				} else {
					None
				};
				Ok(Self::Video {
					player: Box::new(player),
				})
			}
			ViewerMode::Audio => {
				let (player, analysis) = if !config.path.is_empty() {
					let p = audio_mode::open(engine, config)?;
					let a = audio_view::analyze(&config.path, config)?;
					(Some(p), a)
				} else {
					(None, None)
				};
				Ok(Self::Audio {
					player: Box::new(player),
					analysis,
				})
			}
			ViewerMode::Live => Ok(Self::Live),
			ViewerMode::Auto => {
				// Try image → video → audio in donor order.
				if let Ok(texture) = image_mode::load_texture(engine, &config.path) {
					return Ok(Self::Image {
						texture: Box::new(Some(texture)),
						channel_mode: image_mode::ChannelMode::Rgb,
					});
				}
				if let Ok(player) = video_mode::open(engine, config) {
					return Ok(Self::Video {
						player: Box::new(Some(player)),
					});
				}
				if let Ok(player) = audio_mode::open(engine, config) {
					let analysis = audio_view::analyze(&config.path, config)?;
					return Ok(Self::Audio {
						player: Box::new(Some(player)),
						analysis,
					});
				}
				Err(Error::failed_precondition(
					"Viewer Auto mode could not open path as image, video, or audio",
				))
			}
		}
	}

	fn has_visual_content(&self) -> bool {
		matches!(self, Self::Image { .. } | Self::Video { .. } | Self::Live)
	}

	fn mode_name(&self) -> &'static str {
		match self {
			Self::Image { .. } => "Image",
			Self::Video { .. } => "Video",
			Self::Audio { .. } => "Audio",
			Self::Live => "Live",
			Self::Empty => "Empty",
		}
	}

	fn has_timeline(&self) -> bool {
		matches!(self, Self::Video { .. } | Self::Audio { .. })
	}

	fn content_size(&self) -> Option<(u32, u32)> {
		match self {
			Self::Image { texture, .. } => texture
				.as_ref()
				.as_ref()
				.map(|t| (t.width() as u32, t.height() as u32)),
			Self::Video { player } => player
				.as_ref()
				.as_ref()
				.and_then(|p| p.info().map(|info| (info.width(), info.height()))),
			_ => None,
		}
	}

	fn update(&mut self, delta_ms: f32, _engine: &Engine) -> Result<()> {
		if let Self::Video { player } = self
			&& let Some(p) = player.as_mut().as_mut()
		{
			let elapsed = std::time::Duration::from_secs_f32(delta_ms / 1000.0);
			p.tick(elapsed)?;
		}
		Ok(())
	}

	fn handle_event(&mut self, ev: &crate::ui::types::UiEvent, config: &ViewerConfig) -> Result<()> {
		if ev.kind != UiEventKind::KeyDown {
			return Ok(());
		}
		match self {
			Self::Image { channel_mode, .. } => {
				image_mode::handle_event(ev, channel_mode, config);
			}
			Self::Video { player } => {
				if let Some(p) = player.as_mut().as_mut() {
					if ev.key == UiKey::Space {
						p.toggle_play()?;
					}
					if ev.key == UiKey::Left {
						p.step_frames(-1)?;
					}
					if ev.key == UiKey::Right {
						p.advance()?;
					}
				}
			}
			Self::Audio { player, .. } => {
				if let Some(p) = player.as_mut().as_mut()
					&& ev.key == UiKey::Space
				{
					if p.is_playing() {
						p.pause();
					} else {
						p.play()?;
					}
				}
			}
			_ => {}
		}
		Ok(())
	}

	fn close(&mut self) -> Result<()> {
		match self {
			Self::Video { player } => {
				if let Some(player) = player.as_mut().as_mut() {
					player.close();
				}
				Ok(())
			}
			Self::Audio { player, .. } => {
				if let Some(player) = player.as_mut().as_mut() {
					player.close()?;
				}
				Ok(())
			}
			Self::Image { .. } | Self::Live | Self::Empty => Ok(()),
		}
	}

	pub(crate) fn position_us(&self) -> u64 {
		match self {
			Self::Video { player } => player
				.as_ref()
				.as_ref()
				.and_then(|p| {
					// Approximate from frame index at 30 fps.
					p.current_frame_index().ok().map(|i| i * 1_000_000 / 30)
				})
				.unwrap_or(0),
			Self::Audio { player, .. } => player.as_ref().as_ref().map_or(0, |p| p.position_us()),
			_ => 0,
		}
	}

	pub(crate) fn duration_us(&self) -> u64 {
		match self {
			Self::Video { player } => player.as_ref().as_ref().map_or(0, |p| {
				p.info().map_or(0, |info| {
					let tb = info.time_base();
					let d = info.duration();
					// ticks → µs: d * num / den * 1_000_000
					(d as u128 * tb.numerator() as u128 * 1_000_000 / tb.denominator().max(1) as u128) as u64
				})
			}),
			// AudioPlayer doesn't expose total duration.
			_ => 0,
		}
	}

	#[allow(dead_code)]
	pub(crate) fn is_playing(&self) -> bool {
		match self {
			Self::Video { player } => player.as_ref().as_ref().is_some_and(|p| p.is_playing()),
			Self::Audio { player, .. } => player.as_ref().as_ref().is_some_and(|p| p.is_playing()),
			_ => false,
		}
	}

	#[allow(dead_code)]
	pub(crate) fn toggle_play(&mut self) -> Result<()> {
		match self {
			Self::Video { player } => {
				if let Some(p) = player.as_mut().as_mut() {
					p.toggle_play()
				} else {
					Ok(())
				}
			}
			Self::Audio { player, .. } => {
				if let Some(p) = player.as_mut().as_mut() {
					if p.is_playing() {
						p.pause();
						Ok(())
					} else {
						p.play()
					}
				} else {
					Ok(())
				}
			}
			_ => Ok(()),
		}
	}

	pub(crate) fn seek_fraction(&mut self, fraction: f32) -> Result<()> {
		let fraction = fraction.clamp(0.0, 1.0);
		let dur = self.duration_us();
		if dur == 0 {
			return Ok(());
		}
		let target = (dur as f64 * fraction as f64) as u64;
		match self {
			Self::Video { player } => {
				if let Some(p) = player.as_mut().as_mut() {
					p.seek(target)
				} else {
					Ok(())
				}
			}
			Self::Audio { player, .. } => {
				if let Some(p) = player.as_mut().as_mut() {
					p.seek(target)
				} else {
					Ok(())
				}
			}
			_ => Ok(()),
		}
	}
}
