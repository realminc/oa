//! `oa::plot::Figure` — top-level container for a grid of `oa::plot::Axes`.
//!
//! Three terminal sinks:
//! - `show()` — replay through `Ui`'s GPU compositor in a live window.
//! - `save_to(engine, path)` — headless GPU render → image file (Experimental).
//! - `render(engine)` — return the composition as an `oa::Image` (Experimental).
//!
//! Donor reference: `oa/ui/plot/figure.h`, `oa/ui/plot/figure.cpp`

use super::axes::{Axes, HeatmapCmd};
use crate::ui::{TextLayoutConfig, Ui, types::PixelRect};
use crate::{Color, Engine, Error, Image, Result, Texture};

fn preserve_first(result: &mut Result<()>, next: Result<()>) {
	if result.is_ok() {
		*result = next;
	}
}

// ── Theme ─────────────────────────────────────────────────────────────────────

/// Figure color theme.
///
/// Donor: `oa::plot::Theme`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Theme {
	#[default]
	Dark = 0,
	Light = 1,
}

impl Theme {
	fn background(self) -> Color {
		match self {
			Theme::Dark => Color::new(0.10, 0.11, 0.13, 1.0),
			Theme::Light => Color::new(0.95, 0.96, 0.97, 1.0),
		}
	}
	fn axes_background(self) -> Color {
		match self {
			Theme::Dark => Color::new(0.13, 0.14, 0.17, 1.0),
			Theme::Light => Color::new(0.89, 0.91, 0.94, 1.0),
		}
	}
	fn text(self) -> Color {
		match self {
			Theme::Dark => Color::new(0.88, 0.90, 0.94, 1.0),
			Theme::Light => Color::new(0.10, 0.12, 0.16, 1.0),
		}
	}
	fn subtext(self) -> Color {
		match self {
			Theme::Dark => Color::new(0.63, 0.66, 0.72, 1.0),
			Theme::Light => Color::new(0.40, 0.43, 0.48, 1.0),
		}
	}
	fn grid(self) -> Color {
		match self {
			Theme::Dark => Color::new(0.23, 0.24, 0.28, 1.0),
			Theme::Light => Color::new(0.78, 0.80, 0.84, 1.0),
		}
	}
	fn border(self) -> Color {
		match self {
			Theme::Dark => Color::new(0.28, 0.30, 0.35, 1.0),
			Theme::Light => Color::new(0.60, 0.63, 0.68, 1.0),
		}
	}
	fn palette(self) -> [Color; 6] {
		// Same palette for both themes; only saturation/value differs in light.
		[
			Color::new(0.24, 0.65, 1.00, 1.0), // blue
			Color::new(1.00, 0.47, 0.35, 1.0), // orange
			Color::new(0.35, 0.82, 0.54, 1.0), // green
			Color::new(0.75, 0.45, 1.00, 1.0), // purple
			Color::new(1.00, 0.80, 0.22, 1.0), // yellow
			Color::new(0.30, 0.87, 0.89, 1.0), // cyan
		]
	}
}

// ── FigureConfig ──────────────────────────────────────────────────────────────

/// Top-level figure configuration.
///
/// Donor: `oa::plot::FigureConfig`.
#[derive(Clone, Debug)]
pub struct FigureConfig {
	pub title: String,
	pub rows: i32,
	pub cols: i32,
	pub width: u32,
	pub height: u32,
	/// Horizontal spacing between axes cells in pixels.
	pub h_spacing: i32,
	/// Vertical spacing between axes cells in pixels.
	pub v_spacing: i32,
	/// Outer padding around the entire grid.
	pub padding: i32,
	/// Color theme (Dark or Light).  Donor: `oa::plot::Theme`.
	pub theme: Theme,
	/// Override background color.  Alpha-zero uses the theme background.
	pub background: Color,
}

impl Default for FigureConfig {
	fn default() -> Self {
		Self {
			title: "oa::plot".to_owned(),
			rows: 1,
			cols: 1,
			width: 800,
			height: 600,
			h_spacing: 20,
			v_spacing: 24,
			padding: 20,
			theme: Theme::Dark,
			background: Color::new(0.0, 0.0, 0.0, 0.0), // use theme
		}
	}
}

// ── Figure ────────────────────────────────────────────────────────────────────

/// Pixel rect returned by [`Figure::cell_rect`].
///
/// Donor: `oa::plot::Figure::Rect`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CellRect {
	pub x: i32,
	pub y: i32,
	pub w: i32,
	pub h: i32,
}

/// Grid figure container.
///
/// Created without an Engine; the Engine is supplied only at sink time.
pub struct Figure {
	config: FigureConfig,
	axes: Vec<Axes>,
	fig_x_label: String,
	fig_y_label: String,
}

impl Figure {
	/// Construct a new figure with the given configuration.
	pub fn new(mut config: FigureConfig) -> Self {
		config.rows = config.rows.max(1);
		config.cols = config.cols.max(1);
		config.h_spacing = config.h_spacing.max(0);
		config.v_spacing = config.v_spacing.max(0);
		config.padding = config.padding.max(0);
		let rows = config.rows as usize;
		let cols = config.cols as usize;
		let axes = vec![Axes::default(); rows * cols];
		Self {
			config,
			axes,
			fig_x_label: String::new(),
			fig_y_label: String::new(),
		}
	}

	/// Construct with default configuration.
	pub fn default_config() -> Self {
		Self::new(FigureConfig::default())
	}

	/// Access the (row, col) axes — both 0-indexed.
	pub fn ax(&mut self, row: i32, col: i32) -> &mut Axes {
		let r = row.clamp(0, self.config.rows - 1) as usize;
		let c = col.clamp(0, self.config.cols - 1) as usize;
		let cols = self.config.cols as usize;
		&mut self.axes[r * cols + c]
	}

	/// Set the figure-level title (also replaces the window title in `show`).
	pub fn title(&mut self, text: &str) {
		self.config.title = text.to_owned();
	}

	/// Set the figure-level horizontal axis label (rendered below the grid).
	///
	/// Donor: `oa::plot::Figure::xLabel`.
	pub fn x_label(&mut self, text: &str) {
		self.fig_x_label = text.to_owned();
	}

	/// Set the figure-level vertical axis label (rendered left of the grid).
	///
	/// Donor: `oa::plot::Figure::yLabel`.
	pub fn y_label(&mut self, text: &str) {
		self.fig_y_label = text.to_owned();
	}

	// ── Layout queries ────────────────────────────────────────────────────────

	/// Return the figure configuration.
	///
	/// Donor: `oa::plot::Figure::config()`.
	pub fn config(&self) -> &FigureConfig {
		&self.config
	}

	/// Return the row count.
	///
	/// Donor: `oa::plot::Figure::rows()`.
	pub fn rows(&self) -> i32 {
		self.config.rows.max(1)
	}

	/// Return the column count.
	///
	/// Donor: `oa::plot::Figure::cols()`.
	pub fn cols(&self) -> i32 {
		self.config.cols.max(1)
	}

	/// Compute the pixel rect of the `(row, col)` cell inside a canvas of
	/// `(w × h)` pixels, after all figure-level title/label bands are reserved.
	///
	/// Donor: `oa::plot::Figure::cellRect`.
	pub fn cell_rect(&self, row: i32, col: i32, w: u32, h: u32) -> CellRect {
		let layout = FigureLayout::compute(&self.config, &self.fig_x_label, &self.fig_y_label, w, h);
		let r = row.clamp(0, self.config.rows - 1) as usize;
		let c = col.clamp(0, self.config.cols - 1) as usize;
		let ax_x = layout.pad + c as u32 * (layout.cell_w + layout.hs);
		let ax_y = layout.top_band + r as u32 * (layout.cell_h + layout.vs);
		CellRect {
			x: ax_x as i32,
			y: ax_y as i32,
			w: layout.cell_w as i32,
			h: layout.cell_h as i32,
		}
	}

	// ── Sinks ─────────────────────────────────────────────────────────────────

	/// Headless GPU render → PNG/JPEG/WebP file.
	///
	/// GPU composition → file encode is Experimental.  The current
	/// implementation produces a composed Texture and saves it.
	pub fn save_to(&self, engine: &Engine, path: &str) -> Result<()> {
		let texture = self.render_to_texture(engine)?;
		crate::render::save_texture_file(&texture, std::path::Path::new(path), 90)
	}

	/// Headless GPU render → `oa::Image` (normalized Float32 `[1, 4, H, W]`).
	///
	/// GPU→Image readback is Experimental.
	pub fn render(&self, engine: &Engine) -> Result<Image> {
		let texture = self.render_to_texture(engine)?;
		let rgba8 = texture.read_rgba8()?;
		let w = texture.width();
		let h = texture.height();
		// Convert packed RGBA8 to Float32 NCHW [1, 4, H, W].
		let count = w * h;
		let mut data = vec![0.0f32; 4 * count];
		for i in 0..count {
			data[i] = rgba8[i * 4] as f32 / 255.0;
			data[count + i] = rgba8[i * 4 + 1] as f32 / 255.0;
			data[count * 2 + i] = rgba8[i * 4 + 2] as f32 / 255.0;
			data[count * 3 + i] = rgba8[i * 4 + 3] as f32 / 255.0;
		}
		let matrix = crate::Matrix::from_f32(engine, [1, 4, h, w], &data)?;
		Image::new(matrix, crate::ImageLayout::Nchw, crate::ImageFormat::Rgba)
	}

	/// Open an interactive SDL3 window and present the figure.
	///
	/// GPU composition + SDL3 window loop is Experimental.
	pub fn show(&self, engine: &Engine) -> Result<()> {
		let w = self.config.width;
		let h = self.config.height;
		let window = crate::SdlWindow::new(&self.config.title, w, h)
			.map_err(|e| Error::backend_failure("SDL3", "Figure::show window", e))?;
		let mut presenter = crate::Presenter::new(engine, &window)?;
		if !presenter.supports_swapchain() {
			return Err(Error::missing_capability(
				"Figure::show requires a presentation-capable device",
			));
		}
		if !presenter.has_present() {
			return Err(Error::missing_capability(
				"Figure::show requires a queue family that can present to the window surface",
			));
		}
		presenter.init_swapchain(&window)?;
		let mut ui = Ui::init(engine, w, h)?;
		let mut run_result = (|| -> Result<()> {
			let mut event_pump = window
				.event_pump()
				.map_err(|e| Error::backend_failure("SDL3", "Figure::show event pump", e))?;
			loop {
				let mut quit = false;
				for ev in event_pump.poll_iter() {
					if matches!(ev, sdl3::event::Event::Quit { .. }) {
						quit = true;
					}
					if let sdl3::event::Event::KeyDown {
						keycode: Some(sdl3::keyboard::Keycode::Escape | sdl3::keyboard::Keycode::Q),
						..
					} = ev
					{
						quit = true;
					}
				}
				if quit {
					break;
				}
				let [draw_w, draw_h] = window.drawable_extent();
				if draw_w == 0 || draw_h == 0 {
					continue;
				}
				if ui.compose_extent() != Some([draw_w, draw_h]) {
					ui.resize(draw_w, draw_h)?;
				}
				if presenter.swapchain_extent_px() != Some([draw_w, draw_h]) {
					presenter.notify_pixel_size_changed(draw_w, draw_h);
				}
				if presenter.needs_swapchain_recreate() {
					presenter.wait_presentation_idle()?;
					presenter.recreate_swapchain(&window)?;
				}
				ui.begin_frame(0.0);
				self.record_into(&mut ui, draw_w, draw_h)?;
				ui.end_frame();
				if ui.present_to(&mut presenter)? {
					presenter.notify_pixel_size_changed(draw_w, draw_h);
				}
			}
			Ok(())
		})();
		preserve_first(&mut run_result, presenter.wait_presentation_idle());
		preserve_first(&mut run_result, ui.close());
		run_result
	}

	// ── Composition ──────────────────────────────────────────────────────────

	/// Replay the complete figure into an already-open `Ui` GPU frame.
	///
	/// This is the public seam used by `ViewerLiveSource` implementations that
	/// manage their own window/presenter lifecycle.  The configured aspect ratio
	/// and all title/label bands remain stable on resize.
	///
	/// Donor: `oa::plot::Figure::renderFrame`.
	pub fn render_frame(&self, w: u32, h: u32, ui: &mut Ui<'_>) -> Result<()> {
		self.record_into(ui, w, h)
	}

	/// Compose the figure into a packed RGBA8 Texture.
	///
	/// Commands are lowered through the UI compute compositor and copied through
	/// its explicit blocking headless readback boundary.
	fn render_to_texture(&self, engine: &Engine) -> Result<Texture> {
		let w = self.config.width;
		let h = self.config.height;
		let mut ui = Ui::init(engine, w, h)?;
		ui.begin_frame(0.0);
		let mut render_result = (|| -> Result<Vec<u8>> {
			self.record_into(&mut ui, w, h)?;
			ui.end_frame();
			ui.render_rgba8()
		})();
		if let Err(close_error) = ui.close()
			&& render_result.is_ok()
		{
			render_result = Err(close_error);
		}
		let rgba = render_result?;
		crate::render::texture_from_rgba8(engine, &rgba, w as usize, h as usize)
	}

	/// Replay retained artists into an already-open UI frame.
	fn record_into(&self, ui: &mut Ui<'_>, w: u32, h: u32) -> Result<()> {
		let theme = self.config.theme;
		let bg = if self.config.background.a > 0.0 {
			self.config.background
		} else {
			theme.background()
		};
		let full = PixelRect::new(0, 0, w, h);
		let text_color = theme.text();
		let subtext_color = theme.subtext();
		let label_cfg = TextLayoutConfig {
			size: 11.0,
			..Default::default()
		};

		// Compute the shared layout for this output size.
		let layout = FigureLayout::compute(&self.config, &self.fig_x_label, &self.fig_y_label, w, h);
		if layout.frame.w == 0 || layout.frame.h == 0 {
			return Ok(());
		}
		ui.rect_raw(layout.frame, bg, full, 0.0);

		// Figure-level title.
		if !self.config.title.is_empty() {
			ui.text_at(
				&self.config.title,
				[
					(layout.frame.x + layout.outer_pad as i32) as f32,
					(layout.frame.y + layout.outer_pad as i32 + 22) as f32,
				],
				TextLayoutConfig {
					size: 16.0,
					..Default::default()
				},
				text_color,
				full,
			)?;
		}

		// Figure-level x label (below the grid).
		if !self.fig_x_label.is_empty() {
			let lx = layout.frame.x as f32 + layout.frame.w as f32 / 2.0;
			let ly = (layout.frame.y + layout.frame.h as i32
				- layout.outer_pad as i32
				- (layout.bottom_band / 2) as i32) as f32;
			ui.text_at(&self.fig_x_label, [lx, ly], label_cfg, subtext_color, full)?;
		}

		// Figure-level y label (left of the grid).
		// Full rotated y-axis text requires the text-transform pass (Planned).
		// For now render horizontally in the left band so the label is present.
		if !self.fig_y_label.is_empty() {
			let ly = layout.frame.y as f32 + layout.frame.h as f32 / 2.0;
			let lx = (layout.frame.x + layout.outer_pad as i32 + (layout.left_band / 2) as i32) as f32;
			ui.text_at(&self.fig_y_label, [lx, ly], label_cfg, subtext_color, full)?;
		}

		let rows = layout.rows;
		let cols = layout.cols;

		for r in 0..rows {
			for c in 0..cols {
				let ax_x = layout.pad + c as u32 * (layout.cell_w + layout.hs);
				let ax_y = layout.top_band + r as u32 * (layout.cell_h + layout.vs);
				let rect = PixelRect {
					x: ax_x as i32,
					y: ax_y as i32,
					w: layout.cell_w,
					h: layout.cell_h,
				};
				// Axes background.
				ui.rect_raw(rect, theme.axes_background(), rect, 0.0);

				let axes = &self.axes[r * cols + c];
				let label_config = TextLayoutConfig {
					size: 10.0,
					..Default::default()
				};
				let subtext = theme.subtext();

				// Imshow raster base.
				if let Some(image) = &axes.cmds.imshow {
					ui.image_at(&image.texture, rect)?;
				}

				// Heatmap.
				if let Some(heatmap) = &axes.cmds.heatmap {
					render_heatmap(ui, heatmap, rect)?;
				}

				// Border.
				let border = axes.cmds.border_color.unwrap_or(theme.border());
				ui.rect_outline_raw(rect, border, 1, rect);

				let limits = axes.effective_limits();

				// Axes title.
				if !axes.cmds.title.is_empty() {
					let title_color = if axes.cmds.title_color.a > 0.0 {
						axes.cmds.title_color
					} else {
						text_color
					};
					ui.text_at(
						&axes.cmds.title,
						[rect.x as f32 + 4.0, rect.y as f32 + 12.0],
						label_config,
						title_color,
						rect,
					)?;
				}

				// Caption (below title).
				if !axes.cmds.caption.is_empty() {
					let caption_color = if axes.cmds.caption_color.a > 0.0 {
						axes.cmds.caption_color
					} else {
						subtext
					};
					ui.text_at(
						&axes.cmds.caption,
						[rect.x as f32 + 4.0, rect.y as f32 + 24.0],
						label_config,
						caption_color,
						rect,
					)?;
				}

				// Y label.
				if !axes.cmds.y_label.is_empty() {
					ui.text_at(
						&axes.cmds.y_label,
						[rect.x as f32 + 4.0, rect.y as f32 + 36.0],
						label_config,
						subtext,
						rect,
					)?;
				}

				// X label (near bottom).
				if !axes.cmds.x_label.is_empty() {
					ui.text_at(
						&axes.cmds.x_label,
						[rect.x as f32 + 4.0, (rect.y + rect.h as i32 - 4) as f32],
						label_config,
						subtext,
						rect,
					)?;
				}

				// Tick labels — 5 steps on each axis.
				for step in 0..=4 {
					let fraction = step as f32 / 4.0;
					let x_value = limits.x_min + (limits.x_max - limits.x_min) * fraction;
					let y_value = limits.y_max - (limits.y_max - limits.y_min) * fraction;
					ui.text_at(
						&format_tick(x_value),
						[
							rect.x as f32 + fraction * rect.w as f32 + 2.0,
							(rect.y + rect.h as i32 - 16) as f32,
						],
						label_config,
						subtext,
						rect,
					)?;
					ui.text_at(
						&format_tick(y_value),
						[
							rect.x as f32 + 2.0,
							rect.y as f32 + fraction * rect.h as f32 + 10.0,
						],
						label_config,
						subtext,
						rect,
					)?;
				}

				// Grid lines.
				if axes.cmds.show_grid {
					let grid = theme.grid();
					for step in 1..4 {
						let x = rect.x as f32 + rect.w as f32 * step as f32 / 4.0;
						let y = rect.y as f32 + rect.h as f32 * step as f32 / 4.0;
						ui.line_at(
							[x, rect.y as f32],
							[x, (rect.y + rect.h as i32) as f32],
							grid,
							1.0,
							rect,
						)?;
						ui.line_at(
							[rect.x as f32, y],
							[(rect.x + rect.w as i32) as f32, y],
							grid,
							1.0,
							rect,
						)?;
					}
				}

				let palette = theme.palette();

				// Map data-space → pixel-space closures.
				let map_x = |value: f32| {
					rect.x as f32
						+ ((value - limits.x_min) / (limits.x_max - limits.x_min)).clamp(0.0, 1.0)
							* rect.w.saturating_sub(1) as f32
				};
				let map_y = |value: f32| {
					rect.y as f32
						+ (1.0 - ((value - limits.y_min) / (limits.y_max - limits.y_min)).clamp(0.0, 1.0))
							* rect.h.saturating_sub(1) as f32
				};

				// Replay artists in insertion order (donor ArtistRef contract).
				let mut line_idx = 0usize;
				let mut scatter_idx = 0usize;
				let mut bar_idx = 0usize;
				// Palette index advances globally across all artist kinds.
				let mut palette_idx = 0usize;

				for kind in &axes.cmds.artists {
					match kind {
						super::axes::ArtistKind::Line => {
							if let Some(line) = axes.cmds.lines.get(line_idx) {
								let color = if line.style.color.a == 0.0 {
									let c = palette[palette_idx % palette.len()];
									palette_idx += 1;
									c
								} else {
									line.style.color
								};
								for (xs, ys) in line.x.windows(2).zip(line.y.windows(2)) {
									if xs.iter().chain(ys).all(|v| v.is_finite()) {
										ui.line_at(
											[map_x(xs[0]), map_y(ys[0])],
											[map_x(xs[1]), map_y(ys[1])],
											color,
											line.style.width,
											rect,
										)?;
									}
								}
								line_idx += 1;
							}
						}
						super::axes::ArtistKind::Scatter => {
							if let Some(scatter) = axes.cmds.scatters.get(scatter_idx) {
								let color = if scatter.style.color.a == 0.0 {
									let c = palette[palette_idx % palette.len()];
									palette_idx += 1;
									c
								} else {
									scatter.style.color
								};
								for (&x, &y) in scatter.x.iter().zip(&scatter.y) {
									if x.is_finite() && y.is_finite() {
										let point = [map_x(x), map_y(y)];
										ui.line_at(point, point, color, scatter.style.radius * 2.0, rect)?;
									}
								}
								scatter_idx += 1;
							}
						}
						super::axes::ArtistKind::Bar => {
							if let Some(bars) = axes.cmds.bars.get(bar_idx) {
								let color = if bars.style.color.a == 0.0 {
									let c = palette[palette_idx % palette.len()];
									palette_idx += 1;
									c
								} else {
									bars.style.color
								};
								let slot = rect.w as f32 / bars.y.len().max(1) as f32;
								let width = (slot * (1.0 - bars.style.gap.clamp(0.0, 0.95))).max(1.0) as u32;
								for (index, &value) in bars.y.iter().enumerate() {
									if !value.is_finite() {
										continue;
									}
									let y0 = map_y(0.0);
									let y1 = map_y(value);
									let bar = PixelRect {
										x: (rect.x as f32 + index as f32 * slot + (slot - width as f32) * 0.5) as i32,
										y: y0.min(y1) as i32,
										w: width,
										h: (y0 - y1).abs().max(1.0) as u32,
									};
									ui.rect_raw(bar, color, rect, 0.0);
								}
								bar_idx += 1;
							}
						}
					}
				}

				// Legend (line swatches + text).
				if axes.cmds.show_legend {
					let mut legend_row = 0usize;
					let legend_x = rect.x as f32 + rect.w as f32 - 84.0;
					let draw_legend_entry =
						|ui: &mut Ui<'_>, label: &str, color: Color, row: usize| -> Result<()> {
							if label.is_empty() {
								return Ok(());
							}
							let ly = rect.y as f32 + 14.0 + row as f32 * 13.0;
							ui.line_at(
								[legend_x, ly - 4.0],
								[legend_x + 10.0, ly - 4.0],
								color,
								2.0,
								rect,
							)?;
							ui.text_at(label, [legend_x + 13.0, ly], label_config, color, rect)?;
							Ok(())
						};
					for (series, line) in axes.cmds.lines.iter().enumerate() {
						let color = if line.style.color.a == 0.0 {
							palette[series % palette.len()]
						} else {
							line.style.color
						};
						draw_legend_entry(ui, &line.style.label, color, legend_row)?;
						if !line.style.label.is_empty() {
							legend_row += 1;
						}
					}
					for (series, scatter) in axes.cmds.scatters.iter().enumerate() {
						let color = if scatter.style.color.a == 0.0 {
							palette[(series + axes.cmds.lines.len()) % palette.len()]
						} else {
							scatter.style.color
						};
						draw_legend_entry(ui, &scatter.style.label, color, legend_row)?;
						if !scatter.style.label.is_empty() {
							legend_row += 1;
						}
					}
					for (series, bars) in axes.cmds.bars.iter().enumerate() {
						let color = if bars.style.color.a == 0.0 {
							palette[(series + axes.cmds.lines.len() + axes.cmds.scatters.len()) % palette.len()]
						} else {
							bars.style.color
						};
						draw_legend_entry(ui, &bars.style.label, color, legend_row)?;
						if !bars.style.label.is_empty() {
							legend_row += 1;
						}
					}
					let _ = legend_row; // consumed
				}
			}
		}
		Ok(())
	}
}

// ── Figure layout helper ──────────────────────────────────────────────────────

/// Pre-computed layout bands and cell geometry for one output size.
///
/// Shared between `record_into` and `cell_rect` so the layout is always
/// identical regardless of how the figure is consumed.
struct FigureLayout {
	frame: PixelRect,
	rows: usize,
	cols: usize,
	outer_pad: u32,
	pad: u32,
	hs: u32,
	vs: u32,
	top_band: u32,
	bottom_band: u32,
	left_band: u32,
	cell_w: u32,
	cell_h: u32,
}

impl FigureLayout {
	fn compute(config: &FigureConfig, x_label: &str, y_label: &str, w: u32, h: u32) -> Self {
		let rows = config.rows.max(1) as usize;
		let cols = config.cols.max(1) as usize;
		let (frame_w, frame_h) = if w == 0 || h == 0 || config.width == 0 || config.height == 0 {
			(0, 0)
		} else {
			let figure_aspect = config.width as f64 / config.height as f64;
			let output_aspect = w as f64 / h as f64;
			if output_aspect > figure_aspect {
				((h as f64 * figure_aspect) as u32, h)
			} else {
				(w, (w as f64 / figure_aspect) as u32)
			}
		};
		let frame = PixelRect::new(
			(w.saturating_sub(frame_w) / 2) as i32,
			(h.saturating_sub(frame_h) / 2) as i32,
			frame_w,
			frame_h,
		);
		let scale = if config.width == 0 {
			0.0
		} else {
			frame_w as f64 / config.width as f64
		};
		let scaled = |value: i32| (value.max(0) as f64 * scale) as u32;
		let outer_pad = scaled(config.padding);
		let hs = scaled(config.h_spacing);
		let vs = scaled(config.v_spacing);

		// Reserve bands for figure-level labels.
		let title_h = if config.title.is_empty() {
			0
		} else {
			(38.0 * scale) as u32
		};
		let bottom_band = if x_label.is_empty() {
			0
		} else {
			(30.0 * scale) as u32
		};
		let left_band = if y_label.is_empty() {
			0
		} else {
			(30.0 * scale) as u32
		};
		let top_band = frame.y.max(0) as u32 + outer_pad + title_h;

		let avail_w = frame_w
			.saturating_sub(outer_pad.saturating_mul(2) + left_band.saturating_mul(2))
			.saturating_sub(hs.saturating_mul((cols.saturating_sub(1)) as u32));
		let cell_w = avail_w.checked_div(cols as u32).unwrap_or(1).max(1);

		let avail_h = frame_h
			.saturating_sub(outer_pad.saturating_mul(2) + title_h + bottom_band)
			.saturating_sub(vs.saturating_mul((rows.saturating_sub(1)) as u32));
		let cell_h = avail_h.checked_div(rows as u32).unwrap_or(1).max(1);

		// Shift cell x-origin right when there's a y label band.
		let pad = frame.x.max(0) as u32 + outer_pad + left_band;

		Self {
			frame,
			rows,
			cols,
			outer_pad,
			pad,
			hs,
			vs,
			top_band,
			bottom_band,
			left_band,
			cell_w,
			cell_h,
		}
	}
}

// ── Heatmap rasterizer ────────────────────────────────────────────────────────

fn render_heatmap(ui: &mut Ui<'_>, heatmap: &HeatmapCmd, rect: PixelRect) -> Result<()> {
	let rows = heatmap.rows.max(1) as usize;
	let cols = heatmap.cols.max(1) as usize;
	let values = &heatmap.values;

	// Determine data range.
	let (v_min, v_max) = if heatmap.style.auto_scale {
		let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
		if finite.is_empty() {
			(0.0_f32, 1.0_f32)
		} else {
			let mn = finite.iter().cloned().fold(f32::INFINITY, f32::min);
			let mx = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
			let range = (mx - mn).max(1e-8);
			(mn, mn + range)
		}
	} else {
		let range = (heatmap.style.v_max - heatmap.style.v_min).max(1e-8);
		(heatmap.style.v_min, heatmap.style.v_min + range)
	};

	let cell_w = (rect.w as f32 / cols as f32).max(1.0);
	let cell_h = (rect.h as f32 / rows as f32).max(1.0);

	for row in 0..rows {
		for col in 0..cols {
			let idx = row * cols + col;
			let v = values.get(idx).copied().unwrap_or(0.0);
			let t = ((v - v_min) / (v_max - v_min)).clamp(0.0, 1.0);
			let color = colormap(heatmap.style.colormap, t);
			let cx = rect.x as f32 + col as f32 * cell_w;
			let cy = rect.y as f32 + row as f32 * cell_h;
			let cell = PixelRect {
				x: cx as i32,
				y: cy as i32,
				w: (cell_w as u32).max(1),
				h: (cell_h as u32).max(1),
			};
			ui.rect_raw(cell, color, rect, 0.0);
		}
	}

	// Optional thin grid between cells.
	if heatmap.style.show_grid {
		let grid_color = Color::new(0.0, 0.0, 0.0, 0.40);
		for col in 1..cols {
			let x = rect.x as f32 + col as f32 * cell_w;
			ui.line_at(
				[x, rect.y as f32],
				[x, (rect.y + rect.h as i32) as f32],
				grid_color,
				1.0,
				rect,
			)?;
		}
		for row in 1..rows {
			let y = rect.y as f32 + row as f32 * cell_h;
			ui.line_at(
				[rect.x as f32, y],
				[(rect.x + rect.w as i32) as f32, y],
				grid_color,
				1.0,
				rect,
			)?;
		}
	}
	Ok(())
}

/// Map a normalized `t ∈ [0, 1]` through a built-in colormap.
///
/// 0 = inferno, 1 = viridis (default), 2 = greens.
fn colormap(index: u32, t: f32) -> Color {
	match index {
		0 => {
			// inferno: black → purple → orange → yellow-white
			let stops: &[(f32, f32, f32)] = &[
				(0.00, 0.00, 0.04),
				(0.23, 0.06, 0.29),
				(0.58, 0.10, 0.36),
				(0.87, 0.30, 0.17),
				(0.99, 0.65, 0.06),
				(0.99, 0.99, 0.75),
			];
			lerp_colormap(stops, t)
		}
		2 => {
			// greens
			let stops: &[(f32, f32, f32)] = &[
				(0.97, 0.98, 0.97),
				(0.78, 0.91, 0.75),
				(0.50, 0.80, 0.49),
				(0.22, 0.67, 0.31),
				(0.00, 0.46, 0.18),
				(0.00, 0.27, 0.11),
			];
			lerp_colormap(stops, t)
		}
		_ => {
			// viridis (default)
			let stops: &[(f32, f32, f32)] = &[
				(0.27, 0.00, 0.33),
				(0.27, 0.33, 0.61),
				(0.13, 0.57, 0.55),
				(0.38, 0.75, 0.38),
				(0.79, 0.88, 0.10),
				(0.99, 0.91, 0.14),
			];
			lerp_colormap(stops, t)
		}
	}
}

fn lerp_colormap(stops: &[(f32, f32, f32)], t: f32) -> Color {
	let n = stops.len();
	if n == 0 {
		return Color::new(0.0, 0.0, 0.0, 1.0);
	}
	if n == 1 {
		return Color::new(stops[0].0, stops[0].1, stops[0].2, 1.0);
	}
	let scaled = t.clamp(0.0, 1.0) * (n - 1) as f32;
	let lo = scaled.floor() as usize;
	let hi = (lo + 1).min(n - 1);
	let frac = scaled - lo as f32;
	let (r0, g0, b0) = stops[lo];
	let (r1, g1, b1) = stops[hi];
	Color::new(
		r0 + (r1 - r0) * frac,
		g0 + (g1 - g0) * frac,
		b0 + (b1 - b0) * frac,
		1.0,
	)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn format_tick(value: f32) -> String {
	if value.abs() >= 10_000.0 || (value != 0.0 && value.abs() < 0.01) {
		format!("{value:.1e}")
	} else {
		format!("{value:.2}")
	}
}
