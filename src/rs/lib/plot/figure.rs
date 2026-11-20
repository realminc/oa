//! `oa::plot::Figure` — top-level container for a grid of `oa::plot::Axes`.
//!
//! Three terminal sinks:
//! - `show()` — compose once and open the figure in the image viewer.
//! - `save_to(engine, path)` — headless GPU render → image file (Experimental).
//! - `render(engine)` — return the composition as an `oa::Image` (Experimental).
//!
//! Donor reference: `oa/ui/plot/figure.h`, `oa/ui/plot/figure.cpp`

use super::axes::{Axes, AxesLayout, Scale, TickFormat, legend_row_rect, scale_value};
pub use super::types::FigureConfig;
use crate::ui::{TextLayoutConfig, Ui, UiGridConfig, UiStyle, types::PixelRect};
use crate::{Color, Engine, Error, Image, Result, Texture};

// ── Theme ─────────────────────────────────────────────────────────────────────

/// Figure theme preset selector.
///
/// A theme resolves to a `UiStyle` preset; all figure colors come from that
/// shared token set, so plots, widgets, and viewer chrome cannot diverge.
///
/// Donor: `oa::plot::Theme`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Theme {
	#[default]
	Dark = 0,
	Light = 1,
}

impl Theme {
	/// Resolve the shared UI style behind this figure theme.
	pub fn ui_style(self) -> crate::ui::UiStyle {
		match self {
			Theme::Dark => crate::ui::UiStyle::default(),
			Theme::Light => crate::ui::UiStyle::realm_light(),
		}
	}
}

// ── FigureConfig ──────────────────────────────────────────────────────────────

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

	/// Recorded axes in row-major cell order.
	pub fn axes(&self) -> &[Axes] {
		&self.axes
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
		let matrix = crate::Matrix::new_from_f32(engine, [1, 4, h, w], &data)?;
		Image::new(matrix, crate::ImageLayout::Nchw, crate::ImageFormat::Rgba)
	}

	/// Compose the figure once and open it in the OA image viewer.
	///
	/// The window uses the image canvas with pan, zoom and fit controls.
	/// The supplied engine must support desktop presentation. This call
	/// returns when the viewer closes.
	///
	/// # Errors
	///
	/// Returns an error if composition fails or the viewer cannot open or
	/// present the texture on the supplied engine.
	pub fn show(&self, engine: &Engine) -> Result<()> {
		let texture = self.render_to_texture(engine)?;
		crate::viewer::Viewer::show_with_config(
			engine,
			&texture,
			crate::viewer::ViewerConfig {
				title: self.config.title.clone(),
				width: self.config.width,
				height: self.config.height,
				..Default::default()
			},
		)
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
		self.record_into(ui, PixelRect::new(0, 0, w, h))
	}

	/// Replay the figure inside `rect` of an already-open `Ui` GPU frame.
	///
	/// Same contract as [`Figure::render_frame`], but the figure layout is
	/// computed for `rect`'s extent and translated to its origin, so the
	/// figure can be embedded as a pane inside a composed surface (e.g. a
	/// [`crate::viewer::Workspace`] leaf).
	pub fn render_frame_at(&self, rect: PixelRect, ui: &mut Ui<'_>) -> Result<()> {
		self.record_into(ui, rect)
	}

	/// Compose the figure into a packed RGBA8 Texture.
	///
	/// Commands are lowered through the UI GPU compositor and copied through
	/// its explicit blocking headless readback boundary.
	fn render_to_texture(&self, engine: &Engine) -> Result<Texture> {
		let w = self.config.width;
		let h = self.config.height;
		let mut ui = Ui::init(engine, w, h)?;
		ui.begin_frame(0.0);
		let mut render_result = (|| -> Result<Vec<u8>> {
			self.record_into(&mut ui, PixelRect::new(0, 0, w, h))?;
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

	/// Replay retained artists into an already-open UI frame, translated to
	/// `rect`'s origin and clipped to it.
	fn record_into(&self, ui: &mut Ui<'_>, rect: PixelRect) -> Result<()> {
		let style = self.config.theme.ui_style();
		let bg = if self.config.background.a > 0.0 {
			self.config.background
		} else {
			style.plot_background
		};
		let full = rect;
		let text_color = style.text;
		let subtext_color = style.text_secondary;
		let label_cfg = TextLayoutConfig {
			size: 11.0,
			..Default::default()
		};

		// Compute the shared layout for this output size, then translate all
		// absolute positions to the target rect's origin.
		let mut layout = FigureLayout::compute(
			&self.config,
			&self.fig_x_label,
			&self.fig_y_label,
			rect.w,
			rect.h,
		);
		layout.frame.x += rect.x;
		layout.frame.y += rect.y;
		layout.pad = (layout.pad as i64 + rect.x as i64).max(0) as u32;
		layout.top_band = (layout.top_band as i64 + rect.y as i64).max(0) as u32;
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
			let text_w = ui.measure_text(&self.fig_x_label, label_cfg)?;
			let lx = layout.frame.x as f32 + (layout.frame.w as f32 - text_w) / 2.0;
			let ly = (layout.frame.y + layout.frame.h as i32
				- layout.outer_pad as i32
				- (layout.bottom_band / 2) as i32) as f32;
			ui.text_at(&self.fig_x_label, [lx, ly], label_cfg, subtext_color, full)?;
		}

		// Figure-level y label (left of the grid), bottom-to-top like
		// matplotlib's rotated axis title.
		if !self.fig_y_label.is_empty() {
			let text_w = ui.measure_text(&self.fig_y_label, label_cfg)?;
			let baseline_x =
				(layout.frame.x + layout.outer_pad as i32 + layout.left_band as i32 - 4) as f32;
			let pen_y = layout.frame.y as f32 + layout.frame.h as f32 / 2.0 + text_w / 2.0;
			ui.text_vertical_at(
				&self.fig_y_label,
				[baseline_x, pen_y],
				label_cfg,
				subtext_color,
				full,
			)?;
		}

		let rows = layout.rows;
		let cols = layout.cols;

		for r in 0..rows {
			for c in 0..cols {
				let ax_x = layout.pad + c as u32 * (layout.cell_w + layout.hs);
				let ax_y = layout.top_band + r as u32 * (layout.cell_h + layout.vs);
				let cell = PixelRect {
					x: ax_x as i32,
					y: ax_y as i32,
					w: layout.cell_w,
					h: layout.cell_h,
				};
				let axes = &self.axes[r * cols + c];
				let label_config = TextLayoutConfig {
					size: 11.0,
					..Default::default()
				};
				let subtext = style.text_secondary;
				let axes_layout = axes.layout(cell);
				let rect = axes_layout.plot;
				let has_ticks = axes.has_ticks();

				// Display-space limits; axis scales fold into every mapping.
				let limits = axes.scaled_limits();
				let x_scale = axes.cmds.x_scale;
				let y_scale = axes.cmds.y_scale;

				// Ticks and grid share one placement so labels and lines always
				// agree.  Entries are `(fraction of axis range, label)`.
				let x_ticks: Vec<(f32, String)> = if !has_ticks {
					Vec::new()
				} else if !axes.cmds.categories.is_empty() {
					let slots = axes.cmds.categories.len() as f32;
					axes
						.cmds
						.categories
						.iter()
						.enumerate()
						.map(|(index, label)| ((index as f32 + 0.5) / slots, label.clone()))
						.collect()
				} else if let Some(values) = &axes.cmds.x_ticks {
					explicit_ticks(
						values,
						&axes.cmds.x_tick_format,
						limits.x_min,
						limits.x_max,
						x_scale,
					)
				} else {
					axis_ticks(
						limits.x_min,
						limits.x_max,
						x_scale,
						(rect.w as usize / 68).clamp(2, 8),
					)
				};
				let y_ticks = if !has_ticks {
					Vec::new()
				} else if let Some(values) = &axes.cmds.y_ticks {
					explicit_ticks(
						values,
						&axes.cmds.y_tick_format,
						limits.y_min,
						limits.y_max,
						y_scale,
					)
				} else {
					axis_ticks(
						limits.y_min,
						limits.y_max,
						y_scale,
						(rect.h as usize / 48).clamp(2, 8),
					)
				};

				record_axes_text(ui, axes, &axes_layout, &style, &x_ticks, &y_ticks)?;
				record_plot_background(
					ui,
					rect,
					axes,
					self.config.theme,
					&style,
					&x_ticks,
					&y_ticks,
				);

				// Imshow raster base.
				if let Some(image) = &axes.cmds.imshow {
					let destination = if has_ticks {
						rect
					} else {
						fit_plot_image(rect, &image.texture)
					};
					ui.image_at(&image.texture, destination)?;
				}

				// Host-recorded dense heatmap — one upload per frame, lowered
				// to a single `draw_heatmap` dispatch by the compositor.
				let mut colorbar = None;
				if let Some(heatmap) = &axes.cmds.heatmap {
					let (v_min, v_max) = if heatmap.style.auto_scale {
						let mut lo = f32::INFINITY;
						let mut hi = f32::NEG_INFINITY;
						for &v in &heatmap.values {
							if v.is_finite() {
								lo = lo.min(v);
								hi = hi.max(v);
							}
						}
						if lo.is_finite() && hi.is_finite() && lo < hi {
							(lo, hi)
						} else {
							(0.0, 1.0)
						}
					} else {
						(heatmap.style.v_min, heatmap.style.v_max)
					};
					let values = crate::Matrix::new_from_f32(
						ui.engine(),
						[heatmap.rows.max(1) as usize, heatmap.cols.max(1) as usize],
						&heatmap.values,
					)?;
					ui.heatmap_matrix_at(
						&values,
						rect,
						[v_min, v_max],
						heatmap.style.colormap,
						heatmap.style.show_grid,
					)?;
					if heatmap.style.colorbar {
						colorbar = Some((v_min, v_max, heatmap.style.colormap));
					}
				}

				// Device-resident heatmap — zero-copy dispatch.
				if let Some(heatmap) = &axes.cmds.heatmap_matrix {
					if heatmap.style.auto_scale {
						return Err(Error::invalid_argument(
							"heatmap_matrix requires an explicit v_min/v_max range",
						));
					}
					ui.heatmap_matrix_at(
						&heatmap.values,
						rect,
						[heatmap.style.v_min, heatmap.style.v_max],
						heatmap.style.colormap,
						heatmap.style.show_grid,
					)?;
					if heatmap.style.colorbar {
						colorbar = Some((
							heatmap.style.v_min,
							heatmap.style.v_max,
							heatmap.style.colormap,
						));
					}
				}

				record_plot_axes(ui, rect, axes, &style, &x_ticks, &y_ticks)?;

				let palette = style.data_palette;

				// Map data-space → pixel-space closures (scale-aware; a
				// scale-dropped value maps to NaN and is skipped by callers).
				let map_x = |value: f32| {
					rect.x as f32
						+ ((scale_value(value, x_scale) - limits.x_min) / (limits.x_max - limits.x_min))
							.clamp(0.0, 1.0)
							* rect.w.saturating_sub(1) as f32
				};
				let map_y = |value: f32| {
					rect.y as f32
						+ (1.0
							- ((scale_value(value, y_scale) - limits.y_min) / (limits.y_max - limits.y_min))
								.clamp(0.0, 1.0))
							* rect.h.saturating_sub(1) as f32
				};

				// Replay artists in insertion order (donor ArtistRef contract).
				let mut line_idx = 0usize;
				let mut scatter_idx = 0usize;
				let mut bar_idx = 0usize;
				let mut matrix_idx = 0usize;
				let mut stream_idx = 0usize;
				let mut band_idx = 0usize;
				let mut box_idx = 0usize;
				// Palette index advances globally across all artist kinds,
				// hidden artists included, so toggling keeps colors stable.
				let mut palette_idx = 0usize;

				for (artist_ord, kind) in axes.cmds.artists.iter().enumerate() {
					let dim = axes.highlight.is_some_and(|h| h != artist_ord);
					match kind {
						super::axes::ArtistKind::Line => {
							let Some(line) = axes.cmds.lines.get(line_idx) else {
								continue;
							};
							line_idx += 1;
							let color = if line.style.color.a == 0.0 {
								let c = palette[palette_idx % palette.len()];
								palette_idx += 1;
								c
							} else {
								line.style.color
							};
							if line.style.hidden {
								continue;
							}
							let color = dim_series(color, dim);
							// One draw_plot_line dispatch per series;
							// non-finite points break the polyline.
							if line.style.step || x_scale != Scale::Linear || y_scale != Scale::Linear {
								let (x, y) =
									transformed_points(&line.x, &line.y, x_scale, y_scale, line.style.step);
								ui.plot_xy_at(
									&x,
									&y,
									rect,
									limits.x_min,
									limits.x_max,
									limits.y_min,
									limits.y_max,
									color,
									line.style.antialias_samples,
									line.style.width,
								)?;
							} else {
								ui.plot_xy_at(
									&line.x,
									&line.y,
									rect,
									limits.x_min,
									limits.x_max,
									limits.y_min,
									limits.y_max,
									color,
									line.style.antialias_samples,
									line.style.width,
								)?;
							}
						}
						super::axes::ArtistKind::Scatter => {
							let Some(scatter) = axes.cmds.scatters.get(scatter_idx) else {
								continue;
							};
							scatter_idx += 1;
							let color = if scatter.style.color.a == 0.0 {
								let c = palette[palette_idx % palette.len()];
								palette_idx += 1;
								c
							} else {
								scatter.style.color
							};
							if scatter.style.hidden {
								continue;
							}
							let color = dim_series(color, dim);
							for (&x, &y) in scatter.x.iter().zip(&scatter.y) {
								let point = [map_x(x), map_y(y)];
								if point[0].is_finite() && point[1].is_finite() {
									ui.line_at(point, point, color, scatter.style.radius * 2.0, rect)?;
								}
							}
						}
						super::axes::ArtistKind::Bar => {
							let Some(bars) = axes.cmds.bars.get(bar_idx) else {
								continue;
							};
							bar_idx += 1;
							let color = if bars.style.color.a == 0.0 {
								let c = palette[palette_idx % palette.len()];
								palette_idx += 1;
								c
							} else {
								bars.style.color
							};
							if bars.style.hidden {
								continue;
							}
							let color = dim_series(color, dim);
							let slot = rect.w as f32 / bars.y.len().max(1) as f32;
							let (group_index, group_size) = bars.group.unwrap_or((0, 1));
							let sub_slot = slot / group_size.max(1) as f32;
							let width = (sub_slot * (1.0 - bars.style.gap.clamp(0.0, 0.95))).max(1.0) as u32;
							// A logarithmic y axis has no zero baseline; bars
							// grow from the bottom of the data range instead.
							let zero_base = if y_scale == Scale::Log10 {
								(rect.y + rect.h.saturating_sub(1) as i32) as f32
							} else {
								map_y(0.0)
							};
							for (index, &value) in bars.y.iter().enumerate() {
								let stack = bars.stack_base.as_ref().map_or(0.0, |bases| bases[index]);
								let y0 = if bars.stack_base.is_some() {
									map_y(stack)
								} else {
									zero_base
								};
								let y1 = map_y(stack + value);
								if !y0.is_finite() || !y1.is_finite() {
									continue;
								}
								let bar = PixelRect {
									x: (rect.x as f32
										+ index as f32 * slot
										+ group_index as f32 * sub_slot
										+ (sub_slot - width as f32) * 0.5) as i32,
									y: y0.min(y1) as i32,
									w: width,
									h: (y0 - y1).abs().max(1.0) as u32,
								};
								ui.rect_raw(bar, color, rect, 0.0);
							}
						}
						super::axes::ArtistKind::LineMatrix => {
							let Some(series) = axes.cmds.line_matrices.get(matrix_idx) else {
								continue;
							};
							matrix_idx += 1;
							let color = if series.style.color.a == 0.0 {
								let c = palette[palette_idx % palette.len()];
								palette_idx += 1;
								c
							} else {
								series.style.color
							};
							if series.style.hidden {
								continue;
							}
							let color = dim_series(color, dim);
							// Device data cannot be transformed host-side; a
							// non-linear axis fails rather than drawing the
							// series against mismatched limits.
							if x_scale != Scale::Linear || y_scale != Scale::Linear {
								return Err(Error::invalid_argument(
									"plot_matrix series require linear axes",
								));
							}
							if series.xy {
								ui.plot_xy_matrix_at(
									&series.values,
									rect,
									limits.x_min,
									limits.x_max,
									limits.y_min,
									limits.y_max,
									color,
									series.style.antialias_samples,
									series.style.width,
								)?;
							} else {
								ui.plot_line_matrix_at(
									&series.values,
									rect,
									limits.y_min,
									limits.y_max,
									color,
									series.style.width,
								)?;
							}
						}
						super::axes::ArtistKind::Stream => {
							let Some(stream) = axes.cmds.streams.get(stream_idx) else {
								continue;
							};
							stream_idx += 1;
							let color = if stream.style.color.a == 0.0 {
								let c = palette[palette_idx % palette.len()];
								palette_idx += 1;
								c
							} else {
								stream.style.color
							};
							if stream.style.hidden || stream.len == 0 {
								continue;
							}
							let color = dim_series(color, dim);
							let y: Vec<f32> = stream.ordered().collect();
							let x: Vec<f32> = (0..y.len()).map(|i| i as f32).collect();
							let (x, y) = transformed_points(&x, &y, x_scale, y_scale, stream.style.step);
							ui.plot_xy_at(
								&x,
								&y,
								rect,
								limits.x_min,
								limits.x_max,
								limits.y_min,
								limits.y_max,
								color,
								stream.style.antialias_samples,
								stream.style.width,
							)?;
						}
						super::axes::ArtistKind::Band => {
							let Some(band) = axes.cmds.bands.get(band_idx) else {
								continue;
							};
							band_idx += 1;
							let color = if band.style.color.a == 0.0 {
								let c = palette[palette_idx % palette.len()];
								palette_idx += 1;
								c
							} else {
								band.style.color
							};
							if band.style.hidden {
								continue;
							}
							let color = dim_series(color, dim);
							// Band x positions are uniform sample indices; a
							// non-linear x axis would need explicit positions
							// the band path does not carry.
							if x_scale != Scale::Linear {
								return Err(Error::invalid_argument(
									"band artists require a linear x axis",
								));
							}
							let lo: Vec<f32> = band.y_lo.iter().map(|&v| scale_value(v, y_scale)).collect();
							let hi: Vec<f32> = band.y_hi.iter().map(|&v| scale_value(v, y_scale)).collect();
							ui.plot_band_at(&lo, &hi, rect, limits.y_min, limits.y_max, color)?;
						}
						super::axes::ArtistKind::BoxPlot => {
							let Some(cmd) = axes.cmds.boxes.get(box_idx) else {
								continue;
							};
							box_idx += 1;
							let base = if cmd.style.color.a == 0.0 {
								let c = palette[palette_idx % palette.len()];
								palette_idx += 1;
								c
							} else {
								cmd.style.color
							};
							if cmd.style.hidden {
								continue;
							}
							let color = dim_series(base, dim);
							let count = cmd.summaries.len().max(1) as f32;
							let slot = rect.w as f32 / count;
							let width = (slot * cmd.style.width.clamp(0.0, 0.95)).max(2.0);
							for &(index, q1, median, q3, lo, hi) in &cmd.summaries {
								let (y_q1, y_median, y_q3, y_lo, y_hi) =
									(map_y(q1), map_y(median), map_y(q3), map_y(lo), map_y(hi));
								if ![y_q1, y_median, y_q3, y_lo, y_hi]
									.iter()
									.all(|y| y.is_finite())
								{
									continue;
								}
								let cx = rect.x as f32 + index * slot + slot * 0.5;
								let left = cx - width * 0.5;
								let fill = PixelRect {
									x: left as i32,
									y: y_q3.min(y_q1) as i32,
									w: width as u32,
									h: (y_q1 - y_q3).abs().max(1.0) as u32,
								};
								ui.rect_raw(
									fill,
									Color::new(color.r, color.g, color.b, color.a * 0.4),
									rect,
									0.0,
								);
								ui.rect_outline_raw(fill, color, 1, rect);
								// Whisker spine and caps.
								ui.line_at([cx, y_lo], [cx, y_hi], color, 1.0, rect)?;
								ui.line_at(
									[cx - width * 0.25, y_lo],
									[cx + width * 0.25, y_lo],
									color,
									1.0,
									rect,
								)?;
								ui.line_at(
									[cx - width * 0.25, y_hi],
									[cx + width * 0.25, y_hi],
									color,
									1.0,
									rect,
								)?;
								// Median.
								ui.line_at([left, y_median], [left + width, y_median], color, 1.5, rect)?;
							}
						}
					}
				}

				// Data-space annotations.
				for note in &axes.cmds.annotations {
					let point = [map_x(note.x), map_y(note.y)];
					if point[0].is_finite() && point[1].is_finite() {
						ui.text_at(
							&note.text,
							[point[0] + 3.0, point[1] - 3.0],
							label_config,
							text_color,
							rect,
						)?;
					}
				}

				// Heatmap colorbar — a vertical value-ramp strip with
				// min/mid/max labels in the reserved right band.
				if let Some((v_min, v_max, colormap)) = colorbar
					&& cell.x + cell.w as i32 - (rect.x + rect.w as i32) >= 20
				{
					const RAMP: usize = 32;
					// Row 0 sits at the top of the strip and shows v_max.
					let ramp: Vec<f32> = (0..RAMP)
						.map(|index| v_max - (v_max - v_min) * index as f32 / (RAMP - 1).max(1) as f32)
						.collect();
					let strip = PixelRect::new(
						rect.x + rect.w as i32 + 6,
						rect.y + 8,
						10,
						rect.h.saturating_sub(16).max(1),
					);
					let values = crate::Matrix::new_from_f32(ui.engine(), [RAMP, 1], &ramp)?;
					ui.heatmap_matrix_at(&values, strip, [v_min, v_max], colormap, false)?;
					let step = ((v_max - v_min).abs() / 4.0).max(f32::EPSILON);
					for (fraction, value) in [(0.0, v_max), (0.5, (v_min + v_max) * 0.5), (1.0, v_min)] {
						ui.text_at(
							&format_tick(value, step),
							[
								(strip.x + 14) as f32,
								strip.y as f32 + 8.0 + fraction * strip.h.saturating_sub(16) as f32,
							],
							label_config,
							subtext,
							cell,
						)?;
					}
				}

				if axes.cmds.show_legend {
					record_legend(ui, rect, axes, &style, label_config)?;
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

// ── Helpers ───────────────────────────────────────────────────────────────────

fn fit_plot_image(rect: PixelRect, texture: &Texture) -> PixelRect {
	let width = texture.width() as f64;
	let height = texture.height() as f64;
	let scale = (f64::from(rect.w) / width).min(f64::from(rect.h) / height);
	let w = (width * scale).round().clamp(1.0, f64::from(rect.w)) as u32;
	let h = (height * scale).round().clamp(1.0, f64::from(rect.h)) as u32;
	PixelRect::new(
		rect.x + ((rect.w - w) / 2) as i32,
		rect.y + ((rect.h - h) / 2) as i32,
		w,
		h,
	)
}

fn record_legend(
	ui: &mut Ui<'_>,
	rect: PixelRect,
	axes: &Axes,
	style: &UiStyle,
	config: TextLayoutConfig,
) -> Result<()> {
	let entries = axes.legend_entries(&style.data_palette);
	let visible_rows = entries
		.iter()
		.filter(|entry| entry.row != usize::MAX && legend_row_rect(rect, entry.row).is_non_empty())
		.count();
	if visible_rows == 0 {
		return Ok(());
	}
	let first = legend_row_rect(rect, 0);
	let panel = PixelRect::new(first.x, first.y, first.w, first.h * visible_rows as u32);
	let mut background = style.plot_background;
	background.a = 0.94;
	ui.rect_raw(panel, background, rect, 3.0);

	for entry in entries.iter().filter(|entry| entry.row < visible_rows) {
		let row = legend_row_rect(rect, entry.row);
		let x = row.x as f32;
		let y = row.y as f32 + row.h as f32 / 2.0;
		let dim = entry.hidden || axes.highlight.is_some_and(|artist| artist != entry.artist);
		ui.line_at(
			[x + 5.0, y],
			[x + 18.0, y],
			dim_series(entry.color, dim),
			2.0,
			row,
		)?;
		ui.text_at(
			&entry.label,
			[x + 23.0, y + 4.0],
			config,
			dim_series(style.text_secondary, dim),
			row,
		)?;
	}
	Ok(())
}

fn record_axes_text(
	ui: &mut Ui<'_>,
	axes: &Axes,
	layout: &AxesLayout,
	style: &UiStyle,
	x_ticks: &[(f32, String)],
	y_ticks: &[(f32, String)],
) -> Result<()> {
	let text = TextLayoutConfig {
		size: 12.0,
		..Default::default()
	};
	let ticks = TextLayoutConfig {
		size: 10.0,
		..Default::default()
	};
	let plot = layout.plot;
	let center_x = plot.x as f32 + plot.w as f32 / 2.0;
	let center_y = plot.y as f32 + plot.h as f32 / 2.0;
	let subtext = style.text_secondary;
	for (label, band, config, color) in [
		(
			&axes.cmds.title,
			layout.title,
			text,
			if axes.cmds.title_color.a > 0.0 {
				axes.cmds.title_color
			} else {
				style.text
			},
		),
		(
			&axes.cmds.caption,
			layout.caption,
			ticks,
			if axes.cmds.caption_color.a > 0.0 {
				axes.cmds.caption_color
			} else {
				subtext
			},
		),
		(&axes.cmds.x_label, layout.x_label, text, subtext),
	] {
		if !label.is_empty() && band.h >= 12 {
			let width = ui.measure_text(label, config)?;
			let baseline = band.y as f32 + band.h as f32 / 2.0 + 4.0;
			ui.text_at(
				label,
				[center_x - width / 2.0, baseline],
				config,
				color,
				band,
			)?;
		}
	}
	if !axes.cmds.y_label.is_empty() && layout.y_label.w >= 12 {
		let width = ui.measure_text(&axes.cmds.y_label, text)?;
		let x = layout.y_label.x as f32 + layout.y_label.w as f32 / 2.0 + 4.0;
		ui.text_vertical_at(
			&axes.cmds.y_label,
			[x, center_y + width / 2.0],
			text,
			subtext,
			layout.y_label,
		)?;
	}
	// Tick values are centered below X and right-aligned beside Y. Permit
	// endpoint labels to extend past the viewport into their own gutters.
	if layout.x_ticks.h >= 12 {
		let band = layout.x_ticks;
		let left = band.x as f32;
		let right = left + band.w as f32;
		let mut labels = Vec::with_capacity(x_ticks.len());
		for (fraction, label) in x_ticks {
			let width = ui.measure_text(label, ticks)?;
			let center = plot.x as f32 + 0.5 + fraction * plot.w.saturating_sub(1) as f32;
			let x = (center - width / 2.0).clamp(left, (right - width).max(left));
			labels.push((x, width, label));
		}
		labels.sort_by(|a, b| a.0.total_cmp(&b.0));
		let mut end = left - 4.0;
		for (x, width, label) in labels {
			if x >= end + 4.0 {
				ui.text_at(label, [x, band.y as f32 + 13.0], ticks, subtext, band)?;
				end = x + width;
			}
		}
	}
	if layout.y_ticks.w >= 12 {
		let band = layout.y_ticks;
		let right = (plot.x - 6) as f32;
		let mut previous = f32::INFINITY;
		for (fraction, label) in y_ticks {
			let y = plot.y as f32 + (1.0 - fraction) * plot.h.saturating_sub(1) as f32 + 4.0;
			if (previous - y).abs() >= 12.0 {
				let width = ui.measure_text(label, ticks)?;
				ui.text_at(label, [right - width, y], ticks, subtext, band)?;
				previous = y;
			}
		}
	}
	Ok(())
}

fn record_plot_background(
	ui: &mut Ui<'_>,
	rect: PixelRect,
	axes: &Axes,
	theme: Theme,
	style: &UiStyle,
	x_ticks: &[(f32, String)],
	y_ticks: &[(f32, String)],
) {
	if theme == Theme::Light {
		ui.rect_raw(rect, style.plot_surface, rect, 0.0);
		return;
	}

	// Reuse the donor canvas grid for a neutral gradient and fine subdivisions.
	// Add subdivisions only to regular, automatic linear ticks. Major lines
	// are recorded separately so custom and nonlinear ticks stay exact.
	let surface = style.plot_surface;
	let mut grid = UiGridConfig {
		background_top: Color::new(surface.r + 0.031, surface.g + 0.031, surface.b + 0.031, 1.0),
		background_bottom: Color::new(surface.r - 0.020, surface.g - 0.020, surface.b - 0.020, 1.0),
		draw_axes: false,
		draw_grid: false,
		..Default::default()
	};
	if axes.cmds.show_grid
		&& axes.cmds.categories.is_empty()
		&& axes.cmds.x_ticks.is_none()
		&& axes.cmds.y_ticks.is_none()
		&& axes.cmds.x_scale == Scale::Linear
		&& axes.cmds.y_scale == Scale::Linear
		&& let [x0, x1, ..] = x_ticks
		&& let [y0, y1, ..] = y_ticks
	{
		let w = rect.w.saturating_sub(1) as f32;
		let h = rect.h.saturating_sub(1) as f32;
		let dx = (x1.0 - x0.0) * w / 10.0;
		let dy = (y1.0 - y0.0) * h / 10.0;
		if dx >= 4.0 && dy >= 4.0 {
			grid.origin = [
				rect.x as f32 + 0.5 + x0.0 * w,
				rect.y as f32 + 0.5 + (1.0 - y0.0) * h,
			];
			grid.minor_spacing = [dx, dy];
			grid.major_color = Color::new(1.0, 1.0, 1.0, 0.01);
			grid.super_major_color = grid.major_color;
			grid.draw_grid = true;
		}
	}
	ui.grid(rect, grid);
}

fn record_plot_axes(
	ui: &mut Ui<'_>,
	rect: PixelRect,
	axes: &Axes,
	style: &UiStyle,
	x_ticks: &[(f32, String)],
	y_ticks: &[(f32, String)],
) -> Result<()> {
	let x = rect.x as f32 + 0.5;
	let y = rect.y as f32 + 0.5;
	let w = rect.w.saturating_sub(1) as f32;
	let h = rect.h.saturating_sub(1) as f32;
	if axes.cmds.show_grid {
		for &(fraction, _) in x_ticks {
			if fraction > f32::EPSILON && fraction < 1.0 - f32::EPSILON {
				let tick = x + fraction * w;
				ui.line_at([tick, y], [tick, y + h], style.plot_grid, 1.0, rect)?;
			}
		}
		for &(fraction, _) in y_ticks {
			if fraction > f32::EPSILON && fraction < 1.0 - f32::EPSILON {
				let tick = y + (1.0 - fraction) * h;
				ui.line_at([x, tick], [x + w, tick], style.plot_grid, 1.0, rect)?;
			}
		}
	}
	if let Some(border) = axes.cmds.border_color {
		ui.rect_outline_raw(rect, border, 1, rect);
	} else if axes.has_ticks() {
		let bottom = rect.y + rect.h.saturating_sub(1) as i32;
		ui.rect_raw(
			PixelRect::new(rect.x, rect.y, 1, rect.h),
			style.plot_border,
			rect,
			0.0,
		);
		ui.rect_raw(
			PixelRect::new(rect.x, bottom, rect.w, 1),
			style.plot_border,
			rect,
			0.0,
		);
	}
	Ok(())
}

/// Emphasis factor for legend-hover highlighting — the hovered series keeps
/// full alpha; the rest fade to a quarter.
fn dim_series(color: Color, dim: bool) -> Color {
	if dim {
		Color::new(color.r, color.g, color.b, color.a * 0.25)
	} else {
		color
	}
}

/// Tick positions for one axis as `(fraction of range, label)` pairs.
///
/// Linear axes use nice-number ticks; a `Log10` axis places ticks on whole
/// decades with data-space labels (`1e3` at display position `log10(1e3)`),
/// falling back to linear ticks when the range covers less than a decade.
fn axis_ticks(min: f32, max: f32, scale: Scale, target: usize) -> Vec<(f32, String)> {
	if !min.is_finite() || !max.is_finite() || min >= max {
		return Vec::new();
	}
	let span = (max - min).max(f32::EPSILON);
	let linear = |min: f32, max: f32| {
		let ticks = nice_ticks(min, max, target);
		let step = tick_step(&ticks);
		ticks
			.iter()
			.map(|&value| ((value - min) / span, format_tick(value, step)))
			.collect()
	};
	match scale {
		Scale::Linear => linear(min, max),
		Scale::Log10 => {
			let first = min.ceil() as i32;
			let last = max.floor() as i32;
			if last - first >= 1 {
				(first..=last)
					.map(|k| ((k as f32 - min) / span, format!("1e{k}")))
					.collect()
			} else {
				linear(min, max)
			}
		}
		Scale::Symlog => {
			let mut ticks: Vec<(f32, String)> = Vec::new();
			if min <= 0.0 && max >= 0.0 {
				ticks.push((-min / span, "0".to_owned()));
			}
			for sign in [-1f32, 1f32] {
				for k in 0..32 {
					// Display position `±linthresh·(1+k)` is data `±10^k`.
					let pos = sign * super::axes::SYMLOG_LINTHRESH * (1.0 + k as f32);
					if pos > max {
						break;
					}
					if pos >= min {
						let negative = if sign < 0.0 { "-" } else { "" };
						ticks.push(((pos - min) / span, format!("{negative}1e{k}")));
					}
				}
			}
			if ticks.len() <= 1 {
				linear(min, max)
			} else {
				ticks.sort_by(|a, b| a.0.total_cmp(&b.0));
				ticks
			}
		}
	}
}

/// Format caller-supplied data-space tick positions as `(fraction, label)`.
///
/// Positions map through the axis scale into display space; scale-dropped
/// and out-of-range values are skipped. Labels show the data-space value.
fn explicit_ticks(
	values: &[f32],
	format: &TickFormat,
	min: f32,
	max: f32,
	scale: Scale,
) -> Vec<(f32, String)> {
	let span = (max - min).max(f32::EPSILON);
	let step = values
		.windows(2)
		.map(|pair| (pair[1] - pair[0]).abs())
		.fold(f32::INFINITY, f32::min)
		.max(f32::EPSILON);
	let step = if step.is_finite() { step } else { 1.0 };
	values
		.iter()
		.enumerate()
		.filter_map(|(index, &value)| {
			let position = scale_value(value, scale);
			if !position.is_finite() || position < min || position > max {
				return None;
			}
			let label = match format {
				TickFormat::Labels(labels) => labels.get(index)?.clone(),
				TickFormat::Decimals(d) => format!("{value:.d$}"),
				TickFormat::Scientific => format!("{value:.1e}"),
				TickFormat::Auto => format_tick(value, step),
			};
			Some(((position - min) / span, label))
		})
		.collect()
}

/// Transform a series' raw samples into display space.
///
/// `step` expands the polyline into step-post vertices — each y value holds
/// until the next x — and the axis scales map every sample afterward;
/// scale-dropped values become `NaN` and break the polyline in the shader.
fn transformed_points(
	x: &[f32],
	y: &[f32],
	x_scale: Scale,
	y_scale: Scale,
	step: bool,
) -> (Vec<f32>, Vec<f32>) {
	let count = if step { x.len() * 2 - 1 } else { x.len() }.max(x.len());
	let mut tx = Vec::with_capacity(count);
	let mut ty = Vec::with_capacity(count);
	if step {
		for index in 0..x.len() {
			tx.push(x[index]);
			ty.push(y[index]);
			if index + 1 < x.len() {
				tx.push(x[index + 1]);
				ty.push(y[index]);
			}
		}
	} else {
		tx.extend_from_slice(x);
		ty.extend_from_slice(y);
	}
	for value in &mut tx {
		*value = scale_value(*value, x_scale);
	}
	for value in &mut ty {
		*value = scale_value(*value, y_scale);
	}
	(tx, ty)
}

/// Nice-number tick values spanning `[min, max]` at roughly `target` count,
/// using the 1/2/5 step family.
fn nice_ticks(min: f32, max: f32, target: usize) -> Vec<f32> {
	if !min.is_finite() || !max.is_finite() || min >= max || target == 0 {
		return Vec::new();
	}
	let range = f64::from(max - min);
	let raw = range / target as f64;
	let magnitude = 10f64.powf(raw.log10().floor());
	let normalized = raw / magnitude;
	let step = if normalized < 1.5 {
		magnitude
	} else if normalized < 3.0 {
		2.0 * magnitude
	} else if normalized < 7.0 {
		5.0 * magnitude
	} else {
		10.0 * magnitude
	};
	let start = (f64::from(min) / step).ceil();
	let end = (f64::from(max) / step).floor();
	let count = (end - start).max(0.0) as usize + 1;
	if end < start || count > 64 {
		return vec![min, max];
	}
	(0..count)
		.map(|index| ((start + index as f64) * step) as f32)
		.collect()
}

/// Spacing between consecutive nice ticks, used to pick label precision.
fn tick_step(ticks: &[f32]) -> f32 {
	ticks
		.windows(2)
		.map(|pair| (pair[1] - pair[0]).abs())
		.next()
		.unwrap_or(1.0)
		.max(f32::EPSILON)
}

/// Format a tick value with the precision implied by the tick step.
///
/// Nice steps are 1/2/5 × 10^k, so `max(0, -floor(log10(step)))` decimals
/// always render the step exactly.
fn format_tick(value: f32, step: f32) -> String {
	if value.abs() >= 1.0e6 || (value != 0.0 && value.abs() < 1.0e-4) {
		return format!("{value:.1e}");
	}
	let decimals = (-f64::from(step).abs().log10().floor()).max(0.0) as usize;
	format!("{value:.decimals$}")
}
