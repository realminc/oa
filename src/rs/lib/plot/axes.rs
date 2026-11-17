//! `oa::plot::Axes` — one subplot inside an `oa::plot::Figure`.
//!
//! Records one optional raster base plus ordered vector artists.
//! The parent Figure lowers the same commands through `Ui`'s GPU compositor
//! for both live (`show`) and terminal (`save_to` / `render`) sinks.
//!
//! Donor reference: `oa/ui/plot/axes.h`, `oa/ui/plot/axes.cpp`

use crate::Color;
use crate::ui::types::PixelRect;

// ── Axis scaling ──────────────────────────────────────────────────────────────

/// Data-to-display transform applied per axis at replay.
///
/// The transform happens in display space: ticks, grid, artists, and
/// `FigureView` pixel mapping all share `Axes::scaled_limits`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scale {
	/// Identity mapping.
	#[default]
	Linear,
	/// Base-10 logarithm. Non-positive samples drop from artists; ticks land
	/// on whole decades. Device-resident series (`plot_matrix`) cannot be
	/// transformed host-side and require linear axes.
	Log10,
	/// Symmetric log: linear inside `±SYMLOG_LINTHRESH`, logarithmic beyond.
	/// Zero and negative data stay on the axis; ticks land on symmetric
	/// decades plus zero.
	Symlog,
}

/// Magnitude at which a [`Scale::Symlog`] axis switches from linear to log.
pub const SYMLOG_LINTHRESH: f32 = 1.0;

/// Transform one data value into display space; `NaN` drops the point.
pub(super) fn scale_value(value: f32, scale: Scale) -> f32 {
	match scale {
		Scale::Linear => value,
		Scale::Log10 => {
			if value > 0.0 && value.is_finite() {
				value.log10()
			} else {
				f32::NAN
			}
		}
		Scale::Symlog => {
			if !value.is_finite() {
				return f32::NAN;
			}
			let magnitude = value.abs();
			if magnitude <= SYMLOG_LINTHRESH {
				value
			} else {
				value.signum() * SYMLOG_LINTHRESH * (1.0 + (magnitude / SYMLOG_LINTHRESH).log10())
			}
		}
	}
}

/// Inverse of [`scale_value`], used for readouts and limit write-back.
pub(super) fn unscale_value(value: f32, scale: Scale) -> f32 {
	match scale {
		Scale::Linear => value,
		Scale::Log10 => 10f32.powf(value),
		Scale::Symlog => {
			let magnitude = value.abs();
			if magnitude <= SYMLOG_LINTHRESH {
				value
			} else {
				value.signum() * SYMLOG_LINTHRESH * 10f32.powf(magnitude / SYMLOG_LINTHRESH - 1.0)
			}
		}
	}
}

/// Resolve one axis' raw `[lo, hi]` into display space.
///
/// A logarithmic axis whose lower bound is non-positive floors at the
/// smallest positive recorded value; an axis without any positive data
/// falls back to the `[0, 1]` display decade.
fn scaled_axis(lo: f32, hi: f32, scale: Scale, positive_min: Option<f32>) -> (f32, f32) {
	match scale {
		Scale::Linear => (lo, hi),
		Scale::Log10 => {
			let lo = if lo.is_finite() && lo > 0.0 {
				lo
			} else if let Some(min) = positive_min {
				min
			} else {
				return (0.0, 1.0);
			};
			let hi = if hi.is_finite() && hi > lo {
				hi
			} else {
				lo * 10.0
			};
			(lo.log10(), hi.log10())
		}
		Scale::Symlog => {
			let lo = scale_value(lo, scale);
			let hi = scale_value(hi, scale);
			if lo.is_finite() && hi.is_finite() && lo < hi {
				(lo, hi)
			} else {
				(-SYMLOG_LINTHRESH, SYMLOG_LINTHRESH)
			}
		}
	}
}

/// Axis tick label presentation paired with explicit tick positions.
///
/// Applies to ticks recorded through [`Axes::xticks`]/[`Axes::yticks`];
/// `Labels` pairs one string per explicit value and requires equal lengths.
#[derive(Clone, Debug, Default)]
pub enum TickFormat {
	/// Precision derived from the explicit positions' spacing.
	#[default]
	Auto,
	/// Fixed decimal places — `Decimals(2)` renders `1.25`.
	Decimals(usize),
	/// Scientific notation — `1.2e3`.
	Scientific,
	/// Caller-provided labels, one per explicit tick value in order.
	Labels(Vec<String>),
}

// ── Style descriptors ─────────────────────────────────────────────────────────

/// Per-series line style.
#[derive(Clone, Debug)]
pub struct LineStyle {
	/// Alpha-zero selects the next deterministic theme palette color.
	pub color: Color,
	pub label: String,
	pub width: f32,
	pub antialias_samples: u32,
	/// Draw as a step function: each y value holds to the next x
	/// (matplotlib `steps-post`).
	pub step: bool,
	/// Skip rendering while keeping the palette slot and legend entry.
	pub hidden: bool,
}

impl Default for LineStyle {
	fn default() -> Self {
		Self {
			color: Color::new(0.0, 0.0, 0.0, 0.0), // palette auto
			label: String::new(),
			width: 1.35,
			antialias_samples: 4,
			step: false,
			hidden: false,
		}
	}
}

/// Per-series scatter style.
#[derive(Clone, Debug)]
pub struct ScatterStyle {
	pub color: Color,
	pub label: String,
	pub radius: f32,
	/// Skip rendering while keeping the palette slot and legend entry.
	pub hidden: bool,
}

impl Default for ScatterStyle {
	fn default() -> Self {
		Self {
			color: Color::new(0.0, 0.0, 0.0, 0.0),
			label: String::new(),
			radius: 3.0,
			hidden: false,
		}
	}
}

/// Per-series bar style.
#[derive(Clone, Debug)]
pub struct BarStyle {
	pub color: Color,
	pub label: String,
	/// Gap fraction between bars `[0, 0.95]`.
	pub gap: f32,
	/// Skip rendering while keeping the palette slot and legend entry.
	pub hidden: bool,
}

impl Default for BarStyle {
	fn default() -> Self {
		Self {
			color: Color::new(0.0, 0.0, 0.0, 0.0),
			label: String::new(),
			gap: 0.18,
			hidden: false,
		}
	}
}

/// Dense row-major heatmap style.
///
/// Donor: `oa::plot::HeatmapStyle`.
#[derive(Clone, Debug)]
pub struct HeatmapStyle {
	pub v_min: f32,
	pub v_max: f32,
	/// Colormap index evaluated by the shared `draw_heatmap` kernel:
	/// 0 = plasma, 1 = viridis (default), 2 = blue-white-orange diverging,
	/// 3 = grayscale.
	pub colormap: u32,
	/// Scale `v_min`/`v_max` automatically from data.
	///
	/// Only the host `heatmap` path can inspect values; the device-resident
	/// `heatmap_matrix` path requires an explicit `v_min`/`v_max` range.
	pub auto_scale: bool,
	/// Draw a thin grid between cells.
	pub show_grid: bool,
	/// Reserve a strip at the right of the cell for a value-ramp colorbar
	/// with min/mid/max labels.
	pub colorbar: bool,
}

/// Box-summary style shared by every dataset in one `boxplot` call.
#[derive(Clone, Debug)]
pub struct BoxStyle {
	/// Alpha-zero cycles the theme palette per dataset slot.
	pub color: Color,
	pub label: String,
	/// Box width as a fraction of the dataset slot `[0, 0.95]`.
	pub width: f32,
	/// Whisker reach in interquartile-range units (matplotlib `whis`).
	pub whisk: f32,
	/// Skip rendering while keeping the palette slot and legend entry.
	pub hidden: bool,
}

impl Default for BoxStyle {
	fn default() -> Self {
		Self {
			color: Color::new(0.0, 0.0, 0.0, 0.0),
			label: String::new(),
			width: 0.5,
			whisk: 1.5,
			hidden: false,
		}
	}
}

impl Default for HeatmapStyle {
	fn default() -> Self {
		Self {
			v_min: 0.0,
			v_max: 1.0,
			colormap: 1, // viridis
			auto_scale: true,
			show_grid: true,
			colorbar: false,
		}
	}
}

// ── Internal command records ──────────────────────────────────────────────────

/// Insertion-order artist kind — mirrors the donor's `ArtistRef`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArtistKind {
	Line,
	Scatter,
	Bar,
	/// Device-resident `Matrix` line series.
	LineMatrix,
	/// Host ring-buffer series appended through `Axes::stream_push`.
	Stream,
	/// Filled band between two y curves.
	Band,
	/// Quartile boxes over one or more datasets.
	BoxPlot,
}

/// Retained commands within one Axes.
#[derive(Default, Clone)]
pub(super) struct AxesCommands {
	pub(super) title: String,
	pub(super) title_color: Color,
	pub(super) caption: String,
	pub(super) caption_color: Color,
	pub(super) x_label: String,
	pub(super) y_label: String,
	pub(super) imshow: Option<ImshowCmd>,
	pub(super) heatmap: Option<Heatmap>,
	pub(super) heatmap_matrix: Option<HeatmapMatrixCmd>,
	pub(super) lines: Vec<LineCmd>,
	pub(super) scatters: Vec<ScatterCmd>,
	pub(super) bars: Vec<BarCmd>,
	pub(super) line_matrices: Vec<LineMatrixCmd>,
	pub(super) streams: Vec<StreamCmd>,
	pub(super) bands: Vec<BandCmd>,
	pub(super) boxes: Vec<BoxCmd>,
	pub(super) annotations: Vec<AnnotationCmd>,
	/// Insertion-order list of artists — used by Figure to replay in call order.
	pub(super) artists: Vec<ArtistKind>,
	pub(super) limits: Option<LimitsCmd>,
	pub(super) x_scale: Scale,
	pub(super) y_scale: Scale,
	/// Explicit data-space tick positions; `None` = nice-number locator.
	pub(super) x_ticks: Option<Vec<f32>>,
	pub(super) y_ticks: Option<Vec<f32>>,
	pub(super) x_tick_format: TickFormat,
	pub(super) y_tick_format: TickFormat,
	pub(super) categories: Vec<String>,
	pub(super) show_grid: bool,
	pub(super) show_legend: bool,
	pub(super) border_color: Option<Color>,
}

impl AxesCommands {
	pub(super) fn new() -> Self {
		Self {
			show_grid: true,
			show_legend: true,
			..Default::default()
		}
	}
}

#[derive(Clone)]
pub(super) struct ImshowCmd {
	pub(super) texture: crate::Texture,
}

#[derive(Clone)]
pub(super) struct Heatmap {
	pub(super) values: Vec<f32>,
	pub(super) rows: i32,
	pub(super) cols: i32,
	pub(super) style: HeatmapStyle,
}

/// Device-resident dense heatmap; the matrix stays on the GPU.
#[derive(Clone)]
pub(super) struct HeatmapMatrixCmd {
	pub(super) values: crate::Matrix,
	pub(super) style: HeatmapStyle,
}

#[derive(Clone)]
pub(super) struct LineCmd {
	pub(super) x: Vec<f32>,
	pub(super) y: Vec<f32>,
	pub(super) style: LineStyle,
}

#[derive(Clone)]
pub(super) struct ScatterCmd {
	pub(super) x: Vec<f32>,
	pub(super) y: Vec<f32>,
	pub(super) style: ScatterStyle,
}

#[derive(Clone)]
pub(super) struct BarCmd {
	pub(super) y: Vec<f32>,
	/// `(index, size)` inside each x slot for grouped bars; `None` = full slot.
	pub(super) group: Option<(usize, usize)>,
	/// Running-sum baseline per bar for stacked layers; `None` = zero base.
	pub(super) stack_base: Option<Vec<f32>>,
	pub(super) style: BarStyle,
}

/// Device-resident line series; the matrix stays on the GPU.
///
/// `xy` selects the layout contract: `true` = rank-2 `[2, N]` with one X
/// block followed by one Y block (the `draw_plot_line` explicit-XY ABI);
/// `false` = rank-1 `[N]` y-vs-index.
#[derive(Clone)]
pub(super) struct LineMatrixCmd {
	pub(super) values: crate::Matrix,
	pub(super) xy: bool,
	pub(super) style: LineStyle,
}

/// Host ring-buffer line series for live data.
///
/// The window's x domain is `[0, capacity - 1]` sample indices; samples
/// arrive through `Axes::stream_push` and the oldest drop once `len` reaches
/// `capacity`, so a full window scrolls like a rolling trace.
#[derive(Clone)]
pub(super) struct StreamCmd {
	pub(super) values: Vec<f32>,
	pub(super) head: usize,
	pub(super) len: usize,
	pub(super) style: LineStyle,
}

impl StreamCmd {
	/// Append one sample, dropping the oldest when the window is full.
	fn push(&mut self, value: f32) {
		let capacity = self.values.len();
		if self.len < capacity {
			self.values[(self.head + self.len) % capacity] = value;
			self.len += 1;
		} else {
			self.values[self.head] = value;
			self.head = (self.head + 1) % capacity;
		}
	}

	/// Window contents in draw order (oldest first).
	pub(super) fn ordered(&self) -> impl Iterator<Item = f32> + '_ {
		let capacity = self.values.len();
		(0..self.len).map(move |index| self.values[(self.head + index) % capacity])
	}
}

/// One box-summary artist; each entry is one dataset's quartiles.
///
/// Quartiles are resolved eagerly at record time so replay stays free of
/// data inspection and automatic limits can read the whisker extents.
#[derive(Clone)]
pub(super) struct BoxCmd {
	/// `(slot, q1, median, q3, whisker_lo, whisker_hi)` per dataset.
	pub(super) summaries: Vec<(f32, f32, f32, f32, f32, f32)>,
	pub(super) style: BoxStyle,
}

/// Filled band between two host y curves at x = 0..n-1.
///
/// The band is the matplotlib `fill_between` / error-band primitive: replay
/// shades the region between `y_lo` and `y_hi` with `style.color`.
#[derive(Clone)]
pub(super) struct BandCmd {
	pub(super) y_lo: Vec<f32>,
	pub(super) y_hi: Vec<f32>,
	pub(super) style: LineStyle,
}

/// Data-space text annotation.
#[derive(Clone)]
pub(super) struct AnnotationCmd {
	pub(super) text: String,
	pub(super) x: f32,
	pub(super) y: f32,
}

#[derive(Clone, Debug)]
pub(super) struct LimitsCmd {
	pub(super) x_min: f32,
	pub(super) x_max: f32,
	pub(super) y_min: f32,
	pub(super) y_max: f32,
}

/// One resolved legend entry, in artist insertion order.
///
/// `row` is `usize::MAX` for unlabeled artists (no legend row) and the
/// drawn row index otherwise; `artist` is the `cmds.artists` position used
/// by `set_artist_hidden`.
pub(super) struct LegendEntry {
	pub(super) artist: usize,
	pub(super) label: String,
	pub(super) color: Color,
	pub(super) hidden: bool,
	pub(super) row: usize,
}

/// Pixels reserved at the right of a cell for a heatmap colorbar.
pub(super) const COLORBAR_RESERVE: u32 = 46;

/// Legend geometry shared between figure replay and `FigureView` hit-testing.
pub(super) const LEGEND_WIDTH: f32 = 84.0;
pub(super) const LEGEND_ROW_STEP: f32 = 13.0;
pub(super) const LEGEND_FIRST_BASELINE: f32 = 14.0;

/// Clickable rect for legend row `row` inside plot area `rect`.
///
/// Covers the swatch and text band drawn at `LEGEND_FIRST_BASELINE + row *
/// LEGEND_ROW_STEP`.
pub(super) fn legend_row_rect(rect: PixelRect, row: usize) -> PixelRect {
	PixelRect::new(
		rect.x + rect.w as i32 - LEGEND_WIDTH as i32 - 2,
		rect.y + (LEGEND_FIRST_BASELINE - 10.0 + row as f32 * LEGEND_ROW_STEP) as i32,
		LEGEND_WIDTH as u32 + 2,
		12,
	)
}

// ── Axes ──────────────────────────────────────────────────────────────────────

/// One subplot inside an `oa::plot::Figure`.
#[derive(Clone)]
pub struct Axes {
	pub(super) cmds: AxesCommands,
	/// Artist index kept at full strength while the rest render dimmed —
	/// `FigureView` sets this from legend hover.
	pub(super) highlight: Option<usize>,
}

impl Default for Axes {
	fn default() -> Self {
		Self {
			cmds: AxesCommands::new(),
			highlight: None,
		}
	}
}

impl Axes {
	/// Set the axes title rendered above its plot region.
	///
	/// `color` with alpha zero (the default) uses the theme text color.
	pub fn title(&mut self, text: &str) {
		self.title_color(text, Color::new(0.0, 0.0, 0.0, 0.0));
	}

	/// Set the axes title with an explicit text color.
	pub fn title_color(&mut self, text: &str, color: Color) {
		self.cmds.title = text.to_owned();
		self.cmds.title_color = color;
	}

	/// Set the secondary caption line below the title.
	///
	/// Used by the image-classify tutorial to show the ground-truth label.
	/// Alpha-zero color uses the theme secondary text color.
	pub fn caption(&mut self, text: &str) {
		self.cmds.caption = text.to_owned();
		self.cmds.caption_color = Color::new(0.0, 0.0, 0.0, 0.0);
	}

	/// Set caption with an explicit text color.
	pub fn caption_color(&mut self, text: &str, color: Color) {
		self.cmds.caption = text.to_owned();
		self.cmds.caption_color = color;
	}

	/// Set the horizontal axis label.
	pub fn x_label(&mut self, text: &str) {
		self.cmds.x_label = text.to_owned();
	}

	/// Set the vertical axis label. The first GPU text slice renders it at the
	/// upper-left edge; rotated vertical layout follows the text transform pass.
	pub fn y_label(&mut self, text: &str) {
		self.cmds.y_label = text.to_owned();
	}

	/// Display a texture as a raster base for this axes.
	pub fn imshow(&mut self, texture: crate::Texture) {
		self.cmds.imshow = Some(ImshowCmd { texture });
	}

	/// Dense row-major heatmap.  Donor: `oa::plot::Axes::heatmap`.
	pub fn heatmap(&mut self, values: &[f32], rows: i32, cols: i32, style: HeatmapStyle) {
		let Some(count) = usize::try_from(rows).ok().and_then(|rows| {
			usize::try_from(cols)
				.ok()
				.and_then(|cols| rows.checked_mul(cols))
		}) else {
			return;
		};
		if count == 0 || values.len() < count {
			return;
		}
		self.cmds.heatmap = Some(Heatmap {
			values: values[..count].to_vec(),
			rows,
			cols,
			style,
		});
	}

	/// Dense device-resident heatmap from a rank-2 `F32` `oa::Matrix`.
	///
	/// The matrix stays on the GPU; replay lowers it to one `draw_heatmap`
	/// dispatch through the UI compositor. `style.auto_scale` cannot inspect
	/// device data — callers must supply an explicit `v_min`/`v_max` range or
	/// replay fails validation.
	pub fn heatmap_matrix(&mut self, values: crate::Matrix, style: HeatmapStyle) {
		if values.dtype() != crate::DType::F32 || values.shape().len() != 2 {
			return;
		}
		self.cmds.heatmap_matrix = Some(HeatmapMatrixCmd { values, style });
	}

	/// Line plot: y-values vs index.  Repeated calls append ordered series.
	pub fn plot(&mut self, y: &[f32], style: LineStyle) {
		if y.is_empty() {
			return;
		}
		let x: Vec<f32> = (0..y.len()).map(|i| i as f32).collect();
		self.cmds.lines.push(LineCmd {
			x,
			y: y.to_vec(),
			style,
		});
		self.cmds.artists.push(ArtistKind::Line);
	}

	/// Explicit X/Y line plot.
	pub fn plot_xy(&mut self, x: &[f32], y: &[f32], style: LineStyle) {
		if x.is_empty() || x.len() != y.len() {
			return;
		}
		self.cmds.lines.push(LineCmd {
			x: x.to_vec(),
			y: y.to_vec(),
			style,
		});
		self.cmds.artists.push(ArtistKind::Line);
	}

	/// Device-resident explicit-XY line series from an `F32` `[2, N]`
	/// `oa::Matrix` laid out as one X block followed by one Y block.
	///
	/// The matrix stays on the GPU and is retained until composition
	/// completes. Device data cannot be inspected host-side: the series does
	/// not contribute to automatic limits (set `limits` explicitly or record
	/// a host artist as well) and requires linear axes — a logarithmic axis
	/// fails at replay rather than silently drawing untransformed data.
	pub fn plot_matrix(&mut self, values: &crate::Matrix, style: LineStyle) {
		let shape = values.shape();
		if values.dtype() != crate::DType::F32 || shape.len() != 2 || shape[0] != 2 || shape[1] == 0 {
			return;
		}
		self.cmds.line_matrices.push(LineMatrixCmd {
			values: values.clone(),
			xy: true,
			style,
		});
		self.cmds.artists.push(ArtistKind::LineMatrix);
	}

	/// Device-resident y-vs-index line series from an `F32` `[N]` `oa::Matrix`.
	///
	/// Same device-resident contract as [`Axes::plot_matrix`]; x positions
	/// are the sample indices `0..N-1`, which do contribute to automatic
	/// x limits.
	pub fn plot_matrix_y(&mut self, values: &crate::Matrix, style: LineStyle) {
		if values.dtype() != crate::DType::F32 || values.shape().len() != 1 || values.shape()[0] == 0 {
			return;
		}
		self.cmds.line_matrices.push(LineMatrixCmd {
			values: values.clone(),
			xy: false,
			style,
		});
		self.cmds.artists.push(ArtistKind::LineMatrix);
	}

	/// Data-space text annotation anchored at `(x, y)`.
	///
	/// The text renders just above the data point; non-finite or
	/// scale-dropped points are skipped at replay.
	pub fn annotate(&mut self, text: &str, x: f32, y: f32) {
		if text.is_empty() {
			return;
		}
		self.cmds.annotations.push(AnnotationCmd {
			text: text.to_owned(),
			x,
			y,
		});
	}

	/// Category labels replacing numeric x tick labels at bar positions.
	///
	/// Pass one label per bar slot; an empty list restores numeric ticks.
	pub fn categories(&mut self, labels: &[&str]) {
		self.cmds.categories = labels.iter().map(|label| (*label).to_owned()).collect();
	}

	/// Set the x axis scaling (default: linear).
	pub fn xscale(&mut self, scale: Scale) {
		self.cmds.x_scale = scale;
	}

	/// Set the y axis scaling (default: linear).
	pub fn yscale(&mut self, scale: Scale) {
		self.cmds.y_scale = scale;
	}

	/// Explicit x tick positions in data space.
	///
	/// Positions map through the x scale at replay — a `Log10` axis drops
	/// non-positive values — and share grid placement with the labels.
	/// An empty slice restores the automatic nice-number locator.
	pub fn xticks(&mut self, values: &[f32]) {
		self.cmds.x_ticks = if values.is_empty() {
			None
		} else {
			Some(values.to_vec())
		};
	}

	/// Explicit y tick positions in data space; see [`Axes::xticks`].
	pub fn yticks(&mut self, values: &[f32]) {
		self.cmds.y_ticks = if values.is_empty() {
			None
		} else {
			Some(values.to_vec())
		};
	}

	/// Label presentation for explicit [`Axes::xticks`] positions.
	pub fn xformat(&mut self, format: TickFormat) {
		self.cmds.x_tick_format = format;
	}

	/// Label presentation for explicit [`Axes::yticks`] positions.
	pub fn yformat(&mut self, format: TickFormat) {
		self.cmds.y_tick_format = format;
	}

	/// Point cloud scatter.
	pub fn scatter(&mut self, x: &[f32], y: &[f32], style: ScatterStyle) {
		if x.is_empty() || x.len() != y.len() {
			return;
		}
		self.cmds.scatters.push(ScatterCmd {
			x: x.to_vec(),
			y: y.to_vec(),
			style,
		});
		self.cmds.artists.push(ArtistKind::Scatter);
	}

	/// Vertical bar chart at x = 0..n-1.
	pub fn bar(&mut self, y: &[f32], style: BarStyle) {
		if y.is_empty() {
			return;
		}
		self.cmds.bars.push(BarCmd {
			y: y.to_vec(),
			group: None,
			stack_base: None,
			style,
		});
		self.cmds.artists.push(ArtistKind::Bar);
	}

	/// Grouped bar chart — `series[i]` shares each x slot side-by-side.
	///
	/// `styles` pairs per layer; a short list falls back to defaults (palette
	/// colors). Each layer is its own artist, so legend rows and visibility
	/// toggling work per series.
	pub fn bar_grouped(&mut self, series: &[&[f32]], styles: &[BarStyle]) {
		let size = series.len();
		if size == 0 || series.iter().any(|layer| layer.is_empty()) {
			return;
		}
		for (index, layer) in series.iter().enumerate() {
			self.cmds.bars.push(BarCmd {
				y: (*layer).to_vec(),
				group: Some((index, size)),
				stack_base: None,
				style: styles.get(index).cloned().unwrap_or_default(),
			});
			self.cmds.artists.push(ArtistKind::Bar);
		}
	}

	/// Stacked bar chart — `series[i]` accumulates on top of the layers
	/// before it.
	///
	/// Style and artist semantics match [`Axes::bar_grouped`]; each layer's
	/// baseline is the running sum of the previous layers' values.
	pub fn bar_stacked(&mut self, series: &[&[f32]], styles: &[BarStyle]) {
		if series.is_empty() || series.iter().any(|layer| layer.is_empty()) {
			return;
		}
		let count = series[0].len();
		if series.iter().any(|layer| layer.len() != count) {
			return;
		}
		let mut base = vec![0.0_f32; count];
		for (index, layer) in series.iter().enumerate() {
			self.cmds.bars.push(BarCmd {
				y: (*layer).to_vec(),
				group: None,
				stack_base: Some(base.clone()),
				style: styles.get(index).cloned().unwrap_or_default(),
			});
			self.cmds.artists.push(ArtistKind::Bar);
			for (acc, &value) in base.iter_mut().zip(layer.iter()) {
				*acc += if value.is_finite() { value } else { 0.0 };
			}
		}
	}

	/// Begin a rolling line series of fixed `capacity` samples.
	///
	/// Returns the series handle for [`Axes::stream_push`]. The window's x
	/// domain is the sample indices `0..capacity-1`, so the trace fills from
	/// the left and then scrolls as the oldest samples drop.
	pub fn stream(&mut self, capacity: usize, style: LineStyle) -> usize {
		let index = self.cmds.streams.len();
		self.cmds.streams.push(StreamCmd {
			values: vec![0.0; capacity.max(1)],
			head: 0,
			len: 0,
			style,
		});
		self.cmds.artists.push(ArtistKind::Stream);
		index
	}

	/// Append samples to a [`Axes::stream`] series.
	///
	/// Out-of-range handles are ignored, matching the silent-skip contract of
	/// the other recording methods.
	pub fn stream_push(&mut self, series: usize, values: &[f32]) {
		if let Some(stream) = self.cmds.streams.get_mut(series) {
			for &value in values {
				stream.push(value);
			}
		}
	}

	/// Fill the band between two equal-length y curves at x = 0..n-1.
	///
	/// The error-band / `fill_between` primitive: `style.color` (alpha
	/// included) tints the shaded region. Band x positions are the sample
	/// indices, so a non-linear x axis is rejected at replay.
	pub fn band(&mut self, y_lo: &[f32], y_hi: &[f32], style: LineStyle) {
		if y_lo.is_empty() || y_lo.len() != y_hi.len() {
			return;
		}
		self.cmds.bands.push(BandCmd {
			y_lo: y_lo.to_vec(),
			y_hi: y_hi.to_vec(),
			style,
		});
		self.cmds.artists.push(ArtistKind::Band);
	}

	/// Box-and-whisker summary over one or more datasets.
	///
	/// Each dataset becomes one box centered on its own x slot, palette-colored
	/// in order. Quartiles use linear interpolation (matplotlib default) and
	/// whiskers extend `style.whisk` × IQR, clamped to the data extremes.
	pub fn boxplot(&mut self, datasets: &[&[f32]], style: BoxStyle) {
		if datasets.is_empty() {
			return;
		}
		let mut summaries = Vec::with_capacity(datasets.len());
		for (index, dataset) in datasets.iter().enumerate() {
			let mut sorted: Vec<f32> = dataset.iter().copied().filter(|v| v.is_finite()).collect();
			if sorted.is_empty() {
				continue;
			}
			sorted.sort_by(f32::total_cmp);
			let q1 = percentile(&sorted, 0.25);
			let median = percentile(&sorted, 0.5);
			let q3 = percentile(&sorted, 0.75);
			let iqr = (q3 - q1) * style.whisk;
			let lo = sorted
				.iter()
				.copied()
				.find(|&v| v >= q1 - iqr)
				.unwrap_or(sorted[0]);
			let hi = sorted
				.iter()
				.copied()
				.rev()
				.find(|&v| v <= q3 + iqr)
				.unwrap_or(sorted[sorted.len() - 1]);
			summaries.push((index as f32, q1, median, q3, lo, hi));
		}
		if summaries.is_empty() {
			return;
		}
		self.cmds.boxes.push(BoxCmd { summaries, style });
		self.cmds.artists.push(ArtistKind::BoxPlot);
	}

	/// Histogram: bin `values` into `bins` bars.
	pub fn histogram(&mut self, values: &[f32], bins: i32, style: BarStyle) {
		if bins <= 0 || values.is_empty() {
			return;
		}
		let bins = (bins as usize).min(4096);
		let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
		if finite.is_empty() {
			return;
		}
		let v_min = finite.iter().cloned().fold(f32::INFINITY, f32::min);
		let v_max = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
		let range = (v_max - v_min).max(1e-8);
		let mut counts = vec![0.0f32; bins];
		for &v in &finite {
			let idx = (((v - v_min) / range) * bins as f32) as usize;
			counts[idx.min(bins - 1)] += 1.0;
		}
		self.bar(&counts, style);
	}

	/// Fixed data limits; degenerate input restores auto-limits.
	pub fn limits(&mut self, x_min: f32, x_max: f32, y_min: f32, y_max: f32) {
		if [x_min, x_max, y_min, y_max]
			.iter()
			.all(|value| value.is_finite())
			&& x_min < x_max
			&& y_min < y_max
		{
			self.cmds.limits = Some(LimitsCmd {
				x_min,
				x_max,
				y_min,
				y_max,
			});
		} else {
			self.auto_limits();
		}
	}

	/// Restore automatic data-driven limits.
	pub fn auto_limits(&mut self) {
		self.cmds.limits = None;
	}

	/// Toggle the background grid (default: on).
	pub fn grid(&mut self, visible: bool) {
		self.cmds.show_grid = visible;
	}

	/// Toggle the labeled-series legend (default: on).
	pub fn legend(&mut self, visible: bool) {
		self.cmds.show_legend = visible;
	}

	/// Border color around the axes rect (default: none).
	pub fn border_color(&mut self, color: Color) {
		self.cmds.border_color = Some(color);
	}

	/// Clear all recorded commands.
	pub fn clear(&mut self) {
		self.cmds = AxesCommands::new();
	}

	/// Number of recorded artists in insertion order.
	pub fn artist_count(&self) -> usize {
		self.cmds.artists.len()
	}

	/// Visibility flag of the `index`-th artist in insertion order.
	pub fn artist_hidden(&self, index: usize) -> Option<bool> {
		let (mut line, mut scatter, mut bar, mut matrix, mut stream, mut band, mut bx) =
			(0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
		for (position, kind) in self.cmds.artists.iter().enumerate() {
			let hidden = match kind {
				ArtistKind::Line => self.cmds.lines.get(line).map(|cmd| cmd.style.hidden),
				ArtistKind::Scatter => self.cmds.scatters.get(scatter).map(|cmd| cmd.style.hidden),
				ArtistKind::Bar => self.cmds.bars.get(bar).map(|cmd| cmd.style.hidden),
				ArtistKind::LineMatrix => self
					.cmds
					.line_matrices
					.get(matrix)
					.map(|cmd| cmd.style.hidden),
				ArtistKind::Stream => self.cmds.streams.get(stream).map(|cmd| cmd.style.hidden),
				ArtistKind::Band => self.cmds.bands.get(band).map(|cmd| cmd.style.hidden),
				ArtistKind::BoxPlot => self.cmds.boxes.get(bx).map(|cmd| cmd.style.hidden),
			};
			if position == index {
				return hidden;
			}
			match kind {
				ArtistKind::Line => line += 1,
				ArtistKind::Scatter => scatter += 1,
				ArtistKind::Bar => bar += 1,
				ArtistKind::LineMatrix => matrix += 1,
				ArtistKind::Stream => stream += 1,
				ArtistKind::Band => band += 1,
				ArtistKind::BoxPlot => bx += 1,
			}
		}
		None
	}

	/// Show or hide the `index`-th artist in insertion order.
	///
	/// Hidden artists keep their palette slot and legend entry but skip
	/// rendering; `FigureView` legend toggling writes this flag.
	pub fn set_artist_hidden(&mut self, index: usize, hidden: bool) {
		let (mut line, mut scatter, mut bar, mut matrix, mut stream, mut band, mut bx) =
			(0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
		for (position, kind) in self.cmds.artists.iter().enumerate() {
			if position == index {
				match kind {
					ArtistKind::Line => {
						if let Some(cmd) = self.cmds.lines.get_mut(line) {
							cmd.style.hidden = hidden;
						}
					}
					ArtistKind::Scatter => {
						if let Some(cmd) = self.cmds.scatters.get_mut(scatter) {
							cmd.style.hidden = hidden;
						}
					}
					ArtistKind::Bar => {
						if let Some(cmd) = self.cmds.bars.get_mut(bar) {
							cmd.style.hidden = hidden;
						}
					}
					ArtistKind::LineMatrix => {
						if let Some(cmd) = self.cmds.line_matrices.get_mut(matrix) {
							cmd.style.hidden = hidden;
						}
					}
					ArtistKind::Stream => {
						if let Some(cmd) = self.cmds.streams.get_mut(stream) {
							cmd.style.hidden = hidden;
						}
					}
					ArtistKind::Band => {
						if let Some(cmd) = self.cmds.bands.get_mut(band) {
							cmd.style.hidden = hidden;
						}
					}
					ArtistKind::BoxPlot => {
						if let Some(cmd) = self.cmds.boxes.get_mut(bx) {
							cmd.style.hidden = hidden;
						}
					}
				}
				return;
			}
			match kind {
				ArtistKind::Line => line += 1,
				ArtistKind::Scatter => scatter += 1,
				ArtistKind::Bar => bar += 1,
				ArtistKind::LineMatrix => matrix += 1,
				ArtistKind::Stream => stream += 1,
				ArtistKind::Band => band += 1,
				ArtistKind::BoxPlot => bx += 1,
			}
		}
	}

	/// Highlight the `index`-th artist in insertion order, dimming the rest.
	///
	/// `None` restores uniform emphasis; `FigureView` legend hover writes this.
	pub fn set_series_highlight(&mut self, index: Option<usize>) {
		self.highlight = index;
	}

	// ── Shared layout / legend helpers ──────────────────────────────────────

	/// Plot region inside `cell` after the colorbar reservation.
	///
	/// Figure replay and `FigureView` hit-testing share this helper so
	/// pixel↔data mapping always agrees with the drawn artists.
	pub(super) fn plot_area(&self, cell: PixelRect) -> PixelRect {
		let colorbar = self
			.cmds
			.heatmap
			.as_ref()
			.is_some_and(|heatmap| heatmap.style.colorbar)
			|| self
				.cmds
				.heatmap_matrix
				.as_ref()
				.is_some_and(|heatmap| heatmap.style.colorbar);
		if !colorbar {
			return cell;
		}
		PixelRect {
			w: cell.w.saturating_sub(COLORBAR_RESERVE),
			..cell
		}
	}

	/// Labeled-series legend entries in insertion order.
	///
	/// The palette counter advances for every alpha-zero artist — hidden and
	/// unlabeled included — so colors stay stable across legend toggles.
	/// `row` numbers only the labeled entries, matching the drawn rows.
	pub(super) fn legend_entries(&self, palette: &[crate::Color]) -> Vec<LegendEntry> {
		let mut entries = Vec::new();
		let (mut line, mut scatter, mut bar, mut matrix, mut stream, mut band, mut bx) =
			(0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
		let mut palette_idx = 0usize;
		for (position, kind) in self.cmds.artists.iter().enumerate() {
			let (label, color, hidden) = match kind {
				ArtistKind::Line => {
					let Some(cmd) = self.cmds.lines.get(line) else {
						continue;
					};
					line += 1;
					(&cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
				ArtistKind::Scatter => {
					let Some(cmd) = self.cmds.scatters.get(scatter) else {
						continue;
					};
					scatter += 1;
					(&cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
				ArtistKind::Bar => {
					let Some(cmd) = self.cmds.bars.get(bar) else {
						continue;
					};
					bar += 1;
					(&cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
				ArtistKind::LineMatrix => {
					let Some(cmd) = self.cmds.line_matrices.get(matrix) else {
						continue;
					};
					matrix += 1;
					(&cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
				ArtistKind::Stream => {
					let Some(cmd) = self.cmds.streams.get(stream) else {
						continue;
					};
					stream += 1;
					(&cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
				ArtistKind::Band => {
					let Some(cmd) = self.cmds.bands.get(band) else {
						continue;
					};
					band += 1;
					(&cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
				ArtistKind::BoxPlot => {
					let Some(cmd) = self.cmds.boxes.get(bx) else {
						continue;
					};
					bx += 1;
					(&cmd.style.label, cmd.style.color, cmd.style.hidden)
				}
			};
			let color = if color.a == 0.0 {
				let resolved = palette[palette_idx % palette.len()];
				palette_idx += 1;
				resolved
			} else {
				color
			};
			entries.push(LegendEntry {
				artist: position,
				label: label.clone(),
				color,
				hidden,
				row: usize::MAX,
			});
		}
		let mut row = 0usize;
		for entry in &mut entries {
			if !entry.label.is_empty() {
				entry.row = row;
				row += 1;
			}
		}
		entries
	}

	// ── Computed limits ───────────────────────────────────────────────────────

	/// Compute the data-driven Y range from all recorded artists.
	pub(super) fn data_y_range(&self) -> (f32, f32) {
		let mut y_min = f32::INFINITY;
		let mut y_max = f32::NEG_INFINITY;
		for line in &self.cmds.lines {
			for &v in &line.y {
				if v.is_finite() {
					y_min = y_min.min(v);
					y_max = y_max.max(v);
				}
			}
		}
		for scatter in &self.cmds.scatters {
			for &value in &scatter.y {
				if value.is_finite() {
					y_min = y_min.min(value);
					y_max = y_max.max(value);
				}
			}
		}
		for bar in &self.cmds.bars {
			for (index, &v) in bar.y.iter().enumerate() {
				if v.is_finite() {
					let base = bar.stack_base.as_ref().map_or(0.0, |bases| bases[index]);
					y_min = y_min.min((base + v).min(base.min(0.0)));
					y_max = y_max.max((base + v).max(base.max(0.0)));
				}
			}
		}
		for stream in &self.cmds.streams {
			for value in stream.ordered() {
				if value.is_finite() {
					y_min = y_min.min(value);
					y_max = y_max.max(value);
				}
			}
		}
		for band in &self.cmds.bands {
			for &value in band.y_lo.iter().chain(&band.y_hi) {
				if value.is_finite() {
					y_min = y_min.min(value);
					y_max = y_max.max(value);
				}
			}
		}
		for boxes in &self.cmds.boxes {
			for &(_, _, _, _, lo, hi) in &boxes.summaries {
				y_min = y_min.min(lo);
				y_max = y_max.max(hi);
			}
		}
		if !y_min.is_finite() || !y_max.is_finite() {
			return (0.0, 1.0);
		}
		if (y_max - y_min).abs() < 1e-6 {
			return (y_min - 1.0, y_max + 1.0);
		}
		(y_min, y_max)
	}

	/// Effective data limits `[x_min, x_max, y_min, y_max]` after explicit
	/// `limits` or automatic data-driven resolution. Interactive views use
	/// this to map pixel coordinates into data space.
	pub fn data_limits(&self) -> [f32; 4] {
		let limits = self.effective_limits();
		[limits.x_min, limits.x_max, limits.y_min, limits.y_max]
	}

	/// The explicit `limits` override, or `None` when limits are data-driven.
	pub fn explicit_limits(&self) -> Option<[f32; 4]> {
		self
			.cmds
			.limits
			.as_ref()
			.map(|l| [l.x_min, l.x_max, l.y_min, l.y_max])
	}

	/// Compute the effective data limits (explicit or automatic).
	pub(super) fn effective_limits(&self) -> LimitsCmd {
		if let Some(lim) = &self.cmds.limits {
			return lim.clone();
		}
		let (y_min, y_max) = self.data_y_range();
		let mut x_min = f32::INFINITY;
		let mut x_max = f32::NEG_INFINITY;
		for value in self
			.cmds
			.lines
			.iter()
			.flat_map(|line| line.x.iter())
			.chain(
				self
					.cmds
					.scatters
					.iter()
					.flat_map(|scatter| scatter.x.iter()),
			)
			.copied()
			.filter(|value| value.is_finite())
		{
			x_min = x_min.min(value);
			x_max = x_max.max(value);
		}
		for bar in &self.cmds.bars {
			x_min = x_min.min(0.0);
			x_max = x_max.max(bar.y.len().saturating_sub(1) as f32);
		}
		for matrix in &self.cmds.line_matrices {
			if !matrix.xy {
				let count = matrix.values.shape().first().copied().unwrap_or(0);
				x_min = x_min.min(0.0);
				x_max = x_max.max(count.saturating_sub(1) as f32);
			}
		}
		for stream in &self.cmds.streams {
			if stream.len == 0 {
				continue;
			}
			x_min = x_min.min(0.0);
			x_max = x_max.max(stream.values.len().saturating_sub(1) as f32);
		}
		for band in &self.cmds.bands {
			x_min = x_min.min(0.0);
			x_max = x_max.max(band.y_lo.len().saturating_sub(1) as f32);
		}
		for boxes in &self.cmds.boxes {
			x_min = x_min.min(0.0);
			x_max = x_max.max(boxes.summaries.len().saturating_sub(1) as f32);
		}
		if !x_min.is_finite() || !x_max.is_finite() || (x_max - x_min).abs() < 1e-6 {
			x_min = 0.0;
			x_max = 1.0;
		}
		LimitsCmd {
			x_min,
			x_max,
			y_min,
			y_max,
		}
	}

	/// Effective limits transformed into display space by the axis scales.
	///
	/// Under linear scaling this equals `data_limits`; under `Log10` the
	/// ends are `log10` of the positive range (see [`scaled_axis`]).
	pub(super) fn scaled_limits(&self) -> LimitsCmd {
		let raw = self.effective_limits();
		let (x_min, x_max) = scaled_axis(
			raw.x_min,
			raw.x_max,
			self.cmds.x_scale,
			self.min_positive_x(),
		);
		let (y_min, y_max) = scaled_axis(
			raw.y_min,
			raw.y_max,
			self.cmds.y_scale,
			self.min_positive_y(),
		);
		LimitsCmd {
			x_min,
			x_max,
			y_min,
			y_max,
		}
	}

	/// Smallest positive x value across host artists, for log limit floors.
	fn min_positive_x(&self) -> Option<f32> {
		let mut min = f32::INFINITY;
		for value in self.cmds.lines.iter().flat_map(|line| line.x.iter()).chain(
			self
				.cmds
				.scatters
				.iter()
				.flat_map(|scatter| scatter.x.iter()),
		) {
			if value.is_finite() && *value > 0.0 {
				min = min.min(*value);
			}
		}
		for bars in &self.cmds.bars {
			if bars.y.len() > 1 {
				min = min.min(1.0);
			}
		}
		min.is_finite().then_some(min)
	}

	/// Smallest positive y value across host artists, for log limit floors.
	fn min_positive_y(&self) -> Option<f32> {
		let mut min = f32::INFINITY;
		for value in self
			.cmds
			.lines
			.iter()
			.flat_map(|line| line.y.iter())
			.chain(
				self
					.cmds
					.scatters
					.iter()
					.flat_map(|scatter| scatter.y.iter()),
			)
			.chain(self.cmds.bars.iter().flat_map(|bars| bars.y.iter()))
		{
			if value.is_finite() && *value > 0.0 {
				min = min.min(*value);
			}
		}
		min.is_finite().then_some(min)
	}
}

/// Linear-interpolation percentile over sorted values (matplotlib default).
fn percentile(sorted: &[f32], q: f32) -> f32 {
	let pos = q * (sorted.len() - 1) as f32;
	let lo = pos.floor() as usize;
	let hi = (lo + 1).min(sorted.len() - 1);
	sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f32)
}
