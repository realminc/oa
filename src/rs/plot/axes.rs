//! `oa::plot::Axes` — one subplot inside an `oa::plot::Figure`.
//!
//! Records one optional raster base plus ordered vector artists.
//! The parent Figure lowers the same commands through `Ui`'s GPU compositor
//! for both live (`show`) and terminal (`save_to` / `render`) sinks.
//!
//! Donor reference: `oa/ui/plot/axes.h`, `oa/ui/plot/axes.cpp`

use crate::Color;

// ── Style descriptors ─────────────────────────────────────────────────────────

/// Per-series line style.
#[derive(Clone, Debug)]
pub struct LineStyle {
	/// Alpha-zero selects the next deterministic theme palette color.
	pub color: Color,
	pub label: String,
	pub width: f32,
	pub antialias_samples: u32,
}

impl Default for LineStyle {
	fn default() -> Self {
		Self {
			color: Color::new(0.0, 0.0, 0.0, 0.0), // palette auto
			label: String::new(),
			width: 1.35,
			antialias_samples: 4,
		}
	}
}

/// Per-series scatter style.
#[derive(Clone, Debug)]
pub struct ScatterStyle {
	pub color: Color,
	pub label: String,
	pub radius: f32,
}

impl Default for ScatterStyle {
	fn default() -> Self {
		Self {
			color: Color::new(0.0, 0.0, 0.0, 0.0),
			label: String::new(),
			radius: 3.0,
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
}

impl Default for BarStyle {
	fn default() -> Self {
		Self {
			color: Color::new(0.0, 0.0, 0.0, 0.0),
			label: String::new(),
			gap: 0.18,
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
	/// Colormap index: 0 = inferno, 1 = viridis (default), 2 = greens.
	pub colormap: u32,
	/// Scale `v_min`/`v_max` automatically from data.
	pub auto_scale: bool,
	/// Draw a thin grid between cells.
	pub show_grid: bool,
}

impl Default for HeatmapStyle {
	fn default() -> Self {
		Self {
			v_min: 0.0,
			v_max: 1.0,
			colormap: 1, // viridis
			auto_scale: true,
			show_grid: true,
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
	pub(super) heatmap: Option<HeatmapCmd>,
	pub(super) lines: Vec<LineCmd>,
	pub(super) scatters: Vec<ScatterCmd>,
	pub(super) bars: Vec<BarCmd>,
	/// Insertion-order list of artists — used by Figure to replay in call order.
	pub(super) artists: Vec<ArtistKind>,
	pub(super) limits: Option<LimitsCmd>,
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
pub(super) struct HeatmapCmd {
	pub(super) values: Vec<f32>,
	pub(super) rows: i32,
	pub(super) cols: i32,
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
	pub(super) style: BarStyle,
}

#[derive(Clone, Debug)]
pub(super) struct LimitsCmd {
	pub(super) x_min: f32,
	pub(super) x_max: f32,
	pub(super) y_min: f32,
	pub(super) y_max: f32,
}

// ── Axes ──────────────────────────────────────────────────────────────────────

/// One subplot inside an `oa::plot::Figure`.
#[derive(Clone)]
pub struct Axes {
	pub(super) cmds: AxesCommands,
}

impl Default for Axes {
	fn default() -> Self {
		Self {
			cmds: AxesCommands::new(),
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
		self.cmds.heatmap = Some(HeatmapCmd {
			values: values[..count].to_vec(),
			rows,
			cols,
			style,
		});
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
			style,
		});
		self.cmds.artists.push(ArtistKind::Bar);
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
			for &v in &bar.y {
				if v.is_finite() {
					y_min = y_min.min(v.min(0.0));
					y_max = y_max.max(v.max(0.0));
				}
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
}
