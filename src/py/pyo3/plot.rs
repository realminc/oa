//! Python bindings for the retained `oa::plot` figure and axes API.

use std::{cell::RefCell, rc::Rc};

use pyo3::prelude::*;

use crate::{
	error::python_error, image::PythonImage, matrix::PythonMatrix, runtime::PythonRuntime,
};

type Rgba = (f32, f32, f32, f32);

fn color(value: Rgba) -> oa::Color {
	oa::Color::new(value.0, value.1, value.2, value.3)
}

fn rgba(value: oa::Color) -> Rgba {
	(value.r, value.g, value.b, value.a)
}

/// Resolve the optional `decimals`/`labels` pair into a `TickFormat`.
fn tick_format(decimals: Option<usize>, labels: Option<Vec<String>>) -> oa::plot::TickFormat {
	match (decimals, labels) {
		(_, Some(labels)) => oa::plot::TickFormat::Labels(labels),
		(Some(decimals), None) => oa::plot::TickFormat::Decimals(decimals),
		(None, None) => oa::plot::TickFormat::Auto,
	}
}

#[pyclass(name = "PlotTheme", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonPlotTheme {
	Dark = 0,
	Light = 1,
}

impl From<PythonPlotTheme> for oa::plot::Theme {
	fn from(value: PythonPlotTheme) -> Self {
		match value {
			PythonPlotTheme::Dark => Self::Dark,
			PythonPlotTheme::Light => Self::Light,
		}
	}
}

impl From<oa::plot::Theme> for PythonPlotTheme {
	fn from(value: oa::plot::Theme) -> Self {
		match value {
			oa::plot::Theme::Dark => Self::Dark,
			oa::plot::Theme::Light => Self::Light,
		}
	}
}

#[pyclass(name = "PlotScale", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonPlotScale {
	Linear = 0,
	Log10 = 1,
	Symlog = 2,
}

impl From<PythonPlotScale> for oa::plot::Scale {
	fn from(value: PythonPlotScale) -> Self {
		match value {
			PythonPlotScale::Linear => Self::Linear,
			PythonPlotScale::Log10 => Self::Log10,
			PythonPlotScale::Symlog => Self::Symlog,
		}
	}
}

#[pyclass(name = "LineStyle")]
#[derive(Clone)]
pub(crate) struct PythonLineStyle {
	inner: oa::plot::LineStyle,
}

#[pymethods]
impl PythonLineStyle {
	#[new]
	#[pyo3(signature = (color=(0.0, 0.0, 0.0, 0.0), label=String::new(), width=1.35, antialias_samples=4, step=false, hidden=false))]
	#[allow(clippy::too_many_arguments)]
	fn new(
		color: Rgba,
		label: String,
		width: f32,
		antialias_samples: u32,
		step: bool,
		hidden: bool,
	) -> Self {
		Self {
			inner: oa::plot::LineStyle {
				color: self::color(color),
				label,
				width,
				antialias_samples,
				step,
				hidden,
			},
		}
	}
}

#[pyclass(name = "ScatterStyle")]
#[derive(Clone)]
pub(crate) struct PythonScatterStyle {
	inner: oa::plot::ScatterStyle,
}

#[pymethods]
impl PythonScatterStyle {
	#[new]
	#[pyo3(signature = (color=(0.0, 0.0, 0.0, 0.0), label=String::new(), radius=3.0, hidden=false))]
	fn new(color: Rgba, label: String, radius: f32, hidden: bool) -> Self {
		Self {
			inner: oa::plot::ScatterStyle {
				color: self::color(color),
				label,
				radius,
				hidden,
			},
		}
	}
}

#[pyclass(name = "BarStyle")]
#[derive(Clone)]
pub(crate) struct PythonBarStyle {
	inner: oa::plot::BarStyle,
}

#[pymethods]
impl PythonBarStyle {
	#[new]
	#[pyo3(signature = (color=(0.0, 0.0, 0.0, 0.0), label=String::new(), gap=0.18, hidden=false))]
	fn new(color: Rgba, label: String, gap: f32, hidden: bool) -> Self {
		Self {
			inner: oa::plot::BarStyle {
				color: self::color(color),
				label,
				gap,
				hidden,
			},
		}
	}
}

#[pyclass(name = "BoxStyle")]
#[derive(Clone)]
pub(crate) struct PythonBoxStyle {
	inner: oa::plot::BoxStyle,
}

#[pymethods]
impl PythonBoxStyle {
	#[new]
	#[pyo3(signature = (color=(0.0, 0.0, 0.0, 0.0), label=String::new(), width=0.5, whisk=1.5, hidden=false))]
	fn new(color: Rgba, label: String, width: f32, whisk: f32, hidden: bool) -> Self {
		Self {
			inner: oa::plot::BoxStyle {
				color: self::color(color),
				label,
				width,
				whisk,
				hidden,
			},
		}
	}
}

#[pyclass(name = "HeatmapStyle")]
#[derive(Clone)]
pub(crate) struct PythonHeatmapStyle {
	inner: oa::plot::HeatmapStyle,
}

#[pymethods]
impl PythonHeatmapStyle {
	#[new]
	#[pyo3(signature = (v_min=0.0, v_max=1.0, colormap=1, auto_scale=true, show_grid=true, colorbar=false))]
	#[allow(clippy::too_many_arguments)]
	fn new(
		v_min: f32,
		v_max: f32,
		colormap: u32,
		auto_scale: bool,
		show_grid: bool,
		colorbar: bool,
	) -> Self {
		Self {
			inner: oa::plot::HeatmapStyle {
				v_min,
				v_max,
				colormap,
				auto_scale,
				show_grid,
				colorbar,
			},
		}
	}
}

#[pyclass(name = "FigureConfig")]
#[derive(Clone)]
pub(crate) struct PythonFigureConfig {
	pub(crate) inner: oa::plot::FigureConfig,
}

#[pymethods]
impl PythonFigureConfig {
	#[new]
	#[pyo3(signature = (title="oa::plot".to_owned(), rows=1, cols=1, width=800, height=600, h_spacing=20, v_spacing=24, padding=20, theme=PythonPlotTheme::Dark, background=(0.0, 0.0, 0.0, 0.0)))]
	#[allow(clippy::too_many_arguments)]
	fn new(
		title: String,
		rows: i32,
		cols: i32,
		width: u32,
		height: u32,
		h_spacing: i32,
		v_spacing: i32,
		padding: i32,
		theme: PythonPlotTheme,
		background: Rgba,
	) -> Self {
		Self {
			inner: oa::plot::FigureConfig {
				title,
				rows,
				cols,
				width,
				height,
				h_spacing,
				v_spacing,
				padding,
				theme: theme.into(),
				background: color(background),
			},
		}
	}

	#[getter]
	fn rows(&self) -> i32 {
		self.inner.rows
	}
	#[getter]
	fn cols(&self) -> i32 {
		self.inner.cols
	}
	#[getter]
	fn width(&self) -> u32 {
		self.inner.width
	}
	#[getter]
	fn height(&self) -> u32 {
		self.inner.height
	}
	#[getter]
	fn theme(&self) -> PythonPlotTheme {
		self.inner.theme.into()
	}
	#[getter]
	fn background(&self) -> Rgba {
		rgba(self.inner.background)
	}
}

#[pyclass(name = "CellRect", frozen)]
pub(crate) struct PythonCellRect {
	#[pyo3(get)]
	x: i32,
	#[pyo3(get)]
	y: i32,
	#[pyo3(get)]
	w: i32,
	#[pyo3(get)]
	h: i32,
}

#[pyclass(name = "Figure", unsendable)]
pub(crate) struct PythonFigure {
	inner: Rc<RefCell<oa::plot::Figure>>,
}

#[pymethods]
impl PythonFigure {
	#[new]
	#[pyo3(signature = (config=None))]
	fn new(config: Option<PyRef<'_, PythonFigureConfig>>) -> Self {
		let config = config.map_or_else(oa::plot::FigureConfig::default, |value| value.inner.clone());
		Self {
			inner: Rc::new(RefCell::new(oa::plot::Figure::new(config))),
		}
	}

	fn ax(&self, row: i32, col: i32) -> PythonAxes {
		PythonAxes {
			target: PythonAxesTarget::Figure(self.inner.clone()),
			row,
			col,
		}
	}

	fn title(&self, text: &str) {
		self.inner.borrow_mut().title(text);
	}
	fn x_label(&self, text: &str) {
		self.inner.borrow_mut().x_label(text);
	}
	fn y_label(&self, text: &str) {
		self.inner.borrow_mut().y_label(text);
	}
	fn rows(&self) -> i32 {
		self.inner.borrow().rows()
	}
	fn cols(&self) -> i32 {
		self.inner.borrow().cols()
	}

	fn cell_rect(&self, row: i32, col: i32, width: u32, height: u32) -> PythonCellRect {
		let rect = self.inner.borrow().cell_rect(row, col, width, height);
		PythonCellRect {
			x: rect.x,
			y: rect.y,
			w: rect.w,
			h: rect.h,
		}
	}

	fn save_to(&self, path: &str, runtime: Option<&PythonRuntime>) -> PyResult<()> {
		let runtime = PythonRuntime::resolve(runtime)?;
		self
			.inner
			.borrow()
			.save_to_on(&runtime, path)
			.map_err(python_error)
	}

	fn render(&self, runtime: Option<&PythonRuntime>) -> PyResult<PythonImage> {
		let runtime = PythonRuntime::resolve(runtime)?;
		self
			.inner
			.borrow()
			.render_on(&runtime)
			.map(PythonImage::wrap)
			.map_err(python_error)
	}

	fn show(&self, runtime: Option<&PythonRuntime>) -> PyResult<()> {
		let runtime = PythonRuntime::resolve(runtime)?;
		self.inner.borrow().show_on(&runtime).map_err(python_error)
	}
}

/// Owning target an `Axes` binding delegates to — a recorded figure or the
/// figure held inside an attached `FigureView` (for post-attach updates).
enum PythonAxesTarget {
	Figure(Rc<RefCell<oa::plot::Figure>>),
	View(Rc<RefCell<oa::plot::FigureView>>),
}

#[pyclass(name = "Axes", unsendable)]
pub(crate) struct PythonAxes {
	target: PythonAxesTarget,
	row: i32,
	col: i32,
}

impl PythonAxes {
	fn update(&self, apply: impl FnOnce(&mut oa::plot::Axes)) {
		match &self.target {
			PythonAxesTarget::Figure(figure) => {
				apply(figure.borrow_mut().ax(self.row, self.col));
			}
			PythonAxesTarget::View(view) => {
				apply(view.borrow_mut().figure_mut().ax(self.row, self.col));
			}
		}
	}
}

#[pymethods]
impl PythonAxes {
	fn title(&self, text: &str) {
		self.update(|axes| axes.title(text));
	}
	fn title_color(&self, text: &str, value: Rgba) {
		self.update(|axes| axes.title_color(text, color(value)));
	}
	fn caption(&self, text: &str) {
		self.update(|axes| axes.caption(text));
	}
	fn caption_color(&self, text: &str, value: Rgba) {
		self.update(|axes| axes.caption_color(text, color(value)));
	}
	fn x_label(&self, text: &str) {
		self.update(|axes| axes.x_label(text));
	}
	fn y_label(&self, text: &str) {
		self.update(|axes| axes.y_label(text));
	}
	fn imshow(&self, image: &PythonImage) -> PyResult<()> {
		let texture = oa::render::texture_from_image(&image.inner).map_err(python_error)?;
		self.update(|axes| axes.imshow(texture));
		Ok(())
	}

	#[pyo3(signature = (values, style=None))]
	fn plot(&self, values: Vec<f32>, style: Option<PyRef<'_, PythonLineStyle>>) {
		let style = style.map_or_else(oa::plot::LineStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.plot(&values, style));
	}

	#[pyo3(signature = (x, y, style=None))]
	fn plot_xy(&self, x: Vec<f32>, y: Vec<f32>, style: Option<PyRef<'_, PythonLineStyle>>) {
		let style = style.map_or_else(oa::plot::LineStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.plot_xy(&x, &y, style));
	}

	#[pyo3(signature = (x, y, style=None))]
	fn scatter(&self, x: Vec<f32>, y: Vec<f32>, style: Option<PyRef<'_, PythonScatterStyle>>) {
		let style = style.map_or_else(oa::plot::ScatterStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.scatter(&x, &y, style));
	}

	#[pyo3(signature = (values, style=None))]
	fn bar(&self, values: Vec<f32>, style: Option<PyRef<'_, PythonBarStyle>>) {
		let style = style.map_or_else(oa::plot::BarStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.bar(&values, style));
	}

	#[pyo3(signature = (values, bins, style=None))]
	fn histogram(&self, values: Vec<f32>, bins: i32, style: Option<PyRef<'_, PythonBarStyle>>) {
		let style = style.map_or_else(oa::plot::BarStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.histogram(&values, bins, style));
	}

	#[pyo3(signature = (series, styles=None))]
	fn bar_grouped(&self, series: Vec<Vec<f32>>, styles: Option<Vec<PyRef<'_, PythonBarStyle>>>) {
		let styles = styles.map_or_else(Vec::new, |list| {
			list.iter().map(|value| value.inner.clone()).collect()
		});
		let layers: Vec<&[f32]> = series.iter().map(Vec::as_slice).collect();
		self.update(|axes| axes.bar_grouped(&layers, &styles));
	}

	#[pyo3(signature = (series, styles=None))]
	fn bar_stacked(&self, series: Vec<Vec<f32>>, styles: Option<Vec<PyRef<'_, PythonBarStyle>>>) {
		let styles = styles.map_or_else(Vec::new, |list| {
			list.iter().map(|value| value.inner.clone()).collect()
		});
		let layers: Vec<&[f32]> = series.iter().map(Vec::as_slice).collect();
		self.update(|axes| axes.bar_stacked(&layers, &styles));
	}

	#[pyo3(signature = (values, rows, cols, style=None))]
	fn heatmap(
		&self,
		values: Vec<f32>,
		rows: i32,
		cols: i32,
		style: Option<PyRef<'_, PythonHeatmapStyle>>,
	) {
		let style = style.map_or_else(oa::plot::HeatmapStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.heatmap(&values, rows, cols, style));
	}

	#[pyo3(signature = (values, style=None))]
	fn heatmap_matrix(&self, values: &PythonMatrix, style: Option<PyRef<'_, PythonHeatmapStyle>>) {
		let style = style.map_or_else(oa::plot::HeatmapStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.heatmap_matrix(values.inner.clone(), style));
	}

	#[pyo3(signature = (values, style=None))]
	fn plot_matrix(&self, values: &PythonMatrix, style: Option<PyRef<'_, PythonLineStyle>>) {
		let style = style.map_or_else(oa::plot::LineStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.plot_matrix(&values.inner, style));
	}

	#[pyo3(signature = (values, style=None))]
	fn plot_matrix_y(&self, values: &PythonMatrix, style: Option<PyRef<'_, PythonLineStyle>>) {
		let style = style.map_or_else(oa::plot::LineStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.plot_matrix_y(&values.inner, style));
	}

	fn annotate(&self, text: &str, x: f32, y: f32) {
		self.update(|axes| axes.annotate(text, x, y));
	}

	#[pyo3(signature = (capacity, style=None))]
	fn stream(&self, capacity: usize, style: Option<PyRef<'_, PythonLineStyle>>) -> usize {
		let style = style.map_or_else(oa::plot::LineStyle::default, |value| value.inner.clone());
		let mut handle = 0;
		self.update(|axes| handle = axes.stream(capacity, style));
		handle
	}

	fn stream_push(&self, series: usize, values: Vec<f32>) {
		self.update(|axes| axes.stream_push(series, &values));
	}

	#[pyo3(signature = (y_lo, y_hi, style=None))]
	fn band(&self, y_lo: Vec<f32>, y_hi: Vec<f32>, style: Option<PyRef<'_, PythonLineStyle>>) {
		let style = style.map_or_else(oa::plot::LineStyle::default, |value| value.inner.clone());
		self.update(|axes| axes.band(&y_lo, &y_hi, style));
	}

	#[pyo3(signature = (datasets, style=None))]
	fn boxplot(&self, datasets: Vec<Vec<f32>>, style: Option<PyRef<'_, PythonBoxStyle>>) {
		let style = style.map_or_else(oa::plot::BoxStyle::default, |value| value.inner.clone());
		let layers: Vec<&[f32]> = datasets.iter().map(Vec::as_slice).collect();
		self.update(|axes| axes.boxplot(&layers, style));
	}

	#[pyo3(signature = (values, decimals=None, labels=None))]
	fn xticks(&self, values: Vec<f32>, decimals: Option<usize>, labels: Option<Vec<String>>) {
		self.update(|axes| {
			axes.xticks(&values);
			axes.xformat(tick_format(decimals, labels));
		});
	}

	#[pyo3(signature = (values, decimals=None, labels=None))]
	fn yticks(&self, values: Vec<f32>, decimals: Option<usize>, labels: Option<Vec<String>>) {
		self.update(|axes| {
			axes.yticks(&values);
			axes.yformat(tick_format(decimals, labels));
		});
	}

	fn categories(&self, labels: Vec<String>) {
		let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
		self.update(|axes| axes.categories(&refs));
	}

	fn xscale(&self, scale: PythonPlotScale) {
		self.update(|axes| axes.xscale(scale.into()));
	}

	fn yscale(&self, scale: PythonPlotScale) {
		self.update(|axes| axes.yscale(scale.into()));
	}

	fn data_limits(&self) -> (f32, f32, f32, f32) {
		let mut result = (0.0, 0.0, 0.0, 0.0);
		self.update(|axes| {
			let limits = axes.data_limits();
			result = (limits[0], limits[1], limits[2], limits[3]);
		});
		result
	}

	fn explicit_limits(&self) -> Option<(f32, f32, f32, f32)> {
		let mut result = None;
		self.update(|axes| {
			result = axes.explicit_limits().map(|l| (l[0], l[1], l[2], l[3]));
		});
		result
	}

	fn artist_count(&self) -> usize {
		let mut count = 0;
		self.update(|axes| count = axes.artist_count());
		count
	}

	fn set_artist_hidden(&self, index: usize, hidden: bool) {
		self.update(|axes| axes.set_artist_hidden(index, hidden));
	}

	fn limits(&self, x_min: f32, x_max: f32, y_min: f32, y_max: f32) {
		self.update(|axes| axes.limits(x_min, x_max, y_min, y_max));
	}
	fn auto_limits(&self) {
		self.update(oa::plot::Axes::auto_limits);
	}
	fn grid(&self, visible: bool) {
		self.update(|axes| axes.grid(visible));
	}
	fn legend(&self, visible: bool) {
		self.update(|axes| axes.legend(visible));
	}
	fn border_color(&self, value: Rgba) {
		self.update(|axes| axes.border_color(color(value)));
	}
	fn clear(&self) {
		self.update(oa::plot::Axes::clear);
	}
}

/// Interactive `Figure` presentation through `Viewer` live mode.
///
/// `FigureView` takes ownership of the figure's recorded content at
/// construction: the passed `Figure` is left as an empty figure with the
/// same configuration. Record or stream further artists through
/// `view.ax(row, col)` — they reach the attached figure directly.
#[pyclass(name = "FigureView", unsendable)]
pub(crate) struct PythonFigureView {
	pub(crate) inner: Rc<RefCell<oa::plot::FigureView>>,
}

#[pymethods]
impl PythonFigureView {
	#[new]
	fn new(figure: &PythonFigure) -> Self {
		let config = figure.inner.borrow().config().clone();
		let taken = std::mem::replace(
			&mut *figure.inner.borrow_mut(),
			oa::plot::Figure::new(config),
		);
		Self {
			inner: Rc::new(RefCell::new(oa::plot::FigureView::new(taken))),
		}
	}

	/// Access the (row, col) axes inside the attached figure.
	fn ax(&self, row: i32, col: i32) -> PythonAxes {
		PythonAxes {
			target: PythonAxesTarget::View(self.inner.clone()),
			row,
			col,
		}
	}

	/// Toggle the pointer crosshair and data-coordinate readout.
	fn crosshair(&self, visible: bool) {
		self.inner.borrow_mut().crosshair(visible);
	}

	/// Restore every axes to the limits recorded at attach time.
	fn reset_view(&self) {
		self.inner.borrow_mut().reset_view();
	}

	/// Present the view in a blocking `Viewer` live-mode window.
	#[pyo3(signature = (config=None, runtime=None))]
	fn show(
		&self,
		config: Option<PyRef<'_, crate::viewer::PythonViewerConfig>>,
		runtime: Option<&PythonRuntime>,
	) -> PyResult<()> {
		let runtime = PythonRuntime::resolve(runtime)?;
		let cfg = config.map_or_else(
			|| oa::ViewerConfig {
				mode: oa::ViewerMode::Live,
				..Default::default()
			},
			|c| oa::ViewerConfig {
				mode: oa::ViewerMode::Live,
				..c.inner.clone()
			},
		);
		let mut view = self.inner.borrow_mut();
		let mut viewer = oa::Viewer::new(cfg);
		viewer.set_live_source(&mut *view);
		viewer.run(&runtime).map_err(python_error)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonPlotTheme>()?;
	module.add_class::<PythonPlotScale>()?;
	module.add_class::<PythonLineStyle>()?;
	module.add_class::<PythonScatterStyle>()?;
	module.add_class::<PythonBarStyle>()?;
	module.add_class::<PythonBoxStyle>()?;
	module.add_class::<PythonHeatmapStyle>()?;
	module.add_class::<PythonFigureConfig>()?;
	module.add_class::<PythonCellRect>()?;
	module.add_class::<PythonFigure>()?;
	module.add_class::<PythonAxes>()?;
	module.add_class::<PythonFigureView>()?;
	Ok(())
}
