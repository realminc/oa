//! Python bindings for the retained `oa::plot` figure and axes API.

use std::{cell::RefCell, rc::Rc};

use pyo3::prelude::*;

use crate::{error::python_error, image::PythonImage, runtime::PythonEngine};

type Rgba = (f32, f32, f32, f32);

fn color(value: Rgba) -> oa::Color {
	oa::Color::new(value.0, value.1, value.2, value.3)
}

fn rgba(value: oa::Color) -> Rgba {
	(value.r, value.g, value.b, value.a)
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

#[pyclass(name = "LineStyle")]
#[derive(Clone)]
pub(crate) struct PythonLineStyle {
	inner: oa::plot::LineStyle,
}

#[pymethods]
impl PythonLineStyle {
	#[new]
	#[pyo3(signature = (color=(0.0, 0.0, 0.0, 0.0), label=String::new(), width=1.35, antialias_samples=4))]
	fn new(color: Rgba, label: String, width: f32, antialias_samples: u32) -> Self {
		Self {
			inner: oa::plot::LineStyle {
				color: self::color(color),
				label,
				width,
				antialias_samples,
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
	#[pyo3(signature = (color=(0.0, 0.0, 0.0, 0.0), label=String::new(), radius=3.0))]
	fn new(color: Rgba, label: String, radius: f32) -> Self {
		Self {
			inner: oa::plot::ScatterStyle {
				color: self::color(color),
				label,
				radius,
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
	#[pyo3(signature = (color=(0.0, 0.0, 0.0, 0.0), label=String::new(), gap=0.18))]
	fn new(color: Rgba, label: String, gap: f32) -> Self {
		Self {
			inner: oa::plot::BarStyle {
				color: self::color(color),
				label,
				gap,
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
	#[pyo3(signature = (v_min=0.0, v_max=1.0, colormap=1, auto_scale=true, show_grid=true))]
	fn new(v_min: f32, v_max: f32, colormap: u32, auto_scale: bool, show_grid: bool) -> Self {
		Self {
			inner: oa::plot::HeatmapStyle {
				v_min,
				v_max,
				colormap,
				auto_scale,
				show_grid,
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
			figure: self.inner.clone(),
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

	fn save_to(&self, engine: &PythonEngine, path: &str) -> PyResult<()> {
		self
			.inner
			.borrow()
			.save_to(&engine.inner, path)
			.map_err(python_error)
	}

	fn render(&self, engine: &PythonEngine) -> PyResult<PythonImage> {
		self
			.inner
			.borrow()
			.render(&engine.inner)
			.map(PythonImage::wrap)
			.map_err(python_error)
	}

	fn show(&self, engine: &PythonEngine) -> PyResult<()> {
		self
			.inner
			.borrow()
			.show(&engine.inner)
			.map_err(python_error)
	}
}

#[pyclass(name = "Axes", unsendable)]
pub(crate) struct PythonAxes {
	figure: Rc<RefCell<oa::plot::Figure>>,
	row: i32,
	col: i32,
}

impl PythonAxes {
	fn update(&self, apply: impl FnOnce(&mut oa::plot::Axes)) {
		apply(self.figure.borrow_mut().ax(self.row, self.col));
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonPlotTheme>()?;
	module.add_class::<PythonLineStyle>()?;
	module.add_class::<PythonScatterStyle>()?;
	module.add_class::<PythonBarStyle>()?;
	module.add_class::<PythonHeatmapStyle>()?;
	module.add_class::<PythonFigureConfig>()?;
	module.add_class::<PythonCellRect>()?;
	module.add_class::<PythonFigure>()?;
	module.add_class::<PythonAxes>()?;
	Ok(())
}
