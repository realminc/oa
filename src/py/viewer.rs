//! Python bindings for `oa::Viewer` and its configuration types.
//!
//! Donor reference: `oa/ui/viewer.h`, `oa/ui/viewer.cpp`

use pyo3::prelude::*;

use crate::{error::python_error, runtime::PythonEngine};

// ── ViewerMode ────────────────────────────────────────────────────────────────

/// Media source mode for the Viewer.
#[pyclass(name = "ViewerMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonViewerMode {
	Auto = 0,
	Image = 1,
	Video = 2,
	Audio = 3,
	Live = 4,
}

impl From<PythonViewerMode> for oa::ViewerMode {
	fn from(value: PythonViewerMode) -> Self {
		match value {
			PythonViewerMode::Auto => oa::ViewerMode::Auto,
			PythonViewerMode::Image => oa::ViewerMode::Image,
			PythonViewerMode::Video => oa::ViewerMode::Video,
			PythonViewerMode::Audio => oa::ViewerMode::Audio,
			PythonViewerMode::Live => oa::ViewerMode::Live,
		}
	}
}

// ── ViewerAudioView ───────────────────────────────────────────────────────────

/// Audio visualization mode.
#[pyclass(name = "ViewerAudioView", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonViewerAudioView {
	Waveform = 0,
	Spectrum = 1,
	Mel = 2,
}

impl From<PythonViewerAudioView> for oa::ViewerAudioView {
	fn from(value: PythonViewerAudioView) -> Self {
		match value {
			PythonViewerAudioView::Waveform => oa::ViewerAudioView::Waveform,
			PythonViewerAudioView::Spectrum => oa::ViewerAudioView::Spectrum,
			PythonViewerAudioView::Mel => oa::ViewerAudioView::Mel,
		}
	}
}

// ── ViewerCanvasBackground ────────────────────────────────────────────────────

/// Canvas background style.
#[pyclass(name = "ViewerCanvasBackground", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonViewerCanvasBackground {
	Dark = 0,
	Gradient = 1,
}

impl From<PythonViewerCanvasBackground> for oa::ViewerCanvasBackground {
	fn from(value: PythonViewerCanvasBackground) -> Self {
		match value {
			PythonViewerCanvasBackground::Dark => oa::ViewerCanvasBackground::Dark,
			PythonViewerCanvasBackground::Gradient => oa::ViewerCanvasBackground::Gradient,
		}
	}
}

// ── ViewerConfig ──────────────────────────────────────────────────────────────

/// Configuration for a Viewer window.
#[pyclass(name = "ViewerConfig")]
#[derive(Clone)]
pub(crate) struct PythonViewerConfig {
	pub(crate) inner: oa::ViewerConfig,
}

#[pymethods]
impl PythonViewerConfig {
	/// Create a default Viewer configuration.
	#[new]
	fn new() -> Self {
		Self {
			inner: oa::ViewerConfig::default(),
		}
	}

	#[getter]
	fn mode(&self) -> PythonViewerMode {
		match self.inner.mode {
			oa::ViewerMode::Auto => PythonViewerMode::Auto,
			oa::ViewerMode::Image => PythonViewerMode::Image,
			oa::ViewerMode::Video => PythonViewerMode::Video,
			oa::ViewerMode::Audio => PythonViewerMode::Audio,
			oa::ViewerMode::Live => PythonViewerMode::Live,
		}
	}

	#[setter]
	fn set_mode(&mut self, mode: PythonViewerMode) {
		self.inner.mode = mode.into();
	}

	#[getter]
	fn path(&self) -> &str {
		&self.inner.path
	}

	#[setter]
	fn set_path(&mut self, path: String) {
		self.inner.path = path;
	}

	#[getter]
	fn title(&self) -> &str {
		&self.inner.title
	}

	#[setter]
	fn set_title(&mut self, title: String) {
		self.inner.title = title;
	}

	#[getter]
	fn width(&self) -> u32 {
		self.inner.width
	}

	#[setter]
	fn set_width(&mut self, width: u32) {
		self.inner.width = width;
	}

	#[getter]
	fn height(&self) -> u32 {
		self.inner.height
	}

	#[setter]
	fn set_height(&mut self, height: u32) {
		self.inner.height = height;
	}

	#[getter]
	fn show_timeline(&self) -> bool {
		self.inner.show_timeline
	}

	#[setter]
	fn set_show_timeline(&mut self, v: bool) {
		self.inner.show_timeline = v;
	}

	#[getter]
	fn show_help(&self) -> bool {
		self.inner.show_help
	}

	#[setter]
	fn set_show_help(&mut self, value: bool) {
		self.inner.show_help = value;
	}

	#[getter]
	fn show_stats(&self) -> bool {
		self.inner.show_stats
	}

	#[setter]
	fn set_show_stats(&mut self, value: bool) {
		self.inner.show_stats = value;
	}

	#[getter]
	fn vsync(&self) -> bool {
		self.inner.vsync
	}

	#[setter]
	fn set_vsync(&mut self, value: bool) {
		self.inner.vsync = value;
	}

	#[getter]
	fn loop_media(&self) -> bool {
		self.inner.loop_media
	}

	#[setter]
	fn set_loop_media(&mut self, v: bool) {
		self.inner.loop_media = v;
	}

	#[getter]
	fn start_playing(&self) -> bool {
		self.inner.start_playing
	}

	#[setter]
	fn set_start_playing(&mut self, v: bool) {
		self.inner.start_playing = v;
	}

	#[getter]
	fn canvas_background(&self) -> PythonViewerCanvasBackground {
		match self.inner.canvas_background {
			oa::ViewerCanvasBackground::Dark => PythonViewerCanvasBackground::Dark,
			oa::ViewerCanvasBackground::Gradient => PythonViewerCanvasBackground::Gradient,
		}
	}

	#[setter]
	fn set_canvas_background(&mut self, value: PythonViewerCanvasBackground) {
		self.inner.canvas_background = value.into();
	}

	#[getter]
	fn show_canvas_grid(&self) -> bool {
		self.inner.show_canvas_grid
	}

	#[setter]
	fn set_show_canvas_grid(&mut self, value: bool) {
		self.inner.show_canvas_grid = value;
	}

	#[getter]
	fn audio_view(&self) -> PythonViewerAudioView {
		match self.inner.audio_view {
			oa::ViewerAudioView::Waveform => PythonViewerAudioView::Waveform,
			oa::ViewerAudioView::Spectrum => PythonViewerAudioView::Spectrum,
			oa::ViewerAudioView::Mel => PythonViewerAudioView::Mel,
		}
	}

	#[setter]
	fn set_audio_view(&mut self, value: PythonViewerAudioView) {
		self.inner.audio_view = value.into();
	}

	fn __repr__(&self) -> String {
		format!(
			"ViewerConfig(mode={:?}, title={:?}, size={}×{})",
			self.inner.mode, self.inner.title, self.inner.width, self.inner.height
		)
	}
}

// ── Viewer ────────────────────────────────────────────────────────────────────

/// Windowed or headless media inspection session.
///
/// Opens an SDL3 window and presents image, video, or audio content through the
/// GPU compositor. Blocks until the user closes the window.
#[pyclass(name = "Viewer", unsendable)]
pub(crate) struct PythonViewer;

#[pymethods]
impl PythonViewer {
	/// Open `path` in a window with a fresh presentation-capable engine.
	#[staticmethod]
	#[pyo3(signature = (path, config=None))]
	fn preview(path: &str, config: Option<PyRef<'_, PythonViewerConfig>>) -> PyResult<()> {
		let cfg = config.map_or_else(oa::ViewerConfig::default, |c| c.inner.clone());
		oa::Viewer::preview_with_config(path, cfg).map_err(python_error)
	}

	/// Open `path` borrowing `engine`.
	#[staticmethod]
	#[pyo3(signature = (engine, path, config=None))]
	fn preview_with_engine(
		engine: &PythonEngine,
		path: &str,
		config: Option<PyRef<'_, PythonViewerConfig>>,
	) -> PyResult<()> {
		let cfg = config.map_or_else(oa::ViewerConfig::default, |c| c.inner.clone());
		oa::Viewer::preview_with_engine_and_config(&engine.inner, path, cfg).map_err(python_error)
	}

	/// Display a semantic Image in a blocking window.
	#[staticmethod]
	#[pyo3(signature = (engine, image, config=None))]
	fn show(
		engine: &PythonEngine,
		image: &crate::image::PythonImage,
		config: Option<PyRef<'_, PythonViewerConfig>>,
	) -> PyResult<()> {
		let cfg = config.map_or_else(oa::ViewerConfig::default, |c| c.inner.clone());
		let texture = oa::render::texture_from_image(&image.inner).map_err(python_error)?;
		oa::Viewer::show_with_config(&engine.inner, &texture, cfg).map_err(python_error)
	}
}

// ── Registration ──────────────────────────────────────────────────────────────

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonViewerMode>()?;
	module.add_class::<PythonViewerAudioView>()?;
	module.add_class::<PythonViewerCanvasBackground>()?;
	module.add_class::<PythonViewerConfig>()?;
	module.add_class::<PythonViewer>()?;
	Ok(())
}
