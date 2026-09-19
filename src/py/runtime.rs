use pyo3::prelude::*;

use crate::error::python_error;

// ── LogLevel ──────────────────────────────────────────────────────────────────

/// Severity threshold and record level for OA diagnostics.
#[pyclass(name = "LogLevel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonLogLevel {
	Trace = 0,
	Debug = 1,
	Info = 2,
	Warn = 3,
	Error = 4,
	Fatal = 5,
	Off = 6,
}

impl From<PythonLogLevel> for oa::LogLevel {
	fn from(value: PythonLogLevel) -> Self {
		match value {
			PythonLogLevel::Trace => oa::LogLevel::Trace,
			PythonLogLevel::Debug => oa::LogLevel::Debug,
			PythonLogLevel::Info => oa::LogLevel::Info,
			PythonLogLevel::Warn => oa::LogLevel::Warn,
			PythonLogLevel::Error => oa::LogLevel::Error,
			PythonLogLevel::Fatal => oa::LogLevel::Fatal,
			PythonLogLevel::Off => oa::LogLevel::Off,
		}
	}
}

impl From<oa::LogLevel> for PythonLogLevel {
	fn from(value: oa::LogLevel) -> Self {
		match value {
			oa::LogLevel::Trace => PythonLogLevel::Trace,
			oa::LogLevel::Debug => PythonLogLevel::Debug,
			oa::LogLevel::Info => PythonLogLevel::Info,
			oa::LogLevel::Warn => PythonLogLevel::Warn,
			oa::LogLevel::Error => PythonLogLevel::Error,
			oa::LogLevel::Fatal => PythonLogLevel::Fatal,
			oa::LogLevel::Off => PythonLogLevel::Off,
		}
	}
}

// ── LogOptions ────────────────────────────────────────────────────────────────

/// Configuration for the logging session owned by an Engine.
#[pyclass(name = "LogOptions")]
#[derive(Clone)]
pub(crate) struct PythonLogOptions {
	inner: oa::LogOptions,
}

#[pymethods]
impl PythonLogOptions {
	/// Create default console-logging options.
	#[new]
	pub fn new() -> Self {
		Self {
			inner: oa::LogOptions::new(),
		}
	}

	/// Select the directory used when file output is enabled.
	pub fn directory(mut slf: PyRefMut<'_, Self>, directory: String) -> PyRefMut<'_, Self> {
		slf.inner = slf.inner.clone().directory(directory);
		slf
	}

	/// Select the safe filename prefix used when file output is enabled.
	pub fn prefix(mut slf: PyRefMut<'_, Self>, prefix: String) -> PyRefMut<'_, Self> {
		slf.inner = slf.inner.clone().prefix(prefix);
		slf
	}

	/// Set the minimum emitted severity.
	pub fn minimum_level(mut slf: PyRefMut<'_, Self>, level: PythonLogLevel) -> PyRefMut<'_, Self> {
		slf.inner = slf.inner.clone().minimum_level(level.into());
		slf
	}

	/// Enable or disable synchronous standard-error output.
	pub fn console_output(mut slf: PyRefMut<'_, Self>, enabled: bool) -> PyRefMut<'_, Self> {
		slf.inner = slf.inner.clone().console_output(enabled);
		slf
	}

	/// Enable or disable append-only dated file output.
	pub fn file_output(mut slf: PyRefMut<'_, Self>, enabled: bool) -> PyRefMut<'_, Self> {
		slf.inner = slf.inner.clone().file_output(enabled);
		slf
	}

	/// Enable or disable ANSI severity colors on console output.
	pub fn color_output(mut slf: PyRefMut<'_, Self>, enabled: bool) -> PyRefMut<'_, Self> {
		slf.inner = slf.inner.clone().color_output(enabled);
		slf
	}

	/// Return the configured minimum severity level.
	pub fn level(&self) -> PythonLogLevel {
		self.inner.level().into()
	}

	pub fn __repr__(&self) -> String {
		format!("LogOptions(level={:?})", self.inner.level())
	}
}

// ── Event ─────────────────────────────────────────────────────────────────────

#[pyclass(name = "Event", unsendable)]
pub(crate) struct PythonEvent {
	pub(crate) inner: oa::Event,
}

impl PythonEvent {
	fn wrap(inner: oa::Event) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonEvent {
	fn is_complete(&self) -> PyResult<bool> {
		self.inner.is_complete().map_err(python_error)
	}

	fn wait(&self) -> PyResult<()> {
		self.inner.wait().map_err(python_error)
	}

	fn device_duration_seconds(&self) -> PyResult<f64> {
		self
			.inner
			.device_duration()
			.map(|duration| duration.as_secs_f64())
			.map_err(python_error)
	}

	fn try_device_duration_seconds(&self) -> PyResult<Option<f64>> {
		self
			.inner
			.try_device_duration()
			.map(|duration| duration.map(|value| value.as_secs_f64()))
			.map_err(python_error)
	}

	fn __repr__(&self) -> String {
		format!("{:?}", self.inner)
	}
}

// ── Engine ────────────────────────────────────────────────────────────────────

#[pyclass(name = "Engine", unsendable)]
pub(crate) struct PythonEngine {
	pub(crate) inner: oa::Engine,
}

#[pymethods]
impl PythonEngine {
	#[new]
	#[pyo3(signature = (device=None, logging=None))]
	fn new(device: Option<usize>, logging: Option<PyRef<'_, PythonLogOptions>>) -> PyResult<Self> {
		let mut builder = oa::Engine::builder();
		if let Some(index) = device {
			builder = builder.devices(oa::DeviceSelection::Index(index));
		}
		if let Some(opts) = logging {
			builder = builder.logging(opts.inner.clone());
		}
		let inner = builder.build().map_err(python_error)?;
		Ok(Self { inner })
	}

	fn checkpoint(&self) -> PyResult<PythonEvent> {
		self
			.inner
			.checkpoint()
			.map(PythonEvent::wrap)
			.map_err(python_error)
	}

	fn checkpoint_timed(&self) -> PyResult<PythonEvent> {
		self
			.inner
			.checkpoint_timed()
			.map(PythonEvent::wrap)
			.map_err(python_error)
	}

	/// Set the minimum severity of records emitted by this Engine's logger.
	fn set_log_level(&self, level: PythonLogLevel) {
		self.inner.set_log_level(level.into());
	}

	/// Return the current minimum log severity.
	fn log_level(&self) -> PythonLogLevel {
		self.inner.log_level().into()
	}

	/// Return the current log file path, if file output is enabled.
	fn log_path(&self) -> Option<String> {
		self
			.inner
			.log_path()
			.map(|path| path.to_string_lossy().into_owned())
	}

	/// Flush any pending log file writes.
	fn flush_log(&self) -> PyResult<()> {
		self.inner.flush_log().map_err(python_error)
	}

	fn __repr__(&self) -> &'static str {
		"Engine(backend='vulkan')"
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonLogLevel>()?;
	module.add_class::<PythonLogOptions>()?;
	module.add_class::<PythonEvent>()?;
	module.add_class::<PythonEngine>()?;
	Ok(())
}
