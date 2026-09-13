use pyo3::prelude::*;

use crate::error::python_error;

#[pyclass(name = "Event", unsendable)]
pub(crate) struct PythonEvent {
	inner: oa::Event,
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
		self.inner
			.device_duration()
			.map(|duration| duration.as_secs_f64())
			.map_err(python_error)
	}

	fn try_device_duration_seconds(&self) -> PyResult<Option<f64>> {
		self.inner
			.try_device_duration()
			.map(|duration| duration.map(|value| value.as_secs_f64()))
			.map_err(python_error)
	}

	fn __repr__(&self) -> String {
		format!("{:?}", self.inner)
	}
}

#[pyclass(name = "Engine", unsendable)]
pub(crate) struct PythonEngine {
	pub(crate) inner: oa::Engine,
}

#[pymethods]
impl PythonEngine {
	#[new]
	#[pyo3(signature = (device=None))]
	fn new(device: Option<usize>) -> PyResult<Self> {
		let inner = match device {
			Some(index) => oa::Engine::builder()
				.devices(oa::DeviceSelection::Index(index))
				.build(),
			None => oa::Engine::new(),
		}
		.map_err(python_error)?;
		Ok(Self { inner })
	}

	fn checkpoint(&self) -> PyResult<PythonEvent> {
		self.inner
			.checkpoint()
			.map(PythonEvent::wrap)
			.map_err(python_error)
	}

	fn checkpoint_timed(&self) -> PyResult<PythonEvent> {
		self.inner
			.checkpoint_timed()
			.map(PythonEvent::wrap)
			.map_err(python_error)
	}

	fn __repr__(&self) -> &'static str {
		"Engine(backend='vulkan')"
	}
}
