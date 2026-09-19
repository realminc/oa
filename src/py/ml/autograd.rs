use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

/// Trainable FP32 matrix with one accumulated gradient.
#[pyclass(name = "Parameter", unsendable)]
pub(crate) struct PythonParameter {
	pub(crate) inner: oa::ml::Parameter,
}

#[pymethods]
impl PythonParameter {
	/// Construct a trainable parameter from an existing FP32 matrix.
	#[new]
	pub fn new(name: &str, data: &PythonMatrix) -> PyResult<Self> {
		oa::ml::Parameter::new(name, data.inner.clone())
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Return this parameter's local name.
	pub fn name(&self) -> String {
		self.inner.name()
	}

	/// Return the current parameter data.
	pub fn data(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.data())
	}

	/// Return the current parameter data (alias for `data()`).
	#[getter]
	pub fn value(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.data())
	}

	/// Return the accumulated gradient, when present.
	pub fn gradient(&self) -> Option<PythonMatrix> {
		self.inner.gradient().map(PythonMatrix::wrap)
	}

	/// Enable or disable gradient accumulation.
	pub fn set_requires_grad(&self, requires_grad: bool) {
		self.inner.set_requires_grad(requires_grad);
	}

	/// Return whether gradient accumulation is enabled.
	pub fn requires_grad(&self) -> bool {
		self.inner.requires_grad()
	}

	pub fn __repr__(&self) -> String {
		let grad_str = if self.inner.gradient().is_some() {
			"has_grad"
		} else {
			"no_grad"
		};
		format!("Parameter(name={:?}, {})", self.inner.name(), grad_str)
	}
}

/// Thread-affine reverse-mode recording scope.
///
/// Constructing a tape automatically begins recording ML operations on the
/// current thread. Call `backward(loss)` to run the backward pass.
#[pyclass(name = "GradientTape", unsendable)]
pub(crate) struct PythonGradientTape {
	inner: oa::ml::GradientTape,
}

#[pymethods]
impl PythonGradientTape {
	/// Begin a reverse-mode recording scope.
	#[new]
	pub fn new() -> Self {
		Self {
			inner: oa::ml::GradientTape::new(),
		}
	}

	/// Record the backward pass for one FP32 scalar loss.
	pub fn backward(&self, root: &PythonMatrix) -> PyResult<()> {
		self.inner.backward(&root.inner).map_err(python_error)
	}

	/// Close the recording scope without running backward.
	pub fn close(&self) {
		self.inner.close();
	}

	pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
		slf
	}

	pub fn __exit__(
		&self,
		_exc_type: &Bound<'_, PyAny>,
		_exc_val: &Bound<'_, PyAny>,
		_exc_tb: &Bound<'_, PyAny>,
	) -> bool {
		self.inner.close();
		false
	}
}

/// Clip the combined L2 norm of a list of gradient matrices in place.
///
/// Accepts either a list of `Matrix` objects **or** a list of `Parameter`
/// objects (in which case the current parameter data is used as the gradient
/// accumulator).
#[pyfunction]
pub(crate) fn ml_clip_grad_norm(
	py: Python<'_>,
	gradients: &Bound<'_, pyo3::types::PyList>,
	max_norm: f32,
) -> PyResult<()> {
	let mut mats: Vec<oa::Matrix> = Vec::with_capacity(gradients.len());
	for item in gradients.iter() {
		if let Ok(m) = item.extract::<PyRef<PythonMatrix>>() {
			mats.push(m.inner.clone());
		} else if let Ok(p) = item.extract::<PyRef<PythonParameter>>() {
			// Use the parameter's gradient matrix when present.
			if let Some(grad) = p.inner.gradient() {
				mats.push(grad);
			}
		} else {
			return Err(pyo3::exceptions::PyTypeError::new_err(
				"clip_grad_norm: list items must be Matrix or Parameter",
			));
		}
	}
	let _ = py;
	oa::ml::optim::clip_grad_norm(&mats, max_norm).map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonParameter>()?;
	module.add_class::<PythonGradientTape>()?;
	module.add_function(wrap_pyfunction!(ml_clip_grad_norm, module)?)?;
	Ok(())
}
