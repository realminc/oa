use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

// ── AvgPool2d ─────────────────────────────────────────────────────────────────

/// Parameterless two-dimensional average-pooling module for NCHW matrices.
#[pyclass(name = "AvgPool2d", unsendable)]
pub(crate) struct PythonAvgPool2d {
	inner: oa::ml::nn::AvgPool2d,
}

#[pymethods]
impl PythonAvgPool2d {
	/// Construct with `stride == kernel_size` and no padding, or explicit geometry.
	#[new]
	#[pyo3(signature = (kernel_size, stride = 0, padding = 0))]
	pub fn new(kernel_size: usize, stride: usize, padding: usize) -> PyResult<Self> {
		let effective_stride = if stride == 0 { kernel_size } else { stride };
		oa::ml::nn::AvgPool2d::with_options(kernel_size, effective_stride, padding)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Pool one NCHW Matrix without submitting or waiting.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn kernel_size(&self) -> usize {
		self.inner.kernel_size()
	}

	#[getter]
	pub fn stride(&self) -> usize {
		self.inner.stride()
	}

	#[getter]
	pub fn padding(&self) -> usize {
		self.inner.padding()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"AvgPool2d(kernel_size={}, stride={}, padding={})",
			self.inner.kernel_size(),
			self.inner.stride(),
			self.inner.padding(),
		)
	}
}

// ── MaxPool2d ─────────────────────────────────────────────────────────────────

/// Parameterless two-dimensional max-pooling module for NCHW matrices.
#[pyclass(name = "MaxPool2d", unsendable)]
pub(crate) struct PythonMaxPool2d {
	inner: oa::ml::nn::MaxPool2d,
}

#[pymethods]
impl PythonMaxPool2d {
	/// Construct with `stride == kernel_size` and no padding, or explicit geometry.
	#[new]
	#[pyo3(signature = (kernel_size, stride = 0, padding = 0))]
	pub fn new(kernel_size: usize, stride: usize, padding: usize) -> PyResult<Self> {
		let effective_stride = if stride == 0 { kernel_size } else { stride };
		oa::ml::nn::MaxPool2d::with_options(kernel_size, effective_stride, padding)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Pool one NCHW Matrix and retain argmax state for reverse mode.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn kernel_size(&self) -> usize {
		self.inner.kernel_size()
	}

	#[getter]
	pub fn stride(&self) -> usize {
		self.inner.stride()
	}

	#[getter]
	pub fn padding(&self) -> usize {
		self.inner.padding()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"MaxPool2d(kernel_size={}, stride={}, padding={})",
			self.inner.kernel_size(),
			self.inner.stride(),
			self.inner.padding(),
		)
	}
}

// ── AdaptiveAvgPool2d ─────────────────────────────────────────────────────────

/// Parameterless adaptive average-pooling module for NCHW matrices.
#[pyclass(name = "AdaptiveAvgPool2d", unsendable)]
pub(crate) struct PythonAdaptiveAvgPool2d {
	inner: oa::ml::nn::AdaptiveAvgPool2d,
}

#[pymethods]
impl PythonAdaptiveAvgPool2d {
	/// Construct with a square or independent `(height, width)` output extent.
	#[new]
	#[pyo3(signature = (output_height, output_width = 0))]
	pub fn new(output_height: usize, output_width: usize) -> PyResult<Self> {
		let effective_width = if output_width == 0 {
			output_height
		} else {
			output_width
		};
		oa::ml::nn::AdaptiveAvgPool2d::with_output_size(output_height, effective_width)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Adaptively pool one NCHW Matrix without submitting or waiting.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn output_height(&self) -> usize {
		self.inner.output_height()
	}

	#[getter]
	pub fn output_width(&self) -> usize {
		self.inner.output_width()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"AdaptiveAvgPool2d(output_height={}, output_width={})",
			self.inner.output_height(),
			self.inner.output_width(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonAvgPool2d>()?;
	module.add_class::<PythonMaxPool2d>()?;
	module.add_class::<PythonAdaptiveAvgPool2d>()?;
	Ok(())
}
