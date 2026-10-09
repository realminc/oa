use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

// ── Conv1d ────────────────────────────────────────────────────────────────────

/// Trainable FP32 one-dimensional convolution over NCL matrices.
#[pyclass(name = "Conv1d", unsendable)]
pub(crate) struct PythonConv1d {
	pub(crate) inner: oa::ml::nn::Conv1d,
}

#[pymethods]
impl PythonConv1d {
	/// Construct with symmetric-uniform weight and zero bias.
	#[new]
	#[pyo3(signature = (engine, input_channels, output_channels, kernel_size, stride = 1, padding = 0, dilation = 1, seed = 0))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		engine: &PythonEngine,
		input_channels: usize,
		output_channels: usize,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		dilation: usize,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::Conv1d::with_seed(
			&engine.inner,
			input_channels,
			output_channels,
			kernel_size,
			stride,
			padding,
			dilation,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Apply Conv1d without submitting or waiting.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn input_channels(&self) -> usize {
		self.inner.input_channels()
	}

	#[getter]
	pub fn output_channels(&self) -> usize {
		self.inner.output_channels()
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

	#[getter]
	pub fn dilation(&self) -> usize {
		self.inner.dilation()
	}

	pub fn weight(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.weight(),
		}
	}

	pub fn bias(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.bias(),
		}
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Conv1d(in={}, out={}, kernel={}, stride={}, padding={}, dilation={})",
			self.inner.input_channels(),
			self.inner.output_channels(),
			self.inner.kernel_size(),
			self.inner.stride(),
			self.inner.padding(),
			self.inner.dilation(),
		)
	}
}

// ── ConvTranspose1d ───────────────────────────────────────────────────────────

/// Trainable bias-free FP32 one-dimensional transposed convolution over NCL matrices.
#[pyclass(name = "ConvTranspose1d", unsendable)]
pub(crate) struct PythonConvTranspose1d {
	pub(crate) inner: oa::ml::nn::ConvTranspose1d,
}

#[pymethods]
impl PythonConvTranspose1d {
	/// Construct with symmetric-uniform IOK weight and no bias.
	#[new]
	#[pyo3(signature = (engine, input_channels, output_channels, kernel_size, stride = 1, padding = 0, seed = 0))]
	pub fn new(
		engine: &PythonEngine,
		input_channels: usize,
		output_channels: usize,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::ConvTranspose1d::with_seed(
			&engine.inner,
			input_channels,
			output_channels,
			kernel_size,
			stride,
			padding,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Apply ConvTranspose1d without submitting or waiting.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn input_channels(&self) -> usize {
		self.inner.input_channels()
	}

	#[getter]
	pub fn output_channels(&self) -> usize {
		self.inner.output_channels()
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

	pub fn weight(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.weight(),
		}
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ConvTranspose1d(in={}, out={}, kernel={}, stride={}, padding={})",
			self.inner.input_channels(),
			self.inner.output_channels(),
			self.inner.kernel_size(),
			self.inner.stride(),
			self.inner.padding(),
		)
	}
}

// ── ConvTranspose2d ───────────────────────────────────────────────────────────

/// Trainable FP32 two-dimensional transposed convolution over NCHW matrices.
#[pyclass(name = "ConvTranspose2d", unsendable)]
pub(crate) struct PythonConvTranspose2d {
	pub(crate) inner: oa::ml::nn::ConvTranspose2d,
}

#[pymethods]
impl PythonConvTranspose2d {
	/// Construct with symmetric-uniform IOKK weight and zero bias.
	#[new]
	#[pyo3(signature = (engine, input_channels, output_channels, kernel_size, stride = 1, padding = 0, seed = 0))]
	pub fn new(
		engine: &PythonEngine,
		input_channels: usize,
		output_channels: usize,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::ConvTranspose2d::with_seed(
			&engine.inner,
			input_channels,
			output_channels,
			kernel_size,
			stride,
			padding,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Apply ConvTranspose2d without submitting or waiting.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn input_channels(&self) -> usize {
		self.inner.input_channels()
	}

	#[getter]
	pub fn output_channels(&self) -> usize {
		self.inner.output_channels()
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

	pub fn weight(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.weight(),
		}
	}

	pub fn bias(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.bias(),
		}
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ConvTranspose2d(in={}, out={}, kernel={}, stride={}, padding={})",
			self.inner.input_channels(),
			self.inner.output_channels(),
			self.inner.kernel_size(),
			self.inner.stride(),
			self.inner.padding(),
		)
	}
}

// ── Conv2d ────────────────────────────────────────────────────────────────────

/// Trainable grouped FP32 two-dimensional convolution over NCHW matrices.
#[pyclass(name = "Conv2d", unsendable)]
pub(crate) struct PythonConv2d {
	pub(crate) inner: oa::ml::nn::Conv2d,
}

#[pymethods]
impl PythonConv2d {
	/// Construct with symmetric-uniform OIHW/group weight and zero bias.
	#[new]
	#[pyo3(signature = (engine, input_channels, output_channels, kernel_size, stride = 1, padding = 0, groups = 1, seed = 0))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		engine: &PythonEngine,
		input_channels: usize,
		output_channels: usize,
		kernel_size: usize,
		stride: usize,
		padding: usize,
		groups: usize,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::Conv2d::with_seed(
			&engine.inner,
			input_channels,
			output_channels,
			kernel_size,
			stride,
			padding,
			groups,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Apply grouped convolution without submitting or waiting.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn input_channels(&self) -> usize {
		self.inner.input_channels()
	}

	#[getter]
	pub fn output_channels(&self) -> usize {
		self.inner.output_channels()
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

	#[getter]
	pub fn groups(&self) -> usize {
		self.inner.groups()
	}

	pub fn weight(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.weight(),
		}
	}

	pub fn bias(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.bias(),
		}
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Conv2d(in={}, out={}, kernel={}, stride={}, padding={}, groups={})",
			self.inner.input_channels(),
			self.inner.output_channels(),
			self.inner.kernel_size(),
			self.inner.stride(),
			self.inner.padding(),
			self.inner.groups(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonConv1d>()?;
	module.add_class::<PythonConvTranspose1d>()?;
	module.add_class::<PythonConvTranspose2d>()?;
	module.add_class::<PythonConv2d>()?;
	Ok(())
}
