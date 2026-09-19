use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

// ── UpsampleMode ──────────────────────────────────────────────────────────────

/// Interpolation algorithm for spatial upsampling.
#[pyclass(name = "UpsampleMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonUpsampleMode {
	Nearest = 0,
	Bilinear = 1,
}

impl From<PythonUpsampleMode> for oa::ml::nn::UpsampleMode {
	fn from(value: PythonUpsampleMode) -> Self {
		match value {
			PythonUpsampleMode::Nearest => oa::ml::nn::UpsampleMode::Nearest,
			PythonUpsampleMode::Bilinear => oa::ml::nn::UpsampleMode::Bilinear,
		}
	}
}

impl From<oa::ml::nn::UpsampleMode> for PythonUpsampleMode {
	fn from(value: oa::ml::nn::UpsampleMode) -> Self {
		match value {
			oa::ml::nn::UpsampleMode::Nearest => PythonUpsampleMode::Nearest,
			oa::ml::nn::UpsampleMode::Bilinear => PythonUpsampleMode::Bilinear,
			_ => PythonUpsampleMode::Bilinear,
		}
	}
}

// ── Upsample ──────────────────────────────────────────────────────────────────

/// Parameterless spatial upsampling module for FP32 NCHW matrices.
#[pyclass(name = "Upsample", unsendable)]
pub(crate) struct PythonUpsample {
	inner: oa::ml::nn::Upsample,
}

#[pymethods]
impl PythonUpsample {
	/// Construct with an integer scale factor and optional interpolation mode (default: bilinear).
	#[new]
	#[pyo3(signature = (scale_factor, mode = PythonUpsampleMode::Bilinear))]
	pub fn new(scale_factor: usize, mode: PythonUpsampleMode) -> PyResult<Self> {
		oa::ml::nn::Upsample::with_mode(scale_factor, mode.into())
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Upsample one NCHW Matrix without submitting or waiting.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn scale_factor(&self) -> usize {
		self.inner.scale_factor()
	}

	#[getter]
	pub fn mode(&self) -> PythonUpsampleMode {
		self.inner.mode().into()
	}

	pub fn __repr__(&self) -> String {
		let mode = match self.inner.mode() {
			oa::ml::nn::UpsampleMode::Nearest => "nearest",
			oa::ml::nn::UpsampleMode::Bilinear => "bilinear",
			_ => "unknown",
		};
		format!(
			"Upsample(scale_factor={}, mode={mode:?})",
			self.inner.scale_factor()
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonUpsampleMode>()?;
	module.add_class::<PythonUpsample>()?;
	Ok(())
}
