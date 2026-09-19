use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

/// Trainable FP32 affine projection `[output, input]` weight layout.
#[pyclass(name = "Linear", unsendable)]
pub(crate) struct PythonLinear {
	pub(crate) inner: oa::ml::nn::Linear,
}

#[pymethods]
impl PythonLinear {
	/// Construct a deterministically initialized linear layer.
	///
	/// Accepts `in_features`/`out_features` as aliases for
	/// `input_features`/`output_features`.
	#[new]
	#[pyo3(signature = (engine, input_features=None, output_features=None, seed=0, bias=true, in_features=None, out_features=None))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		engine: &PythonEngine,
		input_features: Option<usize>,
		output_features: Option<usize>,
		seed: u64,
		bias: bool,
		in_features: Option<usize>,
		out_features: Option<usize>,
	) -> PyResult<Self> {
		let in_f = input_features.or(in_features).ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("Linear requires in_features or input_features")
		})?;
		let out_f = output_features.or(out_features).ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("Linear requires out_features or output_features")
		})?;
		oa::ml::nn::Linear::with_seed_and_bias(&engine.inner, in_f, out_f, bias, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Apply this layer to an FP32 matrix with rank at least two.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Make instances callable: `layer(x)` is equivalent to `layer.forward(x)`.
	pub fn __call__(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self.forward(input)
	}

	#[getter]
	pub fn input_features(&self) -> usize {
		self.inner.input_features()
	}

	#[getter]
	pub fn output_features(&self) -> usize {
		self.inner.output_features()
	}

	#[getter]
	pub fn has_bias(&self) -> bool {
		self.inner.has_bias()
	}

	pub fn weight(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.weight(),
		}
	}

	pub fn bias(&self) -> Option<PythonParameter> {
		self.inner.bias().map(|inner| PythonParameter { inner })
	}

	pub fn parameters(&self) -> Vec<PythonParameter> {
		self
			.inner
			.parameters()
			.into_iter()
			.map(|inner| PythonParameter { inner })
			.collect()
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
			"Linear(in={}, out={}, bias={})",
			self.inner.input_features(),
			self.inner.output_features(),
			self.inner.has_bias(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonLinear>()?;
	Ok(())
}
