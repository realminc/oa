use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

/// Trainable FP32 BatchNorm over channels of NCHW matrices.
#[pyclass(name = "BatchNorm2d", unsendable)]
pub(crate) struct PythonBatchNorm2d {
	pub(crate) inner: oa::ml::nn::BatchNorm2d,
}

#[pymethods]
impl PythonBatchNorm2d {
	/// Construct with epsilon `1e-5` and momentum `0.1`, or explicit options.
	#[new]
	#[pyo3(signature = (engine, num_features, epsilon = 1e-5, momentum = 0.1))]
	pub fn new(
		engine: &PythonEngine,
		num_features: usize,
		epsilon: f32,
		momentum: f32,
	) -> PyResult<Self> {
		oa::ml::nn::BatchNorm2d::with_options(&engine.inner, num_features, epsilon, momentum)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Normalize one NCHW Matrix according to the module's current mode.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn num_features(&self) -> usize {
		self.inner.num_features()
	}

	#[getter]
	pub fn epsilon(&self) -> f32 {
		self.inner.epsilon()
	}

	#[getter]
	pub fn momentum(&self) -> f32 {
		self.inner.momentum()
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

	pub fn running_mean(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.running_mean())
	}

	pub fn running_variance(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.running_variance())
	}

	pub fn train(&self) {
		use oa::ml::Module as _;
		self.inner.train(true);
	}

	pub fn eval(&self) {
		use oa::ml::Module as _;
		self.inner.train(false);
	}

	#[getter]
	pub fn is_training(&self) -> bool {
		use oa::ml::Module as _;
		self.inner.is_training()
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
			"BatchNorm2d(num_features={}, epsilon={}, momentum={})",
			self.inner.num_features(),
			self.inner.epsilon(),
			self.inner.momentum(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonBatchNorm2d>()?;
	Ok(())
}
