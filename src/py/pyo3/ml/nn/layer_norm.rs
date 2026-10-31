use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

/// Trainable FP32 LayerNorm over the final input dimension.
#[pyclass(name = "LayerNorm", unsendable)]
pub(crate) struct PythonLayerNorm {
	pub(crate) inner: oa::ml::nn::LayerNorm,
}

#[pymethods]
impl PythonLayerNorm {
	/// Construct a LayerNorm with unit weight and zero bias.
	///
	/// `normalized_shape` may be passed as an integer or a one-element list
	/// `[n]`.  The `seed` keyword is accepted for API compatibility but ignored
	/// (LayerNorm initializes deterministically to unit weight / zero bias).
	#[new]
	#[pyo3(signature = (engine, normalized_shape=None, epsilon=1e-5, seed=None))]
	pub fn new(
		engine: &PythonEngine,
		normalized_shape: Option<pyo3::Bound<'_, pyo3::PyAny>>,
		epsilon: f32,
		seed: Option<u64>,
	) -> PyResult<Self> {
		let _ = seed;
		let shape: usize = match normalized_shape {
			None => {
				return Err(pyo3::exceptions::PyTypeError::new_err(
					"LayerNorm requires normalized_shape",
				));
			}
			Some(ref v) => {
				if let Ok(n) = v.extract::<usize>() {
					n
				} else if let Ok(lst) = v.extract::<Vec<usize>>() {
					if lst.len() != 1 {
						return Err(pyo3::exceptions::PyValueError::new_err(
							"LayerNorm normalized_shape list must have exactly one element",
						));
					}
					lst[0]
				} else {
					return Err(pyo3::exceptions::PyTypeError::new_err(
						"LayerNorm normalized_shape must be an int or [int]",
					));
				}
			}
		};
		oa::ml::nn::LayerNorm::new(&engine.inner, shape, epsilon)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Normalize each row over the final dimension and apply affine parameters.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn __call__(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self.forward(input)
	}

	/// Normalize channels in `[B, C, T]` storage.
	pub fn forward_channel(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_channel(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Normalize channels in `[B, C, T]` storage and fuse ReLU.
	pub fn forward_channel_relu(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_channel_relu(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn normalized_shape(&self) -> usize {
		self.inner.normalized_shape()
	}

	#[getter]
	pub fn epsilon(&self) -> f32 {
		self.inner.epsilon()
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

	pub fn parameters(&self) -> Vec<PythonParameter> {
		self
			.inner
			.parameters()
			.into_iter()
			.map(|inner| PythonParameter { inner })
			.collect()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"LayerNorm(normalized_shape={}, epsilon={})",
			self.inner.normalized_shape(),
			self.inner.epsilon(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonLayerNorm>()?;
	Ok(())
}
