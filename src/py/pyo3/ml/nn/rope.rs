use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

/// Llama-style split-half rotary position embedding module.
#[pyclass(name = "Rope", unsendable)]
pub(crate) struct PythonRope {
	inner: oa::ml::nn::Rope,
}

#[pymethods]
impl PythonRope {
	/// Construct a parameterless rotary-position module.
	///
	/// Accepts `dim`/`seq_len` convenience aliases:
	/// - `dim` → treats `num_heads=1, head_dim=dim`
	/// - `seq_len` is accepted but ignored (RoPE is position-agnostic at construction)
	/// `head_dim` must be even and `theta_base` must be finite and positive.
	#[new]
	#[pyo3(signature = (num_heads=None, head_dim=None, theta_base=10000.0, dim=None, seq_len=None))]
	pub fn new(
		num_heads: Option<usize>,
		head_dim: Option<usize>,
		theta_base: f32,
		dim: Option<usize>,
		seq_len: Option<usize>,
	) -> PyResult<Self> {
		let _ = seq_len;
		// Allow `Rope(dim=d)` as shorthand for `Rope(num_heads=1, head_dim=d)`.
		let (nh, hd) = match (num_heads, head_dim, dim) {
			(Some(nh), Some(hd), _) => (nh, hd),
			(None, None, Some(d)) => (1, d),
			_ => {
				return Err(pyo3::exceptions::PyTypeError::new_err(
					"Rope requires (num_heads, head_dim) or dim",
				));
			}
		};
		oa::ml::nn::Rope::new(nh, hd, theta_base)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Rotate a sequence beginning at absolute position zero.
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

	#[getter]
	pub fn num_heads(&self) -> usize {
		self.inner.num_heads()
	}

	#[getter]
	pub fn head_dim(&self) -> usize {
		self.inner.head_dim()
	}

	#[getter]
	pub fn theta_base(&self) -> f32 {
		self.inner.theta_base()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"Rope(num_heads={}, head_dim={}, theta_base={})",
			self.inner.num_heads(),
			self.inner.head_dim(),
			self.inner.theta_base(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonRope>()?;
	Ok(())
}
