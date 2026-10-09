use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

use super::super::autograd::PythonParameter;

/// Ordered sequential module composition — applies registered children in insertion order.
///
/// Can be constructed from an empty state (`Sequential()`) or from a Python
/// list of `Linear` layers (`Sequential([l1, l2])`).
#[pyclass(name = "Sequential", unsendable)]
pub(crate) struct PythonSequential {
	pub(crate) inner: oa::ml::nn::Sequential,
	/// Python-side layer objects for the list-constructor path.
	py_layers: Vec<Py<PyAny>>,
}

#[pymethods]
impl PythonSequential {
	/// Construct an empty sequence, or a sequence from a Python list of layers.
	///
	/// When passed a list the layers are stored as Python objects and applied
	/// via their `forward` method.  Any layer with a `forward(Matrix) -> Matrix`
	/// method is accepted.
	#[new]
	#[pyo3(signature = (layers=None))]
	pub fn new(py: Python<'_>, layers: Option<&Bound<'_, pyo3::types::PyList>>) -> PyResult<Self> {
		let py_layers = match layers {
			None => vec![],
			Some(lst) => lst.iter().map(|item| item.clone().unbind()).collect(),
		};
		let _ = py;
		Ok(Self {
			inner: oa::ml::nn::Sequential::new(),
			py_layers,
		})
	}

	/// Apply all registered children in insertion order.
	pub fn forward(&self, py: Python<'_>, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		if !self.py_layers.is_empty() {
			// Python-list path: chain calls to each layer's forward/call method.
			let mut current = input.inner.clone();
			for layer in &self.py_layers {
				let layer_bound = layer.bind(py);
				let mat = PythonMatrix::wrap(current);
				let result = if layer_bound.hasattr("forward")? {
					layer_bound.call_method1("forward", (mat,))?
				} else {
					layer_bound.call1((mat,))?
				};
				let out: PyRef<PythonMatrix> = result.extract::<PyRef<PythonMatrix>>().map_err(|e| {
					pyo3::exceptions::PyTypeError::new_err(format!(
						"Sequential layer did not return a Matrix: {e}"
					))
				})?;
				current = out.inner.clone();
			}
			Ok(PythonMatrix::wrap(current))
		} else {
			self
				.inner
				.forward(&input.inner)
				.map(PythonMatrix::wrap)
				.map_err(python_error)
		}
	}

	pub fn __call__(&self, py: Python<'_>, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self.forward(py, input)
	}

	/// Return the number of children currently registered.
	#[getter]
	pub fn len(&self) -> usize {
		if self.py_layers.is_empty() {
			self.inner.len()
		} else {
			self.py_layers.len()
		}
	}

	/// Return whether no children have been registered.
	#[getter]
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Propagate train mode to all children.
	pub fn train(&self) {
		use oa::ml::Module as _;
		self.inner.train(true);
	}

	/// Propagate eval mode to all children.
	pub fn eval(&self) {
		use oa::ml::Module as _;
		self.inner.train(false);
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
		format!("Sequential(layers={})", self.len())
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonSequential>()?;
	Ok(())
}
