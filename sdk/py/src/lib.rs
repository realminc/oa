use pyo3::{exceptions::PyRuntimeError, prelude::*};

fn python_error(error: oa::Error) -> PyErr {
	PyRuntimeError::new_err(error.to_string())
}

#[pyclass(name = "Engine", unsendable)]
struct PythonEngine {
	inner: oa::Engine,
}

#[pymethods]
impl PythonEngine {
	#[new]
	fn new() -> PyResult<Self> {
		Ok(Self {
			inner: oa::Engine::new().map_err(python_error)?,
		})
	}

	fn _ones(&self, shape: Vec<usize>) -> PyResult<PythonMatrix> {
		Ok(PythonMatrix {
			inner: oa::matrix::ones(&self.inner, shape).map_err(python_error)?,
		})
	}

	fn _full(&self, shape: Vec<usize>, value: f32) -> PyResult<PythonMatrix> {
		Ok(PythonMatrix {
			inner: oa::matrix::full(&self.inner, shape, value).map_err(python_error)?,
		})
	}

	fn __repr__(&self) -> &'static str {
		"Engine(backend='vulkan')"
	}
}

#[pyclass(name = "Matrix", unsendable)]
struct PythonMatrix {
	inner: oa::Matrix,
}

#[pymethods]
impl PythonMatrix {
	#[staticmethod]
	fn from_f32(engine: &PythonEngine, shape: Vec<usize>, values: Vec<f32>) -> PyResult<Self> {
		Ok(Self {
			inner: oa::Matrix::from_f32(&engine.inner, shape, &values).map_err(python_error)?,
		})
	}

	#[getter]
	fn shape(&self) -> Vec<usize> {
		self.inner.shape().to_vec()
	}

	#[getter]
	fn dtype(&self) -> &'static str {
		self.inner.dtype().token()
	}

	fn _add(&self, right: &PythonMatrix) -> PyResult<Self> {
		Ok(Self {
			inner: oa::matrix::add(&self.inner, &right.inner).map_err(python_error)?,
		})
	}

	fn _mat_mul_nt(&self, right: &PythonMatrix) -> PyResult<Self> {
		Ok(Self {
			inner: oa::matrix::mat_mul_nt(&self.inner, &right.inner).map_err(python_error)?,
		})
	}

	fn to_list(&self) -> PyResult<Vec<f32>> {
		self.inner.read_f32().map_err(python_error)
	}

	fn read_f32(&self) -> PyResult<Vec<f32>> {
		self.inner.read_f32().map_err(python_error)
	}

	fn __add__(&self, right: &PythonMatrix) -> PyResult<Self> {
		self._add(right)
	}

	fn __repr__(&self) -> String {
		format!(
			"Matrix(shape={:?}, dtype='{}')",
			self.inner.shape(),
			self.inner.dtype().token()
		)
	}
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonEngine>()?;
	module.add_class::<PythonMatrix>()?;
	Ok(())
}
