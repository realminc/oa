use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{error::python_error, runtime::PythonEngine};

#[path = "matrix/matrix.gen.rs"]
mod generated;

#[pyclass(name = "Matrix", unsendable)]
pub(crate) struct PythonMatrix {
	pub(crate) inner: oa::Matrix,
}

impl PythonMatrix {
	pub(crate) fn wrap(inner: oa::Matrix) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonMatrix {
	#[staticmethod]
	fn new_from_f32(engine: &PythonEngine, shape: Vec<usize>, values: Vec<f32>) -> PyResult<Self> {
		oa::Matrix::new_from_slice(&engine.inner, shape, &values)
			.map(Self::wrap)
			.map_err(python_error)
	}

	#[staticmethod]
	fn new_from_i32(engine: &PythonEngine, shape: Vec<usize>, values: Vec<i32>) -> PyResult<Self> {
		oa::Matrix::new_from_slice(&engine.inner, shape, &values)
			.map(Self::wrap)
			.map_err(python_error)
	}

	#[staticmethod]
	fn new_from_u32(engine: &PythonEngine, shape: Vec<usize>, values: Vec<u32>) -> PyResult<Self> {
		oa::Matrix::new_from_slice(&engine.inner, shape, &values)
			.map(Self::wrap)
			.map_err(python_error)
	}

	#[staticmethod]
	fn new_from_u8(engine: &PythonEngine, shape: Vec<usize>, values: Vec<u8>) -> PyResult<Self> {
		oa::Matrix::new_from_slice(&engine.inner, shape, &values)
			.map(Self::wrap)
			.map_err(python_error)
	}

	#[getter]
	fn shape(&self) -> Vec<usize> {
		self.inner.shape().to_vec()
	}

	#[getter]
	fn dtype(&self) -> &'static str {
		self.inner.dtype().token()
	}

	#[getter]
	fn num_elements(&self) -> usize {
		self.inner.num_elements()
	}

	fn read_f32(&self) -> PyResult<Vec<f32>> {
		self.inner.read::<f32>().map_err(python_error)
	}

	fn read_i32(&self) -> PyResult<Vec<i32>> {
		self.inner.read::<i32>().map_err(python_error)
	}

	fn read_u32(&self) -> PyResult<Vec<u32>> {
		self.inner.read::<u32>().map_err(python_error)
	}

	fn read_u8(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyList>> {
		let bytes = self.inner.read::<u8>().map_err(python_error)?;
		let list = pyo3::types::PyList::new(py, bytes.iter().map(|&b| b as u32))?;
		Ok(list.unbind())
	}

	fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
		let values = match self.inner.dtype() {
			oa::DType::F32 => self
				.inner
				.read::<f32>()
				.map_err(python_error)?
				.into_pyobject(py)?
				.into_any()
				.unbind(),
			oa::DType::I32 => self
				.inner
				.read::<i32>()
				.map_err(python_error)?
				.into_pyobject(py)?
				.into_any()
				.unbind(),
			oa::DType::U32 => self
				.inner
				.read::<u32>()
				.map_err(python_error)?
				.into_pyobject(py)?
				.into_any()
				.unbind(),
			oa::DType::U8 => self
				.inner
				.read::<u8>()
				.map_err(python_error)?
				.into_pyobject(py)?
				.into_any()
				.unbind(),
			_ => {
				return Err(PyRuntimeError::new_err(format!(
					"to_list does not support dtype {}",
					self.inner.dtype().token()
				)));
			}
		};
		Ok(values)
	}

	fn __add__(&self, right: &PythonMatrix) -> PyResult<Self> {
		self.add(right)
	}

	fn __sub__(&self, right: &PythonMatrix) -> PyResult<Self> {
		self.sub(right)
	}

	fn __mul__(&self, right: &PythonMatrix) -> PyResult<Self> {
		self.mul(right)
	}

	fn __matmul__(&self, right: &PythonMatrix) -> PyResult<Self> {
		self.mat_mul_nt(right)
	}

	fn __truediv__(&self, right: &PythonMatrix) -> PyResult<Self> {
		self.div(right)
	}

	fn __neg__(&self) -> PyResult<Self> {
		self.neg()
	}

	fn __abs__(&self) -> PyResult<Self> {
		self.abs()
	}

	fn __repr__(&self) -> String {
		format!(
			"Matrix(shape={:?}, dtype='{}')",
			self.inner.shape(),
			self.inner.dtype().token()
		)
	}
}

#[pyfunction]
pub(crate) fn matrix_ones(engine: &PythonEngine, shape: Vec<usize>) -> PyResult<PythonMatrix> {
	oa::matrix::ones(&engine.inner, shape)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_full(
	engine: &PythonEngine,
	shape: Vec<usize>,
	value: f32,
) -> PyResult<PythonMatrix> {
	oa::matrix::full(&engine.inner, shape, value)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

// MoeExpertPlan result type
#[pyclass(name = "MoeExpertPlan", unsendable)]
pub(crate) struct PythonMoeExpertPlan {
	pub counts: PythonMatrix,
	pub offsets: PythonMatrix,
	pub packed_token: PythonMatrix,
	pub packed_expert: PythonMatrix,
	pub packed_slot: PythonMatrix,
	pub inverse: PythonMatrix,
}

#[pymethods]
impl PythonMoeExpertPlan {
	#[getter]
	fn counts(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.counts.inner.clone())
	}

	#[getter]
	fn offsets(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.offsets.inner.clone())
	}

	#[getter]
	fn packed_token(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.packed_token.inner.clone())
	}

	#[getter]
	fn packed_expert(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.packed_expert.inner.clone())
	}

	#[getter]
	fn packed_slot(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.packed_slot.inner.clone())
	}

	#[getter]
	fn inverse(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inverse.inner.clone())
	}
}

// CompactRowsResult result type
#[pyclass(name = "CompactRowsResult", unsendable)]
pub(crate) struct PythonCompactRowsResult {
	pub values: PythonMatrix,
	pub row_map: PythonMatrix,
	pub count: PythonMatrix,
	pub dispatch_args: PythonMatrix,
}

#[pymethods]
impl PythonCompactRowsResult {
	#[getter]
	fn values(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.values.inner.clone())
	}

	#[getter]
	fn row_map(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.row_map.inner.clone())
	}

	#[getter]
	fn count(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.count.inner.clone())
	}

	#[getter]
	fn dispatch_args(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.dispatch_args.inner.clone())
	}
}

#[pyfunction]
pub(crate) fn matrix_set_rng_seed(seed: u64) {
	oa::matrix::set_rng_seed(seed);
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonMatrix>()?;
	module.add_class::<PythonMoeExpertPlan>()?;
	module.add_class::<PythonCompactRowsResult>()?;
	macro_rules! add_functions {
		($($function:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($function, module)?)?;)+ };
	}
	generated::register(module)?;
	add_functions!(matrix_ones, matrix_full, matrix_set_rng_seed,);
	Ok(())
}
