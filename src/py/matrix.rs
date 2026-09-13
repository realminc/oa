use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{error::python_error, runtime::PythonEngine};

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
	fn from_f32(engine: &PythonEngine, shape: Vec<usize>, values: Vec<f32>) -> PyResult<Self> {
		oa::Matrix::from_slice(&engine.inner, shape, &values)
			.map(Self::wrap)
			.map_err(python_error)
	}

	#[staticmethod]
	fn from_i32(engine: &PythonEngine, shape: Vec<usize>, values: Vec<i32>) -> PyResult<Self> {
		oa::Matrix::from_slice(&engine.inner, shape, &values)
			.map(Self::wrap)
			.map_err(python_error)
	}

	#[staticmethod]
	fn from_u32(engine: &PythonEngine, shape: Vec<usize>, values: Vec<u32>) -> PyResult<Self> {
		oa::Matrix::from_slice(&engine.inner, shape, &values)
			.map(Self::wrap)
			.map_err(python_error)
	}

	#[staticmethod]
	fn from_u8(engine: &PythonEngine, shape: Vec<usize>, values: Vec<u8>) -> PyResult<Self> {
		oa::Matrix::from_slice(&engine.inner, shape, &values)
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

	fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
		self.inner
			.reshape(shape)
			.map(Self::wrap)
			.map_err(python_error)
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

	fn read_u8(&self) -> PyResult<Vec<u8>> {
		self.inner.read::<u8>().map_err(python_error)
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
		matrix_add(self, right)
	}

	fn __sub__(&self, right: &PythonMatrix) -> PyResult<Self> {
		matrix_sub(self, right)
	}

	fn __mul__(&self, right: &PythonMatrix) -> PyResult<Self> {
		matrix_mul(self, right)
	}

	fn __truediv__(&self, right: &PythonMatrix) -> PyResult<Self> {
		matrix_div(self, right)
	}

	fn __neg__(&self) -> PyResult<Self> {
		matrix_neg(self)
	}

	fn __abs__(&self) -> PyResult<Self> {
		matrix_abs(self)
	}

	fn __repr__(&self) -> String {
		format!(
			"Matrix(shape={:?}, dtype='{}')",
			self.inner.shape(),
			self.inner.dtype().token()
		)
	}
}

macro_rules! binary_operation {
	($binding:ident, $operation:path) => {
		#[pyfunction]
		pub(crate) fn $binding(
			left: &PythonMatrix,
			right: &PythonMatrix,
		) -> PyResult<PythonMatrix> {
			$operation(&left.inner, &right.inner)
				.map(PythonMatrix::wrap)
				.map_err(python_error)
		}
	};
}

macro_rules! unary_operation {
	($binding:ident, $operation:path) => {
		#[pyfunction]
		pub(crate) fn $binding(input: &PythonMatrix) -> PyResult<PythonMatrix> {
			$operation(&input.inner)
				.map(PythonMatrix::wrap)
				.map_err(python_error)
		}
	};
}

macro_rules! scalar_operation {
	($binding:ident, $operation:path) => {
		#[pyfunction]
		pub(crate) fn $binding(input: &PythonMatrix, scalar: f32) -> PyResult<PythonMatrix> {
			$operation(&input.inner, scalar)
				.map(PythonMatrix::wrap)
				.map_err(python_error)
		}
	};
}

binary_operation!(matrix_add, oa::matrix::add);
binary_operation!(matrix_sub, oa::matrix::sub);
binary_operation!(matrix_mul, oa::matrix::mul);
binary_operation!(matrix_div, oa::matrix::div);
binary_operation!(matrix_mat_mul_nt, oa::matrix::mat_mul_nt);
binary_operation!(matrix_gather, oa::matrix::gather);
binary_operation!(matrix_gather_last_dim, oa::matrix::gather_last_dim);

unary_operation!(matrix_neg, oa::matrix::neg);
unary_operation!(matrix_abs, oa::matrix::abs);
unary_operation!(matrix_log, oa::matrix::log);
unary_operation!(matrix_sqrt, oa::matrix::sqrt);
unary_operation!(matrix_exp, oa::matrix::exp);
unary_operation!(matrix_sin, oa::matrix::sin);
unary_operation!(matrix_cos, oa::matrix::cos);
unary_operation!(matrix_reciprocal, oa::matrix::reciprocal);
unary_operation!(matrix_copy, oa::matrix::copy);

scalar_operation!(matrix_scale, oa::matrix::scale);
scalar_operation!(matrix_pow, oa::matrix::pow);
scalar_operation!(matrix_add_scalar, oa::matrix::add_scalar);
scalar_operation!(matrix_sub_scalar, oa::matrix::sub_scalar);
scalar_operation!(matrix_div_scalar, oa::matrix::div_scalar);
scalar_operation!(matrix_clamp_max, oa::matrix::clamp_max);
scalar_operation!(matrix_clamp_min, oa::matrix::clamp_min);
scalar_operation!(matrix_equal, oa::matrix::equal);

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

#[pyfunction]
pub(crate) fn matrix_reshape(input: &PythonMatrix, shape: Vec<usize>) -> PyResult<PythonMatrix> {
	oa::matrix::reshape(&input.inner, shape)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_softmax(input: &PythonMatrix, dim: i32) -> PyResult<PythonMatrix> {
	oa::matrix::softmax(&input.inner, dim)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_log_softmax(input: &PythonMatrix, dim: i32) -> PyResult<PythonMatrix> {
	oa::matrix::log_softmax(&input.inner, dim)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_sum(input: &PythonMatrix, dim: i32) -> PyResult<PythonMatrix> {
	oa::matrix::sum(&input.inner, dim)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_transpose(
	input: &PythonMatrix,
	dim0: i32,
	dim1: i32,
) -> PyResult<PythonMatrix> {
	oa::matrix::transpose(&input.inner, dim0, dim1)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_slice(
	input: &PythonMatrix,
	dim: i32,
	start: i64,
	end: i64,
) -> PyResult<PythonMatrix> {
	oa::matrix::slice(&input.inner, dim, start, end)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_repeat_interleave(
	input: &PythonMatrix,
	repeats: usize,
	dim: i32,
) -> PyResult<PythonMatrix> {
	oa::matrix::repeat_interleave(&input.inner, repeats, dim)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_concat(
	inputs: Vec<PyRef<'_, PythonMatrix>>,
	dim: i32,
) -> PyResult<PythonMatrix> {
	let matrices = inputs
		.iter()
		.map(|input| input.inner.clone())
		.collect::<Vec<_>>();
	oa::matrix::concat(&matrices, dim)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_top_k(
	input: &PythonMatrix,
	k: i32,
	dim: i32,
) -> PyResult<(PythonMatrix, PythonMatrix)> {
	oa::matrix::top_k(&input.inner, k, dim)
		.map(|result| {
			(
				PythonMatrix::wrap(result.values),
				PythonMatrix::wrap(result.indices),
			)
		})
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_philox_uniform(
	input: &PythonMatrix,
	low: f32,
	high: f32,
	seed: u64,
) -> PyResult<PythonMatrix> {
	oa::matrix::philox_uniform(&input.inner, low, high, seed)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_philox_normal(
	input: &PythonMatrix,
	mean: f32,
	stddev: f32,
	seed: u64,
) -> PyResult<PythonMatrix> {
	oa::matrix::philox_normal(&input.inner, mean, stddev, seed)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_dropout(
	input: &PythonMatrix,
	probability: f32,
	seed: u64,
) -> PyResult<PythonMatrix> {
	oa::matrix::dropout(&input.inner, probability, seed)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_sample_logits(
	input: &PythonMatrix,
	temperature: f32,
	top_k: i32,
	top_p: f32,
	seed: u64,
) -> PyResult<PythonMatrix> {
	oa::matrix::sample_logits(&input.inner, temperature, top_k, top_p, seed)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn matrix_set_rng_seed(seed: u64) {
	oa::matrix::set_rng_seed(seed);
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonMatrix>()?;
	macro_rules! add_functions {
		($($function:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($function, module)?)?;)+ };
	}
	add_functions!(
		matrix_ones,
		matrix_full,
		matrix_reshape,
		matrix_add,
		matrix_sub,
		matrix_mul,
		matrix_div,
		matrix_scale,
		matrix_neg,
		matrix_abs,
		matrix_log,
		matrix_sqrt,
		matrix_pow,
		matrix_add_scalar,
		matrix_sub_scalar,
		matrix_div_scalar,
		matrix_exp,
		matrix_sin,
		matrix_cos,
		matrix_reciprocal,
		matrix_clamp_max,
		matrix_clamp_min,
		matrix_copy,
		matrix_mat_mul_nt,
		matrix_softmax,
		matrix_log_softmax,
		matrix_sum,
		matrix_gather,
		matrix_gather_last_dim,
		matrix_equal,
		matrix_top_k,
		matrix_repeat_interleave,
		matrix_concat,
		matrix_transpose,
		matrix_slice,
		matrix_philox_uniform,
		matrix_philox_normal,
		matrix_sample_logits,
		matrix_dropout,
		matrix_set_rng_seed,
	);
	Ok(())
}
