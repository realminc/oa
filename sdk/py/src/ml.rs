use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

macro_rules! unary_matrix {
	($binding:ident, $operation:path) => {
		#[pyfunction]
		pub(crate) fn $binding(input: &PythonMatrix) -> PyResult<PythonMatrix> {
			$operation(&input.inner)
				.map(PythonMatrix::wrap)
				.map_err(python_error)
		}
	};
}

macro_rules! binary_matrix {
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

unary_matrix!(ml_gelu, oa::ml::matrix::gelu);
unary_matrix!(ml_silu, oa::ml::matrix::silu);
unary_matrix!(ml_relu, oa::ml::matrix::relu);
unary_matrix!(ml_tanh, oa::ml::matrix::tanh);
unary_matrix!(ml_sigmoid, oa::ml::matrix::sigmoid);
unary_matrix!(ml_mish, oa::ml::matrix::mish);
unary_matrix!(ml_softplus, oa::ml::matrix::softplus);
unary_matrix!(ml_detach, oa::ml::matrix::detach);

binary_matrix!(ml_swiglu, oa::ml::matrix::swiglu);
binary_matrix!(ml_loss_smooth_l1, oa::ml::loss::smooth_l1);
binary_matrix!(ml_loss_mse, oa::ml::loss::mse);
binary_matrix!(ml_loss_l1, oa::ml::loss::l1);
binary_matrix!(ml_loss_bce, oa::ml::loss::bce);
binary_matrix!(ml_loss_cross_entropy, oa::ml::loss::cross_entropy);

#[pyfunction]
pub(crate) fn ml_leaky_relu(input: &PythonMatrix, alpha: f32) -> PyResult<PythonMatrix> {
	oa::ml::matrix::leaky_relu(&input.inner, alpha)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_elu(input: &PythonMatrix, alpha: f32) -> PyResult<PythonMatrix> {
	oa::ml::matrix::elu(&input.inner, alpha)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_silu_mul(
	input: &PythonMatrix,
	intermediate_size: usize,
) -> PyResult<PythonMatrix> {
	oa::ml::matrix::silu_mul(&input.inner, intermediate_size)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_loss_masked_cross_entropy(
	logits: &PythonMatrix,
	targets: &PythonMatrix,
	mask: &PythonMatrix,
	valid_count: usize,
) -> PyResult<PythonMatrix> {
	oa::ml::loss::masked_cross_entropy(&logits.inner, &targets.inner, &mask.inner, valid_count)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_metric_scalar_loss(loss: &PythonMatrix) -> PyResult<f32> {
	oa::ml::metric::scalar_loss(&loss.inner).map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_metric_accuracy(logits: &PythonMatrix, labels: &PythonMatrix) -> PyResult<f32> {
	oa::ml::metric::accuracy(&logits.inner, &labels.inner).map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_byte_encode(engine: &PythonEngine, bytes: Vec<u8>) -> PyResult<PythonMatrix> {
	oa::ml::byte::encode(&engine.inner, &bytes)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_byte_encode_batched(
	engine: &PythonEngine,
	bytes: Vec<u8>,
) -> PyResult<PythonMatrix> {
	oa::ml::byte::encode_batched(&engine.inner, &bytes)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_byte_encode_text(engine: &PythonEngine, text: &str) -> PyResult<PythonMatrix> {
	oa::ml::byte::encode_text(&engine.inner, text)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_byte_decode(logits: &PythonMatrix) -> PyResult<Vec<u8>> {
	oa::ml::byte::decode(&logits.inner).map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_byte_decode_text(logits: &PythonMatrix) -> PyResult<String> {
	oa::ml::byte::decode_text(&logits.inner).map_err(python_error)
}

#[pyfunction]
pub(crate) fn ml_byte_sample(
	logits: &PythonMatrix,
	temperature: f32,
	top_k: i32,
	top_p: f32,
	seed: u64,
) -> PyResult<Vec<u8>> {
	oa::ml::byte::sample(&logits.inner, temperature, top_k, top_p, seed).map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	macro_rules! add_functions {
		($($function:ident),+ $(,)?) => { $(module.add_function(wrap_pyfunction!($function, module)?)?;)+ };
	}
	add_functions!(
		ml_gelu,
		ml_silu,
		ml_relu,
		ml_tanh,
		ml_sigmoid,
		ml_leaky_relu,
		ml_elu,
		ml_mish,
		ml_softplus,
		ml_swiglu,
		ml_silu_mul,
		ml_detach,
		ml_loss_smooth_l1,
		ml_loss_mse,
		ml_loss_l1,
		ml_loss_bce,
		ml_loss_cross_entropy,
		ml_loss_masked_cross_entropy,
		ml_metric_scalar_loss,
		ml_metric_accuracy,
		ml_byte_encode,
		ml_byte_encode_batched,
		ml_byte_encode_text,
		ml_byte_decode,
		ml_byte_decode_text,
		ml_byte_sample,
	);
	Ok(())
}
