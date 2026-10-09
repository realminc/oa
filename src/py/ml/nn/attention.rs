use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::transformer::PythonAttentionMode;

// ── AttentionBackend ──────────────────────────────────────────────────────────

/// Execution policy for scaled dot-product attention.
#[pyclass(name = "AttentionBackend", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonAttentionBackend {
	Auto = 0,
	Standard = 1,
	Flash = 2,
}

impl From<PythonAttentionBackend> for oa::ml::nn::AttentionBackend {
	fn from(value: PythonAttentionBackend) -> Self {
		match value {
			PythonAttentionBackend::Auto => oa::ml::nn::AttentionBackend::Auto,
			PythonAttentionBackend::Standard => oa::ml::nn::AttentionBackend::Standard,
			PythonAttentionBackend::Flash => oa::ml::nn::AttentionBackend::Flash,
		}
	}
}

impl From<oa::ml::nn::AttentionBackend> for PythonAttentionBackend {
	fn from(value: oa::ml::nn::AttentionBackend) -> Self {
		match value {
			oa::ml::nn::AttentionBackend::Auto => PythonAttentionBackend::Auto,
			oa::ml::nn::AttentionBackend::Standard => PythonAttentionBackend::Standard,
			oa::ml::nn::AttentionBackend::Flash => PythonAttentionBackend::Flash,
		}
	}
}

// ── MultiHeadAttention ────────────────────────────────────────────────────────

/// Multi-head self-attention over packed `[B*S, D]` values.
#[pyclass(name = "MultiHeadAttention", unsendable)]
pub(crate) struct PythonMultiHeadAttention {
	pub(crate) inner: oa::ml::nn::MultiHeadAttention,
}

#[pymethods]
impl PythonMultiHeadAttention {
	/// Construct four biased projections with causal visibility and automatic backend.
	#[new]
	#[pyo3(signature = (engine, model_width, num_heads, sequence_length, dropout_probability = 0.0, bias = true, seed = 0))]
	pub fn new(
		engine: &PythonEngine,
		model_width: usize,
		num_heads: usize,
		sequence_length: usize,
		dropout_probability: f32,
		bias: bool,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::MultiHeadAttention::with_seed_and_options(
			&engine.inner,
			model_width,
			num_heads,
			sequence_length,
			dropout_probability,
			bias,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Apply self-attention using the configured visibility and provider policy.
	pub fn forward(&self, input: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&input.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Apply standard bidirectional attention with an additive `[B*H*S, S]` mask.
	pub fn forward_masked(
		&self,
		input: &PythonMatrix,
		additive_mask: &PythonMatrix,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_masked(&input.inner, &additive_mask.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn set_backend(&self, backend: PythonAttentionBackend) {
		self.inner.set_backend(backend.into());
	}

	pub fn backend(&self) -> PythonAttentionBackend {
		self.inner.backend().into()
	}

	pub fn last_backend(&self) -> PythonAttentionBackend {
		self.inner.last_backend().into()
	}

	pub fn set_mode(&self, mode: PythonAttentionMode) {
		self.inner.set_mode(mode.into());
	}

	pub fn mode(&self) -> PythonAttentionMode {
		self.inner.mode().into()
	}

	#[getter]
	pub fn model_width(&self) -> usize {
		self.inner.model_width()
	}

	#[getter]
	pub fn num_heads(&self) -> usize {
		self.inner.num_heads()
	}

	#[getter]
	pub fn sequence_length(&self) -> usize {
		self.inner.sequence_length()
	}

	pub fn set_sequence_length(&self, sequence_length: usize) -> PyResult<()> {
		self
			.inner
			.set_sequence_length(sequence_length)
			.map_err(python_error)
	}

	#[getter]
	pub fn has_bias(&self) -> bool {
		self.inner.has_bias()
	}

	#[getter]
	pub fn dropout_probability(&self) -> f32 {
		self.inner.dropout_probability()
	}

	pub fn all_parameters(&self) -> PyResult<Vec<crate::ml::autograd::PythonParameter>> {
		use oa::ml::Module as _;
		self
			.inner
			.all_parameters()
			.map(|params| {
				params
					.into_iter()
					.map(|inner| crate::ml::autograd::PythonParameter { inner })
					.collect()
			})
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"MultiHeadAttention(model_width={}, num_heads={}, seq_len={}, bias={}, dropout={})",
			self.inner.model_width(),
			self.inner.num_heads(),
			self.inner.sequence_length(),
			self.inner.has_bias(),
			self.inner.dropout_probability(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonAttentionBackend>()?;
	module.add_class::<PythonMultiHeadAttention>()?;
	Ok(())
}
