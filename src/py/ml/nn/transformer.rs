use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

// ── AttentionMode ─────────────────────────────────────────────────────────────

/// Token-visibility contract for self-attention.
#[pyclass(name = "AttentionMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PythonAttentionMode {
	Causal = 0,
	Bidirectional = 1,
}

impl From<PythonAttentionMode> for oa::ml::nn::AttentionMode {
	fn from(value: PythonAttentionMode) -> Self {
		match value {
			PythonAttentionMode::Causal => oa::ml::nn::AttentionMode::Causal,
			PythonAttentionMode::Bidirectional => oa::ml::nn::AttentionMode::Bidirectional,
		}
	}
}

impl From<oa::ml::nn::AttentionMode> for PythonAttentionMode {
	fn from(value: oa::ml::nn::AttentionMode) -> Self {
		match value {
			oa::ml::nn::AttentionMode::Causal => PythonAttentionMode::Causal,
			oa::ml::nn::AttentionMode::Bidirectional => PythonAttentionMode::Bidirectional,
		}
	}
}

// ── TransformerBlock ──────────────────────────────────────────────────────────

/// One pre-normalized causal Transformer block.
#[pyclass(name = "TransformerBlock", unsendable)]
pub(crate) struct PythonTransformerBlock {
	pub(crate) inner: oa::ml::nn::TransformerBlock,
}

#[pymethods]
impl PythonTransformerBlock {
	/// Construct the canonical pre-norm attention and GELU feed-forward block.
	///
	/// Accepts `dim`/`heads`/`seq_len` as aliases for
	/// `model_width`/`num_heads`/`sequence_length`.
	#[new]
	#[pyo3(signature = (engine, model_width=None, hidden_width=None, sequence_length=None, num_heads=None, epsilon=1e-5, seed=0, dim=None, heads=None, seq_len=None))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		engine: &PythonEngine,
		model_width: Option<usize>,
		hidden_width: Option<usize>,
		sequence_length: Option<usize>,
		num_heads: Option<usize>,
		epsilon: f32,
		seed: u64,
		dim: Option<usize>,
		heads: Option<usize>,
		seq_len: Option<usize>,
	) -> PyResult<Self> {
		let mw = model_width.or(dim).ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("TransformerBlock requires model_width or dim")
		})?;
		let nh = num_heads.or(heads).ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("TransformerBlock requires num_heads or heads")
		})?;
		let sl = sequence_length.or(seq_len).ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("TransformerBlock requires sequence_length or seq_len")
		})?;
		let hw = hidden_width.unwrap_or(mw * 4);
		oa::ml::nn::TransformerBlock::with_seed(&engine.inner, mw, hw, sl, nh, epsilon, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Construct a block with zero-initialized adaptive conditioning.
	#[staticmethod]
	#[pyo3(signature = (engine, model_width, hidden_width, sequence_length, num_heads, condition_dim, epsilon=1e-5, seed=0))]
	#[allow(clippy::too_many_arguments)]
	pub fn with_conditioning(
		engine: &PythonEngine,
		model_width: usize,
		hidden_width: usize,
		sequence_length: usize,
		num_heads: usize,
		condition_dim: usize,
		epsilon: f32,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::TransformerBlock::with_seed_conditioned(
			&engine.inner,
			model_width,
			hidden_width,
			sequence_length,
			num_heads,
			condition_dim,
			epsilon,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Construct a pre-norm attention block with a sparse MoE residual.
	#[staticmethod]
	#[pyo3(signature = (engine, model_width, expert_hidden_width, sequence_length, num_heads, num_experts, experts_per_token, epsilon=1e-5, seed=0))]
	#[allow(clippy::too_many_arguments)]
	pub fn with_moe(
		engine: &PythonEngine,
		model_width: usize,
		expert_hidden_width: usize,
		sequence_length: usize,
		num_heads: usize,
		num_experts: usize,
		experts_per_token: usize,
		epsilon: f32,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::TransformerBlock::with_seed_moe(
			&engine.inner,
			model_width,
			expert_hidden_width,
			sequence_length,
			num_heads,
			num_experts,
			experts_per_token,
			epsilon,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Apply pre-norm attention/residual followed by pre-norm GELU FFN/residual.
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

	/// Apply this block with an explicit bidirectional additive attention mask.
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

	/// Apply donor AdaLN-Zero conditioning without an explicit attention mask.
	pub fn forward_conditioned(
		&self,
		input: &PythonMatrix,
		condition: &PythonMatrix,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_conditioned(&input.inner, &condition.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Apply donor AdaLN-Zero conditioning with an additive attention mask.
	pub fn forward_conditioned_masked(
		&self,
		input: &PythonMatrix,
		condition: &PythonMatrix,
		additive_mask: &PythonMatrix,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_conditioned_masked(&input.inner, &condition.inner, &additive_mask.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn model_width(&self) -> usize {
		self.inner.model_width()
	}

	#[getter]
	pub fn hidden_width(&self) -> usize {
		self.inner.hidden_width()
	}

	#[getter]
	pub fn sequence_length(&self) -> usize {
		self.inner.sequence_length()
	}

	#[getter]
	pub fn num_heads(&self) -> usize {
		self.inner.num_heads()
	}

	#[getter]
	pub fn is_moe(&self) -> bool {
		self.inner.is_moe()
	}

	#[getter]
	pub fn is_adaptively_conditioned(&self) -> bool {
		self.inner.is_adaptively_conditioned()
	}

	#[getter]
	pub fn condition_dim(&self) -> Option<usize> {
		self.inner.condition_dim()
	}

	pub fn set_attention_mode(&self, mode: PythonAttentionMode) {
		self.inner.set_attention_mode(mode.into());
	}

	pub fn attention_mode(&self) -> PythonAttentionMode {
		self.inner.attention_mode().into()
	}

	pub fn set_sequence_length(&self, sequence_length: usize) -> PyResult<()> {
		self
			.inner
			.set_sequence_length(sequence_length)
			.map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"TransformerBlock(model_width={}, hidden_width={}, seq_len={}, num_heads={}, moe={})",
			self.inner.model_width(),
			self.inner.hidden_width(),
			self.inner.sequence_length(),
			self.inner.num_heads(),
			self.inner.is_moe(),
		)
	}
}

// ── Transformer ───────────────────────────────────────────────────────────────

/// Ready-to-train causal Transformer language model.
#[pyclass(name = "Transformer", unsendable)]
pub(crate) struct PythonTransformer {
	pub(crate) inner: oa::ml::nn::Transformer,
}

#[pymethods]
impl PythonTransformer {
	/// Construct a deterministic causal language model.
	#[new]
	#[pyo3(signature = (engine, vocab_size, context_length, model_width, hidden_width, num_layers, num_heads, epsilon=1e-5, seed=0))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		engine: &PythonEngine,
		vocab_size: usize,
		context_length: usize,
		model_width: usize,
		hidden_width: usize,
		num_layers: usize,
		num_heads: usize,
		epsilon: f32,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::Transformer::with_seed(
			&engine.inner,
			vocab_size,
			context_length,
			model_width,
			hidden_width,
			num_layers,
			num_heads,
			epsilon,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Construct a deterministic causal language model with MoE Transformer blocks.
	#[staticmethod]
	#[pyo3(signature = (engine, vocab_size, context_length, model_width, expert_hidden_width, num_layers, num_heads, num_experts, experts_per_token, epsilon=1e-5, seed=0))]
	#[allow(clippy::too_many_arguments)]
	pub fn with_moe(
		engine: &PythonEngine,
		vocab_size: usize,
		context_length: usize,
		model_width: usize,
		expert_hidden_width: usize,
		num_layers: usize,
		num_heads: usize,
		num_experts: usize,
		experts_per_token: usize,
		epsilon: f32,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::Transformer::with_seed_moe(
			&engine.inner,
			vocab_size,
			context_length,
			model_width,
			expert_hidden_width,
			num_layers,
			num_heads,
			num_experts,
			experts_per_token,
			epsilon,
			seed,
		)
		.map(|inner| Self { inner })
		.map_err(python_error)
	}

	/// Evaluate all-position vocabulary logits for U8 or U32 tokens `[B, S]`.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn vocab_size(&self) -> usize {
		self.inner.vocab_size()
	}

	#[getter]
	pub fn context_length(&self) -> usize {
		self.inner.context_length()
	}

	#[getter]
	pub fn model_width(&self) -> usize {
		self.inner.model_width()
	}

	#[getter]
	pub fn hidden_width(&self) -> usize {
		self.inner.hidden_width()
	}

	#[getter]
	pub fn num_layers(&self) -> usize {
		self.inner.num_layers()
	}

	#[getter]
	pub fn num_heads(&self) -> usize {
		self.inner.num_heads()
	}

	#[getter]
	pub fn is_moe(&self) -> bool {
		self.inner.is_moe()
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
			"Transformer(vocab={}, ctx={}, d={}, ffn={}, layers={}, heads={}, moe={})",
			self.inner.vocab_size(),
			self.inner.context_length(),
			self.inner.model_width(),
			self.inner.hidden_width(),
			self.inner.num_layers(),
			self.inner.num_heads(),
			self.inner.is_moe(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonAttentionMode>()?;
	module.add_class::<PythonTransformerBlock>()?;
	module.add_class::<PythonTransformer>()?;
	Ok(())
}
