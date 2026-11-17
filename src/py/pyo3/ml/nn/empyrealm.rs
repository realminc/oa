use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;
use super::mamba3::PythonMamba3Config;

// ── EmpyrealmConfig ───────────────────────────────────────────────────────────

/// Construction contract for the Empyrealm hybrid stack.
#[pyclass(name = "EmpyrealmConfig")]
#[derive(Clone)]
pub(crate) struct PythonEmpyrealmConfig {
	pub(crate) inner: oa::ml::nn::EmpyrealmConfig,
}

#[pymethods]
impl PythonEmpyrealmConfig {
	/// Construct with OA hybrid defaults for the given model width and
	/// packed sequence length.
	///
	/// Accepts `dim` as an alias for `model_width`. Additional keyword
	/// arguments override the corresponding struct fields after construction.
	#[new]
	#[pyo3(signature = (model_width=None, dim=None, sequence_length=None, layers=None, attention_every=None, attention_heads=None, ffn_hidden=None, moe_experts=None, moe_experts_per_token=None, epsilon=None))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		model_width: Option<usize>,
		dim: Option<usize>,
		sequence_length: Option<usize>,
		layers: Option<usize>,
		attention_every: Option<usize>,
		attention_heads: Option<usize>,
		ffn_hidden: Option<usize>,
		moe_experts: Option<usize>,
		moe_experts_per_token: Option<usize>,
		epsilon: Option<f32>,
	) -> PyResult<Self> {
		let mw = model_width.or(dim).ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("EmpyrealmConfig requires model_width or dim")
		})?;
		let sl = sequence_length.ok_or_else(|| {
			pyo3::exceptions::PyTypeError::new_err("EmpyrealmConfig requires sequence_length")
		})?;
		let mut inner = oa::ml::nn::EmpyrealmConfig::new(mw, sl);
		if let Some(v) = layers {
			inner.layers = v;
		}
		if let Some(v) = attention_every {
			inner.attention_every = v;
		}
		if let Some(v) = attention_heads {
			inner.attention_heads = v;
		}
		if let Some(v) = ffn_hidden {
			inner.ffn_hidden = v;
		}
		if let Some(v) = moe_experts {
			inner.moe_experts = v;
		}
		if let Some(v) = moe_experts_per_token {
			inner.moe_experts_per_token = v;
		}
		if let Some(v) = epsilon {
			inner.epsilon = v;
		}
		Ok(Self { inner })
	}

	#[getter]
	pub fn layers(&self) -> usize {
		self.inner.layers
	}
	#[setter]
	pub fn set_layers(&mut self, value: usize) {
		self.inner.layers = value;
	}

	#[getter]
	pub fn attention_every(&self) -> usize {
		self.inner.attention_every
	}
	#[setter]
	pub fn set_attention_every(&mut self, value: usize) {
		self.inner.attention_every = value;
	}

	#[getter]
	pub fn attention_heads(&self) -> usize {
		self.inner.attention_heads
	}
	#[setter]
	pub fn set_attention_heads(&mut self, value: usize) {
		self.inner.attention_heads = value;
	}

	#[getter]
	pub fn sequence_length(&self) -> usize {
		self.inner.sequence_length
	}
	#[setter]
	pub fn set_sequence_length(&mut self, value: usize) {
		self.inner.sequence_length = value;
	}

	#[getter]
	pub fn ffn_hidden(&self) -> usize {
		self.inner.ffn_hidden
	}
	#[setter]
	pub fn set_ffn_hidden(&mut self, value: usize) {
		self.inner.ffn_hidden = value;
	}

	#[getter]
	pub fn moe_experts(&self) -> usize {
		self.inner.moe_experts
	}
	#[setter]
	pub fn set_moe_experts(&mut self, value: usize) {
		self.inner.moe_experts = value;
	}

	#[getter]
	pub fn moe_experts_per_token(&self) -> usize {
		self.inner.moe_experts_per_token
	}
	#[setter]
	pub fn set_moe_experts_per_token(&mut self, value: usize) {
		self.inner.moe_experts_per_token = value;
	}

	#[getter]
	pub fn epsilon(&self) -> f32 {
		self.inner.epsilon
	}
	#[setter]
	pub fn set_epsilon(&mut self, value: f32) {
		self.inner.epsilon = value;
	}

	#[getter]
	pub fn mamba(&self) -> PythonMamba3Config {
		PythonMamba3Config {
			inner: self.inner.mamba,
		}
	}
	#[setter]
	pub fn set_mamba(&mut self, value: PythonMamba3Config) {
		self.inner.mamba = value.inner;
	}

	pub fn __repr__(&self) -> String {
		format!(
			"EmpyrealmConfig(layers={}, attention_every={}, attention_heads={}, sequence_length={}, ffn_hidden={}, moe_experts={}, epsilon={})",
			self.inner.layers,
			self.inner.attention_every,
			self.inner.attention_heads,
			self.inner.sequence_length,
			self.inner.ffn_hidden,
			self.inner.moe_experts,
			self.inner.epsilon,
		)
	}
}

// ── EmpyrealmState ────────────────────────────────────────────────────────────

/// Caller-owned per-block decode state for one Empyrealm module and batch size.
#[pyclass(name = "EmpyrealmState", unsendable)]
pub(crate) struct PythonEmpyrealmState {
	pub(crate) inner: oa::ml::nn::EmpyrealmState,
}

#[pymethods]
impl PythonEmpyrealmState {
	#[getter]
	pub fn batch_size(&self) -> usize {
		self.inner.batch_size()
	}

	#[getter]
	pub fn layer_count(&self) -> usize {
		self.inner.layer_count()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"EmpyrealmState(batch_size={}, layer_count={})",
			self.inner.batch_size(),
			self.inner.layer_count()
		)
	}
}

// ── Empyrealm ────────────────────────────────────────────────────────────────

/// OA embedding-plus-hybrid-Mamba core with canonical Rust execution path.
#[pyclass(name = "Empyrealm", unsendable)]
pub(crate) struct PythonEmpyrealm {
	pub(crate) inner: oa::ml::nn::Empyrealm,
}

#[pymethods]
impl PythonEmpyrealm {
	/// Construct a deterministic token embedding and hybrid mixer stack.
	#[new]
	pub fn new(
		engine: &PythonEngine,
		vocab_size: usize,
		config: &PythonEmpyrealmConfig,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::Empyrealm::new_with_seed(&engine.inner, vocab_size, config.inner.clone(), seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Embed a U8/U32 `[batch, sequence]` token grid and evaluate the stack.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Evaluate the stack over pre-embedded `[B, S, D]` features.
	pub fn forward_embedded(&self, embedded: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_embedded(&embedded.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Allocate a per-block decode state for `batch_size` sequences.
	pub fn new_state(&self, batch_size: usize) -> PyResult<PythonEmpyrealmState> {
		self
			.inner
			.new_state(batch_size)
			.map(|inner| PythonEmpyrealmState { inner })
			.map_err(python_error)
	}

	/// Embed one `[batch, 1]` token column and advance the decode state.
	pub fn step(
		&self,
		tokens: &PythonMatrix,
		state: &mut PythonEmpyrealmState,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.step(&tokens.inner, &mut state.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Advance one pre-embedded `[B, 1, D]` position through the stack.
	pub fn step_embedded(
		&self,
		embedded: &PythonMatrix,
		state: &mut PythonEmpyrealmState,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.step_embedded(&embedded.inner, &mut state.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Return the number of blocks in the hybrid stack.
	pub fn layer_count(&self) -> usize {
		self.inner.layer_count()
	}

	/// Return the block positions holding self-attention mixers.
	pub fn attention_indices(&self) -> Vec<usize> {
		self.inner.attention_indices()
	}

	pub fn train(&self) {
		use oa::ml::Module as _;
		self.inner.train(true);
	}

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

	pub fn __repr__(&self) -> &str {
		"Empyrealm"
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonEmpyrealmConfig>()?;
	module.add_class::<PythonEmpyrealmState>()?;
	module.add_class::<PythonEmpyrealm>()?;
	Ok(())
}
