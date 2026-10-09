use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::super::autograd::PythonParameter;

// ── FlowTimeEmbedding ─────────────────────────────────────────────────────────

/// GPU sinusoidal embedding for normalized continuous flow/diffusion time.
#[pyclass(name = "FlowTimeEmbedding", unsendable)]
pub(crate) struct PythonFlowTimeEmbedding {
	pub(crate) inner: oa::ml::nn::FlowTimeEmbedding,
}

#[pymethods]
impl PythonFlowTimeEmbedding {
	/// Construct the donor sinusoidal time embedding.
	#[new]
	#[pyo3(signature = (engine, embedding_dim, max_period = 10_000.0, time_scale = 1_000.0))]
	pub fn new(
		engine: &PythonEngine,
		embedding_dim: usize,
		max_period: f32,
		time_scale: f32,
	) -> PyResult<Self> {
		oa::ml::nn::FlowTimeEmbedding::new(&engine.inner, embedding_dim, max_period, time_scale)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Embed F32 time values shaped `[B]` or `[B, 1]` as `[B, D]`.
	pub fn forward(&self, time: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&time.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	#[getter]
	pub fn embedding_dim(&self) -> usize {
		self.inner.embedding_dim()
	}

	#[getter]
	pub fn max_period(&self) -> f32 {
		self.inner.max_period()
	}

	#[getter]
	pub fn time_scale(&self) -> f32 {
		self.inner.time_scale()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"FlowTimeEmbedding(embedding_dim={}, max_period={}, time_scale={})",
			self.inner.embedding_dim(),
			self.inner.max_period(),
			self.inner.time_scale(),
		)
	}
}

// ── FlowTransformerConfig ─────────────────────────────────────────────────────

/// Configuration for a bidirectional flow/diffusion Transformer backbone.
#[pyclass(name = "FlowTransformerConfig")]
#[derive(Clone)]
pub(crate) struct PythonFlowTransformerConfig {
	pub(crate) inner: oa::ml::nn::FlowTransformerConfig,
}

#[pymethods]
impl PythonFlowTransformerConfig {
	#[new]
	#[pyo3(signature = (
		model_width,
		hidden_width,
		sequence_length,
		num_layers = 1,
		num_heads = 1,
		num_experts = 0,
		experts_per_token = 0,
		epsilon = 1e-5,
		adaptive_conditioning = true,
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		model_width: usize,
		hidden_width: usize,
		sequence_length: usize,
		num_layers: usize,
		num_heads: usize,
		num_experts: usize,
		experts_per_token: usize,
		epsilon: f32,
		adaptive_conditioning: bool,
	) -> Self {
		Self {
			inner: oa::ml::nn::FlowTransformerConfig {
				model_width,
				hidden_width,
				sequence_length,
				num_layers,
				num_heads,
				num_experts,
				experts_per_token,
				epsilon,
				adaptive_conditioning,
			},
		}
	}

	#[getter]
	pub fn model_width(&self) -> usize {
		self.inner.model_width
	}

	#[getter]
	pub fn hidden_width(&self) -> usize {
		self.inner.hidden_width
	}

	#[getter]
	pub fn sequence_length(&self) -> usize {
		self.inner.sequence_length
	}

	#[getter]
	pub fn num_layers(&self) -> usize {
		self.inner.num_layers
	}

	#[getter]
	pub fn num_heads(&self) -> usize {
		self.inner.num_heads
	}

	#[getter]
	pub fn num_experts(&self) -> usize {
		self.inner.num_experts
	}

	#[getter]
	pub fn experts_per_token(&self) -> usize {
		self.inner.experts_per_token
	}

	#[getter]
	pub fn epsilon(&self) -> f32 {
		self.inner.epsilon
	}

	#[getter]
	pub fn adaptive_conditioning(&self) -> bool {
		self.inner.adaptive_conditioning
	}

	pub fn __repr__(&self) -> String {
		format!(
			"FlowTransformerConfig(model_width={}, seq_len={}, layers={}, heads={})",
			self.inner.model_width,
			self.inner.sequence_length,
			self.inner.num_layers,
			self.inner.num_heads,
		)
	}
}

// ── FlowTransformer ───────────────────────────────────────────────────────────

/// Reusable bidirectional Transformer backbone for flow/diffusion denoisers.
#[pyclass(name = "FlowTransformer", unsendable)]
pub(crate) struct PythonFlowTransformer {
	pub(crate) inner: oa::ml::nn::FlowTransformer,
}

#[pymethods]
impl PythonFlowTransformer {
	/// Construct a deterministic dense or dropless-MoE flow Transformer.
	#[new]
	pub fn new(
		engine: &PythonEngine,
		config: &PythonFlowTransformerConfig,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::FlowTransformer::with_seed(&engine.inner, config.inner, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Apply the unmasked backbone to rank-two or rank-three token state.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Apply the backbone with a binary key mask shaped `[B,S]` or `[B,S,1]`.
	pub fn forward_masked(
		&self,
		tokens: &PythonMatrix,
		token_mask: &PythonMatrix,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_masked(&tokens.inner, &token_mask.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Apply adaptive conditioning with an optional binary key mask.
	#[pyo3(signature = (tokens, condition, token_mask = None))]
	pub fn forward_conditioned(
		&self,
		tokens: &PythonMatrix,
		condition: &PythonMatrix,
		token_mask: Option<&PythonMatrix>,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_conditioned(
				&tokens.inner,
				&condition.inner,
				token_mask.map(|m| &m.inner),
			)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Update the configured sequence length without rebuilding parameters.
	pub fn set_sequence_length(&self, sequence_length: usize) -> PyResult<()> {
		self
			.inner
			.set_sequence_length(sequence_length)
			.map_err(python_error)
	}

	pub fn config(&self) -> PythonFlowTransformerConfig {
		PythonFlowTransformerConfig {
			inner: self.inner.config(),
		}
	}

	#[getter]
	pub fn is_moe(&self) -> bool {
		self.inner.is_moe()
	}

	#[getter]
	pub fn num_layers(&self) -> usize {
		self.inner.num_layers()
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

	pub fn __repr__(&self) -> String {
		let c = self.inner.config();
		format!(
			"FlowTransformer(model_width={}, seq_len={}, layers={}, moe={})",
			c.model_width,
			c.sequence_length,
			self.inner.num_layers(),
			self.inner.is_moe(),
		)
	}
}

// ── FlowDenoiserConfig ────────────────────────────────────────────────────────

/// Configuration for a modality-independent flow/diffusion denoiser.
#[pyclass(name = "FlowDenoiserConfig")]
#[derive(Clone)]
pub(crate) struct PythonFlowDenoiserConfig {
	pub(crate) inner: oa::ml::nn::FlowDenoiserConfig,
}

#[pymethods]
impl PythonFlowDenoiserConfig {
	#[new]
	#[pyo3(signature = (
		input_dim,
		backbone,
		condition_dim = 0,
		time_max_period = 10_000.0,
		time_scale = 1_000.0,
		condition_dropout_probability = 0.0,
	))]
	pub fn new(
		input_dim: usize,
		backbone: &PythonFlowTransformerConfig,
		condition_dim: usize,
		time_max_period: f32,
		time_scale: f32,
		condition_dropout_probability: f32,
	) -> Self {
		Self {
			inner: oa::ml::nn::FlowDenoiserConfig {
				input_dim,
				condition_dim,
				backbone: backbone.inner,
				time_max_period,
				time_scale,
				condition_dropout_probability,
			},
		}
	}

	#[getter]
	pub fn input_dim(&self) -> usize {
		self.inner.input_dim
	}

	#[getter]
	pub fn condition_dim(&self) -> usize {
		self.inner.condition_dim
	}

	#[getter]
	pub fn time_max_period(&self) -> f32 {
		self.inner.time_max_period
	}

	#[getter]
	pub fn time_scale(&self) -> f32 {
		self.inner.time_scale
	}

	#[getter]
	pub fn condition_dropout_probability(&self) -> f32 {
		self.inner.condition_dropout_probability
	}

	pub fn __repr__(&self) -> String {
		format!(
			"FlowDenoiserConfig(input_dim={}, condition_dim={})",
			self.inner.input_dim, self.inner.condition_dim,
		)
	}
}

// ── FlowDenoiser ──────────────────────────────────────────────────────────────

/// Trainable flow/diffusion denoiser with optional condition features and CFG.
#[pyclass(name = "FlowDenoiser", unsendable)]
pub(crate) struct PythonFlowDenoiser {
	pub(crate) inner: oa::ml::nn::FlowDenoiser,
}

#[pymethods]
impl PythonFlowDenoiser {
	/// Construct a deterministic dense or MoE denoiser.
	#[new]
	pub fn new(
		engine: &PythonEngine,
		config: &PythonFlowDenoiserConfig,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::FlowDenoiser::with_seed(&engine.inner, config.inner, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Denoise with zero time and zero condition.
	pub fn forward(&self, sample: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&sample.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Denoise with explicit time, optional condition, and optional token mask.
	#[pyo3(signature = (sample, time, condition = None, token_mask = None))]
	pub fn forward_conditioned(
		&self,
		sample: &PythonMatrix,
		time: &PythonMatrix,
		condition: Option<&PythonMatrix>,
		token_mask: Option<&PythonMatrix>,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_conditioned(
				&sample.inner,
				&time.inner,
				condition.map(|c| &c.inner),
				token_mask.map(|m| &m.inner),
			)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Apply classifier-free guidance in a bounded eval-mode scope.
	#[pyo3(signature = (sample, time, condition, guidance_scale, token_mask = None))]
	pub fn forward_guided(
		&self,
		sample: &PythonMatrix,
		time: &PythonMatrix,
		condition: &PythonMatrix,
		guidance_scale: f32,
		token_mask: Option<&PythonMatrix>,
	) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward_guided(
				&sample.inner,
				&time.inner,
				&condition.inner,
				guidance_scale,
				token_mask.map(|m| &m.inner),
			)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn config(&self) -> PythonFlowDenoiserConfig {
		PythonFlowDenoiserConfig {
			inner: self.inner.config(),
		}
	}

	#[getter]
	pub fn is_moe(&self) -> bool {
		self.inner.is_moe()
	}

	pub fn position(&self) -> PythonParameter {
		PythonParameter {
			inner: self.inner.position(),
		}
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

	pub fn __repr__(&self) -> String {
		let c = self.inner.config();
		format!(
			"FlowDenoiser(input_dim={}, condition_dim={}, moe={})",
			c.input_dim,
			c.condition_dim,
			self.inner.is_moe(),
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonFlowTimeEmbedding>()?;
	module.add_class::<PythonFlowTransformerConfig>()?;
	module.add_class::<PythonFlowTransformer>()?;
	module.add_class::<PythonFlowDenoiserConfig>()?;
	module.add_class::<PythonFlowDenoiser>()?;
	Ok(())
}
