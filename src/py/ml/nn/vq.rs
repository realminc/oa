use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

// ── VectorQuantizerConfig ─────────────────────────────────────────────────────

/// Configuration for one EMA-trained vector-quantization codebook.
#[pyclass(name = "VectorQuantizerConfig")]
#[derive(Clone)]
pub(crate) struct PythonVectorQuantizerConfig {
	pub(crate) inner: oa::ml::nn::VectorQuantizerConfig,
}

#[pymethods]
impl PythonVectorQuantizerConfig {
	/// Construct with the OA defaults or explicit fields.
	#[new]
	#[pyo3(signature = (
		num_codes = 256,
		code_dim = 64,
		commitment_beta = 0.25,
		ema_decay = 0.99,
		ema_epsilon = 1e-5,
		dead_threshold = 1.0,
		normalize_codes = false,
	))]
	pub fn new(
		num_codes: usize,
		code_dim: usize,
		commitment_beta: f32,
		ema_decay: f32,
		ema_epsilon: f32,
		dead_threshold: f32,
		normalize_codes: bool,
	) -> Self {
		Self {
			inner: oa::ml::nn::VectorQuantizerConfig {
				num_codes,
				code_dim,
				commitment_beta,
				ema_decay,
				ema_epsilon,
				dead_threshold,
				normalize_codes,
			},
		}
	}

	#[getter]
	pub fn num_codes(&self) -> usize {
		self.inner.num_codes
	}

	#[getter]
	pub fn code_dim(&self) -> usize {
		self.inner.code_dim
	}

	#[getter]
	pub fn commitment_beta(&self) -> f32 {
		self.inner.commitment_beta
	}

	#[getter]
	pub fn ema_decay(&self) -> f32 {
		self.inner.ema_decay
	}

	#[getter]
	pub fn ema_epsilon(&self) -> f32 {
		self.inner.ema_epsilon
	}

	#[getter]
	pub fn dead_threshold(&self) -> f32 {
		self.inner.dead_threshold
	}

	#[getter]
	pub fn normalize_codes(&self) -> bool {
		self.inner.normalize_codes
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VectorQuantizerConfig(num_codes={}, code_dim={}, commitment_beta={})",
			self.inner.num_codes, self.inner.code_dim, self.inner.commitment_beta,
		)
	}
}

// ── VqResult ──────────────────────────────────────────────────────────────────

/// Straight-through vector-quantization result.
#[pyclass(name = "VqResult", unsendable)]
pub(crate) struct PythonVqResult {
	pub(crate) inner: oa::ml::nn::VqResult,
}

#[pymethods]
impl PythonVqResult {
	/// Quantized `[N,D]` value whose input adjoint is the identity.
	#[getter]
	pub fn quantized(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.quantized.clone())
	}

	/// I32 nearest-code indices `[N]`.
	#[getter]
	pub fn indices(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.indices.clone())
	}

	/// Rank-zero scalar commitment loss.
	#[getter]
	pub fn commitment_loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.commitment_loss.clone())
	}

	pub fn __repr__(&self) -> &str {
		"VqResult"
	}
}

// ── ResidualVqResult ──────────────────────────────────────────────────────────

/// Multi-level residual vector-quantization result.
#[pyclass(name = "ResidualVqResult", unsendable)]
pub(crate) struct PythonResidualVqResult {
	pub(crate) inner: oa::ml::nn::ResidualVqResult,
}

#[pymethods]
impl PythonResidualVqResult {
	/// Straight-through sum of all selected code vectors.
	#[getter]
	pub fn quantized(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.quantized.clone())
	}

	/// One I32 `[N]` token Matrix per level.
	#[getter]
	pub fn indices(&self) -> Vec<PythonMatrix> {
		self
			.inner
			.indices
			.iter()
			.cloned()
			.map(PythonMatrix::wrap)
			.collect()
	}

	/// Input residual used by each level's EMA update.
	#[getter]
	pub fn residuals(&self) -> Vec<PythonMatrix> {
		self
			.inner
			.residuals
			.iter()
			.cloned()
			.map(PythonMatrix::wrap)
			.collect()
	}

	/// Rank-zero scalar commitment loss against the summed code vectors.
	#[getter]
	pub fn commitment_loss(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.commitment_loss.clone())
	}

	pub fn __repr__(&self) -> &str {
		"ResidualVqResult"
	}
}

// ── VectorQuantizer ───────────────────────────────────────────────────────────

/// EMA-trained vector quantizer with persistent, non-gradient codebook state.
#[pyclass(name = "VectorQuantizer", unsendable)]
pub(crate) struct PythonVectorQuantizer {
	pub(crate) inner: oa::ml::nn::VectorQuantizer,
}

#[pymethods]
impl PythonVectorQuantizer {
	/// Construct a deterministically initialized quantizer.
	#[new]
	#[pyo3(signature = (engine, config, seed = 0))]
	pub fn new(
		engine: &PythonEngine,
		config: &PythonVectorQuantizerConfig,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::VectorQuantizer::with_seed(&engine.inner, config.inner, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Quantize latent rows and return the straight-through result.
	pub fn quantize(&self, latent: &PythonMatrix) -> PyResult<PythonVqResult> {
		self
			.inner
			.quantize(&latent.inner)
			.map(|inner| PythonVqResult { inner })
			.map_err(python_error)
	}

	/// Advance codebook EMA state once from the current latent assignment.
	pub fn ema_update(&self, latent: &PythonMatrix, indices: &PythonMatrix) -> PyResult<()> {
		self
			.inner
			.ema_update(&latent.inner, &indices.inner)
			.map_err(python_error)
	}

	/// Decode I32 token ids into code vectors.
	pub fn lookup(&self, indices: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.lookup(&indices.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Seed codes from the highest-L2-norm latent rows.
	pub fn seed(&self, latents: &PythonMatrix) -> PyResult<()> {
		self.inner.seed(&latents.inner).map_err(python_error)
	}

	/// Return a cheap handle to the current codebook value.
	pub fn codebook(&self) -> PythonMatrix {
		PythonMatrix::wrap(self.inner.codebook())
	}

	/// Return the number of completed EMA transitions modulo `u32`.
	pub fn ema_step(&self) -> u32 {
		self.inner.ema_step()
	}

	#[getter]
	pub fn num_codes(&self) -> usize {
		self.inner.config().num_codes
	}

	#[getter]
	pub fn code_dim(&self) -> usize {
		self.inner.config().code_dim
	}

	pub fn __repr__(&self) -> String {
		format!(
			"VectorQuantizer(num_codes={}, code_dim={})",
			self.inner.config().num_codes,
			self.inner.config().code_dim,
		)
	}
}

// ── ResidualVectorQuantizer ───────────────────────────────────────────────────

/// Stack of independently EMA-trained residual vector quantizers.
#[pyclass(name = "ResidualVectorQuantizer", unsendable)]
pub(crate) struct PythonResidualVectorQuantizer {
	pub(crate) inner: oa::ml::nn::ResidualVectorQuantizer,
}

#[pymethods]
impl PythonResidualVectorQuantizer {
	/// Construct `num_levels` deterministically initialized quantizers.
	#[new]
	#[pyo3(signature = (engine, config, num_levels, seed = 0))]
	pub fn new(
		engine: &PythonEngine,
		config: &PythonVectorQuantizerConfig,
		num_levels: usize,
		seed: u64,
	) -> PyResult<Self> {
		oa::ml::nn::ResidualVectorQuantizer::with_seed(&engine.inner, config.inner, num_levels, seed)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Quantize successive residuals and apply one straight-through estimator.
	pub fn quantize(&self, latent: &PythonMatrix) -> PyResult<PythonResidualVqResult> {
		self
			.inner
			.quantize(&latent.inner)
			.map(|inner| PythonResidualVqResult { inner })
			.map_err(python_error)
	}

	/// Advance every level's EMA state using a matching quantization result.
	pub fn ema_update(&self, result: &PythonResidualVqResult) -> PyResult<()> {
		self.inner.ema_update(&result.inner).map_err(python_error)
	}

	/// Decode one or more shallow-to-deep token levels and sum their codes.
	pub fn lookup(&self, indices: Vec<PyRef<'_, PythonMatrix>>) -> PyResult<PythonMatrix> {
		let inner: Vec<oa::Matrix> = indices.iter().map(|m| m.inner.clone()).collect();
		self
			.inner
			.lookup(&inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	/// Greedily seed each codebook from the residual left by preceding levels.
	pub fn seed(&self, latents: &PythonMatrix) -> PyResult<()> {
		self.inner.seed(&latents.inner).map_err(python_error)
	}

	#[getter]
	pub fn num_levels(&self) -> usize {
		self.inner.num_levels()
	}

	pub fn __repr__(&self) -> String {
		format!(
			"ResidualVectorQuantizer(num_levels={})",
			self.inner.num_levels()
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonVectorQuantizerConfig>()?;
	module.add_class::<PythonVqResult>()?;
	module.add_class::<PythonResidualVqResult>()?;
	module.add_class::<PythonVectorQuantizer>()?;
	module.add_class::<PythonResidualVectorQuantizer>()?;
	Ok(())
}
