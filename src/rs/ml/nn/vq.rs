use std::rc::Rc;

use crate::{DType, Engine, Error, Matrix, Result, matrix as core_matrix};

use super::super::{Module, ModuleRegistry, NamedBuffer, NamedStateU32, matrix, random};

/// Configuration for one EMA-trained vector-quantization codebook.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorQuantizerConfig {
	/// Number of codebook entries.
	pub num_codes: usize,
	/// Width of each latent/code vector.
	pub code_dim: usize,
	/// Commitment-loss multiplier.
	pub commitment_beta: f32,
	/// Exponential moving-average decay.
	pub ema_decay: f32,
	/// Positive division and normalization floor.
	pub ema_epsilon: f32,
	/// Revive codes whose EMA population falls below this value.
	pub dead_threshold: f32,
	/// Rescale codebook rows to unit RMS after each EMA update.
	pub normalize_codes: bool,
}

impl Default for VectorQuantizerConfig {
	fn default() -> Self {
		Self {
			num_codes: 256,
			code_dim: 64,
			commitment_beta: 0.25,
			ema_decay: 0.99,
			ema_epsilon: 1e-5,
			dead_threshold: 1.0,
			normalize_codes: false,
		}
	}
}

/// Straight-through vector-quantization result.
pub struct VqResult {
	/// Quantized `[N,D]` value whose input adjoint is the identity.
	pub quantized: Matrix,
	/// I32 nearest-code indices `[N]`.
	pub indices: Matrix,
	/// Scalar `[1]` commitment loss.
	pub commitment_loss: Matrix,
}

/// EMA-trained vector quantizer with persistent, non-gradient codebook state.
pub struct VectorQuantizer {
	config: VectorQuantizerConfig,
	codebook: NamedBuffer,
	embed_sum: NamedBuffer,
	cluster_size: NamedBuffer,
	ema_step: NamedStateU32,
	registry: ModuleRegistry,
}

impl VectorQuantizer {
	/// Construct a deterministically initialized quantizer.
	///
	/// # Errors
	///
	/// Returns an error for an invalid configuration, shape overflow, or failed
	/// state allocation/upload.
	pub fn with_seed(engine: &Engine, config: VectorQuantizerConfig, seed: u64) -> Result<Self> {
		validate_config(config)?;
		let count = config
			.num_codes
			.checked_mul(config.code_dim)
			.ok_or_else(|| Error::invalid_argument("VQ codebook size overflows usize"))?;
		let denominator = config
			.num_codes
			.checked_add(config.code_dim)
			.ok_or_else(|| Error::invalid_argument("VQ Glorot extent overflows usize"))?;
		let limit = (6.0_f32 / denominator as f32).sqrt();
		let values = random::symmetric_uniform(count, limit, seed);
		let codebook = Matrix::from_f32(engine, [config.num_codes, config.code_dim], &values)?;
		let embed_sum = Matrix::from_f32(engine, [config.num_codes, config.code_dim], &values)?;
		let cluster_size =
			Matrix::from_f32(engine, [config.num_codes], &vec![1.0; config.num_codes])?;
		Self::from_state(config, codebook, embed_sum, cluster_size)
	}

	/// Construct a quantizer using deterministic seed zero.
	///
	/// # Errors
	///
	/// Returns the same errors as [`Self::with_seed`].
	pub fn new(engine: &Engine, config: VectorQuantizerConfig) -> Result<Self> {
		Self::with_seed(engine, config, 0)
	}

	fn from_state(
		config: VectorQuantizerConfig,
		codebook: Matrix,
		embed_sum: Matrix,
		cluster_size: Matrix,
	) -> Result<Self> {
		validate_config(config)?;
		let state_shape = [config.num_codes, config.code_dim];
		if codebook.shape() != state_shape
			|| embed_sum.shape() != state_shape
			|| cluster_size.shape() != [config.num_codes]
			|| [codebook.dtype(), embed_sum.dtype(), cluster_size.dtype()]
				.into_iter()
				.any(|dtype| dtype != DType::F32)
			|| !codebook.engine_handle().same_as(embed_sum.engine_handle())
			|| !codebook
				.engine_handle()
				.same_as(cluster_size.engine_handle())
		{
			return Err(Error::invalid_argument(
				"VQ state must match the configured same-engine FP32 codebook geometry",
			));
		}
		let mut registry = ModuleRegistry::new();
		registry.register_buffer("codebook", codebook, true)?;
		registry.register_buffer("embed_sum", embed_sum, true)?;
		registry.register_buffer("cluster_size", cluster_size, true)?;
		let ema_step = registry.register_state_u32("ema_step", 0)?;
		let codebook = registry
			.buffer_handle("codebook")
			.ok_or_else(|| Error::internal("VQ codebook registration was lost"))?;
		let embed_sum = registry
			.buffer_handle("embed_sum")
			.ok_or_else(|| Error::internal("VQ embed-sum registration was lost"))?;
		let cluster_size = registry
			.buffer_handle("cluster_size")
			.ok_or_else(|| Error::internal("VQ cluster-size registration was lost"))?;
		Ok(Self {
			config,
			codebook,
			embed_sum,
			cluster_size,
			ema_step,
			registry,
		})
	}

	/// Quantize latent rows and build the donor straight-through estimator.
	///
	/// This records entirely device-side work. The codebook receives no gradient;
	/// EMA updates it explicitly through [`Self::ema_update`].
	///
	/// # Errors
	///
	/// Returns an error for an incompatible latent or failed runtime recording.
	pub fn quantize(&self, latent: &Matrix) -> Result<VqResult> {
		let assignment = matrix::vq_assign(latent, &self.codebook.data())?;
		let detached_delta = matrix::detach(&core_matrix::sub(&assignment.quantized, latent)?)?;
		let quantized = core_matrix::add(latent, &detached_delta)?;
		let difference = core_matrix::sub(latent, &assignment.quantized)?;
		let squared = core_matrix::mul(&difference, &difference)?;
		let mean = core_matrix::scale(
			&core_matrix::sum(&squared, -1)?,
			1.0 / squared.num_elements() as f32,
		)?;
		let commitment_loss = core_matrix::scale(&mean, self.config.commitment_beta)?;
		Ok(VqResult {
			quantized,
			indices: assignment.indices,
			commitment_loss,
		})
	}

	/// Advance codebook EMA state once from the current latent assignment.
	///
	/// # Errors
	///
	/// Returns an error for incompatible inputs or failed runtime recording.
	pub fn ema_update(&self, latent: &Matrix, indices: &Matrix) -> Result<()> {
		let step = self.ema_step.value();
		let state = matrix::vq_ema_update(
			latent,
			indices,
			&self.embed_sum.data(),
			&self.cluster_size.data(),
			&self.codebook.data(),
			self.config.ema_decay,
			self.config.ema_epsilon,
			self.config.dead_threshold,
			step,
			self.config.normalize_codes,
		)?;
		self.embed_sum.replace_data(state.embed_sum)?;
		self.cluster_size.replace_data(state.cluster_size)?;
		self.codebook.replace_data(state.codebook)?;
		self.ema_step.replace_value(step.wrapping_add(1));
		Ok(())
	}

	/// Decode I32 token ids into code vectors.
	///
	/// # Errors
	///
	/// Returns an error for incompatible indices or failed runtime recording.
	pub fn lookup(&self, indices: &Matrix) -> Result<Matrix> {
		matrix::vq_lookup(&self.codebook.data(), indices)
	}

	/// Seed codes from the highest-L2-norm latent rows.
	///
	/// Equal norms select lower source-row indices. Selection stays on the GPU;
	/// the final checkpoint is the donor-compatible explicit initialization
	/// completion boundary.
	///
	/// # Errors
	///
	/// Returns an error unless `latents` is FP32 `[N,D]`, `N >= K`, and its code
	/// dimension and engine match this quantizer, or readback/upload fails.
	pub fn seed(&self, latents: &Matrix) -> Result<()> {
		let [rows, code_dim] = latents.shape() else {
			return Err(Error::invalid_argument(
				"VQ seed latents must have shape [N,D]",
			));
		};
		let codebook = self.codebook.data();
		if latents.dtype() != DType::F32
			|| *rows < self.config.num_codes
			|| self.config.num_codes > 512
			|| *code_dim != self.config.code_dim
			|| !latents.engine_handle().same_as(codebook.engine_handle())
		{
			return Err(Error::invalid_argument(
				"VQ seed requires same-engine FP32 [N,D] latents with N >= K",
			));
		}
		let squared = core_matrix::mul(latents, latents)?;
		let norms = core_matrix::sum(&squared, 1)?.reshape([*rows])?;
		let selected = core_matrix::top_k(
			&norms,
			i32::try_from(self.config.num_codes)
				.map_err(|_| Error::invalid_argument("VQ seed code count exceeds i32"))?,
			-1,
		)?;
		let seeded = matrix::vq_lookup(latents, &selected.indices)?;
		let embed_sum = core_matrix::copy(&seeded)?;
		let cluster_size = Matrix::from_slice_handle(
			latents.engine_handle(),
			vec![self.config.num_codes],
			&vec![1.0_f32; self.config.num_codes],
		)?;
		self.codebook.replace_data(seeded)?;
		self.embed_sum.replace_data(embed_sum)?;
		self.cluster_size.replace_data(cluster_size)?;
		latents.engine_handle().checkpoint(false)?.wait()
	}

	/// Return a cheap handle to the current codebook value.
	pub fn codebook(&self) -> Matrix {
		self.codebook.data()
	}

	/// Return this quantizer's immutable configuration.
	pub const fn config(&self) -> &VectorQuantizerConfig {
		&self.config
	}

	/// Return the number of completed EMA transitions modulo `u32`.
	pub fn ema_step(&self) -> u32 {
		self.ema_step.value()
	}
}

impl Module for VectorQuantizer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Ok(self.quantize(input)?.quantized)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Multi-level residual vector-quantization result.
pub struct ResidualVqResult {
	/// Straight-through sum of all selected code vectors.
	pub quantized: Matrix,
	/// One I32 `[N]` token Matrix per level.
	pub indices: Vec<Matrix>,
	/// Input residual used by each level's EMA update.
	pub residuals: Vec<Matrix>,
	/// Scalar `[1]` commitment loss against the summed code vectors.
	pub commitment_loss: Matrix,
}

/// Stack of independently EMA-trained residual vector quantizers.
pub struct ResidualVectorQuantizer {
	config: VectorQuantizerConfig,
	levels: Vec<Rc<VectorQuantizer>>,
	registry: ModuleRegistry,
}

impl ResidualVectorQuantizer {
	/// Construct `num_levels` deterministically initialized quantizers.
	///
	/// # Errors
	///
	/// Returns an error when the level count or configuration is invalid, child
	/// state allocation fails, or module registration fails.
	pub fn with_seed(
		engine: &Engine,
		config: VectorQuantizerConfig,
		num_levels: usize,
		seed: u64,
	) -> Result<Self> {
		if num_levels == 0 {
			return Err(Error::invalid_argument(
				"residual VQ requires at least one level",
			));
		}
		let mut registry = ModuleRegistry::new();
		let mut levels = Vec::with_capacity(num_levels);
		for level in 0..num_levels {
			let quantizer = Rc::new(VectorQuantizer::with_seed(
				engine,
				config,
				seed.wrapping_add(level as u64),
			)?);
			registry.register_module(format!("level{level}"), quantizer.clone())?;
			levels.push(quantizer);
		}
		Ok(Self {
			config,
			levels,
			registry,
		})
	}

	/// Construct a residual quantizer using deterministic seed zero.
	///
	/// # Errors
	///
	/// Returns the same errors as [`Self::with_seed`].
	pub fn new(engine: &Engine, config: VectorQuantizerConfig, num_levels: usize) -> Result<Self> {
		Self::with_seed(engine, config, num_levels, 0)
	}

	/// Quantize successive residuals and apply one straight-through estimator.
	///
	/// # Errors
	///
	/// Returns an error for an incompatible latent or failed runtime recording.
	pub fn quantize(&self, latent: &Matrix) -> Result<ResidualVqResult> {
		let mut residual = latent.clone();
		let mut total = None;
		let mut indices = Vec::with_capacity(self.levels.len());
		let mut residuals = Vec::with_capacity(self.levels.len());
		for level in &self.levels {
			let assignment = matrix::vq_assign(&residual, &level.codebook())?;
			residuals.push(residual.clone());
			indices.push(assignment.indices);
			total = Some(match total {
				Some(previous) => core_matrix::add(&previous, &assignment.quantized)?,
				None => assignment.quantized.clone(),
			});
			residual = core_matrix::sub(&residual, &assignment.quantized)?;
		}
		let total = total.ok_or_else(|| Error::internal("residual VQ lost all levels"))?;
		let quantized =
			core_matrix::add(latent, &matrix::detach(&core_matrix::sub(&total, latent)?)?)?;
		let difference = core_matrix::sub(latent, &total)?;
		let squared = core_matrix::mul(&difference, &difference)?;
		let commitment_loss = core_matrix::scale(
			&core_matrix::sum(&squared, -1)?,
			self.config.commitment_beta / squared.num_elements() as f32,
		)?;
		Ok(ResidualVqResult {
			quantized,
			indices,
			residuals,
			commitment_loss,
		})
	}

	/// Advance every level's EMA state using a matching quantization result.
	///
	/// # Errors
	///
	/// Returns an error when the result has the wrong level count or an update fails.
	pub fn ema_update(&self, result: &ResidualVqResult) -> Result<()> {
		if result.indices.len() != self.levels.len() || result.residuals.len() != self.levels.len()
		{
			return Err(Error::invalid_argument(
				"residual VQ result level count does not match module",
			));
		}
		for ((level, residual), indices) in self
			.levels
			.iter()
			.zip(&result.residuals)
			.zip(&result.indices)
		{
			level.ema_update(residual, indices)?;
		}
		Ok(())
	}

	/// Decode one or more shallow-to-deep token levels and sum their codes.
	///
	/// # Errors
	///
	/// Returns an error when no indices are supplied, too many levels are supplied,
	/// index shapes differ, or a level lookup/runtime operation fails.
	pub fn lookup(&self, indices: &[Matrix]) -> Result<Matrix> {
		if indices.is_empty() || indices.len() > self.levels.len() {
			return Err(Error::invalid_argument(
				"residual VQ lookup requires one through num_levels token matrices",
			));
		}
		let mut total = self.levels[0].lookup(&indices[0])?;
		for (level, level_indices) in self.levels.iter().zip(indices).skip(1) {
			let codes = level.lookup(level_indices)?;
			if codes.shape() != total.shape() {
				return Err(Error::invalid_argument(
					"residual VQ token shapes must match",
				));
			}
			total = core_matrix::add(&total, &codes)?;
		}
		Ok(total)
	}

	/// Greedily seed each codebook from the residual left by preceding levels.
	///
	/// # Errors
	///
	/// Returns an error when a level cannot be seeded or residual recording fails.
	pub fn seed(&self, latents: &Matrix) -> Result<()> {
		let mut residual = latents.clone();
		for level in &self.levels {
			level.seed(&residual)?;
			let assignment = matrix::vq_assign(&residual, &level.codebook())?;
			residual = core_matrix::sub(&residual, &assignment.quantized)?;
		}
		Ok(())
	}

	/// Return the number of quantization levels.
	pub fn num_levels(&self) -> usize {
		self.levels.len()
	}

	/// Return one level by index.
	pub fn level(&self, index: usize) -> Option<&VectorQuantizer> {
		self.levels.get(index).map(Rc::as_ref)
	}
}

impl Module for ResidualVectorQuantizer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Ok(self.quantize(input)?.quantized)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

fn validate_config(config: VectorQuantizerConfig) -> Result<()> {
	if config.num_codes == 0
		|| config.code_dim == 0
		|| !config.commitment_beta.is_finite()
		|| config.commitment_beta < 0.0
		|| !config.ema_decay.is_finite()
		|| !(0.0..=1.0).contains(&config.ema_decay)
		|| !config.ema_epsilon.is_finite()
		|| config.ema_epsilon <= 0.0
		|| !config.dead_threshold.is_finite()
		|| config.dead_threshold < 0.0
	{
		return Err(Error::invalid_argument(
			"invalid vector-quantizer configuration",
		));
	}
	Ok(())
}
