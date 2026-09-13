//! ALM motion-tokenizer workload.

mod clip;
mod model;
mod prior;
mod tokenizer;

pub use clip::{ClipText, ClipTextConfig};
pub use model::{Alm, AlmConfig};
pub use prior::{AlmFfnType, AlmGenerationOptions, AlmPrior, AlmPriorConfig};
pub use tokenizer::{ClipTokenBatch, ClipTokenizer};

use std::rc::Rc;

use crate::ml::{
	Module, ModuleRegistry, NamedBuffer, matrix as ml_matrix,
	nn::{
		Conv1d, ConvTranspose1d, LayerNorm, ResidualVectorQuantizer, ResidualVqResult,
		VectorQuantizerConfig,
	},
};
use crate::{DType, Engine, Error, Matrix, Result, matrix};

const NORMALIZATION_EPSILON: f32 = 1e-5;
const DEFAULT_INITIALIZER_SEED: u64 = 0xC0FFEE;

/// Architecture and quantization policy for the ALM temporal VQ-VAE tokenizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlmTokenizerConfig {
	/// Motion feature width per input frame.
	pub input_dim: usize,
	/// Hidden Conv1d channel width.
	pub width: usize,
	/// Latent and codebook-vector width.
	pub code_dim: usize,
	/// Number of discrete codes in the single EMA codebook.
	pub num_codes: usize,
	/// Number of stride-two temporal downsampling stages.
	pub downsample_stages: usize,
	/// Number of dilated residual blocks per stage.
	pub depth: usize,
	/// Commitment-loss multiplier.
	pub commitment_beta: f32,
	/// Codebook exponential moving-average decay.
	pub ema_decay: f32,
	/// Positive EMA numerical floor.
	pub ema_epsilon: f32,
	/// Population threshold below which a code is revived.
	pub dead_threshold: f32,
}

impl Default for AlmTokenizerConfig {
	fn default() -> Self {
		Self {
			input_dim: 263,
			width: 512,
			code_dim: 512,
			num_codes: 512,
			downsample_stages: 2,
			depth: 3,
			commitment_beta: 0.25,
			ema_decay: 0.99,
			ema_epsilon: 1e-5,
			dead_threshold: 2.0,
		}
	}
}

/// Temporal Conv1d VQ-VAE used by ALM to map motion frames to discrete tokens.
///
/// The model accepts `[batch, frames, input_dim]`, downsamples time by
/// `2^downsample_stages`, performs one level of normalized EMA vector
/// quantization, and reconstructs the original rank-three shape.
pub struct AlmTokenizer {
	config: AlmTokenizerConfig,
	downsample_factor: usize,
	enc_in: Rc<Conv1d>,
	enc_down: Vec<Rc<Conv1d>>,
	enc_res: Vec<Rc<Conv1d>>,
	enc_out: Rc<Conv1d>,
	dec_in: Rc<Conv1d>,
	dec_res: Vec<Rc<Conv1d>>,
	dec_up: Vec<Rc<ConvTranspose1d>>,
	dec_mid: Rc<Conv1d>,
	dec_out: Rc<Conv1d>,
	enc_norms: Vec<Rc<LayerNorm>>,
	dec_norms: Vec<Rc<LayerNorm>>,
	latent_rms_weight: NamedBuffer,
	rvq: Rc<ResidualVectorQuantizer>,
	registry: ModuleRegistry,
}

impl AlmTokenizer {
	/// Construct the tokenizer with the donor's deterministic `0xC0FFEE`
	/// continuous Glorot stream.
	///
	/// # Errors
	///
	/// Returns an error for invalid geometry or quantizer policy, arithmetic
	/// overflow, failed allocation/upload, or module-registration failure.
	pub fn new(engine: &Engine, config: AlmTokenizerConfig) -> Result<Self> {
		Self::with_seed(engine, config, DEFAULT_INITIALIZER_SEED)
	}

	/// Construct the tokenizer with an explicit deterministic initializer seed.
	///
	/// Every convolution consumes one continuous donor-compatible LCG stream;
	/// changing the seed does not change parameter names or topology.
	///
	/// # Errors
	///
	/// Returns the same errors as [`Self::new`].
	pub fn with_seed(engine: &Engine, config: AlmTokenizerConfig, seed: u64) -> Result<Self> {
		validate_config(config)?;
		let downsample_factor =
			1_usize
				.checked_shl(u32::try_from(config.downsample_stages).map_err(|_| {
					Error::invalid_argument("ALM downsample stage count exceeds u32")
				})?)
				.ok_or_else(|| Error::invalid_argument("ALM downsample factor overflows usize"))?;
		let mut rng = seed;
		let mut registry = ModuleRegistry::new();

		let enc_in = make_conv(engine, config.input_dim, config.width, 3, 1, 1, 1, &mut rng)?;
		registry.register_module("enc_in", enc_in.clone())?;

		let mut enc_down = Vec::with_capacity(config.downsample_stages);
		let mut enc_res = Vec::with_capacity(
			config
				.downsample_stages
				.checked_mul(config.depth)
				.and_then(|count| count.checked_mul(2))
				.ok_or_else(|| Error::invalid_argument("ALM encoder depth overflows usize"))?,
		);
		for stage in 0..config.downsample_stages {
			let down = make_conv(engine, config.width, config.width, 4, 2, 1, 1, &mut rng)?;
			registry.register_module(format!("enc_down{stage}"), down.clone())?;
			enc_down.push(down);
			for block in 0..config.depth {
				let first = make_conv(engine, config.width, config.width, 3, 1, 3, 3, &mut rng)?;
				registry.register_module(format!("enc_res{stage}_{block}_a"), first.clone())?;
				enc_res.push(first);
				let second = make_conv(engine, config.width, config.width, 1, 1, 0, 1, &mut rng)?;
				registry.register_module(format!("enc_res{stage}_{block}_b"), second.clone())?;
				enc_res.push(second);
			}
		}
		let enc_out = make_conv(engine, config.width, config.code_dim, 3, 1, 1, 1, &mut rng)?;
		registry.register_module("enc_out", enc_out.clone())?;

		let vq_config = VectorQuantizerConfig {
			num_codes: config.num_codes,
			code_dim: config.code_dim,
			commitment_beta: config.commitment_beta,
			ema_decay: config.ema_decay,
			ema_epsilon: config.ema_epsilon,
			dead_threshold: config.dead_threshold,
			normalize_codes: true,
		};
		let rvq = Rc::new(ResidualVectorQuantizer::with_seed(
			engine, vq_config, 1, rng,
		)?);
		registry.register_module("rvq", rvq.clone())?;

		let dec_in = make_conv(engine, config.code_dim, config.width, 3, 1, 1, 1, &mut rng)?;
		registry.register_module("dec_in", dec_in.clone())?;

		let mut dec_res = Vec::with_capacity(enc_res.capacity());
		let mut dec_up = Vec::with_capacity(config.downsample_stages);
		for stage in 0..config.downsample_stages {
			for block in 0..config.depth {
				let first = make_conv(engine, config.width, config.width, 3, 1, 3, 3, &mut rng)?;
				registry.register_module(format!("dec_res{stage}_{block}_a"), first.clone())?;
				dec_res.push(first);
				let second = make_conv(engine, config.width, config.width, 1, 1, 0, 1, &mut rng)?;
				registry.register_module(format!("dec_res{stage}_{block}_b"), second.clone())?;
				dec_res.push(second);
			}
			let up = make_conv_transpose(engine, config.width, config.width, 4, 2, 1, &mut rng)?;
			registry.register_module(format!("dec_up{stage}"), up.clone())?;
			dec_up.push(up);
		}
		let dec_mid = make_conv(engine, config.width, config.width, 3, 1, 1, 1, &mut rng)?;
		registry.register_module("dec_mid", dec_mid.clone())?;
		let dec_out = make_conv(engine, config.width, config.input_dim, 3, 1, 1, 1, &mut rng)?;
		registry.register_module("dec_out", dec_out.clone())?;

		let encoder_norm_count = 1_usize
			.checked_add(
				config
					.downsample_stages
					.checked_mul(1_usize.checked_add(config.depth).ok_or_else(|| {
						Error::invalid_argument("ALM encoder norm count overflows usize")
					})?)
					.ok_or_else(|| {
						Error::invalid_argument("ALM encoder norm count overflows usize")
					})?,
			)
			.ok_or_else(|| Error::invalid_argument("ALM encoder norm count overflows usize"))?;
		let decoder_norm_count = 2_usize
			.checked_add(
				config
					.downsample_stages
					.checked_mul(1_usize.checked_add(config.depth).ok_or_else(|| {
						Error::invalid_argument("ALM decoder norm count overflows usize")
					})?)
					.ok_or_else(|| {
						Error::invalid_argument("ALM decoder norm count overflows usize")
					})?,
			)
			.ok_or_else(|| Error::invalid_argument("ALM decoder norm count overflows usize"))?;
		let mut enc_norms = Vec::with_capacity(encoder_norm_count);
		for index in 0..encoder_norm_count {
			let norm = Rc::new(LayerNorm::new(engine, config.width, NORMALIZATION_EPSILON)?);
			registry.register_module(format!("enc_ln{index}"), norm.clone())?;
			enc_norms.push(norm);
		}
		let mut dec_norms = Vec::with_capacity(decoder_norm_count);
		for index in 0..decoder_norm_count {
			let norm = Rc::new(LayerNorm::new(engine, config.width, NORMALIZATION_EPSILON)?);
			registry.register_module(format!("dec_ln{index}"), norm.clone())?;
			dec_norms.push(norm);
		}
		registry.register_buffer(
			"latent_rms_weight",
			matrix::ones(engine, [config.code_dim])?,
			false,
		)?;
		let latent_rms_weight = registry
			.buffer_handle("latent_rms_weight")
			.ok_or_else(|| Error::internal("ALM latent RMS weight registration was lost"))?;

		Ok(Self {
			config,
			downsample_factor,
			enc_in,
			enc_down,
			enc_res,
			enc_out,
			dec_in,
			dec_res,
			dec_up,
			dec_mid,
			dec_out,
			enc_norms,
			dec_norms,
			latent_rms_weight,
			rvq,
			registry,
		})
	}

	/// Encode `[batch, frames, input_dim]` motion into unit-RMS
	/// `[batch * tokens, code_dim]` latent rows.
	///
	/// # Errors
	///
	/// Returns an error unless the input is nonempty rank-three F32, its feature
	/// width matches the configuration, and its frame count is divisible by the
	/// downsample factor, or a recorded operation fails.
	pub fn encode(&self, input: &Matrix) -> Result<Matrix> {
		let [batch, frames, input_dim] = input.shape() else {
			return Err(Error::invalid_argument(
				"ALM tokenizer input must have shape [batch,frames,input_dim]",
			));
		};
		if *batch == 0
			|| *frames == 0
			|| *input_dim != self.config.input_dim
			|| input.dtype() != DType::F32
			|| frames % self.downsample_factor != 0
		{
			return Err(Error::invalid_argument(
				"ALM tokenizer input must be nonempty F32 with configured feature width and frame count divisible by its downsample factor",
			));
		}
		let channels_first = matrix::transpose(input, 1, 2)?;
		let mut norm_cursor = 0;
		let mut hidden = self.enc_in.forward(&channels_first)?;
		hidden = self.channel_norm_relu(&self.enc_norms[norm_cursor], &hidden)?;
		norm_cursor += 1;
		let mut residual_cursor = 0;
		for stage in 0..self.config.downsample_stages {
			hidden = self.enc_down[stage].forward(&hidden)?;
			hidden = self.channel_norm_relu(&self.enc_norms[norm_cursor], &hidden)?;
			norm_cursor += 1;
			hidden = self.residual_stack(
				&self.enc_res,
				&mut residual_cursor,
				&self.enc_norms,
				&mut norm_cursor,
				hidden,
			)?;
		}
		let latent = self.enc_out.forward(&hidden)?;
		let latent = matrix::transpose(&latent, 1, 2)?;
		let token_count = batch
			.checked_mul(frames / self.downsample_factor)
			.ok_or_else(|| Error::invalid_argument("ALM token count overflows usize"))?;
		let latent = matrix::reshape(&latent, [token_count, self.config.code_dim])?;
		ml_matrix::rms_norm(
			&latent,
			&self.latent_rms_weight.data(),
			NORMALIZATION_EPSILON,
		)
	}

	/// Quantize encoded latent rows with one straight-through EMA codebook level.
	///
	/// # Errors
	///
	/// Returns an error for incompatible latent geometry or failed recording.
	pub fn quantize(&self, latent: &Matrix) -> Result<ResidualVqResult> {
		self.rvq.quantize(latent)
	}

	/// Decode `[batch * tokens, code_dim]` latent rows into
	/// `[batch, tokens * downsample_factor, input_dim]` motion.
	///
	/// # Errors
	///
	/// Returns an error for zero batch size, incompatible latent geometry, count
	/// overflow, or a failed recorded operation.
	pub fn decode(&self, latent: &Matrix, batch: usize) -> Result<Matrix> {
		let [rows, code_dim] = latent.shape() else {
			return Err(Error::invalid_argument(
				"ALM decoder latent must have shape [batch*tokens,code_dim]",
			));
		};
		if batch == 0
			|| *rows == 0
			|| rows % batch != 0
			|| *code_dim != self.config.code_dim
			|| latent.dtype() != DType::F32
		{
			return Err(Error::invalid_argument(
				"ALM decoder requires nonempty F32 latent rows divisible by batch with configured code width",
			));
		}
		let tokens = rows / batch;
		let latent = matrix::reshape(latent, [batch, tokens, self.config.code_dim])?;
		let channels_first = matrix::transpose(&latent, 1, 2)?;
		let mut norm_cursor = 0;
		let mut hidden = self.dec_in.forward(&channels_first)?;
		hidden = self.channel_norm_relu(&self.dec_norms[norm_cursor], &hidden)?;
		norm_cursor += 1;
		let mut residual_cursor = 0;
		for stage in 0..self.config.downsample_stages {
			hidden = self.residual_stack(
				&self.dec_res,
				&mut residual_cursor,
				&self.dec_norms,
				&mut norm_cursor,
				hidden,
			)?;
			hidden = self.dec_up[stage].forward(&hidden)?;
			hidden = self.channel_norm(&self.dec_norms[norm_cursor], &hidden)?;
			norm_cursor += 1;
		}
		hidden = self.dec_mid.forward(&hidden)?;
		hidden = self.channel_norm_relu(&self.dec_norms[norm_cursor], &hidden)?;
		let output = self.dec_out.forward(&hidden)?;
		matrix::transpose(&output, 1, 2)
	}

	/// Encode motion and return its one-level I32 token stream.
	///
	/// # Errors
	///
	/// Returns an error from encoding or quantization.
	pub fn tokenize(&self, input: &Matrix) -> Result<Vec<Matrix>> {
		Ok(self.quantize(&self.encode(input)?)?.indices)
	}

	/// Look up one-level tokens and decode them into rank-three motion.
	///
	/// # Errors
	///
	/// Returns an error for invalid token levels, batch geometry, or failed
	/// lookup/decoding.
	pub fn detokenize(&self, indices: &[Matrix], batch: usize) -> Result<Matrix> {
		self.decode(&self.rvq.lookup(indices)?, batch)
	}

	/// Advance all codebook EMA state from this step's quantization result.
	///
	/// # Errors
	///
	/// Returns an error for a mismatched result or failed runtime operation.
	pub fn ema_update(&self, result: &ResidualVqResult) -> Result<()> {
		self.rvq.ema_update(result)
	}

	/// Seed the codebook from a warm encoded batch.
	///
	/// # Errors
	///
	/// Returns an error unless enough compatible latent rows are supplied or a
	/// runtime operation fails.
	pub fn seed(&self, latent: &Matrix) -> Result<()> {
		self.rvq.seed(latent)
	}

	/// Return the temporal reduction/expansion factor.
	pub const fn downsample_factor(&self) -> usize {
		self.downsample_factor
	}

	/// Return the immutable tokenizer configuration.
	pub const fn config(&self) -> &AlmTokenizerConfig {
		&self.config
	}

	/// Return the single-level residual quantizer.
	pub fn rvq(&self) -> &ResidualVectorQuantizer {
		&self.rvq
	}

	fn channel_norm(&self, norm: &LayerNorm, input: &Matrix) -> Result<Matrix> {
		let channels_last = matrix::transpose(input, 1, 2)?;
		matrix::transpose(&norm.forward(&channels_last)?, 1, 2)
	}

	fn channel_norm_relu(&self, norm: &LayerNorm, input: &Matrix) -> Result<Matrix> {
		ml_matrix::relu(&self.channel_norm(norm, input)?)
	}

	fn residual_stack(
		&self,
		convolutions: &[Rc<Conv1d>],
		convolution_cursor: &mut usize,
		norms: &[Rc<LayerNorm>],
		norm_cursor: &mut usize,
		mut hidden: Matrix,
	) -> Result<Matrix> {
		for _ in 0..self.config.depth {
			let mut branch = self.channel_norm_relu(&norms[*norm_cursor], &hidden)?;
			*norm_cursor += 1;
			branch = convolutions[*convolution_cursor].forward(&branch)?;
			*convolution_cursor += 1;
			branch = ml_matrix::relu(&branch)?;
			branch = convolutions[*convolution_cursor].forward(&branch)?;
			*convolution_cursor += 1;
			branch = ml_matrix::relu(&branch)?;
			hidden = matrix::add(&hidden, &branch)?;
		}
		Ok(hidden)
	}
}

impl Module for AlmTokenizer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let [batch, ..] = input.shape() else {
			return Err(Error::invalid_argument(
				"ALM tokenizer forward requires rank-three input",
			));
		};
		let quantized = self.quantize(&self.encode(input)?)?;
		self.decode(&quantized.quantized, *batch)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

#[allow(
	clippy::too_many_arguments,
	reason = "Conv1d donor geometry and one shared initializer stream are explicit"
)]
fn make_conv(
	engine: &Engine,
	input_channels: usize,
	output_channels: usize,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	dilation: usize,
	rng: &mut u64,
) -> Result<Rc<Conv1d>> {
	let values = glorot_values(input_channels, output_channels, kernel_size, rng)?;
	let weight = Matrix::from_f32(
		engine,
		[output_channels, input_channels, kernel_size],
		&values,
	)?;
	let bias = Matrix::from_f32(engine, [output_channels], &vec![0.0; output_channels])?;
	Ok(Rc::new(Conv1d::from_matrices(
		weight, bias, stride, padding, dilation,
	)?))
}

fn make_conv_transpose(
	engine: &Engine,
	input_channels: usize,
	output_channels: usize,
	kernel_size: usize,
	stride: usize,
	padding: usize,
	rng: &mut u64,
) -> Result<Rc<ConvTranspose1d>> {
	let values = glorot_values(output_channels, input_channels, kernel_size, rng)?;
	let weight = Matrix::from_f32(
		engine,
		[input_channels, output_channels, kernel_size],
		&values,
	)?;
	Ok(Rc::new(ConvTranspose1d::from_matrix(
		weight, stride, padding,
	)?))
}

fn glorot_values(
	input_channels: usize,
	output_channels: usize,
	kernel_size: usize,
	rng: &mut u64,
) -> Result<Vec<f32>> {
	let count = input_channels
		.checked_mul(output_channels)
		.and_then(|value| value.checked_mul(kernel_size))
		.ok_or_else(|| Error::invalid_argument("ALM convolution weight count overflows usize"))?;
	let denominator = input_channels
		.checked_add(output_channels)
		.and_then(|value| value.checked_mul(kernel_size))
		.ok_or_else(|| Error::invalid_argument("ALM Glorot extent overflows usize"))?;
	let bound = (6.0_f32 / denominator as f32).sqrt();
	let divisor = (u32::MAX >> 1) as f32;
	Ok((0..count)
		.map(|_| {
			*rng = rng
				.wrapping_mul(6_364_136_223_846_793_005)
				.wrapping_add(1_442_695_040_888_963_407);
			let uniform = ((*rng >> 33) as u32) as f32 / divisor;
			((uniform * 2.0) - 1.0) * bound
		})
		.collect())
}

fn validate_config(config: AlmTokenizerConfig) -> Result<()> {
	if config.input_dim == 0
		|| config.width == 0
		|| config.code_dim == 0
		|| config.num_codes == 0
		|| config.downsample_stages == 0
		|| config.depth == 0
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
			"invalid ALM tokenizer configuration",
		));
	}
	Ok(())
}
