use std::rc::Rc;

use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::super::super::{Module, ModuleRegistry, Parameter, autograd};
use super::super::Linear;
use super::{FlowTimeEmbedding, FlowTransformer, FlowTransformerConfig};

/// Configuration for a modality-independent flow/diffusion denoiser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowDenoiserConfig {
	pub input_dim: usize,
	pub condition_dim: usize,
	pub backbone: FlowTransformerConfig,
	pub time_max_period: f32,
	pub time_scale: f32,
	/// Per-sample conditioning dropout used only in training mode.
	pub condition_dropout_probability: f32,
}

impl Default for FlowDenoiserConfig {
	fn default() -> Self {
		Self {
			input_dim: 0,
			condition_dim: 0,
			backbone: FlowTransformerConfig::default(),
			time_max_period: 10_000.0,
			time_scale: 1_000.0,
			condition_dropout_probability: 0.0,
		}
	}
}

/// Trainable flow/diffusion denoiser with optional condition features and CFG.
pub struct FlowDenoiser {
	config: FlowDenoiserConfig,
	input_projection: Rc<Linear>,
	time_embedding: Rc<FlowTimeEmbedding>,
	condition_projection: Option<Rc<Linear>>,
	backbone: Rc<FlowTransformer>,
	output_projection: Rc<Linear>,
	position: Parameter,
	registry: ModuleRegistry,
}

impl FlowDenoiser {
	/// Construct a deterministic dense or MoE denoiser.
	///
	/// # Errors
	///
	/// Returns an error for invalid model/time/dropout geometry or failed child,
	/// parameter, and buffer construction.
	pub fn with_seed(engine: &Engine, config: FlowDenoiserConfig, seed: u64) -> Result<Self> {
		if config.input_dim == 0
			|| config.backbone.model_width == 0
			|| config.backbone.sequence_length == 0
			|| !config.time_max_period.is_finite()
			|| config.time_max_period <= 1.0
			|| !config.time_scale.is_finite()
			|| config.time_scale <= 0.0
			|| !config.condition_dropout_probability.is_finite()
			|| !(0.0..1.0).contains(&config.condition_dropout_probability)
		{
			return Err(Error::invalid_argument(
				"FlowDenoiser requires positive input/model/sequence/time geometry and finite condition dropout in [0, 1)",
			));
		}
		if !config.backbone.adaptive_conditioning {
			return Err(Error::invalid_argument(
				"FlowDenoiser requires adaptive conditioning in its backbone",
			));
		}

		let input_projection = Rc::new(Linear::with_seed(
			engine,
			config.input_dim,
			config.backbone.model_width,
			seed,
		)?);
		let time_embedding = Rc::new(FlowTimeEmbedding::new(
			engine,
			config.backbone.model_width,
			config.time_max_period,
			config.time_scale,
		)?);
		let condition_projection = if config.condition_dim == 0 {
			None
		} else {
			Some(Rc::new(Linear::with_seed(
				engine,
				config.condition_dim,
				config.backbone.model_width,
				seed.wrapping_add(1),
			)?))
		};
		let backbone = Rc::new(FlowTransformer::with_seed(
			engine,
			config.backbone,
			seed.wrapping_add(2),
		)?);
		let output_projection = Rc::new(Linear::with_seed(
			engine,
			config.backbone.model_width,
			config.input_dim,
			seed.wrapping_add(3),
		)?);
		let position_count = config
			.backbone
			.sequence_length
			.checked_mul(config.backbone.model_width)
			.ok_or_else(|| Error::invalid_argument("FlowDenoiser position size overflows usize"))?;
		let position_template = Matrix::allocate(
			&engine.handle(),
			vec![config.backbone.sequence_length, config.backbone.model_width],
			position_count,
			DType::F32,
		)?;
		let position = matrix::philox_normal(&position_template, 0.0, 0.02, seed.wrapping_add(4))?;
		let position = Parameter::new("position", position)?;

		let mut registry = ModuleRegistry::new();
		registry.register_parameter("position", position.clone())?;
		registry.register_module("input_projection", input_projection.clone())?;
		registry.register_module("time_embedding", time_embedding.clone())?;
		if let Some(condition_projection) = &condition_projection {
			registry.register_module("condition_projection", condition_projection.clone())?;
		}
		registry.register_module("backbone", backbone.clone())?;
		registry.register_module("output_projection", output_projection.clone())?;
		Ok(Self {
			config,
			input_projection,
			time_embedding,
			condition_projection,
			backbone,
			output_projection,
			position,
			registry,
		})
	}

	/// Denoise using zero time and, when configured, zero condition features.
	pub fn forward(&self, sample: &Matrix) -> Result<Matrix> {
		let batch = self.validate_sample(sample)?;
		let time = zero_matrix(sample, vec![batch, 1])?;
		let condition = if self.config.condition_dim == 0 {
			None
		} else {
			Some(zero_matrix(sample, vec![batch, self.config.condition_dim])?)
		};
		self.forward_conditioned(sample, &time, condition.as_ref(), None)
	}

	/// Denoise a sample with explicit time, optional condition, and token mask.
	///
	/// # Errors
	///
	/// Returns an error unless inputs match the configured same-engine F32
	/// contracts or a recorded child operation fails.
	pub fn forward_conditioned(
		&self,
		sample: &Matrix,
		time: &Matrix,
		condition: Option<&Matrix>,
		token_mask: Option<&Matrix>,
	) -> Result<Matrix> {
		let batch = self.validate_sample(sample)?;
		let time_valid =
			matches!(time.shape(), [value] if *value == batch) || time.shape() == [batch, 1];
		if !time_valid
			|| time.dtype() != DType::F32
			|| !sample.engine_handle().same_as(time.engine_handle())
		{
			return Err(Error::invalid_argument(
				"FlowDenoiser time must be same-engine F32 [B] or [B,1]",
			));
		}
		match (self.config.condition_dim, condition) {
			(0, None) => {}
			(0, Some(_)) => {
				return Err(Error::invalid_argument(
					"FlowDenoiser was configured without condition features",
				));
			}
			(width, Some(value))
				if value.shape() == [batch, width]
					&& value.dtype() == DType::F32
					&& sample.engine_handle().same_as(value.engine_handle()) => {}
			_ => {
				return Err(Error::invalid_argument(
					"FlowDenoiser condition must be same-engine F32 [B,condition_dim]",
				));
			}
		}

		let rows = batch
			.checked_mul(self.config.backbone.sequence_length)
			.ok_or_else(|| Error::invalid_argument("FlowDenoiser row count overflows usize"))?;
		let sample_rows = matrix::reshape(sample, [rows, self.config.input_dim])?;
		let tokens = self.input_projection.forward(&sample_rows)?;
		let tokens = matrix::reshape(
			&tokens,
			[
				batch,
				self.config.backbone.sequence_length,
				self.config.backbone.model_width,
			],
		)?;
		autograd::record_parameter_leaf(&self.position)?;
		let position = matrix::reshape(
			&self.position.data(),
			[
				1,
				self.config.backbone.sequence_length,
				self.config.backbone.model_width,
			],
		)?;
		let tokens = matrix::add(&tokens, &position)?;
		let mut context = self.time_embedding.forward(time)?;
		if let (Some(projection), Some(condition)) = (&self.condition_projection, condition) {
			let condition = if self.is_training() && self.config.condition_dropout_probability > 0.0 {
				let seed_column = matrix::slice(condition, 1, 0, 1)?;
				let ones = matrix::add_scalar(&matrix::scale(&seed_column, 0.0)?, 1.0)?;
				let keep = matrix::scale(
					&matrix::dropout(&ones, self.config.condition_dropout_probability, 0)?,
					1.0 - self.config.condition_dropout_probability,
				)?;
				matrix::mul(condition, &keep)?
			} else {
				condition.clone()
			};
			context = matrix::add(&context, &projection.forward(&condition)?)?;
		}
		let hidden = self
			.backbone
			.forward_conditioned(&tokens, &context, token_mask)?;
		let hidden = matrix::reshape(&hidden, [rows, self.config.backbone.model_width])?;
		let output = self.output_projection.forward(&hidden)?;
		matrix::reshape(&output, sample.shape().to_vec())
	}

	/// Apply classifier-free guidance in a bounded evaluation-mode scope.
	pub fn forward_guided(
		&self,
		sample: &Matrix,
		time: &Matrix,
		condition: &Matrix,
		guidance_scale: f32,
		token_mask: Option<&Matrix>,
	) -> Result<Matrix> {
		if self.condition_projection.is_none() || !guidance_scale.is_finite() || guidance_scale < 0.0 {
			return Err(Error::invalid_argument(
				"FlowDenoiser guidance requires configured conditions and a finite nonnegative scale",
			));
		}
		let _eval = self.scoped_eval();
		let unconditional_condition = zero_matrix(condition, condition.shape().to_vec())?;
		let unconditional =
			self.forward_conditioned(sample, time, Some(&unconditional_condition), token_mask)?;
		let conditional = self.forward_conditioned(sample, time, Some(condition), token_mask)?;
		matrix::add(
			&unconditional,
			&matrix::scale(&matrix::sub(&conditional, &unconditional)?, guidance_scale)?,
		)
	}

	fn validate_sample(&self, sample: &Matrix) -> Result<usize> {
		let [batch, sequence, features] = sample.shape() else {
			return Err(Error::invalid_argument(
				"FlowDenoiser sample must be F32 [B,S,input_dim]",
			));
		};
		if *batch == 0
			|| *sequence != self.config.backbone.sequence_length
			|| *features != self.config.input_dim
			|| sample.dtype() != DType::F32
		{
			return Err(Error::invalid_argument(
				"FlowDenoiser sample must be nonempty F32 [B,S,input_dim] matching its configuration",
			));
		}
		Ok(*batch)
	}

	pub const fn config(&self) -> FlowDenoiserConfig {
		self.config
	}

	pub fn is_moe(&self) -> bool {
		self.backbone.is_moe()
	}

	pub fn backbone(&self) -> &FlowTransformer {
		&self.backbone
	}

	pub fn position(&self) -> Parameter {
		self.position.clone()
	}
}

impl Module for FlowDenoiser {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		FlowDenoiser::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

fn zero_matrix(reference: &Matrix, shape: Vec<usize>) -> Result<Matrix> {
	let count = shape.iter().try_fold(1_usize, |count, extent| {
		count
			.checked_mul(*extent)
			.ok_or_else(|| Error::invalid_argument("zero Matrix shape overflows usize"))
	})?;
	Matrix::allocate(reference.engine_handle(), shape, count, DType::F32)
}
