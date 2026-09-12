use std::{cell::Cell, rc::Rc};

use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::super::super::{Module, ModuleRegistry};
use super::super::{AttentionMode, LayerNorm, TransformerBlock};

/// Configuration for a bidirectional flow/diffusion Transformer backbone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowTransformerConfig {
	pub model_width: usize,
	pub hidden_width: usize,
	pub sequence_length: usize,
	pub num_layers: usize,
	pub num_heads: usize,
	pub num_experts: usize,
	pub experts_per_token: usize,
	pub epsilon: f32,
	pub adaptive_conditioning: bool,
}

impl Default for FlowTransformerConfig {
	fn default() -> Self {
		Self {
			model_width: 0,
			hidden_width: 0,
			sequence_length: 0,
			num_layers: 1,
			num_heads: 1,
			num_experts: 0,
			experts_per_token: 0,
			epsilon: 1.0e-5,
			adaptive_conditioning: true,
		}
	}
}

/// Reusable bidirectional Transformer backbone for flow/diffusion denoisers.
pub struct FlowTransformer {
	config: Cell<FlowTransformerConfig>,
	blocks: Vec<Rc<TransformerBlock>>,
	output_norm: Rc<LayerNorm>,
	registry: ModuleRegistry,
}

impl FlowTransformer {
	/// Construct a deterministic dense or dropless-MoE flow Transformer.
	///
	/// # Errors
	///
	/// Returns an error for invalid geometry, expert routing, epsilon, or failed
	/// child construction and registration.
	pub fn with_seed(engine: &Engine, config: FlowTransformerConfig, seed: u64) -> Result<Self> {
		validate_config(config)?;
		let mut registry = ModuleRegistry::new();
		let mut blocks = Vec::with_capacity(config.num_layers);
		for index in 0..config.num_layers {
			let block_seed = seed.wrapping_add((index as u64).wrapping_mul(16));
			let block = if config.num_experts == 0 {
				if config.adaptive_conditioning {
					TransformerBlock::with_seed_conditioned(
						engine,
						config.model_width,
						config.hidden_width,
						config.sequence_length,
						config.num_heads,
						config.model_width,
						config.epsilon,
						block_seed,
					)?
				} else {
					TransformerBlock::with_seed(
						engine,
						config.model_width,
						config.hidden_width,
						config.sequence_length,
						config.num_heads,
						config.epsilon,
						block_seed,
					)?
				}
			} else if config.adaptive_conditioning {
				TransformerBlock::with_seed_moe_conditioned(
					engine,
					config.model_width,
					config.hidden_width,
					config.sequence_length,
					config.num_heads,
					config.num_experts,
					config.experts_per_token,
					config.model_width,
					config.epsilon,
					block_seed,
				)?
			} else {
				TransformerBlock::with_seed_moe(
					engine,
					config.model_width,
					config.hidden_width,
					config.sequence_length,
					config.num_heads,
					config.num_experts,
					config.experts_per_token,
					config.epsilon,
					block_seed,
				)?
			};
			block.set_attention_mode(AttentionMode::Bidirectional);
			let block = Rc::new(block);
			registry.register_module(format!("block_{index}"), block.clone())?;
			blocks.push(block);
		}
		let output_norm = Rc::new(LayerNorm::new(engine, config.model_width, config.epsilon)?);
		registry.register_module("output_norm", output_norm.clone())?;
		Ok(Self {
			config: Cell::new(config),
			blocks,
			output_norm,
			registry,
		})
	}

	/// Apply the unmasked backbone to rank-two or rank-three token state.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		self.forward_impl(tokens, None, None)
	}

	/// Apply the backbone with a binary key mask shaped `[B,S]` or `[B,S,1]`.
	pub fn forward_masked(&self, tokens: &Matrix, token_mask: &Matrix) -> Result<Matrix> {
		self.forward_impl(tokens, Some(token_mask), None)
	}

	/// Apply adaptive conditioning with an optional binary key mask.
	pub fn forward_conditioned(
		&self,
		tokens: &Matrix,
		condition: &Matrix,
		token_mask: Option<&Matrix>,
	) -> Result<Matrix> {
		self.forward_impl(tokens, token_mask, Some(condition))
	}

	fn forward_impl(
		&self,
		tokens: &Matrix,
		token_mask: Option<&Matrix>,
		condition: Option<&Matrix>,
	) -> Result<Matrix> {
		let config = self.config.get();
		let (batch, rows, batched) = match tokens.shape() {
			[rows, features]
				if *features == config.model_width
					&& *rows != 0 && rows.is_multiple_of(config.sequence_length) =>
			{
				(rows / config.sequence_length, *rows, false)
			}
			[batch, sequence, features]
				if *features == config.model_width
					&& *sequence == config.sequence_length
					&& *batch != 0 =>
			{
				(
					*batch,
					batch.checked_mul(*sequence).ok_or_else(|| {
						Error::invalid_argument("FlowTransformer token row count overflows usize")
					})?,
					true,
				)
			}
			_ => {
				return Err(Error::invalid_argument(
					"FlowTransformer expects F32 [B*S,D] or [B,S,D] matching its configured geometry",
				));
			}
		};
		if tokens.dtype() != DType::F32 {
			return Err(Error::invalid_argument(
				"FlowTransformer tokens must be F32",
			));
		}
		if let Some(condition) = condition
			&& (!config.adaptive_conditioning
				|| condition.shape() != [batch, config.model_width]
				|| condition.dtype() != DType::F32
				|| !tokens.engine_handle().same_as(condition.engine_handle()))
		{
			return Err(Error::invalid_argument(
				"FlowTransformer condition requires enabled same-engine F32 [B,D] adaptive conditioning",
			));
		}

		let additive_mask = token_mask
			.map(|mask| self.additive_mask(tokens, mask, batch, config))
			.transpose()?;
		let mut output = if batched {
			matrix::reshape(tokens, [rows, config.model_width])?
		} else {
			tokens.clone()
		};
		for block in &self.blocks {
			output = match (condition, additive_mask.as_ref()) {
				(Some(condition), Some(mask)) => {
					block.forward_conditioned_masked(&output, condition, mask)?
				}
				(Some(condition), None) => block.forward_conditioned(&output, condition)?,
				(None, Some(mask)) => block.forward_masked(&output, mask)?,
				(None, None) => block.forward(&output)?,
			};
		}
		let output = self.output_norm.forward(&output)?;
		if batched {
			matrix::reshape(&output, tokens.shape().to_vec())
		} else {
			Ok(output)
		}
	}

	fn additive_mask(
		&self,
		tokens: &Matrix,
		mask: &Matrix,
		batch: usize,
		config: FlowTransformerConfig,
	) -> Result<Matrix> {
		let valid_shape = mask.shape() == [batch, config.sequence_length]
			|| mask.shape() == [batch, config.sequence_length, 1];
		if !valid_shape
			|| mask.dtype() != DType::F32
			|| !tokens.engine_handle().same_as(mask.engine_handle())
		{
			return Err(Error::invalid_argument(
				"FlowTransformer token mask must be same-engine F32 [B,S] or [B,S,1]",
			));
		}
		let key_mask = matrix::reshape(mask, [batch, 1, config.sequence_length])?;
		let key_mask = matrix::scale(&matrix::sub_scalar(&key_mask, 1.0)?, 1.0e4)?;
		let repeats = config
			.num_heads
			.checked_mul(config.sequence_length)
			.ok_or_else(|| {
				Error::invalid_argument("FlowTransformer mask repeat count overflows")
			})?;
		let additive = matrix::repeat_interleave(&key_mask, repeats, 1)?;
		let mask_rows = batch
			.checked_mul(repeats)
			.ok_or_else(|| Error::invalid_argument("FlowTransformer mask row count overflows"))?;
		matrix::reshape(&additive, [mask_rows, config.sequence_length])
	}

	/// Update sequence geometry without rebuilding parameters.
	pub fn set_sequence_length(&self, sequence_length: usize) -> Result<()> {
		if sequence_length == 0 {
			return Err(Error::invalid_argument(
				"FlowTransformer sequence length must be positive",
			));
		}
		for block in &self.blocks {
			block.set_sequence_length(sequence_length)?;
		}
		let mut config = self.config.get();
		config.sequence_length = sequence_length;
		self.config.set(config);
		Ok(())
	}

	pub fn config(&self) -> FlowTransformerConfig {
		self.config.get()
	}

	pub fn is_moe(&self) -> bool {
		self.config.get().num_experts > 0
	}

	pub fn num_layers(&self) -> usize {
		self.blocks.len()
	}

	pub fn block(&self, index: usize) -> Option<&TransformerBlock> {
		self.blocks.get(index).map(Rc::as_ref)
	}
}

impl Module for FlowTransformer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		FlowTransformer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

fn validate_config(config: FlowTransformerConfig) -> Result<()> {
	let dense = config.num_experts == 0 && config.experts_per_token == 0;
	let moe = config.num_experts > 0
		&& config.experts_per_token > 0
		&& config.experts_per_token <= config.num_experts;
	if config.model_width == 0
		|| config.hidden_width == 0
		|| config.sequence_length == 0
		|| config.num_layers == 0
		|| config.num_heads == 0
		|| !config.model_width.is_multiple_of(config.num_heads)
		|| !config.epsilon.is_finite()
		|| config.epsilon <= 0.0
		|| !(dense || moe)
	{
		return Err(Error::invalid_argument(
			"FlowTransformer requires positive geometry, D divisible by H, finite positive epsilon, and either 0/0 dense experts or 0 < K <= E",
		));
	}
	Ok(())
}
