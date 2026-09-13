use std::rc::Rc;

use crate::ml::{
	Module, ModuleRegistry, matrix as ml_matrix,
	nn::{Embedding, LayerNorm, Linear, MultiHeadAttention},
};
use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::ClipTokenizer;

/// Frozen CLIP text-tower architecture consumed by conditioned ALM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipTextConfig {
	pub vocab_size: usize,
	pub context_length: usize,
	pub hidden_size: usize,
	pub intermediate_size: usize,
	pub num_heads: usize,
	pub num_layers: usize,
	pub projection_dim: usize,
	pub layer_norm_epsilon: f32,
	pub quick_gelu_alpha: f32,
	pub bos_token: i32,
	pub eos_token: i32,
	pub pad_token: i32,
}

impl Default for ClipTextConfig {
	fn default() -> Self {
		Self {
			vocab_size: 49_408,
			context_length: 77,
			hidden_size: 768,
			intermediate_size: 3072,
			num_heads: 12,
			num_layers: 12,
			projection_dim: 768,
			layer_norm_epsilon: 1e-5,
			quick_gelu_alpha: 1.702,
			bos_token: 49_406,
			eos_token: 49_407,
			pad_token: 49_407,
		}
	}
}

struct ClipResidualBlock {
	norm_attention: Rc<LayerNorm>,
	attention: Rc<MultiHeadAttention>,
	norm_mlp: Rc<LayerNorm>,
	fc1: Rc<Linear>,
	fc2: Rc<Linear>,
	quick_gelu_alpha: f32,
	registry: ModuleRegistry,
}

impl ClipResidualBlock {
	fn new(engine: &Engine, config: ClipTextConfig, seed: u64) -> Result<Self> {
		let norm_attention = Rc::new(LayerNorm::new(
			engine,
			config.hidden_size,
			config.layer_norm_epsilon,
		)?);
		let attention = Rc::new(MultiHeadAttention::with_seed(
			engine,
			config.hidden_size,
			config.num_heads,
			config.context_length,
			seed,
		)?);
		let norm_mlp = Rc::new(LayerNorm::new(
			engine,
			config.hidden_size,
			config.layer_norm_epsilon,
		)?);
		let fc1 = Rc::new(Linear::with_seed(
			engine,
			config.hidden_size,
			config.intermediate_size,
			seed.wrapping_add(4),
		)?);
		let fc2 = Rc::new(Linear::with_seed(
			engine,
			config.intermediate_size,
			config.hidden_size,
			seed.wrapping_add(5),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("layer_norm1", norm_attention.clone())?;
		registry.register_module("self_attn", attention.clone())?;
		registry.register_module("layer_norm2", norm_mlp.clone())?;
		registry.register_module("mlp_fc1", fc1.clone())?;
		registry.register_module("mlp_fc2", fc2.clone())?;
		Ok(Self {
			norm_attention,
			attention,
			norm_mlp,
			fc1,
			fc2,
			quick_gelu_alpha: config.quick_gelu_alpha,
			registry,
		})
	}
}

impl Module for ClipResidualBlock {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let attention = self
			.attention
			.forward(&self.norm_attention.forward(input)?)?;
		let residual = matrix::add(input, &attention)?;
		let hidden = self.fc1.forward(&self.norm_mlp.forward(&residual)?)?;
		let quick_gelu = matrix::mul(
			&hidden,
			&ml_matrix::sigmoid(&matrix::scale(&hidden, self.quick_gelu_alpha)?)?,
		)?;
		matrix::add(&residual, &self.fc2.forward(&quick_gelu)?)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Exact frozen CLIP text-with-projection tower used by conditioned ALM.
pub struct ClipText {
	config: ClipTextConfig,
	token_embedding: Rc<Embedding>,
	position_embedding: Rc<Embedding>,
	layers: Vec<Rc<ClipResidualBlock>>,
	final_layer_norm: Rc<LayerNorm>,
	text_projection: Rc<Linear>,
	position_ids: Matrix,
	registry: ModuleRegistry,
}

impl ClipText {
	pub fn with_seed(engine: &Engine, config: ClipTextConfig, seed: u64) -> Result<Self> {
		validate_config(config)?;
		let token_embedding = Rc::new(Embedding::with_seed(
			engine,
			config.vocab_size,
			config.hidden_size,
			seed,
		)?);
		let position_embedding = Rc::new(Embedding::with_seed(
			engine,
			config.context_length,
			config.hidden_size,
			seed.wrapping_add(1),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("token_embedding", token_embedding.clone())?;
		registry.register_module("position_embedding", position_embedding.clone())?;
		let mut layers = Vec::with_capacity(config.num_layers);
		for index in 0..config.num_layers {
			let layer = Rc::new(ClipResidualBlock::new(
				engine,
				config,
				seed.wrapping_add(2 + (index as u64).wrapping_mul(6)),
			)?);
			registry.register_module(format!("layer_{index}"), layer.clone())?;
			layers.push(layer);
		}
		let final_layer_norm = Rc::new(LayerNorm::new(
			engine,
			config.hidden_size,
			config.layer_norm_epsilon,
		)?);
		let text_projection = Rc::new(Linear::with_seed_and_bias(
			engine,
			config.hidden_size,
			config.projection_dim,
			false,
			seed.wrapping_add(2 + (config.num_layers as u64).wrapping_mul(6)),
		)?);
		registry.register_module("final_layer_norm", final_layer_norm.clone())?;
		registry.register_module("text_projection", text_projection.clone())?;
		let position_ids = Matrix::from_slice(
			engine,
			[config.context_length],
			&(0..config.context_length)
				.map(|value| i32::try_from(value).expect("validated CLIP context fits I32"))
				.collect::<Vec<_>>(),
		)?;
		let model = Self {
			config,
			token_embedding,
			position_embedding,
			layers,
			final_layer_norm,
			text_projection,
			position_ids,
			registry,
		};
		model.freeze()?;
		Ok(model)
	}

	pub fn new(engine: &Engine, config: ClipTextConfig) -> Result<Self> {
		Self::with_seed(engine, config, 0)
	}

	/// Evaluate pre-tokenized fixed-context text and gather tokenizer-provided EOS rows.
	pub fn forward_tokens(&self, token_ids: &Matrix, flat_eos_rows: &Matrix) -> Result<Matrix> {
		let [batch, context] = token_ids.shape() else {
			return Err(Error::invalid_argument(
				"CLIP token ids must have shape [batch,context]",
			));
		};
		if *batch == 0
			|| *context != self.config.context_length
			|| !matches!(token_ids.dtype(), DType::I32 | DType::U32)
			|| flat_eos_rows.shape() != [*batch]
			|| !matches!(flat_eos_rows.dtype(), DType::I32 | DType::U32)
			|| !token_ids
				.engine_handle()
				.same_as(flat_eos_rows.engine_handle())
		{
			return Err(Error::invalid_argument(
				"CLIP requires same-engine I32/U32 [batch,context] tokens and [batch] EOS rows",
			));
		}
		let token = self.token_embedding.forward(token_ids)?;
		let position = self.position_embedding.forward(&self.position_ids)?;
		let mut hidden = matrix::reshape(
			&matrix::add(&token, &position)?,
			[batch * context, self.config.hidden_size],
		)?;
		for layer in &self.layers {
			hidden = layer.forward(&hidden)?;
		}
		hidden = self.final_layer_norm.forward(&hidden)?;
		self.text_projection
			.forward(&matrix::gather(&hidden, flat_eos_rows)?)
	}

	/// Tokenize raw prompts with an explicit CLIP merge asset and encode them.
	///
	/// # Errors
	///
	/// Returns an error from host tokenization, upload, or text-tower execution.
	pub fn forward_prompts<S: AsRef<str>>(
		&self,
		tokenizer: &ClipTokenizer,
		prompts: &[S],
		truncate: bool,
	) -> Result<Matrix> {
		let batch = tokenizer.encode(prompts, self.config.context_length, truncate)?;
		let tokens = Matrix::from_slice_handle(
			self.position_ids.engine_handle(),
			vec![batch.batch, batch.context_length],
			&batch.token_ids,
		)?;
		let eos = Matrix::from_slice_handle(
			self.position_ids.engine_handle(),
			vec![batch.batch],
			&batch.flat_eos_rows,
		)?;
		self.forward_tokens(&tokens, &eos)
	}

	/// Freeze every parameter recursively and discard any accumulated gradients.
	pub fn freeze(&self) -> Result<()> {
		for parameter in self.all_parameters()? {
			parameter.set_requires_grad(false);
		}
		Ok(())
	}

	pub const fn config(&self) -> &ClipTextConfig {
		&self.config
	}
}

impl Module for ClipText {
	fn forward(&self, token_ids: &Matrix) -> Result<Matrix> {
		let [batch, context] = token_ids.shape() else {
			return Err(Error::invalid_argument(
				"CLIP token ids must have shape [batch,context]",
			));
		};
		if *batch == 0 || *context != self.config.context_length {
			return Err(Error::invalid_argument(
				"CLIP token geometry does not match configuration",
			));
		}
		let values = match token_ids.dtype() {
			DType::I32 => token_ids.read::<i32>()?,
			DType::U32 => token_ids
				.read::<u32>()?
				.into_iter()
				.map(|value| i32::try_from(value).unwrap_or(i32::MAX))
				.collect(),
			_ => return Err(Error::invalid_argument("CLIP token ids must be I32 or U32")),
		};
		let rows = values
			.chunks_exact(*context)
			.enumerate()
			.map(|(batch_index, row)| {
				let mut position = 0;
				for index in 1..row.len() {
					if row[index] > row[position] {
						position = index;
					}
				}
				i32::try_from(batch_index * context + position)
					.map_err(|_| Error::invalid_argument("CLIP EOS row exceeds I32"))
			})
			.collect::<Result<Vec<_>>>()?;
		let eos = Matrix::from_slice_handle(token_ids.engine_handle(), vec![*batch], &rows)?;
		self.forward_tokens(token_ids, &eos)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

fn validate_config(config: ClipTextConfig) -> Result<()> {
	if config.vocab_size == 0
		|| config.context_length == 0
		|| i32::try_from(config.context_length).is_err()
		|| config.hidden_size == 0
		|| config.intermediate_size == 0
		|| config.num_heads == 0
		|| !config.hidden_size.is_multiple_of(config.num_heads)
		|| config.num_layers == 0
		|| config.projection_dim == 0
		|| !config.layer_norm_epsilon.is_finite()
		|| config.layer_norm_epsilon <= 0.0
		|| !config.quick_gelu_alpha.is_finite()
		|| config.quick_gelu_alpha <= 0.0
		|| [config.bos_token, config.eos_token, config.pad_token]
			.into_iter()
			.any(|token| token < 0 || token as usize >= config.vocab_size)
	{
		return Err(Error::invalid_argument("invalid CLIP text configuration"));
	}
	Ok(())
}
