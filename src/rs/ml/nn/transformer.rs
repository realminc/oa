use std::{cell::Cell, rc::Rc};

use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::super::{Module, ModuleRegistry, matrix::gelu};
use super::{AttentionMode, LayerNorm, Linear, Moe, MultiHeadAttention};

enum FeedForward {
	Dense {
		norm: Rc<LayerNorm>,
		input: Rc<Linear>,
		output: Rc<Linear>,
	},
	Moe(Rc<Moe>),
}

/// One pre-normalized causal Transformer block.
pub struct TransformerBlock {
	model_width: usize,
	hidden_width: usize,
	sequence_length: Cell<usize>,
	num_heads: usize,
	norm_attention: Rc<LayerNorm>,
	attention: Rc<MultiHeadAttention>,
	feed_forward: FeedForward,
	adaptive_modulation: Option<Rc<Linear>>,
	condition_dim: Option<usize>,
	registry: ModuleRegistry,
}

impl TransformerBlock {
	/// Construct the canonical pre-norm attention and GELU feed-forward block.
	///
	/// # Errors
	///
	/// Returns an error for invalid dimensions or failed child construction.
	pub fn with_seed(
		engine: &Engine,
		model_width: usize,
		hidden_width: usize,
		sequence_length: usize,
		num_heads: usize,
		epsilon: f32,
		seed: u64,
	) -> Result<Self> {
		if hidden_width == 0 {
			return Err(Error::invalid_argument(
				"Transformer FFN width must be nonzero",
			));
		}
		let norm_attention = Rc::new(LayerNorm::new(engine, model_width, epsilon)?);
		let attention = Rc::new(MultiHeadAttention::with_seed(
			engine,
			model_width,
			num_heads,
			sequence_length,
			seed,
		)?);
		let norm_feed_forward = Rc::new(LayerNorm::new(engine, model_width, epsilon)?);
		let feed_forward_in = Rc::new(Linear::with_seed(
			engine,
			model_width,
			hidden_width,
			seed.wrapping_add(4),
		)?);
		let feed_forward_out = Rc::new(Linear::with_seed(
			engine,
			hidden_width,
			model_width,
			seed.wrapping_add(5),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("ln_attn", norm_attention.clone())?;
		registry.register_module("attention", attention.clone())?;
		registry.register_module("ln_ffn", norm_feed_forward.clone())?;
		registry.register_module("ffn1", feed_forward_in.clone())?;
		registry.register_module("ffn2", feed_forward_out.clone())?;
		Ok(Self {
			model_width,
			hidden_width,
			sequence_length: Cell::new(sequence_length),
			num_heads,
			norm_attention,
			attention,
			feed_forward: FeedForward::Dense {
				norm: norm_feed_forward,
				input: feed_forward_in,
				output: feed_forward_out,
			},
			adaptive_modulation: None,
			condition_dim: None,
			registry,
		})
	}

	/// Construct a dense block with zero-initialized adaptive conditioning.
	///
	/// The complete child tree is registered before the block is returned. This
	/// is the Rust replacement for donor `enableAdaptiveConditioning`, whose
	/// post-construction registry mutation would violate [`ModuleRegistry`]'s
	/// fixed structural-ownership contract.
	///
	/// # Errors
	///
	/// Returns an error for invalid block dimensions, a zero condition width,
	/// shape overflow, allocation failure, or child registration failure.
	#[allow(
		clippy::too_many_arguments,
		reason = "the constructor preserves the donor block and conditioning dimensions"
	)]
	pub fn with_seed_conditioned(
		engine: &Engine,
		model_width: usize,
		hidden_width: usize,
		sequence_length: usize,
		num_heads: usize,
		condition_dim: usize,
		epsilon: f32,
		seed: u64,
	) -> Result<Self> {
		Self::with_seed(
			engine,
			model_width,
			hidden_width,
			sequence_length,
			num_heads,
			epsilon,
			seed,
		)?
		.with_adaptive_conditioning(engine, condition_dim)
	}

	/// Construct the donor pre-norm attention block with a sparse MoE residual.
	///
	/// The MoE owns its RMSNorm and residual connection, so this block does not
	/// add a second feed-forward normalization or residual around it.
	///
	/// # Errors
	///
	/// Returns an error for invalid attention or MoE dimensions, failed child
	/// construction, or registration failure.
	#[allow(
		clippy::too_many_arguments,
		reason = "the constructor preserves the donor attention and MoE dimensions"
	)]
	pub fn with_seed_moe(
		engine: &Engine,
		model_width: usize,
		expert_hidden_width: usize,
		sequence_length: usize,
		num_heads: usize,
		num_experts: usize,
		experts_per_token: usize,
		epsilon: f32,
		seed: u64,
	) -> Result<Self> {
		let norm_attention = Rc::new(LayerNorm::new(engine, model_width, epsilon)?);
		let attention = Rc::new(MultiHeadAttention::with_seed(
			engine,
			model_width,
			num_heads,
			sequence_length,
			seed,
		)?);
		let moe = Rc::new(Moe::with_seed(
			engine,
			model_width,
			expert_hidden_width,
			num_experts,
			experts_per_token,
			epsilon,
			seed.wrapping_add(4),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("ln_attn", norm_attention.clone())?;
		registry.register_module("attention", attention.clone())?;
		registry.register_module("moe", moe.clone())?;
		Ok(Self {
			model_width,
			hidden_width: expert_hidden_width,
			sequence_length: Cell::new(sequence_length),
			num_heads,
			norm_attention,
			attention,
			feed_forward: FeedForward::Moe(moe),
			adaptive_modulation: None,
			condition_dim: None,
			registry,
		})
	}

	/// Construct a sparse-MoE block with zero-initialized adaptive conditioning.
	///
	/// # Errors
	///
	/// Returns an error for invalid attention, expert, or conditioning geometry,
	/// allocation failure, or child registration failure.
	#[allow(
		clippy::too_many_arguments,
		reason = "the constructor preserves the donor attention, MoE, and conditioning dimensions"
	)]
	pub fn with_seed_moe_conditioned(
		engine: &Engine,
		model_width: usize,
		expert_hidden_width: usize,
		sequence_length: usize,
		num_heads: usize,
		num_experts: usize,
		experts_per_token: usize,
		condition_dim: usize,
		epsilon: f32,
		seed: u64,
	) -> Result<Self> {
		Self::with_seed_moe(
			engine,
			model_width,
			expert_hidden_width,
			sequence_length,
			num_heads,
			num_experts,
			experts_per_token,
			epsilon,
			seed,
		)?
		.with_adaptive_conditioning(engine, condition_dim)
	}

	fn with_adaptive_conditioning(mut self, engine: &Engine, condition_dim: usize) -> Result<Self> {
		if condition_dim == 0 {
			return Err(Error::invalid_argument(
				"Transformer adaptive condition width must be nonzero",
			));
		}
		let modulation_width = self
			.model_width
			.checked_mul(6)
			.ok_or_else(|| Error::invalid_argument("Transformer modulation width overflows usize"))?;
		let weight_count = condition_dim
			.checked_mul(modulation_width)
			.ok_or_else(|| Error::invalid_argument("Transformer modulation size overflows usize"))?;
		let weight = Matrix::from_f32(
			engine,
			[modulation_width, condition_dim],
			&vec![0.0; weight_count],
		)?;
		let bias = Matrix::from_f32(engine, [modulation_width], &vec![0.0; modulation_width])?;
		let adaptive_modulation = Rc::new(Linear::from_matrices(weight, bias)?);
		self
			.registry
			.register_module("adaptive_modulation", adaptive_modulation.clone())?;
		self.adaptive_modulation = Some(adaptive_modulation);
		self.condition_dim = Some(condition_dim);
		Ok(self)
	}

	/// Apply pre-norm attention/residual followed by pre-norm GELU FFN/residual.
	///
	/// # Errors
	///
	/// Returns an error unless input is F32 `[B*S, D]` or child recording fails.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		self.forward_impl(input, None)
	}

	/// Apply this block with an explicit bidirectional additive attention mask.
	///
	/// # Errors
	///
	/// Returns an error unless the mask is F32 `[B*H*S, S]` for the current
	/// sequence geometry, or a child operation fails.
	pub fn forward_masked(&self, input: &Matrix, additive_mask: &Matrix) -> Result<Matrix> {
		self.forward_impl(input, Some(additive_mask))
	}

	/// Apply donor AdaLN-Zero conditioning without an explicit attention mask.
	///
	/// # Errors
	///
	/// Returns an error unless this block was constructed with adaptive
	/// conditioning, input is F32 `[B*S, D]`, condition is same-engine F32
	/// `[B, C]`, or a child operation fails.
	pub fn forward_conditioned(&self, input: &Matrix, condition: &Matrix) -> Result<Matrix> {
		self.forward_conditioned_impl(input, condition, None)
	}

	/// Apply donor AdaLN-Zero conditioning with an additive attention mask.
	///
	/// # Errors
	///
	/// Returns an error unless the conditioned input contract and the attention
	/// mask contract are both satisfied, or a child operation fails.
	pub fn forward_conditioned_masked(
		&self,
		input: &Matrix,
		condition: &Matrix,
		additive_mask: &Matrix,
	) -> Result<Matrix> {
		self.forward_conditioned_impl(input, condition, Some(additive_mask))
	}

	fn forward_conditioned_impl(
		&self,
		input: &Matrix,
		condition: &Matrix,
		additive_mask: Option<&Matrix>,
	) -> Result<Matrix> {
		let Some(adaptive_modulation) = &self.adaptive_modulation else {
			return Err(Error::failed_precondition(
				"Transformer block was not constructed with adaptive conditioning",
			));
		};
		let condition_dim = self
			.condition_dim
			.expect("adaptive modulation and condition width are constructed together");
		let [rows, features] = input.shape() else {
			return Err(Error::invalid_argument(
				"conditioned Transformer input must have shape [B*S, D]",
			));
		};
		let [condition_batch, condition_features] = condition.shape() else {
			return Err(Error::invalid_argument(
				"Transformer condition must have shape [B, C]",
			));
		};
		if input.dtype() != DType::F32
			|| condition.dtype() != DType::F32
			|| *features != self.model_width
			|| *rows == 0
			|| !rows.is_multiple_of(self.sequence_length.get())
			|| *condition_batch != rows / self.sequence_length.get()
			|| *condition_features != condition_dim
			|| !input.engine_handle().same_as(condition.engine_handle())
		{
			return Err(Error::invalid_argument(
				"conditioned Transformer requires same-engine F32 [B*S, D] input and [B, C] condition",
			));
		}

		let modulation = adaptive_modulation.forward(condition)?;
		let modulation = matrix::repeat_interleave(&modulation, self.sequence_length.get(), 0)?;
		let mut parts = Vec::with_capacity(6);
		for index in 0..6 {
			let start = index * self.model_width;
			parts.push(matrix::slice(
				&modulation,
				1,
				start as i64,
				(start + self.model_width) as i64,
			)?);
		}

		let attention_scale = matrix::add_scalar(&parts[1], 1.0)?;
		let attention_normalized = self.norm_attention.forward(input)?;
		let attention_normalized = matrix::mul(&attention_normalized, &attention_scale)?;
		let attention_normalized = matrix::add(&attention_normalized, &parts[0])?;
		let attention_delta = match additive_mask {
			Some(mask) => self.attention.forward_masked(&attention_normalized, mask)?,
			None => self.attention.forward(&attention_normalized)?,
		};
		let attention_delta = matrix::mul(&attention_delta, &parts[2])?;
		let residual = matrix::add(input, &attention_delta)?;

		let feed_forward_normalized = match &self.feed_forward {
			FeedForward::Dense { norm, .. } => norm.forward(&residual)?,
			FeedForward::Moe(_) => residual.clone(),
		};
		let feed_forward_scale = matrix::add_scalar(&parts[4], 1.0)?;
		let feed_forward_normalized = matrix::mul(&feed_forward_normalized, &feed_forward_scale)?;
		let feed_forward_normalized = matrix::add(&feed_forward_normalized, &parts[3])?;
		let feed_forward_delta = match &self.feed_forward {
			FeedForward::Dense { input, output, .. } => {
				let hidden = gelu(&input.forward(&feed_forward_normalized)?)?;
				output.forward(&hidden)?
			}
			FeedForward::Moe(moe) => moe.forward(&feed_forward_normalized)?,
		};
		let feed_forward_delta = matrix::mul(&feed_forward_delta, &parts[5])?;
		matrix::add(&residual, &feed_forward_delta)
	}

	fn forward_impl(&self, input: &Matrix, additive_mask: Option<&Matrix>) -> Result<Matrix> {
		let normalized = self.norm_attention.forward(input)?;
		let attended = match additive_mask {
			Some(mask) => self.attention.forward_masked(&normalized, mask)?,
			None => self.attention.forward(&normalized)?,
		};
		let residual = matrix::add(input, &attended)?;
		match &self.feed_forward {
			FeedForward::Dense {
				norm,
				input,
				output,
			} => {
				let normalized = norm.forward(&residual)?;
				let hidden = gelu(&input.forward(&normalized)?)?;
				let projected = output.forward(&hidden)?;
				matrix::add(&residual, &projected)
			}
			FeedForward::Moe(moe) => moe.forward(&residual),
		}
	}

	pub const fn model_width(&self) -> usize {
		self.model_width
	}
	pub const fn hidden_width(&self) -> usize {
		self.hidden_width
	}
	pub fn sequence_length(&self) -> usize {
		self.sequence_length.get()
	}
	pub const fn num_heads(&self) -> usize {
		self.num_heads
	}

	/// Return whether this block uses routed experts instead of a dense FFN.
	pub fn is_moe(&self) -> bool {
		matches!(&self.feed_forward, FeedForward::Moe(_))
	}

	/// Return the routed expert module when this is an MoE block.
	pub fn moe(&self) -> Option<&Moe> {
		match &self.feed_forward {
			FeedForward::Moe(moe) => Some(moe),
			FeedForward::Dense { .. } => None,
		}
	}

	/// Return whether this block owns a registered AdaLN-Zero projection.
	pub fn is_adaptively_conditioned(&self) -> bool {
		self.adaptive_modulation.is_some()
	}

	/// Return the accepted condition width for an adaptive block.
	pub const fn condition_dim(&self) -> Option<usize> {
		self.condition_dim
	}

	/// Select causal or bidirectional attention for this block.
	pub fn set_attention_mode(&self, mode: AttentionMode) {
		self.attention.set_mode(mode);
	}

	/// Return this block's attention visibility contract.
	pub fn attention_mode(&self) -> AttentionMode {
		self.attention.mode()
	}

	/// Change runtime sequence geometry without rebuilding block parameters.
	///
	/// # Errors
	///
	/// Returns an error when `sequence_length` is zero.
	pub fn set_sequence_length(&self, sequence_length: usize) -> Result<()> {
		self.attention.set_sequence_length(sequence_length)?;
		self.sequence_length.set(sequence_length);
		Ok(())
	}
}

impl Module for TransformerBlock {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		TransformerBlock::forward(self, input)
	}
	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Ready-to-train causal Transformer language model.
///
/// This is the idiomatic Rust spelling of donor `oa::NnTransformer`: token and
/// position embeddings, registered Transformer blocks, final normalization,
/// and a vocabulary projection.
pub struct Transformer {
	vocab_size: usize,
	context_length: usize,
	model_width: usize,
	hidden_width: usize,
	num_heads: usize,
	token_embedding: Rc<super::Embedding>,
	position_embedding: Rc<super::Embedding>,
	blocks: Vec<Rc<TransformerBlock>>,
	final_norm: Rc<LayerNorm>,
	head: Rc<Linear>,
	registry: ModuleRegistry,
}

impl Transformer {
	/// Construct a deterministic causal language model.
	///
	/// # Errors
	///
	/// Returns an error unless all dimensions are positive, the model width is
	/// divisible by the head count, positions fit U32, and every child can be
	/// constructed and registered.
	#[allow(clippy::too_many_arguments)]
	pub fn with_seed(
		engine: &Engine,
		vocab_size: usize,
		context_length: usize,
		model_width: usize,
		hidden_width: usize,
		num_layers: usize,
		num_heads: usize,
		epsilon: f32,
		seed: u64,
	) -> Result<Self> {
		if vocab_size == 0
			|| context_length == 0
			|| model_width == 0
			|| hidden_width == 0
			|| num_layers == 0
			|| num_heads == 0
			|| !model_width.is_multiple_of(num_heads)
			|| u32::try_from(context_length - 1).is_err()
		{
			return Err(Error::invalid_argument(
				"Transformer requires positive dimensions, D divisible by H, and U32 positions",
			));
		}
		let token_embedding = Rc::new(super::Embedding::with_seed(
			engine,
			vocab_size,
			model_width,
			seed,
		)?);
		let position_embedding = Rc::new(super::Embedding::with_seed(
			engine,
			context_length,
			model_width,
			seed.wrapping_add(1),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("token_embedding", token_embedding.clone())?;
		registry.register_module("position_embedding", position_embedding.clone())?;
		let mut blocks = Vec::with_capacity(num_layers);
		for index in 0..num_layers {
			let seed_offset = (index as u64).wrapping_mul(6).wrapping_add(2);
			let block = Rc::new(TransformerBlock::with_seed(
				engine,
				model_width,
				hidden_width,
				context_length,
				num_heads,
				epsilon,
				seed.wrapping_add(seed_offset),
			)?);
			registry.register_module(format!("block_{index}"), block.clone())?;
			blocks.push(block);
		}
		let final_norm = Rc::new(LayerNorm::new(engine, model_width, epsilon)?);
		let head_seed = seed.wrapping_add((num_layers as u64).wrapping_mul(6).wrapping_add(2));
		let head = Rc::new(Linear::with_seed(
			engine,
			model_width,
			vocab_size,
			head_seed,
		)?);
		registry.register_module("final_norm", final_norm.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			vocab_size,
			context_length,
			model_width,
			hidden_width,
			num_heads,
			token_embedding,
			position_embedding,
			blocks,
			final_norm,
			head,
			registry,
		})
	}

	/// Construct a deterministic causal language model with MoE Transformer blocks.
	///
	/// Every block preserves the donor topology: pre-norm attention and residual,
	/// followed by one MoE module that owns its own RMSNorm and residual.
	///
	/// # Errors
	///
	/// Returns an error unless all dimensions are valid or when child allocation
	/// and registration fail.
	#[allow(
		clippy::too_many_arguments,
		reason = "the constructor preserves the complete Transformer and MoE recipe"
	)]
	pub fn with_seed_moe(
		engine: &Engine,
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
	) -> Result<Self> {
		if vocab_size == 0
			|| context_length == 0
			|| model_width == 0
			|| expert_hidden_width == 0
			|| num_layers == 0
			|| num_heads == 0
			|| num_experts == 0
			|| !model_width.is_multiple_of(num_heads)
			|| u32::try_from(context_length - 1).is_err()
		{
			return Err(Error::invalid_argument(
				"MoE Transformer requires positive dimensions, D divisible by H, and U32 positions",
			));
		}
		let token_embedding = Rc::new(super::Embedding::with_seed(
			engine,
			vocab_size,
			model_width,
			seed,
		)?);
		let position_embedding = Rc::new(super::Embedding::with_seed(
			engine,
			context_length,
			model_width,
			seed.wrapping_add(1),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("token_embedding", token_embedding.clone())?;
		registry.register_module("position_embedding", position_embedding.clone())?;
		let mut blocks = Vec::with_capacity(num_layers);
		for index in 0..num_layers {
			let seed_offset = (index as u64).wrapping_mul(8).wrapping_add(2);
			let block = Rc::new(TransformerBlock::with_seed_moe(
				engine,
				model_width,
				expert_hidden_width,
				context_length,
				num_heads,
				num_experts,
				experts_per_token,
				epsilon,
				seed.wrapping_add(seed_offset),
			)?);
			registry.register_module(format!("block_{index}"), block.clone())?;
			blocks.push(block);
		}
		let final_norm = Rc::new(LayerNorm::new(engine, model_width, epsilon)?);
		let head_seed = seed.wrapping_add((num_layers as u64).wrapping_mul(8).wrapping_add(2));
		let head = Rc::new(Linear::with_seed(
			engine,
			model_width,
			vocab_size,
			head_seed,
		)?);
		registry.register_module("final_norm", final_norm.clone())?;
		registry.register_module("head", head.clone())?;
		Ok(Self {
			vocab_size,
			context_length,
			model_width,
			hidden_width: expert_hidden_width,
			num_heads,
			token_embedding,
			position_embedding,
			blocks,
			final_norm,
			head,
			registry,
		})
	}

	/// Evaluate all-position vocabulary logits for U8 or U32 tokens `[B, S]`.
	///
	/// # Errors
	///
	/// Returns an error unless `S` equals the configured context length and the
	/// input is nonempty U8/U32, or a child operation fails.
	pub fn forward(&self, tokens: &Matrix) -> Result<Matrix> {
		let [batch, sequence] = tokens.shape() else {
			return Err(Error::invalid_argument(
				"Transformer tokens must have shape [batch, context_length]",
			));
		};
		if *batch == 0
			|| *sequence != self.context_length
			|| !matches!(tokens.dtype(), DType::U8 | DType::U32)
		{
			return Err(Error::invalid_argument(
				"Transformer tokens must be nonempty U8 or U32 [batch, context_length]",
			));
		}
		let rows = batch
			.checked_mul(*sequence)
			.ok_or_else(|| Error::invalid_argument("Transformer row count overflows usize"))?;
		let position_values = (0..rows)
			.map(|index| (index % self.context_length) as u32)
			.collect::<Vec<_>>();
		let position_ids =
			Matrix::from_slice_handle(tokens.engine_handle(), vec![rows], &position_values)?;
		let token_values = self
			.token_embedding
			.forward(tokens)?
			.reshape([rows, self.model_width])?;
		let position_values = self.position_embedding.forward(&position_ids)?;
		let mut value = matrix::add(&token_values, &position_values)?;
		for block in &self.blocks {
			value = block.forward(&value)?;
		}
		self.head.forward(&self.final_norm.forward(&value)?)
	}

	pub const fn vocab_size(&self) -> usize {
		self.vocab_size
	}

	pub const fn context_length(&self) -> usize {
		self.context_length
	}

	pub const fn model_width(&self) -> usize {
		self.model_width
	}

	pub const fn hidden_width(&self) -> usize {
		self.hidden_width
	}

	pub fn num_layers(&self) -> usize {
		self.blocks.len()
	}

	pub const fn num_heads(&self) -> usize {
		self.num_heads
	}

	/// Return whether every block uses routed experts.
	pub fn is_moe(&self) -> bool {
		self.blocks.iter().all(|block| block.is_moe())
	}

	/// Return one Transformer block by layer index.
	pub fn block(&self, index: usize) -> Option<&TransformerBlock> {
		self.blocks.get(index).map(Rc::as_ref)
	}
}

impl Module for Transformer {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Transformer::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
