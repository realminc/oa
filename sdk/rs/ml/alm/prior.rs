use std::{cell::RefCell, rc::Rc};

use crate::ml::{
	Module, ModuleRegistry,
	nn::{Embedding, Linear, MoeRouteStats, RmsNorm, TransformerBlock},
};
use crate::{DType, Engine, Error, Matrix, Result, matrix};

use super::AlmTokenizer;

const NORMALIZATION_EPSILON: f32 = 1e-5;

/// Feed-forward policy used by ALM Transformer layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlmFfnType {
	/// Every layer uses a dense GELU feed-forward network.
	#[default]
	Dense,
	/// Every layer uses sparse routed experts.
	Moe,
	/// Every `moe_every` layer uses sparse experts; other layers remain dense.
	Hybrid,
}

/// Architecture, routing, training, and default-generation policy for ALM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlmPriorConfig {
	/// Complete vocabulary size, including the three special tokens.
	pub vocab_size: usize,
	/// Number of motion codes produced by the tokenizer.
	pub num_codes: usize,
	/// Start-of-motion token.
	pub som_token: i32,
	/// End-of-motion token.
	pub eom_token: i32,
	/// Padding token.
	pub pad_token: i32,
	/// Transformer feature width.
	pub model_width: usize,
	/// Number of causal attention heads.
	pub num_heads: usize,
	/// Number of decoder blocks.
	pub num_layers: usize,
	/// Dense or expert feed-forward hidden width.
	pub hidden_width: usize,
	/// Frozen text-feature width; zero selects an unconditional prior.
	pub text_feature_dim: usize,
	/// Dense, MoE, or hybrid feed-forward policy.
	pub ffn_type: AlmFfnType,
	/// Number of routed experts in each MoE layer.
	pub moe_num_experts: usize,
	/// Number of experts selected per token.
	pub moe_experts_per_token: usize,
	/// Hybrid cadence in one-based layer numbering.
	pub moe_every: usize,
	/// Auxiliary-loss-free routing-bias update rate.
	pub moe_balance_rate: f32,
	/// Switch/GShard auxiliary-loss coefficient.
	pub moe_aux_loss_alpha: f32,
	/// Router z-loss coefficient.
	pub moe_router_z_loss_beta: f32,
	/// Workload training batch size.
	pub batch_size: usize,
	/// Default training sequence length.
	pub sequence_length: usize,
	/// Learned-position table and runtime sequence limit.
	pub max_sequence_length: usize,
	/// Workload optimizer learning rate.
	pub learning_rate: f32,
	/// Workload training epoch count.
	pub num_epochs: usize,
	/// Default sampling temperature.
	pub temperature: f32,
	/// Default top-k restriction; zero disables it.
	pub top_k: i32,
	/// Default nucleus probability.
	pub top_p: f32,
	/// Default maximum number of generation steps after SOM.
	pub max_generation_length: usize,
}

impl Default for AlmPriorConfig {
	fn default() -> Self {
		Self {
			vocab_size: 515,
			num_codes: 512,
			som_token: 512,
			eom_token: 513,
			pad_token: 514,
			model_width: 384,
			num_heads: 1,
			num_layers: 6,
			hidden_width: 1536,
			text_feature_dim: 0,
			ffn_type: AlmFfnType::Dense,
			moe_num_experts: 4,
			moe_experts_per_token: 2,
			moe_every: 2,
			moe_balance_rate: 0.0,
			moe_aux_loss_alpha: 0.0,
			moe_router_z_loss_beta: 0.0,
			batch_size: 32,
			sequence_length: 128,
			max_sequence_length: 260,
			learning_rate: 1e-4,
			num_epochs: 100,
			temperature: 1.0,
			top_k: 0,
			top_p: 0.9,
			max_generation_length: 256,
		}
	}
}

impl AlmPriorConfig {
	/// Synchronize vocabulary and special-token values with a tokenizer code count.
	///
	/// # Errors
	///
	/// Returns an error when `num_codes + 3` exceeds the I32 token representation.
	pub fn sync_vocab(&mut self, num_codes: usize) -> Result<()> {
		let eom = num_codes
			.checked_add(1)
			.ok_or_else(|| Error::invalid_argument("ALM vocabulary overflows usize"))?;
		let pad = num_codes
			.checked_add(2)
			.ok_or_else(|| Error::invalid_argument("ALM vocabulary overflows usize"))?;
		let vocab = num_codes
			.checked_add(3)
			.ok_or_else(|| Error::invalid_argument("ALM vocabulary overflows usize"))?;
		self.num_codes = num_codes;
		self.som_token = i32::try_from(num_codes)
			.map_err(|_| Error::invalid_argument("ALM code count exceeds I32"))?;
		self.eom_token =
			i32::try_from(eom).map_err(|_| Error::invalid_argument("ALM EOM exceeds I32"))?;
		self.pad_token =
			i32::try_from(pad).map_err(|_| Error::invalid_argument("ALM PAD exceeds I32"))?;
		self.vocab_size = vocab;
		Ok(())
	}

	/// Return whether the requested zero-based layer uses routed experts.
	pub fn uses_moe(&self, layer: usize) -> bool {
		match self.ffn_type {
			AlmFfnType::Dense => false,
			AlmFfnType::Moe => true,
			AlmFfnType::Hybrid => self.moe_every != 0 && (layer + 1).is_multiple_of(self.moe_every),
		}
	}
}

/// Explicit autoregressive sampling policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlmGenerationOptions {
	/// Sampling temperature; nonpositive values select greedy argmax.
	pub temperature: f32,
	/// Top-k candidate count; zero disables top-k restriction.
	pub top_k: i32,
	/// Nucleus probability in `(0, 1]` after clamping by the sampler.
	pub top_p: f32,
	/// Maximum generation steps after the initial SOM token.
	pub max_length: usize,
	/// Exact base seed; each generation step adds its zero-based index.
	pub seed: u64,
	/// Reserved request for the not-yet-admitted KV-cache path.
	pub use_cache: bool,
}

impl AlmGenerationOptions {
	fn from_config(config: &AlmPriorConfig) -> Self {
		Self {
			temperature: config.temperature,
			top_k: config.top_k,
			top_p: config.top_p,
			max_length: config.max_generation_length,
			seed: 0,
			use_cache: true,
		}
	}
}

struct PositionCache {
	batch: usize,
	sequence: usize,
	indices: Matrix,
}

/// Decoder-only ALM motion-token prior.
///
/// The permanent backbone is causal attention. Its feed-forward policy may be
/// dense, MoE, or hybrid without changing tokenization, conditioning,
/// generation, or the registered persistence tree.
pub struct AlmPrior {
	config: AlmPriorConfig,
	token_embedding: Rc<Embedding>,
	position_embedding: Rc<Embedding>,
	text_projection: Option<Rc<Linear>>,
	layers: Vec<Rc<TransformerBlock>>,
	final_norm: Rc<RmsNorm>,
	output_head: Rc<Linear>,
	position_cache: RefCell<Option<PositionCache>>,
	registry: ModuleRegistry,
}

impl AlmPrior {
	/// Construct a deterministic ALM prior.
	///
	/// # Errors
	///
	/// Returns an error for inconsistent vocabulary, dimensions, routing policy,
	/// generation policy, failed allocation, or child registration.
	pub fn with_seed(engine: &Engine, config: AlmPriorConfig, seed: u64) -> Result<Self> {
		validate_config(config)?;
		let token_embedding = Rc::new(Embedding::with_seed(
			engine,
			config.vocab_size,
			config.model_width,
			seed,
		)?);
		let position_embedding = Rc::new(Embedding::with_seed(
			engine,
			config.max_sequence_length,
			config.model_width,
			seed.wrapping_add(1),
		)?);
		let text_projection = if config.text_feature_dim == 0 {
			None
		} else {
			Some(Rc::new(Linear::with_seed(
				engine,
				config.text_feature_dim,
				config.model_width,
				seed.wrapping_add(2),
			)?))
		};
		let mut registry = ModuleRegistry::new();
		if let Some(projection) = &text_projection {
			registry.register_module("text_projection", projection.clone())?;
		}
		let mut layers = Vec::with_capacity(config.num_layers);
		for index in 0..config.num_layers {
			let layer_seed = seed
				.wrapping_add(3)
				.wrapping_add((index as u64).wrapping_mul(8));
			let block = if config.uses_moe(index) {
				let block = Rc::new(TransformerBlock::with_seed_moe(
					engine,
					config.model_width,
					config.hidden_width,
					config.sequence_length,
					config.num_heads,
					config.moe_num_experts,
					config.moe_experts_per_token,
					NORMALIZATION_EPSILON,
					layer_seed,
				)?);
				let moe = block
					.moe()
					.expect("MoE constructor always creates an MoE block");
				moe.set_balance_rate(config.moe_balance_rate);
				moe.set_aux_loss_alpha(config.moe_aux_loss_alpha);
				moe.set_router_z_loss_beta(config.moe_router_z_loss_beta);
				block
			} else {
				Rc::new(TransformerBlock::with_seed(
					engine,
					config.model_width,
					config.hidden_width,
					config.sequence_length,
					config.num_heads,
					NORMALIZATION_EPSILON,
					layer_seed,
				)?)
			};
			registry.register_module(format!("layer{index}"), block.clone())?;
			layers.push(block);
		}
		let final_norm = Rc::new(RmsNorm::new(
			engine,
			config.model_width,
			NORMALIZATION_EPSILON,
		)?);
		let output_head = Rc::new(Linear::with_seed_and_bias(
			engine,
			config.model_width,
			config.vocab_size,
			false,
			seed.wrapping_add(3 + (config.num_layers as u64).wrapping_mul(8)),
		)?);
		registry.register_module("token_embed", token_embedding.clone())?;
		registry.register_module("pos_embed", position_embedding.clone())?;
		registry.register_module("final_norm", final_norm.clone())?;
		registry.register_module("output_head", output_head.clone())?;
		Ok(Self {
			config,
			token_embedding,
			position_embedding,
			text_projection,
			layers,
			final_norm,
			output_head,
			position_cache: RefCell::new(None),
			registry,
		})
	}

	/// Construct a prior with deterministic seed zero.
	///
	/// # Errors
	///
	/// Returns the same errors as [`Self::with_seed`].
	pub fn new(engine: &Engine, config: AlmPriorConfig) -> Result<Self> {
		Self::with_seed(engine, config, 0)
	}

	/// Evaluate unconditional `[batch, time, vocab]` next-token logits.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` when this prior requires text features, or an
	/// input/recording error from the common forward path.
	pub fn forward(&self, token_ids: &Matrix) -> Result<Matrix> {
		if self.config.text_feature_dim != 0 {
			return Err(Error::failed_precondition(
				"conditioned ALM prior requires forward_conditioned",
			));
		}
		self.forward_impl(token_ids, None)
	}

	/// Evaluate motion-token logits with one projected frozen-text prefix token.
	///
	/// Returned logits correspond only to motion positions, never the prefix.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` for an unconditional prior, or an
	/// input/recording error from the common forward path.
	pub fn forward_conditioned(&self, token_ids: &Matrix, text_features: &Matrix) -> Result<Matrix> {
		if self.config.text_feature_dim == 0 {
			return Err(Error::failed_precondition(
				"unconditional ALM prior has no text projection",
			));
		}
		self.forward_impl(token_ids, Some(text_features))
	}

	fn forward_impl(&self, token_ids: &Matrix, text_features: Option<&Matrix>) -> Result<Matrix> {
		let [batch, sequence] = token_ids.shape() else {
			return Err(Error::invalid_argument(
				"ALM prior token ids must have shape [batch,time]",
			));
		};
		let prefix = usize::from(text_features.is_some());
		let total_sequence = sequence
			.checked_add(prefix)
			.ok_or_else(|| Error::invalid_argument("ALM prior sequence length overflows usize"))?;
		if *batch == 0
			|| *sequence == 0
			|| total_sequence > self.config.max_sequence_length
			|| !matches!(token_ids.dtype(), DType::U8 | DType::U32 | DType::I32)
		{
			return Err(Error::invalid_argument(
				"ALM prior requires nonempty U8/U32/I32 [batch,time] tokens within max_sequence_length",
			));
		}
		let token_values = self.token_embedding.forward(token_ids)?;
		let embeddings = if let Some(text_features) = text_features {
			if text_features.shape() != [*batch, self.config.text_feature_dim]
				|| text_features.dtype() != DType::F32
				|| !token_ids
					.engine_handle()
					.same_as(text_features.engine_handle())
			{
				return Err(Error::invalid_argument(
					"ALM text features must be same-engine F32 [batch,text_feature_dim]",
				));
			}
			let projection = self
				.text_projection
				.as_ref()
				.expect("conditioned configuration constructs text projection");
			let text = matrix::reshape(
				&projection.forward(text_features)?,
				[*batch, 1, self.config.model_width],
			)?;
			matrix::concat(&[text, token_values], 1)?
		} else {
			token_values
		};
		let positions = self.position_ids(token_ids, *batch, total_sequence)?;
		let position_values = self.position_embedding.forward(&positions)?;
		let rows = batch
			.checked_mul(total_sequence)
			.ok_or_else(|| Error::invalid_argument("ALM prior row count overflows usize"))?;
		let mut hidden = matrix::reshape(
			&matrix::add(&embeddings, &position_values)?,
			[rows, self.config.model_width],
		)?;
		for layer in &self.layers {
			layer.set_sequence_length(total_sequence)?;
			hidden = layer.forward(&hidden)?;
		}
		hidden = self.final_norm.forward(&hidden)?;
		if prefix != 0 {
			hidden = matrix::reshape(&hidden, [*batch, total_sequence, self.config.model_width])?;
			hidden = matrix::slice(
				&hidden,
				1,
				1,
				i64::try_from(total_sequence)
					.map_err(|_| Error::invalid_argument("ALM sequence exceeds i64"))?,
			)?;
			hidden = matrix::reshape(&hidden, [batch * sequence, self.config.model_width])?;
		}
		let logits = self.output_head.forward(&hidden)?;
		matrix::reshape(&logits, [*batch, *sequence, self.config.vocab_size])
	}

	fn position_ids(&self, token_ids: &Matrix, batch: usize, sequence: usize) -> Result<Matrix> {
		let mut cache = self.position_cache.borrow_mut();
		if let Some(cache) = cache.as_ref()
			&& cache.batch == batch
			&& cache.sequence == sequence
			&& cache
				.indices
				.engine_handle()
				.same_as(token_ids.engine_handle())
		{
			return Ok(cache.indices.clone());
		}
		let count = batch
			.checked_mul(sequence)
			.ok_or_else(|| Error::invalid_argument("ALM position count overflows usize"))?;
		let values = (0..count)
			.map(|index| {
				u32::try_from(index % sequence)
					.map_err(|_| Error::invalid_argument("ALM position exceeds u32"))
			})
			.collect::<Result<Vec<_>>>()?;
		let indices =
			Matrix::from_slice_handle(token_ids.engine_handle(), vec![batch, sequence], &values)?;
		*cache = Some(PositionCache {
			batch,
			sequence,
			indices: indices.clone(),
		});
		Ok(indices)
	}

	/// Return the sum of the latest MoE auxiliary losses, or `None` for a dense prior.
	///
	/// # Errors
	///
	/// Returns an error if scalar addition cannot be recorded.
	pub fn moe_aux_loss(&self) -> Result<Option<Matrix>> {
		let mut total = None;
		for layer in &self.layers {
			if let Some(moe) = layer.moe() {
				total = Some(match total {
					Some(previous) => matrix::add(&previous, &moe.aux_loss())?,
					None => moe.aux_loss(),
				});
			}
		}
		Ok(total)
	}

	/// Queue auxiliary-loss-free routing-bias updates for all MoE layers.
	///
	/// # Errors
	///
	/// Returns an error when a device-resident update cannot be recorded.
	pub fn update_moe_routing_bias(&self) -> Result<()> {
		for layer in &self.layers {
			if let Some(moe) = layer.moe() {
				moe.update_routing_bias()?;
			}
		}
		Ok(())
	}

	/// Read reduced routing telemetry for each MoE layer.
	///
	/// # Errors
	///
	/// This is an explicit host boundary and returns an error if reduction or
	/// readback fails.
	pub fn moe_route_stats(&self) -> Result<Vec<MoeRouteStats>> {
		self
			.layers
			.iter()
			.filter_map(|layer| layer.moe())
			.map(|moe| moe.route_stats())
			.collect()
	}

	/// Return the generation policy stored in this prior's configuration.
	pub fn default_generation_options(&self) -> AlmGenerationOptions {
		AlmGenerationOptions::from_config(&self.config)
	}

	/// Generate unconditional token sequences beginning with SOM.
	///
	/// `use_cache` is accepted for donor API compatibility, but the current
	/// implementation deliberately replays the growing prefix because no KV-cache
	/// path has been admitted yet.
	///
	/// # Errors
	///
	/// Returns an error for invalid generation geometry/policy or failed forward,
	/// sampling, submission, or readback.
	pub fn generate(&self, batch: usize, options: AlmGenerationOptions) -> Result<Matrix> {
		if self.config.text_feature_dim != 0 {
			return Err(Error::failed_precondition(
				"conditioned ALM prior requires generate_conditioned",
			));
		}
		self.generate_impl(None, batch, options)
	}

	/// Generate token sequences from frozen caption features.
	///
	/// # Errors
	///
	/// Returns an error for an unconditional prior, invalid text geometry/policy,
	/// or failed forward/sampling/readback.
	pub fn generate_conditioned(
		&self,
		text_features: &Matrix,
		options: AlmGenerationOptions,
	) -> Result<Matrix> {
		if self.config.text_feature_dim == 0 {
			return Err(Error::failed_precondition(
				"unconditional ALM prior cannot generate from text features",
			));
		}
		let [batch, width] = text_features.shape() else {
			return Err(Error::invalid_argument(
				"ALM generation text features must have shape [batch,text_feature_dim]",
			));
		};
		if *batch == 0 || *width != self.config.text_feature_dim {
			return Err(Error::invalid_argument(
				"ALM generation text features have incompatible geometry",
			));
		}
		self.generate_impl(Some(text_features), *batch, options)
	}

	fn generate_impl(
		&self,
		text_features: Option<&Matrix>,
		batch: usize,
		options: AlmGenerationOptions,
	) -> Result<Matrix> {
		let prefix = usize::from(text_features.is_some());
		if batch == 0
			|| options.max_length == 0
			|| options
				.max_length
				.checked_add(prefix)
				.is_none_or(|length| length > self.config.max_sequence_length)
			|| !options.temperature.is_finite()
			|| options.top_k < 0
			|| !options.top_p.is_finite()
		{
			return Err(Error::invalid_argument(
				"invalid ALM generation batch, length, temperature, or top-p policy",
			));
		}
		let _cache_requested = options.use_cache;
		let mut rows = vec![vec![self.config.som_token]; batch];
		let mut done = vec![false; batch];
		for step in 0..options.max_length {
			let current_length = step + 1;
			let count = batch
				.checked_mul(current_length)
				.ok_or_else(|| Error::invalid_argument("ALM generation size overflows usize"))?;
			let mut dense = vec![self.config.pad_token; count];
			for (batch_index, row) in rows.iter().enumerate() {
				for (time, token) in row.iter().enumerate() {
					dense[batch_index * current_length + time] = *token;
				}
			}
			let token_ids = Matrix::from_slice_handle(
				self.token_embedding.weight().data().engine_handle(),
				vec![batch, current_length],
				&dense,
			)?;
			let logits = match text_features {
				Some(features) => self.forward_conditioned(&token_ids, features)?,
				None => self.forward(&token_ids)?,
			};
			let last = matrix::slice(
				&logits,
				1,
				i64::try_from(step)
					.map_err(|_| Error::invalid_argument("ALM generation step exceeds i64"))?,
				i64::try_from(current_length)
					.map_err(|_| Error::invalid_argument("ALM generation length exceeds i64"))?,
			)?;
			let last = matrix::reshape(&last, [batch, self.config.vocab_size])?;
			let sampled = matrix::sample_logits(
				&last,
				options.temperature,
				options.top_k,
				options.top_p,
				options.seed.wrapping_add(step as u64),
			)?
			.read::<i32>()?;
			let mut all_done = true;
			for batch_index in 0..batch {
				if done[batch_index] {
					continue;
				}
				let next = sampled[batch_index];
				rows[batch_index].push(next);
				if next == self.config.eom_token {
					done[batch_index] = true;
				} else {
					all_done = false;
				}
			}
			if all_done {
				break;
			}
		}
		let output_length = rows.iter().map(Vec::len).max().unwrap_or(0);
		let mut output = vec![self.config.pad_token; batch * output_length];
		for (batch_index, row) in rows.iter().enumerate() {
			let offset = batch_index * output_length;
			output[offset..offset + row.len()].copy_from_slice(row);
		}
		Matrix::from_slice_handle(
			self.token_embedding.weight().data().engine_handle(),
			vec![batch, output_length],
			&output,
		)
	}

	/// Decode generated `[batch,time]` sequences through the ALM tokenizer.
	///
	/// The common prefix ending before the earliest EOM is used, SOM is removed,
	/// and `None` is returned when no motion-code position remains.
	///
	/// # Errors
	///
	/// Returns an error for incompatible token dtype/geometry or failed tokenizer
	/// lookup/decoding.
	pub fn decode_to_motion(
		&self,
		token_ids: &Matrix,
		tokenizer: &AlmTokenizer,
	) -> Result<Option<Matrix>> {
		let [batch, sequence] = token_ids.shape() else {
			return Err(Error::invalid_argument(
				"ALM generated tokens must have shape [batch,time]",
			));
		};
		if *batch == 0 || *sequence == 0 || !matches!(token_ids.dtype(), DType::I32 | DType::U32) {
			return Err(Error::invalid_argument(
				"ALM generated tokens must be nonempty I32 or U32 [batch,time]",
			));
		}
		let ids = match token_ids.dtype() {
			DType::I32 => token_ids.read::<i32>()?,
			DType::U32 => token_ids
				.read::<u32>()?
				.into_iter()
				.map(|value| {
					i32::try_from(value).map_err(|_| Error::invalid_argument("ALM U32 token exceeds I32"))
				})
				.collect::<Result<Vec<_>>>()?,
			_ => unreachable!("validated ALM generated-token dtype"),
		};
		let minimum_end = (0..*batch)
			.map(|batch_index| {
				let row = &ids[batch_index * sequence..(batch_index + 1) * sequence];
				row
					.iter()
					.position(|token| *token == self.config.eom_token)
					.unwrap_or(*sequence)
			})
			.min()
			.expect("validated nonempty batch");
		let token_length = minimum_end.saturating_sub(1);
		if token_length == 0 {
			return Ok(None);
		}
		let mut flat = Vec::with_capacity(batch * token_length);
		for batch_index in 0..*batch {
			let start = batch_index * sequence + 1;
			flat.extend_from_slice(&ids[start..start + token_length]);
		}
		let indices =
			Matrix::from_slice_handle(token_ids.engine_handle(), vec![batch * token_length], &flat)?;
		Ok(Some(tokenizer.detokenize(&[indices], *batch)?))
	}

	/// Return this prior's immutable complete configuration.
	pub const fn config(&self) -> &AlmPriorConfig {
		&self.config
	}

	/// Return one Transformer block by index.
	pub fn layer(&self, index: usize) -> Option<&TransformerBlock> {
		self.layers.get(index).map(Rc::as_ref)
	}
}

impl Module for AlmPrior {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		AlmPrior::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

fn validate_config(config: AlmPriorConfig) -> Result<()> {
	let expected_vocab = config.num_codes.checked_add(3);
	let expected_som = i32::try_from(config.num_codes).ok();
	let expected_eom = config
		.num_codes
		.checked_add(1)
		.and_then(|value| i32::try_from(value).ok());
	let expected_pad = config
		.num_codes
		.checked_add(2)
		.and_then(|value| i32::try_from(value).ok());
	if config.num_codes == 0
		|| expected_vocab != Some(config.vocab_size)
		|| expected_som != Some(config.som_token)
		|| expected_eom != Some(config.eom_token)
		|| expected_pad != Some(config.pad_token)
		|| config.model_width == 0
		|| config.num_heads == 0
		|| !config.model_width.is_multiple_of(config.num_heads)
		|| config.num_layers == 0
		|| config.hidden_width == 0
		|| config.sequence_length == 0
		|| config.max_sequence_length == 0
		|| config.sequence_length > config.max_sequence_length
		|| (config.ffn_type != AlmFfnType::Dense
			&& (config.moe_num_experts == 0 || config.moe_experts_per_token == 0))
		|| (config.ffn_type == AlmFfnType::Hybrid && config.moe_every == 0)
		|| !config.moe_balance_rate.is_finite()
		|| !config.moe_aux_loss_alpha.is_finite()
		|| !config.moe_router_z_loss_beta.is_finite()
		|| !config.learning_rate.is_finite()
		|| config.learning_rate < 0.0
		|| config.batch_size == 0
		|| config.num_epochs == 0
		|| !config.temperature.is_finite()
		|| !config.top_p.is_finite()
		|| config.max_generation_length == 0
	{
		return Err(Error::invalid_argument("invalid ALM prior configuration"));
	}
	Ok(())
}
