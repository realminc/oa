use std::{
	cell::{Cell, RefCell},
	rc::Rc,
};

use crate::{DType, Engine, Error, Matrix, Result, matrix as core_matrix};

use super::super::{
	Module, ModuleRegistry,
	matrix::{
		bmm, bmm_nt, flash_attention_causal, merge_heads, scaled_dot_product_attention,
		softmax_scaled_masked, split_heads,
	},
};
use super::Linear;

/// Execution policy for scaled dot-product attention.
///
/// `Auto` selects only a route proven by device-specific performance evidence.
/// It currently preserves the compositional [`AttentionBackend::Standard`]
/// provider on every device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionBackend {
	/// Use the internally proven provider policy.
	#[default]
	Auto,
	/// Use BMM-NT, scaled/masked Softmax, and BMM.
	Standard,
	/// Explicitly request the fused causal Flash provider.
	Flash,
}

/// Token-visibility contract for self-attention.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionMode {
	/// A query may observe only itself and preceding tokens.
	#[default]
	Causal,
	/// Every query may observe every token in its sequence.
	Bidirectional,
}

/// Multi-head self-attention over packed `[B*S, D]` values.
pub struct MultiHeadAttention {
	model_width: usize,
	num_heads: usize,
	sequence_length: Cell<usize>,
	query_projection: Rc<Linear>,
	key_projection: Rc<Linear>,
	value_projection: Rc<Linear>,
	output_projection: Rc<Linear>,
	dropout_probability: f32,
	dropout_seed: u64,
	backend: Cell<AttentionBackend>,
	mode: Cell<AttentionMode>,
	last_backend: Cell<AttentionBackend>,
	mask_cache: RefCell<Vec<AttentionMask>>,
	registry: ModuleRegistry,
}

struct AttentionMask {
	batch: usize,
	sequence_length: usize,
	causal: bool,
	value: Matrix,
}

impl MultiHeadAttention {
	/// Construct four biased projections with causal visibility and automatic
	/// backend policy.
	///
	/// # Errors
	///
	/// Returns an error unless dimensions are positive and `model_width` is
	/// divisible by `num_heads`, or child construction/registration fails.
	pub fn with_seed(
		engine: &Engine,
		model_width: usize,
		num_heads: usize,
		sequence_length: usize,
		seed: u64,
	) -> Result<Self> {
		Self::with_seed_and_options(
			engine,
			model_width,
			num_heads,
			sequence_length,
			0.0,
			true,
			seed,
		)
	}

	/// Construct four projections with explicit optional bias.
	///
	/// # Errors
	///
	/// Returns an error unless dimensions are positive and `model_width` is
	/// divisible by `num_heads`, or child construction/registration fails.
	pub fn with_seed_and_bias(
		engine: &Engine,
		model_width: usize,
		num_heads: usize,
		sequence_length: usize,
		bias: bool,
		seed: u64,
	) -> Result<Self> {
		Self::with_seed_and_options(
			engine,
			model_width,
			num_heads,
			sequence_length,
			0.0,
			bias,
			seed,
		)
	}

	/// Construct attention with explicit dropout and projection-bias policy.
	///
	/// Dropout is applied to attention probabilities only while the module is in
	/// training mode. Its Philox stream is derived from `seed` and advances on
	/// captured-program replay through the ordinary Matrix Dropout contract.
	///
	/// # Errors
	///
	/// Returns an error unless dimensions are valid, `dropout_probability` is
	/// finite and in `[0, 1)`, or child construction/registration fails.
	#[allow(clippy::too_many_arguments)]
	pub fn with_seed_and_options(
		engine: &Engine,
		model_width: usize,
		num_heads: usize,
		sequence_length: usize,
		dropout_probability: f32,
		bias: bool,
		seed: u64,
	) -> Result<Self> {
		if model_width == 0
			|| num_heads == 0
			|| sequence_length == 0
			|| !model_width.is_multiple_of(num_heads)
		{
			return Err(Error::invalid_argument(
				"attention requires positive D/H/S and D divisible by H",
			));
		}
		if !dropout_probability.is_finite() || !(0.0..1.0).contains(&dropout_probability) {
			return Err(Error::invalid_argument(
				"attention dropout probability must be finite and in [0, 1)",
			));
		}
		let query_projection = Rc::new(Linear::with_seed_and_bias(
			engine,
			model_width,
			model_width,
			bias,
			seed,
		)?);
		let key_projection = Rc::new(Linear::with_seed_and_bias(
			engine,
			model_width,
			model_width,
			bias,
			seed.wrapping_add(1),
		)?);
		let value_projection = Rc::new(Linear::with_seed_and_bias(
			engine,
			model_width,
			model_width,
			bias,
			seed.wrapping_add(2),
		)?);
		let output_projection = Rc::new(Linear::with_seed_and_bias(
			engine,
			model_width,
			model_width,
			bias,
			seed.wrapping_add(3),
		)?);
		let mut registry = ModuleRegistry::new();
		registry.register_module("q_proj", query_projection.clone())?;
		registry.register_module("k_proj", key_projection.clone())?;
		registry.register_module("v_proj", value_projection.clone())?;
		registry.register_module("out_proj", output_projection.clone())?;
		Ok(Self {
			model_width,
			num_heads,
			sequence_length: Cell::new(sequence_length),
			query_projection,
			key_projection,
			value_projection,
			output_projection,
			dropout_probability,
			dropout_seed: seed.wrapping_add(4),
			backend: Cell::new(AttentionBackend::Auto),
			mode: Cell::new(AttentionMode::Causal),
			last_backend: Cell::new(AttentionBackend::Standard),
			mask_cache: RefCell::new(Vec::new()),
			registry,
		})
	}

	fn geometry(&self, input: &Matrix) -> Result<usize> {
		let sequence_length = self.sequence_length.get();
		let [rows, width] = input.shape() else {
			return Err(Error::invalid_argument(
				"MultiHeadAttention expects F32 [B*S, D] input",
			));
		};
		if input.dtype() != DType::F32
			|| *width != self.model_width
			|| *rows == 0
			|| !rows.is_multiple_of(sequence_length)
		{
			return Err(Error::invalid_argument(format!(
				"MultiHeadAttention expects nonempty F32 [B*S, {}] with S={}; found {:?} {}",
				self.model_width,
				sequence_length,
				input.shape(),
				input.dtype().token(),
			)));
		}
		Ok(rows / sequence_length)
	}

	fn project(&self, input: &Matrix) -> Result<(Matrix, Matrix, Matrix)> {
		Ok((
			self.query_projection.forward(input)?,
			self.key_projection.forward(input)?,
			self.value_projection.forward(input)?,
		))
	}

	fn forward_standard(
		&self,
		input: &Matrix,
		additive_mask: Option<&Matrix>,
		causal: bool,
	) -> Result<Matrix> {
		let batch = self.geometry(input)?;
		let sequence_length = self.sequence_length.get();
		let (query, key, value) = self.project(input)?;
		let query = split_heads(&query, batch, sequence_length, self.num_heads)?;
		let key = split_heads(&key, batch, sequence_length, self.num_heads)?;
		let value = split_heads(&value, batch, sequence_length, self.num_heads)?;
		let scale = 1.0 / ((self.model_width / self.num_heads) as f32).sqrt();
		let context = if self.is_training() && self.dropout_probability > 0.0 {
			let batch_heads = batch.checked_mul(self.num_heads).ok_or_else(|| {
				Error::invalid_argument("attention batch by head count overflows usize")
			})?;
			let score_rows = batch_heads
				.checked_mul(sequence_length)
				.ok_or_else(|| Error::invalid_argument("attention score rows overflow usize"))?;
			let scores = bmm_nt(&query, &key)?.reshape([score_rows, sequence_length])?;
			let owned_mask;
			let mask = if let Some(mask) = additive_mask {
				mask
			} else {
				owned_mask = self.cached_mask(input, batch, causal)?;
				&owned_mask
			};
			let probability = softmax_scaled_masked(&scores, mask, scale)?;
			let probability =
				core_matrix::dropout(&probability, self.dropout_probability, self.dropout_seed)?;
			bmm(
				&probability.reshape([batch_heads, sequence_length, sequence_length])?,
				&value,
			)?
		} else {
			scaled_dot_product_attention(&query, &key, &value, additive_mask, scale, causal)?
		};
		let context = merge_heads(&context, batch, sequence_length, self.num_heads)?;
		let output = self.output_projection.forward(&context)?;
		self.last_backend.set(AttentionBackend::Standard);
		Ok(output)
	}

	fn cached_mask(&self, input: &Matrix, batch: usize, causal: bool) -> Result<Matrix> {
		let sequence_length = self.sequence_length.get();
		if let Some(mask) = self.mask_cache.borrow().iter().find(|mask| {
			mask.batch == batch && mask.sequence_length == sequence_length && mask.causal == causal
		}) {
			return Ok(mask.value.clone());
		}
		let batch_heads = batch.checked_mul(self.num_heads).ok_or_else(|| {
			Error::invalid_argument("attention batch by head count overflows usize")
		})?;
		let score_rows = batch_heads
			.checked_mul(sequence_length)
			.ok_or_else(|| Error::invalid_argument("attention score rows overflow usize"))?;
		let count = score_rows
			.checked_mul(sequence_length)
			.ok_or_else(|| Error::invalid_argument("attention score size overflows usize"))?;
		let mut values = vec![0.0_f32; count];
		if causal {
			for row in 0..score_rows {
				let query = row % sequence_length;
				for key in query + 1..sequence_length {
					values[row * sequence_length + key] = -1.0e9;
				}
			}
		}
		let mask = Matrix::allocate(
			input.engine_handle(),
			vec![score_rows, sequence_length],
			count,
			DType::F32,
		)?;
		mask.write_values(&values)?;
		self.mask_cache.borrow_mut().push(AttentionMask {
			batch,
			sequence_length,
			causal,
			value: mask.clone(),
		});
		Ok(mask)
	}

	fn forward_flash(&self, input: &Matrix) -> Result<Matrix> {
		let batch = self.geometry(input)?;
		let sequence_length = self.sequence_length.get();
		let (query, key, value) = self.project(input)?;
		let query = split_heads(&query, batch, sequence_length, self.num_heads)?;
		let key = split_heads(&key, batch, sequence_length, self.num_heads)?;
		let value = split_heads(&value, batch, sequence_length, self.num_heads)?;
		let scale = 1.0 / ((self.model_width / self.num_heads) as f32).sqrt();
		let context = flash_attention_causal(&query, &key, &value, scale)?;
		let context = merge_heads(&context, batch, sequence_length, self.num_heads)?;
		let output = self.output_projection.forward(&context)?;
		self.last_backend.set(AttentionBackend::Flash);
		Ok(output)
	}

	/// Apply self-attention using the configured visibility and provider policy.
	///
	/// `Auto` deliberately uses the standard provider until a device-specific
	/// benchmark qualifies another route. Explicit Flash fails closed unless the
	/// operation is causal F32 with `S <= 1024`.
	///
	/// # Errors
	///
	/// Returns an error for invalid input geometry, an ineligible explicit Flash
	/// request, or failed operation recording.
	pub fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let mode = self.mode.get();
		match self.backend.get() {
			AttentionBackend::Flash => {
				if mode != AttentionMode::Causal
					|| input.dtype() != DType::F32
					|| self.dropout_probability != 0.0
					|| self.sequence_length.get() > 1024
				{
					return Err(Error::invalid_argument(
						"FlashAttention requires causal F32 input, dropout=0, and sequence length <= 1024",
					));
				}
				self.forward_flash(input)
			}
			AttentionBackend::Auto | AttentionBackend::Standard => {
				self.forward_standard(input, None, mode == AttentionMode::Causal)
			}
		}
	}

	/// Apply standard bidirectional attention with an additive `[B*H*S, S]` mask.
	///
	/// The arbitrary mask route is never silently sent to Flash.
	///
	/// # Errors
	///
	/// Returns an error when Flash is explicitly selected, mask/input geometry is
	/// invalid, or operation recording fails.
	pub fn forward_masked(&self, input: &Matrix, additive_mask: &Matrix) -> Result<Matrix> {
		if self.backend.get() == AttentionBackend::Flash {
			return Err(Error::invalid_argument(
				"FlashAttention does not accept an arbitrary additive mask",
			));
		}
		self.forward_standard(input, Some(additive_mask), false)
	}

	/// Select the requested attention provider policy.
	pub fn set_backend(&self, backend: AttentionBackend) {
		self.backend.set(backend);
	}

	/// Return the requested attention provider policy.
	pub fn backend(&self) -> AttentionBackend {
		self.backend.get()
	}

	/// Return the provider used by the most recent successful forward call.
	pub fn last_backend(&self) -> AttentionBackend {
		self.last_backend.get()
	}

	/// Select causal or bidirectional token visibility.
	pub fn set_mode(&self, mode: AttentionMode) {
		self.mode.set(mode);
	}

	/// Return the configured token-visibility contract.
	pub fn mode(&self) -> AttentionMode {
		self.mode.get()
	}

	pub const fn model_width(&self) -> usize {
		self.model_width
	}
	pub const fn num_heads(&self) -> usize {
		self.num_heads
	}
	pub fn sequence_length(&self) -> usize {
		self.sequence_length.get()
	}

	/// Change runtime sequence geometry without rebuilding projection weights.
	///
	/// # Errors
	///
	/// Returns an error when `sequence_length` is zero.
	pub fn set_sequence_length(&self, sequence_length: usize) -> Result<()> {
		if sequence_length == 0 {
			return Err(Error::invalid_argument(
				"attention sequence length must be nonzero",
			));
		}
		if self.sequence_length.replace(sequence_length) != sequence_length {
			self.mask_cache.borrow_mut().clear();
		}
		Ok(())
	}

	/// Return whether all four projections own trainable biases.
	pub fn has_bias(&self) -> bool {
		self.query_projection.has_bias()
	}

	/// Return the probability dropped from attention weights during training.
	pub const fn dropout_probability(&self) -> f32 {
		self.dropout_probability
	}
}

impl Module for MultiHeadAttention {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		MultiHeadAttention::forward(self, input)
	}
	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
