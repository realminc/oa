use std::rc::Rc;

use crate::ml::{Module, ModuleRegistry};
use crate::{Engine, Error, Matrix, Result};

use super::{
	AlmGenerationOptions, AlmPrior, AlmPriorConfig, AlmTokenizer, AlmTokenizerConfig, ClipText,
	ClipTextConfig, ClipTokenizer,
};

/// Product-level ALM architecture configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AlmConfig {
	/// Temporal VQ-VAE configuration.
	pub tokenizer: AlmTokenizerConfig,
	/// Autoregressive motion-token prior configuration.
	pub prior: AlmPriorConfig,
	/// Optional native frozen CLIP tower for pre-tokenized prompt input.
	pub clip_text: Option<ClipTextConfig>,
}

/// Complete Animation Language Model ownership and persistence boundary.
///
/// The tokenizer and prior remain independently trainable children, while this
/// module gives checkpoints one unambiguous `tokenizer.*` / `prior.*` tree and
/// exposes their composed tokenization and generation paths.
pub struct Alm {
	config: AlmConfig,
	tokenizer: Rc<AlmTokenizer>,
	prior: Rc<AlmPrior>,
	clip_text: Option<Rc<ClipText>>,
	registry: ModuleRegistry,
}

impl Alm {
	/// Construct the complete ALM with deterministic child seeds.
	///
	/// # Errors
	///
	/// Returns an error unless the tokenizer and prior vocabulary contracts agree,
	/// or when either child or the ownership tree cannot be constructed.
	pub fn with_seed(engine: &Engine, config: AlmConfig, seed: u64) -> Result<Self> {
		let tokenizer = Rc::new(AlmTokenizer::with_seed(engine, config.tokenizer, seed)?);
		let prior = Rc::new(AlmPrior::with_seed(
			engine,
			config.prior,
			seed.wrapping_add(1),
		)?);
		let clip_text = config
			.clip_text
			.map(|clip| ClipText::with_seed(engine, clip, seed.wrapping_add(2)).map(Rc::new))
			.transpose()?;
		Self::from_parts_impl(tokenizer, prior, clip_text)
	}

	/// Construct the complete ALM with deterministic seed zero.
	///
	/// # Errors
	///
	/// Returns the same errors as [`Self::with_seed`].
	pub fn new(engine: &Engine, config: AlmConfig) -> Result<Self> {
		Self::with_seed(engine, config, 0)
	}

	/// Adopt independently constructed tokenizer and prior children.
	///
	/// # Errors
	///
	/// Returns an error unless vocabulary and Engine ownership agree, or child
	/// registration fails.
	pub fn from_parts(tokenizer: Rc<AlmTokenizer>, prior: Rc<AlmPrior>) -> Result<Self> {
		Self::from_parts_impl(tokenizer, prior, None)
	}

	/// Adopt tokenizer, conditioned prior, and native frozen CLIP children.
	///
	/// # Errors
	///
	/// Returns an error unless their dimensions and Engine ownership agree.
	pub fn from_text_parts(
		tokenizer: Rc<AlmTokenizer>,
		prior: Rc<AlmPrior>,
		clip_text: Rc<ClipText>,
	) -> Result<Self> {
		Self::from_parts_impl(tokenizer, prior, Some(clip_text))
	}

	fn from_parts_impl(
		tokenizer: Rc<AlmTokenizer>,
		prior: Rc<AlmPrior>,
		clip_text: Option<Rc<ClipText>>,
	) -> Result<Self> {
		if tokenizer.config().num_codes != prior.config().num_codes
			|| tokenizer.config().num_codes.checked_add(3) != Some(prior.config().vocab_size)
		{
			return Err(Error::invalid_argument(
				"ALM tokenizer and prior vocabulary contracts do not match",
			));
		}
		let tokenizer_parameters = tokenizer.all_parameters()?;
		let prior_parameters = prior.all_parameters()?;
		let tokenizer_data = tokenizer_parameters
			.first()
			.ok_or_else(|| Error::failed_precondition("ALM tokenizer has no parameters"))?
			.data();
		let prior_data = prior_parameters
			.first()
			.ok_or_else(|| Error::failed_precondition("ALM prior has no parameters"))?
			.data();
		if !tokenizer_data
			.engine_handle()
			.same_as(prior_data.engine_handle())
		{
			return Err(Error::invalid_argument(
				"ALM tokenizer and prior must belong to the same engine",
			));
		}
		if let Some(clip) = &clip_text {
			if prior.config().text_feature_dim != clip.config().projection_dim {
				return Err(Error::invalid_argument(
					"ALM CLIP projection width must match prior text_feature_dim",
				));
			}
			let clip_parameter =
				clip.all_parameters()?.into_iter().next().ok_or_else(|| {
					Error::failed_precondition("ALM CLIP tower has no parameters")
				})?;
			if !tokenizer_data
				.engine_handle()
				.same_as(clip_parameter.data().engine_handle())
			{
				return Err(Error::invalid_argument(
					"ALM CLIP tower must belong to the same engine",
				));
			}
		} else if prior.config().text_feature_dim != 0 {
			return Err(Error::invalid_argument(
				"conditioned ALM prior requires a native CLIP child in the product bundle",
			));
		}
		let config = AlmConfig {
			tokenizer: *tokenizer.config(),
			prior: *prior.config(),
			clip_text: clip_text.as_ref().map(|clip| *clip.config()),
		};
		let mut registry = ModuleRegistry::new();
		registry.register_module("tokenizer", tokenizer.clone())?;
		registry.register_module("prior", prior.clone())?;
		if let Some(clip) = &clip_text {
			registry.register_module("text_encoder", clip.clone())?;
		}
		Ok(Self {
			config,
			tokenizer,
			prior,
			clip_text,
			registry,
		})
	}

	/// Evaluate prior logits for token ids.
	///
	/// # Errors
	///
	/// Returns an error from the prior's unconditional forward contract.
	pub fn forward(&self, token_ids: &Matrix) -> Result<Matrix> {
		self.prior.forward(token_ids)
	}

	/// Evaluate prior logits from token ids and frozen text features.
	///
	/// # Errors
	///
	/// Returns an error from the prior's conditioned forward contract.
	pub fn forward_conditioned(
		&self,
		token_ids: &Matrix,
		text_features: &Matrix,
	) -> Result<Matrix> {
		self.prior.forward_conditioned(token_ids, text_features)
	}

	/// Encode motion into one or more discrete token streams.
	///
	/// # Errors
	///
	/// Returns an error from tokenizer encoding or quantization.
	pub fn tokenize(&self, motion: &Matrix) -> Result<Vec<Matrix>> {
		self.tokenizer.tokenize(motion)
	}

	/// Decode discrete token streams into motion.
	///
	/// # Errors
	///
	/// Returns an error from tokenizer lookup or decoding.
	pub fn detokenize(&self, token_ids: &[Matrix], batch: usize) -> Result<Matrix> {
		self.tokenizer.detokenize(token_ids, batch)
	}

	/// Generate unconditional tokens and decode the common motion prefix.
	///
	/// Returns `None` when generation emits EOM immediately after SOM.
	///
	/// # Errors
	///
	/// Returns an error from prior generation/readback or tokenizer decoding.
	pub fn generate_motion(
		&self,
		batch: usize,
		options: AlmGenerationOptions,
	) -> Result<Option<Matrix>> {
		let tokens = self.prior.generate(batch, options)?;
		self.prior.decode_to_motion(&tokens, &self.tokenizer)
	}

	/// Generate text-conditioned tokens and decode the common motion prefix.
	///
	/// Returns `None` when generation emits EOM immediately after SOM.
	///
	/// # Errors
	///
	/// Returns an error from conditioned generation/readback or tokenizer decoding.
	pub fn generate_motion_conditioned(
		&self,
		text_features: &Matrix,
		options: AlmGenerationOptions,
	) -> Result<Option<Matrix>> {
		let tokens = self.prior.generate_conditioned(text_features, options)?;
		self.prior.decode_to_motion(&tokens, &self.tokenizer)
	}

	/// Encode pre-tokenized CLIP input and generate conditioned motion.
	///
	/// String-to-token BPE remains a separate host asset/parser contract and is
	/// not inferred from model weights.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` without a native CLIP child, or an error from
	/// text encoding, prior generation, or motion decoding.
	pub fn generate_motion_text_tokens(
		&self,
		text_token_ids: &Matrix,
		flat_eos_rows: &Matrix,
		options: AlmGenerationOptions,
	) -> Result<Option<Matrix>> {
		let clip = self.clip_text.as_ref().ok_or_else(|| {
			Error::failed_precondition("ALM bundle has no native CLIP text encoder")
		})?;
		let features = clip.forward_tokens(text_token_ids, flat_eos_rows)?;
		self.generate_motion_conditioned(&features, options)
	}

	/// Tokenize raw prompts, encode them with the frozen CLIP child, and generate motion.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` without a native CLIP child, or an error from
	/// tokenization, text encoding, prior generation, or motion decoding.
	pub fn generate_motion_text<S: AsRef<str>>(
		&self,
		tokenizer: &ClipTokenizer,
		prompts: &[S],
		truncate: bool,
		options: AlmGenerationOptions,
	) -> Result<Option<Matrix>> {
		let clip = self.clip_text.as_ref().ok_or_else(|| {
			Error::failed_precondition("ALM bundle has no native CLIP text encoder")
		})?;
		let features = clip.forward_prompts(tokenizer, prompts, truncate)?;
		self.generate_motion_conditioned(&features, options)
	}

	/// Return the temporal VQ-VAE child.
	pub fn tokenizer(&self) -> &AlmTokenizer {
		&self.tokenizer
	}

	/// Return the autoregressive prior child.
	pub fn prior(&self) -> &AlmPrior {
		&self.prior
	}

	/// Return the optional native frozen CLIP text child.
	pub fn clip_text(&self) -> Option<&ClipText> {
		self.clip_text.as_deref()
	}

	/// Return the complete architecture configuration.
	pub const fn config(&self) -> &AlmConfig {
		&self.config
	}
}

impl Module for Alm {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		Alm::forward(self, input)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}
