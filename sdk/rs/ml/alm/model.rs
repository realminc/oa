use std::{path::Path, rc::Rc};

use crate::ml::{
	Module, ModuleArtifact, ModuleArtifactMetadata, ModuleRegistry, NamedBuffer, save_module_artifact,
};
use crate::{Engine, Error, Matrix, Result};

use super::{
	AlmGenerationOptions, AlmPrior, AlmPriorConfig, AlmTokenizer, AlmTokenizerConfig, ClipText,
	ClipTextConfig, ClipTokenizer, clip::clip_artifact_path,
};

const BUNDLE_MAGIC: u32 = 0x4741_4D41;
const BUNDLE_VERSION: u32 = 3;
const BUNDLE_CONFIG_SIZE: usize = 205;
const NATIVE_TEXT_ENCODER: &str = "openai/clip-vit-large-patch14";

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
	text_encoder_identity: Option<String>,
	text_tokenizer_merges: Option<NamedBuffer>,
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
		Self::from_parts_impl(tokenizer, prior, clip_text, None, None)
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
		Self::from_parts_impl(tokenizer, prior, None, None, None)
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
		Self::from_parts_impl(tokenizer, prior, Some(clip_text), None, None)
	}

	/// Adopt a conditioned prior whose text features are supplied externally.
	///
	/// The exact encoder identity is persisted in product bundles, while the
	/// encoder weights and tokenizer asset remain outside this module.
	///
	/// # Errors
	///
	/// Returns an error unless the prior is conditioned, the identity is nonempty,
	/// and the child vocabulary and Engine contracts agree.
	pub fn from_external_text_parts(
		tokenizer: Rc<AlmTokenizer>,
		prior: Rc<AlmPrior>,
		text_encoder_identity: impl Into<String>,
	) -> Result<Self> {
		if prior.config().text_feature_dim == 0 {
			return Err(Error::invalid_argument(
				"external text-encoder identity requires a conditioned ALM prior",
			));
		}
		Self::from_parts_impl(
			tokenizer,
			prior,
			None,
			Some(text_encoder_identity.into()),
			None,
		)
	}

	/// Adopt the pinned native CLIP tower and its canonical byte-BPE merge asset.
	///
	/// # Errors
	///
	/// Returns an error unless the CLIP configuration is the donor ViT-L/14
	/// contract, merges are nonempty and valid, and all children share one Engine.
	pub fn from_native_text_parts(
		tokenizer: Rc<AlmTokenizer>,
		prior: Rc<AlmPrior>,
		clip_text: Rc<ClipText>,
		clip_merges: &[u8],
	) -> Result<Self> {
		if *clip_text.config() != ClipTextConfig::default() {
			return Err(Error::invalid_argument(
				"native ALM bundle requires the pinned CLIP ViT-L/14 configuration",
			));
		}
		let mut tokenizer_check = ClipTokenizer::new();
		tokenizer_check.load_merges(clip_merges)?;
		Self::from_parts_impl(
			tokenizer,
			prior,
			Some(clip_text),
			Some(NATIVE_TEXT_ENCODER.to_owned()),
			Some(clip_merges),
		)
	}

	fn from_parts_impl(
		tokenizer: Rc<AlmTokenizer>,
		prior: Rc<AlmPrior>,
		clip_text: Option<Rc<ClipText>>,
		text_encoder_identity: Option<String>,
		clip_merges: Option<&[u8]>,
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
			let clip_parameter = clip
				.all_parameters()?
				.into_iter()
				.next()
				.ok_or_else(|| Error::failed_precondition("ALM CLIP tower has no parameters"))?;
			if !tokenizer_data
				.engine_handle()
				.same_as(clip_parameter.data().engine_handle())
			{
				return Err(Error::invalid_argument(
					"ALM CLIP tower must belong to the same engine",
				));
			}
		} else if prior.config().text_feature_dim != 0 && text_encoder_identity.is_none() {
			return Err(Error::invalid_argument(
				"conditioned ALM prior requires an exact external text-encoder identity",
			));
		}
		if text_encoder_identity.as_deref().is_some_and(str::is_empty) {
			return Err(Error::invalid_argument(
				"ALM text-encoder identity cannot be empty",
			));
		}
		if clip_merges.is_some() && text_encoder_identity.as_deref() != Some(NATIVE_TEXT_ENCODER) {
			return Err(Error::invalid_argument(
				"native ALM CLIP merges require the pinned text-encoder identity",
			));
		}
		if clip_merges.is_some()
			&& clip_text
				.as_ref()
				.is_none_or(|clip| *clip.config() != ClipTextConfig::default())
		{
			return Err(Error::invalid_argument(
				"native ALM CLIP merges require the pinned CLIP ViT-L/14 child",
			));
		}
		let config = AlmConfig {
			tokenizer: *tokenizer.config(),
			prior: *prior.config(),
			clip_text: clip_text.as_ref().map(|clip| *clip.config()),
		};
		let mut registry = ModuleRegistry::new();
		let text_tokenizer_merges = if let Some(bytes) = clip_merges {
			if bytes.is_empty() {
				return Err(Error::invalid_argument(
					"native ALM CLIP merges cannot be empty",
				));
			}
			let data =
				Matrix::from_slice_handle(tokenizer_data.engine_handle(), vec![bytes.len()], bytes)?;
			registry.register_buffer("text_tokenizer_merges", data, true)?;
			Some(
				registry
					.buffer_handle("text_tokenizer_merges")
					.ok_or_else(|| Error::internal("ALM tokenizer merge registration was lost"))?,
			)
		} else {
			None
		};
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
			text_encoder_identity,
			text_tokenizer_merges,
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
	pub fn forward_conditioned(&self, token_ids: &Matrix, text_features: &Matrix) -> Result<Matrix> {
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
		let clip = self
			.clip_text
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("ALM bundle has no native CLIP text encoder"))?;
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
		let clip = self
			.clip_text
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("ALM bundle has no native CLIP text encoder"))?;
		let features = clip.forward_prompts(tokenizer, prompts, truncate)?;
		self.generate_motion_conditioned(&features, options)
	}

	/// Encode raw prompts with the CLIP tower and merge table owned by this bundle.
	///
	/// This is an explicit host boundary: the small persistent merge buffer is
	/// read, validated, and used by the host byte-BPE tokenizer before token upload.
	///
	/// # Errors
	///
	/// Returns `FailedPrecondition` unless both native CLIP assets are bundled, or
	/// a readback, tokenizer, upload, or text-tower operation fails.
	pub fn encode_prompts<S: AsRef<str>>(&self, prompts: &[S]) -> Result<Matrix> {
		let clip = self
			.clip_text
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("ALM bundle has no native CLIP text encoder"))?;
		let merges = self
			.text_tokenizer_merges
			.as_ref()
			.ok_or_else(|| Error::failed_precondition("ALM bundle has no native CLIP tokenizer"))?;
		let mut tokenizer = ClipTokenizer::new();
		tokenizer.load_merges(&merges.data().read::<u8>()?)?;
		clip.forward_prompts(&tokenizer, prompts, true)
	}

	/// Encode one prompt with the complete native text bundle.
	///
	/// # Errors
	///
	/// Returns the same errors as [`Self::encode_prompts`].
	pub fn encode_prompt(&self, prompt: &str) -> Result<Matrix> {
		self.encode_prompts(&[prompt])
	}

	/// Encode one prompt, sample motion tokens, and decode their common prefix.
	///
	/// # Errors
	///
	/// Returns an error from native prompt encoding, conditioned generation, or
	/// motion decoding.
	pub fn generate_motion_prompt(
		&self,
		prompt: &str,
		options: AlmGenerationOptions,
	) -> Result<Option<Matrix>> {
		let features = self.encode_prompt(prompt)?;
		self.generate_motion_conditioned(&features, options)
	}

	/// Save one versioned product artifact containing every child and persistent buffer.
	///
	/// Optimizer state is intentionally excluded; training checkpoints remain a
	/// separate contract. Tensor paths and the packed 205-byte payload match the
	/// donor C++ `OaAlmAg` bundle.
	///
	/// # Errors
	///
	/// Returns an error for an incomplete native text bundle, unsupported integer
	/// widths, module traversal, readback, serialization, or filesystem failure.
	pub fn save_bundle(&self, path: impl AsRef<Path>) -> Result<()> {
		if self.clip_text.is_some() != self.text_tokenizer_merges.is_some() {
			return Err(Error::failed_precondition(
				"ALM native text encoder and tokenizer merges must be bundled together",
			));
		}
		let arch_config = encode_bundle_config(self)?;
		save_module_artifact(
			path.as_ref(),
			self,
			ModuleArtifactMetadata {
				architecture: "OaAlmAg".to_owned(),
				config_version: BUNDLE_VERSION,
				d_model: to_u32(self.config.prior.model_width, "ALM model width")?,
				n_layers: to_u32(self.config.prior.num_layers, "ALM layer count")?,
				d_vocab: to_u32(self.config.prior.vocab_size, "ALM vocabulary")?,
				arch_config,
			},
			alm_artifact_path,
		)
	}

	/// Load a C++/Rust `OaAlmAg` v3 product bundle.
	///
	/// The file and packed architecture payload are validated before construction;
	/// all tensors are then validated and uploaded before any module handle changes.
	///
	/// # Errors
	///
	/// Returns `CheckpointCorrupt` for an unsupported or inconsistent artifact,
	/// and another error for invalid architecture, allocation, upload, or runtime
	/// failure.
	pub fn load_bundle(engine: &Engine, path: impl AsRef<Path>) -> Result<Self> {
		let artifact = ModuleArtifact::load(path.as_ref())?;
		let metadata = artifact.metadata();
		if metadata.architecture != "OaAlmAg"
			|| metadata.config_version != BUNDLE_VERSION
			|| metadata.arch_config.len() != BUNDLE_CONFIG_SIZE
		{
			return Err(Error::checkpoint_corrupt(
				"checkpoint is not a supported OaAlmAg v3 bundle",
			));
		}
		let decoded = decode_bundle_config(&metadata.arch_config)?;
		if metadata.d_model != to_u32(decoded.config.prior.model_width, "ALM model width")?
			|| metadata.n_layers != to_u32(decoded.config.prior.num_layers, "ALM layer count")?
			|| metadata.d_vocab != to_u32(decoded.config.prior.vocab_size, "ALM vocabulary")?
		{
			return Err(Error::checkpoint_corrupt(
				"ALM summary metadata disagrees with its architecture payload",
			));
		}

		let tokenizer = Rc::new(AlmTokenizer::new(engine, decoded.config.tokenizer)?);
		let prior = Rc::new(AlmPrior::new(engine, decoded.config.prior)?);
		let model = if decoded.merge_bytes == 0 {
			Self::from_parts_impl(tokenizer, prior, None, decoded.text_encoder, None)?
		} else {
			let clip = Rc::new(ClipText::new(engine, ClipTextConfig::default())?);
			let placeholder = vec![0_u8; decoded.merge_bytes];
			Self::from_parts_impl(
				tokenizer,
				prior,
				Some(clip),
				decoded.text_encoder,
				Some(&placeholder),
			)?
		};
		artifact.restore(engine, &model, alm_artifact_path)?;
		Ok(model)
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

	/// Return the exact external or bundled text-encoder identity, when conditioned.
	pub fn text_encoder_identity(&self) -> Option<&str> {
		self.text_encoder_identity.as_deref()
	}

	/// Whether this artifact owns both the native CLIP tower and tokenizer asset.
	pub fn has_native_text_encoder(&self) -> bool {
		self.clip_text.is_some() && self.text_tokenizer_merges.is_some()
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

struct DecodedBundle {
	config: AlmConfig,
	merge_bytes: usize,
	text_encoder: Option<String>,
}

fn alm_artifact_path(path: &str) -> Result<String> {
	if let Some(suffix) = path.strip_prefix("text_encoder.") {
		return Ok(format!("text_encoder.{}", clip_artifact_path(suffix)?));
	}
	Ok(path.to_owned())
}

fn encode_bundle_config(model: &Alm) -> Result<Vec<u8>> {
	let tokenizer = model.config.tokenizer;
	let prior = model.config.prior;
	let merge_bytes = model
		.text_tokenizer_merges
		.as_ref()
		.map_or(0, |buffer| buffer.data().num_elements());
	let identity = model.text_encoder_identity.as_deref().unwrap_or("");
	if identity.as_bytes().contains(&0) || identity.len() >= 96 {
		return Err(Error::invalid_argument(
			"ALM text-encoder identity must fit a NUL-terminated 96-byte field",
		));
	}
	if merge_bytes > 0
		&& (identity != NATIVE_TEXT_ENCODER
			|| model.clip_text.as_ref().map(|clip| *clip.config()) != Some(ClipTextConfig::default()))
	{
		return Err(Error::failed_precondition(
			"native ALM bundle does not match the pinned CLIP ViT-L/14 contract",
		));
	}

	let mut bytes = Vec::with_capacity(BUNDLE_CONFIG_SIZE);
	push_u32(&mut bytes, BUNDLE_MAGIC);
	push_u32(&mut bytes, BUNDLE_VERSION);
	for (value, label) in [
		(tokenizer.input_dim, "tokenizer input width"),
		(tokenizer.width, "tokenizer hidden width"),
		(tokenizer.code_dim, "tokenizer code width"),
		(tokenizer.num_codes, "tokenizer code count"),
		(tokenizer.downsample_stages, "tokenizer downsample stages"),
		(tokenizer.depth, "tokenizer depth"),
	] {
		push_i32(&mut bytes, to_i32(value, label)?);
	}
	for value in [
		tokenizer.commitment_beta,
		tokenizer.ema_decay,
		tokenizer.ema_epsilon,
		tokenizer.dead_threshold,
	] {
		push_f32(&mut bytes, value);
	}
	for (value, label) in [
		(prior.model_width, "prior model width"),
		(prior.num_heads, "prior head count"),
		(prior.num_layers, "prior layer count"),
		(prior.hidden_width, "prior hidden width"),
		(prior.text_feature_dim, "prior text feature width"),
	] {
		push_i32(&mut bytes, to_i32(value, label)?);
	}
	bytes.push(match prior.ffn_type {
		super::AlmFfnType::Dense => 0,
		super::AlmFfnType::Moe => 1,
		super::AlmFfnType::Hybrid => 2,
	});
	for (value, label) in [
		(prior.moe_num_experts, "prior expert count"),
		(prior.moe_experts_per_token, "prior experts per token"),
		(prior.moe_every, "prior MoE cadence"),
	] {
		push_i32(&mut bytes, to_i32(value, label)?);
	}
	for value in [
		prior.moe_balance_rate,
		prior.moe_aux_loss_alpha,
		prior.moe_router_z_loss_beta,
	] {
		push_f32(&mut bytes, value);
	}
	for (value, label) in [
		(prior.sequence_length, "prior sequence length"),
		(prior.max_sequence_length, "prior maximum sequence length"),
		(
			prior.max_generation_length,
			"prior maximum generation length",
		),
	] {
		push_i32(&mut bytes, to_i32(value, label)?);
	}
	push_u32(
		&mut bytes,
		to_u32(merge_bytes, "CLIP merge-table byte count")?,
	);
	bytes.extend_from_slice(identity.as_bytes());
	bytes.resize(BUNDLE_CONFIG_SIZE, 0);
	debug_assert_eq!(bytes.len(), BUNDLE_CONFIG_SIZE);
	Ok(bytes)
}

fn decode_bundle_config(bytes: &[u8]) -> Result<DecodedBundle> {
	if bytes.len() != BUNDLE_CONFIG_SIZE {
		return Err(Error::checkpoint_corrupt(
			"ALM architecture payload must contain 205 bytes",
		));
	}
	let mut reader = BundleReader::new(bytes);
	if reader.u32()? != BUNDLE_MAGIC || reader.u32()? != BUNDLE_VERSION {
		return Err(Error::checkpoint_corrupt(
			"ALM architecture payload has an invalid version",
		));
	}
	let tokenizer = AlmTokenizerConfig {
		input_dim: reader.positive("tokenizer input width")?,
		width: reader.positive("tokenizer hidden width")?,
		code_dim: reader.positive("tokenizer code width")?,
		num_codes: reader.positive("tokenizer code count")?,
		downsample_stages: reader.positive("tokenizer downsample stages")?,
		depth: reader.positive("tokenizer depth")?,
		commitment_beta: reader.f32()?,
		ema_decay: reader.f32()?,
		ema_epsilon: reader.f32()?,
		dead_threshold: reader.f32()?,
	};
	let mut prior = AlmPriorConfig::default();
	prior.sync_vocab(tokenizer.num_codes)?;
	prior.model_width = reader.positive("prior model width")?;
	prior.num_heads = reader.positive("prior head count")?;
	prior.num_layers = reader.positive("prior layer count")?;
	prior.hidden_width = reader.positive("prior hidden width")?;
	prior.text_feature_dim = reader.nonnegative("prior text feature width")?;
	prior.ffn_type = match reader.u8()? {
		0 => super::AlmFfnType::Dense,
		1 => super::AlmFfnType::Moe,
		2 => super::AlmFfnType::Hybrid,
		_ => {
			return Err(Error::checkpoint_corrupt(
				"ALM bundle has an invalid FFN type",
			));
		}
	};
	prior.moe_num_experts = reader.nonnegative("prior expert count")?;
	prior.moe_experts_per_token = reader.nonnegative("prior experts per token")?;
	prior.moe_every = reader.nonnegative("prior MoE cadence")?;
	prior.moe_balance_rate = reader.f32()?;
	prior.moe_aux_loss_alpha = reader.f32()?;
	prior.moe_router_z_loss_beta = reader.f32()?;
	prior.sequence_length = reader.positive("prior sequence length")?;
	prior.max_sequence_length = reader.positive("prior maximum sequence length")?;
	prior.max_generation_length = reader.positive("prior maximum generation length")?;
	let merge_bytes = usize::try_from(reader.u32()?)
		.map_err(|_| Error::checkpoint_corrupt("CLIP merge-table size exceeds usize"))?;
	let text_encoder = reader.fixed_string(96)?;
	if prior.text_feature_dim > 0 && text_encoder.is_none() {
		return Err(Error::invalid_argument(
			"conditioned ALM bundle requires an exact text-encoder identity",
		));
	}
	if merge_bytes > 0
		&& (prior.text_feature_dim != ClipTextConfig::default().projection_dim
			|| text_encoder.as_deref() != Some(NATIVE_TEXT_ENCODER))
	{
		return Err(Error::invalid_argument(
			"native ALM bundle requires the pinned CLIP ViT-L/14 identity and feature width",
		));
	}
	Ok(DecodedBundle {
		config: AlmConfig {
			tokenizer,
			prior,
			clip_text: (merge_bytes > 0).then_some(ClipTextConfig::default()),
		},
		merge_bytes,
		text_encoder,
	})
}

struct BundleReader<'a> {
	bytes: &'a [u8],
	offset: usize,
}

impl<'a> BundleReader<'a> {
	const fn new(bytes: &'a [u8]) -> Self {
		Self { bytes, offset: 0 }
	}

	fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
		let end = self
			.offset
			.checked_add(N)
			.ok_or_else(|| Error::checkpoint_corrupt("ALM payload offset overflows usize"))?;
		let value = self
			.bytes
			.get(self.offset..end)
			.and_then(|value| value.try_into().ok())
			.ok_or_else(|| Error::checkpoint_corrupt("truncated ALM architecture payload"))?;
		self.offset = end;
		Ok(value)
	}

	fn u8(&mut self) -> Result<u8> {
		Ok(self.take::<1>()?[0])
	}

	fn u32(&mut self) -> Result<u32> {
		Ok(u32::from_le_bytes(self.take()?))
	}

	fn i32(&mut self) -> Result<i32> {
		Ok(i32::from_le_bytes(self.take()?))
	}

	fn f32(&mut self) -> Result<f32> {
		Ok(f32::from_le_bytes(self.take()?))
	}

	fn positive(&mut self, label: &str) -> Result<usize> {
		let value = self.i32()?;
		if value <= 0 {
			return Err(Error::invalid_argument(format!(
				"ALM {label} must be positive"
			)));
		}
		usize::try_from(value)
			.map_err(|_| Error::checkpoint_corrupt(format!("ALM {label} exceeds usize")))
	}

	fn nonnegative(&mut self, label: &str) -> Result<usize> {
		usize::try_from(self.i32()?)
			.map_err(|_| Error::invalid_argument(format!("ALM {label} cannot be negative")))
	}

	fn fixed_string(&mut self, size: usize) -> Result<Option<String>> {
		let end = self
			.offset
			.checked_add(size)
			.ok_or_else(|| Error::checkpoint_corrupt("ALM string offset overflows usize"))?;
		let field = self
			.bytes
			.get(self.offset..end)
			.ok_or_else(|| Error::checkpoint_corrupt("truncated ALM text-encoder identity"))?;
		self.offset = end;
		let length = field
			.iter()
			.position(|byte| *byte == 0)
			.unwrap_or(field.len());
		let value = std::str::from_utf8(&field[..length])
			.map_err(|_| Error::checkpoint_corrupt("ALM text-encoder identity is not UTF-8"))?;
		Ok((!value.is_empty()).then(|| value.to_owned()))
	}
}

fn to_i32(value: usize, label: &str) -> Result<i32> {
	i32::try_from(value).map_err(|_| Error::resource_exhausted(format!("{label} exceeds i32")))
}

fn to_u32(value: usize, label: &str) -> Result<u32> {
	u32::try_from(value).map_err(|_| Error::resource_exhausted(format!("{label} exceeds u32")))
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
	bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
	bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
	bytes.extend_from_slice(&value.to_le_bytes());
}
