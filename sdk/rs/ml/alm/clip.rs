use std::{collections::HashSet, path::Path, rc::Rc};

use crate::ml::{
	DenseArtifactTensor, Module, ModuleArtifact, ModuleArtifactMetadata, ModuleRegistry,
	SafeTensorsSource, matrix as ml_matrix,
	nn::{Embedding, LayerNorm, Linear, MultiHeadAttention},
	save_dense_artifact, save_module_artifact,
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

/// Summary of one external CLIP SafeTensors translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipImportReport {
	/// Complete tensor count in the source container.
	pub source_tensors: usize,
	/// Text-tower tensors consumed by the exact translator.
	pub used_source_tensors: usize,
	/// Native `.oam` weight count.
	pub output_tensors: usize,
	/// Total dense FP32 tensor bytes written to Weights.
	pub output_bytes: usize,
	/// Source tensors intentionally excluded, including the vision tower.
	pub unused_source_tensors: usize,
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
	/// Translate the published CLIP ViT-L/14 SafeTensors image into native `.oam`.
	///
	/// The translator admits every and only text-tower FP32 tensor, validates exact
	/// shapes, ignores unrelated vision/logit-scale tensors, and rejects unknown
	/// names below the text namespaces. No Vulkan Engine is needed for this host
	/// conversion.
	///
	/// # Errors
	///
	/// Returns an error for malformed SafeTensors data, a missing/unexpected text
	/// tensor, dtype or shape mismatch, arithmetic overflow, or output failure.
	pub fn import_safetensors(
		input: impl AsRef<Path>,
		output: impl AsRef<Path>,
	) -> Result<ClipImportReport> {
		import_safetensors_config(input.as_ref(), output.as_ref(), ClipTextConfig::default())
	}

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

	/// Save this frozen text tower as an OA `OaClipTextAg` v1 model file.
	///
	/// Tensor names and the 48-byte architecture payload match the C++ model
	/// translator, allowing the same `.oam` artifact to cross language boundaries.
	///
	/// # Errors
	///
	/// Returns an error for unsupported integer widths, incomplete module
	/// ownership, readback, serialization, or filesystem failure.
	pub fn save_model(&self, path: impl AsRef<Path>) -> Result<()> {
		save_module_artifact(
			path.as_ref(),
			self,
			ModuleArtifactMetadata {
				architecture: "OaClipTextAg".to_owned(),
				config_version: 1,
				d_model: u32::try_from(self.config.hidden_size)
					.map_err(|_| Error::resource_exhausted("CLIP hidden size exceeds u32"))?,
				n_layers: u32::try_from(self.config.num_layers)
					.map_err(|_| Error::resource_exhausted("CLIP layer count exceeds u32"))?,
				d_vocab: u32::try_from(self.config.vocab_size)
					.map_err(|_| Error::resource_exhausted("CLIP vocabulary exceeds u32"))?,
				arch_config: encode_config(self.config)?,
			},
			clip_artifact_path,
		)
	}

	/// Load an OA C++/Rust `OaClipTextAg` v1 model file.
	///
	/// The wire file is fully validated before its architecture is constructed;
	/// every tensor is then shape/dtype/name checked before any live parameter is
	/// replaced.
	///
	/// # Errors
	///
	/// Returns `CheckpointCorrupt` for the wrong architecture/version/payload or
	/// malformed wire data, and another error for invalid model geometry,
	/// allocation, upload, or destination mismatch.
	pub fn load_model(engine: &Engine, path: impl AsRef<Path>) -> Result<Self> {
		let artifact = ModuleArtifact::load(path.as_ref())?;
		let metadata = artifact.metadata();
		if metadata.architecture != "OaClipTextAg"
			|| metadata.config_version != 1
			|| metadata.arch_config.len() != 48
		{
			return Err(Error::checkpoint_corrupt(
				"checkpoint is not an OaClipTextAg v1 model",
			));
		}
		let config = decode_config(&metadata.arch_config)?;
		let d_model = u32::try_from(config.hidden_size)
			.map_err(|_| Error::checkpoint_corrupt("CLIP hidden size exceeds u32"))?;
		let n_layers = u32::try_from(config.num_layers)
			.map_err(|_| Error::checkpoint_corrupt("CLIP layer count exceeds u32"))?;
		let d_vocab = u32::try_from(config.vocab_size)
			.map_err(|_| Error::checkpoint_corrupt("CLIP vocabulary exceeds u32"))?;
		if metadata.d_model != d_model || metadata.n_layers != n_layers || metadata.d_vocab != d_vocab {
			return Err(Error::checkpoint_corrupt(
				"CLIP summary metadata disagrees with its architecture payload",
			));
		}
		let model = Self::new(engine, config)?;
		artifact.restore(engine, &model, clip_artifact_path)?;
		model.freeze()?;
		Ok(model)
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
		self
			.text_projection
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
				.map(|value| {
					i32::try_from(value)
						.map_err(|_| Error::out_of_range("CLIP token ID exceeds the supported I32 vocabulary"))
				})
				.collect::<Result<Vec<_>>>()?,
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

pub(super) fn clip_artifact_path(path: &str) -> Result<String> {
	if let Some(suffix) = path.strip_prefix("token_embedding") {
		return Ok(format!("text_model.embeddings.token_embedding{suffix}"));
	}
	if let Some(suffix) = path.strip_prefix("position_embedding") {
		return Ok(format!("text_model.embeddings.position_embedding{suffix}"));
	}
	if let Some(suffix) = path.strip_prefix("final_layer_norm") {
		return Ok(format!("text_model.final_layer_norm{suffix}"));
	}
	if path == "text_projection.weight" {
		return Ok(path.to_owned());
	}
	let (layer, suffix) = path
		.split_once('.')
		.ok_or_else(|| Error::internal(format!("unrecognized CLIP parameter path {path}")))?;
	let index = layer
		.strip_prefix("layer_")
		.ok_or_else(|| Error::internal(format!("unrecognized CLIP parameter path {path}")))?;
	if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
		return Err(Error::internal(format!("invalid CLIP layer path {path}")));
	}
	let suffix = if let Some(value) = suffix.strip_prefix("mlp_fc1") {
		format!("mlp.fc1{value}")
	} else if let Some(value) = suffix.strip_prefix("mlp_fc2") {
		format!("mlp.fc2{value}")
	} else {
		suffix.to_owned()
	};
	Ok(format!("text_model.encoder.layers.{index}.{suffix}"))
}

fn encode_config(config: ClipTextConfig) -> Result<Vec<u8>> {
	let integers = [
		config.vocab_size,
		config.context_length,
		config.hidden_size,
		config.intermediate_size,
		config.num_heads,
		config.num_layers,
		config.projection_dim,
	];
	let tokens = [config.bos_token, config.eos_token, config.pad_token];
	let mut bytes = Vec::with_capacity(48);
	for value in integers {
		bytes.extend_from_slice(
			&i32::try_from(value)
				.map_err(|_| Error::resource_exhausted("CLIP configuration exceeds i32"))?
				.to_le_bytes(),
		);
	}
	bytes.extend_from_slice(&config.layer_norm_epsilon.to_le_bytes());
	bytes.extend_from_slice(&config.quick_gelu_alpha.to_le_bytes());
	for value in tokens {
		bytes.extend_from_slice(&value.to_le_bytes());
	}
	debug_assert_eq!(bytes.len(), 48);
	Ok(bytes)
}

fn import_safetensors_config(
	input: &Path,
	output: &Path,
	config: ClipTextConfig,
) -> Result<ClipImportReport> {
	validate_config(config)?;
	let source = SafeTensorsSource::open(input)?;
	let expected = expected_tensors(config);
	let expected_names = expected
		.iter()
		.map(|(name, _)| name.as_str())
		.collect::<HashSet<_>>();
	if let Some(shape) = source.shape("text_model.embeddings.position_ids")
		&& shape != [1, config.context_length]
	{
		return Err(Error::invalid_argument(
			"CLIP position_ids buffer has the wrong shape",
		));
	}
	for name in source.names() {
		let text_tensor = name.starts_with("text_model.") || name.starts_with("text_projection");
		if text_tensor && name != "text_model.embeddings.position_ids" && !expected_names.contains(name)
		{
			return Err(Error::failed_precondition(format!(
				"unexpected CLIP text tensor: {name}"
			)));
		}
	}

	let mut tensors = Vec::with_capacity(expected.len());
	let mut output_bytes = 0_usize;
	for (name, shape) in expected {
		let source_shape = source
			.shape(&name)
			.ok_or_else(|| Error::not_found(format!("CLIP tensor missing: {name}")))?;
		if source_shape != shape {
			return Err(Error::invalid_argument(format!(
				"CLIP shape mismatch: {name}"
			)));
		}
		let data = source.f32_bytes(&name)?.to_vec();
		output_bytes = output_bytes
			.checked_add(data.len())
			.ok_or_else(|| Error::resource_exhausted("CLIP output byte count overflows usize"))?;
		tensors.push(DenseArtifactTensor { name, shape, data });
	}
	let source_tensors = source.names().count();
	let output_tensors = tensors.len();
	save_dense_artifact(
		output,
		ModuleArtifactMetadata {
			architecture: "OaClipTextAg".to_owned(),
			config_version: 1,
			d_model: u32::try_from(config.hidden_size)
				.map_err(|_| Error::resource_exhausted("CLIP hidden size exceeds u32"))?,
			n_layers: u32::try_from(config.num_layers)
				.map_err(|_| Error::resource_exhausted("CLIP layer count exceeds u32"))?,
			d_vocab: u32::try_from(config.vocab_size)
				.map_err(|_| Error::resource_exhausted("CLIP vocabulary exceeds u32"))?,
			arch_config: encode_config(config)?,
		},
		tensors,
	)?;
	Ok(ClipImportReport {
		source_tensors,
		used_source_tensors: output_tensors,
		output_tensors,
		output_bytes,
		unused_source_tensors: source_tensors.saturating_sub(output_tensors),
	})
}

fn expected_tensors(config: ClipTextConfig) -> Vec<(String, Vec<usize>)> {
	let mut tensors = Vec::with_capacity(5 + config.num_layers * 16);
	tensors.push((
		"text_model.embeddings.token_embedding.weight".to_owned(),
		vec![config.vocab_size, config.hidden_size],
	));
	tensors.push((
		"text_model.embeddings.position_embedding.weight".to_owned(),
		vec![config.context_length, config.hidden_size],
	));
	for layer in 0..config.num_layers {
		let root = format!("text_model.encoder.layers.{layer}");
		for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
			let base = format!("{root}.self_attn.{projection}");
			tensors.push((
				format!("{base}.weight"),
				vec![config.hidden_size, config.hidden_size],
			));
			tensors.push((format!("{base}.bias"), vec![config.hidden_size]));
		}
		for norm in ["layer_norm1", "layer_norm2"] {
			let base = format!("{root}.{norm}");
			tensors.push((format!("{base}.weight"), vec![config.hidden_size]));
			tensors.push((format!("{base}.bias"), vec![config.hidden_size]));
		}
		tensors.push((
			format!("{root}.mlp.fc1.weight"),
			vec![config.intermediate_size, config.hidden_size],
		));
		tensors.push((
			format!("{root}.mlp.fc1.bias"),
			vec![config.intermediate_size],
		));
		tensors.push((
			format!("{root}.mlp.fc2.weight"),
			vec![config.hidden_size, config.intermediate_size],
		));
		tensors.push((format!("{root}.mlp.fc2.bias"), vec![config.hidden_size]));
	}
	tensors.push((
		"text_model.final_layer_norm.weight".to_owned(),
		vec![config.hidden_size],
	));
	tensors.push((
		"text_model.final_layer_norm.bias".to_owned(),
		vec![config.hidden_size],
	));
	tensors.push((
		"text_projection.weight".to_owned(),
		vec![config.projection_dim, config.hidden_size],
	));
	tensors
}

fn decode_config(bytes: &[u8]) -> Result<ClipTextConfig> {
	if bytes.len() != 48 {
		return Err(Error::checkpoint_corrupt(
			"CLIP architecture payload must contain 48 bytes",
		));
	}
	let mut offset = 0_usize;
	let mut word = || {
		let value = bytes
			.get(offset..offset + 4)
			.and_then(|value| value.try_into().ok())
			.map(i32::from_le_bytes)
			.ok_or_else(|| Error::checkpoint_corrupt("truncated CLIP architecture payload"));
		offset += 4;
		value
	};
	let vocab_size = positive_usize(word()?, "vocabulary")?;
	let context_length = positive_usize(word()?, "context length")?;
	let hidden_size = positive_usize(word()?, "hidden size")?;
	let intermediate_size = positive_usize(word()?, "intermediate size")?;
	let num_heads = positive_usize(word()?, "head count")?;
	let num_layers = positive_usize(word()?, "layer count")?;
	let projection_dim = positive_usize(word()?, "projection width")?;
	let layer_norm_epsilon = f32::from_bits(u32::from_le_bytes(word()?.to_le_bytes()));
	let quick_gelu_alpha = f32::from_bits(u32::from_le_bytes(word()?.to_le_bytes()));
	let bos_token = word()?;
	let eos_token = word()?;
	let pad_token = word()?;
	let config = ClipTextConfig {
		vocab_size,
		context_length,
		hidden_size,
		intermediate_size,
		num_heads,
		num_layers,
		projection_dim,
		layer_norm_epsilon,
		quick_gelu_alpha,
		bos_token,
		eos_token,
		pad_token,
	};
	validate_config(config)?;
	Ok(config)
}

fn positive_usize(value: i32, field: &str) -> Result<usize> {
	if value <= 0 {
		return Err(Error::checkpoint_corrupt(format!(
			"CLIP {field} must be a positive integer"
		)));
	}
	usize::try_from(value)
		.map_err(|_| Error::checkpoint_corrupt(format!("CLIP {field} exceeds usize")))
}

#[cfg(test)]
mod tests {
	use std::time::SystemTime;

	use serde_json::{Map, Value, json};

	use super::{ClipTextConfig, expected_tensors, import_safetensors_config};

	#[test]
	fn safe_tensors_translation_emits_exact_text_inventory() -> crate::Result<()> {
		let config = ClipTextConfig {
			vocab_size: 8,
			context_length: 4,
			hidden_size: 4,
			intermediate_size: 8,
			num_heads: 1,
			num_layers: 1,
			projection_dim: 3,
			layer_norm_epsilon: 1.0e-5,
			quick_gelu_alpha: 1.702,
			bos_token: 6,
			eos_token: 7,
			pad_token: 7,
		};
		let directory = std::env::temp_dir().join(format!(
			"oars-clip-import-{}-{}",
			std::process::id(),
			SystemTime::now()
				.duration_since(SystemTime::UNIX_EPOCH)
				.expect("system clock predates Unix epoch")
				.as_nanos()
		));
		std::fs::create_dir_all(&directory).expect("create CLIP import test directory");
		let input = directory.join("clip.safetensors");
		let output = directory.join("clip.oam");

		let mut header = Map::new();
		let mut payload = Vec::new();
		for (name, shape) in expected_tensors(config) {
			let start = payload.len();
			let elements = shape.iter().product::<usize>();
			payload.resize(start + elements * 4, 0);
			header.insert(
				name,
				json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
			);
		}
		let start = payload.len();
		payload.resize(start + config.context_length * 8, 0);
		header.insert(
			"text_model.embeddings.position_ids".to_owned(),
			json!({
				"dtype": "I64",
				"shape": [1, config.context_length],
				"data_offsets": [start, payload.len()]
			}),
		);
		let start = payload.len();
		payload.resize(start + 4, 0);
		header.insert(
			"visual.unused.weight".to_owned(),
			json!({"dtype": "F32", "shape": [1], "data_offsets": [start, payload.len()]}),
		);
		header.insert(
			"__metadata__".to_owned(),
			Value::Object(Map::from_iter([(
				"format".to_owned(),
				Value::String("pt".to_owned()),
			)])),
		);
		let header = serde_json::to_vec(&Value::Object(header)).expect("serialize test header");
		let mut file = Vec::with_capacity(8 + header.len() + payload.len());
		file.extend_from_slice(&(header.len() as u64).to_le_bytes());
		file.extend_from_slice(&header);
		file.extend_from_slice(&payload);
		std::fs::write(&input, file).expect("write synthetic SafeTensors");

		let report = import_safetensors_config(&input, &output, config)?;
		assert_eq!(report.output_tensors, 21);
		assert_eq!(report.source_tensors, 23);
		assert_eq!(report.used_source_tensors, 21);
		assert_eq!(report.unused_source_tensors, 2);
		assert_eq!(
			report.output_bytes,
			payload.len() - config.context_length * 8 - 4
		);
		let artifact = std::fs::read(&output).expect("read translated CLIP model");
		assert_eq!(&artifact[..4], b"OAM\0");
		assert!(
			artifact
				.windows(b"text_model.encoder.layers.0.mlp.fc1.weight".len())
				.any(|window| window == b"text_model.encoder.layers.0.mlp.fc1.weight")
		);
		std::fs::remove_dir_all(directory).expect("remove CLIP import test directory");
		Ok(())
	}
}
