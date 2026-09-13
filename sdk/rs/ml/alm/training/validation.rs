//! Held-out ALM tokenizer and prior evaluation.
//!
//! These passes preserve the donor `trainalm` validation boundaries: no
//! optimizer or EMA state is changed, the final partial batch is not wrapped,
//! and only compact scalar/count results cross the device boundary before the
//! tokenizer's motion diagnostics are computed on host data.

use crate::{
	Engine, Error, Matrix, Result, matrix,
	ml::{ValidationResult, loss},
	sdk::data::{HumanMl3dDataset, human_ml3d_evaluate_motion},
};

use super::{
	PriorConditioning, PriorSpecialTokens, build_prior_windows, build_tokenizer_windows,
	gather_prior_batch, gather_tokenizer_batch,
};
use crate::sdk::ml::alm::{AlmPrior, AlmTokenizer};

/// Batching policy for one held-out tokenizer pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenizerValidationConfig {
	/// Frames in each held-out motion window.
	pub sequence_len: usize,
	/// Maximum rows submitted together.
	pub batch_size: usize,
	/// Maximum batches to evaluate; zero consumes the complete split.
	pub max_batches: usize,
}

/// Complete held-out tokenizer diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenizerValidationReport {
	/// Sample-weighted reconstruction Smooth L1.
	pub reconstruction_loss: f64,
	/// Sample-weighted temporal-velocity Smooth L1.
	pub velocity_loss: f64,
	/// Sample-weighted mean per-joint position error in centimeters.
	pub mpjpe_cm: f64,
	/// Sample-weighted contact classification accuracy in `[0, 1]`.
	pub contact_accuracy: f64,
	/// Sample-weighted planted-foot motion in centimeters per frame.
	pub foot_skate_cm_per_frame: f64,
	/// Number of codebook entries selected at least once.
	pub live_codes: usize,
	/// Exponential entropy of the held-out token distribution.
	pub codebook_perplexity: f64,
	/// Number of emitted motion tokens.
	pub tokens: usize,
	/// Number of submitted validation batches.
	pub batches: usize,
	/// Number of evaluated motion windows.
	pub samples: usize,
}

impl TokenizerValidationReport {
	/// Convert to the shared callback result, using reconstruction as the
	/// checkpoint/early-stop authority.
	pub fn callback_result(self) -> ValidationResult {
		ValidationResult {
			loss: self.reconstruction_loss,
			batches: self.batches as u64,
			samples: self.samples as u64,
		}
	}
}

/// Batching policy for one held-out prior pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PriorValidationConfig {
	/// Next-token pairs in each true-boundary row.
	pub window_len: usize,
	/// Maximum rows submitted together.
	pub batch_size: usize,
	/// Maximum batches to evaluate; zero consumes every window.
	pub max_batches: usize,
}

/// Complete held-out prior diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PriorValidationReport {
	/// Valid-token-weighted masked cross entropy.
	pub loss: f64,
	/// Exponential of [`Self::loss`].
	pub perplexity: f64,
	/// Correct predictions across all unmasked targets.
	pub correct_tokens: u64,
	/// Number of unmasked target tokens.
	pub valid_tokens: u64,
	/// Correct-token fraction in `[0, 1]`.
	pub token_accuracy: f64,
	/// Correct predictions at true EOM targets.
	pub correct_eos: u64,
	/// Number of true EOM targets.
	pub eos_tokens: u64,
	/// Correct-EOM fraction in `[0, 1]`, or zero when no EOM was observed.
	pub eos_accuracy: f64,
	/// Number of submitted validation batches.
	pub batches: usize,
}

impl PriorValidationReport {
	/// Convert to the shared callback result.
	pub fn callback_result(self) -> ValidationResult {
		ValidationResult {
			loss: self.loss,
			batches: self.batches as u64,
			samples: self.valid_tokens,
		}
	}
}

/// Evaluate a tokenizer over one standardized HumanML3D validation split.
///
/// # Errors
///
/// Returns an error for incompatible model/dataset geometry, empty validation
/// inventory, invalid batching, allocation, inference, synchronization,
/// readback, denormalization, or HumanML3D diagnostic failure.
pub fn evaluate_tokenizer(
	engine: &Engine,
	tokenizer: &AlmTokenizer,
	dataset: &HumanMl3dDataset,
	config: TokenizerValidationConfig,
) -> Result<TokenizerValidationReport> {
	if dataset.feature_dim() != tokenizer.config().input_dim
		|| config.sequence_len < 2
		|| config.batch_size == 0
		|| !config
			.sequence_len
			.is_multiple_of(tokenizer.downsample_factor())
	{
		return Err(Error::invalid_argument(
			"ALM tokenizer validation geometry is incompatible",
		));
	}
	let windows = build_tokenizer_windows(
		(0..dataset.len()).filter_map(|index| dataset.clip_frames(index)),
		config.sequence_len,
	)?;
	if windows.is_empty() {
		return Err(Error::invalid_argument(
			"ALM tokenizer validation split has no complete window",
		));
	}
	let all_batches = windows.len().div_ceil(config.batch_size);
	let batches = if config.max_batches == 0 {
		all_batches
	} else {
		all_batches.min(config.max_batches)
	};
	let mut weighted_reconstruction = 0.0_f64;
	let mut weighted_velocity = 0.0_f64;
	let mut weighted_mpjpe = 0.0_f64;
	let mut weighted_contact = 0.0_f64;
	let mut weighted_foot_skate = 0.0_f64;
	let mut samples = 0_usize;
	let mut token_histogram = vec![0_u64; tokenizer.config().num_codes];

	for batch_index in 0..batches {
		let begin = batch_index
			.checked_mul(config.batch_size)
			.ok_or_else(|| Error::resource_exhausted("validation cursor exceeds usize"))?;
		let rows = config.batch_size.min(windows.len() - begin);
		let mut target =
			gather_tokenizer_batch(dataset, &windows, begin, rows, config.sequence_len)?;
		let input = Matrix::from_slice(
			engine,
			[rows, config.sequence_len, dataset.feature_dim()],
			&target,
		)?;
		let quantized = tokenizer.quantize(&tokenizer.encode(&input)?)?;
		let reconstruction = tokenizer.decode(&quantized.quantized, rows)?;
		let reconstruction_loss = loss::smooth_l1(&reconstruction, &input)?;
		let velocity_loss =
			super::tokenizer::velocity_smooth_l1(&reconstruction, &input, config.sequence_len)?;
		let reconstruction_loss = scalar_f32(&reconstruction_loss)?;
		let velocity_loss = scalar_f32(&velocity_loss)?;
		let mut predicted = reconstruction.read_f32()?;
		let indices = quantized
			.indices
			.first()
			.ok_or_else(|| Error::internal("ALM tokenizer returned no code level"))?
			.read::<i32>()?;
		for token in indices {
			let index = usize::try_from(token)
				.map_err(|_| Error::data_loss("ALM tokenizer emitted a negative code"))?;
			let count = token_histogram
				.get_mut(index)
				.ok_or_else(|| Error::data_loss("ALM tokenizer emitted an invalid code"))?;
			*count = count
				.checked_add(1)
				.ok_or_else(|| Error::resource_exhausted("validation token count exceeds u64"))?;
		}
		dataset.denormalize(&mut predicted)?;
		dataset.denormalize(&mut target)?;
		let row_values = config
			.sequence_len
			.checked_mul(dataset.feature_dim())
			.ok_or_else(|| Error::resource_exhausted("validation row size exceeds usize"))?;
		for row in 0..rows {
			let range = row * row_values..(row + 1) * row_values;
			let metrics = human_ml3d_evaluate_motion(
				&predicted[range.clone()],
				&target[range],
				config.sequence_len,
				dataset.feature_dim(),
				0.5,
			)?;
			weighted_mpjpe += metrics.mpjpe_cm;
			weighted_contact += metrics.contact_accuracy;
			weighted_foot_skate += metrics.foot_skate_cm_per_frame;
		}
		weighted_reconstruction += f64::from(reconstruction_loss) * rows as f64;
		weighted_velocity += f64::from(velocity_loss) * rows as f64;
		samples = samples
			.checked_add(rows)
			.ok_or_else(|| Error::resource_exhausted("validation sample count exceeds usize"))?;
	}
	let token_count = token_histogram.iter().try_fold(0_u64, |total, count| {
		total
			.checked_add(*count)
			.ok_or_else(|| Error::resource_exhausted("validation token count exceeds u64"))
	})?;
	let entropy = if token_count == 0 {
		0.0
	} else {
		token_histogram
			.iter()
			.filter(|&&count| count > 0)
			.map(|&count| {
				let probability = count as f64 / token_count as f64;
				-probability * probability.ln()
			})
			.sum::<f64>()
	};
	let denominator = samples as f64;
	Ok(TokenizerValidationReport {
		reconstruction_loss: weighted_reconstruction / denominator,
		velocity_loss: weighted_velocity / denominator,
		mpjpe_cm: weighted_mpjpe / denominator,
		contact_accuracy: weighted_contact / denominator,
		foot_skate_cm_per_frame: weighted_foot_skate / denominator,
		live_codes: token_histogram.iter().filter(|&&count| count > 0).count(),
		codebook_perplexity: entropy.exp(),
		tokens: usize::try_from(token_count)
			.map_err(|_| Error::resource_exhausted("validation token count exceeds usize"))?,
		batches,
		samples,
	})
}

/// Evaluate an ALM prior over true-boundary held-out token windows.
///
/// Conditioned validation uses the first cached caption feature for each
/// sequence, matching the donor rather than the epoch-rotating training rule.
///
/// # Errors
///
/// Returns an error for incompatible model, sequence, conditioning, or batch
/// geometry, empty validation data, allocation, inference, synchronization,
/// or compact metric readback failure.
pub fn evaluate_prior(
	engine: &Engine,
	prior: &AlmPrior,
	sequences: &[Vec<i32>],
	conditioning: Option<PriorConditioning<'_>>,
	config: PriorValidationConfig,
) -> Result<PriorValidationReport> {
	validate_prior_evaluation(prior, sequences, conditioning, config)?;
	let windows = build_prior_windows(sequences, config.window_len)?;
	let all_batches = windows.len().div_ceil(config.batch_size);
	let batches = if config.max_batches == 0 {
		all_batches
	} else {
		all_batches.min(config.max_batches)
	};
	let special = PriorSpecialTokens {
		som: prior.config().som_token,
		eom: prior.config().eom_token,
		pad: prior.config().pad_token,
	};
	let mut weighted_loss = 0.0_f64;
	let mut correct_tokens = 0_u64;
	let mut correct_eos = 0_u64;
	let mut eos_tokens = 0_u64;
	let mut valid_tokens = 0_u64;
	for batch_index in 0..batches {
		let begin = batch_index
			.checked_mul(config.batch_size)
			.ok_or_else(|| Error::resource_exhausted("validation cursor exceeds usize"))?;
		let rows = config.batch_size.min(windows.len() - begin);
		let batch =
			gather_prior_batch(sequences, &windows, begin, rows, config.window_len, special)?;
		let element_count = rows
			.checked_mul(config.window_len)
			.ok_or_else(|| Error::resource_exhausted("validation batch size exceeds usize"))?;
		let input = Matrix::from_slice(engine, [rows, config.window_len], &batch.input_ids)?;
		let targets = Matrix::from_slice(engine, [element_count], &batch.target_ids)?;
		let mask = Matrix::from_slice(engine, [element_count], &batch.loss_mask)?;
		let eos_mask_values = batch
			.target_ids
			.iter()
			.zip(&batch.loss_mask)
			.map(|(&target, &enabled)| {
				if enabled != 0.0 && target == special.eom {
					1.0
				} else {
					0.0
				}
			})
			.collect::<Vec<_>>();
		let batch_eos = eos_mask_values
			.iter()
			.filter(|&&value| value != 0.0)
			.count() as u64;
		let eos_mask = Matrix::from_slice(engine, [element_count], &eos_mask_values)?;
		let text = conditioning
			.map(|source| gather_validation_text(engine, source, &windows[begin..begin + rows]))
			.transpose()?;
		let logits = match text.as_ref() {
			Some(text) => prior.forward_conditioned(&input, text)?,
			None => prior.forward(&input)?,
		};
		let logits = matrix::reshape(&logits, [element_count, prior.config().vocab_size])?;
		let batch_loss = loss::masked_cross_entropy(&logits, &targets, &mask, batch.valid_count)?;
		let correct = matrix::masked_categorical_accuracy_count(&logits, &targets, &mask)?;
		let eos_correct = matrix::masked_categorical_accuracy_count(&logits, &targets, &eos_mask)?;
		weighted_loss += f64::from(scalar_f32(&batch_loss)?) * batch.valid_count as f64;
		correct_tokens = correct_tokens
			.checked_add(u64::from(scalar_u32(&correct)?))
			.ok_or_else(|| Error::resource_exhausted("validation correct count exceeds u64"))?;
		correct_eos = correct_eos
			.checked_add(u64::from(scalar_u32(&eos_correct)?))
			.ok_or_else(|| Error::resource_exhausted("validation EOM count exceeds u64"))?;
		eos_tokens = eos_tokens
			.checked_add(batch_eos)
			.ok_or_else(|| Error::resource_exhausted("validation EOM count exceeds u64"))?;
		valid_tokens = valid_tokens
			.checked_add(batch.valid_count as u64)
			.ok_or_else(|| Error::resource_exhausted("validation token count exceeds u64"))?;
	}
	let loss = weighted_loss / valid_tokens as f64;
	Ok(PriorValidationReport {
		loss,
		perplexity: loss.exp(),
		correct_tokens,
		valid_tokens,
		token_accuracy: correct_tokens as f64 / valid_tokens as f64,
		correct_eos,
		eos_tokens,
		eos_accuracy: if eos_tokens == 0 {
			0.0
		} else {
			correct_eos as f64 / eos_tokens as f64
		},
		batches,
	})
}

fn validate_prior_evaluation(
	prior: &AlmPrior,
	sequences: &[Vec<i32>],
	conditioning: Option<PriorConditioning<'_>>,
	config: PriorValidationConfig,
) -> Result<()> {
	if sequences.is_empty() || config.batch_size == 0 || config.window_len == 0 {
		return Err(Error::invalid_argument(
			"ALM prior validation requires sequences and positive batch geometry",
		));
	}
	let model_sequence = config
		.window_len
		.checked_add(usize::from(conditioning.is_some()))
		.ok_or_else(|| Error::resource_exhausted("validation sequence length exceeds usize"))?;
	if model_sequence > prior.config().max_sequence_length {
		return Err(Error::invalid_argument(
			"ALM prior validation sequence exceeds its position table",
		));
	}
	match conditioning {
		Some(source)
			if prior.config().text_feature_dim == source.feature_dim
				&& source.feature_dim > 0
				&& source.by_sequence.len() == sequences.len()
				&& source.by_sequence.iter().all(|features| {
					features.len() >= source.feature_dim
						&& features.len().is_multiple_of(source.feature_dim)
						&& features.iter().all(|value| value.is_finite())
				}) => {}
		Some(_) => {
			return Err(Error::invalid_argument(
				"ALM prior validation conditioning is incompatible",
			));
		}
		None if prior.config().text_feature_dim != 0 => {
			return Err(Error::invalid_argument(
				"conditioned ALM prior validation requires text features",
			));
		}
		None => {}
	}
	Ok(())
}

fn gather_validation_text(
	engine: &Engine,
	source: PriorConditioning<'_>,
	windows: &[super::PriorWindow],
) -> Result<Matrix> {
	let mut values = Vec::with_capacity(windows.len() * source.feature_dim);
	for window in windows {
		values.extend_from_slice(&source.by_sequence[window.sequence][..source.feature_dim]);
	}
	Matrix::from_slice(engine, [windows.len(), source.feature_dim], &values)
}

fn scalar_f32(value: &Matrix) -> Result<f32> {
	value
		.read_f32()?
		.into_iter()
		.next()
		.ok_or_else(|| Error::internal("scalar FP32 readback was empty"))
}

fn scalar_u32(value: &Matrix) -> Result<u32> {
	value
		.read::<u32>()?
		.into_iter()
		.next()
		.ok_or_else(|| Error::internal("scalar U32 readback was empty"))
}
