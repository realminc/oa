//! End-to-end HumanML3D ALM training orchestration.

use std::rc::Rc;

use crate::{Engine, Error, Matrix, Result, sdk::data::HumanMl3dDataset};

use super::{
	PriorConditioning, PriorTrainingConfig, PriorTrainingReport, PriorValidation,
	PriorValidationConfig, TokenizerTrainingConfig, TokenizerTrainingReport, TokenizerValidation,
	TokenizerValidationConfig, train_prior_with_validation, train_tokenizer_with_validation,
};
use crate::sdk::ml::alm::{Alm, AlmPrior, AlmTokenizer};

/// Stage policies for one complete tokenizer-then-prior ALM run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AlmTrainingConfig {
	/// Temporal VQ-VAE stage policy.
	pub tokenizer: TokenizerTrainingConfig,
	/// Autoregressive prior stage policy.
	pub prior: PriorTrainingConfig,
	/// Caption selection seed for cached text features.
	pub text_seed: u64,
}

/// Evidence and resulting product model from one complete ALM run.
pub struct AlmTrainingReport {
	/// Trained composed model. Save this with [`Alm::save_bundle`].
	pub model: Alm,
	/// Tokenizer-stage timing and loss evidence.
	pub tokenizer: TokenizerTrainingReport,
	/// Prior-stage timing and loss evidence.
	pub prior: PriorTrainingReport,
	/// Number of motion tokens used to construct prior windows.
	pub corpus_tokens: usize,
}

/// Borrowed held-out dataset and per-stage evaluation policy.
#[derive(Clone, Copy)]
pub struct AlmValidation<'data> {
	/// HumanML3D validation split.
	pub dataset: &'data HumanMl3dDataset,
	/// Tokenizer validation geometry and batch limit.
	pub tokenizer: TokenizerValidationConfig,
	/// Prior validation geometry and batch limit.
	pub prior: PriorValidationConfig,
}

/// Tokenize every eligible normalized dataset clip with one trained tokenizer.
///
/// Clips shorter than one downsample interval are rejected because silently
/// dropping a dataset row would break caption/conditioning alignment.
///
/// # Errors
///
/// Returns an error for feature-width mismatch, a short/malformed clip,
/// allocation, tokenizer execution, synchronization, or readback failure.
pub fn tokenize_corpus(
	engine: &Engine,
	tokenizer: &AlmTokenizer,
	dataset: &HumanMl3dDataset,
) -> Result<Vec<Vec<i32>>> {
	if dataset.feature_dim() != tokenizer.config().input_dim {
		return Err(Error::invalid_argument(
			"HumanML3D feature width does not match the ALM tokenizer",
		));
	}
	let mut sequences = Vec::with_capacity(dataset.len());
	for index in 0..dataset.len() {
		let frames = dataset
			.clip_frames(index)
			.ok_or_else(|| Error::internal("HumanML3D dataset lost a clip frame count"))?;
		let features = dataset
			.clip_data(index)
			.ok_or_else(|| Error::internal("HumanML3D dataset lost a clip value buffer"))?;
		if frames < tokenizer.downsample_factor() {
			return Err(Error::invalid_argument(format!(
				"HumanML3D clip {index} is shorter than one tokenizer downsample interval"
			)));
		}
		let input =
			Matrix::from_slice(engine, [1, frames, tokenizer.config().input_dim], features)?;
		let mut levels = tokenizer.tokenize(&input)?;
		let tokens = levels
			.pop()
			.ok_or_else(|| Error::internal("ALM tokenizer returned no code level"))?;
		if !levels.is_empty() {
			return Err(Error::failed_precondition(
				"ALM workflow currently requires the donor single-level tokenizer",
			));
		}
		sequences.push(tokens.read::<i32>()?);
	}
	Ok(sequences)
}

/// Train tokenizer and prior stages over one HumanML3D dataset, then compose
/// the independently trainable children into one product-level ALM model.
///
/// A prior with `text_feature_dim == 0` is unconditional. A conditioned prior
/// requires the dataset's validated `oa_clip_text_v1` rows and preserves the
/// manifest model identity in the resulting bundle.
///
/// # Errors
///
/// Returns an error from either stage, token-corpus construction, cached text
/// contract validation, product ownership composition, or count overflow.
pub fn train_alm(
	engine: &Engine,
	dataset: &HumanMl3dDataset,
	tokenizer: Rc<AlmTokenizer>,
	prior: Rc<AlmPrior>,
	config: AlmTrainingConfig,
) -> Result<AlmTrainingReport> {
	train_alm_impl(engine, dataset, tokenizer, prior, None, config)
}

/// Train both ALM stages while evaluating a held-out HumanML3D split at every
/// completed stage epoch.
///
/// Tokenizer validation observes motion reconstruction directly. After that
/// stage is complete, the same frozen tokenizer creates both train and held-out
/// token corpora so prior validation cannot drift to a different vocabulary.
///
/// # Errors
///
/// Returns the same errors as [`train_alm`], plus held-out split, tokenization,
/// conditioning, evaluation, or callback failures.
pub fn train_alm_with_validation(
	engine: &Engine,
	dataset: &HumanMl3dDataset,
	validation: AlmValidation<'_>,
	tokenizer: Rc<AlmTokenizer>,
	prior: Rc<AlmPrior>,
	config: AlmTrainingConfig,
) -> Result<AlmTrainingReport> {
	train_alm_impl(engine, dataset, tokenizer, prior, Some(validation), config)
}

fn train_alm_impl(
	engine: &Engine,
	dataset: &HumanMl3dDataset,
	tokenizer: Rc<AlmTokenizer>,
	prior: Rc<AlmPrior>,
	validation: Option<AlmValidation<'_>>,
	config: AlmTrainingConfig,
) -> Result<AlmTrainingReport> {
	if dataset.feature_dim() != tokenizer.config().input_dim {
		return Err(Error::invalid_argument(
			"HumanML3D feature width does not match the ALM tokenizer",
		));
	}
	if tokenizer.config().num_codes != prior.config().num_codes {
		return Err(Error::invalid_argument(
			"ALM tokenizer and prior code counts do not match",
		));
	}
	let clips = (0..dataset.len())
		.map(|index| {
			dataset.clip_data(index).ok_or_else(|| {
				Error::internal("HumanML3D dataset lost a clip inside its advertised length")
			})
		})
		.collect::<Result<Vec<_>>>()?;
	let tokenizer_report = train_tokenizer_with_validation(
		engine,
		&tokenizer,
		&clips,
		validation.map(|specification| TokenizerValidation {
			dataset: specification.dataset,
			config: specification.tokenizer,
		}),
		config.tokenizer,
	)?;
	let sequences = tokenize_corpus(engine, &tokenizer, dataset)?;
	let validation_sequences = validation
		.map(|specification| tokenize_corpus(engine, &tokenizer, specification.dataset))
		.transpose()?;
	let corpus_tokens = sequences.iter().try_fold(0_usize, |total, sequence| {
		total
			.checked_add(sequence.len())
			.ok_or_else(|| Error::resource_exhausted("ALM token corpus size overflows usize"))
	})?;

	let text_features = if prior.config().text_feature_dim == 0 {
		None
	} else {
		Some(cached_conditioning_rows(dataset, &prior)?)
	};
	let validation_text_features = match validation {
		Some(specification) if prior.config().text_feature_dim != 0 => {
			if specification.dataset.text_feature_model() != dataset.text_feature_model() {
				return Err(Error::invalid_argument(
					"training and validation text encoder identities differ",
				));
			}
			Some(cached_conditioning_rows(specification.dataset, &prior)?)
		}
		_ => None,
	};
	let conditioning = text_features.as_deref().map(|rows| PriorConditioning {
		by_sequence: rows,
		feature_dim: prior.config().text_feature_dim,
		seed: config.text_seed,
	});
	let prior_validation = match (validation, validation_sequences.as_deref()) {
		(Some(specification), Some(validation_sequences)) => Some(PriorValidation {
			sequences: validation_sequences,
			conditioning: validation_text_features
				.as_deref()
				.map(|rows| PriorConditioning {
					by_sequence: rows,
					feature_dim: prior.config().text_feature_dim,
					seed: config.text_seed,
				}),
			config: specification.prior,
		}),
		_ => None,
	};
	let prior_report = train_prior_with_validation(
		engine,
		&prior,
		&sequences,
		conditioning,
		prior_validation,
		config.prior,
	)?;
	let model = if prior.config().text_feature_dim == 0 {
		Alm::from_parts(tokenizer, prior)?
	} else {
		Alm::from_external_text_parts(
			tokenizer,
			prior,
			dataset
				.text_feature_model()
				.ok_or_else(|| Error::internal("validated text model identity was lost"))?,
		)?
	};
	Ok(AlmTrainingReport {
		model,
		tokenizer: tokenizer_report,
		prior: prior_report,
		corpus_tokens,
	})
}

fn cached_conditioning_rows(dataset: &HumanMl3dDataset, prior: &AlmPrior) -> Result<Vec<Vec<f32>>> {
	if dataset.text_feature_dim() != prior.config().text_feature_dim
		|| dataset.text_feature_format() != Some("oa_clip_text_v1")
		|| dataset.text_feature_model().is_none_or(str::is_empty)
	{
		return Err(Error::invalid_argument(
			"HumanML3D cached text contract does not match the conditioned ALM prior",
		));
	}
	(0..dataset.len())
		.map(|index| {
			let features = dataset
				.clip_text_features(index)
				.ok_or_else(|| Error::internal("HumanML3D dataset lost cached text features"))?;
			let captions = dataset
				.clip_captions(index)
				.ok_or_else(|| Error::internal("HumanML3D dataset lost clip captions"))?;
			let expected = captions
				.len()
				.checked_mul(prior.config().text_feature_dim)
				.ok_or_else(|| Error::resource_exhausted("ALM caption feature size overflows"))?;
			if captions.is_empty() || features.len() != expected {
				return Err(Error::invalid_argument(format!(
					"HumanML3D clip {index} lacks one cached feature per caption"
				)));
			}
			Ok(features.to_vec())
		})
		.collect()
}
