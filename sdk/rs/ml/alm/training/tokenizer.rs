//! ALM temporal VQ-VAE tokenizer training lifecycle.

use std::{cell::Cell, rc::Rc};

use crate::{
	Engine, Error, Matrix, Result, matrix,
	ml::{
		AdamW, Checkpoint, CheckpointManager, CheckpointManagerConfig, GradientTape, ItTraining,
		ItTrainingConfig, LinearWarmupCosineScheduler, LossAggregation, LossMetric, LrScheduler,
		Module, ProgressBar, TrainingSummary, Validation, loss,
	},
	sdk::data::HumanMl3dDataset,
};

use super::{
	StageCheckpointConfig, TokenizerValidationConfig, TokenizerValidationReport,
	build_tokenizer_windows, evaluate_tokenizer, gather_tokenizer_clip_batch,
};
use crate::sdk::ml::alm::AlmTokenizer;

/// Complete optimizer and iteration policy for [`train_tokenizer`].
#[derive(Clone, Debug, PartialEq)]
pub struct TokenizerTrainingConfig {
	/// Number of complete passes through the window inventory.
	pub epochs: u64,
	/// Fixed motion windows per optimizer step.
	pub batch_size: usize,
	/// Frames per motion window.
	pub sequence_len: usize,
	/// Peak AdamW learning rate.
	pub learning_rate: f32,
	/// Final cosine learning rate.
	pub minimum_learning_rate: f32,
	/// Number of optimizer steps in the linear warmup.
	pub warmup_steps: u64,
	/// Decoupled AdamW weight decay.
	pub weight_decay: f32,
	/// Whether to initialize the EMA codebook from the first batch when it has
	/// at least one latent row per code.
	pub seed_codebook: bool,
	/// Collect exact Vulkan timestamp evidence for every step.
	pub enable_gpu_timing: bool,
	/// Print the standard OA progress bar and final training summary.
	pub show_progress: bool,
	/// Optional native model/optimizer/progress checkpoint policy.
	pub checkpoint: Option<StageCheckpointConfig>,
}

impl Default for TokenizerTrainingConfig {
	fn default() -> Self {
		Self {
			epochs: 100,
			batch_size: 32,
			sequence_len: 64,
			learning_rate: 2.0e-4,
			minimum_learning_rate: 1.0e-6,
			warmup_steps: 100,
			weight_decay: 0.0,
			seed_codebook: true,
			enable_gpu_timing: true,
			show_progress: true,
			checkpoint: None,
		}
	}
}

/// Final evidence from one ALM tokenizer training run.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenizerTrainingReport {
	/// Completed optimizer steps.
	pub steps: u64,
	/// Number of logical steps in each epoch.
	pub steps_per_epoch: u64,
	/// First completed reconstruction loss.
	pub initial_reconstruction_loss: f32,
	/// Last completed reconstruction loss.
	pub final_reconstruction_loss: f32,
	/// Mean completed reconstruction loss.
	pub mean_reconstruction_loss: f64,
	/// Total measured wall time in seconds.
	pub wall_seconds: f64,
	/// Mean measured device milliseconds per step, or zero when disabled.
	pub gpu_mean_ms: f64,
	/// Median measured device milliseconds per step, or zero when disabled.
	pub gpu_median_ms: f64,
	/// 95th-percentile device milliseconds per step, or zero when disabled.
	pub gpu_p95_ms: f64,
	/// Quantizer EMA transition count after the run.
	pub ema_steps: u32,
	/// Completed step restored before this process began, or zero for a fresh run.
	pub resumed_from_step: u64,
	/// Most recent held-out result, when validation was configured.
	pub validation: Option<TokenizerValidationReport>,
}

/// Borrowed held-out split evaluated at every completed tokenizer epoch.
#[derive(Clone, Copy)]
pub struct TokenizerValidation<'data> {
	/// Standardized HumanML3D validation split.
	pub dataset: &'data HumanMl3dDataset,
	/// Validation batching policy.
	pub config: TokenizerValidationConfig,
}

/// Train the temporal ALM VQ-VAE over normalized row-major motion clips.
///
/// The optimized objective combines reconstruction Smooth L1, velocity Smooth
/// L1, and the configured VQ commitment loss.
///
/// The iterator's primary loss remains
/// reconstruction, matching the donor `trainalm` progress/checkpoint contract.
/// Codebook updates use EMA and are not AdamW parameters.
///
/// # Errors
///
/// Returns an error for invalid configuration/corpus geometry, allocation,
/// model, autograd, EMA, optimizer, runtime, timing, metric, or callback
/// failure. A non-finite completed reconstruction loss is rejected.
pub fn train_tokenizer<T: AsRef<[f32]>>(
	engine: &Engine,
	tokenizer: &AlmTokenizer,
	clips: &[T],
	config: TokenizerTrainingConfig,
) -> Result<TokenizerTrainingReport> {
	train_tokenizer_with_validation(engine, tokenizer, clips, None, config)
}

/// Train the tokenizer and evaluate one held-out split at every epoch end.
///
/// This is the same training implementation as [`train_tokenizer`]; the extra
/// argument only attaches OA's shared [`Validation`] callback.
///
/// # Errors
///
/// Returns the same errors as [`train_tokenizer`], plus held-out evaluation or
/// callback failures.
pub fn train_tokenizer_with_validation<T: AsRef<[f32]>>(
	engine: &Engine,
	tokenizer: &AlmTokenizer,
	clips: &[T],
	validation: Option<TokenizerValidation<'_>>,
	config: TokenizerTrainingConfig,
) -> Result<TokenizerTrainingReport> {
	validate_training_config(tokenizer, clips, &config)?;
	let frame_counts = clips
		.iter()
		.map(|clip| clip.as_ref().len() / tokenizer.config().input_dim);
	let windows = build_tokenizer_windows(frame_counts, config.sequence_len)?;
	if windows.is_empty() {
		return Err(Error::invalid_argument(
			"ALM tokenizer corpus has no clip long enough for sequence_len",
		));
	}
	let steps_per_epoch = u64::try_from(windows.len().div_ceil(config.batch_size))
		.map_err(|_| Error::resource_exhausted("ALM tokenizer steps per epoch exceed u64"))?;
	let total_steps = config
		.epochs
		.checked_mul(steps_per_epoch)
		.ok_or_else(|| Error::resource_exhausted("ALM tokenizer total step count exceeds u64"))?;
	let batch_values = config
		.batch_size
		.checked_mul(config.sequence_len)
		.and_then(|count| count.checked_mul(tokenizer.config().input_dim))
		.ok_or_else(|| Error::resource_exhausted("ALM tokenizer batch size overflows usize"))?;

	if config.seed_codebook {
		let latent_rows = config
			.batch_size
			.checked_mul(config.sequence_len / tokenizer.downsample_factor())
			.ok_or_else(|| Error::resource_exhausted("ALM tokenizer latent row count overflows"))?;
		if latent_rows >= tokenizer.config().num_codes {
			let values = gather_tokenizer_clip_batch(
				clips,
				tokenizer.config().input_dim,
				&windows,
				0,
				config.batch_size,
				config.sequence_len,
			)?;
			let input = Matrix::from_slice(
				engine,
				[
					config.batch_size,
					config.sequence_len,
					tokenizer.config().input_dim,
				],
				&values,
			)?;
			tokenizer.seed(&tokenizer.encode(&input)?)?;
		}
	}

	let mut optimizer = AdamW::with_hyperparameters(
		tokenizer.all_parameters()?,
		config.learning_rate,
		0.9,
		0.99,
		1.0e-8,
		config.weight_decay,
	)?;
	let mut checkpoint_manager = config
		.checkpoint
		.as_ref()
		.map(|checkpoint| {
			CheckpointManager::new(
				engine,
				CheckpointManagerConfig {
					directory: checkpoint.directory.clone(),
					model_name: checkpoint.model_name.clone(),
					context: checkpoint.context.clone(),
					max_keep: checkpoint.max_keep,
					save_best: true,
					metric_name: if validation.is_some() {
						"val_loss".to_owned()
					} else {
						"reconstruction".to_owned()
					},
					lower_is_better: true,
				},
			)
		})
		.transpose()?;
	let resumed_from_step = match (config.checkpoint.as_ref(), checkpoint_manager.as_mut()) {
		(Some(checkpoint), Some(manager)) if checkpoint.resume => {
			manager.resume_latest_into(tokenizer, &mut optimizer)?
		}
		_ => 0,
	};
	let schedule = LinearWarmupCosineScheduler::new(
		config.warmup_steps,
		total_steps,
		config.learning_rate,
		config.minimum_learning_rate,
	)?;
	optimizer.set_learning_rate(schedule.learning_rate(resumed_from_step.saturating_add(1)))?;
	let mut schedule_callback = crate::ml::LearningRateScheduler::new(&schedule);
	let mut loss_metric = LossMetric::new("reconstruction", LossAggregation::Mean);
	let mut progress = ProgressBar::default();
	let mut summary = TrainingSummary::new(true);
	let latest_validation = Rc::new(Cell::new(None));
	let mut validation_callback = validation
		.map(|specification| {
			let latest_validation = Rc::clone(&latest_validation);
			Validation::new(
				move |_| {
					let report = evaluate_tokenizer(
						engine,
						tokenizer,
						specification.dataset,
						specification.config,
					)?;
					latest_validation.set(Some(report));
					Ok(report.callback_result())
				},
				"val_loss",
				0,
			)
		})
		.transpose()?;
	let validation_metric = validation_callback.as_ref().map(Validation::metric);
	let mut checkpoint_callback = match (config.checkpoint.as_ref(), checkpoint_manager.as_mut()) {
		(Some(checkpoint), Some(manager)) => {
			let mut callback = Checkpoint::new(manager, tokenizer);
			callback.set_save_every(checkpoint.save_every);
			callback.set_restore_best(checkpoint.restore_best);
			callback.set_verbose(checkpoint.verbose);
			if let Some(metric) = validation_metric {
				callback.set_validation_metric(metric);
			}
			Some(callback)
		}
		_ => None,
	};
	let mut training = ItTraining::new_eager_checkpointable(
		engine,
		&mut optimizer,
		ItTrainingConfig {
			initial_step: resumed_from_step,
			total_steps,
			steps_per_epoch,
			batch_size: config.batch_size as u64,
			sequence_length: config.sequence_len as u64,
			sequence_unit: "frame".into(),
			timer_name: "alm_tokenizer_step".into(),
			enable_gpu_timing: config.enable_gpu_timing,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	if let Some(callback) = validation_callback.as_mut() {
		training.add_callback(callback);
	}
	if let Some(callback) = checkpoint_callback.as_mut() {
		training.add_callback(callback);
	}
	training.add_callback(&mut schedule_callback);
	if config.show_progress {
		training.add_callback(&mut progress);
		training.add_callback(&mut summary);
	}

	let mut completed_steps = resumed_from_step;
	let mut initial_loss = None;
	while training.begin_step()? {
		let completed_usize = usize::try_from(completed_steps)
			.map_err(|_| Error::resource_exhausted("ALM tokenizer step index exceeds usize"))?;
		let cursor = completed_usize
			.checked_mul(config.batch_size)
			.ok_or_else(|| Error::resource_exhausted("ALM tokenizer cursor overflows usize"))?
			% windows.len();
		let values = gather_tokenizer_clip_batch(
			clips,
			tokenizer.config().input_dim,
			&windows,
			cursor,
			config.batch_size,
			config.sequence_len,
		)?;
		debug_assert_eq!(values.len(), batch_values);
		let input = Matrix::from_slice(
			engine,
			[
				config.batch_size,
				config.sequence_len,
				tokenizer.config().input_dim,
			],
			&values,
		)?;
		training.seal_replay_inputs()?;
		training.zero_grad();
		let tape = GradientTape::new();
		let latent = tokenizer.encode(&input)?;
		let quantized = tokenizer.quantize(&latent)?;
		let reconstruction = tokenizer.decode(&quantized.quantized, config.batch_size)?;
		let reconstruction_loss = loss::smooth_l1(&reconstruction, &input)?;
		let velocity_loss = velocity_smooth_l1(&reconstruction, &input, config.sequence_len)?;
		let objective = matrix::add(
			&matrix::add(&reconstruction_loss, &velocity_loss)?,
			&quantized.commitment_loss,
		)?;
		tape.backward(&objective)?;
		tokenizer.ema_update(&quantized)?;
		training.complete_step(&reconstruction_loss)?;
		completed_steps += 1;
		let latest = training
			.snapshot()
			.last_loss()
			.ok_or_else(|| Error::internal("ALM tokenizer step completed without a loss"))?;
		if !latest.is_finite() {
			return Err(Error::data_loss(format!(
				"ALM tokenizer loss became non-finite at step {completed_steps}"
			)));
		}
		initial_loss.get_or_insert(latest);
	}
	let snapshot = training.finish()?;
	let gpu = snapshot.gpu_timing_stats();
	Ok(TokenizerTrainingReport {
		steps: snapshot.step_count(),
		steps_per_epoch,
		initial_reconstruction_loss: initial_loss.unwrap_or_default(),
		final_reconstruction_loss: snapshot.last_loss().unwrap_or_default(),
		mean_reconstruction_loss: snapshot.training_mean_loss(),
		wall_seconds: snapshot.elapsed().as_secs_f64(),
		gpu_mean_ms: gpu.mean_ms,
		gpu_median_ms: gpu.median_ms,
		gpu_p95_ms: gpu.p95_ms,
		ema_steps: tokenizer.rvq().level(0).map_or(0, |level| level.ema_step()),
		resumed_from_step,
		validation: latest_validation.get(),
	})
}

pub(super) fn velocity_smooth_l1(
	prediction: &Matrix,
	target: &Matrix,
	sequence_len: usize,
) -> Result<Matrix> {
	let end = i64::try_from(sequence_len)
		.map_err(|_| Error::resource_exhausted("ALM tokenizer sequence length exceeds i64"))?;
	let prediction_left = matrix::slice(prediction, 1, 0, end - 1)?;
	let prediction_right = matrix::slice(prediction, 1, 1, end)?;
	let target_left = matrix::slice(target, 1, 0, end - 1)?;
	let target_right = matrix::slice(target, 1, 1, end)?;
	loss::smooth_l1(
		&matrix::sub(&prediction_right, &prediction_left)?,
		&matrix::sub(&target_right, &target_left)?,
	)
}

fn validate_training_config<T: AsRef<[f32]>>(
	tokenizer: &AlmTokenizer,
	clips: &[T],
	config: &TokenizerTrainingConfig,
) -> Result<()> {
	if config.epochs == 0 || config.batch_size == 0 || config.sequence_len < 2 {
		return Err(Error::invalid_argument(
			"ALM tokenizer epochs/batch must be positive and sequence_len must be at least two",
		));
	}
	if !config
		.sequence_len
		.is_multiple_of(tokenizer.downsample_factor())
	{
		return Err(Error::invalid_argument(
			"ALM tokenizer sequence_len must be divisible by its downsample factor",
		));
	}
	if !config.learning_rate.is_finite()
		|| config.learning_rate <= 0.0
		|| !config.minimum_learning_rate.is_finite()
		|| config.minimum_learning_rate < 0.0
		|| config.minimum_learning_rate > config.learning_rate
		|| !config.weight_decay.is_finite()
		|| config.weight_decay < 0.0
	{
		return Err(Error::invalid_argument(
			"ALM tokenizer learning rates and weight decay are invalid",
		));
	}
	let feature_dim = tokenizer.config().input_dim;
	if clips.is_empty()
		|| clips.iter().any(|clip| {
			clip.as_ref().is_empty()
				|| !clip.as_ref().len().is_multiple_of(feature_dim)
				|| clip.as_ref().iter().any(|value| !value.is_finite())
		}) {
		return Err(Error::invalid_argument(
			"ALM tokenizer training requires finite complete motion clips",
		));
	}
	let provisional_steps = u64::try_from(clips.len().div_ceil(config.batch_size))
		.map_err(|_| Error::resource_exhausted("ALM tokenizer provisional steps exceed u64"))?
		.max(1);
	let provisional_total = config
		.epochs
		.checked_mul(provisional_steps)
		.ok_or_else(|| Error::resource_exhausted("ALM tokenizer provisional total steps exceed u64"))?;
	if config.warmup_steps >= provisional_total {
		return Err(Error::invalid_argument(
			"ALM tokenizer warmup must be shorter than the training run",
		));
	}
	Ok(())
}
