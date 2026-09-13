//! ALM prior training lifecycle.

use crate::{
	Engine, Error, Matrix, Result, matrix,
	ml::{
		AdamW, GradientTape, ItTraining, ItTrainingConfig, LinearWarmupCosineScheduler,
		LossAggregation, LossMetric, LrScheduler, Module, ProgressBar, TrainingSummary, loss,
	},
};

use super::{PriorSpecialTokens, build_prior_windows, gather_prior_batch};
use crate::sdk::ml::alm::AlmPrior;

/// Frozen text features associated with each motion-token sequence.
///
/// Each entry of `by_sequence` is a row-major collection of one or more
/// `feature_dim` caption embeddings. Training selects captions with the donor's
/// deterministic `(seed + epoch + sequence) % caption_count` rule.
#[derive(Clone, Copy, Debug)]
pub struct PriorConditioning<'data> {
	/// Flattened caption-feature rows for every token sequence.
	pub by_sequence: &'data [Vec<f32>],
	/// Scalar width of one caption feature.
	pub feature_dim: usize,
	/// Base caption-selection seed.
	pub seed: u64,
}

/// Complete optimizer and iteration policy for [`train_prior`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PriorTrainingConfig {
	/// Number of complete passes through the true-boundary window inventory.
	pub epochs: u64,
	/// Fixed rows per optimizer step. The final logical batch wraps cyclically,
	/// matching the donor application.
	pub batch_size: usize,
	/// Number of next-token pairs in each row.
	pub window_len: usize,
	/// Peak AdamW learning rate.
	pub learning_rate: f32,
	/// Final cosine learning rate.
	pub minimum_learning_rate: f32,
	/// Number of optimizer steps in the linear warmup.
	pub warmup_steps: u64,
	/// Decoupled AdamW weight decay.
	pub weight_decay: f32,
	/// Collect exact Vulkan timestamp evidence for every step.
	pub enable_gpu_timing: bool,
	/// Print the standard OA progress bar and final training summary.
	pub show_progress: bool,
}

impl Default for PriorTrainingConfig {
	fn default() -> Self {
		Self {
			epochs: 100,
			batch_size: 32,
			window_len: 129,
			learning_rate: 1.0e-4,
			minimum_learning_rate: 1.0e-6,
			warmup_steps: 100,
			weight_decay: 0.01,
			enable_gpu_timing: true,
			show_progress: true,
		}
	}
}

/// Final evidence from one ALM prior training run.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorTrainingReport {
	/// Completed optimizer steps.
	pub steps: u64,
	/// Number of logical steps in each epoch.
	pub steps_per_epoch: u64,
	/// First completed loss.
	pub initial_loss: f32,
	/// Last completed loss.
	pub final_loss: f32,
	/// Arithmetic mean of all completed step losses.
	pub mean_loss: f64,
	/// Total measured wall time in seconds.
	pub wall_seconds: f64,
	/// Mean measured device milliseconds per step, or zero when disabled.
	pub gpu_mean_ms: f64,
	/// Median measured device milliseconds per step, or zero when disabled.
	pub gpu_median_ms: f64,
	/// 95th-percentile device milliseconds per step, or zero when disabled.
	pub gpu_p95_ms: f64,
}

/// Train an ALM prior with true-boundary token windows and the shared OA
/// iterator/callback lifecycle.
///
/// This path deliberately uses the eager iterator mode because a padded batch's
/// `valid_count` is a host scalar that can change between steps. Replaying a
/// fixed captured push constant would silently apply the wrong normalization.
/// Once dynamic executable-graph scalars are admitted, this restriction can be
/// removed without changing the public training contract.
///
/// # Errors
///
/// Returns an error for invalid configuration, token or conditioning data,
/// model/configuration mismatch, allocation, autograd, optimizer, runtime,
/// timing, metric, or callback failure. A non-finite completed loss is rejected.
pub fn train_prior(
	engine: &Engine,
	prior: &AlmPrior,
	sequences: &[Vec<i32>],
	conditioning: Option<PriorConditioning<'_>>,
	config: PriorTrainingConfig,
) -> Result<PriorTrainingReport> {
	validate_training_config(prior, sequences, conditioning, config)?;
	let windows = build_prior_windows(sequences, config.window_len)?;
	if windows.is_empty() {
		return Err(Error::invalid_argument(
			"ALM prior training requires at least one token sequence",
		));
	}
	let steps_per_epoch = u64::try_from(windows.len().div_ceil(config.batch_size))
		.map_err(|_| Error::resource_exhausted("ALM prior steps per epoch exceed u64"))?;
	let total_steps = config
		.epochs
		.checked_mul(steps_per_epoch)
		.ok_or_else(|| Error::resource_exhausted("ALM prior total step count exceeds u64"))?;
	let special = PriorSpecialTokens {
		som: prior.config().som_token,
		eom: prior.config().eom_token,
		pad: prior.config().pad_token,
	};
	let batch_tokens = config
		.batch_size
		.checked_mul(config.window_len)
		.ok_or_else(|| Error::resource_exhausted("ALM prior batch token count overflows usize"))?;
	let mut optimizer = AdamW::with_hyperparameters(
		prior.all_parameters()?,
		config.learning_rate,
		0.9,
		0.99,
		1.0e-8,
		config.weight_decay,
	)?;
	let schedule = LinearWarmupCosineScheduler::new(
		config.warmup_steps,
		total_steps,
		config.learning_rate,
		config.minimum_learning_rate,
	)?;
	optimizer.set_learning_rate(schedule.learning_rate(1))?;
	let mut schedule_callback = crate::ml::LearningRateScheduler::new(&schedule);
	let mut loss_metric = LossMetric::new("cross_entropy", LossAggregation::Mean);
	let mut progress = ProgressBar::default();
	let mut summary = TrainingSummary::new(true);
	let mut training = ItTraining::new_eager_checkpointable(
		engine,
		&mut optimizer,
		ItTrainingConfig {
			total_steps,
			steps_per_epoch,
			batch_size: config.batch_size as u64,
			sequence_length: config.window_len as u64,
			sequence_unit: "token".into(),
			timer_name: "alm_prior_step".into(),
			enable_gpu_timing: config.enable_gpu_timing,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	training.add_callback(&mut schedule_callback);
	if config.show_progress {
		training.add_callback(&mut progress);
		training.add_callback(&mut summary);
	}

	let mut completed_steps = 0_u64;
	let mut initial_loss = None;
	while training.begin_step()? {
		let completed_usize = usize::try_from(completed_steps)
			.map_err(|_| Error::resource_exhausted("ALM prior step index exceeds usize"))?;
		let cursor = completed_usize
			.checked_mul(config.batch_size)
			.ok_or_else(|| Error::resource_exhausted("ALM prior batch cursor overflows usize"))?
			% windows.len();
		let batch = gather_prior_batch(
			sequences,
			&windows,
			cursor,
			config.batch_size,
			config.window_len,
			special,
		)?;
		let input_ids = Matrix::from_slice(
			engine,
			[config.batch_size, config.window_len],
			&batch.input_ids,
		)?;
		let target_ids = Matrix::from_slice(engine, [batch_tokens], &batch.target_ids)?;
		let loss_mask = Matrix::from_slice(engine, [batch_tokens], &batch.loss_mask)?;
		let text_features = conditioning
			.map(|source| {
				gather_text_features(
					engine,
					source,
					&windows,
					cursor,
					config.batch_size,
					completed_steps / steps_per_epoch + 1,
				)
			})
			.transpose()?;
		training.seal_replay_inputs()?;
		training.zero_grad();
		let tape = GradientTape::new();
		let logits = match text_features.as_ref() {
			Some(text) => prior.forward_conditioned(&input_ids, text)?,
			None => prior.forward(&input_ids)?,
		};
		let logits = matrix::reshape(&logits, [batch_tokens, prior.config().vocab_size])?;
		let cross_entropy =
			loss::masked_cross_entropy(&logits, &target_ids, &loss_mask, batch.valid_count)?;
		let objective = match prior.moe_aux_loss()? {
			Some(auxiliary) => matrix::add(&cross_entropy, &auxiliary)?,
			None => cross_entropy.clone(),
		};
		tape.backward(&objective)?;
		prior.update_moe_routing_bias()?;
		training.complete_step(&cross_entropy)?;
		completed_steps += 1;
		let latest = training
			.snapshot()
			.last_loss()
			.ok_or_else(|| Error::internal("ALM prior training completed a step without a loss"))?;
		if !latest.is_finite() {
			return Err(Error::data_loss(format!(
				"ALM prior loss became non-finite at step {completed_steps}"
			)));
		}
		initial_loss.get_or_insert(latest);
	}
	let snapshot = training.finish()?;
	let gpu = snapshot.gpu_timing_stats();
	Ok(PriorTrainingReport {
		steps: snapshot.step_count(),
		steps_per_epoch,
		initial_loss: initial_loss.unwrap_or_default(),
		final_loss: snapshot.last_loss().unwrap_or_default(),
		mean_loss: snapshot.training_mean_loss(),
		wall_seconds: snapshot.elapsed().as_secs_f64(),
		gpu_mean_ms: gpu.mean_ms,
		gpu_median_ms: gpu.median_ms,
		gpu_p95_ms: gpu.p95_ms,
	})
}

fn validate_training_config(
	prior: &AlmPrior,
	sequences: &[Vec<i32>],
	conditioning: Option<PriorConditioning<'_>>,
	config: PriorTrainingConfig,
) -> Result<()> {
	if config.epochs == 0 || config.batch_size == 0 || config.window_len == 0 {
		return Err(Error::invalid_argument(
			"ALM prior epochs, batch size, and window length must be positive",
		));
	}
	let model_sequence = config
		.window_len
		.checked_add(usize::from(conditioning.is_some()))
		.ok_or_else(|| Error::resource_exhausted("ALM prior sequence length overflows usize"))?;
	if model_sequence > prior.config().max_sequence_length {
		return Err(Error::invalid_argument(
			"ALM prior training sequence exceeds the model position table",
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
			"ALM prior learning rates and weight decay are invalid",
		));
	}
	if sequences.is_empty() {
		return Err(Error::invalid_argument(
			"ALM prior training requires token sequences",
		));
	}
	match conditioning {
		Some(source) => {
			if prior.config().text_feature_dim == 0
				|| source.feature_dim != prior.config().text_feature_dim
				|| source.by_sequence.len() != sequences.len()
				|| source.feature_dim == 0
			{
				return Err(Error::invalid_argument(
					"ALM prior conditioning does not match the model or token corpus",
				));
			}
			if source.by_sequence.iter().any(|features| {
				features.is_empty()
					|| !features.len().is_multiple_of(source.feature_dim)
					|| features.iter().any(|value| !value.is_finite())
			}) {
				return Err(Error::invalid_argument(
					"every ALM sequence requires finite complete caption-feature rows",
				));
			}
		}
		None if prior.config().text_feature_dim != 0 => {
			return Err(Error::invalid_argument(
				"conditioned ALM prior training requires text features",
			));
		}
		None => {}
	}
	let steps_per_epoch = sequences.len().div_ceil(config.batch_size).max(1) as u64;
	let total_steps = config
		.epochs
		.checked_mul(steps_per_epoch)
		.ok_or_else(|| Error::resource_exhausted("ALM prior provisional step count exceeds u64"))?;
	if config.warmup_steps >= total_steps {
		return Err(Error::invalid_argument(
			"ALM prior warmup must be shorter than the training run",
		));
	}
	Ok(())
}

fn gather_text_features(
	engine: &Engine,
	source: PriorConditioning<'_>,
	windows: &[super::PriorWindow],
	cursor: usize,
	batch_size: usize,
	epoch: u64,
) -> Result<Matrix> {
	let count = batch_size
		.checked_mul(source.feature_dim)
		.ok_or_else(|| Error::resource_exhausted("ALM text batch size overflows usize"))?;
	let mut batch = vec![0.0; count];
	for row in 0..batch_size {
		let window = windows[(cursor + row) % windows.len()];
		let features = &source.by_sequence[window.sequence];
		let captions = features.len() / source.feature_dim;
		let selector = source
			.seed
			.wrapping_add(epoch)
			.wrapping_add(window.sequence as u64);
		let caption = usize::try_from(selector % captions as u64)
			.map_err(|_| Error::resource_exhausted("ALM caption index exceeds usize"))?;
		let start = caption * source.feature_dim;
		batch[row * source.feature_dim..(row + 1) * source.feature_dim]
			.copy_from_slice(&features[start..start + source.feature_dim]);
	}
	Matrix::from_slice(engine, [batch_size, source.feature_dim], &batch)
}
