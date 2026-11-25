//! Adaptive log-softmax output layer (`nn.AdaptiveLogSoftmaxWithLoss`).

use std::rc::Rc;

use crate::{DType, Error, Matrix, Result, Runtime, matrix};

use super::super::{Module, ModuleRegistry};
use super::Linear;

/// Output bundle of [`AdaptiveLogSoftmaxWithLoss::forward`]: the assembled
/// `[N, V]` log-probabilities and the mean negative log-likelihood scalar.
pub struct AdaptiveLogSoftmaxResult {
	pub output: Matrix,
	pub loss: Matrix,
}

/// Two-level output layer: a head linear scores the most frequent classes plus
/// one logit per tail cluster, while each tail cluster owns a projection and
/// output linear that rescore its classes conditioned on the head cluster
/// probability. Matches the `nn.AdaptiveLogSoftmaxWithLoss` contract: cutoffs
/// partition `0..n_classes` into a head range `0..cutoffs[0]` and one tail
/// cluster per cutoff, with tail `i` projecting `in_features` down to
/// `round(in_features / div_value^(i+1))` before its output linear.
pub struct AdaptiveLogSoftmaxWithLoss {
	in_features: usize,
	n_classes: usize,
	cutoffs: Vec<usize>,
	head: Rc<Linear>,
	tails: Vec<(Rc<Linear>, Rc<Linear>)>,
	registry: ModuleRegistry,
}

impl AdaptiveLogSoftmaxWithLoss {
	/// Construct on the current runtime with the thread-local seed source.
	///
	/// # Errors
	///
	/// Returns an error when the contract or initialization fails.
	pub fn new(in_features: usize, n_classes: usize, cutoffs: &[usize]) -> Result<Self> {
		Self::new_with_seed(in_features, n_classes, cutoffs, 0)
	}

	/// Construct on the current runtime with an explicit seed.
	///
	/// Uses `div_value` 4.0 and a bias-free head, matching the torch defaults.
	///
	/// # Errors
	///
	/// Returns an error when the contract or initialization fails.
	pub fn new_with_seed(
		in_features: usize,
		n_classes: usize,
		cutoffs: &[usize],
		seed: u64,
	) -> Result<Self> {
		Self::new_with_seed_on(
			&Runtime::current()?,
			in_features,
			n_classes,
			cutoffs,
			4.0,
			false,
			true,
			seed,
		)
	}

	/// Construct a deterministically initialized adaptive softmax layer.
	///
	/// `cutoffs` must be a nonempty strictly increasing list below `n_classes`;
	/// each cutoff opens a tail cluster. `div_value` sets each tail's
	/// projection width and `head_bias` adds a bias to the head linear.
	///
	/// # Errors
	///
	/// Returns an error when the contract or initialization fails.
	#[allow(clippy::too_many_arguments)]
	pub fn new_with_seed_on(
		runtime: &Runtime,
		in_features: usize,
		n_classes: usize,
		cutoffs: &[usize],
		div_value: f64,
		head_bias: bool,
		_tail_bias: bool,
		seed: u64,
	) -> Result<Self> {
		let (head_size, tail_sizes) = validate_adaptive_shape(
			"AdaptiveLogSoftmaxWithLoss::new_with_seed_on",
			in_features,
			n_classes,
			cutoffs,
		)?;
		if div_value <= 0.0 || !div_value.is_finite() {
			return Err(Error::invalid_argument(format!(
				"AdaptiveLogSoftmaxWithLoss requires a positive finite div_value; found {div_value}"
			)));
		}
		let head = Rc::new(Linear::new_with_seed_and_bias_on(
			runtime,
			in_features,
			head_size,
			head_bias,
			seed,
		)?);
		let mut tails = Vec::with_capacity(tail_sizes.len());
		for (index, size) in tail_sizes.iter().enumerate() {
			let hidden = (in_features as f64 / div_value.powi(index as i32 + 1))
				.round()
				.max(1.0) as usize;
			let projection = Rc::new(Linear::new_with_seed_and_bias_on(
				runtime,
				in_features,
				hidden,
				true,
				seed + 1 + index as u64,
			)?);
			let output = Rc::new(Linear::new_with_seed_and_bias_on(
				runtime,
				hidden,
				*size,
				true,
				seed + 1 + tail_sizes.len() as u64 + index as u64,
			)?);
			tails.push((projection, output));
		}
		Self::from_parts(in_features, n_classes, cutoffs, head, tails)
	}

	/// Construct from explicit head and tail linears (deterministic tests and
	/// checkpoint assembly).
	///
	/// # Errors
	///
	/// Returns an error when the contract or any linear's shape is invalid.
	pub fn from_parts(
		in_features: usize,
		n_classes: usize,
		cutoffs: &[usize],
		head: Rc<Linear>,
		tails: Vec<(Rc<Linear>, Rc<Linear>)>,
	) -> Result<Self> {
		const OPERATION: &str = "AdaptiveLogSoftmaxWithLoss::from_parts";
		let (head_size, tail_sizes) =
			validate_adaptive_shape(OPERATION, in_features, n_classes, cutoffs)?;
		if head.input_features() != in_features || head.output_features() != head_size {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} head must map {in_features} inputs to {head_size} outputs"
			)));
		}
		if tails.len() != tail_sizes.len() {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires {} tail clusters; found {}",
				tail_sizes.len(),
				tails.len()
			)));
		}
		for (index, ((projection, output), size)) in tails.iter().zip(&tail_sizes).enumerate() {
			if projection.input_features() != in_features
				|| output.input_features() != projection.output_features()
				|| output.output_features() != *size
			{
				return Err(Error::invalid_argument(format!(
					"{OPERATION} tail {index} must chain {in_features} -> hidden -> {size}"
				)));
			}
		}
		let mut registry = ModuleRegistry::new();
		registry.register_module("head", head.clone())?;
		for (index, (projection, output)) in tails.iter().enumerate() {
			registry.register_module(format!("tail_{index}_projection"), projection.clone())?;
			registry.register_module(format!("tail_{index}_output"), output.clone())?;
		}
		Ok(Self {
			in_features,
			n_classes,
			cutoffs: cutoffs.to_vec(),
			head,
			tails,
			registry,
		})
	}

	/// Return the assembled `[N, V]` log-probabilities for a rank-two FP32
	/// input, matching `AdaptiveLogSoftmaxWithLoss.log_prob`.
	///
	/// # Errors
	///
	/// Returns an error when the input contract or runtime recording fails.
	pub fn log_prob(&self, input: &Matrix) -> Result<Matrix> {
		let (head_log_probs, tail_log_probs, _rows) = self.distributions(input)?;
		let mut parts = Vec::with_capacity(self.tails.len() + 1);
		parts.push(matrix::slice(
			&head_log_probs,
			1,
			0,
			self.cutoffs[0] as i64,
		)?);
		for (index, tail_log_probs) in tail_log_probs.iter().enumerate() {
			let cluster_logit = self.cluster_log_probs(&head_log_probs, index)?;
			parts.push(matrix::add(tail_log_probs, &cluster_logit)?);
		}
		matrix::concat(&parts, 1)
	}

	/// Compute the assembled log-probabilities and the mean negative
	/// log-likelihood for a rank-two FP32 `input` `[N, I]` and a rank-one
	/// `target` `[N]` holding class indices as I32, U32, or FP32.
	///
	/// # Errors
	///
	/// Returns an error when the contract or runtime recording fails.
	pub fn forward(&self, input: &Matrix, target: &Matrix) -> Result<AdaptiveLogSoftmaxResult> {
		const OPERATION: &str = "AdaptiveLogSoftmaxWithLoss::forward";
		let (head_log_probs, tail_log_probs, rows) = self.distributions(input)?;
		if target.shape().len() != 1 || target.shape()[0] != rows {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} target must have shape [N] matching the input rows"
			)));
		}
		if !target.runtime().same_as(input.runtime()) {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} input and target must belong to the same runtime"
			)));
		}
		let target_f32 = match target.dtype() {
			DType::F32 => target.clone(),
			DType::I32 | DType::U32 => matrix::cast_i32_f32(target)?,
			other => {
				return Err(Error::invalid_argument(format!(
					"{OPERATION} requires I32, U32, or F32 targets; found {}",
					other.token()
				)));
			}
		};
		// Cluster membership masks: mask for cluster 0 is 1 - ge(c0); mask for
		// tail cluster i (0-based) is ge(c_i) - ge(c_{i+1}) with c_k = V.
		let mut previous = matrix::greater_equal(&target_f32, self.cutoffs[0] as f32)?;
		let mask0 = matrix::add_scalar(&matrix::neg(&previous)?, 1.0)?;
		let mut masks = Vec::with_capacity(self.tails.len());
		let boundaries = self
			.cutoffs
			.iter()
			.copied()
			.skip(1)
			.chain(std::iter::once(self.n_classes));
		for upper in boundaries {
			let next = matrix::greater_equal(&target_f32, upper as f32)?;
			masks.push(matrix::sub(&previous, &next)?);
			previous = next;
		}
		// Per-item head column: the class index for head items, the cluster
		// logit column c0 + i for tail cluster i.
		let mut head_index = matrix::mul(&mask0, &target_f32)?;
		for (index, mask) in masks.iter().enumerate() {
			let column_index = (self.cutoffs[0] + index) as f32;
			let contribution = matrix::div_scalar(mask, 1.0 / column_index)?;
			head_index = matrix::add(&head_index, &contribution)?;
		}
		let head_index = matrix::cast_f32_i32(&head_index)?.reshape([rows, 1])?;
		let mut nll = matrix::neg(&matrix::gather_last_dim(&head_log_probs, &head_index)?)?;
		let target_column = target_f32.reshape([rows, 1])?;
		for (index, tail_log_probs) in tail_log_probs.iter().enumerate() {
			let local = matrix::sub_scalar(&target_column, self.cutoffs[index] as f32)?;
			let local = matrix::cast_f32_i32(&local)?;
			let gathered = matrix::gather_last_dim(tail_log_probs, &local)?;
			let term = matrix::mul(&gathered, &masks[index].reshape([rows, 1])?)?;
			nll = matrix::sub(&nll, &term)?;
		}
		let loss = matrix::mean(&nll, -1)?.reshape([])?;
		let mut parts = Vec::with_capacity(self.tails.len() + 1);
		parts.push(matrix::slice(
			&head_log_probs,
			1,
			0,
			self.cutoffs[0] as i64,
		)?);
		for (index, tail_log_probs) in tail_log_probs.iter().enumerate() {
			let cluster_logit = self.cluster_log_probs(&head_log_probs, index)?;
			parts.push(matrix::add(tail_log_probs, &cluster_logit)?);
		}
		let output = matrix::concat(&parts, 1)?;
		Ok(AdaptiveLogSoftmaxResult { output, loss })
	}

	/// Head and tail log-softmax distributions for a rank-two `[N, I]` input.
	fn distributions(&self, input: &Matrix) -> Result<(Matrix, Vec<Matrix>, usize)> {
		const OPERATION: &str = "AdaptiveLogSoftmaxWithLoss";
		if input.dtype() != DType::F32 || input.shape().len() != 2 {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires a rank-two FP32 input"
			)));
		}
		if input.shape()[1] != self.in_features {
			return Err(Error::invalid_argument(format!(
				"{OPERATION} requires {} input features; found {}",
				self.in_features,
				input.shape()[1]
			)));
		}
		let rows = input.shape()[0];
		let head_log_probs = matrix::log_softmax(&self.head.forward(input)?, -1)?;
		let mut tail_log_probs = Vec::with_capacity(self.tails.len());
		for (projection, output) in &self.tails {
			let logits = output.forward(&projection.forward(input)?)?;
			tail_log_probs.push(matrix::log_softmax(&logits, -1)?);
		}
		Ok((head_log_probs, tail_log_probs, rows))
	}

	/// Broadcast `[N, 1]` cluster-logit column of tail `index` from the head
	/// log-probabilities.
	fn cluster_log_probs(&self, head_log_probs: &Matrix, index: usize) -> Result<Matrix> {
		let rows = head_log_probs.shape()[0];
		let column = self.cutoffs[0] + index;
		let index =
			head_log_probs.new_from_slice_on_same_runtime(vec![rows, 1], &vec![column as i32; rows])?;
		matrix::gather_last_dim(head_log_probs, &index)
	}

	/// Return the input feature count.
	pub const fn in_features(&self) -> usize {
		self.in_features
	}

	/// Return the class count.
	pub const fn n_classes(&self) -> usize {
		self.n_classes
	}
}

impl Module for AdaptiveLogSoftmaxWithLoss {
	fn forward(&self, input: &Matrix) -> Result<Matrix> {
		let _ = input;
		Err(Error::invalid_argument(
			"AdaptiveLogSoftmaxWithLoss requires a target; call forward(input, target) or log_prob(input)",
		))
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

/// Validate the shared shape contract and return `(head_size, tail_sizes)`.
fn validate_adaptive_shape(
	operation: &'static str,
	in_features: usize,
	n_classes: usize,
	cutoffs: &[usize],
) -> Result<(usize, Vec<usize>)> {
	if in_features == 0 || n_classes == 0 || cutoffs.is_empty() {
		return Err(Error::invalid_argument(format!(
			"{operation} requires nonzero in_features and n_classes with at least one cutoff"
		)));
	}
	let mut previous = 0usize;
	for &cutoff in cutoffs {
		if cutoff <= previous || cutoff >= n_classes {
			return Err(Error::invalid_argument(format!(
				"{operation} cutoffs must strictly increase inside 0..{n_classes}"
			)));
		}
		previous = cutoff;
	}
	let tail_sizes: Vec<usize> = cutoffs
		.iter()
		.copied()
		.chain(std::iter::once(n_classes))
		.skip(1)
		.zip(cutoffs.iter().copied())
		.map(|(upper, lower)| upper - lower)
		.collect();
	Ok((cutoffs[0] + cutoffs.len(), tail_sizes))
}
