//! Explicit host observations for machine-learning metrics.

use crate::{DType, Error, Matrix, Result, matrix};

/// Read one scalar FP32 loss value.
///
/// This is an explicit host-observation boundary: pending work is submitted and
/// the producing event is awaited before the value is returned.
///
/// # Errors
///
/// Returns an error unless `loss` contains exactly one F32 value, or submission,
/// completion, or readback fails.
pub fn scalar_loss(loss: &Matrix) -> Result<f32> {
	if loss.dtype() != DType::F32 || loss.num_elements() != 1 {
		return Err(Error::invalid_argument(format!(
			"ml scalar_loss requires exactly one F32 value; found shape {:?} {}",
			loss.shape(),
			loss.dtype().token()
		)));
	}
	Ok(loss.read_f32()?[0])
}

/// Compute last-axis categorical accuracy as a fraction in `[0, 1]`.
///
/// Argmax and counting execute on the owning device. Only the resulting U32
/// scalar crosses the host boundary; this function then waits for that scalar.
/// Equal maximum logits select the first class, matching the OA C++ contract.
///
/// # Errors
///
/// Returns an error when [`matrix::categorical_accuracy_count`] rejects the
/// logits or labels, or submission, completion, or scalar readback fails.
pub fn accuracy(logits: &Matrix, labels: &Matrix) -> Result<f32> {
	let count = matrix::categorical_accuracy_count(logits, labels)?;
	let correct = count.read::<u32>()?[0];
	Ok(correct as f32 / labels.num_elements() as f32)
}

/// Per-class prediction/label counts behind one-vs-rest metrics.
///
/// Logits and labels are read back and the last-axis argmax runs on the host —
/// the same explicit host-observation boundary as [`scalar_loss`], sized for
/// evaluation batches rather than hot loops. A class absent from both
/// predictions and labels reports `0.0` for all derived values rather than a
/// NaN. Equal maximum logits select the first class, matching the OA device
/// argmax contract.
///
/// # Errors
///
/// Returns an error when `logits` is not rank-two F32 `["samples", C]`,
/// `labels` does not carry one U32/I32 label per row, `class_index` is out of
/// range for the class axis, or readback fails.
pub fn precision_recall_f1(
	logits: &Matrix,
	labels: &Matrix,
	class_index: usize,
) -> Result<ClassificationCounts> {
	let &[samples, num_classes] = logits.shape() else {
		return Err(Error::invalid_argument(format!(
			"metric logits must be rank two; found {:?}",
			logits.shape(),
		)));
	};
	if logits.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"metric logits must be F32; found {}",
			logits.dtype().token(),
		)));
	}
	if class_index >= num_classes {
		return Err(Error::invalid_argument(format!(
			"class index {class_index} exceeds the class axis {num_classes}"
		)));
	}
	if labels.num_elements() != samples {
		return Err(Error::invalid_argument(format!(
			"metric logits predict {samples} rows; found {} labels",
			labels.num_elements(),
		)));
	}
	let logits_flat = logits.read_f32()?;
	let labels_flat = labels.read::<u32>()?;
	let mut counts = ClassificationCounts::default();
	for (row, label) in labels_flat.iter().enumerate() {
		let prediction = logits_flat[row * num_classes..(row + 1) * num_classes]
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
			.map(|(index, _)| index)
			.unwrap_or(0);
		let predicted_class = prediction == class_index;
		let actual_class = *label as usize == class_index;
		match (predicted_class, actual_class) {
			(true, true) => counts.true_positives += 1,
			(true, false) => counts.false_positives += 1,
			(false, true) => counts.false_negatives += 1,
			(false, false) => counts.true_negatives += 1,
		}
	}
	Ok(counts)
}

/// Per-class prediction/label counts behind [`precision_recall_f1`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassificationCounts {
	pub true_positives: u64,
	pub false_positives: u64,
	pub true_negatives: u64,
	pub false_negatives: u64,
}

impl ClassificationCounts {
	/// Precision `TP / (TP + FP)`; `0.0` when the class was never predicted.
	pub fn precision(&self) -> f32 {
		let predicted = self.true_positives + self.false_positives;
		if predicted == 0 {
			return 0.0;
		}
		self.true_positives as f32 / predicted as f32
	}

	/// Recall `TP / (TP + FN)`; `0.0` when the class has no labels.
	pub fn recall(&self) -> f32 {
		let actual = self.true_positives + self.false_negatives;
		if actual == 0 {
			return 0.0;
		}
		self.true_positives as f32 / actual as f32
	}

	/// Harmonic mean of precision and recall; `0.0` when both are zero.
	pub fn f1(&self) -> f32 {
		let (p, r) = (self.precision(), self.recall());
		if p + r == 0.0 {
			return 0.0;
		}
		2.0 * p * r / (p + r)
	}
}

/// Macro-averaged F1 over `num_classes` one-vs-rest counts.
///
/// # Errors
///
/// Returns an error when `num_classes` is zero or any per-class count fails.
pub fn macro_f1(logits: &Matrix, labels: &Matrix, num_classes: usize) -> Result<f32> {
	if num_classes == 0 {
		return Err(Error::invalid_argument(
			"macro_f1 requires at least one class",
		));
	}
	let mut total = 0.0;
	for class_index in 0..num_classes {
		total += precision_recall_f1(logits, labels, class_index)?.f1();
	}
	Ok(total / num_classes as f32)
}

/// Binary ROC AUC from positive-class `scores` against binary `labels`.
///
/// Scores and labels are read back and ranked on the host with the
/// Mann-Whitney formulation (tied scores share average rank), so the value is
/// exactly the probability that a random positive outranks a random negative.
/// This is an evaluation-side host metric, not a capture-safe device op.
///
/// # Errors
///
/// Returns an error when `scores` is not F32, shapes disagree, labels are not
/// `0`/`1`, or one class is absent.
pub fn roc_auc(scores: &Matrix, labels: &Matrix) -> Result<f32> {
	if scores.dtype() != DType::F32 {
		return Err(Error::invalid_argument(format!(
			"roc_auc requires F32 scores; found {}",
			scores.dtype().token()
		)));
	}
	let scores = scores.read_f32()?;
	let labels = labels.read::<u32>()?;
	if scores.len() != labels.len() {
		return Err(Error::invalid_argument(format!(
			"roc_auc has {} scores for {} labels",
			scores.len(),
			labels.len(),
		)));
	}
	if scores.is_empty() || labels.iter().any(|label| *label > 1) {
		return Err(Error::invalid_argument(
			"roc_auc requires a nonempty binary label set",
		));
	}
	let positives = labels.iter().filter(|label| **label == 1).count();
	if positives == 0 || positives == labels.len() {
		return Err(Error::invalid_argument(
			"roc_auc requires at least one positive and one negative label",
		));
	}
	// Average-rank assignment for ties, ascending by score.
	let mut order: Vec<usize> = (0..scores.len()).collect();
	order.sort_by(|a, b| scores[*a].total_cmp(&scores[*b]));
	let mut rank_sum_positive = 0.0_f64;
	let mut index = 0;
	while index < order.len() {
		let mut end = index + 1;
		while end < order.len() && scores[order[end]] == scores[order[index]] {
			end += 1;
		}
		let average_rank = (index + 1 + end) as f64 / 2.0;
		for &sample in &order[index..end] {
			if labels[sample] == 1 {
				rank_sum_positive += average_rank;
			}
		}
		index = end;
	}
	let negatives = (labels.len() - positives) as f64;
	let positives = positives as f64;
	Ok(((rank_sum_positive - positives * (positives + 1.0) / 2.0) / (positives * negatives)) as f32)
}

/// Corpus BLEU over token sequences with a brevity penalty, no smoothing.
///
/// Counts clipped n-gram matches up to `max_order` across the corpus and
/// combines them geometrically. A corpus with no matches at any order returns
/// `0.0`; hypotheses shorter than references receive the standard
/// `exp(1 - ref/hyp)` penalty. Inputs are host token sequences — this is an
/// evaluation-side metric, not a device op.
///
/// # Errors
///
/// Returns an error when `max_order` is zero, the corpora differ in length, a
/// hypothesis is empty, or any sequence is shorter than `max_order`.
pub fn bleu(references: &[Vec<u32>], hypotheses: &[Vec<u32>], max_order: usize) -> Result<f32> {
	use std::collections::HashMap;
	if max_order == 0 {
		return Err(Error::invalid_argument("bleu requires max_order >= 1"));
	}
	if references.len() != hypotheses.len() {
		return Err(Error::invalid_argument(format!(
			"bleu received {} references for {} hypotheses",
			references.len(),
			hypotheses.len(),
		)));
	}
	if hypotheses.iter().any(Vec::is_empty) {
		return Err(Error::invalid_argument("bleu hypotheses must be nonempty"));
	}
	let ngrams = |sequence: &[u32], order: usize| -> HashMap<Vec<u32>, u64> {
		let mut counts = HashMap::new();
		for window in sequence.windows(order) {
			*counts.entry(window.to_vec()).or_insert(0) += 1;
		}
		counts
	};
	let mut matches = vec![0_u64; max_order];
	let mut candidates = vec![0_u64; max_order];
	let (mut ref_len, mut hyp_len) = (0_u64, 0_u64);
	for (reference, hypothesis) in references.iter().zip(hypotheses.iter()) {
		ref_len += reference.len() as u64;
		hyp_len += hypothesis.len() as u64;
		for (order, (matched, total)) in
			(1..=max_order).zip(matches.iter_mut().zip(candidates.iter_mut()))
		{
			if hypothesis.len() < order {
				continue;
			}
			let reference_counts = ngrams(reference, order);
			*total += (hypothesis.len() + 1 - order) as u64;
			for (gram, count) in ngrams(hypothesis, order) {
				*matched += count.min(reference_counts.get(&gram).copied().unwrap_or(0));
			}
		}
	}
	if candidates.contains(&0) {
		return Err(Error::invalid_argument(
			"bleu requires every hypothesis to cover every n-gram order",
		));
	}
	if matches.contains(&0) {
		return Ok(0.0);
	}
	let log_precision: f64 = matches
		.iter()
		.zip(candidates.iter())
		.map(|(matched, total)| (*matched as f64 / *total as f64).ln())
		.sum::<f64>()
		/ max_order as f64;
	let brevity = if hyp_len >= ref_len {
		1.0
	} else {
		(1.0 - ref_len as f64 / hyp_len as f64).exp()
	};
	Ok((brevity * log_precision.exp()) as f32)
}
