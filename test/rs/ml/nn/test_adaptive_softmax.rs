//! AdaptiveLogSoftmaxWithLoss contracts: two-level nll oracle, log_prob
//! assembly, gradient flow, and shape validation.

use std::rc::Rc;

use oa::ml::nn::{AdaptiveLogSoftmaxWithLoss, Linear};
use oa::ml::{GradientTape, Module};

const IN_FEATURES: usize = 4;
const N_CLASSES: usize = 10;
const CUTOFFS: [usize; 2] = [4, 7];
const HEAD_SIZE: usize = 6; // CUTOFFS[0] + number of tails
const TAIL0_HIDDEN: usize = 2;
const TAIL1_HIDDEN: usize = 3;
const TAIL0_SIZE: usize = 3; // classes 4..7
const TAIL1_SIZE: usize = 3; // classes 7..10

const INPUT: [f32; 12] = [
	0.5, -0.25, 0.1, 0.3, //
	-0.4, 0.2, 0.05, -0.1, //
	0.15, 0.6, -0.35, 0.2, //
];
const TARGETS: [i32; 3] = [2, 5, 8]; // head, tail 0, tail 1 items

const HEAD_WEIGHT: [f32; 24] = [
	0.2, -0.1, 0.05, 0.1, //
	-0.15, 0.25, -0.05, 0.0, //
	0.1, 0.0, -0.2, 0.15, //
	0.05, -0.1, 0.3, -0.05, //
	-0.2, 0.1, 0.05, 0.1, //
	0.15, -0.05, -0.1, 0.2, //
];
const HEAD_BIAS: [f32; 6] = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02];
const TAIL0_PROJ_W: [f32; 8] = [0.1, -0.2, 0.15, 0.05, -0.05, 0.1, -0.1, 0.2];
const TAIL0_PROJ_B: [f32; 2] = [0.02, -0.01];
const TAIL0_OUT_W: [f32; 6] = [0.2, -0.1, -0.15, 0.25, 0.1, -0.05];
const TAIL0_OUT_B: [f32; 3] = [0.01, 0.0, -0.02];
const TAIL1_PROJ_W: [f32; 12] = [
	0.05, 0.1, -0.15, 0.2, //
	-0.1, 0.2, 0.05, -0.05, //
	0.15, -0.1, 0.1, 0.0, //
];
const TAIL1_PROJ_B: [f32; 3] = [0.0, 0.01, -0.01];
const TAIL1_OUT_W: [f32; 9] = [
	0.1, -0.2, 0.15, //
	-0.05, 0.1, -0.1, //
	0.2, -0.05, 0.1, //
];
const TAIL1_OUT_B: [f32; 3] = [-0.01, 0.02, 0.0];

/// `x [N,I] @ w [O,I]^T + b [O]` → `[N,O]`.
fn linear_rows(
	x: &[f32],
	rows: usize,
	in_features: usize,
	w: &[f32],
	b: &[f32],
	out: usize,
) -> Vec<f32> {
	let mut result = vec![0.0; rows * out];
	for n in 0..rows {
		for o in 0..out {
			let mut sum = b[o];
			for i in 0..in_features {
				sum += x[n * in_features + i] * w[o * in_features + i];
			}
			result[n * out + o] = sum;
		}
	}
	result
}

fn log_softmax_rows(v: &mut [f32], rows: usize, cols: usize) {
	for n in 0..rows {
		let row = &mut v[n * cols..(n + 1) * cols];
		let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		let log_sum = row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
		for x in row.iter_mut() {
			*x -= max + log_sum;
		}
	}
}

/// Head/tail log-probs plus the torch two-level nll for the fixture above.
fn oracle() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
	let mut head_lp = linear_rows(&INPUT, 3, IN_FEATURES, &HEAD_WEIGHT, &HEAD_BIAS, HEAD_SIZE);
	log_softmax_rows(&mut head_lp, 3, HEAD_SIZE);
	let proj0 = linear_rows(
		&INPUT,
		3,
		IN_FEATURES,
		&TAIL0_PROJ_W,
		&TAIL0_PROJ_B,
		TAIL0_HIDDEN,
	);
	let mut tail0_lp = linear_rows(
		&proj0,
		3,
		TAIL0_HIDDEN,
		&TAIL0_OUT_W,
		&TAIL0_OUT_B,
		TAIL0_SIZE,
	);
	log_softmax_rows(&mut tail0_lp, 3, TAIL0_SIZE);
	let proj1 = linear_rows(
		&INPUT,
		3,
		IN_FEATURES,
		&TAIL1_PROJ_W,
		&TAIL1_PROJ_B,
		TAIL1_HIDDEN,
	);
	let mut tail1_lp = linear_rows(
		&proj1,
		3,
		TAIL1_HIDDEN,
		&TAIL1_OUT_W,
		&TAIL1_OUT_B,
		TAIL1_SIZE,
	);
	log_softmax_rows(&mut tail1_lp, 3, TAIL1_SIZE);
	(head_lp, tail0_lp, tail1_lp)
}

/// Per-item nll following the torch cluster decomposition.
fn oracle_nll(
	head_lp: &[f32],
	tail0_lp: &[f32],
	tail1_lp: &[f32],
	target: usize,
	row: usize,
) -> f32 {
	let c0 = CUTOFFS[0];
	if target < c0 {
		-head_lp[row * HEAD_SIZE + target]
	} else if target < CUTOFFS[1] {
		-(head_lp[row * HEAD_SIZE + c0] + tail0_lp[row * TAIL0_SIZE + (target - c0)])
	} else {
		-(head_lp[row * HEAD_SIZE + c0 + 1] + tail1_lp[row * TAIL1_SIZE + (target - CUTOFFS[1])])
	}
}

/// Assembled `[N, V]` log-prob contract of `log_prob`.
fn oracle_log_prob(
	head_lp: &[f32],
	tail0_lp: &[f32],
	tail1_lp: &[f32],
	row: usize,
	class: usize,
) -> f32 {
	let c0 = CUTOFFS[0];
	if class < c0 {
		head_lp[row * HEAD_SIZE + class]
	} else if class < CUTOFFS[1] {
		head_lp[row * HEAD_SIZE + c0] + tail0_lp[row * TAIL0_SIZE + (class - c0)]
	} else {
		head_lp[row * HEAD_SIZE + c0 + 1] + tail1_lp[row * TAIL1_SIZE + (class - CUTOFFS[1])]
	}
}

fn build_module() -> Result<AdaptiveLogSoftmaxWithLoss, oa::Error> {
	let head = Rc::new(Linear::new_from_matrices(
		oa::Matrix::new_from_f32([HEAD_SIZE, IN_FEATURES], &HEAD_WEIGHT)?,
		oa::Matrix::new_from_f32([HEAD_SIZE], &HEAD_BIAS)?,
	)?);
	let tail0 = (
		Rc::new(Linear::new_from_matrices(
			oa::Matrix::new_from_f32([TAIL0_HIDDEN, IN_FEATURES], &TAIL0_PROJ_W)?,
			oa::Matrix::new_from_f32([TAIL0_HIDDEN], &TAIL0_PROJ_B)?,
		)?),
		Rc::new(Linear::new_from_matrices(
			oa::Matrix::new_from_f32([TAIL0_SIZE, TAIL0_HIDDEN], &TAIL0_OUT_W)?,
			oa::Matrix::new_from_f32([TAIL0_SIZE], &TAIL0_OUT_B)?,
		)?),
	);
	let tail1 = (
		Rc::new(Linear::new_from_matrices(
			oa::Matrix::new_from_f32([TAIL1_HIDDEN, IN_FEATURES], &TAIL1_PROJ_W)?,
			oa::Matrix::new_from_f32([TAIL1_HIDDEN], &TAIL1_PROJ_B)?,
		)?),
		Rc::new(Linear::new_from_matrices(
			oa::Matrix::new_from_f32([TAIL1_SIZE, TAIL1_HIDDEN], &TAIL1_OUT_W)?,
			oa::Matrix::new_from_f32([TAIL1_SIZE], &TAIL1_OUT_B)?,
		)?),
	);
	AdaptiveLogSoftmaxWithLoss::from_parts(IN_FEATURES, N_CLASSES, &CUTOFFS, head, vec![tail0, tail1])
}

test_vk!(adaptive_log_softmax_forward_matches_oracle, runtime, {
	let module = build_module()?;
	let input = oa::Matrix::new_from_f32([3, IN_FEATURES], &INPUT)?;
	let target = oa::Matrix::new_from_slice([3], &TARGETS)?;
	let result = module.forward(&input, &target)?;
	assert_eq!(result.output.shape(), &[3, N_CLASSES]);
	assert_eq!(result.loss.shape(), &[] as &[usize]);
	let (head_lp, tail0_lp, tail1_lp) = oracle();
	let expected_loss = TARGETS
		.iter()
		.enumerate()
		.map(|(row, &t)| oracle_nll(&head_lp, &tail0_lp, &tail1_lp, t as usize, row))
		.sum::<f32>()
		/ 3.0;
	let actual_loss = result.loss.read_f32()?[0];
	assert!(
		(actual_loss - expected_loss).abs() <= 1.0e-5,
		"loss deviates: {actual_loss} vs {expected_loss}"
	);
	let output = result.output.read_f32()?;
	for row in 0..3 {
		for class in 0..N_CLASSES {
			let expected = oracle_log_prob(&head_lp, &tail0_lp, &tail1_lp, row, class);
			let actual = output[row * N_CLASSES + class];
			assert!(
				(actual - expected).abs() <= 1.0e-5,
				"output[{row},{class}] deviates: {actual} vs {expected}"
			);
		}
	}
	// log_prob agrees with the forward output.
	let log_prob = module.log_prob(&input)?.read_f32()?;
	for (a, b) in log_prob.iter().zip(output.iter()) {
		assert!((a - b).abs() <= 1.0e-6, "log_prob deviates from output");
	}
	Ok(())
});

test_vk!(
	adaptive_log_softmax_backward_reaches_all_parameters,
	runtime,
	{
		let module = build_module()?;
		let input = oa::Matrix::new_from_f32([3, IN_FEATURES], &INPUT)?;
		let target = oa::Matrix::new_from_slice([3], &TARGETS)?;
		let tape = GradientTape::new();
		let result = module.forward(&input, &target)?;
		tape.backward(&result.loss)?;
		let parameters = module.all_parameters()?;
		assert_eq!(parameters.len(), 10); // 2 (head) + 4 per tail
		for (index, parameter) in parameters.iter().enumerate() {
			let gradient = parameter
				.gradient()
				.unwrap_or_else(|| panic!("parameter {index} missing gradient"));
			assert!(
				gradient.read_f32()?.iter().all(|v| v.is_finite()),
				"parameter {index} gradient contains non-finite values"
			);
		}
		Ok(())
	}
);

test_vk!(adaptive_log_softmax_contract_errors, runtime, {
	let module = build_module()?;
	let input = oa::Matrix::new_from_f32([3, IN_FEATURES], &INPUT)?;
	// The Module trait forward requires a target.
	assert!(Module::forward(&module, &input).is_err());
	// Rank and feature contracts.
	assert!(
		module
			.forward(
				&oa::Matrix::new_from_f32([3, 2], &[0.0; 6])?,
				&oa::Matrix::new_from_slice([3], &TARGETS)?
			)
			.is_err()
	);
	assert!(
		module
			.forward(&input, &oa::Matrix::new_from_slice([2], &[0_i32, 1])?)
			.is_err()
	);
	// Cutoff contracts: empty, unordered, out of range.
	assert!(AdaptiveLogSoftmaxWithLoss::new(IN_FEATURES, N_CLASSES, &[]).is_err());
	assert!(AdaptiveLogSoftmaxWithLoss::new(IN_FEATURES, N_CLASSES, &[7, 4]).is_err());
	assert!(AdaptiveLogSoftmaxWithLoss::new(IN_FEATURES, N_CLASSES, &[10]).is_err());
	Ok(())
});
