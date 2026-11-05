//! `oa::GemmTuner` — measured GEMM candidate qualification and route-cache
//! publication. This is the donor's `runtime/gemm/tuner.cpp` ported into the
//! device-gated qualification harness: the donor's private class is invoked
//! from `benchGemmVariants` test binaries, and Rust test code is the only
//! consumer that can reach the private planner, launch, and publication
//! machinery. Production code reads the cache but never writes it.
//!
//! The tuner measures candidates through the exact production path —
//! `plan` with a required-variant pin, `lower_mat_mul_nt_plan`, and
//! `record_semantic` + `checkpoint_timed` — and publishes a winner only
//! after the deterministic host oracle re-validates its numerics. Timing
//! alone never authorizes route publication.

use super::{
	GemmEpilogue, GemmKernel, MatmulPreference, MatmulProblem, MatmulRouteValue, MatmulVariant,
	VARIANTS, calculate_workspace_bytes, find_variant, lower_mat_mul_nt_plan, plan,
	resolve_precision, route_key, variant_legal,
};
use crate::runtime::{ComputeDispatch, EngineHandle, LogComponent, SemanticDispatch};
use crate::{Engine, Error, Result};

/// Sentinel returned by `benchmark_candidate` for any failed or invalid
/// measurement — the donor's `kInvalidMeasurementMs`.
const INVALID_MEASUREMENT_MS: f32 = 1.0e9;

/// Below this output size the oracle validates every element; above it the
/// donor probes tile edges plus deterministic interior coordinates.
const FULL_VALIDATION_OUTPUT_LIMIT: usize = 4096;

/// One benchmarked shape — the donor's `GemmTunerShape`.
#[derive(Clone, Copy, Debug)]
pub(super) struct GemmTunerShape {
	pub m: u32,
	pub n: u32,
	pub k: u32,
	pub name: &'static str,
	pub epilogue: GemmEpilogue,
	pub max_workspace_bytes: u64,
}

/// Measured result for one candidate — the donor's
/// `GemmTunerCandidateResult`.
#[derive(Clone, Copy, Debug)]
pub(super) struct GemmTunerCandidateResult {
	pub variant: u64,
	pub kernel: GemmKernel,
	pub name: &'static str,
	pub median_time_ms: f32,
	pub p95_time_ms: f32,
	pub sample_count: u32,
}

/// Benchmark result for one shape — the donor's `GemmTunerResult`. The
/// ranked list deliberately keeps the complete legal ranking, fastest
/// first, so diagnostics can compare a stable second choice without
/// re-running an exhaustive tune.
#[derive(Debug)]
pub(super) struct GemmTunerResult {
	pub shape_name: &'static str,
	pub m: u32,
	pub n: u32,
	pub k: u32,
	pub best_variant: u64,
	pub best_kernel: GemmKernel,
	pub best_time_ms: f32,
	pub best_gflops: f32,
	pub ranked_candidates: Vec<GemmTunerCandidateResult>,
}

/// Default shape suite — the donor's `kDefaultShapes` table of common
/// training and inference patterns.
const DEFAULT_SHAPES: &[GemmTunerShape] = &[
	// Training shapes (batch × sequence × hidden patterns)
	GemmTunerShape {
		m: 64,
		n: 256,
		k: 128,
		name: "train_tiny",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 256,
		n: 256,
		k: 256,
		name: "train_atom",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 512,
		n: 512,
		k: 512,
		name: "train_small",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 1024,
		n: 1024,
		k: 1024,
		name: "train_base",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 1024,
		n: 32,
		k: 32,
		name: "nlp_qkv_bias",
		epilogue: GemmEpilogue::Bias,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 1024,
		n: 64,
		k: 32,
		name: "nlp_ffn1_bias_gelu",
		epilogue: GemmEpilogue::BiasGelu,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 1024,
		n: 32,
		k: 64,
		name: "nlp_ffn2_bias",
		epilogue: GemmEpilogue::Bias,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 4096,
		n: 384,
		k: 384,
		name: "alm_qkv",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 4096,
		n: 1536,
		k: 384,
		name: "alm_ffn1",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 4096,
		n: 384,
		k: 1536,
		name: "alm_ffn2",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	// Irregular tall-skinny and short-wide occupancy cases
	GemmTunerShape {
		m: 1000,
		n: 256,
		k: 512,
		name: "irregular_tall",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 500,
		n: 512,
		k: 256,
		name: "irregular_wide",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 750,
		n: 1024,
		k: 512,
		name: "irregular_med",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	// Inference decode (M=1 cooperative-vector candidates)
	GemmTunerShape {
		m: 1,
		n: 256,
		k: 256,
		name: "decode_tiny",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 1,
		n: 512,
		k: 512,
		name: "decode_small",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 1,
		n: 1024,
		k: 1024,
		name: "decode_base",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 1,
		n: 2048,
		k: 2048,
		name: "decode_large",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	// Small-batch inference
	GemmTunerShape {
		m: 4,
		n: 1024,
		k: 1024,
		name: "batch4_base",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 8,
		n: 512,
		k: 512,
		name: "batch8_small",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
	GemmTunerShape {
		m: 16,
		n: 256,
		k: 256,
		name: "batch16_tiny",
		epilogue: GemmEpilogue::None,
		max_workspace_bytes: 0,
	},
];

/// The donor's `getDefaultShapes`.
pub(super) fn default_shapes() -> &'static [GemmTunerShape] {
	DEFAULT_SHAPES
}

/// Host-side epilogue application for the oracle — the donor's
/// `applyEpilogue`.
fn apply_epilogue(value: f32, epilogue: GemmEpilogue) -> f32 {
	match epilogue {
		GemmEpilogue::None | GemmEpilogue::Bias => value,
		GemmEpilogue::BiasRelu => value.max(0.0),
		GemmEpilogue::BiasGelu => {
			let x3 = value * value * value;
			0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * x3)).tanh())
		}
		GemmEpilogue::BiasSilu => value / (1.0 + (-value).exp()),
		GemmEpilogue::SiluDual => f32::NAN,
	}
}

/// Independent host oracle — the donor's `validateNumericalOutput`.
///
/// Small problems validate every output; large problems validate tile
/// edges plus 256 reproducible interior coordinates.
pub(super) fn validate_numerical_output(
	shape: &GemmTunerShape,
	a: &[f32],
	b: &[f32],
	bias: &[f32],
	output: &[f32],
) -> Result<()> {
	let a_count = usize::try_from(u64::from(shape.m) * u64::from(shape.k))
		.map_err(|_| Error::invalid_argument("gemm tuner oracle: unbounded A extent"))?;
	let b_count = usize::try_from(u64::from(shape.n) * u64::from(shape.k))
		.map_err(|_| Error::invalid_argument("gemm tuner oracle: unbounded B extent"))?;
	let output_count = usize::try_from(u64::from(shape.m) * u64::from(shape.n))
		.map_err(|_| Error::invalid_argument("gemm tuner oracle: unbounded output extent"))?;
	if shape.m == 0 || shape.n == 0 || shape.k == 0 {
		return Err(Error::invalid_argument(
			"gemm tuner oracle requires non-zero bounded dimensions",
		));
	}
	if a.len() != a_count || b.len() != b_count || output.len() != output_count {
		return Err(Error::invalid_argument(
			"gemm tuner oracle buffer sizes do not match the shape",
		));
	}
	if shape.epilogue == GemmEpilogue::SiluDual {
		return Err(Error::invalid_argument(
			"gemm tuner oracle: dual-output SiLU requires a two-output validation contract",
		));
	}
	let has_bias = shape.epilogue != GemmEpilogue::None;
	if (has_bias && bias.len() != shape.n as usize) || (!has_bias && !bias.is_empty()) {
		return Err(Error::invalid_argument(
			"gemm tuner oracle bias contract does not match the epilogue",
		));
	}

	let validate_at = |row: u32, col: u32| -> Result<()> {
		let mut expected = if has_bias { bias[col as usize] } else { 0.0 };
		for index in 0..shape.k {
			expected += a[row as usize * shape.k as usize + index as usize]
				* b[col as usize * shape.k as usize + index as usize];
		}
		expected = apply_epilogue(expected, shape.epilogue);
		let actual = output[row as usize * shape.n as usize + col as usize];
		let tolerance = 2.0e-4 + 4.0e-6 * shape.k as f32 + 2.0e-4 * expected.abs();
		if !expected.is_finite() || !actual.is_finite() || (actual - expected).abs() > tolerance {
			return Err(Error::data_loss(format!(
				"gemm tuner numerical validation failed at row={row} col={col} \
				 expected={expected} actual={actual} tolerance={tolerance}"
			)));
		}
		Ok(())
	};

	if output_count <= FULL_VALIDATION_OUTPUT_LIMIT {
		for row in 0..shape.m {
			for col in 0..shape.n {
				validate_at(row, col)?;
			}
		}
		return Ok(());
	}

	// Always probe tile boundaries and tails, then add reproducible
	// interior coordinates so product-sized tune shapes stay bounded while
	// covering the regions most likely to expose dispatch/tail defects.
	let rows = [
		0,
		1.min(shape.m - 1),
		shape.m / 2,
		shape.m.saturating_sub(2),
		shape.m - 1,
	];
	let cols = [
		0,
		1.min(shape.n - 1),
		shape.n / 2,
		shape.n.saturating_sub(2),
		shape.n - 1,
	];
	for row in rows {
		for col in cols {
			validate_at(row, col)?;
		}
	}
	let mut state =
		0x9e37_79b9_u32 ^ shape.m ^ (shape.n << 7) ^ (shape.k << 15) ^ shape.epilogue as u32;
	for _ in 0..256 {
		state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
		let row = state % shape.m;
		state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
		let col = state % shape.n;
		validate_at(row, col)?;
	}
	Ok(())
}

/// Deterministic validation inputs — the donor's exact generator formulas,
/// chosen so every value is exactly representable in F32.
fn validation_data(
	count: usize,
	modulus: u32,
	multiplier: u32,
	offset: i32,
	scale: f32,
) -> Vec<f32> {
	(0..count)
		.map(|index| (((index as u32 % modulus) * multiplier % modulus) as i32 - offset) as f32 * scale)
		.collect()
}

/// Collect legal candidates for one exact production contract — the donor's
/// `collectCandidates`, including its sweep skip rules: Naive is excluded
/// for non-trivial products (it cannot win and wastes tuning time), and the
/// strided fallback families are excluded because the canonical tune problem
/// can never select them.
fn collect_candidates(
	contract: &super::MatmulDeviceContract,
	problem: &MatmulProblem,
	max_workspace_bytes: u64,
) -> Vec<&'static MatmulVariant> {
	let precision = resolve_precision(problem);
	VARIANTS
		.iter()
		.filter(|variant| {
			if variant.epilogue != problem.epilogue {
				return false;
			}
			if calculate_workspace_bytes(variant, problem) > max_workspace_bytes {
				return false;
			}
			if variant.kernel == GemmKernel::Naive && u64::from(problem.m) * u64::from(problem.n) >= 64 {
				return false;
			}
			if matches!(
				variant.kernel,
				GemmKernel::StridedF32 | GemmKernel::StridedTiledF32
			) {
				return false;
			}
			variant_legal(contract, variant, problem, precision).is_none()
		})
		.collect()
}

/// Benchmark one candidate through the same plan → launch → semantic
/// dispatch path production records — the donor's `benchmarkCandidate`.
///
/// Warmup submissions run untimed so pipeline creation and clock settling
/// stay outside the samples; each timed sample is one record, one timed
/// checkpoint, and one device-timestamp duration. Returns mean GPU
/// milliseconds or the invalid-measurement sentinel.
fn benchmark_candidate(
	engine: &Engine,
	problem: &MatmulProblem,
	variant_id: u64,
	max_workspace_bytes: u64,
	warm_iterations: u32,
	bench_iterations: u32,
) -> f32 {
	if bench_iterations == 0 {
		return INVALID_MEASUREMENT_MS;
	}
	let (m, n, k) = (problem.m, problem.n, problem.k);
	let a_data = validation_data(m as usize * k as usize, 29, 17, 14, 0.031_25);
	let b_data = validation_data(n as usize * k as usize, 23, 11, 11, 0.025);
	let bias_data = if problem.epilogue == GemmEpilogue::None {
		Vec::new()
	} else {
		validation_data(n as usize, 13, 7, 6, 0.02)
	};
	let (Ok(a), Ok(b)) = (
		crate::matrix::from_f32_on(engine, [m as usize, k as usize], &a_data),
		crate::matrix::from_f32_on(engine, [n as usize, k as usize], &b_data),
	) else {
		return INVALID_MEASUREMENT_MS;
	};
	let Ok(output) = crate::Matrix::allocate(
		&engine.handle(),
		vec![m as usize, n as usize],
		m as usize * n as usize,
		crate::DType::F32,
	) else {
		return INVALID_MEASUREMENT_MS;
	};
	let bias = if bias_data.is_empty() {
		None
	} else {
		match crate::matrix::from_f32_on(engine, [n as usize], &bias_data) {
			Ok(bias) => Some(bias),
			Err(_) => return INVALID_MEASUREMENT_MS,
		}
	};

	let pinned = MatmulPreference {
		required_variant: variant_id,
		max_workspace_bytes,
		..MatmulPreference::default()
	};
	let contract = engine.handle().matmul_device_contract();
	let Ok(candidate_plan) = plan(&contract, problem, &pinned, None) else {
		return INVALID_MEASUREMENT_MS;
	};
	let Ok(launch) = lower_mat_mul_nt_plan(&candidate_plan, problem) else {
		return INVALID_MEASUREMENT_MS;
	};
	// Problem-buffer slots [A, B, C, workspace, epilogue]; unused slots repeat
	// the output so the epilogue launch can bind bias at index 4.
	let epilogue_storage = bias
		.as_ref()
		.map_or_else(|| output.storage(), |bias| bias.storage());
	let problem_buffers = [
		a.storage(),
		b.storage(),
		output.storage(),
		output.storage(),
		epilogue_storage,
	];
	let Ok(buffers) = launch.bindings(&problem_buffers) else {
		return INVALID_MEASUREMENT_MS;
	};

	let record_once = || -> Result<()> {
		engine.handle().record_semantic(
			ComputeDispatch {
				kernel: launch.kernel_id,
				buffers: &buffers,
				push_constants: &launch.push_constants[..launch.push_count],
				workgroups: launch.grid,
				indirect: None,
			},
			SemanticDispatch {
				contract: crate::core::operation::matrix::MAT_MUL_NT,
				inputs: &[&a, &b],
				outputs: &[&output],
				attributes: &[],
			},
		)
	};

	for _ in 0..warm_iterations {
		if record_once()
			.and_then(|()| engine.checkpoint().and_then(|event| event.wait()))
			.is_err()
		{
			return INVALID_MEASUREMENT_MS;
		}
	}

	let mut total_gpu_ms = 0.0_f64;
	for _ in 0..bench_iterations {
		let Ok(event) = record_once().and_then(|()| engine.checkpoint_timed()) else {
			return INVALID_MEASUREMENT_MS;
		};
		let Ok(duration) = event.device_duration() else {
			return INVALID_MEASUREMENT_MS;
		};
		let sample_ms = duration.as_secs_f64() * 1_000.0;
		if !sample_ms.is_finite() || sample_ms <= 0.0 {
			return INVALID_MEASUREMENT_MS;
		}
		total_gpu_ms += sample_ms;
	}
	(total_gpu_ms / f64::from(bench_iterations)) as f32
}

/// Re-run one candidate on deterministic data and compare against the host
/// oracle — the donor's `validateCandidate`. A mismatch leaves the prior
/// route cache untouched.
fn validate_candidate(
	engine: &Engine,
	problem: &MatmulProblem,
	variant_id: u64,
	shape: &GemmTunerShape,
	max_workspace_bytes: u64,
) -> Result<()> {
	let (m, n, k) = (problem.m, problem.n, problem.k);
	let a_data = validation_data(m as usize * k as usize, 29, 17, 14, 0.031_25);
	let b_data = validation_data(n as usize * k as usize, 23, 11, 11, 0.025);
	let bias_data = if problem.epilogue == GemmEpilogue::None {
		Vec::new()
	} else {
		validation_data(n as usize, 13, 7, 6, 0.02)
	};
	let a = crate::matrix::from_f32_on(engine, [m as usize, k as usize], &a_data)?;
	let b = crate::matrix::from_f32_on(engine, [n as usize, k as usize], &b_data)?;
	let bias = if bias_data.is_empty() {
		None
	} else {
		Some(crate::matrix::from_f32_on(
			engine,
			[n as usize],
			&bias_data,
		)?)
	};
	let output = crate::Matrix::allocate(
		&engine.handle(),
		vec![m as usize, n as usize],
		m as usize * n as usize,
		crate::DType::F32,
	)?;

	let pinned = MatmulPreference {
		required_variant: variant_id,
		max_workspace_bytes,
		..MatmulPreference::default()
	};
	let contract = engine.handle().matmul_device_contract();
	let candidate_plan = plan(&contract, problem, &pinned, None)?;
	let launch = lower_mat_mul_nt_plan(&candidate_plan, problem)?;
	// Problem-buffer slots [A, B, C, workspace, epilogue]; unused slots repeat
	// the output so the epilogue launch can bind bias at index 4.
	let epilogue_storage = bias
		.as_ref()
		.map_or_else(|| output.storage(), |bias| bias.storage());
	let buffers = launch.bindings(&[
		a.storage(),
		b.storage(),
		output.storage(),
		output.storage(),
		epilogue_storage,
	])?;
	engine.handle().record_semantic(
		ComputeDispatch {
			kernel: launch.kernel_id,
			buffers: &buffers,
			push_constants: &launch.push_constants[..launch.push_count],
			workgroups: launch.grid,
			indirect: None,
		},
		SemanticDispatch {
			contract: crate::core::operation::matrix::MAT_MUL_NT,
			inputs: &[&a, &b],
			outputs: &[&output],
			attributes: &[],
		},
	)?;
	engine.checkpoint()?.wait()?;
	let output_data = output.read_f32()?;
	validate_numerical_output(shape, &a_data, &b_data, &bias_data, &output_data)
}

/// Benchmark one shape across every legal candidate — the donor's
/// `benchmarkShape`. Candidates are measured in alternating forward/reverse
/// blocks so no candidate systematically sees a more favorable clock or
/// thermal state; the median block mean rejects a single scheduler
/// excursion without hiding stable changes. The measured winner is
/// published only after the deterministic host oracle confirms it.
pub(super) fn benchmark_shape(
	engine: &Engine,
	shape: &GemmTunerShape,
	warm_iterations: u32,
	bench_iterations: u32,
) -> Result<GemmTunerResult> {
	if bench_iterations == 0 {
		return Err(Error::invalid_argument(
			"gemm tuner requires at least one benchmark iteration",
		));
	}
	let mut problem = MatmulProblem::contiguous(shape.m, shape.n, shape.k);
	problem.epilogue = shape.epilogue;

	let contract = engine.handle().matmul_device_contract();
	let candidates = collect_candidates(&contract, &problem, shape.max_workspace_bytes);
	if candidates.is_empty() {
		return Err(Error::failed_precondition(format!(
			"no legal matmul candidate for {}x{}x{} on this device",
			shape.m, shape.n, shape.k
		)));
	}

	let block_count = 4.min(bench_iterations);
	let iterations_per_block = bench_iterations.div_ceil(block_count);
	let mut block_means: Vec<Vec<f32>> = vec![Vec::new(); candidates.len()];
	for block in 0..block_count {
		for order in 0..candidates.len() {
			let index = if block % 2 == 0 {
				order
			} else {
				candidates.len() - 1 - order
			};
			block_means[index].push(benchmark_candidate(
				engine,
				&problem,
				candidates[index].id,
				shape.max_workspace_bytes,
				warm_iterations,
				iterations_per_block,
			));
		}
	}

	let mut ranked_candidates = Vec::with_capacity(candidates.len());
	let mut best: Option<(u64, GemmKernel, &'static str, f32, f32, u32)> = None;
	for (index, candidate) in candidates.iter().enumerate() {
		let mut samples: Vec<f32> = block_means[index]
			.iter()
			.copied()
			.filter(|sample| sample.is_finite() && *sample > 0.0 && *sample < INVALID_MEASUREMENT_MS)
			.collect();
		if samples.is_empty() {
			crate::log_warn!(
				LogComponent::COMPUTE,
				"gemm tuner candidate {}: rejected (no valid GPU timing blocks)",
				candidate.name
			);
			continue;
		}
		samples.sort_by(f32::total_cmp);
		let middle = samples.len() / 2;
		let median = if samples.len() % 2 == 1 {
			samples[middle]
		} else {
			0.5 * (samples[middle - 1] + samples[middle])
		};
		let p95_index = (0.95 * samples.len() as f64).ceil() as usize;
		let p95 = samples[p95_index.saturating_sub(1).min(samples.len() - 1)];
		ranked_candidates.push(GemmTunerCandidateResult {
			variant: candidate.id,
			kernel: candidate.kernel,
			name: candidate.name,
			median_time_ms: median,
			p95_time_ms: p95,
			sample_count: samples.len() as u32,
		});

		let flops = 2.0 * f64::from(shape.m) * f64::from(shape.n) * f64::from(shape.k);
		let gflops = flops / (f64::from(median) * 1.0e-3) / 1.0e9;
		crate::log_info!(
			LogComponent::COMPUTE,
			"gemm tuner candidate {}: {median:.4} ms ({gflops:.1} GFLOP/s, median of {} blocks)",
			candidate.name,
			samples.len()
		);

		if best.is_none_or(|(_, _, _, best_ms, _, _)| median < best_ms) {
			best = Some((
				candidate.id,
				candidate.kernel,
				candidate.name,
				median,
				p95,
				samples.len() as u32,
			));
		}
	}
	ranked_candidates.sort_by(|a, b| a.median_time_ms.total_cmp(&b.median_time_ms));

	let Some((best_variant, best_kernel, best_name, best_ms, best_p95, best_samples)) = best else {
		return Err(Error::failed_precondition(
			"gemm tuner: every legal candidate failed GPU timing",
		));
	};

	// Timing alone never authorizes route publication: the exact winner is
	// re-run on deterministic data and must match the independent oracle.
	if let Err(error) = validate_candidate(
		engine,
		&problem,
		best_variant,
		shape,
		shape.max_workspace_bytes,
	) {
		crate::log_error!(
			LogComponent::COMPUTE,
			"gemm tuner rejected winner {best_name} for {}x{}x{}: {error}",
			shape.m,
			shape.n,
			shape.k
		);
		return Err(error);
	}

	record_measured_matmul_route(
		&engine.handle(),
		&problem,
		MatmulRouteValue {
			winner_variant: best_variant,
			median_gpu_time_ms: best_ms,
			p95_gpu_time_ms: best_p95,
			sample_count: best_samples,
			last_updated_step: 0,
		},
	)?;
	crate::log_info!(
		LogComponent::COMPUTE,
		"gemm tuner winner for {}x{}x{} epilogue={:?}: {best_name} ({best_ms:.4} ms) -> route cache",
		shape.m,
		shape.n,
		shape.k,
		problem.epilogue
	);

	let flops = 2.0 * f64::from(shape.m) * f64::from(shape.n) * f64::from(shape.k);
	Ok(GemmTunerResult {
		shape_name: shape.name,
		m: shape.m,
		n: shape.n,
		k: shape.k,
		best_variant,
		best_kernel,
		best_time_ms: best_ms,
		best_gflops: (flops / (f64::from(best_ms) * 1.0e-3) / 1.0e9) as f32,
		ranked_candidates,
	})
}

/// Run the default shape suite and publish every qualified winner — the
/// donor's `GemmTuner::run`. Failed shapes are logged and skipped; the
/// cache is explicitly persisted after the sweep, matching the donor's
/// `saveCache` call.
pub(super) fn run(
	engine: &Engine,
	warm_iterations: u32,
	bench_iterations: u32,
) -> Result<Vec<GemmTunerResult>> {
	crate::log_info!(
		LogComponent::COMPUTE,
		"gemm tuner: starting benchmark suite"
	);
	let mut results = Vec::new();
	for shape in default_shapes() {
		match benchmark_shape(engine, shape, warm_iterations, bench_iterations) {
			Ok(result) => {
				crate::log_info!(
					LogComponent::COMPUTE,
					"  [{}] {}x{}x{}: {:.3} ms, {:.1} GFLOPS",
					result.shape_name,
					result.m,
					result.n,
					result.k,
					result.best_time_ms,
					result.best_gflops
				);
				results.push(result);
			}
			Err(error) => {
				crate::log_warn!(
					LogComponent::COMPUTE,
					"  [{}] {}x{}x{}: FAILED - {error}",
					shape.name,
					shape.m,
					shape.n,
					shape.k
				);
			}
		}
	}
	engine.handle().save_matmul_route_cache()?;
	crate::log_info!(
		LogComponent::COMPUTE,
		"gemm tuner: benchmark complete ({} shapes qualified)",
		results.len()
	);
	Ok(results)
}

#[test]
#[ignore = "requires a Vulkan device; pinned SmallM numerical qualification"]
fn small_m_pinned_candidate_matches_host_oracle() {
	let engine = Engine::new().expect("SmallM qualification requires an admitted Vulkan device");
	for (m, n, k) in [
		(1, 1, 1),
		(1, 65, 33),
		(2, 17, 257),
		(4, 70, 17),
		(1, 512, 512),
	] {
		let shape = GemmTunerShape {
			m,
			n,
			k,
			name: "small_m_oracle",
			epilogue: GemmEpilogue::None,
			max_workspace_bytes: 0,
		};
		validate_candidate(
			&engine,
			&MatmulProblem::contiguous(m, n, k),
			super::matmul_variant_id_from_name("GemmSmallM"),
			&shape,
			0,
		)
		.unwrap();
	}
}

/// Record one measured qualification result into the engine's route cache.
///
/// The winner must be registered and legal for the complete problem on this
/// device — measured evidence never admits an illegal route.
fn record_measured_matmul_route(
	engine: &EngineHandle,
	problem: &MatmulProblem,
	value: MatmulRouteValue,
) -> Result<()> {
	let contract = engine.matmul_device_contract();
	let variant = find_variant(value.winner_variant).ok_or_else(|| {
		Error::invalid_argument(format!(
			"measured matmul route records an unregistered variant {:#x}",
			value.winner_variant
		))
	})?;
	let precision = resolve_precision(problem);
	if let Some(rejection) = variant_legal(&contract, variant, problem, precision) {
		return Err(Error::invalid_argument(format!(
			"measured matmul route {} is not legal on this device: {rejection}",
			variant.name
		)));
	}
	engine.record_matmul_route(route_key(&contract, problem), value);
	Ok(())
}
