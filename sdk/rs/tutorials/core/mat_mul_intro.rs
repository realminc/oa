use std::{env, hint::black_box, time::Instant};

use anyhow::{Context, bail, ensure};
use oa::{Engine, Matrix, matrix};

const CORRECTNESS_CASES: &[ShapeCase] = &[
	ShapeCase::new("tiny-square", 8, 8, 8),
	ShapeCase::new("square", 128, 128, 128),
	ShapeCase::new("tall-skinny", 512, 64, 128),
	ShapeCase::new("short-wide", 64, 512, 128),
	ShapeCase::new("irregular", 100, 130, 77),
	ShapeCase::new("gemv-decode", 1, 256, 256),
	ShapeCase::new("mnist-hidden", 64, 128, 784),
	ShapeCase::new("mnist-logits", 64, 10, 128),
];

const PERFORMANCE_CASES: &[ShapeCase] = &[
	ShapeCase::new("square-512", 512, 512, 512),
	ShapeCase::new("square-1024", 1024, 1024, 1024),
	ShapeCase::new("square-2048", 2048, 2048, 2048),
	ShapeCase::new("tall-skinny", 4096, 128, 1024),
	ShapeCase::new("short-wide", 128, 4096, 1024),
	ShapeCase::new("gemv-decode", 1, 4096, 4096),
];

#[derive(Clone, Copy)]
struct ShapeCase {
	name: &'static str,
	m: usize,
	n: usize,
	k: usize,
}

impl ShapeCase {
	const fn new(name: &'static str, m: usize, n: usize, k: usize) -> Self {
		Self { name, m, n, k }
	}
}

struct Options {
	correctness: bool,
	benchmark: bool,
	warmup: usize,
	samples: usize,
	case: Option<String>,
}

struct PerfResult {
	p50_ms: f64,
	p95_ms: f64,
	gflops: f64,
}

fn main() -> anyhow::Result<()> {
	let options = parse_options()?;
	let engine = Engine::new().context("create the OA Vulkan engine")?;

	println!("OA Rust Core MatMul Intro — C = A @ Bᵀ");
	println!("contract: FP32 [M,K] × [N,K] -> [M,N]");
	demo_basic_syntax(&engine)?;

	if options.correctness {
		run_correctness_suite(&engine)?;
	}
	if options.benchmark {
		run_performance_suite(&engine, &options)?;
	}
	Ok(())
}

fn parse_options() -> anyhow::Result<Options> {
	let mut options = Options {
		correctness: true,
		benchmark: true,
		warmup: 5,
		samples: 20,
		case: None,
	};
	let mut arguments = env::args().skip(1);
	while let Some(argument) = arguments.next() {
		match argument.as_str() {
			"--correctness-only" => options.benchmark = false,
			"--benchmark-only" => options.correctness = false,
			"--warmup" => options.warmup = parse_count("--warmup", arguments.next())?,
			"--samples" => options.samples = parse_count("--samples", arguments.next())?,
			"--case" => {
				options.case = Some(
					arguments
						.next()
						.context("--case requires a performance-case name")?,
				);
			}
			"--help" | "-h" => {
				println!(
					"usage: core_mat_mul_intro [--correctness-only|--benchmark-only] \
					 [--case NAME] [--warmup N] [--samples N]"
				);
				std::process::exit(0);
			}
			_ => bail!("unknown argument: {argument}"),
		}
	}
	ensure!(options.samples > 0, "--samples must be greater than zero");
	Ok(options)
}

fn parse_count(flag: &str, value: Option<String>) -> anyhow::Result<usize> {
	value
		.with_context(|| format!("{flag} requires a non-negative integer"))?
		.parse()
		.with_context(|| format!("{flag} requires a non-negative integer"))
}

fn demo_basic_syntax(engine: &Engine) -> anyhow::Result<()> {
	println!("\n1. Basic syntax");
	let one = matrix::ones(engine, [2, 3])?;
	let two = matrix::full(engine, [2, 3], 2.0)?;
	let product = matrix::mat_mul_nt(&one, &two)?;
	let values = product.read_f32()?;
	ensure!(product.shape() == [2, 2], "unexpected output shape");
	ensure!(
		values == [6.0; 4],
		"unexpected introductory result: {values:?}"
	);
	println!("   [2,3] × [2,3]ᵀ -> {:?}: {values:?}", product.shape());
	Ok(())
}

fn run_correctness_suite(engine: &Engine) -> anyhow::Result<()> {
	println!("\n2. Correctness vs independent CPU reference");
	for (index, &case) in CORRECTNESS_CASES.iter().enumerate() {
		let left_values = values(case.m * case.k, index * 2 + 3);
		let right_values = values(case.n * case.k, index * 2 + 11);
		let expected = cpu_mat_mul_nt(&left_values, &right_values, case.m, case.n, case.k);
		let left = Matrix::from_f32(engine, [case.m, case.k], &left_values)?;
		let right = Matrix::from_f32(engine, [case.n, case.k], &right_values)?;
		let output = matrix::mat_mul_nt(&left, &right)?;
		let actual = output.read_f32()?;
		let error = normalized_error(&actual, &expected)?;
		ensure!(error <= 1.0e-4, "{} exceeded the FP32 tolerance", case.name);
		println!(
			"   {:<16} [{:5},{:5},{:5}] norm_err={error:.2e} ok",
			case.name, case.m, case.n, case.k
		);
	}
	println!("   All correctness checks passed.");
	Ok(())
}

fn run_performance_suite(engine: &Engine, options: &Options) -> anyhow::Result<()> {
	println!("\n3. Exploratory eager end-to-end performance");
	println!("   boundary: mat_mul_nt call through synchronized full output readback");
	println!(
		"   warmup={} samples={} (one process; not release benchmark evidence)",
		options.warmup, options.samples
	);

	let mut matched = options.case.is_none();
	for &case in PERFORMANCE_CASES {
		if options
			.case
			.as_deref()
			.is_some_and(|name| name != case.name)
		{
			continue;
		}
		matched = true;
		let result = bench_one(engine, case, options.warmup, options.samples)?;
		println!(
			"   {:<16} [{:5},{:5},{:5}] p50={:9.3} ms p95={:9.3} ms {:9.2} GFLOP/s",
			case.name, case.m, case.n, case.k, result.p50_ms, result.p95_ms, result.gflops
		);
		println!(
			"BENCH name={} m={} n={} k={} boundary=eager_submit_readback p50_ms={:.6} p95_ms={:.6} gflops={:.6}",
			case.name, case.m, case.n, case.k, result.p50_ms, result.p95_ms, result.gflops
		);
	}
	ensure!(matched, "unknown performance case: {:?}", options.case);
	Ok(())
}

fn bench_one(
	engine: &Engine,
	case: ShapeCase,
	warmup: usize,
	sample_count: usize,
) -> anyhow::Result<PerfResult> {
	let left = matrix::full(engine, [case.m, case.k], 0.25)?;
	let right = matrix::full(engine, [case.n, case.k], 0.5)?;
	for _ in 0..warmup {
		let output = matrix::mat_mul_nt(&left, &right)?;
		black_box(output.read_f32()?);
	}

	let mut samples = Vec::with_capacity(sample_count);
	for _ in 0..sample_count {
		let start = Instant::now();
		let output = matrix::mat_mul_nt(&left, &right)?;
		black_box(output.read_f32()?);
		samples.push(start.elapsed().as_secs_f64() * 1_000.0);
	}
	samples.sort_by(f64::total_cmp);
	let p50_ms = percentile(&samples, 0.50);
	let p95_ms = percentile(&samples, 0.95);
	let gflop = 2.0 * case.m as f64 * case.n as f64 * case.k as f64 / 1.0e9;
	Ok(PerfResult {
		p50_ms,
		p95_ms,
		gflops: gflop / (p50_ms / 1_000.0),
	})
}

fn percentile(samples: &[f64], percentile: f64) -> f64 {
	let index = ((percentile * samples.len() as f64).ceil() as usize)
		.saturating_sub(1)
		.min(samples.len() - 1);
	samples[index]
}

fn values(length: usize, salt: usize) -> Vec<f32> {
	(0..length)
		.map(|index| (((index * 17 + salt) % 29) as f32 - 14.0) / 7.0)
		.collect()
}

fn cpu_mat_mul_nt(left: &[f32], right: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
	let mut output = vec![0.0; m * n];
	for row in 0..m {
		for column in 0..n {
			for inner in 0..k {
				output[row * n + column] += left[row * k + inner] * right[column * k + inner];
			}
		}
	}
	output
}

fn normalized_error(actual: &[f32], expected: &[f32]) -> anyhow::Result<f64> {
	ensure!(
		actual.len() == expected.len(),
		"output length differs from CPU oracle"
	);
	let mut max_error = 0.0_f64;
	let mut max_reference = 1.0e-6_f64;
	for (&actual, &expected) in actual.iter().zip(expected) {
		max_error = max_error.max((f64::from(actual) - f64::from(expected)).abs());
		max_reference = max_reference.max(f64::from(expected).abs());
	}
	Ok(max_error / max_reference)
}
