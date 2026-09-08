use std::{env, time::Instant};

use anyhow::{Context, bail, ensure};
use oa::{DeviceSelection, Engine, matrix};

const ROUTE: &str = "matrix_mat_mul_nt_tiled_f32";

struct Options {
	name: String,
	m: usize,
	n: usize,
	k: usize,
	device: usize,
	warmup: usize,
	samples: usize,
}

fn main() -> anyhow::Result<()> {
	let options = parse_options()?;
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(options.device))
		.build()
		.context("create the OA Vulkan engine")?;
	let left = matrix::full(&engine, [options.m, options.k], 0.25)?;
	let right = matrix::full(&engine, [options.n, options.k], 0.5)?;
	let (plan, output) = engine.capture(|| matrix::mat_mul_nt(&left, &right))?;

	engine.submit(&plan)?.wait()?;
	let expected = options.k as f32 * 0.125;
	let actual = output.read_f32()?;
	let max_error = actual
		.iter()
		.map(|value| (value - expected).abs())
		.fold(0.0_f32, f32::max);
	let tolerance = expected.abs().max(1.0) * 1.0e-5;
	ensure!(
		max_error <= tolerance,
		"independent constant-input oracle failed: max_error={max_error} tolerance={tolerance}"
	);

	for _ in 0..options.warmup {
		engine.submit_timed(&plan)?.device_duration()?;
	}

	let mut gpu_ms = Vec::with_capacity(options.samples);
	let mut wall_ms = Vec::with_capacity(options.samples);
	for index in 0..options.samples {
		let started = Instant::now();
		let event = engine.submit_timed(&plan)?;
		let gpu = event.device_duration()?.as_secs_f64() * 1_000.0;
		let wall = started.elapsed().as_secs_f64() * 1_000.0;
		ensure!(
			gpu.is_finite() && gpu > 0.0,
			"device duration is not positive"
		);
		ensure!(
			wall.is_finite() && wall > 0.0,
			"wall duration is not positive"
		);
		println!("OARS_SAMPLE index={index} gpu_ms={gpu:.9} synchronized_wall_ms={wall:.9}");
		gpu_ms.push(gpu);
		wall_ms.push(wall);
	}

	let gpu = Statistics::new(&gpu_ms)?;
	let wall = Statistics::new(&wall_ms)?;
	let operations = checked_flop_count(options.m, options.n, options.k)?;
	let gflops = operations / (gpu.median / 1_000.0) / 1.0e9;
	println!(
		"OARS_BENCH schema=oars.matmul.v1 name={} dtype=f32 m={} n={} k={} \
		 route={} warmup={} samples={} correctness=pass unexpected_fallbacks=0 \
		 gpu_median_ms={:.9} gpu_mad_ms={:.9} gpu_p95_ms={:.9} \
		 synchronized_wall_median_ms={:.9} synchronized_wall_p95_ms={:.9} gflops={:.6}",
		options.name,
		options.m,
		options.n,
		options.k,
		ROUTE,
		options.warmup,
		options.samples,
		gpu.median,
		gpu.mad,
		gpu.p95,
		wall.median,
		wall.p95,
		gflops,
	);
	Ok(())
}

fn parse_options() -> anyhow::Result<Options> {
	let mut options = Options {
		name: "square-1024".to_owned(),
		m: 1024,
		n: 1024,
		k: 1024,
		device: 0,
		warmup: 5,
		samples: 20,
	};
	let mut arguments = env::args().skip(1);
	while let Some(argument) = arguments.next() {
		match argument.as_str() {
			"--name" => options.name = parse_string("--name", arguments.next())?,
			"--m" => options.m = parse_count("--m", arguments.next())?,
			"--n" => options.n = parse_count("--n", arguments.next())?,
			"--k" => options.k = parse_count("--k", arguments.next())?,
			"--device" => options.device = parse_count("--device", arguments.next())?,
			"--warmup" => options.warmup = parse_count("--warmup", arguments.next())?,
			"--samples" => options.samples = parse_count("--samples", arguments.next())?,
			"--help" | "-h" => {
				println!(
					"usage: core_mat_mul_bench [--name NAME] [--m M] [--n N] [--k K] \
					 [--device INDEX] [--warmup N] [--samples N]"
				);
				std::process::exit(0);
			}
			_ => bail!("unknown argument: {argument}"),
		}
	}
	ensure!(!options.name.is_empty(), "--name must not be empty");
	ensure!(
		options.m > 0 && options.n > 0 && options.k > 0,
		"matrix dimensions must be greater than zero"
	);
	ensure!(options.samples > 0, "--samples must be greater than zero");
	checked_element_count(options.m, options.k, "left")?;
	checked_element_count(options.n, options.k, "right")?;
	checked_element_count(options.m, options.n, "output")?;
	Ok(options)
}

fn parse_string(flag: &str, value: Option<String>) -> anyhow::Result<String> {
	value.with_context(|| format!("{flag} requires a value"))
}

fn parse_count(flag: &str, value: Option<String>) -> anyhow::Result<usize> {
	value
		.with_context(|| format!("{flag} requires a non-negative integer"))?
		.parse()
		.with_context(|| format!("{flag} requires a non-negative integer"))
}

fn checked_element_count(left: usize, right: usize, label: &str) -> anyhow::Result<usize> {
	left.checked_mul(right)
		.with_context(|| format!("{label} element count overflows usize"))
}

fn checked_flop_count(m: usize, n: usize, k: usize) -> anyhow::Result<f64> {
	let elements = checked_element_count(m, n, "output")?;
	let multiply_accumulates = checked_element_count(elements, k, "operation")?;
	let operations = multiply_accumulates
		.checked_mul(2)
		.context("operation count overflows usize")?;
	Ok(operations as f64)
}

struct Statistics {
	median: f64,
	mad: f64,
	p95: f64,
}

impl Statistics {
	fn new(samples: &[f64]) -> anyhow::Result<Self> {
		ensure!(
			!samples.is_empty(),
			"statistics require at least one sample"
		);
		ensure!(
			samples.iter().all(|sample| sample.is_finite()),
			"statistics require finite samples"
		);
		let mut ordered = samples.to_vec();
		ordered.sort_by(f64::total_cmp);
		let median = percentile(&ordered, 0.50);
		let mut deviations = ordered
			.iter()
			.map(|sample| (sample - median).abs())
			.collect::<Vec<_>>();
		deviations.sort_by(f64::total_cmp);
		Ok(Self {
			median,
			mad: percentile(&deviations, 0.50),
			p95: percentile(&ordered, 0.95),
		})
	}
}

fn percentile(ordered: &[f64], fraction: f64) -> f64 {
	let position = (ordered.len() - 1) as f64 * fraction;
	let lower = position.floor() as usize;
	let upper = position.ceil() as usize;
	let weight = position - lower as f64;
	ordered[lower] * (1.0 - weight) + ordered[upper] * weight
}
