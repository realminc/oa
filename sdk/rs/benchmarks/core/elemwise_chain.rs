use std::{env, time::Instant};

use anyhow::{Context, bail, ensure};
use oa::{DeviceSelection, Engine, matrix, ml};

struct Options {
	name: String,
	elements: usize,
	chain: usize,
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
	let x = matrix::full(&engine, [options.elements], 0.25)?;
	let y = matrix::full(&engine, [options.elements], 0.5)?;
	let z = matrix::full(&engine, [options.elements], 0.75)?;
	let (plan, output) = engine.capture(|| {
		let mut value = matrix::mul(&x, &y)?;
		for _ in 1..options.chain {
			value = ml::matrix::gelu(&matrix::scale(&matrix::add(&value, &z)?, 1.25)?)?;
		}
		Ok(value)
	})?;

	let diagnostics = plan.diagnostics();
	let fused_nodes = diagnostics.node_count();
	println!(
		"OARS_FUSION nodes={fused_nodes} fused_nodes={} fused_ops={}",
		diagnostics.semantic_fused_node_count(),
		diagnostics.semantic_fused_operation_count(),
	);

	engine.submit(&plan)?.wait()?;
	let expected = cpu_reference(options.chain);
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
	println!(
		"OARS_BENCH schema=oars.elemwise_chain.v1 name={} dtype=f32 elements={} \
		 chain={} nodes={fused_nodes} warmup={} samples={} correctness=pass \
		 gpu_median_ms={:.9} gpu_mad_ms={:.9} gpu_p95_ms={:.9} \
		 synchronized_wall_median_ms={:.9} synchronized_wall_p95_ms={:.9}",
		options.name,
		options.elements,
		options.chain,
		options.warmup,
		options.samples,
		gpu.median,
		gpu.mad,
		gpu.p95,
		wall.median,
		wall.p95,
	);
	Ok(())
}

/// CPU oracle for `mul(x,y)` followed by `chain - 1` rounds of
/// `gelu(scale(add(v, z), 1.25))` over constant inputs.
fn cpu_reference(chain: usize) -> f32 {
	let mut value = 0.25_f64 * 0.5;
	for _ in 1..chain {
		value = gelu_ref((value + 0.75) * 1.25);
	}
	value as f32
}

fn gelu_ref(x: f64) -> f64 {
	0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())
}

fn parse_options() -> anyhow::Result<Options> {
	let mut options = Options {
		name: "chain-4x1m".to_owned(),
		elements: 1_048_576,
		chain: 4,
		device: 0,
		warmup: 5,
		samples: 20,
	};
	let mut arguments = env::args().skip(1);
	while let Some(argument) = arguments.next() {
		match argument.as_str() {
			"--name" => options.name = parse_string("--name", arguments.next())?,
			"--elements" => options.elements = parse_count("--elements", arguments.next())?,
			"--chain" => options.chain = parse_count("--chain", arguments.next())?,
			"--device" => options.device = parse_count("--device", arguments.next())?,
			"--warmup" => options.warmup = parse_count("--warmup", arguments.next())?,
			"--samples" => options.samples = parse_count("--samples", arguments.next())?,
			"--help" | "-h" => {
				println!(
					"usage: core_elemwise_chain_bench [--name NAME] [--elements N] \
					 [--chain N] [--device INDEX] [--warmup N] [--samples N]"
				);
				std::process::exit(0);
			}
			_ => bail!("unknown argument: {argument}"),
		}
	}
	ensure!(!options.name.is_empty(), "--name must not be empty");
	ensure!(options.elements > 0, "--elements must be greater than zero");
	ensure!(options.chain >= 2, "--chain must be at least two");
	ensure!(options.samples > 0, "--samples must be greater than zero");
	Ok(options)
}

fn parse_string(flag: &str, value: Option<String>) -> anyhow::Result<String> {
	value.with_context(|| format!("{flag} requires a value"))
}

fn parse_count(flag: &str, value: Option<String>) -> anyhow::Result<usize> {
	value
		.with_context(|| format!("{flag} requires a value"))?
		.parse::<usize>()
		.with_context(|| format!("{flag} requires a positive integer"))
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
		let mut sorted = samples.to_vec();
		sorted.sort_by(f64::total_cmp);
		let median = sorted[sorted.len() / 2];
		let deviations = sorted
			.iter()
			.map(|sample| (sample - median).abs())
			.collect::<Vec<_>>();
		let mut deviations = deviations;
		deviations.sort_by(f64::total_cmp);
		Ok(Self {
			median,
			mad: deviations[deviations.len() / 2],
			p95: sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)],
		})
	}
}
