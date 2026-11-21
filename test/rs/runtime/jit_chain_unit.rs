//! Phase 1 differential oracle and matched replay experiment, explicitly invoked.
//! No graph matcher or production JIT route is introduced by this fixture.

use super::*;
use crate::runtime::graph::executable::QueueFamilies;
use crate::runtime::{
	BufferAccess, PushConstant, Storage,
	device::{
		features::ExecutionProfile,
		logical::command::jit_lab::{ChainContract, PreparedChain},
	},
	dna::{
		compiler::{JitArtifacts, SlangCompiler},
		fusion,
		ir::elementwise::Program,
		kernel::KernelId,
		provider::blas::{lower_mat_mul_nt_reduction, matmul_variant_id_from_name},
	},
	graph::executable::{BufferUse, ComputeNode, ExecutableGraph},
};
use std::time::Instant;

const GUARD_WORDS: usize = 17;
const POISON: u32 = 0x7fa5_b6c7;

struct Fixture {
	inputs: Vec<Storage>,
	outputs: Vec<Storage>,
	graph: ExecutableGraph,
	jit_output: Buffer,
	host: [Vec<f32>; 3],
	count: u32,
}

impl Fixture {
	fn new(device: &Arc<Device>, host: [Vec<f32>; 3]) -> Result<Self> {
		Self::with_recipe(
			device,
			host,
			&std::iter::once(KernelId::MatrixMulF32)
				.chain(
					[
						KernelId::MatrixAddF32,
						KernelId::MatrixScaleF32,
						KernelId::MlGeluF32,
					]
					.repeat(7),
				)
				.collect::<Vec<_>>(),
		)
	}

	fn with_recipe(device: &Arc<Device>, host: [Vec<f32>; 3], recipe: &[KernelId]) -> Result<Self> {
		let count = u32::try_from(host[0].len()).expect("bounded fixture count");
		assert!(host.iter().all(|values| values.len() == count as usize));
		let poison = vec![POISON; count as usize + GUARD_WORDS]
			.iter()
			.flat_map(|word| word.to_le_bytes())
			.collect::<Vec<_>>();
		let mut inputs = Vec::new();
		for values in &host {
			let mut bytes = values
				.iter()
				.flat_map(|value| value.to_le_bytes())
				.collect::<Vec<_>>();
			bytes.extend_from_slice(&poison[count as usize * 4..]);
			inputs.push(Storage::new_from_bytes(device, &bytes)?);
		}
		let mut outputs = Vec::new();
		let mut nodes = Vec::new();
		for &kernel in recipe {
			device.require_kernel(kernel)?;
			let output = Storage::new_from_bytes(device, &poison)?;
			let previous: &Storage = outputs.last().unwrap_or(&inputs[0]);
			let mut buffers = vec![BufferUse {
				buffer: previous.buffer().unwrap().clone(),
				access: BufferAccess::Read,
			}];
			if matches!(kernel, KernelId::MatrixMulF32 | KernelId::MatrixAddF32) {
				buffers.push(BufferUse {
					buffer: inputs[if nodes.is_empty() { 1 } else { 2 }]
						.buffer()
						.unwrap()
						.clone(),
					access: BufferAccess::Read,
				});
			}
			buffers.push(BufferUse {
				buffer: output.buffer().unwrap().clone(),
				access: BufferAccess::Write,
			});
			let mut push_constants = vec![PushConstant::U32(count)];
			if kernel == KernelId::MatrixScaleF32 {
				push_constants.push(PushConstant::F32(1.25));
			}
			nodes.push(ComputeNode {
				operation: kernel.report_name(),
				kernel,
				buffers,
				push_constants,
				workgroups: [count.div_ceil(256).max(1), 1, 1],
				indirect: None,
				semantic_ops: Vec::new(),
				op_contract_hash: 0,
				jit: None,
				fusion_signature: None,
			});
			outputs.push(output);
		}
		let jit_output = Buffer::host_visible_storage(device, poison.len())?;
		jit_output.write(0, &poison)?;
		Ok(Self {
			inputs,
			outputs,
			graph: ExecutableGraph::from_compute_nodes(
				nodes,
				QueueFamilies::from_physical(device.physical()),
			)?,
			jit_output,
			host,
			count,
		})
	}

	fn compute_nodes(&self) -> Vec<ComputeNode> {
		self.graph.compute_nodes().cloned().collect()
	}

	fn tape_graph(&self, device: &Arc<Device>) -> Result<(ExecutableGraph, Vec<Storage>)> {
		let refs = self.inputs.iter().chain(&self.outputs).collect::<Vec<_>>();
		let (nodes, tapes) =
			fusion::jit_lab::diagnostic_elementwise_chain(device, self.compute_nodes(), &refs);
		assert_eq!(
			nodes.len(),
			1,
			"diagnostic tape must actually fuse all 22 operations"
		);
		assert_eq!(nodes[0].kernel, KernelId::MatrixFusedChainF32);
		Ok((
			ExecutableGraph::from_compute_nodes(nodes, QueueFamilies::from_physical(device.physical()))?,
			tapes,
		))
	}

	fn output(&self) -> Buffer {
		self.outputs.last().unwrap().buffer().unwrap().clone()
	}

	fn jit_buffers(&self) -> [Buffer; 4] {
		[
			self.inputs[0].buffer().unwrap().clone(),
			self.inputs[1].buffer().unwrap().clone(),
			self.inputs[2].buffer().unwrap().clone(),
			self.jit_output.clone(),
		]
	}

	fn check(&self, output: &Buffer) -> Result<Vec<f32>> {
		let mut bytes = vec![0; output.byte_len()];
		output.read(0, &mut bytes)?;
		let words = bytes
			.as_chunks::<4>()
			.0
			.iter()
			.map(|word| u32::from_le_bytes(*word))
			.collect::<Vec<_>>();
		assert!(
			words[self.count as usize..]
				.iter()
				.all(|&word| word == POISON),
			"tail canary overwritten"
		);
		let values = words[..self.count as usize]
			.iter()
			.map(|&word| f32::from_bits(word))
			.collect::<Vec<_>>();
		for (i, &actual) in values.iter().enumerate() {
			let expected = reference(self.host[0][i], self.host[1][i], self.host[2][i]);
			assert_equivalent(actual, expected, i);
		}
		for (input, expected) in self.inputs.iter().zip(&self.host) {
			let mut actual = vec![0; input.byte_len()];
			input.buffer().unwrap().read(0, &mut actual)?;
			for (bytes, value) in actual[..self.count as usize * 4]
				.as_chunks::<4>()
				.0
				.iter()
				.zip(expected)
			{
				assert_eq!(
					u32::from_le_bytes(*bytes),
					value.to_bits(),
					"read-only input changed"
				);
			}
			assert!(
				actual[self.count as usize * 4..]
					.as_chunks::<4>()
					.0
					.iter()
					.all(|word| u32::from_le_bytes(*word) == POISON)
			);
		}
		Ok(values)
	}
}

fn reference(x: f32, y: f32, z: f32) -> f32 {
	let mut value = x * y;
	for _ in 0..7 {
		let x = ((value + z) * 1.25) as f64;
		value = (0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())) as f32;
	}
	value
}

fn assert_equivalent(actual: f32, expected: f32, index: usize) {
	if expected.is_nan() {
		assert!(actual.is_nan(), "index {index}: {actual} vs NaN");
	} else if expected.is_infinite() {
		assert_eq!(actual, expected, "index {index}");
	} else {
		assert!(
			actual.is_finite() && (actual - expected).abs() <= expected.abs().max(1.0) * 1e-5,
			"index {index}: {actual} vs {expected}"
		);
		if actual == 0.0 && expected == 0.0 {
			assert_eq!(
				actual.to_bits(),
				expected.to_bits(),
				"signed zero at {index}"
			);
		}
	}
}

fn submit_sample(
	engine: &Engine,
	command: &super::super::ReusableCommandBuffer,
) -> Result<(f64, f64)> {
	let device = engine.handle.state.borrow().device.clone();
	let started = Instant::now();
	let timed = device.time_reusable_command(command.submission())?;
	let event = submit_recorded(&mut engine.handle.state.borrow_mut(), timed)?;
	let gpu_ms = event.device_duration()?.as_secs_f64() * 1000.0;
	Ok((gpu_ms, started.elapsed().as_secs_f64() * 1000.0))
}

fn count_env(name: &str, default: usize) -> usize {
	std::env::var(name).map_or(default, |value| value.parse().expect("laboratory count"))
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and exploratory replay timing"]
fn jit_chain_correctness_and_timings() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let bounded = device.physical().profile == ExecutionProfile::Compatibility;
	let path = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit trusted Slang library");
	let started = Instant::now();
	let compiler = SlangCompiler::open(std::path::Path::new(&path))?;
	let startup_ms = started.elapsed().as_secs_f64() * 1000.0;
	let artifact = compiler.compile_chain(bounded)?;
	let contract = ChainContract::fixed(3, 1);
	let info = engine.device_info();
	println!(
		"OA_JIT_COMPILER version={} profile={} startup_ms={startup_ms:.6} compile_ms={:.6} validation_ms={:.6} spirv_bytes={} device={:?} driver={:?} driver_version={}",
		compiler.version,
		if bounded { "compatibility" } else { "core" },
		artifact.compile_time.as_secs_f64() * 1000.0,
		artifact.validation_time.as_secs_f64() * 1000.0,
		artifact.words().len() * 4,
		info.name(),
		info.driver_info(),
		info.driver_version()
	);
	// Small, odd, nonfinite and zero-count proofs precede every timing sweep.
	let qualification_started = Instant::now();
	for count in [0, 1, 17, 257, 1025] {
		let mut host: [Vec<f32>; 3] = std::array::from_fn(|operand| {
			(0..count)
				.map(|i| (((i * (17 + operand * 8) + operand * 11) % 97) as f32 - 48.0) / 16.0)
				.collect()
		});
		let corners = [
			0.0,
			-0.0,
			f32::NAN,
			f32::INFINITY,
			f32::NEG_INFINITY,
			f32::from_bits(1),
		];
		for (i, &value) in corners.iter().take(count).enumerate() {
			host[0][i] = value;
			host[1][i] = 1.0;
			host[2][i] = -0.0;
		}
		let fixture = Fixture::new(&device, host)?;
		let unfused = device.record_reusable_compute_graph(&fixture.graph)?;
		// The production decoder intentionally declines zero-element tape chains.
		let tape = if count == 0 {
			None
		} else {
			let (graph, _tapes) = fixture.tape_graph(&device)?;
			Some(device.record_reusable_compute_graph(&graph)?)
		};
		if count == 17 {
			let mut aliased = fixture.jit_buffers();
			aliased[3] = aliased[0].clone();
			assert!(PreparedChain::new(&device, &artifact, &contract, aliased, fixture.count).is_err());
			let mut short = fixture.jit_buffers();
			short[3] = Buffer::host_visible_storage(&device, 4)?;
			assert!(PreparedChain::new(&device, &artifact, &contract, short, fixture.count).is_err());
			let other = Engine::builder()
				.devices(DeviceSelection::Index(index))
				.build()?;
			let foreign_device = other.handle.state.borrow().device.clone();
			let mut foreign = fixture.jit_buffers();
			foreign[0] = Buffer::host_visible_storage(&foreign_device, fixture.output().byte_len())?;
			assert!(PreparedChain::new(&device, &artifact, &contract, foreign, fixture.count).is_err());
		}
		let pipeline_started = Instant::now();
		let prepared = PreparedChain::new(
			&device,
			&artifact,
			&contract,
			fixture.jit_buffers(),
			fixture.count,
		)?;
		if count == 0 {
			println!(
				"OA_JIT_PIPELINE phase=first_use creation_ms={:.6}",
				pipeline_started.elapsed().as_secs_f64() * 1000.0
			);
		}
		let jit = device.record_jit_chain(&prepared)?;
		// Drop caller-owned pipeline state before execution: command owns it.
		drop(prepared);
		submit_sample(&engine, &unfused)?;
		let reference = fixture.check(&fixture.output())?;
		let mut routes = vec![(&jit, fixture.jit_output.clone())];
		if let Some(tape) = &tape {
			routes.push((tape, fixture.output()));
		}
		for (command, output) in routes {
			for _ in 0..2 {
				submit_sample(&engine, command)?;
			}
			let actual = fixture.check(&output)?;
			for (i, (&actual, &expected)) in actual.iter().zip(&reference).enumerate() {
				assert_equivalent(actual, expected, i);
			}
		}
	}
	println!(
		"OA_JIT_CORRECTNESS varied_inputs=true tails=true readonly_inputs=true nonfinite=true signed_zero=true command_retention=true qualification_ms={:.6} result=pass",
		qualification_started.elapsed().as_secs_f64() * 1000.0
	);
	let warmup = count_env("OA_JIT_LAB_WARMUP", 5);
	let samples = count_env("OA_JIT_LAB_SAMPLES", 20);
	assert!((1..=1000).contains(&samples) && warmup <= 1000);
	let sizes = std::env::var("OA_JIT_LAB_ELEMENTS").map_or_else(
		|_| vec![1024, 16384, 65536, 1048576],
		|value| vec![value.parse::<usize>().expect("element count")],
	);
	for count in sizes {
		assert!((1..=1048576).contains(&count));
		let fixture = Fixture::new(
			&device,
			[vec![0.25; count], vec![0.5; count], vec![0.75; count]],
		)?;
		let unfused = device.record_reusable_compute_graph(&fixture.graph)?;
		assert_eq!(fixture.graph.nodes().len(), 22);
		let (tape_graph, _tapes) = fixture.tape_graph(&device)?;
		let tape = device.record_reusable_compute_graph(&tape_graph)?;
		let started = Instant::now();
		let prepared = PreparedChain::new(
			&device,
			&artifact,
			&contract,
			fixture.jit_buffers(),
			fixture.count,
		)?;
		let pipeline_ms = started.elapsed().as_secs_f64() * 1000.0;
		let jit = device.record_jit_chain(&prepared)?;
		drop(prepared);
		let routes = [
			(&unfused, fixture.output(), "unfused", 22),
			(&tape, fixture.output(), "diagnostic_tape", 1),
			(&jit, fixture.jit_output.clone(), "specialized", 1),
		];
		for (command, output, _, _) in &routes {
			submit_sample(&engine, command)?;
			fixture.check(output)?;
		}
		for _ in 0..warmup {
			for (command, _, _, _) in &routes {
				submit_sample(&engine, command)?;
			}
		}
		let mut measurements: [Vec<(f64, f64)>; 3] = std::array::from_fn(|_| Vec::new());
		for sample in 0..samples {
			// Rotate order so one route does not always receive the first sample.
			for offset in 0..3 {
				let route = (sample + offset) % 3;
				let (gpu, wall) = submit_sample(&engine, routes[route].0)?;
				println!(
					"OA_JIT_SAMPLE elements={count} route={} sample={sample} gpu_ms={gpu:.9} wall_ms={wall:.9}",
					routes[route].2
				);
				measurements[route].push((gpu, wall));
			}
		}
		for (index, (_, output, name, nodes)) in routes.iter().enumerate() {
			fixture.check(output)?;
			let mut gpu = measurements[index]
				.iter()
				.map(|sample| sample.0)
				.collect::<Vec<_>>();
			let mut wall = measurements[index]
				.iter()
				.map(|sample| sample.1)
				.collect::<Vec<_>>();
			gpu.sort_by(f64::total_cmp);
			wall.sort_by(f64::total_cmp);
			println!(
				"OA_JIT_RESULT elements={count} route={name} nodes={nodes} warmup={warmup} samples={samples} gpu_p50_ms={:.9} gpu_p95_ms={:.9} wall_p50_ms={:.9} wall_p95_ms={:.9} specialized_pipeline_ms={pipeline_ms:.6} correctness=pass exploratory=true",
				gpu[samples / 2],
				gpu[(samples * 95 / 100).min(samples - 1)],
				wall[samples / 2],
				wall[(samples * 95 / 100).min(samples - 1)]
			);
		}
	}
	Ok(())
}

// Phase 2: discover source graph regions, then lower the shared SSA both ways.

fn expected(recipe: &[KernelId], x: f32, y: f32, z: f32) -> Vec<f32> {
	let mut value = x;
	recipe
		.iter()
		.enumerate()
		.map(|(index, &kernel)| {
			value = match kernel {
				KernelId::MatrixMulF32 => value * if index == 0 { y } else { z },
				KernelId::MatrixAddF32 => value + z,
				KernelId::MatrixScaleF32 => value * 1.25,
				KernelId::MatrixNegF32 => -value,
				KernelId::MatrixAbsF32 => value.abs(),
				KernelId::MlReluF32 => value.max(0.0),
				KernelId::MlGeluF32 => {
					let x = f64::from(value);
					(0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())) as f32
				}
				_ => panic!("fixture reference missing"),
			};
			value
		})
		.collect()
}

fn read_checked(buffer: &Buffer, count: usize) -> Result<Vec<f32>> {
	let mut bytes = vec![0; buffer.byte_len()];
	buffer.read(0, &mut bytes)?;
	let words = bytes
		.as_chunks::<4>()
		.0
		.iter()
		.map(|bytes| u32::from_le_bytes(*bytes))
		.collect::<Vec<_>>();
	assert!(
		words[count..].iter().all(|&word| word == POISON),
		"tail overwritten"
	);
	Ok(
		words[..count]
			.iter()
			.map(|&word| f32::from_bits(word))
			.collect(),
	)
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and discovered-chain replay timing"]
fn discovered_chains_correctness_and_timings() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let bounded = device.physical().profile == ExecutionProfile::Compatibility;
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let compiler = SlangCompiler::open(std::path::Path::new(&library))?;
	let recipes = [
		(
			"chain22",
			std::iter::once(KernelId::MatrixMulF32)
				.chain(
					[
						KernelId::MatrixAddF32,
						KernelId::MatrixScaleF32,
						KernelId::MlGeluF32,
					]
					.repeat(7),
				)
				.collect::<Vec<_>>(),
		),
		(
			"affine3",
			vec![
				KernelId::MatrixMulF32,
				KernelId::MatrixAddF32,
				KernelId::MatrixScaleF32,
			],
		),
		(
			"activation5",
			vec![
				KernelId::MatrixMulF32,
				KernelId::MatrixNegF32,
				KernelId::MatrixAbsF32,
				KernelId::MlReluF32,
				KernelId::MlGeluF32,
			],
		),
	];
	let mut artifacts = JitArtifacts::new();
	let mut misses = 0usize;
	let warmup = count_env("OA_JIT_LAB_WARMUP", 5);
	let samples = count_env("OA_JIT_LAB_SAMPLES", 20);
	assert!((1..=1000).contains(&samples) && warmup <= 1000);
	let sizes = std::env::var("OA_JIT_LAB_ELEMENTS").map_or_else(
		|_| vec![1, 17, 257, 16384, 1048576],
		|value| vec![value.parse::<usize>().unwrap()],
	);
	for (name, recipe) in &recipes {
		for &count in &sizes {
			assert!((1..=1048576).contains(&count));
			for retain in [false, true] {
				let fixture = Fixture::with_recipe(
					&device,
					std::array::from_fn(|operand| {
						(0..count)
							.map(|i| (((i * (17 + operand * 8) + operand * 11) % 97) as f32 - 48.0) / 64.0)
							.collect()
					}),
					recipe,
				)?;
				// A live semantic owner requires every corresponding intermediate store.
				let retained = if retain {
					fixture.outputs.iter().take(2).cloned().collect::<Vec<_>>()
				} else {
					vec![]
				};
				let refs = fixture
					.inputs
					.iter()
					.chain(&fixture.outputs)
					.collect::<Vec<_>>();
				let mut candidates = fusion::jit_lab::discover_programs(&fixture.compute_nodes(), &refs);
				assert_eq!(candidates.len(), 1, "one discovered region");
				let (program, inputs, outputs) = candidates.pop().unwrap();
				assert_eq!(program.instructions.len(), recipe.len());
				assert_eq!(program.outputs.len(), if retain { 3 } else { 1 });
				let (artifact, miss) = artifacts.chain(&compiler, &program, bounded)?;
				misses += usize::from(miss);
				let mut generated = Vec::new();
				for output in &outputs {
					let buffer = Buffer::host_visible_storage(&device, output.byte_len())?;
					let poison = vec![POISON; count + GUARD_WORDS]
						.iter()
						.flat_map(|word| word.to_le_bytes())
						.collect::<Vec<_>>();
					buffer.write(0, &poison)?;
					generated.push(buffer);
				}
				let mut bindings = inputs;
				bindings.extend(generated.iter().cloned());
				let prepared = PreparedChain::new(
					&device,
					artifact.as_ref(),
					&ChainContract::of(&program),
					bindings,
					count as u32,
				)?;
				let specialized = device.record_jit_chain(&prepared)?;
				drop(prepared);
				let unfused = device.record_reusable_compute_graph(&fixture.graph)?;
				let (nodes, _tapes) =
					fusion::jit_lab::diagnostic_elementwise_chain(&device, fixture.compute_nodes(), &refs);
				assert_eq!(nodes.len(), 1);
				let tape = device.record_reusable_compute_graph(&ExecutableGraph::from_compute_nodes(
					nodes,
					QueueFamilies::from_physical(device.physical()),
				)?)?;
				for command in [&unfused, &tape, &specialized] {
					submit_sample(&engine, command)?;
					let actual = if std::ptr::eq(command, &specialized) {
						&generated
					} else {
						&outputs
					};
					for (slot, buffer) in actual.iter().enumerate() {
						let values = read_checked(buffer, count)?;
						for (i, &actual) in values.iter().enumerate() {
							let reference = expected(
								recipe,
								fixture.host[0][i],
								fixture.host[1][i],
								fixture.host[2][i],
							);
							assert_equivalent(actual, reference[program.outputs[slot]], i);
						}
					}
				}
				for (buffer, host) in fixture.inputs.iter().zip(&fixture.host) {
					let actual = read_checked(buffer.buffer().unwrap(), count)?;
					assert!(
						actual
							.iter()
							.zip(host)
							.all(|(a, b)| a.to_bits() == b.to_bits())
					);
				}
				let routes = [
					("unfused", &unfused),
					("diagnostic_tape", &tape),
					("specialized", &specialized),
				];
				for _ in 0..warmup {
					for (_, command) in routes {
						submit_sample(&engine, command)?;
					}
				}
				let mut times: [Vec<f64>; 3] = std::array::from_fn(|_| vec![]);
				for sample in 0..samples {
					for offset in 0..3 {
						let route = (sample + offset) % 3;
						times[route].push(submit_sample(&engine, routes[route].1)?.0);
					}
				}
				for (route, (label, _)) in routes.iter().enumerate() {
					times[route].sort_by(f64::total_cmp);
					println!(
						"OA_JIT_DISCOVERED name={name} elements={count} retained={retain} source_nodes={} outputs={} route={label} gpu_p50_ms={:.9} compile_ms={:.6} validation_ms={:.6} cache={} correctness=pass exploratory=true",
						recipe.len(),
						program.outputs.len(),
						times[route][samples / 2],
						artifact.compile_time.as_secs_f64() * 1000.0,
						artifact.validation_time.as_secs_f64() * 1000.0,
						if miss { "miss" } else { "hit" }
					);
				}
				// Wrong launch geometry and in-place writes must remain outside the IR.
				let mut wrong = fixture.compute_nodes();
				wrong[0].workgroups[0] += 1;
				assert!(
					fusion::jit_lab::discover_programs(&wrong, &refs)
						.iter()
						.all(|(p, _, _)| p.instructions.len() < recipe.len())
				);
				drop(retained);
			}
		}
	}
	// Separate geometries are separate regions, and both remain discoverable.
	let first = Fixture::with_recipe(
		&device,
		[vec![0.25; 17], vec![0.5; 17], vec![0.75; 17]],
		&recipes[1].1,
	)?;
	let second = Fixture::with_recipe(
		&device,
		[vec![0.25; 257], vec![0.5; 257], vec![0.75; 257]],
		&recipes[1].1,
	)?;
	let mut nodes = first.compute_nodes();
	nodes.extend(second.graph.compute_nodes().cloned());
	let refs = first
		.inputs
		.iter()
		.chain(&first.outputs)
		.chain(&second.inputs)
		.chain(&second.outputs)
		.collect::<Vec<_>>();
	let candidates = fusion::jit_lab::discover_programs(&nodes, &refs);
	assert_eq!(
		candidates
			.iter()
			.map(|(p, _, _)| p.element_count)
			.collect::<Vec<_>>(),
		[17, 257]
	);
	// Extent-free artifact identity: one compile per recipe/output set, shared
	// across every legal element count.
	assert_eq!(misses, recipes.len() * 2);
	println!(
		"OA_JIT_DISCOVERY_CORRECTNESS recipes=3 retained_outputs=true tails=true readonly_inputs=true result=pass"
	);
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and temp artifact path"]
fn jit_artifacts_persist_across_restarts() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let bounded = device.physical().profile == ExecutionProfile::Compatibility;
	let path = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit trusted Slang library");
	let compiler = SlangCompiler::open(std::path::Path::new(&path))?;
	let scope = crate::runtime::device::logical::vkcache::VkCacheScope::new(device.physical());
	let artifact_path = crate::Path::new_from_path_buf(
		std::env::temp_dir().join(format!("oa-jit-persist-{}.vkcache", std::process::id())),
	);

	let fixture = Fixture::with_recipe(
		&device,
		[vec![0.25; 17], vec![0.5; 17], vec![0.75; 17]],
		&[
			KernelId::MatrixMulF32,
			KernelId::MatrixAddF32,
			KernelId::MatrixScaleF32,
		],
	)?;
	let refs = fixture
		.inputs
		.iter()
		.chain(&fixture.outputs)
		.collect::<Vec<_>>();
	let mut candidates = fusion::jit_lab::discover_programs(&fixture.compute_nodes(), &refs);
	assert_eq!(candidates.len(), 1);
	let (program, _, _) = candidates.pop().unwrap();

	let mut artifacts = JitArtifacts::new();
	let (compiled, miss) = artifacts.chain(&compiler, &program, bounded)?;
	assert!(miss, "first compile is a cache miss");
	artifacts.persist(&artifact_path, &scope)?;
	drop(artifacts);

	// Simulated restart: a fresh index restores the artifact and serves the
	// same program without invoking the compiler again.
	let mut restored = JitArtifacts::restore(&artifact_path, &scope)?;
	let (reused, miss) = restored.chain(&compiler, &program, bounded)?;
	assert!(!miss, "restored artifact must serve without recompiling");
	assert_eq!(compiled.words(), reused.words());

	// Checksummed cache metadata is still untrusted: reject binding-count
	// overflow and unknown section versions before computing push extents.
	{
		use crate::runtime::device::logical::vkcache::{self, SectionUpdate, VkCacheFile};
		let file = VkCacheFile::open(&artifact_path, &scope)?;
		let mut records = vkcache::jit::decode_index(
			&file
				.section(vkcache::SECTION_JIT_ARTIFACT_INDEX)
				.unwrap()
				.bytes,
		)?;
		records[0].binding_count = u32::MAX;
		vkcache::update(
			&artifact_path,
			&scope,
			&[SectionUpdate::replace(
				vkcache::SECTION_JIT_ARTIFACT_INDEX,
				1,
				vkcache::jit::encode_index(&records)?,
			)],
		)?;
		assert!(
			JitArtifacts::restore(&artifact_path, &scope)?
				.get(&program, bounded)
				.is_none()
		);
		restored.persist(&artifact_path, &scope)?;
		let file = VkCacheFile::open(&artifact_path, &scope)?;
		vkcache::update(
			&artifact_path,
			&scope,
			&[SectionUpdate::replace(
				vkcache::SECTION_JIT_ARTIFACT_INDEX,
				99,
				file
					.section(vkcache::SECTION_JIT_ARTIFACT_INDEX)
					.unwrap()
					.bytes
					.clone(),
			)],
		)?;
		assert!(
			JitArtifacts::restore(&artifact_path, &scope)?
				.get(&program, bounded)
				.is_none()
		);
		restored.persist(&artifact_path, &scope)?;
	}

	// A corrupted artifact fails closed: restore yields an empty index and
	// the next request recompiles deterministically instead of trusting
	// damaged bytes.
	let mut damaged = crate::Filesystem::read_binary(&artifact_path)?;
	damaged[64] ^= 0xff;
	crate::Filesystem::write_binary(&artifact_path, &damaged)?;
	let mut recovered = JitArtifacts::restore(&artifact_path, &scope)?;
	let (recompiled, miss) = recovered.chain(&compiler, &program, bounded)?;
	assert!(miss, "corrupted artifact must force recompilation");
	assert_eq!(compiled.words(), recompiled.words());
	let _ = std::fs::remove_file(&*artifact_path);
	Ok(())
}

/// Poll admission until the bounded background compile publishes or the
/// budget expires; compilation is deliberately asynchronous (OA-JIT-04), so
/// the first miss queues work and returns `None`.
fn pump_jit_admission(device: &Device, program: &Program) -> Vec<u8> {
	for _ in 0..4_000 {
		if let Some(key) = device.jit_admit(program, false) {
			return key;
		}
		std::thread::sleep(std::time::Duration::from_millis(5));
	}
	panic!("background specialization was not admitted within the test budget");
}

/// Process-global opt-in flag guard: a leaked `OA_JIT_FUSION` would silently
/// reroute every other test's fusion path, so the flag is always removed, and
/// every flag-sensitive test holds one lock for its whole body — the cargo
/// runner still executes ignored tests on multiple threads.
static JIT_FLAG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Restore the enclosing qualification runner's cache root on every exit.
struct ScopedVar {
	name: &'static str,
	previous: Option<std::ffi::OsString>,
}

impl ScopedVar {
	fn set(name: &'static str, value: &std::ffi::OsStr) -> Self {
		let previous = std::env::var_os(name);
		// SAFETY: ignored environment-mutating qualification runs are serial.
		unsafe { std::env::set_var(name, value) };
		Self { name, previous }
	}
}

impl Drop for ScopedVar {
	fn drop(&mut self) {
		// SAFETY: same serial qualification scope as construction.
		unsafe {
			match &self.previous {
				Some(value) => std::env::set_var(self.name, value),
				None => std::env::remove_var(self.name),
			}
		}
	}
}

struct JitFlag {
	_guard: std::sync::MutexGuard<'static, ()>,
}

impl JitFlag {
	/// Serialize flag-sensitive phases without raising the flag.
	fn hold() -> Self {
		Self {
			_guard: JIT_FLAG_LOCK
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner()),
		}
	}

	/// Serialize and raise the flag.
	fn set() -> Self {
		let flag = Self::hold();
		flag.raise();
		flag
	}

	fn raise(&self) {
		// SAFETY: every flag-sensitive test holds `JIT_FLAG_LOCK` for its
		// whole body, so no concurrent reader observes the flag.
		unsafe { std::env::set_var("OA_JIT_FUSION", "1") };
	}

	/// Lower the flag while the guard keeps the serialization lock.
	fn clear(&self) {
		// SAFETY: see `raise`.
		unsafe { std::env::remove_var("OA_JIT_FUSION") };
	}
}

impl Drop for JitFlag {
	fn drop(&mut self) {
		// SAFETY: see `set`.
		unsafe { std::env::remove_var("OA_JIT_FUSION") };
	}
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in fusion flag"]
fn production_jit_route_matches_canonical_paths() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	// The flag contract is exercised in both directions: the guard is held
	// for the whole test so no concurrent flag user can interleave.
	let flag = JitFlag::hold();

	// 48K elements is above the tape interpreter's production cutoff: without
	// the flag the segment must stay unfused; with it, the admitted
	// specialization is the only route that still fuses.
	let above_cutoff = 48 * 1024;
	let mut host: [Vec<f32>; 3] = std::array::from_fn(|operand| {
		(0..above_cutoff)
			.map(|i| (((i * (17 + operand * 8) + operand * 11) % 97) as f32 - 48.0) / 16.0)
			.collect()
	});
	host[0][0] = f32::NAN;
	host[1][0] = 1.0;
	host[2][0] = -0.0;
	host[0][1] = -0.0;
	host[0][2] = f32::INFINITY;

	let unfused_fixture = Fixture::new(&device, host.clone())?;
	let unfused_refs = unfused_fixture
		.inputs
		.iter()
		.chain(&unfused_fixture.outputs)
		.collect::<Vec<_>>();
	let unfused_nodes = unfused_fixture.graph.nodes().len();
	let before = device.fusion_census().snapshot();
	let (nodes, _tapes) =
		fusion::fuse_elementwise_nodes(&device, unfused_fixture.compute_nodes(), &unfused_refs);
	assert_eq!(
		nodes.len(),
		unfused_nodes,
		"gated route must leave an above-cutoff segment on the canonical kernels"
	);
	assert!(nodes.iter().all(|node| node.jit.is_none()));
	let census = device.fusion_census().snapshot();
	assert_eq!(
		census.examined_node_count() - before.examined_node_count(),
		unfused_nodes as u64
	);
	assert_eq!(
		census.tape_bound_node_count() - before.tape_bound_node_count(),
		unfused_nodes as u64,
		"every above-cutoff node records the tape bound: {census}"
	);

	flag.raise();
	let fixture = Fixture::new(&device, host)?;
	let refs = fixture
		.inputs
		.iter()
		.chain(&fixture.outputs)
		.collect::<Vec<_>>();
	let fixture_nodes = fixture.graph.nodes().len() as u64;
	// Compilation runs on the background worker: pump admission until the
	// queued specialization publishes, then the next fusion pass attaches it.
	let candidates = fusion::jit_lab::discover_programs(&fixture.compute_nodes(), &refs);
	assert_eq!(candidates.len(), 1, "one discovered region");
	pump_jit_admission(&device, &candidates[0].0);
	let before = device.fusion_census().snapshot();
	let (nodes, tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
	assert_eq!(nodes.len(), 1, "admitted chain must fuse all operations");
	let census = device.fusion_census().snapshot();
	assert_eq!(
		census.specialized_fused_node_count() - before.specialized_fused_node_count(),
		fixture_nodes
	);
	assert_eq!(
		census.specialized_region_count() - before.specialized_region_count(),
		1
	);
	assert!(
		census.jit_admitted_count() + census.jit_reused_count()
			> before.jit_admitted_count() + before.jit_reused_count(),
		"the admitted specialization is counted: {census}"
	);
	assert!(
		nodes[0].jit.is_some(),
		"opt-in flag must admit the specialization above the tape cutoff"
	);
	let jit_graph =
		ExecutableGraph::from_compute_nodes(nodes, QueueFamilies::from_physical(device.physical()))?;
	let mut changed = jit_graph.compute_nodes().cloned().collect::<Vec<_>>();
	changed[0].jit.as_mut().unwrap().key.push(0);
	assert_ne!(
		jit_graph.identity(),
		ExecutableGraph::from_compute_nodes(changed, jit_graph.queue_families())?.identity(),
		"execution identity must include the dynamic artifact key"
	);
	let jit_command = device.record_reusable_compute_graph(&jit_graph)?;
	submit_sample(&engine, &jit_command)?;
	let jit_values = fixture.check(&fixture.output())?;
	drop(tapes);

	// The canonical tape route interprets the same discovered chain; the
	// specialized dispatch must match it element for element.
	let (tape_graph, tape_tapes) = fixture.tape_graph(&device)?;
	let tape_command = device.record_reusable_compute_graph(&tape_graph)?;
	submit_sample(&engine, &tape_command)?;
	let tape_values = fixture.check(&fixture.output())?;
	drop(tape_tapes);
	for (i, (&actual, &expected)) in jit_values.iter().zip(&tape_values).enumerate() {
		assert_equivalent(actual, expected, i);
	}
	flag.clear();

	// The default route is unchanged: fusing without the flag below the
	// cutoff still produces the canonical tape node with no jit admission.
	let small = Fixture::new(&device, [vec![0.25; 257], vec![0.5; 257], vec![0.75; 257]])?;
	let small_refs = small
		.inputs
		.iter()
		.chain(&small.outputs)
		.collect::<Vec<_>>();
	let before = device.fusion_census().snapshot();
	let (nodes, _tapes) = fusion::fuse_elementwise_nodes(&device, small.compute_nodes(), &small_refs);
	assert_eq!(nodes.len(), 1);
	assert!(nodes[0].jit.is_none(), "default route must not admit jit");
	let census = device.fusion_census().snapshot();
	assert_eq!(
		census.tape_fused_node_count() - before.tape_fused_node_count(),
		small.graph.nodes().len() as u64
	);
	assert_eq!(census.jit_disabled_count() - before.jit_disabled_count(), 1);

	println!(
		"OA_JIT_PRODUCTION above_cutoff={above_cutoff} jit_admitted=true tape_parity=pass default_route=unchanged result=pass"
	);
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in fusion flag"]
fn production_jit_route_retains_observed_intermediates() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _flag = JitFlag::set();

	// Above the tape cutoff the admitted specialization is the only fused
	// route, so every fused dispatch deterministically runs the jit pipeline.
	let count = 48 * 1024;
	let host: [Vec<f32>; 3] = std::array::from_fn(|operand| {
		(0..count)
			.map(|i| (((i * (17 + operand * 8) + operand * 11) % 97) as f32 - 48.0) / 16.0)
			.collect()
	});
	let fixture = Fixture::new(&device, host)?;

	// Canonical values first: the unfused graph writes every intermediate,
	// including the buffer the fused run must still materialize.
	let unfused = device.record_reusable_compute_graph(&fixture.graph)?;
	submit_sample(&engine, &unfused)?;
	let mid = fixture.outputs[2].buffer().unwrap().clone();
	let expected_mid = read_checked(&mid, count)?;
	let expected_final = fixture.check(&fixture.output())?;

	// An autograd save or live Matrix keeps the mid-chain storage observable:
	// one extra owner turns the intermediate into a second program output.
	let _held = fixture.outputs[2].clone();
	let mid_poison = vec![POISON; mid.byte_len() / 4]
		.iter()
		.flat_map(|word| word.to_le_bytes())
		.collect::<Vec<_>>();
	mid.write(0, &mid_poison)?;
	let refs = fixture
		.inputs
		.iter()
		.chain(&fixture.outputs)
		.collect::<Vec<_>>();
	// The background compile must publish before fusion can attach the jit
	// route; the retained intermediate stays observable through `_held`.
	let candidates = fusion::jit_lab::discover_programs(&fixture.compute_nodes(), &refs);
	assert_eq!(
		candidates.len(),
		1,
		"observed chain must produce one region"
	);
	pump_jit_admission(&device, &candidates[0].0);
	let (nodes, tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
	assert_eq!(nodes.len(), 1, "observed chain must still fuse");
	let jit = nodes[0]
		.jit
		.as_ref()
		.expect("above-cutoff chain must admit jit");
	assert_eq!(
		jit
			.buffers
			.iter()
			.filter(|use_| use_.access == BufferAccess::Write)
			.count(),
		2,
		"specialized dispatch must bind the observed intermediate and the final output"
	);
	let graph =
		ExecutableGraph::from_compute_nodes(nodes, QueueFamilies::from_physical(device.physical()))?;
	let command = device.record_compute_graph(&graph)?;
	submit_recorded(&mut engine.handle.state.borrow_mut(), command)?.wait()?;
	drop(tapes);

	// The retained store is the autograd-visible intermediate: it must hold
	// the canonical value, not the poison left behind the fused dispatch.
	let actual_mid = read_checked(&mid, count)?;
	for (i, (&actual, &expected)) in actual_mid.iter().zip(&expected_mid).enumerate() {
		assert_equivalent(actual, expected, i);
	}
	let actual_final = fixture.check(&fixture.output())?;
	for (i, (&actual, &expected)) in actual_final.iter().zip(&expected_final).enumerate() {
		assert_equivalent(actual, expected, i);
	}
	println!(
		"OA_JIT_RETAINED elements={count} stores=2 intermediate_parity=pass final_parity=pass result=pass"
	);
	Ok(())
}

/// `OA_JIT_ROUTE` pins every discovered region to one route so matched
/// pairs can isolate tape, specialized and canonical costs. An
/// unrecognized value keeps the adaptive policy instead of failing.
#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in fusion flag"]
fn jit_route_override_forces_each_region_route() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	// A private `var` dir keeps the specialization cold: the `bogus` phase
	// below must observe adaptive admission before any publish.
	let var_dir = std::env::temp_dir().join(format!("oa-jit-route-{}", std::process::id()));
	std::fs::create_dir_all(&var_dir).expect("private var dir");
	let _var = ScopedVar::set("OA_VAR", var_dir.as_os_str());
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _flag = JitFlag::set();
	let route = ScopedVar::set("OA_JIT_ROUTE", "unfused".as_ref());
	let fixture = Fixture::new(&device, [vec![0.25; 257], vec![0.5; 257], vec![0.75; 257]])?;
	let refs = fixture
		.inputs
		.iter()
		.chain(&fixture.outputs)
		.collect::<Vec<_>>();
	let unfused_nodes = fixture.graph.nodes().len();

	// `unfused` declines every region at discovery: the canonical dispatches
	// stay exactly as recorded.
	let (nodes, _tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
	assert_eq!(
		nodes.len(),
		unfused_nodes,
		"OA_JIT_ROUTE=unfused must leave canonical dispatches"
	);
	assert!(nodes.iter().all(|node| node.jit.is_none()));

	// `tape` still fuses the region but admission is declined, so the tape
	// interpreter carries the chain.
	// SAFETY: the route override is process-global; `JitFlag`'s held lock
	// serializes every flag-sensitive test phase.
	unsafe { std::env::set_var("OA_JIT_ROUTE", "tape") };
	let before = device.fusion_census().snapshot();
	let (nodes, _tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
	assert_eq!(nodes.len(), 1, "tape route still fuses the region");
	assert!(nodes[0].jit.is_none(), "tape route declines admission");
	let census = device.fusion_census().snapshot();
	assert!(
		census.jit_disabled_count() > before.jit_disabled_count(),
		"the forced tape route records declined admission: {census}"
	);

	// An unrecognized value falls back to the adaptive policy: the region
	// fuses and admission runs normally instead of being declined.
	unsafe { std::env::set_var("OA_JIT_ROUTE", "bogus") };
	let before = device.fusion_census().snapshot();
	let (nodes, _tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
	assert_eq!(nodes.len(), 1);
	assert!(
		nodes[0].jit.is_none(),
		"an unpublished specialization still runs tape"
	);
	let census = device.fusion_census().snapshot();
	assert_eq!(
		census.jit_disabled_count(),
		before.jit_disabled_count(),
		"adaptive admission must not record the disabled outcome: {census}"
	);

	// `specialized` bypasses the hysteresis policy: once the background
	// compile publishes, the fused node dispatches the compiled pipeline.
	unsafe { std::env::set_var("OA_JIT_ROUTE", "specialized") };
	let candidates = fusion::jit_lab::discover_programs(&fixture.compute_nodes(), &refs);
	assert_eq!(candidates.len(), 1, "one discovered region");
	pump_jit_admission(&device, &candidates[0].0);
	let (nodes, _tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
	assert_eq!(nodes.len(), 1);
	assert!(
		nodes[0].jit.is_some(),
		"forced specialized route must attach the admitted pipeline"
	);
	drop(route);
	drop(engine);
	let _ = std::fs::remove_dir_all(&var_dir);

	println!(
		"OA_JIT_ROUTE_OVERRIDE unfused=canonical tape=declined bogus=adaptive specialized=attached result=pass"
	);
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in fusion flag"]
fn jit_admission_is_safe_across_threads() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _flag = JitFlag::set();

	// Distinct discovered programs; fixtures stay on this thread — only
	// admission, lookup and section snapshots race.
	let recipes: [Vec<KernelId>; 5] = [
		vec![
			KernelId::MatrixMulF32,
			KernelId::MatrixAddF32,
			KernelId::MatrixScaleF32,
		],
		vec![
			KernelId::MatrixMulF32,
			KernelId::MatrixNegF32,
			KernelId::MatrixAbsF32,
		],
		vec![
			KernelId::MatrixMulF32,
			KernelId::MatrixAddF32,
			KernelId::MlReluF32,
			KernelId::MlGeluF32,
		],
		vec![
			KernelId::MatrixMulF32,
			KernelId::MatrixNegF32,
			KernelId::MatrixAbsF32,
			KernelId::MlReluF32,
			KernelId::MlGeluF32,
		],
		vec![
			KernelId::MatrixMulF32,
			KernelId::MatrixScaleF32,
			KernelId::MlReluF32,
			KernelId::MatrixAddF32,
			KernelId::MlGeluF32,
			KernelId::MatrixNegF32,
		],
	];
	let mut programs = Vec::new();
	for recipe in &recipes {
		let fixture = Fixture::with_recipe(
			&device,
			[vec![0.25; 257], vec![0.5; 257], vec![0.75; 257]],
			recipe,
		)?;
		let refs = fixture
			.inputs
			.iter()
			.chain(&fixture.outputs)
			.collect::<Vec<_>>();
		let mut candidates = fusion::jit_lab::discover_programs(&fixture.compute_nodes(), &refs);
		assert_eq!(candidates.len(), 1, "one discovered region");
		programs.push(candidates.pop().unwrap().0);
	}

	// Compile four of the five here so racing threads exercise both the
	// dedup-hit path and a genuine first-compile race on the fifth.
	let fresh = programs.pop().unwrap();
	let expected = programs
		.iter()
		.map(|program| pump_jit_admission(&device, program))
		.collect::<Vec<_>>();
	assert!(
		expected
			.iter()
			.collect::<std::collections::HashSet<_>>()
			.len()
			== programs.len(),
		"distinct programs must produce distinct artifact keys"
	);

	let fresh_keys = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new()));
	let mut handles = Vec::new();
	for _ in 0..4 {
		let device = device.clone();
		let programs = programs.clone();
		let expected = expected.clone();
		let fresh = fresh.clone();
		let fresh_keys = fresh_keys.clone();
		handles.push(std::thread::spawn(move || {
			for _ in 0..8 {
				for (program, key) in programs.iter().zip(&expected) {
					assert_eq!(
						device.jit_admit(program, false).as_deref(),
						Some(key.as_slice()),
						"concurrent admission must return the deduplicated key"
					);
					assert!(
						device.jit_pipeline(key).is_some(),
						"admitted pipeline must resolve"
					);
				}
				let key = {
					let mut admitted = None;
					for _ in 0..200_000 {
						if let Some(key) = device.jit_admit(&fresh, false) {
							admitted = Some(key);
							break;
						}
						std::thread::yield_now();
					}
					admitted.expect("racing admission")
				};
				assert!(
					device.jit_pipeline(&key).is_some(),
					"racing pipeline must resolve"
				);
				fresh_keys.lock().expect("key set").insert(key);
				assert!(!device.jit_section_updates().is_empty());
			}
		}));
	}
	for handle in handles {
		handle.join().expect("admission thread");
	}
	assert_eq!(
		fresh_keys.lock().expect("key set").len(),
		1,
		"racing first-compile admissions must converge on one artifact key"
	);
	let updates = device.jit_section_updates();
	println!(
		"OA_JIT_THREADS programs={} threads=4 section_updates={} result=pass",
		programs.len() + 1,
		updates.len()
	);
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in fusion flag"]
fn production_route_observation_measures_and_selects() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	// A private `var` dir gives the observation store a cold `.vkcache` so
	// the probe cycle is deterministic regardless of earlier runs.
	let var_dir = std::env::temp_dir().join(format!("oa-jit-observe-{}", std::process::id()));
	std::fs::create_dir_all(&var_dir).expect("private var dir");
	let _var = ScopedVar::set("OA_VAR", var_dir.as_os_str());
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _flag = JitFlag::set();

	// A small tape-eligible chain: the hysteresis policy must gather
	// specialized samples first, probe tape, then keep the faster route.
	let fixture = Fixture::new(
		&device,
		[vec![0.25; 1024], vec![0.5; 1024], vec![0.75; 1024]],
	)?;
	let refs = fixture
		.inputs
		.iter()
		.chain(&fixture.outputs)
		.collect::<Vec<_>>();
	// The background compile must publish before the probe cycle, otherwise
	// the pass loop outruns the worker and only tape is ever measured.
	let candidates = fusion::jit_lab::discover_programs(&fixture.compute_nodes(), &refs);
	assert_eq!(candidates.len(), 1, "one discovered region");
	pump_jit_admission(&device, &candidates[0].0);
	let mut routes = (0_usize, 0_usize);
	for _ in 0..12 {
		let (nodes, tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
		assert_eq!(nodes.len(), 1, "small chain must fuse");
		assert!(
			nodes[0].fusion_signature.is_some(),
			"opt-in fused node must carry its observation signature"
		);
		if nodes[0].jit.is_some() {
			routes.0 += 1;
		} else {
			routes.1 += 1;
		}
		let graph =
			ExecutableGraph::from_compute_nodes(nodes, QueueFamilies::from_physical(device.physical()))?;
		let command = device.record_compute_graph(&graph)?;
		submit_recorded(&mut engine.handle.state.borrow_mut(), command)?.wait()?;
		drop(tapes);
	}
	let observations = device.jit_observations();
	assert_eq!(
		observations.len(),
		1,
		"one fused signature must produce one observation"
	);
	let (_, specialized, tape, selected) = observations[0];
	assert!(
		specialized >= 4 && tape >= 4,
		"hysteresis must measure both routes: specialized={specialized} tape={tape}"
	);
	assert!(routes.0 > 0 && routes.1 > 0, "both routes must have run");
	// Engine teardown publishes the observation section into the private
	// cache before it is removed.
	drop(engine);
	let _ = std::fs::remove_dir_all(&var_dir);
	println!(
		"OA_JIT_OBSERVE jit_routes={} tape_routes={} specialized_samples={specialized} tape_samples={tape} selected_specialized={selected} result=pass",
		routes.0, routes.1
	);
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in fusion flag"]
fn production_route_observations_persist_across_restart() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let var_dir = std::env::temp_dir().join(format!("oa-jit-restart-{}", std::process::id()));
	std::fs::create_dir_all(&var_dir).expect("private var dir");
	let _var = ScopedVar::set("OA_VAR", var_dir.as_os_str());
	let _flag = JitFlag::set();

	let run_dispatches = |engine: &Engine, rounds: usize| -> Result<(Arc<Device>, u64)> {
		let device = engine.handle.state.borrow().device.clone();
		let fixture = Fixture::new(
			&device,
			[vec![0.25; 1024], vec![0.5; 1024], vec![0.75; 1024]],
		)?;
		let refs = fixture
			.inputs
			.iter()
			.chain(&fixture.outputs)
			.collect::<Vec<_>>();
		// Publish the queued specialization before the observed rounds so the
		// probe cycle — not compile latency — decides the persisted medians.
		let candidates = fusion::jit_lab::discover_programs(&fixture.compute_nodes(), &refs);
		assert_eq!(candidates.len(), 1, "one discovered region");
		pump_jit_admission(&device, &candidates[0].0);
		let mut signature = None;
		for _ in 0..rounds {
			let (nodes, tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
			assert_eq!(nodes.len(), 1, "small chain must fuse");
			signature = nodes[0].fusion_signature;
			let graph = ExecutableGraph::from_compute_nodes(
				nodes,
				QueueFamilies::from_physical(device.physical()),
			)?;
			let command = device.record_compute_graph(&graph)?;
			submit_recorded(&mut engine.handle.state.borrow_mut(), command)?.wait()?;
			drop(tapes);
		}
		Ok((device, signature.expect("observed signature")))
	};

	// Phase 1: populate the observation store, then let Engine teardown
	// commit section 6 into the private cache.
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let (device, signature) = run_dispatches(&engine, 12)?;
	let before = device.jit_observations();
	let first = before
		.iter()
		.find(|(key, ..)| *key == signature)
		.copied()
		.expect("signature observed before restart");
	let expected_selected = first.3;
	drop(engine);

	// Phase 2: a fresh device restores section 6 on its first admission;
	// the persisted medians must reproduce the earlier policy decision
	// for the same signature instead of restarting the probe cycle.
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let (device, signature_after) = run_dispatches(&engine, 2)?;
	assert_eq!(signature_after, signature, "same chain keeps its signature");
	let after = device.jit_observations();
	let (.., selected) = after
		.iter()
		.find(|(key, ..)| *key == signature)
		.copied()
		.expect("restored signature present after restart");
	assert_eq!(
		selected, expected_selected,
		"restored medians must reproduce the persisted policy decision"
	);
	drop(engine);
	let _ = std::fs::remove_dir_all(&var_dir);
	println!("OA_JIT_RESTART signature={signature:016x} selected={selected} result=pass");
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in fusion flag"]
fn production_route_observation_queue_stays_bounded() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _flag = JitFlag::set();

	let fixture = Fixture::new(
		&device,
		[vec![0.25; 1024], vec![0.5; 1024], vec![0.75; 1024]],
	)?;
	let refs = fixture
		.inputs
		.iter()
		.chain(&fixture.outputs)
		.collect::<Vec<_>>();
	// More in-flight instrumented dispatches than MAX_PENDING_TIMINGS: the
	// pool must evict unread query sets instead of growing unboundedly, and
	// the observation store must stay consistent afterwards.
	let mut events = Vec::new();
	let mut signature = None;
	for _ in 0..80 {
		let (nodes, tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
		assert_eq!(nodes.len(), 1, "small chain must fuse");
		signature = nodes[0].fusion_signature;
		let graph =
			ExecutableGraph::from_compute_nodes(nodes, QueueFamilies::from_physical(device.physical()))?;
		let command = device.record_compute_graph(&graph)?;
		events.push(submit_recorded(
			&mut engine.handle.state.borrow_mut(),
			command,
		)?);
		drop(tapes);
	}
	for event in events {
		event.wait()?;
	}
	// Other signatures may be present — restored section-6 records from
	// earlier engine commits are part of the store by design.
	let (_, specialized, tape, _) = device
		.jit_observations()
		.into_iter()
		.find(|(key, ..)| Some(*key) == signature)
		.expect("fused signature must be observed");
	assert!(
		specialized <= 32 && tape <= 32,
		"sample windows stay bounded: specialized={specialized} tape={tape}"
	);
	assert!(
		specialized + tape > 0,
		"completed dispatches must still land samples"
	);
	println!(
		"OA_JIT_BOUND dispatches=80 specialized_samples={specialized} tape_samples={tape} result=pass"
	);
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in fusion flag"]
fn rejected_prefixes_do_not_consume_jit_artifacts() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: this qualification runs with one test thread.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let _flag = JitFlag::set();
	let var_dir = std::env::temp_dir().join(format!("oa-jit-prefix-{}", std::process::id()));
	std::fs::create_dir_all(&var_dir).unwrap();
	let _var = ScopedVar::set("OA_VAR", var_dir.as_os_str());
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let fixture = Fixture::with_recipe(
		&device,
		[vec![0.25; 257], vec![0.5; 257], vec![0.75; 257]],
		&[KernelId::MatrixNegF32; 70],
	)?;
	let refs = fixture
		.inputs
		.iter()
		.chain(&fixture.outputs)
		.collect::<Vec<_>>();
	// Both tape-capacity splits compile on the background worker: keep fusing
	// until the queued specializations publish and attach.
	let mut nodes = Vec::new();
	for _ in 0..4_000 {
		let (fused, _tapes) = fusion::fuse_elementwise_nodes(&device, fixture.compute_nodes(), &refs);
		nodes = fused;
		if nodes.iter().all(|node| node.jit.is_some()) {
			break;
		}
		std::thread::sleep(std::time::Duration::from_millis(5));
	}
	assert_eq!(
		nodes.len(),
		2,
		"70 unary ops fit into 63-op and 7-op tape regions"
	);
	assert!(nodes.iter().all(|node| node.jit.is_some()));
	let updates = device.jit_section_updates();
	let index = updates
		.iter()
		.find(|update| {
			update.section_type == crate::runtime::device::logical::vkcache::SECTION_JIT_ARTIFACT_INDEX
		})
		.unwrap();
	let entries =
		crate::runtime::device::logical::vkcache::jit::decode_index(index.payload.as_ref().unwrap())?;
	assert_eq!(
		entries.len(),
		2,
		"only executable regions may consume compilation entries"
	);
	let command = device.record_reusable_compute_graph(&ExecutableGraph::from_compute_nodes(
		nodes,
		QueueFamilies::from_physical(device.physical()),
	)?)?;
	submit_sample(&engine, &command)?;
	assert!(
		read_checked(&fixture.output(), 257)?
			.iter()
			.all(|&value| value == 0.25)
	);
	drop(engine);
	let _ = std::fs::remove_dir_all(&var_dir);
	println!("OA_JIT_PREFIX source_nodes=70 fused_nodes=2 compiled_artifacts=2 correctness=pass");
	Ok(())
}

fn gemm_values(length: usize, salt: usize) -> Vec<f32> {
	(0..length)
		.map(|index| (((index * 17 + salt) % 29) as f32 - 14.0) / 7.0)
		.collect()
}

fn gemm_cpu_oracle(left: &[f32], right: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
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

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flag"]
fn tile_gemm_route_matches_cpu_oracle() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	// The guard serializes every flag-sensitive phase in this file.
	let _hold = JitFlag::hold();
	let _tile = ScopedVar::set("OA_TILE_GEMM", std::ffi::OsStr::new("1"));
	let _fusion1 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
	let _forced1 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
	let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
	let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
	// m*n >= 64 lands the tiled incumbent variant — the tile route's only
	// match — across exact-tile, odd, tail and vector-ish shapes; the
	// (4,4) case keeps proving the planner's naive fallback is untouched.
	let shapes = [
		(64usize, 64usize, 16usize),
		(65, 129, 33),
		(17, 5, 9),
		(128, 96, 64),
		(1, 64, 32),
		(4, 4, 7),
	];
	let before = device.fusion_census().snapshot();
	for (m, n, k) in shapes {
		let left_values = gemm_values(m * k, 3);
		let right_values = gemm_values(n * k, 11);
		let expected = gemm_cpu_oracle(&left_values, &right_values, m, n, k);
		let left = crate::Matrix::new_from_f32([m, k], &left_values)?;
		let right = crate::Matrix::new_from_f32([n, k], &right_values)?;
		let output = crate::matrix::mat_mul_nt(&left, &right)?;
		let actual = output.read_f32()?;
		for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
			let error = (actual - expected).abs();
			assert!(
				error <= f32::from_bits(0x38d1b717),
				"({m},{n},{k}) element {index}: expected {expected}, found {actual}, error {error}"
			);
		}
	}
	let census = device.fusion_census().snapshot();
	assert!(
		census.jit_admitted_count() + census.jit_reused_count()
			> before.jit_admitted_count() + before.jit_reused_count(),
		"the tile specialization must be admitted and counted: {census}"
	);
	// CC-12: `OA_TILE_GEMM_SEARCH` offers the bounded 32/64 geometry
	// set as measured candidates — repeated dispatches probe the
	// least-sampled signature, so multiple tile geometries accumulate
	// route evidence while every run stays inside the oracle bound.
	{
		let _search = ScopedVar::set("OA_TILE_GEMM_SEARCH", std::ffi::OsStr::new("1"));
		let (m, n, k) = (64usize, 64usize, 32usize);
		let left_values = gemm_values(m * k, 3);
		let right_values = gemm_values(n * k, 11);
		let expected = gemm_cpu_oracle(&left_values, &right_values, m, n, k);
		let left = crate::Matrix::new_from_f32([m, k], &left_values)?;
		let right = crate::Matrix::new_from_f32([n, k], &right_values)?;
		for _ in 0..12 {
			let actual = crate::matrix::mat_mul_nt(&left, &right)?.read_f32()?;
			for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
				let error = (actual - expected).abs();
				assert!(
					error <= f32::from_bits(0x38d1b717),
					"({m},{n},{k}) element {index}: expected {expected}, \
					 found {actual}, error {error}"
				);
			}
		}
		let mut probed = 0usize;
		for tile_rows in [32u32, 64] {
			for tile_columns in [32u32, 64] {
				let program = crate::runtime::dna::ir::tile::BlockProgram::mat_mul_nt(
					tile_rows,
					tile_columns,
					16,
					4,
					4,
					4,
				)?;
				let (_, samples, _) = device.tile_route_stats(device.tile_signature(&program));
				probed += usize::from(samples > 0);
			}
		}
		assert!(
			probed >= 2,
			"the geometry search must probe multiple candidates: {probed}"
		);
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flag"]
fn tile_softmax_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	// (shape, dim): last-axis rows, an odd axis interior (inner > 1
	// exercises the strided layout), a multi-iteration long row and the
	// degenerate single-element row.
	let cases: [(Vec<usize>, i32); 4] = [
		(vec![64, 32], -1),
		(vec![4, 8, 3], 1),
		(vec![5, 1031], -1),
		(vec![1, 1], -1),
	];
	for (shape, dim) in &cases {
		let count = shape.iter().product::<usize>();
		let values: Vec<f32> = (0..count)
			.map(|i| ((i * 2654435761usize % 1009) as f32 - 500.0) / 127.0)
			.collect();
		let input = crate::Matrix::new_from_f32(shape.as_slice(), &values)?;
		// Incumbent first — the emitted template repeats the donor's
		// reduction order, so the oracle is bit-exact.
		let expected = crate::matrix::softmax(&input, *dim)?.read_f32()?;
		let before = device.fusion_census().snapshot();
		let output = {
			let _tile = ScopedVar::set("OA_TILE_SOFTMAX", std::ffi::OsStr::new("1"));
			let _fusion2 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced2 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			crate::matrix::softmax(&input, *dim)?.read_f32()?
		};
		let census = device.fusion_census().snapshot();
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"{shape:?} dim {dim}: the softmax specialization must be admitted: {census}"
		);
		for (index, (actual, expected)) in output.iter().zip(expected.iter()).enumerate() {
			assert_eq!(
				actual.to_bits(),
				expected.to_bits(),
				"{shape:?} dim {dim} element {index}: expected {expected}, found {actual}"
			);
		}
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flag"]
fn tile_gemv_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	// m=1 decode-class shapes select the GemmSmallM/GEMV variant; the
	// tile route repeats the donor's strided fold + tree order so the
	// incumbent is a bit-exact oracle across exact and tail extents.
	for (m, n, k) in [(1usize, 512usize, 512usize), (1, 777, 1024), (1, 520, 513)] {
		let left_values = gemm_values(m * k, 3);
		let right_values = gemm_values(n * k, 11);
		let left = crate::Matrix::new_from_f32([m, k], &left_values)?;
		let right = crate::Matrix::new_from_f32([n, k], &right_values)?;
		let expected = crate::matrix::mat_mul_nt(&left, &right)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_GEMV",
			|| crate::matrix::mat_mul_nt(&left, &right)?.read_f32(),
			&expected,
			"gemv",
		)?;
	}
	Ok(())
}

fn tile_route_bit_exact(
	device: &crate::runtime::device::Device,
	flag: &'static str,
	run: impl FnOnce() -> Result<Vec<f32>>,
	expected: &[f32],
	label: &str,
) -> Result<()> {
	let before = device.fusion_census().snapshot();
	let actual = {
		let _route = ScopedVar::set(flag, std::ffi::OsStr::new("1"));
		// The qualification contract is the tile route's numerics, not the
		// measured-selection policy: force specialized so an accumulated
		// `.vkcache` record can never probe the incumbent mid-assert.
		let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
		run()?
	};
	let census = device.fusion_census().snapshot();
	assert!(
		census.jit_admitted_count() + census.jit_reused_count()
			> before.jit_admitted_count() + before.jit_reused_count(),
		"{label}: the specialization must be admitted: {census}"
	);
	for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
		assert_eq!(
			actual.to_bits(),
			expected.to_bits(),
			"{label} element {index}: expected {expected}, found {actual}"
		);
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_reduce_norm_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x9e3779b9u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2011) as f32 - 1000.0) / 251.0
			})
			.collect()
	};
	// The emitted templates repeat each donor's reduction order, so the
	// incumbent kernels are bit-exact oracles.
	for (shape, dim) in [
		(vec![64usize, 32usize], -1i32),
		(vec![4, 8, 3], 1),
		(vec![5, 1031], -1),
	] {
		let input = crate::Matrix::new_from_f32(shape.as_slice(), &values(shape.iter().product()))?;
		let expected = crate::matrix::log_softmax(&input, dim)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_LOG_SOFTMAX",
			|| crate::matrix::log_softmax(&input, dim).and_then(|m| m.read_f32()),
			&expected,
			&format!("log_softmax {shape:?} dim {dim}"),
		)?;
	}
	for (rows, columns) in [(64usize, 32usize), (3, 1031), (1, 1), (8, 300)] {
		let input = crate::Matrix::new_from_f32([rows, columns], &values(rows * columns))?;
		let weight = crate::Matrix::new_from_f32([columns], &values(columns))?;
		// Incumbent pass first; each route then re-runs under its flag.
		let expected = crate::ml::nn::RmsNorm::new_from_matrix(weight.clone(), 1e-5)?
			.forward(&input)?
			.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_RMS_NORM",
			|| {
				crate::ml::nn::RmsNorm::new_from_matrix(weight.clone(), 1e-5)
					.and_then(|m| m.forward(&input))
					.and_then(|m| m.read_f32())
			},
			&expected,
			&format!("rms_norm [{rows},{columns}]"),
		)?;
		let bias = crate::Matrix::new_from_f32([columns], &values(columns))?;
		let expected = crate::ml::nn::LayerNorm::new_from_matrices(weight.clone(), bias.clone(), 1e-5)?
			.forward(&input)?
			.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_LAYER_NORM",
			|| {
				crate::ml::nn::LayerNorm::new_from_matrices(weight.clone(), bias.clone(), 1e-5)
					.and_then(|m| m.forward(&input))
					.and_then(|m| m.read_f32())
			},
			&expected,
			&format!("layer_norm [{rows},{columns}]"),
		)?;
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_softmax_backward_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x517cc1b727220a95u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2011) as f32 - 1000.0) / 251.0
			})
			.collect()
	};
	for (shape, dim) in [
		(vec![64usize, 32usize], -1i32),
		(vec![4, 8, 3], 1),
		(vec![5, 1031], -1),
		(vec![1, 1], -1),
	] {
		let count = shape.iter().product::<usize>();
		let y = crate::Matrix::new_from_f32(shape.as_slice(), &values(count))?;
		let dy = crate::Matrix::new_from_f32(shape.as_slice(), &values(count))?;
		let expected = crate::matrix::softmax_backward(&y, &dy, dim)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_SOFTMAX_BACKWARD",
			|| crate::matrix::softmax_backward(&y, &dy, dim).and_then(|m| m.read_f32()),
			&expected,
			&format!("softmax_backward {shape:?} dim {dim}"),
		)?;
		let expected = crate::matrix::log_softmax_backward(&y, &dy, dim)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_LOG_SOFTMAX_BACKWARD",
			|| crate::matrix::log_softmax_backward(&y, &dy, dim).and_then(|m| m.read_f32()),
			&expected,
			&format!("log_softmax_backward {shape:?} dim {dim}"),
		)?;
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_linear_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x9e3779b97f4a7c15u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2011) as f32 - 1000.0) / 509.0
			})
			.collect()
	};
	for (batch, input_features, output_features) in [
		(64usize, 128usize, 64usize),
		(17, 33, 9),
		(65, 40, 129),
		(3, 7, 5),
	] {
		let input =
			crate::Matrix::new_from_f32([batch, input_features], &values(batch * input_features))?;
		let weight = crate::Matrix::new_from_f32(
			[output_features, input_features],
			&values(output_features * input_features),
		)?;
		let bias = crate::Matrix::new_from_f32([output_features], &values(output_features))?;
		let layer = crate::ml::nn::Linear::new_from_matrices(weight.clone(), bias.clone())?;
		let expected_linear = layer.forward(&input)?.read_f32()?;
		let expected_act = crate::ml::matrix::relu(&crate::Matrix::new_from_f32(
			[batch, output_features],
			&expected_linear,
		)?)?
		.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_LINEAR",
			|| layer.forward(&input).and_then(|m| m.read_f32()),
			&expected_linear,
			&format!("linear {batch}x{input_features}x{output_features}"),
		)?;
		// The fused `linear + relu` node forms at flush; both materialized
		// buffers (pre-act and act) must match the incumbent fused kernel.
		let fused = || -> Result<(Vec<f32>, Vec<f32>)> {
			let linear = layer.forward(&input)?;
			let act = crate::ml::matrix::relu(&linear)?;
			Ok((linear.read_f32()?, act.read_f32()?))
		};
		let expected_pair = fused()?;
		assert_eq!(expected_pair.0, expected_linear);
		assert_eq!(expected_pair.1, expected_act);
		let before = device.fusion_census().snapshot();
		let hold = ScopedVar::set("OA_TILE_LINEAR_ACT", std::ffi::OsStr::new("1"));
		let _fusion3 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		let _forced3 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
		let actual = fused()?;
		drop(hold);
		let census = device.fusion_census().snapshot();
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"the fused linear_act specialization must be admitted: {census}"
		);
		assert_eq!(
			actual.0, expected_pair.0,
			"fused linear output differs {batch}x{input_features}x{output_features}"
		);
		assert_eq!(
			actual.1, expected_pair.1,
			"fused act output differs {batch}x{input_features}x{output_features}"
		);

		// Residual family: linear → add (pair) and linear → add → relu
		// (triple) fuse into MlLinearResidual{Act}F32; all three
		// materialized outputs must match the incumbent fused kernels.
		let residual =
			crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
		let fused_residual = || -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
			let linear = layer.forward(&input)?;
			let sum = linear.add(&residual)?;
			let act = crate::ml::matrix::relu(&sum)?;
			Ok((linear.read_f32()?, sum.read_f32()?, act.read_f32()?))
		};
		let pair = || -> Result<(Vec<f32>, Vec<f32>)> {
			let linear = layer.forward(&input)?;
			let sum = linear.add(&residual)?;
			Ok((linear.read_f32()?, sum.read_f32()?))
		};
		let expected_pair = pair()?;
		let before = device.fusion_census().snapshot();
		let hold = ScopedVar::set("OA_TILE_LINEAR_ACT", std::ffi::OsStr::new("1"));
		let _fusion4 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		let _forced4 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
		let actual_pair = pair()?;
		drop(hold);
		let census = device.fusion_census().snapshot();
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"the fused linear_residual specialization must be admitted: {census}"
		);
		assert_eq!(actual_pair, expected_pair);
		let expected_residual = fused_residual()?;
		let before = device.fusion_census().snapshot();
		let hold = ScopedVar::set("OA_TILE_LINEAR_ACT", std::ffi::OsStr::new("1"));
		let _fusion5 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		let _forced5 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
		let actual_residual = fused_residual()?;
		drop(hold);
		let census = device.fusion_census().snapshot();
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"the fused residual specialization must be admitted: {census}"
		);
		assert_eq!(actual_residual, expected_residual);

		// QKV-style triple: three same-geometry linears sharing one
		// input fuse into the z-dispatched MlLinearTripleF32 node.
		let layer_b = crate::ml::nn::Linear::new_from_matrices(
			crate::Matrix::new_from_f32(
				[output_features, input_features],
				&values(output_features * input_features),
			)?,
			crate::Matrix::new_from_f32([output_features], &values(output_features))?,
		)?;
		let layer_c = crate::ml::nn::Linear::new_from_matrices(
			crate::Matrix::new_from_f32(
				[output_features, input_features],
				&values(output_features * input_features),
			)?,
			crate::Matrix::new_from_f32([output_features], &values(output_features))?,
		)?;
		let triple = || -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
			let a = layer.forward(&input)?;
			let b = layer_b.forward(&input)?;
			let c = layer_c.forward(&input)?;
			Ok((a.read_f32()?, b.read_f32()?, c.read_f32()?))
		};
		let expected_triple = triple()?;
		let before = device.fusion_census().snapshot();
		let hold = ScopedVar::set("OA_TILE_LINEAR_TRIPLE", std::ffi::OsStr::new("1"));
		let _fusion6 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		let _forced6 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
		let actual_triple = triple()?;
		drop(hold);
		let census = device.fusion_census().snapshot();
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"the fused linear_triple specialization must be admitted: {census}"
		);
		for (which, (actual, expected)) in [
			(&actual_triple.0, &expected_triple.0),
			(&actual_triple.1, &expected_triple.1),
			(&actual_triple.2, &expected_triple.2),
		]
		.into_iter()
		.enumerate()
		{
			let diffs: Vec<usize> = actual
				.iter()
				.zip(expected.iter())
				.enumerate()
				.filter_map(|(i, (a, e))| (a.to_bits() != e.to_bits()).then_some(i))
				.collect();
			if !diffs.is_empty() {
				eprintln!(
					"triple output {which}: {} diffs at {:?}; actual[0..4]={:?} expected {:?}/{:?}/{:?}",
					diffs.len(),
					&diffs[..diffs.len().min(8)],
					&actual[..4],
					&expected_triple.0[..4],
					&expected_triple.1[..4],
					&expected_triple.2[..4]
				);
				panic!("triple output {which} mismatched");
			}
		}

		// Gated silu_mul: a packed [gate|up] projection halves into the
		// fused MlLinearGatedActF32 node; the packed linear output and
		// the half-width act output must both match the incumbent.
		if output_features % 2 == 0 {
			let width = output_features / 2;
			let gated = || -> Result<(Vec<f32>, Vec<f32>)> {
				let linear = layer.forward(&input)?;
				let act = crate::ml::matrix::silu_mul(&linear, width)?;
				Ok((linear.read_f32()?, act.read_f32()?))
			};
			let expected_gated = gated()?;
			let before = device.fusion_census().snapshot();
			let hold = ScopedVar::set("OA_TILE_LINEAR_ACT", std::ffi::OsStr::new("1"));
			let _fusion7 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced7 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			let actual_gated = gated()?;
			drop(hold);
			let census = device.fusion_census().snapshot();
			assert!(
				census.jit_admitted_count() + census.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"the fused linear_gated_act specialization must be admitted: {census}"
			);
			assert_eq!(
				actual_gated, expected_gated,
				"fused gated outputs differ {batch}x{input_features}x{output_features}"
			);
		}

		// Concat-linear: concat part writes into one shared buffer
		// followed by a linear over the packed K axis fuse into
		// MlLinearConcatF32; the materialized concat and the linear
		// output must match the incumbent fused kernel.
		let width0 = input_features / 3;
		let width1 = input_features / 3;
		let width2 = input_features - width0 - width1;
		let parts = [
			crate::Matrix::new_from_f32([batch, width0], &values(batch * width0))?,
			crate::Matrix::new_from_f32([batch, width1], &values(batch * width1))?,
			crate::Matrix::new_from_f32([batch, width2], &values(batch * width2))?,
		];
		let concat_layer = crate::ml::nn::Linear::new_from_matrices(
			crate::Matrix::new_from_f32(
				[output_features, input_features],
				&values(output_features * input_features),
			)?,
			crate::Matrix::new_from_f32([output_features], &values(output_features))?,
		)?;
		let fused_concat = || -> Result<(Vec<f32>, Vec<f32>)> {
			let concat = crate::matrix::concat(&parts, 1)?;
			let linear = concat_layer.forward(&concat)?;
			Ok((concat.read_f32()?, linear.read_f32()?))
		};
		let expected_concat = fused_concat()?;
		let before = device.fusion_census().snapshot();
		let hold = ScopedVar::set("OA_TILE_LINEAR_CONCAT", std::ffi::OsStr::new("1"));
		let _fusion8 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		let _forced8 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
		let actual_concat = fused_concat()?;
		drop(hold);
		let census = device.fusion_census().snapshot();
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"the fused linear_concat specialization must be admitted: {census}"
		);
		assert_eq!(
			actual_concat, expected_concat,
			"fused concat outputs differ {batch}x{input_features}x{output_features}"
		);
		// The fused `linear -> add -> layer_norm` node forms at flush when
		// the row fits one column tile; all three materialized outputs
		// (linear, sum, norm) must match the incumbent fused kernel.
		if output_features <= 64 {
			let residual =
				crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
			let norm_weight = crate::Matrix::new_from_f32([output_features], &values(output_features))?;
			let norm_bias = crate::Matrix::new_from_f32([output_features], &values(output_features))?;
			let norm =
				crate::ml::nn::LayerNorm::new_from_matrices(norm_weight.clone(), norm_bias.clone(), 1e-5)?;
			let fused_norm = || -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
				let linear = layer.forward(&input)?;
				let sum = linear.add(&residual)?;
				let output = norm.forward(&sum)?;
				Ok((linear.read_f32()?, sum.read_f32()?, output.read_f32()?))
			};
			let expected_norm = fused_norm()?;
			let before = device.fusion_census().snapshot();
			let hold = ScopedVar::set("OA_TILE_RESIDUAL_NORM", std::ffi::OsStr::new("1"));
			let _fusion9 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced9 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			let actual_norm = fused_norm()?;
			drop(hold);
			let census = device.fusion_census().snapshot();
			assert!(
				census.jit_admitted_count() + census.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"the fused linear_residual_norm specialization must be admitted: {census}"
			);
			assert_eq!(
				actual_norm, expected_norm,
				"fused residual_norm outputs differ {batch}x{input_features}x{output_features}"
			);
		}
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_sum_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x51ed270b16eb320bu64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2011) as f32 - 1000.0) / 383.0
			})
			.collect()
	};
	for (shape, dim) in [
		(vec![64usize, 32usize], -1i32),
		(vec![4, 8, 3], 1),
		(vec![5, 1031], -1),
		(vec![1, 1], -1),
		(vec![7, 511, 3], 1),
	] {
		let count = shape.iter().product::<usize>();
		let data = values(count);
		let x = crate::Matrix::new_from_f32(shape.as_slice(), &data)?;
		let expected = crate::matrix::sum(&x, dim)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_SUM",
			|| crate::matrix::sum(&x, -1).and_then(|m| m.read_f32()),
			&crate::matrix::sum(&x, -1)?.read_f32()?,
			&format!("sum {shape:?}"),
		)?;
		// The wave candidate changes reduction order versus the
		// shared tree, so it qualifies against a host f64 oracle
		// under the f32 reduction bound (count-1)·eps·Σ|x|, not
		// incumbent bits.
		{
			let before = device.fusion_census().snapshot();
			let actual = {
				let _route = ScopedVar::set("OA_TILE_SUM_WAVE", std::ffi::OsStr::new("1"));
				let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
				let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
				crate::matrix::sum(&x, -1)?.read_f32()?
			};
			let census = device.fusion_census().snapshot();
			assert!(
				census.jit_admitted_count() + census.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"sum_wave {shape:?}: the specialization must be admitted: {census}"
			);
			let host: Vec<f64> = data.iter().map(|v| *v as f64).collect();
			let oracle: f64 = host.iter().sum();
			let bound =
				(count - 1) as f64 * f32::EPSILON as f64 * host.iter().map(|v| v.abs()).sum::<f64>();
			let error = (actual[0] as f64 - oracle).abs();
			assert!(
				error <= bound,
				"sum_wave {shape:?}: error {error} exceeds the f32 \
				 reduction bound {bound} (oracle {oracle}, actual {})",
				actual[0]
			);
		}
		// `OA_TILE_REDUCE_ORDER=throughput` offers the wave candidate
		// without its per-kernel flag: both candidates stay inside the
		// f32 reduction bound, and measured selection probes the
		// least-sampled signature.
		{
			let before = device.fusion_census().snapshot();
			let actual = {
				let _base = ScopedVar::set("OA_TILE_SUM", std::ffi::OsStr::new("1"));
				let _order = ScopedVar::set("OA_TILE_REDUCE_ORDER", std::ffi::OsStr::new("throughput"));
				let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
				let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
				crate::matrix::sum(&x, -1)?.read_f32()?
			};
			let census = device.fusion_census().snapshot();
			assert!(
				census.jit_admitted_count() + census.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"sum throughput {shape:?}: the specialization must be admitted: {census}"
			);
			let host: Vec<f64> = data.iter().map(|v| *v as f64).collect();
			let oracle: f64 = host.iter().sum();
			let bound =
				(count - 1) as f64 * f32::EPSILON as f64 * host.iter().map(|v| v.abs()).sum::<f64>();
			let error = (actual[0] as f64 - oracle).abs();
			assert!(
				error <= bound,
				"sum throughput {shape:?}: error {error} exceeds the f32 \
				 reduction bound {bound} (oracle {oracle}, actual {})",
				actual[0]
			);
		}
		// `deterministic` suppresses the order-changing wave candidate
		// even under its per-kernel flag: only the shared tree remains,
		// so incumbent bits are the oracle again.
		{
			let before = device.fusion_census().snapshot();
			let actual = {
				let _base = ScopedVar::set("OA_TILE_SUM", std::ffi::OsStr::new("1"));
				let _wave = ScopedVar::set("OA_TILE_SUM_WAVE", std::ffi::OsStr::new("1"));
				let _order = ScopedVar::set(
					"OA_TILE_REDUCE_ORDER",
					std::ffi::OsStr::new("deterministic"),
				);
				let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
				let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
				crate::matrix::sum(&x, -1)?.read_f32()?
			};
			let census = device.fusion_census().snapshot();
			assert!(
				census.jit_admitted_count() + census.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"sum deterministic {shape:?}: the tree specialization must run: {census}"
			);
			let expected_scalar = crate::matrix::sum(&x, -1)?.read_f32()?;
			assert_eq!(
				actual[0].to_bits(),
				expected_scalar[0].to_bits(),
				"sum deterministic {shape:?}: the tree route must reproduce incumbent bits"
			);
		}
		if dim != -1 {
			tile_route_bit_exact(
				&device,
				"OA_TILE_SUM_AXIS",
				|| crate::matrix::sum(&x, dim).and_then(|m| m.read_f32()),
				&expected,
				&format!("sum_axis {shape:?} dim {dim}"),
			)?;
		}
	}
	// The cooperative `sum_axis` variant engages once the reduced axis
	// holds at least two elements per lane (`dim >= 512`): one workgroup
	// per `(outer, inner)` slice replaces the flat serial scan, and its
	// tree order differs from the donor — so it qualifies against the
	// f64 host oracle under the per-row f32 reduction bound.
	{
		let shape = [3usize, 1031usize, 2usize];
		let dim_size = shape[1];
		let data = values(shape.iter().product());
		let x = crate::Matrix::new_from_f32(shape, &data)?;
		let before = device.fusion_census().snapshot();
		let actual = {
			let _route = ScopedVar::set("OA_TILE_SUM_AXIS", std::ffi::OsStr::new("1"));
			let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			crate::matrix::sum(&x, 1)?.read_f32()?
		};
		let census = device.fusion_census().snapshot();
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"sum_axis cooperative: the specialization must be admitted: {census}"
		);
		assert_eq!(actual.len(), shape[0] * shape[2]);
		for outer in 0..shape[0] {
			for inner in 0..shape[2] {
				let row: Vec<f64> = (0..dim_size)
					.map(|d| data[outer * dim_size * shape[2] + d * shape[2] + inner] as f64)
					.collect();
				let oracle: f64 = row.iter().sum();
				let bound =
					(dim_size - 1) as f64 * f32::EPSILON as f64 * row.iter().map(|v| v.abs()).sum::<f64>();
				let actual = actual[outer * shape[2] + inner] as f64;
				let error = (actual - oracle).abs();
				assert!(
					error <= bound,
					"sum_axis cooperative ({outer},{inner}): error {error} exceeds the f32 \
					 reduction bound {bound} (oracle {oracle}, actual {actual})"
				);
			}
		}
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_dnn_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x85ebca6b9e3779b9u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2011) as f32 - 1000.0) / 509.0
			})
			.collect()
	};
	let input = crate::Matrix::new_from_f32([1024, 32], &values(1024 * 32))?;
	let mut projection = |_engine: &Engine| -> Result<crate::ml::nn::Linear> {
		crate::ml::nn::Linear::new_from_matrices(
			crate::Matrix::new_from_f32([32, 32], &values(32 * 32))?,
			crate::Matrix::new_from_f32([32], &values(32))?,
		)
	};
	let query = projection(&engine)?;
	let key = projection(&engine)?;
	let value = projection(&engine)?;
	// The captured QKV partition only forms with epilogue fusion disabled —
	// L1's `linear_triple` claims the three projections first otherwise.
	let qkv = || -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
		let disabled = crate::runtime::EpilogueFusionDisabled::enter();
		let captured = engine.capture(|| {
			Ok((
				query.forward(&input)?,
				key.forward(&input)?,
				value.forward(&input)?,
			))
		});
		drop(disabled);
		let (plan, outputs) = captured?;
		engine.submit(&plan)?.wait()?;
		Ok((
			outputs.0.read_f32()?,
			outputs.1.read_f32()?,
			outputs.2.read_f32()?,
		))
	};
	let expected_qkv = qkv()?;
	let before = device.fusion_census().snapshot();
	let actual_qkv = {
		let _route = ScopedVar::set("OA_TILE_QKV", std::ffi::OsStr::new("1"));
		let _fusion10 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		let _forced10 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
		qkv()?
	};
	let census = device.fusion_census().snapshot();
	assert!(
		census.jit_admitted_count() + census.jit_reused_count()
			> before.jit_admitted_count() + before.jit_reused_count(),
		"the qkv_projection_bias specialization must be admitted: {census}"
	);
	assert_eq!(actual_qkv, expected_qkv, "qkv outputs differ");

	let gate = crate::ml::nn::Linear::new_from_matrices(
		crate::Matrix::new_from_f32([64, 32], &values(64 * 32))?,
		crate::Matrix::new_from_f32([64], &values(64))?,
	)?;
	let up = crate::ml::nn::Linear::new_from_matrices(
		crate::Matrix::new_from_f32([64, 32], &values(64 * 32))?,
		crate::Matrix::new_from_f32([64], &values(64))?,
	)?;
	let gate_up = || -> Result<Vec<f32>> {
		let (plan, output) = engine.capture(|| {
			let gate_output = gate.forward(&input)?;
			let up_output = up.forward(&input)?;
			crate::ml::matrix::swiglu(&gate_output, &up_output)
		})?;
		engine.submit(&plan)?.wait()?;
		output.read_f32()
	};
	let expected_gate_up = gate_up()?;
	let before = device.fusion_census().snapshot();
	let actual_gate_up = {
		let _route = ScopedVar::set("OA_TILE_GATE_UP", std::ffi::OsStr::new("1"));
		let _fusion11 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		let _forced11 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
		gate_up()?
	};
	let census = device.fusion_census().snapshot();
	assert!(
		census.jit_admitted_count() + census.jit_reused_count()
			> before.jit_admitted_count() + before.jit_reused_count(),
		"the gate_up_swiglu_bias specialization must be admitted: {census}"
	);
	assert_eq!(actual_gate_up, expected_gate_up, "gate/up outputs differ");
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_grouped_linear_act_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x51f15eeda5u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 997.0
			})
			.collect()
	};
	// (rows, inner, outputs, offsets): tile-aligned single-expert
	// tiles, an expert boundary inside a 32-row tile, and every tail.
	for (rows, inner, outputs, boundaries) in [
		(96usize, 48usize, 64usize, vec![0u32, 32, 64, 96]),
		(70, 24, 48, vec![0, 20, 45, 70]),
		(8, 20, 16, vec![0, 4, 8]),
		(33, 16, 34, vec![0, 33, 33]),
	] {
		let experts = boundaries.len() - 1;
		let input = crate::Matrix::new_from_f32([rows, inner], &values(rows * inner))?;
		let weight = crate::Matrix::new_from_f32(
			[experts, outputs, inner],
			&values(experts * outputs * inner),
		)?;
		let bias = crate::Matrix::new_from_f32([experts, outputs], &values(experts * outputs))?;
		let offsets = crate::Matrix::new_from_slice([experts + 1], &boundaries)?;
		let gate_width = outputs / 2;
		let fused = || -> Result<(Vec<f32>, Vec<f32>)> {
			let linear = crate::ml::matrix::grouped_linear_m(&input, &weight, &bias, &offsets)?;
			let act = crate::ml::matrix::silu_mul(&linear, gate_width)?;
			Ok((linear.read_f32()?, act.read_f32()?))
		};
		let expected = fused()?;
		let label = format!("grouped {rows}x{inner}x{outputs} e{experts} {boundaries:?}");
		// Unfused donors: the same schedule without the act epilogue
		// (`MlGroupedLinearMF32`) and without bias (`MlGroupedGemmMF32`).
		let expected_linear_only =
			crate::ml::matrix::grouped_linear_m(&input, &weight, &bias, &offsets)?.read_f32()?;
		let expected_gemm = crate::ml::matrix::grouped_gemm_m(&input, &weight, &offsets)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_GROUPED_LINEAR",
			|| {
				crate::ml::matrix::grouped_linear_m(&input, &weight, &bias, &offsets)
					.and_then(|m| m.read_f32())
			},
			&expected_linear_only,
			&format!("{label} unfused"),
		)?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_GROUPED_GEMM",
			|| crate::ml::matrix::grouped_gemm_m(&input, &weight, &offsets).and_then(|m| m.read_f32()),
			&expected_gemm,
			&format!("{label} gemm"),
		)?;
		let before = device.fusion_census().snapshot();
		let actual = {
			let _route = ScopedVar::set("OA_TILE_GROUPED_LINEAR_ACT", std::ffi::OsStr::new("1"));
			let _fusion12 = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced12 = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			fused()?
		};
		let census = device.fusion_census().snapshot();
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"{label}: the grouped specialization must be admitted: {census}"
		);
		for (name, actual, expected) in [
			("linear", &actual.0, &expected.0),
			("act", &actual.1, &expected.1),
		] {
			for (element, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
				assert_eq!(
					actual.to_bits(),
					expected.to_bits(),
					"{label} {name} element {element}: expected {expected}, found {actual}"
				);
			}
		}
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flag"]
fn tile_grouped_data_backward_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0xbac4_9a55u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 997.0
			})
			.collect()
	};
	// (rows, inputs, outputs, boundaries): tile-aligned single-expert
	// tiles, an expert boundary inside a 32-row tile, and every tail.
	for (rows, inputs, outputs, boundaries) in [
		(96usize, 48usize, 64usize, vec![0u32, 32, 64, 96]),
		(70, 24, 48, vec![0, 20, 45, 70]),
		(8, 20, 16, vec![0, 4, 8]),
		(33, 16, 34, vec![0, 33, 33]),
	] {
		let experts = boundaries.len() - 1;
		let output_gradient = crate::Matrix::new_from_f32([rows, outputs], &values(rows * outputs))?;
		let input = crate::Matrix::new_from_f32([rows, inputs], &values(rows * inputs))?;
		let weight = crate::Matrix::new_from_f32(
			[experts, outputs, inputs],
			&values(experts * outputs * inputs),
		)?;
		let offsets = crate::Matrix::new_from_slice([experts + 1], &boundaries)?;
		// The `moe.rs` record path verbatim: the split-semantic pair
		// (data adjoint + parameter adjoint) under one contract. Only the
		// data adjoint is tile-eligible; the parameter adjoint stays
		// incumbent and anchors that the record itself is faithful.
		let record = || -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
			let input_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![rows, inputs],
				rows * inputs,
				crate::DType::F32,
			)?;
			let weight_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![experts, outputs, inputs],
				experts * outputs * inputs,
				crate::DType::F32,
			)?;
			let bias_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![experts, outputs],
				experts * outputs,
				crate::DType::F32,
			)?;
			let push = [
				PushConstant::U32(rows as u32),
				PushConstant::U32(outputs as u32),
				PushConstant::U32(inputs as u32),
				PushConstant::U32(experts as u32),
			];
			let data_buffers = [
				crate::runtime::BufferBinding::read(output_gradient.storage()),
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(offsets.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
			];
			let parameter_buffers = [
				crate::runtime::BufferBinding::read(output_gradient.storage()),
				crate::runtime::BufferBinding::read(input.storage()),
				crate::runtime::BufferBinding::read(offsets.storage()),
				crate::runtime::BufferBinding::write(weight_gradient.storage()),
				crate::runtime::BufferBinding::write(bias_gradient.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlGroupedLinearMDataBackwardF32,
					buffers: &data_buffers,
					push_constants: &push,
					workgroups: [rows.div_ceil(32) as u32, inputs.div_ceil(32) as u32, 1],
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlGroupedLinearMParameterBackwardF32,
					buffers: &parameter_buffers,
					push_constants: &push,
					workgroups: [
						outputs.div_ceil(32) as u32,
						inputs.div_ceil(32) as u32,
						experts as u32,
					],
					indirect: None,
				},
			];
			input.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::GROUPED_LINEAR_M_BACKWARD,
					inputs: &[&output_gradient, &input, &weight, &offsets],
					outputs: &[&input_gradient, &weight_gradient, &bias_gradient],
					attributes: &[],
				},
			)?;
			Ok((
				input_gradient.read_f32()?,
				weight_gradient.read_f32()?,
				bias_gradient.read_f32()?,
			))
		};
		let expected = record()?;
		let label = format!("grouped-bwd {rows}x{inputs}x{outputs} e{experts} {boundaries:?}");
		// The gemm sibling shares the data-adjoint donor body — the same
		// record path without the bias output.
		let record_gemm = || -> Result<(Vec<f32>, Vec<f32>)> {
			let input_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![rows, inputs],
				rows * inputs,
				crate::DType::F32,
			)?;
			let weight_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![experts, outputs, inputs],
				experts * outputs * inputs,
				crate::DType::F32,
			)?;
			let push = [
				PushConstant::U32(rows as u32),
				PushConstant::U32(outputs as u32),
				PushConstant::U32(inputs as u32),
				PushConstant::U32(experts as u32),
			];
			let data_buffers = [
				crate::runtime::BufferBinding::read(output_gradient.storage()),
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(offsets.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
			];
			let parameter_buffers = [
				crate::runtime::BufferBinding::read(output_gradient.storage()),
				crate::runtime::BufferBinding::read(input.storage()),
				crate::runtime::BufferBinding::read(offsets.storage()),
				crate::runtime::BufferBinding::write(weight_gradient.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlGroupedGemmMDataBackwardF32,
					buffers: &data_buffers,
					push_constants: &push,
					workgroups: [rows.div_ceil(32) as u32, inputs.div_ceil(32) as u32, 1],
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlGroupedGemmMParameterBackwardF32,
					buffers: &parameter_buffers,
					push_constants: &push,
					workgroups: [
						outputs.div_ceil(32) as u32,
						inputs.div_ceil(32) as u32,
						experts as u32,
					],
					indirect: None,
				},
			];
			input.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::GROUPED_GEMM_M_BACKWARD,
					inputs: &[&output_gradient, &input, &weight, &offsets],
					outputs: &[&input_gradient, &weight_gradient],
					attributes: &[],
				},
			)?;
			Ok((input_gradient.read_f32()?, weight_gradient.read_f32()?))
		};
		let expected_gemm = record_gemm()?;
		// `tile_route_bit_exact` asserts on a single output vector; the
		// data gradient is the route's product. Run the pair closure
		// manually so the parameter gradients compare too.
		let before = device.fusion_census().snapshot();
		let (actual, actual_gemm) = {
			let _route = ScopedVar::set("OA_TILE_GROUPED_BACKWARD", std::ffi::OsStr::new("1"));
			let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			(record()?, record_gemm()?)
		};
		let census = device.fusion_census().snapshot();
		// Both record paths carry two tile-eligible nodes (data +
		// parameter adjoint) — four jit admissions/reuses total prove
		// the whole backward family routed, not just the data adjoint.
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				>= before.jit_admitted_count() + before.jit_reused_count() + 4,
			"{label}: both backward specializations must be admitted: {census}"
		);
		for (name, actual, expected) in [
			("gemm-input", &actual_gemm.0, &expected_gemm.0),
			("gemm-weight", &actual_gemm.1, &expected_gemm.1),
		] {
			for (element, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
				assert_eq!(
					actual.to_bits(),
					expected.to_bits(),
					"{label} {name} element {element}: expected {expected}, found {actual}"
				);
			}
		}
		for (name, actual, expected) in [
			("input", &actual.0, &expected.0),
			("weight", &actual.1, &expected.1),
			("bias", &actual.2, &expected.2),
		] {
			for (element, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
				assert_eq!(
					actual.to_bits(),
					expected.to_bits(),
					"{label} {name} element {element}: expected {expected}, found {actual}"
				);
			}
		}
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flag"]
fn tile_grouped_backward_prologue_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0xf3ed_a11cu64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 997.0
			})
			.collect()
	};
	// (rows, inputs, outputs, boundaries): single-expert tiles, an expert
	// boundary inside the row tile (segment loop covers it), empty
	// experts and tails. `outputs` must stay even for the packed
	// silu_mul leg.
	for (rows, inputs, outputs, boundaries) in [
		(96usize, 48usize, 64usize, vec![0u32, 32, 64, 96]),
		(70, 24, 48, vec![0, 20, 45, 70]),
		(33, 16, 34, vec![0, 33, 33]),
	] {
		let experts = boundaries.len() - 1;
		let saved = crate::Matrix::new_from_f32([rows, outputs], &values(rows * outputs))?;
		let upstream = crate::Matrix::new_from_f32([rows, outputs], &values(rows * outputs))?;
		let input = crate::Matrix::new_from_f32([rows, inputs], &values(rows * inputs))?;
		let weight = crate::Matrix::new_from_f32(
			[experts, outputs, inputs],
			&values(experts * outputs * inputs),
		)?;
		let offsets = crate::Matrix::new_from_slice([experts + 1], &boundaries)?;
		// The production record sequence: a `relu_backward`-shaped
		// producer materializing `dy`, then the grouped data and
		// parameter adjoints consuming it. At submit the pair fuses into
		// `grouped_backward_prologue`; the parameter adjoint stays
		// incumbent and reads the materialized `dy`.
		type Outputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
		let record = || -> Result<Outputs> {
			let dy = crate::Matrix::allocate(
				input.engine_handle(),
				vec![rows, outputs],
				rows * outputs,
				crate::DType::F32,
			)?;
			let input_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![rows, inputs],
				rows * inputs,
				crate::DType::F32,
			)?;
			let weight_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![experts, outputs, inputs],
				experts * outputs * inputs,
				crate::DType::F32,
			)?;
			let bias_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![experts, outputs],
				experts * outputs,
				crate::DType::F32,
			)?;
			let elements = (rows * outputs) as u32;
			let adjoint_push = [
				PushConstant::U32(rows as u32),
				PushConstant::U32(outputs as u32),
				PushConstant::U32(inputs as u32),
				PushConstant::U32(experts as u32),
			];
			let producer_buffers = [
				crate::runtime::BufferBinding::read(saved.storage()),
				crate::runtime::BufferBinding::read(upstream.storage()),
				crate::runtime::BufferBinding::write(dy.storage()),
			];
			let data_buffers = [
				crate::runtime::BufferBinding::read(dy.storage()),
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(offsets.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
			];
			let parameter_buffers = [
				crate::runtime::BufferBinding::read(dy.storage()),
				crate::runtime::BufferBinding::read(input.storage()),
				crate::runtime::BufferBinding::read(offsets.storage()),
				crate::runtime::BufferBinding::write(weight_gradient.storage()),
				crate::runtime::BufferBinding::write(bias_gradient.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlReluBackwardF32,
					buffers: &producer_buffers,
					push_constants: &[PushConstant::U32(elements)],
					workgroups: KernelId::MlReluBackwardF32.linear_workgroups(elements),
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlGroupedLinearMDataBackwardF32,
					buffers: &data_buffers,
					push_constants: &adjoint_push,
					workgroups: [rows.div_ceil(32) as u32, inputs.div_ceil(32) as u32, 1],
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlGroupedLinearMParameterBackwardF32,
					buffers: &parameter_buffers,
					push_constants: &adjoint_push,
					workgroups: [
						outputs.div_ceil(32) as u32,
						inputs.div_ceil(32) as u32,
						experts as u32,
					],
					indirect: None,
				},
			];
			input.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::GROUPED_LINEAR_M_BACKWARD,
					inputs: &[&upstream, &input, &weight, &offsets],
					outputs: &[&input_gradient, &weight_gradient, &bias_gradient],
					attributes: &[],
				},
			)?;
			Ok((
				dy.read_f32()?,
				input_gradient.read_f32()?,
				weight_gradient.read_f32()?,
				bias_gradient.read_f32()?,
			))
		};
		let expected = record()?;
		let label = format!("grouped-prologue {rows}x{inputs}x{outputs} e{experts}");
		let before = device.fusion_census().snapshot();
		let actual = {
			let _route = ScopedVar::set("OA_TILE_GROUPED_BACKWARD", std::ffi::OsStr::new("1"));
			let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			record()?
		};
		let census = device.fusion_census().snapshot();
		// The fused prologue node and the surviving parameter adjoint are
		// both tile-eligible — two jit admissions/reuses prove the fused
		// route fired, not just the unfused parameter path.
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				>= before.jit_admitted_count() + before.jit_reused_count() + 2,
			"{label}: prologue tile route never admitted"
		);
		// The prologue artifact can only be published by
		// `specialize_grouped_backward_prologue_node`, so a live pipeline
		// under its block key proves the fused node itself admitted the
		// tile route — not merely one of the sibling adjoints.
		let program = crate::runtime::dna::ir::tile::BlockProgram::grouped_backward_prologue()?;
		let key = crate::runtime::device::logical::specialization::block_key(
			&program,
			cfg!(oa_profile = "compatibility"),
		);
		let applied = device
			.fusion_census()
			.pair_diagnostics()
			.iter()
			.map(|stats| stats.to_string())
			.find(|stats| stats.contains("rule=grouped_backward_prologue"))
			.and_then(|stats| {
				stats
					.split_whitespace()
					.find_map(|field| field.strip_prefix("applied=").map(str::to_string))
			})
			.and_then(|count| count.parse::<u64>().ok())
			.unwrap_or(0);
		assert!(
			applied > 0,
			"{label}: grouped_backward_prologue pair never fused"
		);
		assert!(
			device.jit_pipeline(&key).is_some(),
			"{label}: fused prologue block artifact was never admitted"
		);
		for (name, expected, actual) in [
			("dy", &expected.0, &actual.0),
			("input_gradient", &expected.1, &actual.1),
			("weight_gradient", &expected.2, &actual.2),
			("bias_gradient", &expected.3, &actual.3),
		] {
			assert_eq!(expected.len(), actual.len(), "{label} {name} length");
			for (element, (expected, actual)) in expected.iter().zip(actual).enumerate() {
				assert_eq!(
					actual.to_bits(),
					expected.to_bits(),
					"{label} {name} element {element}: expected {expected}, found {actual}"
				);
			}
		}
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flag"]
fn tile_moe_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x51f0_a3d7u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 997.0
			})
			.collect()
	};

	// Router: the fused softmax + top_k + top_k_mask chain is produced
	// by the RouterSelect merge; `OA_TILE_MOE` substitutes the tile
	// program at the fusion site. Tie-heavy logits exercise the
	// value-then-index argmax order; a repeated flagged run proves
	// stable selection.
	for (rows, experts, k) in [
		(1usize, 1usize, 1usize),
		(3, 7, 2),
		(2, 8, 8),
		(2, 64, 5),
		(2, 256, 3),
	] {
		for uniform in [true, false] {
			let host = if uniform {
				vec![0.25f32; rows * experts]
			} else {
				values(rows * experts)
			};
			let logits = crate::Matrix::new_from_f32([rows, experts], &host)?;
			type RouterOutputs = (Vec<f32>, Vec<f32>, Vec<i32>, Vec<f32>);
			let router = || -> Result<RouterOutputs> {
				let probability = crate::matrix::softmax(&logits, 1)?;
				let top = crate::matrix::top_k(&logits, k as i32, 1)?;
				let mask = crate::matrix::top_k_mask(&top.indices, experts)?;
				Ok((
					probability.read_f32()?,
					top.values.read_f32()?,
					top.indices.read::<i32>()?,
					mask.read_f32()?,
				))
			};
			let expected = router()?;
			let label = format!("router {rows}x{experts} k{k} uniform={uniform}");
			let before = device.fusion_census().snapshot();
			let actual = {
				let _route = ScopedVar::set("OA_TILE_MOE", std::ffi::OsStr::new("1"));
				let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
				let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
				router()?
			};
			let repeated = {
				let _route = ScopedVar::set("OA_TILE_MOE", std::ffi::OsStr::new("1"));
				let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
				let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
				router()?
			};
			let census = device.fusion_census().snapshot();
			assert!(
				census.jit_admitted_count() + census.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"{label}: the router tile route must be admitted: {census}"
			);
			for (name, actual, repeated, expected) in [
				("probabilities", &actual.0, &repeated.0, &expected.0),
				("values", &actual.1, &repeated.1, &expected.1),
				("mask", &actual.3, &repeated.3, &expected.3),
			] {
				for (index, ((actual, repeated), expected)) in actual
					.iter()
					.zip(repeated.iter())
					.zip(expected.iter())
					.enumerate()
				{
					assert_eq!(
						actual.to_bits(),
						expected.to_bits(),
						"{label} {name} element {index}: expected {expected}, found {actual}"
					);
					assert_eq!(
						actual.to_bits(),
						repeated.to_bits(),
						"{label} {name} repeat {index}"
					);
				}
			}
			for (index, ((actual, repeated), expected)) in actual
				.2
				.iter()
				.zip(repeated.2.iter())
				.zip(expected.2.iter())
				.enumerate()
			{
				assert_eq!(actual, expected, "{label} indices element {index}");
				assert_eq!(actual, repeated, "{label} indices repeat {index}");
			}
		}
	}

	// Flat donors: expert-plan fixtures with tail and multi-workgroup
	// coverage; each route is an exact-ABI substitution.
	for (tokens, experts, k, width) in [
		(3usize, 4usize, 2usize, 2usize),
		(2, 5, 3, 300),
		(1, 8, 8, 17),
	] {
		let routes = tokens * k;
		let expert_host: Vec<i32> = (0..tokens * k)
			.map(|i| ((i * 7 + 1) % experts) as i32)
			.collect();
		let expert_indices = crate::Matrix::new_from_slice([tokens, k], &expert_host)?;
		let probabilities = crate::Matrix::new_from_f32([tokens, experts], &values(tokens * experts))?;
		let plan = crate::matrix::moe_expert_plan(&expert_indices, experts)?;
		let input = crate::Matrix::new_from_f32([tokens, width], &values(tokens * width))?;
		let gates = crate::Matrix::new_from_f32([tokens, k], &values(tokens * k))?;
		let label = format!("moe {tokens}x{experts} k{k} w{width}");

		// route_weights (forward)
		let route = || -> Result<Vec<f32>> {
			crate::ml::matrix::moe_route_weights(&probabilities, &expert_indices)
				.and_then(|m| m.read_f32())
		};
		let expected_route = route()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_MOE",
			route,
			&expected_route,
			&format!("{label} route_weights"),
		)?;

		// gather
		let gather = || -> Result<Vec<f32>> {
			crate::ml::matrix::moe_gather(&input, &plan.packed_token, &plan.inverse)
				.and_then(|m| m.read_f32())
		};
		let packed_host = gather()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_MOE",
			gather,
			&packed_host,
			&format!("{label} gather"),
		)?;

		// combine
		let packed = crate::Matrix::new_from_f32([routes, width], &packed_host)?;
		let combine = || -> Result<Vec<f32>> {
			crate::ml::matrix::moe_combine(&packed, &gates, &plan.inverse, &plan.packed_slot)
				.and_then(|m| m.read_f32())
		};
		let expected_combine = combine()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_MOE",
			combine,
			&expected_combine,
			&format!("{label} combine"),
		)?;

		// gather_backward: donor record path — [source, inverse, output]
		// + push [routes, row_width, tokens, routes_per_token].
		let source = crate::Matrix::new_from_f32([routes, width], &values(routes * width))?;
		let gather_backward = || -> Result<Vec<f32>> {
			let output = crate::Matrix::allocate(
				source.engine_handle(),
				vec![tokens, width],
				tokens * width,
				crate::DType::F32,
			)?;
			let push = [
				PushConstant::U32(routes as u32),
				PushConstant::U32(width as u32),
				PushConstant::U32(tokens as u32),
				PushConstant::U32(k as u32),
			];
			let kernel = KernelId::MlMoeGatherBackwardF32;
			source.engine_handle().record_semantic(
				crate::runtime::ComputeDispatch {
					kernel,
					buffers: &[
						crate::runtime::BufferBinding::read(source.storage()),
						crate::runtime::BufferBinding::read(plan.inverse.storage()),
						crate::runtime::BufferBinding::write(output.storage()),
					],
					push_constants: &push,
					workgroups: kernel.linear_workgroups((tokens * width) as u32),
					indirect: None,
				},
				crate::runtime::SemanticDispatch {
					contract: kernel
						.semantic_contract()
						.expect("gather backward contract"),
					inputs: &[&source, &plan.inverse],
					outputs: &[&output],
					attributes: &[crate::OpAttribute::SignedInteger {
						name: "output_rows".into(),
						value: tokens as i64,
					}],
				},
			)?;
			output.read_f32()
		};
		let expected_gb = gather_backward()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_MOE",
			gather_backward,
			&expected_gb,
			&format!("{label} gather_backward"),
		)?;

		// combine_backward: donor record path — [output_gradient,
		// packed, route_gate, inverse, packed_slot, packed_gradient,
		// gate_gradient] + push [tokens, k, width, routes].
		let output_gradient = crate::Matrix::new_from_f32([tokens, width], &values(tokens * width))?;
		let combine_backward = || -> Result<(Vec<f32>, Vec<f32>)> {
			let packed_gradient = crate::Matrix::allocate(
				output_gradient.engine_handle(),
				vec![routes, width],
				routes * width,
				crate::DType::F32,
			)?;
			let gate_gradient = crate::Matrix::allocate(
				output_gradient.engine_handle(),
				vec![routes],
				routes,
				crate::DType::F32,
			)?;
			let push = [
				PushConstant::U32(tokens as u32),
				PushConstant::U32(k as u32),
				PushConstant::U32(width as u32),
				PushConstant::U32(routes as u32),
			];
			let kernel = KernelId::MlMoeCombineBackwardF32;
			output_gradient.engine_handle().record_semantic(
				crate::runtime::ComputeDispatch {
					kernel,
					buffers: &[
						crate::runtime::BufferBinding::read(output_gradient.storage()),
						crate::runtime::BufferBinding::read(packed.storage()),
						crate::runtime::BufferBinding::read(gates.storage()),
						crate::runtime::BufferBinding::read(plan.inverse.storage()),
						crate::runtime::BufferBinding::read(plan.packed_slot.storage()),
						crate::runtime::BufferBinding::write(packed_gradient.storage()),
						crate::runtime::BufferBinding::write(gate_gradient.storage()),
					],
					push_constants: &push,
					workgroups: kernel.linear_workgroups((routes * width) as u32),
					indirect: None,
				},
				crate::runtime::SemanticDispatch {
					contract: kernel
						.semantic_contract()
						.expect("combine backward contract"),
					inputs: &[
						&output_gradient,
						&packed,
						&gates,
						&plan.inverse,
						&plan.packed_slot,
					],
					outputs: &[&packed_gradient, &gate_gradient],
					attributes: &[],
				},
			)?;
			Ok((packed_gradient.read_f32()?, gate_gradient.read_f32()?))
		};
		let expected_cb = combine_backward()?;
		let cb_label = format!("{label} combine_backward");
		for (which, offset) in [("packed", 0usize), ("gate", 1)] {
			let expected_part = if which == "packed" {
				expected_cb.0.clone()
			} else {
				expected_cb.1.clone()
			};
			let run = || -> Result<Vec<f32>> {
				combine_backward().map(|pair| if which == "packed" { pair.0 } else { pair.1 })
			};
			let before = device.fusion_census().snapshot();
			let actual = {
				let _route = ScopedVar::set("OA_TILE_MOE", std::ffi::OsStr::new("1"));
				let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
				let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
				run()?
			};
			let census = device.fusion_census().snapshot();
			assert!(
				census.jit_admitted_count() + census.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"{cb_label} {which}: the specialization must be admitted: {census}"
			);
			for (index, (actual, expected)) in actual.iter().zip(expected_part.iter()).enumerate() {
				assert_eq!(
					actual.to_bits(),
					expected.to_bits(),
					"{cb_label} {which} element {index}: expected {expected}, found {actual}"
				);
			}
			let _ = offset;
		}

		// route_weights_backward: [output_gradient, probabilities,
		// expert_indices, route_weights, probability_gradient] +
		// push [tokens, experts, k].
		let route_weights = crate::Matrix::new_from_f32([tokens, k], &expected_route)?;
		let og_route = crate::Matrix::new_from_f32([tokens, k], &values(tokens * k))?;
		let rw_backward = || -> Result<Vec<f32>> {
			let probability_gradient = crate::Matrix::allocate(
				probabilities.engine_handle(),
				vec![tokens, experts],
				tokens * experts,
				crate::DType::F32,
			)?;
			let push = [
				PushConstant::U32(tokens as u32),
				PushConstant::U32(experts as u32),
				PushConstant::U32(k as u32),
			];
			let kernel = KernelId::MlMoeRouteWeightsBackwardF32;
			probabilities.engine_handle().record_semantic(
				crate::runtime::ComputeDispatch {
					kernel,
					buffers: &[
						crate::runtime::BufferBinding::read(og_route.storage()),
						crate::runtime::BufferBinding::read(probabilities.storage()),
						crate::runtime::BufferBinding::read(expert_indices.storage()),
						crate::runtime::BufferBinding::read(route_weights.storage()),
						crate::runtime::BufferBinding::write(probability_gradient.storage()),
					],
					push_constants: &push,
					workgroups: kernel.linear_workgroups((tokens * experts) as u32),
					indirect: None,
				},
				crate::runtime::SemanticDispatch {
					contract: kernel
						.semantic_contract()
						.expect("route weights backward contract"),
					inputs: &[&og_route, &probabilities, &expert_indices, &route_weights],
					outputs: &[&probability_gradient],
					attributes: &[],
				},
			)?;
			probability_gradient.read_f32()
		};
		let expected_rwb = rw_backward()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_MOE",
			rw_backward,
			&expected_rwb,
			&format!("{label} route_weights_backward"),
		)?;
	}
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn tile_norm_backward_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x9c4e_7b12u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 997.0
			})
			.collect()
	};

	// layer_norm_backward: donor record path — the data adjoint
	// (`[normalized, inverse_stddev, weight, output_gradient,
	// input_gradient, weight_contribution]` + `[rows, columns]` push)
	// followed by the incumbent `sum_axis_pair` parameter reduce.
	for (rows, columns) in [(4usize, 96usize), (3, 577), (1, 1)] {
		let normalized = crate::Matrix::new_from_f32([rows, columns], &values(rows * columns))?;
		let inverse_stddev = crate::Matrix::new_from_f32([rows], &values(rows))?;
		let weight = crate::Matrix::new_from_f32([columns], &values(columns))?;
		let output_gradient = crate::Matrix::new_from_f32([rows, columns], &values(rows * columns))?;
		let record = || -> Result<Vec<f32>> {
			let input_gradient = crate::Matrix::allocate(
				normalized.engine_handle(),
				vec![rows, columns],
				rows * columns,
				crate::DType::F32,
			)?;
			let weight_contribution = crate::Matrix::allocate(
				normalized.engine_handle(),
				vec![rows, columns],
				rows * columns,
				crate::DType::F32,
			)?;
			let weight_gradient = crate::Matrix::allocate(
				normalized.engine_handle(),
				vec![columns],
				columns,
				crate::DType::F32,
			)?;
			let bias_gradient = crate::Matrix::allocate(
				normalized.engine_handle(),
				vec![columns],
				columns,
				crate::DType::F32,
			)?;
			let input_push = [
				PushConstant::U32(rows as u32),
				PushConstant::U32(columns as u32),
			];
			let sum_push = [
				PushConstant::U32(1),
				PushConstant::U32(rows as u32),
				PushConstant::U32(columns as u32),
			];
			let input_buffers = [
				crate::runtime::BufferBinding::read(normalized.storage()),
				crate::runtime::BufferBinding::read(inverse_stddev.storage()),
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(output_gradient.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
				crate::runtime::BufferBinding::write(weight_contribution.storage()),
			];
			let parameter_buffers = [
				crate::runtime::BufferBinding::read(weight_contribution.storage()),
				crate::runtime::BufferBinding::read(output_gradient.storage()),
				crate::runtime::BufferBinding::write(weight_gradient.storage()),
				crate::runtime::BufferBinding::write(bias_gradient.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlLayerNormBackwardF32,
					buffers: &input_buffers,
					push_constants: &input_push,
					workgroups: [rows as u32, 1, 1],
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlSumAxisPairF32,
					buffers: &parameter_buffers,
					push_constants: &sum_push,
					workgroups: KernelId::MlSumAxisPairF32.linear_workgroups(columns as u32),
					indirect: None,
				},
			];
			normalized.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::LAYER_NORM_BACKWARD,
					inputs: &[&normalized, &inverse_stddev, &weight, &output_gradient],
					outputs: &[&input_gradient, &weight_gradient, &bias_gradient],
					attributes: &[],
				},
			)?;
			let mut flat = input_gradient.read_f32()?;
			flat.extend(weight_gradient.read_f32()?);
			flat.extend(bias_gradient.read_f32()?);
			Ok(flat)
		};
		let expected = record()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_LAYER_NORM_BACKWARD",
			record,
			&expected,
			&format!("layer_norm_backward {rows}x{columns}"),
		)?;
	}

	// rms_norm_backward: `[input, weight, output_gradient,
	// input_gradient, weight_contribution]` + `[rows, columns,
	// epsilon]` push, then the incumbent `sum_axis` parameter reduce.
	for (rows, columns) in [(4usize, 96usize), (3, 577), (1, 1)] {
		let input = crate::Matrix::new_from_f32([rows, columns], &values(rows * columns))?;
		let weight = crate::Matrix::new_from_f32([columns], &values(columns))?;
		let output_gradient = crate::Matrix::new_from_f32([rows, columns], &values(rows * columns))?;
		let epsilon = 1.0e-5f32;
		let record = || -> Result<Vec<f32>> {
			let input_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![rows, columns],
				rows * columns,
				crate::DType::F32,
			)?;
			let weight_contribution = crate::Matrix::allocate(
				input.engine_handle(),
				vec![rows, columns],
				rows * columns,
				crate::DType::F32,
			)?;
			let weight_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![columns],
				columns,
				crate::DType::F32,
			)?;
			let backward_push = [
				PushConstant::U32(rows as u32),
				PushConstant::U32(columns as u32),
				PushConstant::F32(epsilon),
			];
			let parameter_push = [
				PushConstant::U32(1),
				PushConstant::U32(rows as u32),
				PushConstant::U32(columns as u32),
			];
			let backward_buffers = [
				crate::runtime::BufferBinding::read(input.storage()),
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(output_gradient.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
				crate::runtime::BufferBinding::write(weight_contribution.storage()),
			];
			let parameter_buffers = [
				crate::runtime::BufferBinding::read(weight_contribution.storage()),
				crate::runtime::BufferBinding::write(weight_gradient.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlRmsNormBackwardF32,
					buffers: &backward_buffers,
					push_constants: &backward_push,
					workgroups: [rows as u32, 1, 1],
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MatrixSumAxisF32,
					buffers: &parameter_buffers,
					push_constants: &parameter_push,
					workgroups: KernelId::MatrixSumAxisF32.linear_workgroups(columns as u32),
					indirect: None,
				},
			];
			input.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::RMS_NORM_BACKWARD,
					inputs: &[&input, &weight, &output_gradient],
					outputs: &[&input_gradient, &weight_gradient],
					attributes: &[crate::OpAttribute::Float {
						name: "epsilon".into(),
						value: f64::from(epsilon),
					}],
				},
			)?;
			let mut flat = input_gradient.read_f32()?;
			flat.extend(weight_gradient.read_f32()?);
			Ok(flat)
		};
		let expected = record()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_RMS_NORM_BACKWARD",
			record,
			&expected,
			&format!("rms_norm_backward {rows}x{columns}"),
		)?;
	}
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn tile_linear_backward_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x2bd9_7f3au64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2011) as f32 - 1000.0) / 503.0
			})
			.collect()
	};
	let admitted = |rule: &str| -> bool {
		device
			.fusion_census()
			.merge_diagnostics()
			.iter()
			.map(|stats| stats.to_string())
			.chain(
				device
					.fusion_census()
					.pair_diagnostics()
					.iter()
					.map(|stats| stats.to_string()),
			)
			.find(|stats| stats.contains(&format!("rule={rule} ")))
			.and_then(|stats| {
				stats
					.split_whitespace()
					.find_map(|field| field.strip_prefix("applied=").map(str::to_string))
			})
			.and_then(|count| count.parse::<u64>().ok())
			.unwrap_or(0)
			> 0
	};
	let pipeline_admitted =
		|program: &crate::runtime::dna::ir::tile::BlockProgram, label: &str| -> Result<()> {
			let key = crate::runtime::device::logical::specialization::block_key(
				program,
				cfg!(oa_profile = "compatibility"),
			);
			assert!(
				device.jit_pipeline(&key).is_some(),
				"{label}: fused block artifact was never admitted"
			);
			Ok(())
		};
	let bit_exact = |label: &str, expected: &[(String, Vec<f32>)], actual: &[(String, Vec<f32>)]| {
		for ((name, expected), (_, actual)) in expected.iter().zip(actual) {
			assert_eq!(expected.len(), actual.len(), "{label} {name} length");
			for (element, (expected, actual)) in expected.iter().zip(actual).enumerate() {
				assert_eq!(
					actual.to_bits(),
					expected.to_bits(),
					"{label} {name} element {element}: expected {expected}, found {actual}"
				);
			}
		}
	};
	let dimensions = |batch: usize, input: usize, output: usize| {
		[
			PushConstant::U32(batch as u32),
			PushConstant::U32(input as u32),
			PushConstant::U32(output as u32),
		]
	};

	// Member: `MlLinearBackwardF32` standalone — the staged NT GEMM.
	for (batch, input_features, output_features) in [(64usize, 48usize, 32usize), (33, 20, 34)] {
		let weight = crate::Matrix::new_from_f32(
			[output_features, input_features],
			&values(output_features * input_features),
		)?;
		let output_gradient =
			crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
		let record = || -> Result<Vec<f32>> {
			let input_gradient = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, input_features],
				batch * input_features,
				crate::DType::F32,
			)?;
			let buffers = [
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(output_gradient.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
			];
			let dispatch = crate::runtime::ComputeDispatch {
				kernel: KernelId::MlLinearBackwardF32,
				buffers: &buffers,
				push_constants: &dimensions(batch, input_features, output_features),
				workgroups: KernelId::MlLinearBackwardF32
					.output_workgroups(batch as u32, input_features as u32),
				indirect: None,
			};
			weight.engine_handle().record_semantic(
				dispatch,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::LINEAR_BACKWARD,
					inputs: &[&weight, &output_gradient],
					outputs: &[&input_gradient],
					attributes: &[],
				},
			)?;
			input_gradient.read_f32()
		};
		let expected = record()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_LINEAR_BACKWARD",
			record,
			&expected,
			&format!("linear_backward {batch}x{input_features}x{output_features}"),
		)?;
	}

	// Prologue region: relu_backward producer → linear_backward consumer,
	// fused into `MlLinearBackwardPrologueF32` which republishes `dy`.
	for (batch, input_features, output_features) in [(64usize, 48usize, 32usize), (33, 20, 34)] {
		let saved =
			crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
		let upstream =
			crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
		let weight = crate::Matrix::new_from_f32(
			[output_features, input_features],
			&values(output_features * input_features),
		)?;
		type Outputs = (Vec<f32>, Vec<f32>);
		let record = || -> Result<Outputs> {
			let dy = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, output_features],
				batch * output_features,
				crate::DType::F32,
			)?;
			let input_gradient = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, input_features],
				batch * input_features,
				crate::DType::F32,
			)?;
			let elements = (batch * output_features) as u32;
			let producer_buffers = [
				crate::runtime::BufferBinding::read(saved.storage()),
				crate::runtime::BufferBinding::read(upstream.storage()),
				crate::runtime::BufferBinding::write(dy.storage()),
			];
			let backward_buffers = [
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(dy.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlReluBackwardF32,
					buffers: &producer_buffers,
					push_constants: &[PushConstant::U32(elements)],
					workgroups: KernelId::MlReluBackwardF32.linear_workgroups(elements),
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlLinearBackwardF32,
					buffers: &backward_buffers,
					push_constants: &dimensions(batch, input_features, output_features),
					workgroups: KernelId::MlLinearBackwardF32
						.output_workgroups(batch as u32, input_features as u32),
					indirect: None,
				},
			];
			weight.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::LINEAR_BACKWARD,
					inputs: &[&weight, &upstream],
					outputs: &[&input_gradient],
					attributes: &[],
				},
			)?;
			Ok((dy.read_f32()?, input_gradient.read_f32()?))
		};
		let expected = record()?;
		let label = format!("prologue {batch}x{input_features}x{output_features}");
		let actual = {
			let _route = ScopedVar::set("OA_TILE_LINEAR_BACKWARD", std::ffi::OsStr::new("1"));
			let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			record()?
		};
		assert!(
			admitted("linear_backward_prologue"),
			"{label}: prologue pair never fused"
		);
		pipeline_admitted(
			&crate::runtime::dna::ir::tile::BlockProgram::linear_backward_prologue(),
			&label,
		)?;
		bit_exact(
			&label,
			&[
				("dy".to_string(), expected.0),
				("input_gradient".to_string(), expected.1),
			],
			&[
				("dy".to_string(), actual.0),
				("input_gradient".to_string(), actual.1),
			],
		);
	}

	// Norm region: layer_norm_backward → sum_axis_pair → linear_backward
	// fuses into `MlLinearBackwardNormF32`; the reduction stays incumbent.
	for (batch, input_features, output_features) in [(64usize, 48usize, 32usize), (33, 20, 34)] {
		let normalized =
			crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
		let inverse_stddev = crate::Matrix::new_from_f32([batch], &values(batch))?;
		let gamma = crate::Matrix::new_from_f32([output_features], &values(output_features))?;
		let upstream =
			crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
		let weight = crate::Matrix::new_from_f32(
			[output_features, input_features],
			&values(output_features * input_features),
		)?;
		type Outputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
		let record = || -> Result<Outputs> {
			let dx_norm = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, output_features],
				batch * output_features,
				crate::DType::F32,
			)?;
			let contribution = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, output_features],
				batch * output_features,
				crate::DType::F32,
			)?;
			let gamma_gradient = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![output_features],
				output_features,
				crate::DType::F32,
			)?;
			let beta_gradient = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![output_features],
				output_features,
				crate::DType::F32,
			)?;
			let input_gradient = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, input_features],
				batch * input_features,
				crate::DType::F32,
			)?;
			let norm_buffers = [
				crate::runtime::BufferBinding::read(normalized.storage()),
				crate::runtime::BufferBinding::read(inverse_stddev.storage()),
				crate::runtime::BufferBinding::read(gamma.storage()),
				crate::runtime::BufferBinding::read(upstream.storage()),
				crate::runtime::BufferBinding::write(dx_norm.storage()),
				crate::runtime::BufferBinding::write(contribution.storage()),
			];
			let reduction_buffers = [
				crate::runtime::BufferBinding::read(contribution.storage()),
				crate::runtime::BufferBinding::read(upstream.storage()),
				crate::runtime::BufferBinding::write(gamma_gradient.storage()),
				crate::runtime::BufferBinding::write(beta_gradient.storage()),
			];
			let backward_buffers = [
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(dx_norm.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlLayerNormBackwardF32,
					buffers: &norm_buffers,
					push_constants: &[
						PushConstant::U32(batch as u32),
						PushConstant::U32(output_features as u32),
					],
					workgroups: [batch as u32, 1, 1],
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlSumAxisPairF32,
					buffers: &reduction_buffers,
					push_constants: &[
						PushConstant::U32(1),
						PushConstant::U32(batch as u32),
						PushConstant::U32(output_features as u32),
					],
					workgroups: KernelId::MlSumAxisPairF32.linear_workgroups(output_features as u32),
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlLinearBackwardF32,
					buffers: &backward_buffers,
					push_constants: &dimensions(batch, input_features, output_features),
					workgroups: KernelId::MlLinearBackwardF32
						.output_workgroups(batch as u32, input_features as u32),
					indirect: None,
				},
			];
			weight.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::LAYER_NORM_BACKWARD,
					inputs: &[&normalized, &inverse_stddev, &gamma, &upstream],
					outputs: &[&input_gradient, &gamma_gradient, &beta_gradient],
					attributes: &[],
				},
			)?;
			Ok((
				dx_norm.read_f32()?,
				contribution.read_f32()?,
				gamma_gradient.read_f32()?,
				beta_gradient.read_f32()?,
				input_gradient.read_f32()?,
			))
		};
		let expected = record()?;
		let label = format!("norm {batch}x{input_features}x{output_features}");
		let actual = {
			let _route = ScopedVar::set("OA_TILE_LINEAR_BACKWARD", std::ffi::OsStr::new("1"));
			let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			record()?
		};
		assert!(
			admitted("linear_backward_norm"),
			"{label}: norm merge never fused"
		);
		pipeline_admitted(
			&crate::runtime::dna::ir::tile::BlockProgram::linear_backward_norm(),
			&label,
		)?;
		bit_exact(
			&label,
			&[
				("dx_norm".to_string(), expected.0),
				("contribution".to_string(), expected.1),
				("gamma_gradient".to_string(), expected.2),
				("beta_gradient".to_string(), expected.3),
				("input_gradient".to_string(), expected.4),
			],
			&[
				("dx_norm".to_string(), actual.0),
				("contribution".to_string(), actual.1),
				("gamma_gradient".to_string(), actual.2),
				("beta_gradient".to_string(), actual.3),
				("input_gradient".to_string(), actual.4),
			],
		);
	}

	// Norm+add region: layer_norm_backward → sum_axis_pair → matrix.add →
	// linear_backward fuses into `MlLinearBackwardNormAddF32`.
	for (batch, input_features, output_features) in [(64usize, 48usize, 32usize), (33, 20, 34)] {
		let normalized =
			crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
		let inverse_stddev = crate::Matrix::new_from_f32([batch], &values(batch))?;
		let gamma = crate::Matrix::new_from_f32([output_features], &values(output_features))?;
		let upstream =
			crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
		let addend =
			crate::Matrix::new_from_f32([batch, output_features], &values(batch * output_features))?;
		let weight = crate::Matrix::new_from_f32(
			[output_features, input_features],
			&values(output_features * input_features),
		)?;
		type Outputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
		let record = || -> Result<Outputs> {
			let dx_norm = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, output_features],
				batch * output_features,
				crate::DType::F32,
			)?;
			let contribution = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, output_features],
				batch * output_features,
				crate::DType::F32,
			)?;
			let gamma_gradient = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![output_features],
				output_features,
				crate::DType::F32,
			)?;
			let beta_gradient = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![output_features],
				output_features,
				crate::DType::F32,
			)?;
			let combined = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, output_features],
				batch * output_features,
				crate::DType::F32,
			)?;
			let input_gradient = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, input_features],
				batch * input_features,
				crate::DType::F32,
			)?;
			let elements = (batch * output_features) as u32;
			let norm_buffers = [
				crate::runtime::BufferBinding::read(normalized.storage()),
				crate::runtime::BufferBinding::read(inverse_stddev.storage()),
				crate::runtime::BufferBinding::read(gamma.storage()),
				crate::runtime::BufferBinding::read(upstream.storage()),
				crate::runtime::BufferBinding::write(dx_norm.storage()),
				crate::runtime::BufferBinding::write(contribution.storage()),
			];
			let reduction_buffers = [
				crate::runtime::BufferBinding::read(contribution.storage()),
				crate::runtime::BufferBinding::read(upstream.storage()),
				crate::runtime::BufferBinding::write(gamma_gradient.storage()),
				crate::runtime::BufferBinding::write(beta_gradient.storage()),
			];
			let combine_buffers = [
				crate::runtime::BufferBinding::read(dx_norm.storage()),
				crate::runtime::BufferBinding::read(addend.storage()),
				crate::runtime::BufferBinding::write(combined.storage()),
			];
			let backward_buffers = [
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(combined.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlLayerNormBackwardF32,
					buffers: &norm_buffers,
					push_constants: &[
						PushConstant::U32(batch as u32),
						PushConstant::U32(output_features as u32),
					],
					workgroups: [batch as u32, 1, 1],
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlSumAxisPairF32,
					buffers: &reduction_buffers,
					push_constants: &[
						PushConstant::U32(1),
						PushConstant::U32(batch as u32),
						PushConstant::U32(output_features as u32),
					],
					workgroups: KernelId::MlSumAxisPairF32.linear_workgroups(output_features as u32),
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MatrixAddF32,
					buffers: &combine_buffers,
					push_constants: &[PushConstant::U32(elements)],
					workgroups: KernelId::MatrixAddF32.linear_workgroups(elements),
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlLinearBackwardF32,
					buffers: &backward_buffers,
					push_constants: &dimensions(batch, input_features, output_features),
					workgroups: KernelId::MlLinearBackwardF32
						.output_workgroups(batch as u32, input_features as u32),
					indirect: None,
				},
			];
			weight.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::LAYER_NORM_BACKWARD,
					inputs: &[&normalized, &inverse_stddev, &gamma, &upstream],
					outputs: &[&input_gradient, &gamma_gradient, &beta_gradient],
					attributes: &[],
				},
			)?;
			Ok((
				dx_norm.read_f32()?,
				contribution.read_f32()?,
				gamma_gradient.read_f32()?,
				beta_gradient.read_f32()?,
				combined.read_f32()?,
				input_gradient.read_f32()?,
			))
		};
		let expected = record()?;
		let label = format!("norm-add {batch}x{input_features}x{output_features}");
		let actual = {
			let _route = ScopedVar::set("OA_TILE_LINEAR_BACKWARD", std::ffi::OsStr::new("1"));
			let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			record()?
		};
		assert!(
			admitted("linear_backward_norm_add"),
			"{label}: norm-add merge never fused"
		);
		pipeline_admitted(
			&crate::runtime::dna::ir::tile::BlockProgram::linear_backward_norm_add(),
			&label,
		)?;
		bit_exact(
			&label,
			&[
				("dx_norm".to_string(), expected.0),
				("contribution".to_string(), expected.1),
				("gamma_gradient".to_string(), expected.2),
				("beta_gradient".to_string(), expected.3),
				("combined".to_string(), expected.4),
				("input_gradient".to_string(), expected.5),
			],
			&[
				("dx_norm".to_string(), actual.0),
				("contribution".to_string(), actual.1),
				("gamma_gradient".to_string(), actual.2),
				("beta_gradient".to_string(), actual.3),
				("combined".to_string(), actual.4),
				("input_gradient".to_string(), actual.5),
			],
		);
	}

	// Bmm-chain region: 16×16 bmm producer → linear_backward consumer
	// fuses into `MlLinearBackwardBmmChainF32`, which recomputes the bmm
	// band and republishes it. `columns` and `input_features` stay ≤ 32 —
	// the matcher's shared-tile budget.
	for (kernel, batch, rows, inner, columns, input_features) in [
		(
			KernelId::MlBmmTiled16F32,
			3usize,
			24usize,
			40usize,
			24usize,
			16usize,
		),
		(KernelId::MlBmmTnTiled16F32, 2, 20, 18, 30, 28),
	] {
		// kind 0 reads the left operand as `[B,R,I]`; kind 1 as `[B,I,R]` —
		// the element count is identical.
		let left = crate::Matrix::new_from_f32([batch, rows, inner], &values(batch * rows * inner))?;
		let right =
			crate::Matrix::new_from_f32([batch, inner, columns], &values(batch * inner * columns))?;
		let weight =
			crate::Matrix::new_from_f32([columns, input_features], &values(columns * input_features))?;
		type Outputs = (Vec<f32>, Vec<f32>);
		let record = || -> Result<Outputs> {
			let dy = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch, rows, columns],
				batch * rows * columns,
				crate::DType::F32,
			)?;
			let input_gradient = crate::Matrix::allocate(
				weight.engine_handle(),
				vec![batch * rows, input_features],
				batch * rows * input_features,
				crate::DType::F32,
			)?;
			let producer_buffers = [
				crate::runtime::BufferBinding::read(left.storage()),
				crate::runtime::BufferBinding::read(right.storage()),
				crate::runtime::BufferBinding::write(dy.storage()),
			];
			let backward_buffers = [
				crate::runtime::BufferBinding::read(weight.storage()),
				crate::runtime::BufferBinding::read(dy.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel,
					buffers: &producer_buffers,
					push_constants: &[
						PushConstant::U32(batch as u32),
						PushConstant::U32(rows as u32),
						PushConstant::U32(inner as u32),
						PushConstant::U32(columns as u32),
					],
					workgroups: [
						(columns as u32).div_ceil(16),
						(rows as u32).div_ceil(16),
						batch as u32,
					],
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlLinearBackwardF32,
					buffers: &backward_buffers,
					push_constants: &dimensions(batch * rows, input_features, columns),
					workgroups: KernelId::MlLinearBackwardF32
						.output_workgroups((batch * rows) as u32, input_features as u32),
					indirect: None,
				},
			];
			weight.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::LINEAR_BACKWARD,
					inputs: &[&weight, &right],
					outputs: &[&input_gradient],
					attributes: &[],
				},
			)?;
			Ok((dy.read_f32()?, input_gradient.read_f32()?))
		};
		let expected = record()?;
		let label = format!(
			"bmm-chain {:?} {batch}x{rows}x{inner}x{columns}x{input_features}",
			kernel
		);
		let actual = {
			let _route = ScopedVar::set("OA_TILE_LINEAR_BACKWARD", std::ffi::OsStr::new("1"));
			let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			record()?
		};
		assert!(
			admitted("linear_backward_bmm_chain"),
			"{label}: bmm-chain merge never fused"
		);
		pipeline_admitted(
			&crate::runtime::dna::ir::tile::BlockProgram::linear_backward_bmm_chain(),
			&label,
		)?;
		bit_exact(
			&label,
			&[
				("dy".to_string(), expected.0),
				("input_gradient".to_string(), expected.1),
			],
			&[
				("dy".to_string(), actual.0),
				("input_gradient".to_string(), actual.1),
			],
		);
	}
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn tile_cross_entropy_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x3b9a_c077u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 97.0
			})
			.collect()
	};
	let portable = |rows: u32| [rows.min(65535), rows.div_ceil(65535), 1];

	for (rows, classes) in [(4usize, 96usize), (3, 577), (1, 1), (5, 300)] {
		let logits = crate::Matrix::new_from_f32([rows, classes], &values(rows * classes))?;
		let mut target_host = (0..rows)
			.map(|row| ((row * 37 + 11) % classes) as u32)
			.collect::<Vec<u32>>();
		// An out-of-range target exercises the donor's NaN path.
		if rows > 2 {
			target_host[2] = classes as u32 + 5;
		}
		let targets = crate::Matrix::new_from_slice([rows], &target_host)?;

		// forward: [logits, targets, per_row] + [rows, classes, dtype]
		// then the incumbent SumScale mean.
		let forward = || -> Result<Vec<f32>> {
			let per_row =
				crate::Matrix::allocate(logits.engine_handle(), vec![rows], rows, crate::DType::F32)?;
			let output =
				crate::Matrix::allocate(logits.engine_handle(), Vec::new(), 1, crate::DType::F32)?;
			let loss_push = [
				PushConstant::U32(rows as u32),
				PushConstant::U32(classes as u32),
				PushConstant::U32(1),
			];
			let mean_push = [
				PushConstant::U32(rows as u32),
				PushConstant::F32(1.0 / rows as f32),
			];
			let loss_buffers = [
				crate::runtime::BufferBinding::read(logits.storage()),
				crate::runtime::BufferBinding::read(targets.storage()),
				crate::runtime::BufferBinding::write(per_row.storage()),
			];
			let mean_buffers = [
				crate::runtime::BufferBinding::read(per_row.storage()),
				crate::runtime::BufferBinding::write(output.storage()),
			];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlCrossEntropyF32,
					buffers: &loss_buffers,
					push_constants: &loss_push,
					workgroups: portable(rows as u32),
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlSumScaleF32,
					buffers: &mean_buffers,
					push_constants: &mean_push,
					workgroups: [1, 1, 1],
					indirect: None,
				},
			];
			logits.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::CROSS_ENTROPY,
					inputs: &[&logits, &targets],
					outputs: &[&output],
					attributes: &[],
				},
			)?;
			let mut flat = per_row.read_f32()?;
			flat.extend(output.read_f32()?);
			Ok(flat)
		};
		let expected = forward()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_CROSS_ENTROPY",
			forward,
			&expected,
			&format!("cross_entropy {rows}x{classes}"),
		)?;

		// backward: [logits, targets, gradient] + [rows, classes, dtype].
		let backward = || -> Result<Vec<f32>> {
			let gradient = crate::Matrix::allocate(
				logits.engine_handle(),
				vec![rows, classes],
				rows * classes,
				crate::DType::F32,
			)?;
			let push = [
				PushConstant::U32(rows as u32),
				PushConstant::U32(classes as u32),
				PushConstant::U32(1),
			];
			let kernel = KernelId::MlCrossEntropyBackwardF32;
			logits.engine_handle().record_semantic(
				crate::runtime::ComputeDispatch {
					kernel,
					buffers: &[
						crate::runtime::BufferBinding::read(logits.storage()),
						crate::runtime::BufferBinding::read(targets.storage()),
						crate::runtime::BufferBinding::write(gradient.storage()),
					],
					push_constants: &push,
					workgroups: portable(rows as u32),
					indirect: None,
				},
				crate::runtime::SemanticDispatch {
					contract: kernel.semantic_contract().expect("ce backward contract"),
					inputs: &[&logits, &targets],
					outputs: &[&gradient],
					attributes: &[],
				},
			)?;
			gradient.read_f32()
		};
		let expected = backward()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_CROSS_ENTROPY",
			backward,
			&expected,
			&format!("cross_entropy_backward {rows}x{classes}"),
		)?;
	}

	// Masked pair: half the rows masked out, valid_count driving the
	// mean and the adjoint's `inverse_valid_count`.
	for (rows, classes) in [(4usize, 96usize), (3, 577)] {
		let logits = crate::Matrix::new_from_f32([rows, classes], &values(rows * classes))?;
		let target_host = (0..rows)
			.map(|row| ((row * 29 + 7) % classes) as u32)
			.collect::<Vec<u32>>();
		let targets = crate::Matrix::new_from_slice([rows], &target_host)?;
		let mask_host = (0..rows)
			.map(|row| if row % 2 == 0 { 1.0f32 } else { 0.0 })
			.collect::<Vec<f32>>();
		let mask = crate::Matrix::new_from_f32([rows], &mask_host)?;
		let valid_count = mask_host.iter().filter(|v| **v != 0.0).count();

		let forward = || -> Result<Vec<f32>> {
			let per_row =
				crate::Matrix::allocate(logits.engine_handle(), vec![rows], rows, crate::DType::F32)?;
			let output =
				crate::Matrix::allocate(logits.engine_handle(), Vec::new(), 1, crate::DType::F32)?;
			let loss_push = [
				PushConstant::U32(rows as u32),
				PushConstant::U32(classes as u32),
				PushConstant::U32(1),
			];
			let mean_push = [
				PushConstant::U32(rows as u32),
				PushConstant::F32(1.0 / valid_count as f32),
			];
			let loss_buffers = [
				crate::runtime::BufferBinding::read(logits.storage()),
				crate::runtime::BufferBinding::read(targets.storage()),
				crate::runtime::BufferBinding::read(mask.storage()),
				crate::runtime::BufferBinding::write(per_row.storage()),
			];
			let mean_buffers = [
				crate::runtime::BufferBinding::read(per_row.storage()),
				crate::runtime::BufferBinding::write(output.storage()),
			];
			let attributes = [crate::OpAttribute::SignedInteger {
				name: "validCount".into(),
				value: valid_count as i64,
			}];
			let dispatches = [
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlMaskedCrossEntropyF32,
					buffers: &loss_buffers,
					push_constants: &loss_push,
					workgroups: portable(rows as u32),
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: KernelId::MlSumScaleF32,
					buffers: &mean_buffers,
					push_constants: &mean_push,
					workgroups: [1, 1, 1],
					indirect: None,
				},
			];
			logits.engine_handle().record_split_semantic(
				&dispatches,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::ml::MASKED_CROSS_ENTROPY,
					inputs: &[&logits, &targets, &mask],
					outputs: &[&output],
					attributes: &attributes,
				},
			)?;
			let mut flat = per_row.read_f32()?;
			flat.extend(output.read_f32()?);
			Ok(flat)
		};
		let expected = forward()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_CROSS_ENTROPY",
			forward,
			&expected,
			&format!("masked_cross_entropy {rows}x{classes}"),
		)?;

		let backward = || -> Result<Vec<f32>> {
			let gradient = crate::Matrix::allocate(
				logits.engine_handle(),
				vec![rows, classes],
				rows * classes,
				crate::DType::F32,
			)?;
			let push = [
				PushConstant::U32(rows as u32),
				PushConstant::U32(classes as u32),
				PushConstant::U32(1),
				PushConstant::U32(valid_count as u32),
			];
			let kernel = KernelId::MlMaskedCrossEntropyBackwardF32;
			let attributes = [crate::OpAttribute::SignedInteger {
				name: "valid_count".into(),
				value: valid_count as i64,
			}];
			logits.engine_handle().record_semantic(
				crate::runtime::ComputeDispatch {
					kernel,
					buffers: &[
						crate::runtime::BufferBinding::read(logits.storage()),
						crate::runtime::BufferBinding::read(targets.storage()),
						crate::runtime::BufferBinding::read(mask.storage()),
						crate::runtime::BufferBinding::write(gradient.storage()),
					],
					push_constants: &push,
					workgroups: portable(rows as u32),
					indirect: None,
				},
				crate::runtime::SemanticDispatch {
					contract: kernel
						.semantic_contract()
						.expect("masked ce backward contract"),
					inputs: &[&logits, &targets, &mask],
					outputs: &[&gradient],
					attributes: &attributes,
				},
			)?;
			gradient.read_f32()
		};
		let expected = backward()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_CROSS_ENTROPY",
			backward,
			&expected,
			&format!("masked_cross_entropy_backward {rows}x{classes}"),
		)?;
	}
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn tile_gradient_sum_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x51f2_9a3cu64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 997.0
			})
			.collect()
	};

	// layer_norm_backward_sum: MatrixAddF32 upstream sum feeding the
	// data adjoint — two-addend (a + b) and three-addend ((a + b) + c)
	// trees, then the incumbent sum_axis_pair parameter reduce.
	for (rows, columns) in [(4usize, 96usize), (3, 577), (1, 1)] {
		for addends in 2..=3usize {
			let normalized = crate::Matrix::new_from_f32([rows, columns], &values(rows * columns))?;
			let inverse_stddev = crate::Matrix::new_from_f32([rows], &values(rows))?;
			let weight = crate::Matrix::new_from_f32([columns], &values(columns))?;
			let addend_a = crate::Matrix::new_from_f32([rows, columns], &values(rows * columns))?;
			let addend_b = crate::Matrix::new_from_f32([rows, columns], &values(rows * columns))?;
			let addend_c = crate::Matrix::new_from_f32([rows, columns], &values(rows * columns))?;
			let record = || -> Result<Vec<f32>> {
				let dy = crate::Matrix::allocate(
					normalized.engine_handle(),
					vec![rows, columns],
					rows * columns,
					crate::DType::F32,
				)?;
				let input_gradient = crate::Matrix::allocate(
					normalized.engine_handle(),
					vec![rows, columns],
					rows * columns,
					crate::DType::F32,
				)?;
				let weight_contribution = crate::Matrix::allocate(
					normalized.engine_handle(),
					vec![rows, columns],
					rows * columns,
					crate::DType::F32,
				)?;
				let weight_gradient = crate::Matrix::allocate(
					normalized.engine_handle(),
					vec![columns],
					columns,
					crate::DType::F32,
				)?;
				let bias_gradient = crate::Matrix::allocate(
					normalized.engine_handle(),
					vec![columns],
					columns,
					crate::DType::F32,
				)?;
				let element_count = (rows * columns) as u32;
				let add_push = [PushConstant::U32(element_count)];
				let input_push = [
					PushConstant::U32(rows as u32),
					PushConstant::U32(columns as u32),
				];
				let sum_push = [
					PushConstant::U32(1),
					PushConstant::U32(rows as u32),
					PushConstant::U32(columns as u32),
				];
				let partial = crate::Matrix::allocate(
					normalized.engine_handle(),
					vec![rows, columns],
					rows * columns,
					crate::DType::F32,
				)?;
				let first_buffers = [
					crate::runtime::BufferBinding::read(addend_a.storage()),
					crate::runtime::BufferBinding::read(addend_b.storage()),
					crate::runtime::BufferBinding::write(partial.storage()),
				];
				let second_buffers = [
					crate::runtime::BufferBinding::read(partial.storage()),
					crate::runtime::BufferBinding::read(addend_c.storage()),
					crate::runtime::BufferBinding::write(dy.storage()),
				];
				let add_buffers = [
					crate::runtime::BufferBinding::read(addend_a.storage()),
					crate::runtime::BufferBinding::read(addend_b.storage()),
					crate::runtime::BufferBinding::write(dy.storage()),
				];
				let mut dispatches = Vec::new();
				if addends == 3 {
					dispatches.push(crate::runtime::ComputeDispatch {
						kernel: KernelId::MatrixAddF32,
						buffers: &first_buffers,
						push_constants: &add_push,
						workgroups: KernelId::MatrixAddF32.linear_workgroups(element_count),
						indirect: None,
					});
					dispatches.push(crate::runtime::ComputeDispatch {
						kernel: KernelId::MatrixAddF32,
						buffers: &second_buffers,
						push_constants: &add_push,
						workgroups: KernelId::MatrixAddF32.linear_workgroups(element_count),
						indirect: None,
					});
				} else {
					dispatches.push(crate::runtime::ComputeDispatch {
						kernel: KernelId::MatrixAddF32,
						buffers: &add_buffers,
						push_constants: &add_push,
						workgroups: KernelId::MatrixAddF32.linear_workgroups(element_count),
						indirect: None,
					});
				}
				let input_buffers = [
					crate::runtime::BufferBinding::read(normalized.storage()),
					crate::runtime::BufferBinding::read(inverse_stddev.storage()),
					crate::runtime::BufferBinding::read(weight.storage()),
					crate::runtime::BufferBinding::read(dy.storage()),
					crate::runtime::BufferBinding::write(input_gradient.storage()),
					crate::runtime::BufferBinding::write(weight_contribution.storage()),
				];
				let parameter_buffers = [
					crate::runtime::BufferBinding::read(weight_contribution.storage()),
					crate::runtime::BufferBinding::read(dy.storage()),
					crate::runtime::BufferBinding::write(weight_gradient.storage()),
					crate::runtime::BufferBinding::write(bias_gradient.storage()),
				];
				dispatches.push(crate::runtime::ComputeDispatch {
					kernel: KernelId::MlLayerNormBackwardF32,
					buffers: &input_buffers,
					push_constants: &input_push,
					workgroups: [rows as u32, 1, 1],
					indirect: None,
				});
				dispatches.push(crate::runtime::ComputeDispatch {
					kernel: KernelId::MlSumAxisPairF32,
					buffers: &parameter_buffers,
					push_constants: &sum_push,
					workgroups: KernelId::MlSumAxisPairF32.linear_workgroups(columns as u32),
					indirect: None,
				});
				normalized.engine_handle().record_split_semantic(
					&dispatches,
					crate::runtime::SemanticDispatch {
						contract: crate::core::operation::ml::LAYER_NORM_BACKWARD,
						inputs: &[&normalized, &inverse_stddev, &weight, &dy],
						outputs: &[&input_gradient, &weight_gradient, &bias_gradient],
						attributes: &[],
					},
				)?;
				let mut flat = input_gradient.read_f32()?;
				flat.extend(weight_gradient.read_f32()?);
				flat.extend(bias_gradient.read_f32()?);
				flat.extend(dy.read_f32()?);
				drop(partial);
				Ok(flat)
			};
			// The fused route is preferred over the member tile
			// substitution, so the oracle is the incumbent
			// `layer_norm_backward_sum` region under the same fusion
			// admission the specialized run sees.
			let expected = {
				let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
				record()?
			};
			tile_route_bit_exact(
				&device,
				"OA_TILE_LAYER_NORM_BACKWARD",
				record,
				&expected,
				&format!("layer_norm_backward_sum{addends} {rows}x{columns}"),
			)?;
		}
	}

	// embedding_backward_sum: the tape-internal `matrix.add` upstream
	// sum feeding both embedding table adjoints — a live Matrix would
	// keep the sum observable and block the fusion, so the exercise
	// runs the production autograd path. Two identically-seeded layer
	// sets keep the incumbent and tile backward runs independent.
	let tokens = crate::Matrix::new_from_slice([4], &[0u32, 2, 1, 3])?;
	let positions = crate::Matrix::new_from_slice([4], &[0u32, 1, 2, 3])?;
	let target_values: Vec<f32> = (0..8).map(|i| ((i * 7 % 9) as f32 - 4.0) / 3.0).collect();
	let target = crate::Matrix::new_from_f32([4, 2], &target_values)?;
	let layers = || -> Result<(
		crate::ml::nn::Embedding,
		crate::ml::nn::Embedding,
		crate::ml::nn::Linear,
		crate::ml::nn::Linear,
	)> {
		Ok((
			crate::ml::nn::Embedding::new_with_seed(4, 3, 0x7a1)?,
			crate::ml::nn::Embedding::new_with_seed(4, 3, 0x7a2)?,
			crate::ml::nn::Linear::new_with_seed(3, 2, 0x7a3)?,
			crate::ml::nn::Linear::new_with_seed(3, 2, 0x7a4)?,
		))
	};
	let reference = layers()?;
	let candidate = layers()?;
	let backward = |layers: &(
		crate::ml::nn::Embedding,
		crate::ml::nn::Embedding,
		crate::ml::nn::Linear,
		crate::ml::nn::Linear,
	)|
	 -> Result<Vec<f32>> {
		let tape = crate::ml::GradientTape::new();
		let summed = crate::matrix::add(&layers.0.forward(&tokens)?, &layers.1.forward(&positions)?)?;
		let loss = crate::matrix::add(
			&crate::ml::loss::mse(&layers.2.forward(&summed)?, &target)?,
			&crate::ml::loss::mse(&layers.3.forward(&summed)?, &target)?,
		)?;
		tape.backward(&loss)?;
		let mut flat = Vec::new();
		for parameter in layers
			.0
			.parameters()
			.iter()
			.chain(layers.1.parameters().iter())
		{
			flat.extend(parameter.gradient().unwrap().read_f32()?);
		}
		Ok(flat)
	};
	// Same fused-preferred oracle: the incumbent `embedding_backward_sum`
	// region, not the unfused member sequence.
	let expected = {
		let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		backward(&reference)?
	};
	tile_route_bit_exact(
		&device,
		"OA_TILE_EMBEDDING",
		|| backward(&candidate),
		&expected,
		"embedding_backward_sum 4x4x3",
	)?;

	// embedding_backward: the unfused scatter adjoint — both the u32 and
	// packed-u8 index-table variants fold the position loop in dispatch
	// order.
	for (index_count, num_embeddings, embedding_dim) in [(64usize, 40usize, 24usize), (37, 25, 9)] {
		for packed in [false, true] {
			let kernel = if packed {
				KernelId::MlEmbeddingBackwardU8F32
			} else {
				KernelId::MlEmbeddingBackwardF32
			};
			let index_storage = if packed {
				let index_values: Vec<u8> = (0..index_count)
					.map(|index| (index * 7 % num_embeddings) as u8)
					.collect();
				crate::Matrix::new_from_slice([index_count], &index_values)?
			} else {
				let index_values: Vec<u32> = (0..index_count)
					.map(|index| (index * 7 % num_embeddings) as u32)
					.collect();
				crate::Matrix::new_from_slice([index_count], &index_values)?
			};
			let output_gradient = crate::Matrix::new_from_f32(
				[index_count, embedding_dim],
				&values(index_count * embedding_dim),
			)?;
			let table = crate::Matrix::new_from_f32(
				[num_embeddings, embedding_dim],
				&values(num_embeddings * embedding_dim),
			)?;
			let record = || -> Result<Vec<f32>> {
				let weight_gradient = crate::Matrix::allocate(
					table.engine_handle(),
					vec![num_embeddings, embedding_dim],
					num_embeddings * embedding_dim,
					crate::DType::F32,
				)?;
				let buffers = [
					crate::runtime::BufferBinding::read(index_storage.storage()),
					crate::runtime::BufferBinding::read(output_gradient.storage()),
					crate::runtime::BufferBinding::write(weight_gradient.storage()),
				];
				let dispatch = crate::runtime::ComputeDispatch {
					kernel,
					buffers: &buffers,
					push_constants: &[
						PushConstant::U32(index_count as u32),
						PushConstant::U32(num_embeddings as u32),
						PushConstant::U32(embedding_dim as u32),
					],
					workgroups: kernel.linear_workgroups((num_embeddings * embedding_dim) as u32),
					indirect: None,
				};
				table.engine_handle().record_semantic(
					dispatch,
					crate::runtime::SemanticDispatch {
						contract: crate::core::operation::ml::EMBEDDING_BACKWARD,
						inputs: &[&index_storage, &output_gradient, &table],
						outputs: &[&weight_gradient],
						attributes: &[],
					},
				)?;
				weight_gradient.read_f32()
			};
			let expected = record()?;
			tile_route_bit_exact(
				&device,
				"OA_TILE_EMBEDDING",
				record,
				&expected,
				&format!(
					"embedding_backward packed={packed} {index_count}x{num_embeddings}x{embedding_dim}"
				),
			)?;
		}
	}

	// sum_backward: the broadcast adjoint copies each `(outer, inner)`
	// gradient slot across the reduced axis.
	for (outer_size, dim_size, inner_size) in [(3usize, 5usize, 7usize), (2, 1, 130)] {
		let input_count = outer_size * dim_size * inner_size;
		let input =
			crate::Matrix::new_from_f32([outer_size, dim_size, inner_size], &values(input_count))?;
		let output_gradient =
			crate::Matrix::new_from_f32([outer_size, inner_size], &values(outer_size * inner_size))?;
		let record = || -> Result<Vec<f32>> {
			let input_gradient = crate::Matrix::allocate(
				input.engine_handle(),
				vec![outer_size, dim_size, inner_size],
				input_count,
				crate::DType::F32,
			)?;
			let buffers = [
				crate::runtime::BufferBinding::read(output_gradient.storage()),
				crate::runtime::BufferBinding::write(input_gradient.storage()),
			];
			let dispatch = crate::runtime::ComputeDispatch {
				kernel: KernelId::MatrixSumBackwardF32,
				buffers: &buffers,
				push_constants: &[
					PushConstant::U32(outer_size as u32),
					PushConstant::U32(dim_size as u32),
					PushConstant::U32(inner_size as u32),
				],
				workgroups: KernelId::MatrixSumBackwardF32.linear_workgroups(input_count as u32),
				indirect: None,
			};
			input.engine_handle().record_semantic(
				dispatch,
				crate::runtime::SemanticDispatch {
					contract: crate::core::operation::matrix::SUM_BACKWARD,
					inputs: &[&input, &output_gradient],
					outputs: &[&input_gradient],
					attributes: &[crate::OpAttribute::SignedInteger {
						name: "dim".into(),
						value: 1,
					}],
				},
			)?;
			input_gradient.read_f32()
		};
		let expected = record()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_SUM",
			record,
			&expected,
			&format!("sum_backward {outer_size}x{dim_size}x{inner_size}"),
		)?;
	}
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn tile_max_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x2f81_d5e6u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 997.0
			})
			.collect()
	};

	// Full-tensor max/argmax over exact, tail and single extents — the
	// argmax inputs duplicate the maximum so the donor's lowest-index
	// tie-break is exercised.
	for count in [4096usize, 257, 1, 1027] {
		let mut host = values(count);
		if count > 8 {
			host[3] = 42.0;
			host[count - 2] = 42.0;
		}
		let input = crate::Matrix::new_from_f32([count], &host)?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_MAX",
			|| crate::matrix::max(&input, -1)?.read_f32(),
			&crate::matrix::max(&input, -1)?.read_f32()?,
			&format!("max {count}"),
		)?;
		// The subgroup candidate replaces the shared tree with two
		// WaveActiveMax reduces; max is order-insensitive so the
		// bit-exact contract still holds.
		tile_route_bit_exact(
			&device,
			"OA_TILE_MAX_WAVE",
			|| crate::matrix::max(&input, -1)?.read_f32(),
			&crate::matrix::max(&input, -1)?.read_f32()?,
			&format!("max_wave {count}"),
		)?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_MAX",
			|| Ok(vec![crate::matrix::argmax(&input, -1)? as f32]),
			&[crate::matrix::argmax(&input, -1)? as f32],
			&format!("argmax {count}"),
		)?;
		// The wave argmax pairs WaveActiveMax with WaveActiveMin over
		// index contributions — the donor's lowest-index tie-break is
		// preserved, so the route is bit-exact.
		tile_route_bit_exact(
			&device,
			"OA_TILE_MAX_WAVE",
			|| Ok(vec![crate::matrix::argmax(&input, -1)? as f32]),
			&[crate::matrix::argmax(&input, -1)? as f32],
			&format!("argmax_wave {count}"),
		)?;

		// max_backward: `(value == maximum) ? output_gradient : 0` over
		// the flat element grid.
		let maximum = crate::matrix::max(&input, -1)?;
		let output_gradient = crate::Matrix::new_from_f32([1], &values(1))?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_MAX",
			|| crate::matrix::max_backward(&input, &maximum, &output_gradient)?.read_f32(),
			&crate::matrix::max_backward(&input, &maximum, &output_gradient)?.read_f32()?,
			&format!("max_backward {count}"),
		)?;
	}
	// CC-12: with both gated candidates enabled the search probes the
	// least-sampled candidate per record, so repeated dispatches
	// accumulate specialized samples under each candidate signature.
	{
		use crate::runtime::dna::ir::tile::BlockProgram;
		let input = crate::Matrix::new_from_f32([4096], &values(4096))?;
		let _route = ScopedVar::set("OA_TILE_MAX", std::ffi::OsStr::new("1"));
		let _wave = ScopedVar::set("OA_TILE_MAX_WAVE", std::ffi::OsStr::new("1"));
		let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
		let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
		for _ in 0..8 {
			crate::matrix::max(&input, -1)?.read_f32()?;
		}
		let (_, tree_samples, _) =
			device.tile_route_stats(device.tile_signature(&BlockProgram::max(256).unwrap()));
		let (_, wave_samples, _) =
			device.tile_route_stats(device.tile_signature(&BlockProgram::max_wave(256).unwrap()));
		assert!(
			tree_samples > 0 && wave_samples > 0,
			"multi-candidate search must probe both gated candidates: \
			 tree={tree_samples} wave={wave_samples}"
		);
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flag"]
fn tile_gemm_bias_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	// The epilogue launch lowering binds problem slots
	// [A, B, C, workspace, epilogue] into kernel order [left, right,
	// bias, output] with the donor's `[m, n, k]` push ABI — the tile
	// template repeats the donor's 64x64x16 schedule and inline
	// activation literal, so the incumbent is a bit-exact oracle.
	for (epilogue, kernel, label) in [
		(
			crate::runtime::GemmEpilogue::Bias,
			KernelId::MatrixMatMulNtBiasF32,
			"bias",
		),
		(
			crate::runtime::GemmEpilogue::BiasRelu,
			KernelId::MatrixMatMulNtBiasReluF32,
			"bias_relu",
		),
		(
			crate::runtime::GemmEpilogue::BiasGelu,
			KernelId::MatrixMatMulNtBiasGeluF32,
			"bias_gelu",
		),
		(
			crate::runtime::GemmEpilogue::BiasSilu,
			KernelId::MatrixMatMulNtBiasSiluF32,
			"bias_silu",
		),
	] {
		for (m, n, k) in [(64usize, 64usize, 64usize), (33, 70, 47), (128, 96, 32)] {
			let left = crate::Matrix::new_from_f32([m, k], &gemm_values(m * k, 3))?;
			let right = crate::Matrix::new_from_f32([n, k], &gemm_values(n * k, 11))?;
			let bias = crate::Matrix::new_from_f32([n], &gemm_values(n, 7))?;
			let mut problem = crate::runtime::MatmulProblem::contiguous(m as u32, n as u32, k as u32);
			problem.epilogue = epilogue;
			let plan = crate::runtime::plan_mat_mul_nt_problem(
				&engine.handle(),
				&problem,
				&crate::runtime::MatmulPreference::default(),
			)?;
			assert_eq!(
				plan.kernel_id, kernel,
				"{label}: the planner must select the tiled epilogue donor"
			);
			let launch = crate::runtime::lower_mat_mul_nt_plan(&plan, &problem)?;
			let run = |engine: &Engine| -> Result<Vec<f32>> {
				let output =
					crate::Matrix::allocate(&engine.handle(), vec![m, n], m * n, crate::DType::F32)?;
				let buffers = launch.bindings(&[
					left.storage(),
					right.storage(),
					output.storage(),
					output.storage(),
					bias.storage(),
				])?;
				engine.handle().record_semantic(
					crate::runtime::ComputeDispatch {
						kernel: launch.kernel_id,
						buffers: &buffers,
						push_constants: &launch.push_constants[..launch.push_count],
						workgroups: launch.grid,
						indirect: None,
					},
					crate::runtime::SemanticDispatch {
						contract: crate::core::operation::matrix::MAT_MUL_NT,
						inputs: &[&left, &right],
						outputs: &[&output],
						attributes: &[],
					},
				)?;
				engine.checkpoint()?.wait()?;
				output.read_f32()
			};
			let expected = run(&engine)?;
			tile_route_bit_exact(
				&device,
				"OA_TILE_GEMM_BIAS",
				|| run(&engine),
				&expected,
				&format!("{label} {m}x{n}x{k}"),
			)?;
		}
	}
	Ok(())
}

/// Element-address contract for one strided operand — mirrors `MatmulLayout`.
#[derive(Clone, Copy)]
struct StridedLayout {
	offset: usize,
	row_stride: usize,
	col_stride: usize,
	batch_stride: usize,
}

impl StridedLayout {
	fn required_len(&self, rows: usize, cols: usize, batch: usize) -> usize {
		self.offset
			+ (batch - 1) * self.batch_stride
			+ (rows - 1) * self.row_stride
			+ (cols - 1) * self.col_stride
			+ 1
	}
}

#[test]
#[ignore = "selected Vulkan device"]
fn mat_mul_nt_strided_routes_match_host_oracle() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;

	struct Case {
		m: usize,
		n: usize,
		k: usize,
		batch: usize,
		a: StridedLayout,
		b: StridedLayout,
		c: StridedLayout,
		expected: KernelId,
	}

	// Padded rows, element offsets, a column-strided B, and batch counts that
	// exceed one exercise every layout word the donors read; tail extents
	// (33x70x47 crosses the 64x64 tile in both axes) cover partial tiles.
	let cases = [
		Case {
			m: 8,
			n: 6,
			k: 5,
			batch: 3,
			a: StridedLayout {
				offset: 3,
				row_stride: 7,
				col_stride: 1,
				batch_stride: 60,
			},
			b: StridedLayout {
				offset: 2,
				row_stride: 6,
				col_stride: 1,
				batch_stride: 40,
			},
			c: StridedLayout {
				offset: 4,
				row_stride: 9,
				col_stride: 1,
				batch_stride: 78,
			},
			expected: KernelId::MatrixMatMulNtStridedF32,
		},
		Case {
			m: 33,
			n: 70,
			k: 47,
			batch: 2,
			a: StridedLayout {
				offset: 5,
				row_stride: 50,
				col_stride: 1,
				batch_stride: 1700,
			},
			b: StridedLayout {
				offset: 11,
				row_stride: 48,
				col_stride: 1,
				batch_stride: 3400,
			},
			c: StridedLayout {
				offset: 7,
				row_stride: 72,
				col_stride: 1,
				batch_stride: 2400,
			},
			expected: KernelId::MatrixMatMulNtStridedTiledF32,
		},
		Case {
			m: 17,
			n: 12,
			k: 9,
			batch: 1,
			a: StridedLayout {
				offset: 0,
				row_stride: 18,
				col_stride: 2,
				batch_stride: 0,
			},
			b: StridedLayout {
				offset: 1,
				row_stride: 1,
				col_stride: 13,
				batch_stride: 0,
			},
			c: StridedLayout {
				offset: 2,
				row_stride: 14,
				col_stride: 1,
				batch_stride: 0,
			},
			expected: KernelId::MatrixMatMulNtStridedF32,
		},
	];

	for case in &cases {
		let a_len = case.a.required_len(case.m, case.k, case.batch);
		let b_len = case.b.required_len(case.n, case.k, case.batch);
		let c_len = case.c.required_len(case.m, case.n, case.batch);
		// Sentinel padding makes an out-of-layout read numerically visible.
		let mut a_data = vec![1.0e30_f32; a_len];
		let mut b_data = vec![1.0e30_f32; b_len];
		let mut expected = vec![-1.0e30_f32; c_len];
		let a_values = gemm_values(case.m * case.k * case.batch, 3);
		let b_values = gemm_values(case.n * case.k * case.batch, 11);
		for batch in 0..case.batch {
			for row in 0..case.m {
				for inner in 0..case.k {
					a_data[case.a.offset
						+ batch * case.a.batch_stride
						+ row * case.a.row_stride
						+ inner * case.a.col_stride] = a_values[(batch * case.m + row) * case.k + inner];
				}
			}
			for row in 0..case.n {
				for inner in 0..case.k {
					b_data[case.b.offset
						+ batch * case.b.batch_stride
						+ row * case.b.row_stride
						+ inner * case.b.col_stride] = b_values[(batch * case.n + row) * case.k + inner];
				}
			}
			// Sequential f32 accumulation matches the donor's k-order sum.
			for row in 0..case.m {
				for column in 0..case.n {
					let mut accumulator = 0.0_f32;
					for inner in 0..case.k {
						accumulator += a_values[(batch * case.m + row) * case.k + inner]
							* b_values[(batch * case.n + column) * case.k + inner];
					}
					expected[case.c.offset
						+ batch * case.c.batch_stride
						+ row * case.c.row_stride
						+ column * case.c.col_stride] = accumulator;
				}
			}
		}

		let a = crate::Matrix::new_from_f32([a_len], &a_data)?;
		let b = crate::Matrix::new_from_f32([b_len], &b_data)?;
		let output = crate::Matrix::allocate(&engine.handle(), vec![c_len], c_len, crate::DType::F32)?;

		let mut problem =
			crate::runtime::MatmulProblem::contiguous(case.m as u32, case.n as u32, case.k as u32);
		problem.batch_count = case.batch as u32;
		for (layout, field) in [
			(&case.a, &mut problem.a),
			(&case.b, &mut problem.b),
			(&case.c, &mut problem.c),
		] {
			field.offset = layout.offset as u32;
			field.row_stride = layout.row_stride as u32;
			field.col_stride = layout.col_stride as u32;
			field.batch_stride = layout.batch_stride as u32;
		}
		problem.a_contiguous = false;
		problem.b_contiguous = false;
		let plan = crate::runtime::plan_mat_mul_nt_problem(
			&engine.handle(),
			&problem,
			&crate::runtime::MatmulPreference::default(),
		)?;
		assert_eq!(
			plan.kernel_id, case.expected,
			"the planner must route non-canonical layouts to the strided family"
		);
		let launch = crate::runtime::lower_mat_mul_nt_plan(&plan, &problem)?;
		assert_eq!(launch.grid[2], case.batch as u32);
		let buffers = launch.bindings(&[
			a.storage(),
			b.storage(),
			output.storage(),
			output.storage(),
			output.storage(),
		])?;
		engine.handle().record_semantic(
			crate::runtime::ComputeDispatch {
				kernel: launch.kernel_id,
				buffers: &buffers,
				push_constants: &launch.push_constants[..launch.push_count],
				workgroups: launch.grid,
				indirect: None,
			},
			crate::runtime::SemanticDispatch {
				contract: crate::core::operation::matrix::MAT_MUL_NT,
				inputs: &[&a, &b],
				outputs: &[&output],
				attributes: &[],
			},
		)?;
		engine.checkpoint()?.wait()?;
		let actual = output.read_f32()?;
		assert_eq!(actual.len(), expected.len());
		for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
			if *expected == -1.0e30_f32 {
				continue;
			}
			let tolerance = 2.0e-4 + 4.0e-6 * case.k as f32 + 2.0e-4 * expected.abs();
			assert!(
				expected.is_finite() && (actual - expected).abs() <= tolerance,
				"{:?} element {index}: expected {expected} got {actual}",
				case.expected
			);
		}
	}
	Ok(())
}

#[test]
#[ignore = "selected Vulkan device"]
fn mat_mul_nt_split_k_route_matches_host_oracle() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;

	for (m, n, k) in [(64usize, 48usize, 200usize), (33, 19, 97), (16, 16, 4)] {
		let a = crate::Matrix::new_from_f32([m, k], &gemm_values(m * k, 3))?;
		let b = crate::Matrix::new_from_f32([n, k], &gemm_values(n * k, 11))?;
		let output = crate::Matrix::allocate(&engine.handle(), vec![m, n], m * n, crate::DType::F32)?;
		let workspace_elements = m * n * 4;
		let workspace = crate::Matrix::allocate(
			&engine.handle(),
			vec![workspace_elements],
			workspace_elements,
			crate::DType::F32,
		)?;

		let problem = crate::runtime::MatmulProblem::contiguous(m as u32, n as u32, k as u32);
		let preference = crate::runtime::MatmulPreference {
			required_variant: matmul_variant_id_from_name("GemmSplitK"),
			max_workspace_bytes: (workspace_elements * 4) as u64,
			..crate::runtime::MatmulPreference::default()
		};
		let plan = crate::runtime::plan_mat_mul_nt_problem(&engine.handle(), &problem, &preference)?;
		assert_eq!(
			plan.kernel_id,
			KernelId::MatrixMatMulNtSplitKF32,
			"the required-variant pin must select the split-K donor"
		);
		let launch = crate::runtime::lower_mat_mul_nt_plan(&plan, &problem)?;
		let reduction =
			lower_mat_mul_nt_reduction(&plan, &problem)?.expect("split-K plans carry a reduction launch");
		let problem_buffers = [
			a.storage(),
			b.storage(),
			output.storage(),
			workspace.storage(),
			output.storage(),
		];
		let buffers = launch.bindings(&problem_buffers)?;
		let reduce_buffers = reduction.bindings(&problem_buffers)?;
		engine.handle().record_split_semantic(
			&[
				crate::runtime::ComputeDispatch {
					kernel: launch.kernel_id,
					buffers: &buffers,
					push_constants: &launch.push_constants[..launch.push_count],
					workgroups: launch.grid,
					indirect: None,
				},
				crate::runtime::ComputeDispatch {
					kernel: reduction.kernel_id,
					buffers: &reduce_buffers,
					push_constants: &reduction.push_constants[..reduction.push_count],
					workgroups: reduction.grid,
					indirect: None,
				},
			],
			crate::runtime::SemanticDispatch {
				contract: crate::core::operation::matrix::MAT_MUL_NT,
				inputs: &[&a, &b],
				outputs: &[&output],
				attributes: &[],
			},
		)?;
		engine.checkpoint()?.wait()?;
		drop(workspace);
		let actual = output.read_f32()?;
		let expected = gemm_cpu_oracle(&gemm_values(m * k, 3), &gemm_values(n * k, 11), m, n, k);
		for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
			let tolerance = 2.0e-4 + 4.0e-6 * k as f32 + 2.0e-4 * expected.abs();
			assert!(
				expected.is_finite() && (actual - expected).abs() <= tolerance,
				"{m}x{n}x{k} element {index}: expected {expected} got {actual}"
			);
		}
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_conv_1d_im2_col_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x6a09e667f3bcc909u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2011) as f32 - 1000.0) / 383.0
			})
			.collect()
	};
	// (batch, in_channels, in_length, out_channels, kernel, stride,
	// padding, dilation): cover plain, strided, padded, dilated and a
	// columns count that tails past the 256-lane flat grid.
	for (batch, channels, length, out_channels, kernel, stride, padding, dilation) in [
		(
			2usize, 4usize, 96usize, 6usize, 3usize, 1usize, 0usize, 1usize,
		),
		(1, 3, 130, 5, 5, 2, 3, 1),
		(3, 2, 40, 4, 4, 1, 2, 3),
		(1, 1, 300, 2, 3, 1, 1, 1),
	] {
		let input = crate::Matrix::new_from_f32(
			[batch, channels, length],
			&values(batch * channels * length),
		)?;
		let weight = crate::Matrix::new_from_f32(
			[out_channels, channels, kernel],
			&values(out_channels * channels * kernel),
		)?;
		let bias = crate::Matrix::new_from_f32([out_channels], &values(out_channels))?;
		let expected =
			crate::ml::matrix::conv_1d(&input, &weight, &bias, stride, padding, dilation)?.read_f32()?;
		let before = device.fusion_census().snapshot();
		tile_route_bit_exact(
			&device,
			"OA_TILE_CONV",
			|| {
				crate::ml::matrix::conv_1d(&input, &weight, &bias, stride, padding, dilation)
					.and_then(|m| m.read_f32())
			},
			&expected,
			&format!(
				"conv_1d im2col {batch}x{channels}x{length} k{kernel} s{stride} p{padding} d{dilation}"
			),
		)?;
		let census = device.fusion_census().snapshot();
		let specialized = census.jit_admitted_count() + census.jit_reused_count()
			- before.jit_admitted_count()
			- before.jit_reused_count();
		assert_eq!(
			specialized, 4,
			"conv_1d must specialize im2col and both tail members before the \
			 tail merge substitutes `conv_1d_bias_transpose`: {census}"
		);
	}
	// (batch, in_channels, height, width, out_channels, kernel, stride,
	// padding, groups): cover plain, strided, padded, grouped and a
	// non-256-aligned output count tail.
	for (batch, channels, height, width, out_channels, kernel, stride, padding, groups) in [
		(
			2usize, 4usize, 9usize, 11usize, 6usize, 3usize, 1usize, 0usize, 1usize,
		),
		(1, 3, 17, 13, 5, 3, 2, 2, 1),
		(2, 4, 12, 12, 8, 3, 1, 1, 2),
		(1, 1, 33, 41, 2, 5, 1, 0, 1),
	] {
		let input = crate::Matrix::new_from_f32(
			[batch, channels, height, width],
			&values(batch * channels * height * width),
		)?;
		let weight = crate::Matrix::new_from_f32(
			[out_channels, channels / groups, kernel, kernel],
			&values(out_channels * (channels / groups) * kernel * kernel),
		)?;
		let bias = crate::Matrix::new_from_f32([out_channels], &values(out_channels))?;
		let expected =
			crate::ml::matrix::conv_2d(&input, &weight, &bias, stride, padding, groups)?.read_f32()?;
		let before = device.fusion_census().snapshot();
		tile_route_bit_exact(
			&device,
			"OA_TILE_CONV",
			|| {
				crate::ml::matrix::conv_2d(&input, &weight, &bias, stride, padding, groups)
					.and_then(|m| m.read_f32())
			},
			&expected,
			&format!(
				"conv_2d {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding} g{groups}"
			),
		)?;
		let census = device.fusion_census().snapshot();
		let specialized = census.jit_admitted_count() + census.jit_reused_count()
			- before.jit_admitted_count()
			- before.jit_reused_count();
		assert_eq!(
			specialized, 1,
			"conv_2d must specialize its dispatch: {census}"
		);
	}
	// The submit-time Conv2d→ReLU fusion donor: `relu(conv_2d(…))`
	// merges into one `conv_2d_relu` dispatch, which the tile route
	// substitutes end-to-end.
	for (batch, channels, height, width, out_channels, kernel, stride, padding, groups) in [
		(
			2usize, 4usize, 9usize, 11usize, 6usize, 3usize, 1usize, 0usize, 1usize,
		),
		(1, 3, 17, 13, 5, 3, 2, 2, 1),
	] {
		let input = crate::Matrix::new_from_f32(
			[batch, channels, height, width],
			&values(batch * channels * height * width),
		)?;
		let weight = crate::Matrix::new_from_f32(
			[out_channels, channels / groups, kernel, kernel],
			&values(out_channels * (channels / groups) * kernel * kernel),
		)?;
		let bias = crate::Matrix::new_from_f32([out_channels], &values(out_channels))?;
		let expected = crate::ml::matrix::relu(&crate::ml::matrix::conv_2d(
			&input, &weight, &bias, stride, padding, groups,
		)?)?
		.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_CONV",
			|| {
				crate::ml::matrix::relu(&crate::ml::matrix::conv_2d(
					&input, &weight, &bias, stride, padding, groups,
				)?)
				.and_then(|m| m.read_f32())
			},
			&expected,
			&format!(
				"conv_2d_relu {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding} g{groups}"
			),
		)?;
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_conv_1d_parameter_backward_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x9e3779b97f4a7c15u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 421.0
			})
			.collect()
	};
	// (batch, in_channels, in_length, out_channels, kernel, stride,
	// padding, dilation): cover plain, strided, padded, dilated and a
	// parameter count that tails past the 256-lane flat grid.
	for (batch, channels, length, out_channels, kernel, stride, padding, dilation) in [
		(
			2usize, 4usize, 96usize, 6usize, 3usize, 1usize, 0usize, 1usize,
		),
		(1, 3, 130, 5, 5, 2, 3, 1),
		(3, 2, 40, 4, 4, 1, 2, 3),
		(1, 1, 300, 2, 3, 1, 1, 1),
	] {
		let weight_values = values(out_channels * channels * kernel);
		let bias_values = values(out_channels);
		// The input adjoint only dispatches for a watched input, so an
		// Embedding — `length` features per token — feeds the conv: its
		// weight is a registered parameter and a `[batch, channels]`
		// index grid gathers exactly the conv's `[batch, channels,
		// length]` layout.
		let embedding_values = values(64 * length);
		let indices = crate::Matrix::new_from_slice(
			[batch, channels],
			&(0..batch * channels)
				.map(|index| (index % 64) as u32)
				.collect::<Vec<_>>(),
		)?;
		let output_length = (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
		let target = crate::Matrix::new_from_f32(
			[batch, out_channels, output_length],
			&values(batch * out_channels * output_length),
		)?;
		let backward = || -> Result<Vec<f32>> {
			let embedding = crate::ml::nn::Embedding::new_from_matrix(crate::Matrix::new_from_f32(
				[64, length],
				&embedding_values,
			)?)?;
			let module = crate::ml::nn::Conv1d::new_from_matrices(
				crate::Matrix::new_from_f32([out_channels, channels, kernel], &weight_values)?,
				crate::Matrix::new_from_f32([out_channels], &bias_values)?,
				stride,
				padding,
				dilation,
			)?;
			let tape = crate::ml::GradientTape::new();
			let loss = crate::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
			tape.backward(&loss)?;
			let mut flat = embedding.weight().gradient().unwrap().read_f32()?;
			flat.reserve(out_channels * channels * kernel + out_channels);
			for parameter in module.parameters() {
				flat.extend(parameter.gradient().unwrap().read_f32()?);
			}
			Ok(flat)
		};
		let expected = backward()?;
		let before = device.fusion_census().snapshot();
		tile_route_bit_exact(
			&device,
			"OA_TILE_CONV",
			backward,
			&expected,
			&format!(
				"conv_1d parameter backward {batch}x{channels}x{length} k{kernel} s{stride} p{padding} d{dilation}"
			),
		)?;
		let census = device.fusion_census().snapshot();
		let specialized = census.jit_admitted_count() + census.jit_reused_count()
			- before.jit_admitted_count()
			- before.jit_reused_count();
		assert_eq!(
			specialized, 6,
			"conv_1d run must specialize im2col, both tail members, the fused \
			 tail and both backward adjoints: {census}"
		);
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_conv_transpose_1d_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x9e3779b97f4a7c15u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 421.0
			})
			.collect()
	};
	// (batch, in_channels, in_length, out_channels, kernel, stride,
	// padding): cover plain, strided, padded, and a transposed output
	// that tails past the 256-lane flat grid. The module fixes
	// dilation to one.
	for (batch, channels, length, out_channels, kernel, stride, padding) in [
		(2usize, 4usize, 48usize, 6usize, 3usize, 1usize, 0usize),
		(1, 3, 40, 5, 4, 2, 1),
		(3, 2, 17, 4, 3, 2, 2),
		(1, 1, 120, 2, 5, 1, 0),
	] {
		let weight_values = values(channels * out_channels * kernel);
		// An Embedding feeding the transpose-conv watches the input so
		// the backward dispatches the unfold/GEMM/transpose input
		// adjoint alongside the shared parameter adjoint.
		let embedding_values = values(64 * length);
		let indices = crate::Matrix::new_from_slice(
			[batch, channels],
			&(0..batch * channels)
				.map(|index| (index % 64) as u32)
				.collect::<Vec<_>>(),
		)?;
		let output_length = (length - 1) * stride + kernel - 2 * padding;
		let target = crate::Matrix::new_from_f32(
			[batch, out_channels, output_length],
			&values(batch * out_channels * output_length),
		)?;
		let backward = || -> Result<Vec<f32>> {
			let embedding = crate::ml::nn::Embedding::new_from_matrix(crate::Matrix::new_from_f32(
				[64, length],
				&embedding_values,
			)?)?;
			let module = crate::ml::nn::ConvTranspose1d::new_from_matrix(
				crate::Matrix::new_from_f32([channels, out_channels, kernel], &weight_values)?,
				stride,
				padding,
			)?;
			let tape = crate::ml::GradientTape::new();
			let loss = crate::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
			tape.backward(&loss)?;
			let mut flat = embedding.weight().gradient().unwrap().read_f32()?;
			for parameter in module.parameters() {
				flat.extend(parameter.gradient().unwrap().read_f32()?);
			}
			Ok(flat)
		};
		let expected = backward()?;
		let before = device.fusion_census().snapshot();
		tile_route_bit_exact(
			&device,
			"OA_TILE_CONV",
			backward,
			&expected,
			&format!("conv_transpose_1d {batch}x{channels}x{length} k{kernel} s{stride} p{padding}"),
		)?;
		let census = device.fusion_census().snapshot();
		let specialized = census.jit_admitted_count() + census.jit_reused_count()
			- before.jit_admitted_count()
			- before.jit_reused_count();
		assert_eq!(
			specialized, 4,
			"conv_transpose_1d run must specialize forward, unfold, transpose, \
			 and parameter adjoints: {census}"
		);
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_conv_2d_backward_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x9e3779b97f4a7c15u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 421.0
			})
			.collect()
	};
	// (batch, channels, height, width, out_channels, kernel, stride,
	// padding, groups): cover plain, strided, padded, grouped and
	// 256-tail geometries. Conv2d uses square kernels.
	for (batch, channels, height, width, out_channels, kernel, stride, padding, groups) in [
		(
			2usize, 4usize, 16usize, 16usize, 6usize, 3usize, 1usize, 0usize, 1usize,
		),
		(1, 6, 18, 14, 9, 3, 2, 1, 3),
		(1, 3, 21, 19, 5, 5, 1, 2, 1),
		(2, 2, 33, 29, 4, 2, 1, 0, 2),
	] {
		let channels_per_group = channels / groups;
		let weight_values = values(out_channels * channels_per_group * kernel * kernel);
		let bias_values = values(out_channels);
		// An Embedding over width-length rows feeds a `[batch,
		// channels, height]` index grid so its `width` features land
		// exactly in the conv's NCHW input — a watched input is what
		// dispatches the data adjoint.
		let embedding_values = values(64 * width);
		let indices = crate::Matrix::new_from_slice(
			[batch, channels, height],
			&(0..batch * channels * height)
				.map(|index| (index % 64) as u32)
				.collect::<Vec<_>>(),
		)?;
		let output_height = (height + 2 * padding - kernel) / stride + 1;
		let output_width = (width + 2 * padding - kernel) / stride + 1;
		let target = crate::Matrix::new_from_f32(
			[batch, out_channels, output_height, output_width],
			&values(batch * out_channels * output_height * output_width),
		)?;
		let backward = || -> Result<Vec<f32>> {
			let embedding = crate::ml::nn::Embedding::new_from_matrix(crate::Matrix::new_from_f32(
				[64, width],
				&embedding_values,
			)?)?;
			let module = crate::ml::nn::Conv2d::new_from_matrices(
				crate::Matrix::new_from_f32(
					[out_channels, channels_per_group, kernel, kernel],
					&weight_values,
				)?,
				crate::Matrix::new_from_f32([out_channels], &bias_values)?,
				stride,
				padding,
				groups,
			)?;
			let tape = crate::ml::GradientTape::new();
			let loss = crate::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
			tape.backward(&loss)?;
			let mut flat = embedding.weight().gradient().unwrap().read_f32()?;
			for parameter in module.parameters() {
				flat.extend(parameter.gradient().unwrap().read_f32()?);
			}
			Ok(flat)
		};
		let expected = backward()?;
		let before = device.fusion_census().snapshot();
		tile_route_bit_exact(
			&device,
			"OA_TILE_CONV",
			backward,
			&expected,
			&format!(
				"conv_2d backward {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding} g{groups}"
			),
		)?;
		let census = device.fusion_census().snapshot();
		let specialized = census.jit_admitted_count() + census.jit_reused_count()
			- before.jit_admitted_count()
			- before.jit_reused_count();
		assert_eq!(
			specialized, 3,
			"conv_2d run must specialize forward, data, and parameter adjoints: {census}"
		);
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_conv_transpose_2d_route_matches_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x9e3779b97f4a7c15u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 421.0
			})
			.collect()
	};
	// (batch, channels, height, width, out_channels, kernel, stride,
	// padding): cover plain, strided, padded, and 256-tail
	// transposed outputs. ConvTranspose2d uses square kernels.
	for (batch, channels, height, width, out_channels, kernel, stride, padding) in [
		(
			2usize, 4usize, 12usize, 10usize, 6usize, 3usize, 1usize, 0usize,
		),
		(1, 3, 9, 11, 5, 4, 2, 1),
		(3, 2, 7, 8, 4, 3, 2, 2),
		(1, 1, 20, 18, 2, 5, 1, 0),
	] {
		let weight_values = values(channels * out_channels * kernel * kernel);
		let bias_values = values(out_channels);
		// The Embedding input watches the transpose-conv's input so
		// the data adjoint dispatches beside the parameter adjoint.
		let embedding_values = values(64 * width);
		let indices = crate::Matrix::new_from_slice(
			[batch, channels, height],
			&(0..batch * channels * height)
				.map(|index| (index % 64) as u32)
				.collect::<Vec<_>>(),
		)?;
		let output_height = (height - 1) * stride + kernel - 2 * padding;
		let output_width = (width - 1) * stride + kernel - 2 * padding;
		let target = crate::Matrix::new_from_f32(
			[batch, out_channels, output_height, output_width],
			&values(batch * out_channels * output_height * output_width),
		)?;
		let backward = || -> Result<Vec<f32>> {
			let embedding = crate::ml::nn::Embedding::new_from_matrix(crate::Matrix::new_from_f32(
				[64, width],
				&embedding_values,
			)?)?;
			let module = crate::ml::nn::ConvTranspose2d::new_from_matrices(
				crate::Matrix::new_from_f32([channels, out_channels, kernel, kernel], &weight_values)?,
				crate::Matrix::new_from_f32([out_channels], &bias_values)?,
				stride,
				padding,
			)?;
			let tape = crate::ml::GradientTape::new();
			let loss = crate::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
			tape.backward(&loss)?;
			let mut flat = embedding.weight().gradient().unwrap().read_f32()?;
			for parameter in module.parameters() {
				flat.extend(parameter.gradient().unwrap().read_f32()?);
			}
			Ok(flat)
		};
		let expected = backward()?;
		let before = device.fusion_census().snapshot();
		tile_route_bit_exact(
			&device,
			"OA_TILE_CONV",
			backward,
			&expected,
			&format!(
				"conv_transpose_2d {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding}"
			),
		)?;
		let census = device.fusion_census().snapshot();
		let specialized = census.jit_admitted_count() + census.jit_reused_count()
			- before.jit_admitted_count()
			- before.jit_reused_count();
		assert_eq!(
			specialized, 4,
			"conv_transpose_2d run must specialize forward, bias-add, and both \
			 adjoints: {census}"
		);
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_conv_1d_bias_tail_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x1b873593u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2011) as f32 - 1000.0) / 383.0
			})
			.collect()
	};
	// The fused tails only form under capture: in eager mode every
	// semantic intermediate stays observable, so `projected` is never
	// provably free. Each leg captures the chain, replays it, and reads
	// the output storage the plan binds.
	let input = crate::Matrix::new_from_f32([2, 3, 40], &values(2 * 3 * 40))?;
	let weight = crate::Matrix::new_from_f32([5, 3, 3], &values(5 * 3 * 3))?;
	let bias = crate::Matrix::new_from_f32([5], &values(5))?;
	let conv = || crate::ml::matrix::conv_1d(&input, &weight, &bias, 1, 1, 1);
	let fused_tail =
		|label: &str, kernel: &str, run: &dyn Fn() -> Result<crate::Matrix>| -> Result<()> {
			let expected = run()?.read_f32()?;
			let before = device.fusion_census().snapshot();
			let (plan, output) = {
				let _route = ScopedVar::set("OA_TILE_CONV", std::ffi::OsStr::new("1"));
				let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
				let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
				engine.capture(run)?
			};
			let census = device.fusion_census().snapshot();
			let specialized = census.jit_admitted_count() + census.jit_reused_count()
				- before.jit_admitted_count()
				- before.jit_reused_count();
			let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("epilogue"))
				.expect("fused epilogue executable report must be valid JSON");
			assert!(
				report["nodes"]
					.as_array()
					.expect("nodes")
					.iter()
					.any(|node| { node["kernel"].as_str().unwrap().ends_with(kernel) }),
				"{label}: the {kernel} fused node must exist in the plan"
			);
			assert_eq!(
				specialized, 4,
				"{label}: conv_1d must specialize im2col, both tail members and the \
			 fused tail: {census}"
			);
			engine.submit(&plan)?.wait()?;
			let actual = output.read_f32()?;
			for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
				assert_eq!(
					actual.to_bits(),
					expected.to_bits(),
					"{label} element {index}: expected {expected}, found {actual}"
				);
			}
			Ok(())
		};
	fused_tail("bias_transpose", "conv_1d_bias_transpose.f32", &conv)?;
	fused_tail(
		"bias_transpose_relu",
		"conv_1d_bias_transpose_relu.f32",
		&|| crate::ml::matrix::relu(&conv()?),
	)?;
	let residual = crate::Matrix::new_from_f32([2, 5, 40], &values(2 * 5 * 40))?;
	fused_tail(
		"bias_transpose_residual_relu",
		"conv_1d_bias_transpose_residual_relu.f32",
		&|| crate::matrix::add(&crate::ml::matrix::relu(&conv()?)?, &residual),
	)?;
	let norm_weight = crate::Matrix::new_from_f32([5], &values(5))?;
	let norm_bias = crate::Matrix::new_from_f32([5], &values(5))?;
	fused_tail(
		"bias_transpose_norm",
		"conv_1d_bias_transpose_norm.f32",
		&|| crate::ml::matrix::channel_norm(&conv()?, &norm_weight, &norm_bias, 1e-5),
	)?;
	fused_tail(
		"bias_transpose_norm_relu",
		"conv_1d_bias_transpose_norm_relu.f32",
		&|| crate::ml::matrix::channel_norm_relu(&conv()?, &norm_weight, &norm_bias, 1e-5),
	)?;
	fused_tail("bias_copy", "conv_1d_bias_copy.f32", &|| {
		crate::matrix::transpose(&conv()?, 1, 2)
	})?;
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_pool_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x9e3779b97f4a7c15u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 421.0
			})
			.collect()
	};
	// (batch, channels, height, width, kernel, stride, padding): plain,
	// strided, padded and tail-extent geometries.
	for (batch, channels, height, width, kernel, stride, padding) in [
		(1usize, 2usize, 4usize, 5usize, 3usize, 2usize, 1usize),
		(2, 4, 16, 16, 2, 2, 0),
		(1, 3, 9, 11, 2, 1, 1),
		(2, 2, 17, 13, 3, 3, 2),
	] {
		let input_values = values(batch * channels * height * width);
		let input = crate::Matrix::new_from_f32([batch, channels, height, width], &input_values)?;
		let expected = crate::ml::matrix::avg_pool_2d(&input, kernel, stride, padding)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_POOL",
			|| crate::ml::matrix::avg_pool_2d(&input, kernel, stride, padding)?.read_f32(),
			&expected,
			&format!("avg_pool_2d {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding}"),
		)?;
	}
	// Max pool: the output plus the saved argmax indices (u32 read of the
	// donor's absolute-offset index buffer).
	for (batch, channels, height, width, kernel, stride, padding) in [
		(1usize, 2usize, 4usize, 5usize, 3usize, 2usize, 1usize),
		(2, 4, 16, 16, 2, 2, 0),
		(1, 3, 9, 11, 2, 1, 1),
		(2, 2, 17, 13, 3, 3, 2),
	] {
		let input_values = values(batch * channels * height * width);
		let input = crate::Matrix::new_from_f32([batch, channels, height, width], &input_values)?;
		let incumbent = crate::ml::matrix::max_pool_2d(&input, kernel, stride, padding)?;
		let expected_output = incumbent.output.read_f32()?;
		let expected_indices = incumbent.indices.read::<u32>()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_POOL",
			|| {
				crate::ml::matrix::max_pool_2d(&input, kernel, stride, padding)?
					.output
					.read_f32()
			},
			&expected_output,
			&format!("max_pool_2d {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding}"),
		)?;
		let before = device.fusion_census().snapshot();
		let actual_indices = {
			let _route = ScopedVar::set("OA_TILE_POOL", std::ffi::OsStr::new("1"));
			let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			let _forced = ScopedVar::set("OA_JIT_ROUTE", std::ffi::OsStr::new("specialized"));
			crate::ml::matrix::max_pool_2d(&input, kernel, stride, padding)?
				.indices
				.read::<u32>()?
		};
		let census = device.fusion_census().snapshot();
		assert!(
			census.jit_admitted_count() + census.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"max_pool_2d indices: the specialization must be admitted: {census}"
		);
		assert_eq!(
			actual_indices, expected_indices,
			"max_pool_2d indices {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding}"
		);
	}
	// Adaptive average pool: computed window bounds, two output sizes
	// including a non-divisible one.
	for (batch, channels, height, width, output_height, output_width) in [
		(2usize, 3usize, 7usize, 9usize, 4usize, 5usize),
		(1, 2, 8, 8, 2, 2),
		(1, 4, 11, 13, 11, 13),
		(2, 2, 9, 6, 3, 5),
	] {
		let input_values = values(batch * channels * height * width);
		let input = crate::Matrix::new_from_f32([batch, channels, height, width], &input_values)?;
		let expected =
			crate::ml::matrix::adaptive_avg_pool_2d(&input, output_height, output_width)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_POOL",
			|| crate::ml::matrix::adaptive_avg_pool_2d(&input, output_height, output_width)?.read_f32(),
			&expected,
			&format!(
				"adaptive_avg_pool_2d {batch}x{channels}x{height}x{width} -> {output_height}x{output_width}"
			),
		)?;
	}
	// Pool input adjoints: a watched embedding input forces the data
	// adjoint through the tape; gradients must match the incumbent bits.
	for (batch, channels, height, width, kernel, stride, padding) in [
		(1usize, 2usize, 4usize, 5usize, 3usize, 2usize, 1usize),
		(2, 4, 16, 16, 2, 2, 0),
		(1, 3, 9, 11, 2, 1, 1),
	] {
		let embedding_values = values(64 * width);
		let indices = crate::Matrix::new_from_slice(
			[batch, channels, height],
			&(0..batch * channels * height)
				.map(|index| (index % 64) as u32)
				.collect::<Vec<_>>(),
		)?;
		let output_height = (height + 2 * padding - kernel) / stride + 1;
		let output_width = (width + 2 * padding - kernel) / stride + 1;
		let target = crate::Matrix::new_from_f32(
			[batch, channels, output_height, output_width],
			&values(batch * channels * output_height * output_width),
		)?;
		let backward = |relu: bool| -> Result<Vec<f32>> {
			let embedding = crate::ml::nn::Embedding::new_from_matrix(crate::Matrix::new_from_f32(
				[64, width],
				&embedding_values,
			)?)?;
			let tape = crate::ml::GradientTape::new();
			let pooled = if relu {
				let activation = crate::ml::nn::Relu::new();
				let module = crate::ml::nn::MaxPool2d::new_with_options(kernel, stride, padding)?;
				module.forward(&activation.forward(&embedding.forward(&indices)?)?)?
			} else {
				let module = crate::ml::nn::AvgPool2d::new_with_options(kernel, stride, padding)?;
				module.forward(&embedding.forward(&indices)?)?
			};
			let loss = crate::ml::loss::mse(&pooled, &target)?;
			tape.backward(&loss)?;
			embedding.weight().gradient().unwrap().read_f32()
		};
		let expected = backward(false)?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_POOL",
			|| backward(false),
			&expected,
			&format!(
				"avg_pool_2d backward {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding}"
			),
		)?;
		let expected = backward(true)?;
		let before = device.fusion_census().snapshot();
		tile_route_bit_exact(
			&device,
			"OA_TILE_POOL",
			|| backward(true),
			&expected,
			&format!(
				"max_pool_2d relu backward {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding}"
			),
		)?;
		let census = device.fusion_census().snapshot();
		let specialized = census.jit_admitted_count() + census.jit_reused_count()
			- before.jit_admitted_count()
			- before.jit_reused_count();
		// max_pool_2d forward + the member max_pool_2d_backward admission
		// at build + the fused max_pool_2d_relu_backward admission at the
		// merge — same member-then-fused census as the conv tails.
		assert_eq!(
			specialized, 3,
			"relu+max_pool run must specialize the forward pool, the member adjoint and the fused adjoint: {census}"
		);
	}
	// Adaptive pool input adjoint.
	for (batch, channels, height, width, output_height, output_width) in [
		(1usize, 2usize, 7usize, 9usize, 4usize, 5usize),
		(2, 2, 8, 8, 2, 2),
	] {
		let embedding_values = values(64 * width);
		let indices = crate::Matrix::new_from_slice(
			[batch, channels, height],
			&(0..batch * channels * height)
				.map(|index| (index % 64) as u32)
				.collect::<Vec<_>>(),
		)?;
		let target = crate::Matrix::new_from_f32(
			[batch, channels, output_height, output_width],
			&values(batch * channels * output_height * output_width),
		)?;
		let backward = || -> Result<Vec<f32>> {
			let embedding = crate::ml::nn::Embedding::new_from_matrix(crate::Matrix::new_from_f32(
				[64, width],
				&embedding_values,
			)?)?;
			let module =
				crate::ml::nn::AdaptiveAvgPool2d::new_with_output_size(output_height, output_width)?;
			let tape = crate::ml::GradientTape::new();
			let loss = crate::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
			tape.backward(&loss)?;
			embedding.weight().gradient().unwrap().read_f32()
		};
		let expected = backward()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_POOL",
			backward,
			&expected,
			&format!(
				"adaptive_avg_pool_2d backward {batch}x{channels}x{height}x{width} -> {output_height}x{output_width}"
			),
		)?;
	}
	// The plain max_pool_2d input adjoint through the tape.
	for (batch, channels, height, width, kernel, stride, padding) in [
		(1usize, 2usize, 4usize, 5usize, 3usize, 2usize, 1usize),
		(2, 4, 16, 16, 2, 2, 0),
	] {
		let embedding_values = values(64 * width);
		let indices = crate::Matrix::new_from_slice(
			[batch, channels, height],
			&(0..batch * channels * height)
				.map(|index| (index % 64) as u32)
				.collect::<Vec<_>>(),
		)?;
		let output_height = (height + 2 * padding - kernel) / stride + 1;
		let output_width = (width + 2 * padding - kernel) / stride + 1;
		let target = crate::Matrix::new_from_f32(
			[batch, channels, output_height, output_width],
			&values(batch * channels * output_height * output_width),
		)?;
		let backward = || -> Result<Vec<f32>> {
			let embedding = crate::ml::nn::Embedding::new_from_matrix(crate::Matrix::new_from_f32(
				[64, width],
				&embedding_values,
			)?)?;
			let module = crate::ml::nn::MaxPool2d::new_with_options(kernel, stride, padding)?;
			let tape = crate::ml::GradientTape::new();
			let loss = crate::ml::loss::mse(&module.forward(&embedding.forward(&indices)?)?, &target)?;
			tape.backward(&loss)?;
			embedding.weight().gradient().unwrap().read_f32()
		};
		let expected = backward()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_POOL",
			backward,
			&expected,
			&format!(
				"max_pool_2d backward {batch}x{channels}x{height}x{width} k{kernel} s{stride} p{padding}"
			),
		)?;
	}
	Ok(())
}

#[test]
#[ignore = "explicit Slang library, selected Vulkan device and opt-in tile flags"]
fn tile_attention_routes_match_incumbent() -> Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device index")
		.parse::<usize>()
		.expect("device index");
	if let Some(library) = std::env::var_os("OA_TEST_SLANG_LIBRARY") {
		// SAFETY: same single-threaded qualification scope as `JitFlag`.
		unsafe { std::env::set_var("OA_SLANG_LIBRARY", library) };
	}
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(index))
		.build()?;
	let device = engine.handle.state.borrow().device.clone();
	let _hold = JitFlag::hold();
	let mut seeded = 0x1bf52f5f6d4e7c91u64;
	let mut values = |count: usize| -> Vec<f32> {
		(0..count)
			.map(|_| {
				seeded ^= seeded << 13;
				seeded ^= seeded >> 7;
				seeded ^= seeded << 17;
				((seeded % 2003) as f32 - 1000.0) / 379.0
			})
			.collect()
	};
	// `split_heads` — packed [B*S, H*D] into [B*H, S, D]; heads > 1 so the
	// permutation is real, with an odd head dimension for the tail guard.
	for (batch, sequence, heads, head_dim) in [(2usize, 5usize, 2usize, 4usize), (1, 7, 4, 3)] {
		let input = crate::Matrix::new_from_f32(
			[batch * sequence, heads * head_dim],
			&values(batch * sequence * heads * head_dim),
		)?;
		let expected = crate::ml::matrix::split_heads(&input, batch, sequence, heads)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_ATTENTION",
			|| crate::ml::matrix::split_heads(&input, batch, sequence, heads)?.read_f32(),
			&expected,
			&format!("split_heads b{batch} s{sequence} h{heads} d{head_dim}"),
		)?;
	}
	// `merge_heads` — the [B*H, S, D] inverse.
	for (batch, sequence, heads, head_dim) in [(2usize, 5usize, 2usize, 4usize), (1, 7, 4, 3)] {
		let input = crate::Matrix::new_from_f32(
			[batch * heads, sequence, head_dim],
			&values(batch * heads * sequence * head_dim),
		)?;
		let expected = crate::ml::matrix::merge_heads(&input, batch, sequence, heads)?.read_f32()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_ATTENTION",
			|| crate::ml::matrix::merge_heads(&input, batch, sequence, heads)?.read_f32(),
			&expected,
			&format!("merge_heads b{batch} s{sequence} h{heads} d{head_dim}"),
		)?;
	}
	// `scaled_dot_product_attention` — the score-softmax leaf at both
	// workgroup widths: sequence <= 32 selects `sdpa_softmax_n32`, wider
	// sequences select the 256-lane `sdpa_softmax`. Causal, additive-mask
	// and scaled variants cover the donor's select chain.
	for (batch_heads, sequence, head_dim, masked, causal, scale) in [
		(4usize, 7usize, 4usize, false, true, 1.0_f32),
		(2, 24, 8, true, false, 0.25),
		(2, 40, 4, false, true, 0.5),
		(1, 37, 6, true, true, 0.125),
	] {
		let count = batch_heads * sequence * head_dim;
		let query = crate::Matrix::new_from_f32([batch_heads, sequence, head_dim], &values(count))?;
		let key = crate::Matrix::new_from_f32([batch_heads, sequence, head_dim], &values(count))?;
		let value = crate::Matrix::new_from_f32([batch_heads, sequence, head_dim], &values(count))?;
		let mask = masked.then(|| {
			crate::Matrix::new_from_f32(
				[batch_heads * sequence, sequence],
				&values(batch_heads * sequence * sequence),
			)
		});
		let mask = mask.transpose()?;
		let run = |mask: Option<&crate::Matrix>| -> Result<Vec<f32>> {
			crate::ml::matrix::scaled_dot_product_attention(&query, &key, &value, mask, scale, causal)?
				.read_f32()
		};
		let expected = run(mask.as_ref())?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_ATTENTION",
			|| run(mask.as_ref()),
			&expected,
			&format!("sdpa bh{batch_heads} s{sequence} d{head_dim} mask{masked} causal{causal}"),
		)?;
	}
	// `MlSmallCausalSdpaF32` — the fused seven-node causal route. The
	// region only forms under capture (the head-repack intermediates are
	// observable on the eager path), so the tile substitution runs on the
	// fused node's own ABI.
	for (batch, sequence, heads, model_width) in [
		(2usize, 4usize, 2usize, 4usize),
		(1, 16, 4, 8),
		(1, 37, 2, 6),
	] {
		let input = crate::Matrix::new_from_f32(
			[batch * sequence, model_width],
			&values(batch * sequence * model_width),
		)?;
		// The fused incumbent kernel's serial row-softmax differs in
		// reduction order from the member `sdpa_softmax` tree route, so the
		// oracle must be the fused donor itself: capture the incumbent
		// under `OA_JIT_FUSION`, then the tile substitution against it.
		let run = || -> Result<Vec<f32>> {
			let attention = crate::ml::nn::MultiHeadAttention::new_with_seed(
				model_width,
				heads,
				sequence,
				0x4d41_534b,
			)?;
			let (plan, output) = engine.capture(|| attention.forward(&input))?;
			let _event = engine.submit(&plan)?;
			plan.wait()?;
			output.read_f32()
		};
		let expected = {
			let _fusion = ScopedVar::set("OA_JIT_FUSION", std::ffi::OsStr::new("1"));
			run()?
		};
		tile_route_bit_exact(
			&device,
			"OA_TILE_ATTENTION",
			run,
			&expected,
			&format!("fused sdpa b{batch} s{sequence} h{heads} d{model_width}"),
		)?;
	}

	for (batch_heads, sequence, head_dim) in [(2usize, 5usize, 4usize), (1, 9, 3)] {
		let count = batch_heads * sequence * head_dim;
		let scale = 1.0 / (head_dim as f32).sqrt();
		let query = crate::Matrix::new_from_f32([batch_heads, sequence, head_dim], &values(count))?;
		let key = crate::Matrix::new_from_f32([batch_heads, sequence, head_dim], &values(count))?;
		let value = crate::Matrix::new_from_f32([batch_heads, sequence, head_dim], &values(count))?;
		let run = || -> Result<Vec<f32>> {
			crate::ml::matrix::flash_attention_causal(&query, &key, &value, scale)?.read_f32()
		};
		let expected = run()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_ATTENTION",
			run,
			&expected,
			&format!("flash causal b{batch_heads} s{sequence} d{head_dim}"),
		)?;
	}

	for (sequence, head_dim) in [(4usize, 2usize), (7, 3)] {
		let count = sequence * head_dim;
		let scale = 0.7_f32;
		let target = crate::Matrix::new_from_f32([1, sequence, head_dim], &values(count))?;
		let indices =
			crate::Matrix::new_from_slice([sequence], &(0..sequence as u32).collect::<Vec<_>>())?;
		let run = || -> Result<Vec<f32>> {
			let make = |salt: usize| -> Result<crate::ml::nn::Embedding> {
				crate::ml::nn::Embedding::new_from_matrix(crate::Matrix::new_from_f32(
					[sequence, head_dim],
					&gemm_values(count, salt),
				)?)
			};
			let query = make(0x11)?;
			let key = make(0x23)?;
			let value = make(0x37)?;
			let tape = crate::ml::GradientTape::new();
			let output = crate::ml::matrix::flash_attention_causal(
				&query.forward(&indices)?.reshape([1, sequence, head_dim])?,
				&key.forward(&indices)?.reshape([1, sequence, head_dim])?,
				&value.forward(&indices)?.reshape([1, sequence, head_dim])?,
				scale,
			)?;
			let loss = crate::ml::loss::mse(&output, &target)?;
			tape.backward(&loss)?;
			let mut gradients = Vec::with_capacity(3 * count);
			for parameter in [&query, &key, &value] {
				gradients.extend(
					parameter
						.weight()
						.gradient()
						.expect("flash operand gradient is missing")
						.read_f32()?,
				);
			}
			Ok(gradients)
		};
		let expected = run()?;
		tile_route_bit_exact(
			&device,
			"OA_TILE_ATTENTION",
			run,
			&expected,
			&format!("flash backward s{sequence} d{head_dim}"),
		)?;
	}
	Ok(())
}
