//! Phase 1 differential oracle and matched replay experiment, explicitly invoked.
//! No graph matcher or production JIT route is introduced by this fixture.

use super::*;
use crate::runtime::{
	BufferAccess, PushConstant, Storage,
	device::{
		command::jit_lab::{ChainContract, PreparedChain},
		features::ExecutionProfile,
	},
	dna::{
		elementwise::Program,
		kernel::{
			KernelId,
			jit::{JitArtifacts, SlangCompiler},
		},
	},
	executable_graph::{BufferUse, ComputeNode, ExecutableGraph},
	fusion,
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
			graph: ExecutableGraph::from_nodes(nodes)?,
			jit_output,
			host,
			count,
		})
	}

	fn tape_graph(&self, device: &Arc<Device>) -> Result<(ExecutableGraph, Vec<Storage>)> {
		let refs = self.inputs.iter().chain(&self.outputs).collect::<Vec<_>>();
		let (nodes, tapes) =
			fusion::jit_lab::diagnostic_elementwise_chain(device, self.graph.nodes().to_vec(), &refs);
		assert_eq!(
			nodes.len(),
			1,
			"diagnostic tape must actually fuse all 22 operations"
		);
		assert_eq!(nodes[0].kernel, KernelId::MatrixFusedChainF32);
		Ok((ExecutableGraph::from_nodes(nodes)?, tapes))
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
		if bounded { "compatibility" } else { "strict" },
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
				let mut candidates = fusion::jit_lab::discover_programs(fixture.graph.nodes(), &refs);
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
				let (nodes, _tapes) = fusion::jit_lab::diagnostic_elementwise_chain(
					&device,
					fixture.graph.nodes().to_vec(),
					&refs,
				);
				assert_eq!(nodes.len(), 1);
				let tape = device.record_reusable_compute_graph(&ExecutableGraph::from_nodes(nodes)?)?;
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
				let mut wrong = fixture.graph.nodes().to_vec();
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
	let mut nodes = first.graph.nodes().to_vec();
	nodes.extend_from_slice(second.graph.nodes());
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
	let scope = crate::runtime::device::vkcache::VkCacheScope::new(device.physical());
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
	let mut candidates = fusion::jit_lab::discover_programs(fixture.graph.nodes(), &refs);
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
		use crate::runtime::device::vkcache::{self, SectionUpdate, VkCacheFile};
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
	let (nodes, _tapes) = fusion::fuse_elementwise_nodes(
		&device,
		unfused_fixture.graph.nodes().to_vec(),
		&unfused_refs,
	);
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
	let candidates = fusion::jit_lab::discover_programs(fixture.graph.nodes(), &refs);
	assert_eq!(candidates.len(), 1, "one discovered region");
	pump_jit_admission(&device, &candidates[0].0);
	let before = device.fusion_census().snapshot();
	let (nodes, tapes) =
		fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
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
	let jit_graph = ExecutableGraph::from_nodes(nodes)?;
	let mut changed = jit_graph.nodes().to_vec();
	changed[0].jit.as_mut().unwrap().key.push(0);
	assert_ne!(
		jit_graph.identity(),
		ExecutableGraph::from_nodes(changed)?.identity(),
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
	let (nodes, _tapes) =
		fusion::fuse_elementwise_nodes(&device, small.graph.nodes().to_vec(), &small_refs);
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
	let candidates = fusion::jit_lab::discover_programs(fixture.graph.nodes(), &refs);
	assert_eq!(
		candidates.len(),
		1,
		"observed chain must produce one region"
	);
	pump_jit_admission(&device, &candidates[0].0);
	let (nodes, tapes) =
		fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
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
	let graph = ExecutableGraph::from_nodes(nodes)?;
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
	let (nodes, _tapes) =
		fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
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
	let (nodes, _tapes) =
		fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
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
	let (nodes, _tapes) =
		fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
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
	let candidates = fusion::jit_lab::discover_programs(fixture.graph.nodes(), &refs);
	assert_eq!(candidates.len(), 1, "one discovered region");
	pump_jit_admission(&device, &candidates[0].0);
	let (nodes, _tapes) =
		fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
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
		let mut candidates = fusion::jit_lab::discover_programs(fixture.graph.nodes(), &refs);
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
	let candidates = fusion::jit_lab::discover_programs(fixture.graph.nodes(), &refs);
	assert_eq!(candidates.len(), 1, "one discovered region");
	pump_jit_admission(&device, &candidates[0].0);
	let mut routes = (0_usize, 0_usize);
	for _ in 0..12 {
		let (nodes, tapes) =
			fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
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
		let graph = ExecutableGraph::from_nodes(nodes)?;
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
		let candidates = fusion::jit_lab::discover_programs(fixture.graph.nodes(), &refs);
		assert_eq!(candidates.len(), 1, "one discovered region");
		pump_jit_admission(&device, &candidates[0].0);
		let mut signature = None;
		for _ in 0..rounds {
			let (nodes, tapes) =
				fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
			assert_eq!(nodes.len(), 1, "small chain must fuse");
			signature = nodes[0].fusion_signature;
			let graph = ExecutableGraph::from_nodes(nodes)?;
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
		let (nodes, tapes) =
			fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
		assert_eq!(nodes.len(), 1, "small chain must fuse");
		signature = nodes[0].fusion_signature;
		let graph = ExecutableGraph::from_nodes(nodes)?;
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
		let (fused, _tapes) =
			fusion::fuse_elementwise_nodes(&device, fixture.graph.nodes().to_vec(), &refs);
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
			update.section_type == crate::runtime::device::vkcache::SECTION_JIT_ARTIFACT_INDEX
		})
		.unwrap();
	let entries =
		crate::runtime::device::vkcache::jit::decode_index(index.payload.as_ref().unwrap())?;
	assert_eq!(
		entries.len(),
		2,
		"only executable regions may consume compilation entries"
	);
	let command = device.record_reusable_compute_graph(&ExecutableGraph::from_nodes(nodes)?)?;
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
