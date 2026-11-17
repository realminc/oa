use super::*;
use crate::runtime::dna::tuning::ObservedRoute;

#[test]
#[ignore = "explicit Vulkan device; proves unread timing pools do not form a Device cycle"]
fn unread_timing_pool_does_not_retain_owning_device() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let engine = crate::Engine::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	let value = crate::matrix::from_f32_on(&engine, [1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);
	let weak = Arc::downgrade(&device);
	let timings = device.route_timings(vec![(1, ObservedRoute::Tape)])?;
	device.jit_observe_timings(timings.clone());
	drop(timings);
	drop(engine);
	drop(device);
	assert!(
		weak.upgrade().is_none(),
		"unsubmitted queries must not form an ownership cycle"
	);
	Ok(())
}

#[test]
#[ignore = "explicit Vulkan device; bounds failed JIT admissions without invoking Slang"]
fn negative_pipeline_admissions_stay_bounded() -> crate::Result<()> {
	use crate::runtime::dna::ir::elementwise::{Instruction, Source};
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let engine = crate::Engine::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	let value = crate::matrix::from_f32_on(&engine, [1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);
	let mut fusion = JitFusion::new();
	fusion.restored = true;
	fusion.compiler = Some(Err(jit_missing("deliberately absent test compiler")));
	for constant in 0..MAX_PIPELINE_ENTRIES + 16 {
		let program = Program {
			element_count: 1,
			input_bytes: vec![4],
			outputs: vec![0],
			instructions: vec![Instruction {
				opcode: 12,
				sources: vec![Source::Input(0), Source::Constant(constant as u32)],
				parameters: vec![],
			}],
		};
		assert!(fusion.admit(&device, &program, false).is_none());
	}
	assert_eq!(fusion.pipelines.len(), MAX_PIPELINE_ENTRIES);
	Ok(())
}

#[test]
#[ignore = "explicit trusted Slang library and spirv-val; cached artifact needs no compiler session"]
fn cached_artifact_is_usable_with_unavailable_compiler() -> crate::Result<()> {
	use crate::runtime::dna::ir::elementwise::{Instruction, Source};
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let compiler = SlangCompiler::open(std::path::Path::new(&library))?;
	let program = Program {
		element_count: 17,
		input_bytes: vec![68],
		outputs: vec![0],
		instructions: vec![Instruction {
			opcode: 12,
			sources: vec![Source::Input(0), Source::Constant(1.25f32.to_bits())],
			parameters: vec![],
		}],
	};
	let mut fusion = JitFusion::new();
	fusion.artifacts.chain(&compiler, &program, true)?;
	drop(compiler);
	fusion.compiler = Some(Err(jit_missing("deliberately absent test compiler")));
	// The cached artifact serves admissions without the compiler; the
	// descriptor profile remains part of artifact identity.
	assert!(fusion.artifacts.get(&program, true).is_some());
	assert!(fusion.artifacts.get(&program, false).is_none());
	let census = FusionCensus::new();
	fusion.queue_compile(&census, &program, false);
	assert_eq!(census.snapshot().jit_compiler_unavailable_count(), 1);
	assert!(
		fusion
			.pipelines
			.get(&program.artifact_key())
			.is_some_and(Option::is_none),
		"an unavailable compiler is remembered as a negative admission"
	);
	Ok(())
}

impl JitFusion {
	/// Completed observations `(signature, specialized samples, tape samples,
	/// selected)`, used by gated tests.
	pub(in crate::runtime::device) fn observations(&mut self) -> Vec<(u64, usize, usize, bool)> {
		self.drain_pending();
		self
			.observations
			.iter()
			.map(|(signature, observation)| {
				(
					*signature,
					observation.specialized_ns.len(),
					observation.tape_ns.len(),
					observation.specialized,
				)
			})
			.collect()
	}
}
