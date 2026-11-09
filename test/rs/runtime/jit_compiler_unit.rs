//! Phase 1 laboratory tests over the production specialization compiler.
//!
//! The manual chain-8 recipe artifact remains test-only; the compiler,
//! artifact index, persistence codecs and ABI validation live in
//! `runtime::dna::kernel::jit`.

use crate::Result;

use super::{CompiledChain, SlangCompiler, validate_program_abi, validate_with_spirv_tools};

impl SlangCompiler {
	pub(in crate::runtime) fn compile_chain(&self, bounded: bool) -> Result<CompiledChain> {
		let started = std::time::Instant::now();
		let words = self.compile_source(&chain_source(bounded), bounded)?;
		let compile_time = started.elapsed();
		let started = std::time::Instant::now();
		validate_with_spirv_tools(&words, bounded)?;
		validate_chain_abi(&words, bounded)?;
		Ok(CompiledChain {
			words,
			key: Vec::new(),
			bindings: 4,
			compile_time,
			validation_time: started.elapsed(),
			bounded,
		})
	}
}

fn validate_chain_abi(words: &[u32], bounded: bool) -> Result<()> {
	validate_program_abi(words, bounded, 4)
}

/// Manual specialization of the SDK chain-8 benchmark, not a general emitter.
/// Formula provenance: src/slang/matrix/elemwise/{mul,add,scale}.gen.slang and
/// src/slang/ml/nn/activation/gelu.slang; donor FP32 operation order retained.
/// OA donor: core/shader/compute/flat/{mul,add,scale}.gen.slang and
/// ml/shader/compute/nn/activation/gelu.slang, under source/cpp/lib/oa.
/// Adaptation: mechanical straight-line lowering; no donor JIT exists.
fn chain_source(bounded: bool) -> String {
	let descriptors = if bounded { "4" } else { "" };
	let mut source = format!(
		r#"
struct PushConstants {{ uint x; uint y; uint z; uint output; uint count; }};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[{descriptors}];
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {{
	uint index = thread.x;
	if (index >= push.count) return;
	float x = asfloat(storage_buffers[push.x].Load(index * 4u));
	float y = asfloat(storage_buffers[push.y].Load(index * 4u));
	float z = asfloat(storage_buffers[push.z].Load(index * 4u));
	float value = x * y;
"#
	);
	for round in 0..7 {
		source.push_str(&format!(r#"
	float add_{round} = value + z;
	float x_{round} = add_{round} * 1.25f;
	float inner_{round} = 0.7978845608028654f * (x_{round} + 0.044715f * x_{round} * x_{round} * x_{round});
	value = 0.5f * x_{round} * (1.0f + tanh(inner_{round}));
"#));
	}
	// Invocation i exclusively owns output[i]; all inputs are read-only and
	// disjoint from output, and the tail exits before every load/store.
	source.push_str("\tstorage_buffers[push.output].Store(index * 4u, asuint(value));\n}\n");
	source
}

#[test]
#[ignore = "requires explicit OA_TEST_SLANG_LIBRARY and spirv-val"]
fn compiler_errors_release_request_and_allow_recovery() -> Result<()> {
	let path = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit trusted Slang library");
	let compiler = SlangCompiler::open(std::path::Path::new(&path))?;
	assert!(
		compiler
			.compile_source("this is not a shader;", true)
			.is_err()
	);
	let first = compiler.compile_chain(true)?;
	let second = compiler.compile_chain(true)?;
	assert_eq!(first.words, second.words, "deterministic specialization");
	// Compile/validate both profiles; this does not qualify Strict GPU execution.
	compiler.compile_chain(false)?;
	for source in [
		chain_source(true).replace("storage_buffers[4]", "storage_buffers[5]"),
		chain_source(true).replace("numthreads(256", "numthreads(128"),
		chain_source(true).replace("uint count;", "float count;"),
	] {
		let words = compiler.compile_source(&source, true)?;
		assert!(
			validate_chain_abi(&words, true).is_err(),
			"reject incompatible generated ABI"
		);
	}
	assert!(SlangCompiler::open(std::path::Path::new("/nonexistent/oa-slang-library")).is_err());
	let mut damaged = first.words.clone();
	damaged[1] = 0;
	assert!(validate_chain_abi(&damaged, true).is_err());
	Ok(())
}

#[test]
#[ignore = "explicit trusted Slang library and installed spirv-val"]
fn shared_ir_compiles_both_descriptor_profiles() -> Result<()> {
	use crate::runtime::dna::elementwise::{Instruction, Program, Source};
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let compiler = SlangCompiler::open(std::path::Path::new(&library))?;
	let operations = vec![
		Instruction {
			opcode: 4,
			sources: vec![Source::Input(0), Source::Input(1)],
			parameters: vec![4, 3, 0],
		},
		Instruction {
			opcode: 10,
			sources: vec![
				Source::Value(0),
				Source::Broadcast {
					input: 2,
					rank: 2,
					dimensions: [3, 3, 0, 0, 0, 0, 0, 0],
					strides: [0, 1, 0, 0, 0, 0, 0, 0],
				},
			],
			parameters: vec![],
		},
		Instruction {
			opcode: 47,
			sources: vec![Source::Value(1), Source::Constant(0.5f32.to_bits())],
			parameters: vec![1.0f32.to_bits()],
		},
	];
	let program = Program {
		element_count: 9,
		input_bytes: vec![48, 4, 12],
		instructions: operations,
		outputs: vec![0, 2],
	};
	assert!(program.validate());
	for bounded in [true, false] {
		let artifact = compiler.compile_program(&program, bounded)?;
		assert!(validate_program_abi(artifact.words(), bounded, 4).is_err());
		assert_eq!(
			artifact.words(),
			compiler.compile_program(&program, bounded)?.words()
		);
	}
	Ok(())
}

impl super::JitArtifacts {
	/// Persist every compiled artifact through the envelope writer policy.
	/// Production publication flows through `section_updates` inside the
	/// Engine's single atomic commit; direct persistence is a laboratory path.
	pub(in crate::runtime) fn persist(
		&self,
		path: &crate::Path,
		scope: &crate::runtime::device::vkcache::VkCacheScope,
	) -> Result<()> {
		crate::runtime::device::vkcache::update(path, scope, &self.section_updates()?)
	}
}
