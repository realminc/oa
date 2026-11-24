//! Phase 1 laboratory tests over the production specialization compiler.
//!
//! The manual chain-8 recipe artifact remains test-only; the compiler,
//! artifact index, persistence codecs and ABI validation live in
//! `runtime::dna::compiler`.

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
	// Compile/validate both profiles; this does not qualify Core GPU execution.
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
	use crate::runtime::dna::ir::elementwise::{Instruction, Program, Source};
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
		scope: &crate::runtime::device::logical::vkcache::VkCacheScope,
	) -> Result<()> {
		crate::runtime::device::logical::vkcache::update(path, scope, &self.section_updates()?)
	}
}

#[test]
#[ignore = "requires explicit OA_TEST_SLANG_LIBRARY and spirv-val"]
fn block_gemm_template_compiles_and_validates() -> Result<()> {
	use crate::runtime::dna::ir::tile::{BiasAct, BlockProgram};
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let compiler = SlangCompiler::open(std::path::Path::new(&library))?;
	// Donor geometry: src/slang/matrix/blas/mat_mul_nt_tiled.gen.slang.
	let program = BlockProgram::mat_mul_nt(64, 64, 16, 4, 4, 4).unwrap();
	for bounded in [true, false] {
		let source = program.slang_source(bounded)?;
		let words = compiler.compile_source(&source, bounded)?;
		validate_with_spirv_tools(&words, bounded)?;
		// Block templates carry `bindings + scalars` u32 push fields —
		// the elementwise push contract (bindings + count) does not apply.
		assert_eq!(
			crate::runtime::dna::kernel::reflect_push_constant_size(&words)?,
			(3 + 3) * 4
		);
		assert_eq!(words, compiler.compile_source(&source, bounded)?);
	}
	// The seed candidate set and a tail-geometry variant (staging width 1,
	// where the masked path is the only staging route) must also compile.
	for args in [
		(64, 32, 16, 4, 4, 4),
		(32, 64, 16, 4, 4, 4),
		(16, 16, 4, 2, 2, 1),
	] {
		let program = BlockProgram::mat_mul_nt(args.0, args.1, args.2, args.3, args.4, args.5).unwrap();
		let source = program.slang_source(true)?;
		let words = compiler.compile_source(&source, true)?;
		validate_with_spirv_tools(&words, true)?;
		assert_eq!(
			crate::runtime::dna::kernel::reflect_push_constant_size(&words)?,
			(3 + 3) * 4
		);
	}
	// The softmax row-reduce template (2 buffers + 3 scalars) exercises
	// lane folds, tree reductions and the group guard through the same
	// production compile path.
	// The reduce/normalization family exercises lane folds, tree
	// reductions, group guards and typed push scalars through the same
	// production compile path.
	for (program, push_u32s) in [
		(BlockProgram::softmax(256).unwrap(), 3u32),
		(BlockProgram::log_softmax(256).unwrap(), 3),
		(BlockProgram::rms_norm(256).unwrap(), 3),
		(BlockProgram::layer_norm(256).unwrap(), 3),
		(BlockProgram::softmax_backward(256).unwrap(), 3),
		(BlockProgram::log_softmax_backward(256).unwrap(), 3),
		(BlockProgram::linear(64, 64, 16, 4, 4, 4).unwrap(), 3),
		(BlockProgram::linear_act(64, 64, 16, 4, 4, 4).unwrap(), 5),
		(
			BlockProgram::linear_residual(64, 64, 16, 4, 4, 4).unwrap(),
			3,
		),
		(
			BlockProgram::linear_residual_act(64, 64, 16, 4, 4, 4).unwrap(),
			5,
		),
		(
			BlockProgram::linear_gated_act(64, 64, 16, 4, 4, 4).unwrap(),
			5,
		),
		(BlockProgram::linear_triple(64, 64, 16, 4, 4, 4).unwrap(), 3),
		(BlockProgram::linear_concat(64, 64, 16, 4, 4, 4).unwrap(), 6),
		(
			BlockProgram::linear_residual_norm(64, 64, 16, 4, 4, 4).unwrap(),
			4,
		),
		(
			BlockProgram::linear_residual_norm(64, 32, 16, 4, 4, 4).unwrap(),
			4,
		),
		(
			BlockProgram::qkv_projection_bias(32, 32, 32, 1, 4, 1).unwrap(),
			3,
		),
		(
			BlockProgram::gate_up_swiglu_bias(32, 32, 32, 1, 4, 1).unwrap(),
			3,
		),
		(BlockProgram::grouped_linear_m_act().unwrap(), 5),
		(BlockProgram::grouped_linear_m().unwrap(), 4),
		(BlockProgram::grouped_gemm_m().unwrap(), 4),
		(BlockProgram::grouped_linear_m_data_backward().unwrap(), 4),
		(
			BlockProgram::grouped_linear_m_parameter_backward().unwrap(),
			4,
		),
		(
			BlockProgram::grouped_gemm_m_parameter_backward().unwrap(),
			4,
		),
		(BlockProgram::mat_mul_nt_gemv(256).unwrap(), 3),
		// 6 bindings + 5 u32 scalars + 1 f32 alpha = 48-byte push block.
		(BlockProgram::grouped_backward_prologue().unwrap(), 6),
		(BlockProgram::moe_router_select(), 3),
		(BlockProgram::moe_gather(), 2),
		(BlockProgram::moe_gather_backward(), 4),
		(BlockProgram::moe_combine(), 4),
		(BlockProgram::moe_combine_backward(), 4),
		(BlockProgram::moe_route_weights(), 3),
		(BlockProgram::moe_route_weights_backward(), 3),
		(BlockProgram::layer_norm_backward(256).unwrap(), 2),
		// 5 bindings + 2 u32 scalars + 1 f32 epsilon = 32-byte push.
		(BlockProgram::rms_norm_backward(256).unwrap(), 3),
		(BlockProgram::cross_entropy(256, false).unwrap(), 3),
		(BlockProgram::cross_entropy(256, true).unwrap(), 3),
		(BlockProgram::cross_entropy_backward(256, false).unwrap(), 3),
		(BlockProgram::cross_entropy_backward(256, true).unwrap(), 4),
		(BlockProgram::layer_norm_backward_sum(256).unwrap(), 3),
		(BlockProgram::embedding_backward_sum(), 3),
		(BlockProgram::embedding_backward(), 3),
		(BlockProgram::embedding_backward_u8(), 3),
		(BlockProgram::sum_backward(), 3),
		(BlockProgram::max(256).unwrap(), 1),
		(BlockProgram::argmax(256).unwrap(), 1),
		(BlockProgram::max_backward(), 1),
		(BlockProgram::max_wave(256).unwrap(), 1),
		(BlockProgram::sum_wave(256).unwrap(), 1),
		(BlockProgram::sum_axis_cooperative(256).unwrap(), 3),
		(BlockProgram::conv_1d_im2_col(), 8),
		(BlockProgram::conv_1d_bias_add(), 2),
		(BlockProgram::conv_1d_layout_transpose(), 3),
		(BlockProgram::conv_2d(), 11),
		(BlockProgram::conv_2d_relu(), 11),
		(BlockProgram::conv_1d_parameter_backward(), 10),
		(BlockProgram::conv_1d_backward(), 9),
		(BlockProgram::conv_1d_transpose(), 9),
		(BlockProgram::conv_2d_backward(), 11),
		(BlockProgram::conv_2d_parameter_backward(), 12),
		(BlockProgram::conv_2d_transpose(), 10),
		(BlockProgram::conv_2d_transpose_bias_add(), 4),
		(BlockProgram::conv_2d_transpose_backward(), 10),
		(BlockProgram::conv_2d_transpose_parameter_backward(), 11),
		(BlockProgram::conv_1d_bias_layout_transpose(), 3),
		(BlockProgram::conv_1d_bias_layout_transpose_relu(), 3),
		(
			BlockProgram::conv_1d_bias_layout_transpose_residual_relu(),
			3,
		),
		(BlockProgram::conv_1d_bias_copy(), 2),
		(BlockProgram::conv_1d_bias_layout_transpose_norm(false), 4),
		(BlockProgram::conv_1d_bias_layout_transpose_norm(true), 4),
		(BlockProgram::avg_pool_2d(), 9),
		(BlockProgram::max_pool_2d(), 9),
		(BlockProgram::adaptive_avg_pool_2d(), 6),
		(BlockProgram::avg_pool_2d_backward(), 9),
		(BlockProgram::max_pool_2d_backward(), 9),
		(BlockProgram::max_pool_2d_relu_backward(), 9),
		(BlockProgram::adaptive_avg_pool_2d_backward(), 6),
		(BlockProgram::split_heads(), 4),
		(BlockProgram::merge_heads(), 4),
		(BlockProgram::sdpa_softmax(256).unwrap(), 5),
		(BlockProgram::sdpa_softmax(32).unwrap(), 5),
		(BlockProgram::small_causal_sdpa(), 5),
		(BlockProgram::linear_backward(), 3),
		(BlockProgram::linear_backward_prologue(), 5),
		(BlockProgram::linear_backward_norm(), 3),
		(BlockProgram::linear_backward_norm_add(), 3),
		(BlockProgram::linear_backward_bmm_chain(), 6),
		(BlockProgram::flash_causal(), 4),
		(BlockProgram::flash_causal_backward_q(), 4),
		(BlockProgram::flash_causal_backward_kv(), 4),
		(BlockProgram::argmax_wave(256).unwrap(), 1),
		(
			BlockProgram::mat_mul_nt_bias(64, 64, 16, 4, 4, 4, None).unwrap(),
			3,
		),
		(
			BlockProgram::mat_mul_nt_bias(64, 64, 16, 4, 4, 4, Some(BiasAct::Relu)).unwrap(),
			3,
		),
		(
			BlockProgram::mat_mul_nt_bias(64, 64, 16, 4, 4, 4, Some(BiasAct::Gelu)).unwrap(),
			3,
		),
		(
			BlockProgram::mat_mul_nt_bias(64, 64, 16, 4, 4, 4, Some(BiasAct::Silu)).unwrap(),
			3,
		),
	] {
		for bounded in [true, false] {
			let source = program.slang_source(bounded)?;
			let words = compiler.compile_source(&source, bounded)?;
			validate_with_spirv_tools(&words, bounded)?;
			assert_eq!(
				crate::runtime::dna::kernel::reflect_push_constant_size(&words)?,
				(program.bindings.len() as u32 + push_u32s) * 4
			);
			assert_eq!(words, compiler.compile_source(&source, bounded)?);
		}
	}
	Ok(())
}
