use super::*;
use crate::runtime::dna::ir::elementwise::{Instruction, Program, Source};

fn chain() -> Program {
	Program {
		element_count: 17,
		input_bytes: vec![68],
		instructions: vec![
			Instruction {
				opcode: 12,
				sources: vec![Source::Input(0), Source::Constant(1.25f32.to_bits())],
				parameters: vec![],
			},
			Instruction {
				opcode: 10,
				sources: vec![Source::Value(0), Source::Input(0)],
				parameters: vec![],
			},
		],
		outputs: vec![1],
	}
}

#[test]
fn pointwise_lowering_preserves_emission_order_and_bindings() {
	let tile = TileProgram::from_elementwise(&chain());
	assert_eq!(tile.input_count(), 1);
	assert_eq!(tile.bindings, vec![Binding::Input(0), Binding::Output(0)]);
	match &tile.ops[..] {
		[
			TileOp::LoadLinear { slot: 0 },
			TileOp::Pointwise {
				value: 0,
				opcode: 12,
				..
			},
			TileOp::Pointwise {
				value: 1,
				opcode: 10,
				..
			},
			TileOp::Store { slot: 0, value: 1 },
		] => {}
		other => panic!("unexpected op order: {other:?}"),
	}
	let source = tile.slang_source(true).unwrap();
	assert!(source.contains("float input0 = asfloat"));
	assert!(source.contains("float v0 = eval_element(12u, input0, asfloat(0x3fa00000u)"));
	assert!(source.contains("float v1 = eval_element(10u, v0, input0"));
	assert!(source.contains("storage_buffers[push.buffer1].Store(thread.x * 4u, asuint(v1))"));
	assert_eq!(source, tile.slang_source(true).unwrap());
}

#[test]
fn lowering_rejects_empty_program() {
	let tile = TileProgram::from_elementwise(&Program {
		element_count: 0,
		input_bytes: vec![],
		instructions: vec![],
		outputs: vec![],
	});
	assert!(tile.slang_source(true).is_err());
}

fn gemm() -> BlockProgram {
	// Donor geometry: src/slang/matrix/blas/mat_mul_nt_tiled.gen.slang
	// (BM=64, BN=64, BK=16, TM=4, TN=4, LOAD_WIDTH=4, 256 lanes).
	BlockProgram::mat_mul_nt(64, 64, 16, 4, 4, 4).unwrap()
}

#[test]
fn mat_mul_nt_template_lowers_to_tiled_gemm() {
	let program = gemm();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.shape.lanes, 256);
	assert_eq!(program.workgroups(128, 65), [2, 2, 1]);
	assert_eq!(program.workgroups(64, 64), [1, 1, 1]);
	match &program.ops[..] {
		[
			TileOp::ClearTile { tile: 2 },
			TileOp::Loop {
				extent: Extent::Push(2),
				step: 16,
				start: None,
				body,
			},
			TileOp::StoreTile { acc: 2, .. },
		] => match &body[..] {
			[
				TileOp::LoadShared { tile: 0, .. },
				TileOp::LoadShared { tile: 1, .. },
				TileOp::Barrier,
				TileOp::DotShared { acc: 2, .. },
				TileOp::Barrier,
			] => {}
			other => panic!("unexpected loop body: {other:?}"),
		},
		other => panic!("unexpected op order: {other:?}"),
	}
}

#[test]
fn mat_mul_nt_emission_is_deterministic_and_masked() {
	let program = gemm();
	let source = program.slang_source(true).unwrap();
	assert!(source.contains("groupshared float shared0[64u * 17u]"));
	assert!(source.contains("groupshared float shared1[64u * 17u]"));
	assert!(source.contains("[numthreads(256, 1, 1)]"));
	assert!(source.contains("for (uint loop2 = 0u, bound2 = push.k; loop2 < bound2; loop2 += 16u)"));
	assert!(source.contains("Load<float4>"));
	assert!(source.contains("global_row < push.m && global_column + 4u <= push.k"));
	assert!(source.contains("global_column + j < push.k"));
	assert_eq!(
		source.matches("GroupMemoryBarrierWithGroupSync()").count(),
		2
	);
	assert!(source.contains("tile2[0u * 4u + 0u] += left_fragment[0] * right_fragment[0]"));
	assert!(source.contains("output_row < push.m && output_column < push.n"));
	assert_eq!(source, program.slang_source(true).unwrap());
	assert_ne!(
		source,
		program.slang_source(false).unwrap(),
		"descriptor profiles differ"
	);
}

#[test]
fn mat_mul_nt_rejects_invalid_geometry() {
	for args in [
		(64, 64, 18, 4, 4, 4),   // staging width does not divide tile_k
		(64, 64, 16, 3, 4, 4),   // fragment rows do not divide tile_rows
		(64, 64, 16, 4, 3, 4),   // fragment columns do not divide tile_columns
		(128, 128, 16, 1, 1, 4), // fragment grid exceeds workgroup floor
		(64, 64, 16, 4, 4, 8),   // vec over the 16B load ceiling
	] {
		assert!(
			BlockProgram::mat_mul_nt(args.0, args.1, args.2, args.3, args.4, args.5).is_err(),
			"accepts invalid geometry {args:?}"
		);
	}
	// A non-multiple staging width is also rejected at emission.
	let mut program = gemm();
	program.ops[1] = match &program.ops[1] {
		TileOp::Loop {
			extent,
			step,
			start,
			body,
		} => {
			let mut body = body.clone();
			body[0] = match &body[0] {
				TileOp::LoadShared {
					tile,
					binding,
					row_origin,
					column_origin,
					row_extent,
					column_extent,
					vec: _,
					row_offset,
					mask_origin,
					z,
					column_offset,
					transposed,
				} => TileOp::LoadShared {
					tile: *tile,
					binding: *binding,
					row_origin: *row_origin,
					column_origin: *column_origin,
					row_extent: *row_extent,
					column_extent: *column_extent,
					vec: 3,
					row_offset: *row_offset,
					mask_origin: *mask_origin,
					z: *z,
					column_offset: *column_offset,
					transposed: *transposed,
				},
				_ => unreachable!(),
			};
			TileOp::Loop {
				extent: *extent,
				step: *step,
				start: *start,
				body,
			}
		}
		_ => unreachable!(),
	};
	assert!(program.slang_source(true).is_err());
}

#[test]
fn mat_mul_nt_bias_templates_emit_donor_epilogues() {
	// Donor geometry and ABI: src/slang/matrix/blas/mat_mul_nt_bias*.gen.slang
	// — `[left, right, bias, output]` bindings, `[m, n, k]` push scalars,
	// and a fixed inline activation on the single output store.
	for (act, store) in [
		(None, "asuint(linear_value)"),
		(Some(BiasAct::Relu), "asuint(max(0.0f, linear_value))"),
		(
			Some(BiasAct::Gelu),
			"asuint(0.5f * linear_value * (1.0f + tanh(0.7978845608f * (linear_value + 0.044715f * linear_value * linear_value * linear_value))))",
		),
		(
			Some(BiasAct::Silu),
			"asuint(linear_value * (1.0f / (1.0f + exp(-linear_value))))",
		),
	] {
		let program = BlockProgram::mat_mul_nt_bias(64, 64, 16, 4, 4, 4, act).expect("donor geometry");
		assert_eq!(program.bindings.len(), 4);
		assert_eq!(program.push_scalars, ["m", "n", "k"]);
		assert!(matches!(
			program.ops.last(),
			Some(TileOp::StoreTileBias {
				acc: 2,
				bias_binding: 2,
				binding: 3,
				..
			})
		));
		let source = program.slang_source(true).expect("emission");
		assert!(
			source.contains("asfloat(storage_buffers[buf2].Load(output_column * 4u))"),
			"{act:?}"
		);
		assert!(
			source.contains(&format!(
				"storage_buffers[buf3].Store(element * 4u, {store});"
			)),
			"{act:?}"
		);
		assert_eq!(
			source,
			program.slang_source(true).expect("re-emission"),
			"{act:?}"
		);
	}
}

#[test]
fn tile_programs_reject_cross_vocabulary_ops() {
	// Elementwise ops cannot appear in a block program.
	let mut block = gemm();
	block.ops.push(TileOp::Store { slot: 0, value: 0 });
	assert!(block.slang_source(true).is_err());
	// Block ops cannot appear in an elementwise program.
	let mut elementwise = TileProgram::from_elementwise(&chain());
	elementwise.ops.push(TileOp::Barrier);
	assert!(elementwise.slang_source(true).is_err());
}

#[test]
fn block_softmax_template_emits_donor_schedule() {
	let program = BlockProgram::softmax(256).expect("256 lanes is a legal geometry");
	let source = program.slang_source(true).expect("emission");
	assert_eq!(source, program.slang_source(true).expect("re-emission"));
	assert!(source.contains("[numthreads(256, 1, 1)]"));
	assert!(source.contains("groupshared float shared0[1u * 256u]"));
	// Workgroup-uniform guard, per-lane folds, tree reductions and the
	// final strided store all follow the donor's phase order.
	let guard = source
		.find("if (group_id.x >= (push.outer_size * push.inner_size))")
		.unwrap();
	let fold_max = source.find("tile1 = max(tile1, asfloat(").unwrap();
	let store_max = source.find("shared0[lane] = tile1;").unwrap();
	let tree = source
		.find("shared0[lane] = max(shared0[lane], shared0[lane + stride1]);")
		.unwrap();
	let fold_sum = source.find("tile1 = tile1 + (exp(").unwrap();
	let store = source.find("storage_buffers[buf1].Store(").unwrap();
	assert!(
		guard < fold_max
			&& fold_max < store_max
			&& store_max < tree
			&& tree < fold_sum
			&& fold_sum < store
	);
	// The donated tail barrier protects the last shared read.
	assert!(source.matches("GroupMemoryBarrierWithGroupSync();").count() >= 4);
}

#[test]
fn block_softmax_rejects_non_power_of_two_lanes() {
	assert!(BlockProgram::softmax(192).is_err());
	assert!(BlockProgram::softmax(2048).is_err());
	assert!(BlockProgram::softmax(1).is_ok());
}

#[test]
fn block_reduce_templates_emit_donor_schedules() {
	for program in [
		BlockProgram::log_softmax(256).unwrap(),
		BlockProgram::rms_norm(256).unwrap(),
		BlockProgram::layer_norm(256).unwrap(),
	] {
		let source = program.slang_source(true).expect("emission");
		assert_eq!(source, program.slang_source(true).expect("deterministic"));
		assert!(source.contains("[numthreads(256, 1, 1)]"));
		assert!(source.contains("GroupMemoryBarrierWithGroupSync();"));
	}
	let log_softmax = BlockProgram::log_softmax(256)
		.unwrap()
		.slang_source(true)
		.unwrap();
	assert!(log_softmax.contains("(tile2 + log(shared0[0u]))"));
	assert!(log_softmax.contains("- tile3)"));
	let rms = BlockProgram::rms_norm(256)
		.unwrap()
		.slang_source(true)
		.unwrap();
	assert!(rms.contains("float epsilon;"));
	assert!(rms.contains("(asfloat(0x3f800000u) / sqrt("));
	let ln = BlockProgram::layer_norm(256)
		.unwrap()
		.slang_source(true)
		.unwrap();
	assert!(ln.contains("rsqrt("));
	assert!(ln.contains("if (lane == 0u)"));
	assert!(ln.contains("buf5].Store(group_id.x * 4u"));
}

#[test]
fn block_reduce_templates_reject_bad_lanes() {
	assert!(BlockProgram::log_softmax(192).is_err());
	assert!(BlockProgram::rms_norm(192).is_err());
	assert!(BlockProgram::layer_norm(192).is_err());
}

#[test]
fn grouped_linear_m_act_template_is_deterministic() {
	let program = BlockProgram::grouped_linear_m_act().expect("grouped template");
	assert_eq!(program.bindings.len(), 6);
	assert_eq!(program.push_scalars.len(), 5);
	assert_eq!(program.workgroups(70, 24), [3, 1, 1]);
	let unbounded = program.slang_source(false).expect("unbounded source");
	let bounded = program.slang_source(true).expect("bounded source");
	assert_eq!(unbounded, program.slang_source(false).unwrap());
	assert_eq!(bounded, program.slang_source(true).unwrap());
	// Donor schedule anchors: dual expert scan, boundary fallback with
	// an early return, expert-offset weight staging, and the gated
	// epilogue under expert-relative bias addressing.
	assert!(unbounded.contains("numthreads(64, 1, 1)"));
	assert!(unbounded.matches("while (").count() >= 3);
	assert!(unbounded.contains("!= uint(tile6"));
	assert!(unbounded.contains("uint(tile5) * push.output_features"));
	assert!(unbounded.contains("GroupMemoryBarrierWithGroupSync"));
	assert!(unbounded.contains("tile_swiglu"));
	assert!(unbounded.contains("buf5].Store"));
}

#[test]
fn grouped_data_backward_template_is_deterministic() {
	let program = BlockProgram::grouped_linear_m_data_backward().expect("backward template");
	assert_eq!(program.bindings.len(), 4);
	assert_eq!(program.push_scalars.len(), 4);
	assert_eq!(program.workgroups(70, 24), [3, 1, 1]);
	let unbounded = program.slang_source(false).expect("unbounded source");
	let bounded = program.slang_source(true).expect("bounded source");
	assert_eq!(unbounded, program.slang_source(false).unwrap());
	assert_eq!(bounded, program.slang_source(true).unwrap());
	// Donor schedule anchors: dual expert scan, boundary fallback with
	// an early return, transpose-in-load weight staging carrying the
	// expert term on the load column, and reduction over outputs.
	assert!(unbounded.contains("numthreads(64, 1, 1)"));
	assert!(unbounded.matches("while (").count() >= 3);
	assert!(unbounded.contains("!= uint(tile"));
	assert!(
		unbounded
			.contains("(global_column + (uint(tile3) * push.output_features)) * push.input_features")
	);
	assert!(unbounded.contains("boundary_row * push.output_features + k"));
	assert!(unbounded.contains("GroupMemoryBarrierWithGroupSync"));
	assert!(unbounded.contains("buf3].Store"));
}

#[test]
fn grouped_parameter_backward_template_is_deterministic() {
	let program = BlockProgram::grouped_linear_m_parameter_backward().expect("parameter template");
	assert_eq!(program.bindings.len(), 5);
	assert_eq!(program.push_scalars.len(), 4);
	assert_eq!(program.workgroups(24, 70), [1, 3, 1]);
	let unbounded = program.slang_source(false).expect("unbounded source");
	assert_eq!(unbounded, program.slang_source(false).unwrap());
	// Donor schedule anchors: the offsets-range `begin..end` loop,
	// transpose-in-load staging masked `row < end`, the gated bias
	// fold inside the dot, and the expert-strided stores.
	assert!(unbounded.contains("numthreads(64, 1, 1)"));
	assert!(unbounded.contains("loop2 = uint(tile4), bound2 = uint(tile5); loop2 < bound2"));
	assert!(unbounded.contains("storage_buffers[buf2].Load((group_id.z) * 4u)"));
	assert!(unbounded.contains("storage_buffers[buf2].Load(((group_id.z + 1u)) * 4u)"));
	assert!(unbounded.contains("if ((((lane % 8u) == 0u) && (group_id.y == 0u)))"));
	assert!(unbounded.contains("tile3[0u] += left_fragment[0]"));
	assert!(unbounded.contains("buf3].Store"));
	assert!(unbounded.contains("buf4].Store"));
	assert!(unbounded.contains("GroupMemoryBarrierWithGroupSync"));
}
#[test]
fn grouped_backward_prologue_template_is_deterministic() {
	let source = BlockProgram::grouped_backward_prologue()
		.unwrap()
		.slang_source(true)
		.unwrap();
	assert!(source.contains("for (uint loop2 = uint(tile3), bound2 ="));
	assert!(source.contains("tile_prologue_value(global_row, global_column)"));
	assert!(source.contains("loop2 * push.output_features"));
	assert!(source.contains("storage_buffers[buf2].Store("));
	assert!(source.contains("store_row >= max("));
	assert!(source.contains("segment * 16u"));
}

#[test]
fn moe_templates_are_deterministic() {
	let router = BlockProgram::moe_router_select();
	assert_eq!(router.bindings.len(), 6);
	assert_eq!(router.push_scalars.len(), 3);
	assert_eq!(router.workgroups(7, 1), [7, 1, 1]);
	let source = router.slang_source(true).expect("router source");
	assert_eq!(source, router.slang_source(true).unwrap());
	// Donor schedule anchors: row guard, strided max/sum folds into
	// shared tree reductions, the serial top-K pass with the
	// value-then-lower-index argmax pair reduction over `taken[]`, and
	// the lane-strided mask materialization.
	assert!(source.contains("numthreads(256, 1, 1)"));
	assert!(source.contains("if (group_id.x >= push.rows) return;"));
	assert!(source.contains("for (uint loop2 = 0u, bound2 = push.k; loop2 < bound2; loop2 += 1u)"));
	assert!(source.contains("for (uint loop4 = 0u, bound4 = loop2; loop4 < bound4; loop4 += 1u)"));
	assert!(source.contains("shared2[loop4] == loop3"));
	assert!(source.contains("shared1[lane + stride2] < shared1[lane]"));
	assert!(source.contains("if (lane == 0u)"));
	assert!(source.contains("buf4].Store"));
	assert!(source.contains("float(min(tile9, 1u))"));
	assert_eq!(
		source.matches("GroupMemoryBarrierWithGroupSync").count(),
		10
	);

	for (name, program, bindings, scalars) in [
		("gather", BlockProgram::moe_gather(), 3, 2),
		("gather_backward", BlockProgram::moe_gather_backward(), 3, 4),
		("combine", BlockProgram::moe_combine(), 4, 4),
		(
			"combine_backward",
			BlockProgram::moe_combine_backward(),
			7,
			4,
		),
		("route_weights", BlockProgram::moe_route_weights(), 3, 3),
		(
			"route_weights_backward",
			BlockProgram::moe_route_weights_backward(),
			5,
			3,
		),
	] {
		assert_eq!(program.bindings.len(), bindings, "{name} bindings");
		assert_eq!(program.push_scalars.len(), scalars, "{name} scalars");
		let bounded = program.slang_source(true).expect("bounded source");
		let unbounded = program.slang_source(false).expect("unbounded source");
		assert_eq!(
			bounded,
			program.slang_source(true).unwrap(),
			"{name} bounded"
		);
		assert_eq!(
			unbounded,
			program.slang_source(false).unwrap(),
			"{name} unbounded"
		);
		assert!(bounded.contains("numthreads(256, 1, 1)"), "{name} lanes");
		assert!(bounded.contains("group_id.x * 256u"), "{name} flat index");
	}
}

#[test]
fn cross_entropy_templates_are_deterministic() {
	for (masked, bindings, scalars) in [(false, 3, 3), (true, 4, 3)] {
		let program = BlockProgram::cross_entropy(256, masked).unwrap();
		assert_eq!(program.bindings.len(), bindings);
		assert_eq!(program.push_scalars.len(), scalars);
		let source = program.slang_source(true).expect("forward source");
		assert_eq!(source, program.slang_source(true).unwrap());
		assert!(source.contains("group_id.y * 65535u"));
		assert!(source.contains("exp("));
		assert!(source.contains("log("));
		assert!(source.contains("if (lane == 0u)"));
		assert!(source.contains("asfloat(0xffc00000u)"));
		if masked {
			assert!(source.contains("return;"));
		}
	}
	for (masked, bindings, scalars) in [(false, 3, 3), (true, 4, 4)] {
		let program = BlockProgram::cross_entropy_backward(256, masked).unwrap();
		assert_eq!(program.bindings.len(), bindings);
		assert_eq!(program.push_scalars.len(), scalars);
		let source = program.slang_source(true).expect("backward source");
		assert_eq!(source, program.slang_source(true).unwrap());
		assert!(source.contains("group_id.y * 65535u"));
		assert!(source.contains("push.target_dtype == 1u"));
		assert!(source.contains("asfloat(0xffc00000u)"));
	}
}
#[test]
fn gradient_sum_templates_are_deterministic() {
	let sum = BlockProgram::layer_norm_backward_sum(256).unwrap();
	assert_eq!(sum.bindings.len(), 9);
	assert_eq!(sum.push_scalars.len(), 3);
	let source = sum.slang_source(true).expect("sum source");
	assert_eq!(source, sum.slang_source(true).unwrap());
	assert!(source.contains("push.addend_count == 3u"));
	assert!(source.contains("GroupMemoryBarrierWithGroupSync"));
	assert_eq!(
		source.matches("GroupMemoryBarrierWithGroupSync").count(),
		// fold store + first result + second store + one per stride loop
		3 + 2
	);
	let embedding = BlockProgram::embedding_backward_sum();
	assert_eq!(embedding.bindings.len(), 4);
	assert_eq!(embedding.push_scalars.len(), 3);
	let source = embedding.slang_source(true).expect("embedding source");
	assert_eq!(source, embedding.slang_source(true).unwrap());
	assert!(source.contains("push.index_count"));
	assert!(source.contains("% push.embedding_dim"));
	assert!(!source.contains("GroupMemoryBarrierWithGroupSync"));
}

#[test]
fn max_family_templates_are_deterministic() {
	let max = BlockProgram::max(256).unwrap();
	assert_eq!(max.bindings.len(), 2);
	assert_eq!(max.push_scalars.len(), 1);
	let source = max.slang_source(true).expect("max source");
	assert_eq!(source, max.slang_source(true).unwrap());
	assert!(source.contains("asfloat(0xff7fffffu)"));
	assert!(source.contains("max(shared0[lane], shared0[lane + stride1])"));
	let argmax = BlockProgram::argmax(256).unwrap();
	assert_eq!(argmax.bindings.len(), 2);
	let source = argmax.slang_source(true).expect("argmax source");
	assert_eq!(source, argmax.slang_source(true).unwrap());
	assert!(source.contains("groupshared uint shared1[256u]"));
	assert!(source.contains("shared1[lane + stride1] < shared1[lane]"));
	assert!(source.contains("Store(0u * 4u, shared1[0u])"));
	let backward = BlockProgram::max_backward();
	assert_eq!(backward.bindings.len(), 4);
	let source = backward.slang_source(true).expect("max_backward source");
	assert_eq!(source, backward.slang_source(true).unwrap());
	assert!(source.contains("asfloat(0x00000000u)"));
	assert!(!source.contains("GroupMemoryBarrierWithGroupSync"));
}

#[test]
fn max_wave_template_is_deterministic() {
	let program = BlockProgram::max_wave(256).unwrap();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 1);
	let source = program.slang_source(true).expect("wave source");
	assert_eq!(source, program.slang_source(true).unwrap());
	// The wave ops evaluate on every lane of their subgroup — only
	// the shared stage and final store are gated.
	assert_eq!(source.matches("WaveActiveMax").count(), 2);
	assert!(source.contains("WaveGetLaneCount()"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 1);
}

#[test]

fn sum_wave_template_is_deterministic() {
	let program = BlockProgram::sum_wave(256).unwrap();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 1);
	let source = program.slang_source(true).expect("wave source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert_eq!(source.matches("WaveActiveSum").count(), 2);
	assert!(source.contains("asfloat(0x00000000u)"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 1);
}

#[test]
fn split_heads_template_is_deterministic() {
	let program = BlockProgram::split_heads();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 4);
	let source = program.slang_source(true).expect("split_heads source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.num_heads"));
	assert!(source.contains("push.head_dim"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn merge_heads_template_is_deterministic() {
	let program = BlockProgram::merge_heads();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 4);
	let source = program.slang_source(true).expect("merge_heads source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.sequence_length"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn sdpa_softmax_template_is_deterministic() {
	let program = BlockProgram::sdpa_softmax(256).expect("sdpa_softmax");
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 4);
	assert_eq!(program.push_f32_scalars.len(), 1);
	let source = program.slang_source(true).expect("sdpa_softmax source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.scale"));
	assert!(source.contains("exp("));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 6);
	let narrow = BlockProgram::sdpa_softmax(32).expect("sdpa_softmax n32");
	assert_eq!(narrow.shape.lanes, 32);
}

#[test]
fn small_causal_sdpa_template_is_deterministic() {
	let program = BlockProgram::small_causal_sdpa();
	assert_eq!(program.bindings.len(), 10);
	assert_eq!(program.push_scalars.len(), 4);
	assert_eq!(program.push_f32_scalars.len(), 1);
	let source = program
		.slang_source(true)
		.expect("small_causal_sdpa source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.scale"));
	assert!(source.contains("push.num_heads"));
	assert!(source.contains("exp("));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 4);
}

#[test]
fn argmax_wave_template_is_deterministic() {
	let program = BlockProgram::argmax_wave(256).unwrap();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 1);
	let source = program.slang_source(true).expect("wave source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert_eq!(source.matches("WaveActiveMax").count(), 2);
	assert_eq!(source.matches("WaveActiveMin").count(), 2);
	assert!(source.contains("4294967295u"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 1);
}

#[test]
fn sum_axis_cooperative_template_is_deterministic() {
	let program = BlockProgram::sum_axis_cooperative(256).unwrap();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 3);
	let source = program.slang_source(true).expect("cooperative source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("bound2 = push.dim_size"));
	assert!(source.contains("storage_buffers[buf1].Store(group_id.x * 4u"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 2);
}

#[test]
fn conv_1d_im2_col_template_is_deterministic() {
	let program = BlockProgram::conv_1d_im2_col();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 8);
	let source = program.slang_source(true).expect("im2col source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.output_length"));
	assert!(source.contains("push.padding"));
	assert!(source.contains("push.dilation"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_1d_bias_add_template_is_deterministic() {
	let program = BlockProgram::conv_1d_bias_add();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 2);
	let source = program.slang_source(true).expect("bias add source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.columns"));
	assert!(source.contains("push.buffer0"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_1d_layout_transpose_template_is_deterministic() {
	let program = BlockProgram::conv_1d_layout_transpose();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 3);
	assert_eq!(program.shape.lanes, 1024);
	let source = program.slang_source(true).expect("transpose source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.columns"));
	assert!(source.contains("groupshared float shared0[32u * 33u];"));
	assert!(source.contains("group_id.z"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 1);
}

#[test]
fn conv_2d_template_is_deterministic() {
	let program = BlockProgram::conv_2d();
	assert_eq!(program.bindings.len(), 4);
	assert_eq!(program.push_scalars.len(), 11);
	let source = program.slang_source(true).expect("conv_2d source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.groups"));
	assert!(source.contains("loop2"));
	assert!(source.contains("loop3"));
	assert!(source.contains("loop5"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_2d_relu_template_is_deterministic() {
	let program = BlockProgram::conv_2d_relu();
	assert_eq!(program.bindings.len(), 4);
	assert_eq!(program.push_scalars.len(), 11);
	let source = program.slang_source(true).expect("conv_2d_relu source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("max(asfloat(0x00000000u), tile0)"));
}

#[test]
fn conv_1d_parameter_backward_template_is_deterministic() {
	let program = BlockProgram::conv_1d_parameter_backward();
	assert_eq!(program.bindings.len(), 4);
	assert_eq!(program.push_scalars.len(), 10);
	let source = program.slang_source(true).expect("parameter source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.parameter_count"));
	assert!(source.contains("push.dilation"));
	assert!(source.contains("loop3"));
	assert!(source.contains("loop4"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_1d_backward_template_is_deterministic() {
	let program = BlockProgram::conv_1d_backward();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 9);
	let source = program.slang_source(true).expect("backward source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.stride"));
	assert!(source.contains("loop2"));
	assert!(source.contains("loop3"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_2d_backward_template_is_deterministic() {
	let program = BlockProgram::conv_2d_backward();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 11);
	let source = program.slang_source(true).expect("conv_2d_backward source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.groups"));
	assert!(source.contains("loop2"));
	assert!(source.contains("loop5"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_2d_parameter_backward_template_is_deterministic() {
	let program = BlockProgram::conv_2d_parameter_backward();
	assert_eq!(program.bindings.len(), 4);
	assert_eq!(program.push_scalars.len(), 12);
	let source = program
		.slang_source(true)
		.expect("conv_2d_parameter_backward source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.parameter_count"));
	assert!(source.contains("loop3"));
	assert!(source.contains("loop5"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_2d_transpose_template_is_deterministic() {
	let program = BlockProgram::conv_2d_transpose();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 10);
	let source = program
		.slang_source(true)
		.expect("conv_2d_transpose source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.output_width"));
	assert!(source.contains("loop2"));
	assert!(source.contains("loop5"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_2d_transpose_bias_add_template_is_deterministic() {
	let program = BlockProgram::conv_2d_transpose_bias_add();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 4);
	let source = program
		.slang_source(true)
		.expect("conv_2d_transpose_bias_add source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.channels"));
	assert!(source.contains("push.height"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_2d_transpose_backward_template_is_deterministic() {
	let program = BlockProgram::conv_2d_transpose_backward();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 10);
	let source = program
		.slang_source(true)
		.expect("conv_2d_transpose_backward source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.output_height"));
	assert!(source.contains("loop2"));
	assert!(source.contains("loop5"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_2d_transpose_parameter_backward_template_is_deterministic() {
	let program = BlockProgram::conv_2d_transpose_parameter_backward();
	assert_eq!(program.bindings.len(), 4);
	assert_eq!(program.push_scalars.len(), 11);
	let source = program
		.slang_source(true)
		.expect("conv_2d_transpose_parameter_backward source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.parameter_count"));
	assert!(source.contains("loop3"));
	assert!(source.contains("loop5"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_1d_bias_layout_transpose_template_is_deterministic() {
	let program = BlockProgram::conv_1d_bias_layout_transpose();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 3);
	let source = program
		.slang_source(true)
		.expect("conv_1d_bias_layout_transpose source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.columns"));
	assert!(source.contains("push.rows"));
	assert!(source.contains("groupshared"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 1);
}

#[test]
fn conv_1d_bias_layout_transpose_relu_template_is_deterministic() {
	let program = BlockProgram::conv_1d_bias_layout_transpose_relu();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 3);
	let source = program
		.slang_source(true)
		.expect("conv_1d_bias_layout_transpose_relu source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("max("));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 1);
}

#[test]
fn conv_1d_bias_layout_transpose_residual_relu_template_is_deterministic() {
	let program = BlockProgram::conv_1d_bias_layout_transpose_residual_relu();
	assert_eq!(program.bindings.len(), 5);
	assert_eq!(program.push_scalars.len(), 3);
	let source = program
		.slang_source(true)
		.expect("conv_1d_bias_layout_transpose_residual_relu source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("max("));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 1);
}

#[test]
fn conv_1d_bias_copy_template_is_deterministic() {
	let program = BlockProgram::conv_1d_bias_copy();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 2);
	let source = program
		.slang_source(true)
		.expect("conv_1d_bias_copy source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.columns"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_1d_bias_layout_transpose_norm_template_is_deterministic() {
	let program = BlockProgram::conv_1d_bias_layout_transpose_norm(false);
	assert_eq!(program.bindings.len(), 6);
	assert_eq!(program.push_scalars.len(), 3);
	assert_eq!(program.push_f32_scalars.len(), 1);
	let source = program
		.slang_source(true)
		.expect("conv_1d_bias_layout_transpose_norm source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.epsilon"));
	assert!(source.contains("rsqrt"));
	assert!(source.matches("GroupMemoryBarrierWithGroupSync").count() >= 3);
	assert!(!source.contains("max("));
}

#[test]
fn conv_1d_bias_layout_transpose_norm_relu_template_is_deterministic() {
	let program = BlockProgram::conv_1d_bias_layout_transpose_norm(true);
	assert_eq!(program.bindings.len(), 6);
	assert_eq!(program.push_scalars.len(), 3);
	assert_eq!(program.push_f32_scalars.len(), 1);
	let source = program
		.slang_source(true)
		.expect("conv_1d_bias_layout_transpose_norm_relu source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("max("));
	assert!(source.contains("rsqrt"));
	assert!(source.matches("GroupMemoryBarrierWithGroupSync").count() >= 3);
}

#[test]
fn avg_pool_2d_template_is_deterministic() {
	let program = BlockProgram::avg_pool_2d();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 9);
	let source = program.slang_source(true).expect("avg_pool_2d source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.padding"));
	assert!(source.contains("group_id.z"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn max_pool_2d_template_is_deterministic() {
	let program = BlockProgram::max_pool_2d();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 9);
	let source = program.slang_source(true).expect("max_pool_2d source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.kernel_size"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn adaptive_avg_pool_2d_template_is_deterministic() {
	let program = BlockProgram::adaptive_avg_pool_2d();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 6);
	let source = program
		.slang_source(true)
		.expect("adaptive_avg_pool_2d source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.output_height"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn avg_pool_2d_backward_template_is_deterministic() {
	let program = BlockProgram::avg_pool_2d_backward();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 9);
	let source = program
		.slang_source(true)
		.expect("avg_pool_2d_backward source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.stride"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn max_pool_2d_backward_template_is_deterministic() {
	let program = BlockProgram::max_pool_2d_backward();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 9);
	let source = program
		.slang_source(true)
		.expect("max_pool_2d_backward source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.output_width"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn max_pool_2d_relu_backward_template_is_deterministic() {
	let program = BlockProgram::max_pool_2d_relu_backward();
	assert_eq!(program.bindings.len(), 4);
	assert_eq!(program.push_scalars.len(), 9);
	let source = program
		.slang_source(true)
		.expect("max_pool_2d_relu_backward source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.kernel_size"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn adaptive_avg_pool_2d_backward_template_is_deterministic() {
	let program = BlockProgram::adaptive_avg_pool_2d_backward();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 6);
	let source = program
		.slang_source(true)
		.expect("adaptive_avg_pool_2d_backward source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.input_width"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn conv_1d_transpose_template_is_deterministic() {
	let program = BlockProgram::conv_1d_transpose();
	assert_eq!(program.bindings.len(), 3);
	assert_eq!(program.push_scalars.len(), 9);
	let source = program
		.slang_source(true)
		.expect("conv_1d_transpose source");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.output_channels"));
	assert!(source.contains("push.dilation"));
	assert!(source.contains("loop2"));
	assert!(source.contains("loop3"));
	assert_eq!(source.matches("GroupMemoryBarrierWithGroupSync").count(), 0);
}

#[test]
fn flash_attention_templates_are_deterministic() {
	for program in [
		BlockProgram::flash_causal(),
		BlockProgram::flash_causal_backward_q(),
		BlockProgram::flash_causal_backward_kv(),
	] {
		let source = program.slang_source(true).expect("flash source");
		assert!(source.contains("numthreads(64, 1, 1)"));
		assert!(source.contains("group_id.x"));
		assert!(!source.contains("subgroup"));
		assert_eq!(source, program.slang_source(true).unwrap());
	}
}

#[test]
fn linear_backward_templates_are_deterministic() {
	// Member adjoint: NT GEMM of the staged upstream gradient against the
	// transposed weight operand.
	let member = BlockProgram::linear_backward();
	assert_eq!(member.bindings.len(), 3);
	assert_eq!(member.push_scalars.len(), 3);
	assert_eq!(member.workgroups(70, 40), [3, 2, 1]);
	let member_source = member.slang_source(true).expect("member emission");
	assert_eq!(member_source, member.slang_source(true).unwrap());
	assert!(member_source.contains("numthreads(64, 1, 1)"));
	assert!(member_source.contains("global_column * push.input_features + global_row"));
	// Prologue region: prologue-value helper feeds the staged tile and the
	// group_id.y == 0 column republishes the combined dY.
	let prologue = BlockProgram::linear_backward_prologue();
	assert_eq!(prologue.bindings.len(), 5);
	assert_eq!(prologue.push_scalars.len(), 4);
	assert_eq!(prologue.push_f32_scalars.len(), 1);
	let prologue_source = prologue.slang_source(true).expect("prologue emission");
	assert_eq!(prologue_source, prologue.slang_source(true).unwrap());
	assert!(prologue_source.contains("tile_prologue_value"));
	assert!(prologue_source.contains("group_id.y == 0u"));
	// Norm region: per-row statistic pre-pass, gated dx/weight-contribution
	// republish, then the shared GEMM schedule.
	let norm = BlockProgram::linear_backward_norm();
	assert_eq!(norm.bindings.len(), 8);
	assert_eq!(norm.push_scalars.len(), 3);
	let norm_source = norm.slang_source(true).expect("norm emission");
	assert_eq!(norm_source, norm.slang_source(true).unwrap());
	assert!(norm_source.contains("upstream0 * gamma0"));
	assert!(norm_source.contains("upstream0 * normalized0"));
	assert!(norm_source.contains("group_id.y == 0u"));
	// Norm+add region: the addend folds into the staged dY and a third
	// republished output carries the combined gradient.
	let norm_add = BlockProgram::linear_backward_norm_add();
	assert_eq!(norm_add.bindings.len(), 10);
	assert_eq!(norm_add.push_scalars.len(), 3);
	let norm_add_source = norm_add.slang_source(true).expect("norm-add emission");
	assert_eq!(norm_add_source, norm_add.slang_source(true).unwrap());
	assert!(norm_add_source.contains("float dyv0 = adjoint0 +"));
	assert_eq!(norm_add_source.matches("group_id.y == 0u").count(), 1);
	// Bmm-chain region: 256 lanes stage the producer bmm band and the
	// transposed weight, republishing `stage1` before the second GEMM.
	let bmm = BlockProgram::linear_backward_bmm_chain();
	assert_eq!(bmm.bindings.len(), 5);
	assert_eq!(bmm.push_scalars.len(), 6);
	let bmm_source = bmm.slang_source(true).expect("bmm-chain emission");
	assert_eq!(bmm_source, bmm.slang_source(true).unwrap());
	assert!(bmm_source.contains("numthreads(256, 1, 1)"));
	assert!(bmm_source.contains("push.kind"));
	assert!(bmm_source.contains("push.input_features"));
	assert_eq!(
		bmm_source
			.matches("GroupMemoryBarrierWithGroupSync")
			.count(),
		3
	);
}

#[test]
fn scatter_backward_templates_are_deterministic() {
	// Embedding scatter adjoint: one invocation per table scalar folds
	// the position loop; the u8 variant byte-extracts the index read.
	for (program, byte_indexed) in [
		(BlockProgram::embedding_backward(), false),
		(BlockProgram::embedding_backward_u8(), true),
	] {
		assert_eq!(program.bindings.len(), 3);
		assert_eq!(program.push_scalars.len(), 3);
		let source = program.slang_source(true).expect("emission");
		assert_eq!(source, program.slang_source(true).unwrap());
		assert_eq!(
			source.contains("& 255u"),
			byte_indexed,
			"packed={byte_indexed}"
		);
	}
	// Sum broadcast adjoint: the flat gather copy.
	let program = BlockProgram::sum_backward();
	assert_eq!(program.bindings.len(), 2);
	assert_eq!(program.push_scalars.len(), 3);
	let source = program.slang_source(true).expect("emission");
	assert_eq!(source, program.slang_source(true).unwrap());
	assert!(source.contains("push.outer_size"));
	assert!(source.contains("push.inner_size"));
}
