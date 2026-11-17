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
