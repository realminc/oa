use super::*;

fn chain(length: usize) -> Program {
	Program {
		element_count: 17,
		input_bytes: vec![68],
		instructions: (0..length)
			.map(|index| Instruction {
				opcode: 12,
				sources: vec![
					if index == 0 {
						Source::Input(0)
					} else {
						Source::Value(index - 1)
					},
					Source::Constant(1.25f32.to_bits()),
				],
				parameters: vec![],
			})
			.collect(),
		outputs: vec![length - 1],
	}
}

#[test]
fn exact_key_preserves_constants_and_materialized_values() {
	let program = chain(3);
	assert!(program.validate());
	assert_eq!(program.canonical_key(), program.clone().canonical_key());
	let mut different = program.clone();
	different.instructions[0].sources[1] = Source::Constant((-0.0f32).to_bits());
	let negative_zero = different.canonical_key();
	different.instructions[0].sources[1] = Source::Constant(0.0f32.to_bits());
	assert_ne!(negative_zero, different.canonical_key());
	different = program.clone();
	different.outputs.insert(0, 0);
	assert_ne!(program.canonical_key(), different.canonical_key());
	assert_eq!(
		different
			.tape(&[0], &[16, 17])
			.unwrap()
			.as_chunks::<4>()
			.0
			.iter()
			.filter(|quad| quad[0] == 2)
			.count(),
		2
	);
}

#[test]
fn artifact_key_shares_code_across_extents() {
	let program = chain(3);
	let artifact = program.artifact_key();
	let signature = program.canonical_key();
	// Element count and input sizes are dispatch facts; the emitted Slang
	// text is identical, so the artifact key is shared while the full
	// fusion signature still distinguishes them.
	let mut resized = program.clone();
	resized.element_count = 1024;
	resized.input_bytes[0] = 4096;
	assert_eq!(artifact, resized.artifact_key());
	assert_ne!(signature, resized.canonical_key());
	let mut different = program.clone();
	different.instructions[0].sources[1] = Source::Constant((-0.0f32).to_bits());
	assert_ne!(artifact, different.artifact_key());
	different = program.clone();
	different.outputs.insert(0, 0);
	assert_ne!(artifact, different.artifact_key());
	different = program.clone();
	different.input_bytes.push(4);
	assert_ne!(artifact, different.artifact_key());
}

#[test]
fn semantic_legality_is_independent_of_tape_capacity() {
	let program = chain(70);
	assert!(program.validate());
	assert!(program.tape(&[0], &[16]).is_none());
	assert!(program.slang_source(true).unwrap().contains("float v69"));
	let small = chain(3);
	assert_eq!(small.tape(&[0], &[16]).unwrap().len() / 4, 7);
}

#[test]
fn invalid_ssa_extents_and_broadcasts_fail_closed() {
	let mut program = chain(2);
	program.instructions[0].sources[0] = Source::Value(0);
	assert!(!program.validate());
	assert!(program.slang_source(true).is_err());
	program = chain(2);
	program.input_bytes[0] = 67;
	assert!(!program.validate());
	program = chain(2);
	program.element_count = 0;
	assert!(!program.validate());
	program = chain(2);
	program.instructions[0].sources[0] = Source::Broadcast {
		input: 0,
		rank: 1,
		dimensions: [17, 0, 0, 0, 0, 0, 0, 0],
		strides: [0; 8],
	};
	program.input_bytes[0] = 4;
	assert!(program.validate());
	assert!(program.tape(&[0], &[16]).is_some());
	if let Source::Broadcast { dimensions, .. } = &mut program.instructions[0].sources[0] {
		dimensions[0] = 16;
	}
	assert!(!program.validate());
}

#[test]
fn packed_gather_requires_complete_load_words() {
	let mut program = Program {
		element_count: 3,
		input_bytes: vec![16, 3],
		instructions: vec![Instruction {
			opcode: 4,
			sources: vec![Source::Input(0), Source::Input(1)],
			parameters: vec![4, 1, 0],
		}],
		outputs: vec![0],
	};
	assert!(!program.validate());
	program.input_bytes[1] = 4;
	assert!(program.validate());
	program.instructions[0].parameters[0] = u32::MAX;
	assert!(!program.validate());
}

#[test]
fn live_values_exceeding_register_budget_keep_valid_specialization() {
	let mut program = chain(9);
	for op in &mut program.instructions {
		op.sources[0] = Source::Input(0);
	}
	for index in 0..9 {
		program.instructions.push(Instruction {
			opcode: 10,
			sources: vec![Source::Value(index), Source::Constant(0)],
			parameters: vec![],
		});
	}
	program.outputs = vec![17];
	assert!(program.validate());
	assert!(program.tape(&[0], &[16]).is_none());
	assert!(program.slang_source(false).is_ok());
}
