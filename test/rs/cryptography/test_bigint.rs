//! Public-data unsigned device arithmetic against Python-integer fixtures.
use oa::{Matrix, Result, cryptography::bigint};

type Binary = fn(&Matrix, &Matrix) -> Result<Matrix>;
const BINARY: [(&str, Binary); 8] = [
	("add", bigint::add),
	("subtract", bigint::subtract),
	("multiply_low", bigint::multiply_low),
	("multiply_wide", bigint::multiply_wide),
	("bit_and", bigint::bit_and),
	("bit_or", bigint::bit_or),
	("bit_xor", bigint::bit_xor),
	("compare", bigint::compare),
];

fn fixture() -> serde_json::Value {
	serde_json::from_str(include_str!(
		"../../fixtures/cryptography/bigint/arithmetic.json"
	))
	.unwrap()
}

fn words(value: &serde_json::Value) -> Vec<u32> {
	value
		.as_array()
		.unwrap()
		.iter()
		.map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
		.collect()
}

test_vk!(bigint_batch_matches_integer_oracles_and_replay, engine, {
	for case in fixture()["cases"].as_array().unwrap() {
		let width = usize::try_from(case["words"].as_u64().unwrap()).unwrap();
		let rows = usize::try_from(case["rows"].as_u64().unwrap()).unwrap();
		let left_words = words(&case["left"]);
		let right_words = words(&case["right"]);
		let left = Matrix::new_from_slice(&engine, [rows, width], &left_words)?;
		let right = Matrix::new_from_slice(&engine, [rows, width], &right_words)?;
		let (plan, outputs) = engine.capture(|| {
			let mut outputs = Vec::new();
			for (name, op) in BINARY {
				outputs.push((op(&left, &right)?, words(&case["results"][name])));
			}
			let (quotient, remainder, status) = bigint::div_rem(&left, &right)?;
			outputs.push((quotient, words(&case["results"]["quotient"])));
			outputs.push((remainder, words(&case["results"]["remainder"])));
			outputs.push((status, words(&case["results"]["status"])));
			for shift in case["shifts"].as_array().unwrap() {
				let amount = u32::try_from(shift["shift"].as_u64().unwrap()).unwrap();
				outputs.push((bigint::shift_left(&left, amount)?, words(&shift["left"])));
				outputs.push((bigint::shift_right(&left, amount)?, words(&shift["right"])));
			}
			Ok(outputs)
		})?;
		assert_eq!(plan.semantic_graph().operations().len(), 27);
		assert_eq!(outputs[3].0.shape(), [rows, width * 2]);
		assert_eq!(outputs[7].0.shape(), [rows]);
		assert_eq!(outputs[10].0.shape(), [rows]);
		for (output, _) in &outputs {
			assert_eq!(output.dtype(), oa::DType::U32);
			assert_eq!(
				output.try_read::<u32>().unwrap_err().kind(),
				oa::ErrorKind::NotReady
			);
		}
		for _ in 0..2 {
			engine.submit(&plan)?.wait()?;
			for (index, (output, expected)) in outputs.iter().enumerate() {
				assert_eq!(
					output.read::<u32>()?,
					*expected,
					"words={width}, output={index}"
				);
			}
		}
		assert_eq!(left.read::<u32>()?, left_words);
		assert_eq!(right.read::<u32>()?, right_words);
	}
	Ok(())
});

test_vk!(bigint_chain_rebinds_and_crosses_workgroup_tail, engine, {
	let first = (0_u32..33)
		.map(|i| i.wrapping_mul(0xFEDCBA98))
		.collect::<Vec<_>>();
	let second = first.iter().map(|v| !v).collect::<Vec<_>>();
	let left = Matrix::new_from_slice(&engine, [33, 1], &first)?;
	let right = Matrix::new_from_slice(&engine, [33, 1], &[u32::MAX; 33])?;
	let (mut plan, output) = engine.capture(|| {
		let sum = bigint::add(&left, &right)?;
		bigint::subtract(&sum, &right)
	})?;
	assert_eq!(plan.diagnostics().node_count(), 2);
	assert_eq!(plan.diagnostics().barrier_count(), 1);
	assert_eq!(plan.semantic_graph().operations().len(), 2);
	engine.submit(&plan)?.wait()?;
	assert_eq!(output.read::<u32>()?, first);
	let replacement = Matrix::new_from_slice(&engine, [33, 1], &second)?;
	plan.bind_matrix_input(&left, &replacement)?;
	for _ in 0..2 {
		engine.submit(&plan)?.wait()?;
		assert_eq!(output.read::<u32>()?, second);
	}
	Ok(())
});

test_vk!(bigint_contracts_and_ready_empty_batches, engine, {
	let empty = Matrix::new_from_slice(&engine, [0, 2], &[] as &[u32])?;
	for (_, op) in BINARY {
		let output = op(&empty, &empty)?;
		assert!(output.try_read::<u32>()?.is_empty());
	}
	let (q, r, status) = bigint::div_rem(&empty, &empty)?;
	for output in [
		q,
		r,
		status,
		bigint::shift_left(&empty, u32::MAX)?,
		bigint::shift_right(&empty, 0)?,
	] {
		assert!(output.try_read::<u32>()?.is_empty());
	}
	let valid = Matrix::new_from_slice(&engine, [1, 2], &[1_u32, 2])?;
	let wrong_dtype = Matrix::new_from_f32(&engine, [1, 2], &[1.0, 2.0])?;
	let rank_one = Matrix::new_from_slice(&engine, [2], &[1_u32, 2])?;
	let wrong_shape = Matrix::new_from_slice(&engine, [2, 1], &[1_u32, 2])?;
	let zero_width = Matrix::new_from_slice(&engine, [1, 0], &[] as &[u32])?;
	let too_wide = Matrix::new_from_slice(&engine, [1, 129], &[0_u32; 129])?;
	for input in [
		&wrong_dtype,
		&rank_one,
		&wrong_shape,
		&zero_width,
		&too_wide,
	] {
		for (_, op) in BINARY {
			assert_eq!(
				op(&valid, input).err().unwrap().kind(),
				oa::ErrorKind::InvalidArgument
			);
		}
		assert_eq!(
			bigint::div_rem(&valid, input).err().unwrap().kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	for input in [&wrong_dtype, &rank_one, &zero_width, &too_wide] {
		assert_eq!(
			bigint::shift_left(input, 1).err().unwrap().kind(),
			oa::ErrorKind::InvalidArgument
		);
		assert_eq!(
			bigint::shift_right(input, 1).err().unwrap().kind(),
			oa::ErrorKind::InvalidArgument
		);
	}
	Ok(())
});
