//! Complement, bit-length and exact-product surfaces over public limb rows.
use oa::{Matrix, Result, cryptography::bigint};

type Unary = fn(&Matrix) -> Result<Matrix>;
const UNARY: [(&str, Unary); 3] = [
	("square_wide", bigint::square_wide),
	("bit_not", bigint::bit_not),
	("leading_zeros", bigint::leading_zeros),
];

fn fixture() -> serde_json::Value {
	serde_json::from_str(include_str!(
		"../../fixtures/cryptography/bigint/breadth.json"
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

test_vk!(bigint_breadth_matches_integer_oracles_and_replay, engine, {
	for case in fixture()["cases"].as_array().unwrap() {
		let width = usize::try_from(case["words"].as_u64().unwrap()).unwrap();
		let rows = usize::try_from(case["rows"].as_u64().unwrap()).unwrap();
		let left_words = words(&case["left"]);
		let right_words = words(&case["right"]);
		let left = Matrix::new_from_slice(&engine, [rows, width], &left_words)?;
		let right = Matrix::new_from_slice(&engine, [rows, width], &right_words)?;
		let (plan, outputs) = engine.capture(|| {
			let mut outputs = vec![(
				bigint::multiply_high(&left, &right)?,
				words(&case["results"]["multiply_high"]),
			)];
			for (name, op) in UNARY {
				outputs.push((op(&left)?, words(&case["results"][name])));
			}
			Ok(outputs)
		})?;
		assert_eq!(plan.semantic_graph().operations().len(), 4);
		assert_eq!(outputs[0].0.shape(), [rows, width]);
		assert_eq!(outputs[1].0.shape(), [rows, width * 2]);
		assert_eq!(outputs[2].0.shape(), [rows, width]);
		assert_eq!(outputs[3].0.shape(), [rows]);
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

test_vk!(
	bigint_breadth_chain_rebinds_without_host_intermediates,
	engine,
	{
		let values = (0_u32..33).collect::<Vec<_>>();
		let replacement_values = values.iter().map(|v| !v).collect::<Vec<_>>();
		let input = Matrix::new_from_slice(&engine, [33, 1], &values)?;
		let (mut plan, (restored, zeros)) = engine.capture(|| {
			let complement = bigint::bit_not(&input)?;
			let restored = bigint::bit_not(&complement)?;
			let zeros = bigint::leading_zeros(&restored)?;
			Ok((restored, zeros))
		})?;
		assert_eq!(plan.diagnostics().node_count(), 3);
		assert_eq!(plan.diagnostics().barrier_count(), 2);
		engine.submit(&plan)?.wait()?;
		assert_eq!(restored.read::<u32>()?, values);
		assert_eq!(
			zeros.read::<u32>()?,
			values.iter().map(|v| v.leading_zeros()).collect::<Vec<_>>()
		);
		let replacement = Matrix::new_from_slice(&engine, [33, 1], &replacement_values)?;
		plan.bind_matrix_input(&input, &replacement)?;
		for _ in 0..2 {
			engine.submit(&plan)?.wait()?;
			assert_eq!(restored.read::<u32>()?, replacement_values);
			assert_eq!(
				zeros.read::<u32>()?,
				replacement_values
					.iter()
					.map(|v| v.leading_zeros())
					.collect::<Vec<_>>()
			);
		}
		Ok(())
	}
);

test_vk!(
	bigint_breadth_rejects_invalid_metadata_and_accepts_empty,
	engine,
	{
		let empty = Matrix::new_from_slice(&engine, [0, 2], &[] as &[u32])?;
		assert!(
			bigint::multiply_high(&empty, &empty)?
				.try_read::<u32>()?
				.is_empty()
		);
		for (_, op) in UNARY {
			assert!(op(&empty)?.try_read::<u32>()?.is_empty());
		}
		let valid = Matrix::new_from_slice(&engine, [1, 2], &[1_u32, 2])?;
		let wrong_dtype = Matrix::new_from_f32(&engine, [1, 2], &[1.0, 2.0])?;
		let rank_one = Matrix::new_from_slice(&engine, [2], &[1_u32, 2])?;
		let zero_width = Matrix::new_from_slice(&engine, [1, 0], &[] as &[u32])?;
		let too_wide = Matrix::new_from_slice(&engine, [1, 129], &[0_u32; 129])?;
		for input in [&wrong_dtype, &rank_one, &zero_width, &too_wide] {
			for (_, op) in UNARY {
				assert_eq!(
					op(input).err().unwrap().kind(),
					oa::ErrorKind::InvalidArgument
				);
			}
			assert_eq!(
				bigint::multiply_high(&valid, input).err().unwrap().kind(),
				oa::ErrorKind::InvalidArgument
			);
			assert_eq!(
				bigint::multiply_high(input, &valid).err().unwrap().kind(),
				oa::ErrorKind::InvalidArgument
			);
		}
		let wrong_shape = Matrix::new_from_slice(&engine, [2, 1], &[1_u32, 2])?;
		assert_eq!(
			bigint::multiply_high(&valid, &wrong_shape)
				.err()
				.unwrap()
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);
