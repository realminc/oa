//! Modular/Montgomery/Barrett Engine routes against Python-integer fixtures.
use oa::{Matrix, Result, cryptography::bigint};

type Unary = fn(&Matrix, &Matrix) -> Result<(Matrix, Matrix)>;
type Binary = fn(&Matrix, &Matrix, &Matrix) -> Result<(Matrix, Matrix)>;
const UNARY: [(&str, Unary); 5] = [
	("reduce_mod", bigint::reduce_mod),
	("square_mod", bigint::square_mod),
	("inverse_mod", bigint::inverse_mod),
	("to_montgomery", bigint::to_montgomery),
	("from_montgomery", bigint::from_montgomery),
];
const BINARY: [(&str, Binary); 6] = [
	("add_mod", bigint::add_mod),
	("subtract_mod", bigint::subtract_mod),
	("multiply_mod", bigint::multiply_mod),
	("pow_mod", bigint::pow_mod),
	("montgomery_multiply", bigint::montgomery_multiply),
	("barrett_multiply", bigint::barrett_multiply),
];

fn fixture() -> serde_json::Value {
	serde_json::from_str(include_str!(
		"../../fixtures/cryptography/bigint/modular.json"
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

test_vk!(bigint_modular_oracles_and_multi_output_replay, engine, {
	for case in fixture()["cases"].as_array().unwrap() {
		let width = usize::try_from(case["words"].as_u64().unwrap()).unwrap();
		let rows = usize::try_from(case["rows"].as_u64().unwrap()).unwrap();
		let left = Matrix::new_from_slice(&engine, [rows, width], &words(&case["left"]))?;
		let right = Matrix::new_from_slice(&engine, [rows, width], &words(&case["right"]))?;
		let modulus = Matrix::new_from_slice(&engine, [rows, width], &words(&case["modulus"]))?;
		let (plan, outputs) = engine.capture(|| {
			let mut outputs = Vec::new();
			for (name, op) in UNARY {
				outputs.push((name, op(&left, &modulus)?));
			}
			for (name, op) in BINARY {
				outputs.push((name, op(&left, &right, &modulus)?));
			}
			Ok(outputs)
		})?;
		assert_eq!(plan.semantic_graph().operations().len(), 11);
		for (_, (output, status)) in &outputs {
			assert_eq!(output.shape(), [rows, width]);
			assert_eq!(status.shape(), [rows]);
			assert_eq!(
				output.try_read::<u32>().unwrap_err().kind(),
				oa::ErrorKind::NotReady
			);
			assert_eq!(
				status.try_read::<u32>().unwrap_err().kind(),
				oa::ErrorKind::NotReady
			);
		}
		for _ in 0..2 {
			engine.submit(&plan)?.wait()?;
			for (name, (output, status)) in &outputs {
				assert_eq!(
					output.read::<u32>()?,
					words(&case["results"][name]["output"]),
					"{name}, words={width}"
				);
				assert_eq!(
					status.read::<u32>()?,
					words(&case["results"][name]["status"]),
					"{name}, words={width}"
				);
			}
		}
		assert_eq!(left.read::<u32>()?, words(&case["left"]));
		assert_eq!(right.read::<u32>()?, words(&case["right"]));
		assert_eq!(modulus.read::<u32>()?, words(&case["modulus"]));
	}
	Ok(())
});

test_vk!(
	bigint_shared_modulus_chain_rebinds_without_host_intermediates,
	engine,
	{
		let values = (1_u32..=33).collect::<Vec<_>>();
		let left = Matrix::new_from_slice(&engine, [33, 1], &values)?;
		let right = Matrix::new_from_slice(&engine, [33, 1], &[7_u32; 33])?;
		let modulus = Matrix::new_from_slice(&engine, [1, 1], &[97_u32])?;
		let (mut plan, (output, status)) = engine.capture(|| {
			let (a, _) = bigint::to_montgomery(&left, &modulus)?;
			let (b, _) = bigint::to_montgomery(&right, &modulus)?;
			let (product, _) = bigint::montgomery_multiply(&a, &b, &modulus)?;
			bigint::from_montgomery(&product, &modulus)
		})?;
		assert_eq!(plan.semantic_graph().operations().len(), 4);
		for prime in [97_u32, 101] {
			if prime != 97 {
				let replacement = Matrix::new_from_slice(&engine, [1, 1], &[prime])?;
				plan.bind_matrix_input(&modulus, &replacement)?;
			}
			engine.submit(&plan)?.wait()?;
			assert_eq!(
				output.read::<u32>()?,
				values.iter().map(|v| v * 7 % prime).collect::<Vec<_>>()
			);
			assert_eq!(status.read::<u32>()?, vec![1; 33]);
		}
		let even = Matrix::new_from_slice(&engine, [1, 1], &[96_u32])?;
		plan.bind_matrix_input(&modulus, &even)?;
		engine.submit(&plan)?.wait()?;
		assert_eq!(output.read::<u32>()?, vec![0; 33]);
		assert_eq!(status.read::<u32>()?, vec![0; 33]);
		Ok(())
	}
);

test_vk!(bigint_modular_metadata_and_empty_batches, engine, {
	let left = Matrix::new_from_slice(&engine, [2, 2], &[1_u32; 4])?;
	let valid = Matrix::new_from_slice(&engine, [1, 2], &[97_u32, 0])?;
	let wrong_dtype = Matrix::new_from_f32(&engine, [1, 2], &[97.0, 0.0])?;
	let rank_one = Matrix::new_from_slice(&engine, [2], &[97_u32, 0])?;
	let wrong_width = Matrix::new_from_slice(&engine, [1, 1], &[97_u32])?;
	let wrong_rows = Matrix::new_from_slice(&engine, [3, 2], &[97_u32; 6])?;
	for modulus in [&wrong_dtype, &rank_one, &wrong_width, &wrong_rows] {
		for (_, op) in UNARY {
			assert_eq!(
				op(&left, modulus).err().unwrap().kind(),
				oa::ErrorKind::InvalidArgument
			);
		}
		for (_, op) in BINARY {
			assert_eq!(
				op(&left, &left, modulus).err().unwrap().kind(),
				oa::ErrorKind::InvalidArgument
			);
		}
	}
	let empty = Matrix::new_from_slice(&engine, [0, 2], &[] as &[u32])?;
	for (_, op) in UNARY {
		let (output, status) = op(&empty, &valid)?;
		assert!(output.try_read::<u32>()?.is_empty());
		assert!(status.try_read::<u32>()?.is_empty());
	}
	for (_, op) in BINARY {
		let (output, status) = op(&empty, &empty, &valid)?;
		assert!(output.try_read::<u32>()?.is_empty());
		assert!(status.try_read::<u32>()?.is_empty());
	}
	Ok(())
});
