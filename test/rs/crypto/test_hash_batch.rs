//! Fixed-digest device adapters against independent hashlib fixtures.
use oa::{Matrix, Result, crypto::hash};

type BatchHash = fn(&Matrix) -> Result<Matrix>;
const HASHES: [(&str, BatchHash, usize); 10] = [
	("sha224", hash::sha224, 28),
	("sha256", hash::sha256, 32),
	("sha384", hash::sha384, 48),
	("sha512", hash::sha512, 64),
	("sha512_224", hash::sha512_224, 28),
	("sha512_256", hash::sha512_256, 32),
	("sha3_224", hash::sha3_224, 28),
	("sha3_256", hash::sha3_256, 32),
	("sha3_384", hash::sha3_384, 48),
	("sha3_512", hash::sha3_512, 64),
];

fn fixture() -> serde_json::Value {
	serde_json::from_str(include_str!("../../fixtures/crypto/hash/batch.json")).unwrap()
}

fn unhex(value: &str) -> Vec<u8> {
	value
		.as_bytes()
		.as_chunks::<2>()
		.0
		.iter()
		.map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
		.collect()
}

fn flatten(value: &serde_json::Value) -> Vec<u8> {
	value
		.as_array()
		.unwrap()
		.iter()
		.flat_map(|row| unhex(row.as_str().unwrap()))
		.collect()
}

test_vk!(
	fixed_digest_batch_boundaries_match_independent_oracle,
	engine,
	{
		for case in fixture()["cases"].as_array().unwrap() {
			let length = usize::try_from(case["length"].as_u64().unwrap()).unwrap();
			let bytes = flatten(&case["messages"]);
			let input = Matrix::new_from_slice(&engine, [3, length], &bytes)?;
			for (name, op, digest_bytes) in HASHES {
				let output = op(&input)?;
				assert_eq!(output.dtype(), oa::DType::U8);
				assert_eq!(output.shape(), [3, digest_bytes]);
				assert_eq!(
					output.read::<u8>()?,
					flatten(&case["digests"][name]),
					"{name}: length {length}"
				);
			}
		}
		Ok(())
	}
);

test_vk!(
	fixed_digest_batch_contracts_reject_invalid_inputs,
	engine,
	{
		let wrong_dtype = Matrix::new_from_f32(&engine, [1, 3], &[0.0; 3])?;
		let rank_one = Matrix::new_from_slice(&engine, [3], b"abc")?;
		let empty_batch = Matrix::new_from_slice(&engine, [0, 3], &[] as &[u8])?;
		for (_, op, _) in HASHES {
			for input in [&wrong_dtype, &rank_one, &empty_batch] {
				assert_eq!(
					op(input).err().expect("invalid hash input accepted").kind(),
					oa::ErrorKind::InvalidArgument
				);
			}
		}
		Ok(())
	}
);

test_vk!(
	fixed_digest_chain_replays_and_rebinds_without_host_intermediates,
	engine,
	{
		let fixture = fixture();
		let chains = fixture["chains"].as_array().unwrap();
		let input = Matrix::new_from_slice(
			&engine,
			[3, 3],
			&unhex(chains[0]["messages"].as_str().unwrap()),
		)?;
		let (mut plan, output) = engine.capture(|| {
			let digest = hash::sha256(&input)?;
			hash::sha3_256(&digest)
		})?;
		assert_eq!(
			output.try_read::<u8>().unwrap_err().kind(),
			oa::ErrorKind::NotReady
		);
		assert_eq!(plan.diagnostics().node_count(), 2);
		assert_eq!(plan.diagnostics().barrier_count(), 1);
		assert_eq!(plan.semantic_graph().operations().len(), 2);
		for (index, chain) in chains.iter().enumerate() {
			if index != 0 {
				let replacement =
					Matrix::new_from_slice(&engine, [3, 3], &unhex(chain["messages"].as_str().unwrap()))?;
				plan.bind_matrix_input(&input, &replacement)?;
			}
			let first = engine.submit(&plan)?;
			let second = engine.submit(&plan)?;
			first.wait()?;
			second.wait()?;
			assert_eq!(output.read::<u8>()?, flatten(&chain["sha256_sha3_256"]));
		}
		assert_eq!(plan.diagnostics().command_recording_count(), 2);
		assert_eq!(plan.diagnostics().command_cache_hit_count(), 2);
		Ok(())
	}
);

// Independent SHAKE-256 tree oracle, including singleton, odd levels and a tail
// beyond two workgroups. Every level stays on the device during capture/replay.
test_vk!(
	general_merkle_roots_match_independent_oracle_and_replay,
	engine,
	{
		for case in fixture()["merkle_cases"].as_array().unwrap() {
			let rows = case["leaves"].as_array().unwrap().len();
			let bytes = flatten(&case["leaves"]);
			let input = Matrix::new_from_slice(&engine, [rows, 32], &bytes)?;
			let (plan, root) = engine.capture(|| hash::merkle_root(&input))?;
			assert_eq!(plan.semantic_graph().operations().len(), 1);
			assert_eq!(
				root.try_read::<u8>().unwrap_err().kind(),
				oa::ErrorKind::NotReady
			);
			let expected = unhex(case["root"].as_str().unwrap());
			for _ in 0..2 {
				engine.submit(&plan)?.wait()?;
				assert_eq!(root.read::<u8>()?, expected, "Merkle leaf count {rows}");
			}
			assert_eq!(input.read::<u8>()?, bytes);
		}
		Ok(())
	}
);

// Repeated independent rows cross a workgroup boundary and leave non-empty
// tails; host observation also proves that output writes did not alias input.
test_vk!(
	fixed_digest_batch_crosses_workgroup_tails_without_aliasing,
	engine,
	{
		let fixture = fixture();
		let case = &fixture["cases"][2];
		let row_bytes = flatten(&case["messages"]);
		for repetitions in [11, 21] {
			let bytes = row_bytes.repeat(repetitions);
			let rows = 3 * repetitions;
			let input = Matrix::new_from_slice(&engine, [rows, 3], &bytes)?;
			for (name, op, digest_bytes) in HASHES {
				let expected = flatten(&case["digests"][name]).repeat(repetitions);
				let output = op(&input)?;
				assert_eq!(output.shape(), [rows, digest_bytes]);
				assert_eq!(output.read::<u8>()?, expected, "{name}: {rows} rows");
			}
			assert_eq!(input.read::<u8>()?, bytes);
		}
		Ok(())
	}
);
