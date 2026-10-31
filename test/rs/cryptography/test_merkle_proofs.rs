//! Selected device paths and closed malformed-proof handling.
use oa::{Matrix, cryptography::hash};

fn fixture() -> serde_json::Value {
	serde_json::from_str(include_str!(
		"../../fixtures/cryptography/hash/merkle_proofs.json"
	))
	.unwrap()
}
fn unhex(text: &str) -> Vec<u8> {
	text
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
		.flat_map(|v| unhex(v.as_str().unwrap()))
		.collect()
}

test_vk!(
	merkle_proof_selected_indices_match_independent_oracle_and_replay,
	engine,
	{
		for case in fixture()["cases"].as_array().unwrap() {
			let count = case["leaves"].as_array().unwrap().len();
			let leaves = Matrix::new_from_slice(&engine, [count, 32], &flatten(&case["leaves"]))?;
			let selected = Matrix::new_from_slice(&engine, [5, 32], &flatten(&case["selected_leaves"]))?;
			let indices = case["indices"]
				.as_array()
				.unwrap()
				.iter()
				.map(|i| u32::try_from(i.as_u64().unwrap()).unwrap())
				.collect::<Vec<_>>();
			let indices = Matrix::new_from_slice(&engine, [5], &indices)?;
			let root = Matrix::new_from_slice(&engine, [1, 32], &unhex(case["root"].as_str().unwrap()))?;
			let (plan, (proofs, valid)) = engine.capture(|| {
				let proofs = hash::merkle_proofs(&leaves, &indices)?;
				let valid =
					hash::verify_merkle_proofs(&selected, &proofs, &root, u32::try_from(count).unwrap())?;
				Ok((proofs, valid))
			})?;
			assert_eq!(plan.semantic_graph().operations().len(), 2);
			assert_eq!(
				valid.try_read::<u32>().unwrap_err().kind(),
				oa::ErrorKind::NotReady
			);
			for _ in 0..2 {
				engine.submit(&plan)?.wait()?;
				assert_eq!(
					proofs.read::<u8>()?,
					flatten(&case["frames"]),
					"leaf count {count}"
				);
				assert_eq!(valid.read::<u32>()?, [1, 1, 1, 0, 0], "leaf count {count}");
			}
		}
		Ok(())
	}
);

test_vk!(
	merkle_proof_headers_paths_leaves_and_roots_reject_tampering,
	engine,
	{
		let fixture = fixture();
		let case = &fixture["cases"][3]; // five leaves, including duplicate-last paths
		let source = flatten(&case["frames"]);
		let frame_bytes = source.len() / 5;
		let leaves_bytes = flatten(&case["selected_leaves"]);
		let leaves = Matrix::new_from_slice(&engine, [5, 32], &leaves_bytes)?;
		let root_bytes = unhex(case["root"].as_str().unwrap());
		let root = Matrix::new_from_slice(&engine, [1, 32], &root_bytes)?;
		for offset in [0, 4, 8, 12, 16, frame_bytes - 1] {
			let mut bytes = source.clone();
			bytes[offset] ^= 0x80;
			let proof = Matrix::new_from_slice(&engine, [5, frame_bytes], &bytes)?;
			let flags = hash::verify_merkle_proofs(&leaves, &proof, &root, 5)?.read::<u32>()?;
			assert_eq!(flags, [0, 1, 1, 0, 0], "tampered byte {offset}");
		}
		let proof = Matrix::new_from_slice(&engine, [5, frame_bytes], &source)?;
		// Same-depth cardinalities must still match trusted tree metadata.
		assert_eq!(
			hash::verify_merkle_proofs(&leaves, &proof, &root, 6)?.read::<u32>()?,
			[0; 5]
		);
		let mut changed = leaves_bytes.clone();
		changed[0] ^= 1;
		let changed = Matrix::new_from_slice(&engine, [5, 32], &changed)?;
		assert_eq!(
			hash::verify_merkle_proofs(&changed, &proof, &root, 5)?.read::<u32>()?,
			[0, 1, 1, 0, 0]
		);
		let mut changed_root = root_bytes.repeat(5);
		changed_root[0] ^= 1;
		let changed_root = Matrix::new_from_slice(&engine, [5, 32], &changed_root)?;
		assert_eq!(
			hash::verify_merkle_proofs(&leaves, &proof, &changed_root, 5)?.read::<u32>()?,
			[0, 1, 1, 0, 0]
		);
		Ok(())
	}
);

test_vk!(
	merkle_proof_empty_selections_and_invalid_metadata,
	engine,
	{
		let leaves = Matrix::new_from_slice(&engine, [1, 32], &[1_u8; 32])?;
		let empty_indices = Matrix::new_from_slice(&engine, [0], &[] as &[u32])?;
		let frames = hash::merkle_proofs(&leaves, &empty_indices)?;
		assert_eq!(frames.shape(), [0, 16]);
		assert!(frames.try_read::<u8>()?.is_empty());
		let empty_leaves = Matrix::new_from_slice(&engine, [0, 32], &[] as &[u8])?;
		assert!(
			hash::verify_merkle_proofs(&empty_leaves, &frames, &leaves, 1)?
				.try_read::<u32>()?
				.is_empty()
		);
		let wrong_indices = Matrix::new_from_slice(&engine, [1], &[0_u8])?;
		assert_eq!(
			hash::merkle_proofs(&leaves, &wrong_indices)
				.err()
				.unwrap()
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let wrong_frame = Matrix::new_from_slice(&engine, [1, 17], &[0_u8; 17])?;
		assert_eq!(
			hash::verify_merkle_proofs(&leaves, &wrong_frame, &leaves, 1)
				.err()
				.unwrap()
				.kind(),
			oa::ErrorKind::InvalidArgument
		);
		Ok(())
	}
);

// Two workgroups, an odd final invocation and a device-produced flag vector.
test_vk!(
	merkle_proof_workgroup_tail_chains_and_rebinds_device_indices,
	engine,
	{
		let fixture = fixture();
		let case = &fixture["cases"][3];
		let leaves_bytes = flatten(&case["leaves"]);
		let leaves = Matrix::new_from_slice(&engine, [5, 32], &leaves_bytes)?;
		let indices = Matrix::new_from_slice(&engine, [33], &[0_u32; 33])?;
		let selected = Matrix::new_from_slice(&engine, [33, 32], &leaves_bytes[..32].repeat(33))?;
		let root = Matrix::new_from_slice(&engine, [1, 32], &unhex(case["root"].as_str().unwrap()))?;
		let (mut plan, valid) = engine.capture(|| {
			let frames = hash::merkle_proofs(&leaves, &indices)?;
			hash::verify_merkle_proofs(&selected, &frames, &root, 5)
		})?;
		engine.submit(&plan)?.wait()?;
		assert_eq!(valid.read::<u32>()?, [1; 33]);
		let mut replacements = [0_u32; 33];
		replacements[32] = 5;
		let replacement = Matrix::new_from_slice(&engine, [33], &replacements)?;
		plan.bind_matrix_input(&indices, &replacement)?;
		engine.submit(&plan)?.wait()?;
		let mut expected = [1_u32; 33];
		expected[32] = 0;
		assert_eq!(valid.read::<u32>()?, expected);
		Ok(())
	}
);
