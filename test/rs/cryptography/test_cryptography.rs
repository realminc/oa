use oa::cryptography::{
	Hash, Hasher, Shake128, Shake256, build_merkle_tree, hash, keccak_f1600, kmac256, merkle_proof,
	merkle_root, shake128, shake256, verify_merkle_proof,
};
use oa::cryptography::{
	SecureBuffer,
	pqc::{
		PUBLIC_KEY_SIZE, SIGNATURE_SIZE, Signature, generate_keypair, sign, sign_hash, verify,
		verify_hash,
	},
};

fn hexadecimal(bytes: &[u8]) -> String {
	bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_hex(text: &str) -> Vec<u8> {
	let (pairs, remainder) = text.as_bytes().as_chunks::<2>();
	assert!(remainder.is_empty());
	pairs
		.iter()
		.map(|pair| {
			let pair = std::str::from_utf8(pair).unwrap();
			u8::from_str_radix(pair, 16).unwrap()
		})
		.collect()
}

#[test]
fn keccak_zero_state_matches_reference_permutation() {
	let mut state = [0_u64; 25];
	keccak_f1600(&mut state);
	assert_eq!(state[0], 0xf125_8f79_40e1_dde7);
	assert_eq!(state[1], 0x84d5_ccf9_33c0_478a);
	assert_eq!(state[2], 0xd598_261e_a65a_a9ee);
	assert_eq!(state[3], 0xbd15_4730_6f80_494d);
	assert_eq!(state[4], 0x8b28_4e05_6253_d057);
}

#[test]
fn shake_known_answer_tests_match_fips_202() {
	let mut shake256_empty = [0_u8; 32];
	shake256(b"", &mut shake256_empty);
	assert_eq!(
		hexadecimal(&shake256_empty),
		"46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"
	);

	let mut shake128_empty = [0_u8; 32];
	shake128(b"", &mut shake128_empty);
	assert_eq!(
		hexadecimal(&shake128_empty),
		"7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26"
	);

	let mut single_byte = [0_u8; 32];
	shake256(b"a", &mut single_byte);
	assert_eq!(
		hexadecimal(&single_byte),
		"867e2cb04f5a04dcbd592501a5e8fe9ceaafca50255626ca736c138042530ba4"
	);
}

#[test]
fn incremental_shake_matches_one_shot_across_absorb_and_squeeze_boundaries() -> oa::Result<()> {
	let input: Vec<u8> = (0..512).map(|index| index as u8).collect();
	let mut expected256 = [0_u8; 257];
	shake256(&input, &mut expected256);
	let mut incremental256 = Shake256::new();
	for chunk in input.chunks(37) {
		incremental256.update(chunk)?;
	}
	let mut actual256 = [0_u8; 257];
	for chunk in actual256.chunks_mut(29) {
		incremental256.squeeze(chunk);
	}
	assert_eq!(actual256, expected256);
	assert!(incremental256.update(b"too late").is_err());

	let mut expected128 = [0_u8; 211];
	shake128(&input, &mut expected128);
	let mut incremental128 = Shake128::new();
	incremental128.update(&input[..173])?;
	incremental128.update(&input[173..])?;
	let mut actual128 = [0_u8; 211];
	incremental128.squeeze(&mut actual128[..17]);
	incremental128.squeeze(&mut actual128[17..]);
	assert_eq!(actual128, expected128);
	Ok(())
}

#[test]
fn kmac256_matches_nist_sp_800_185_sample_four() -> oa::Result<()> {
	let key = parse_hex("404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f");
	let message = parse_hex("00010203");
	let mac = kmac256(&key, &message, b"My Tagged Application", 64)?;
	assert_eq!(
		hexadecimal(&mac),
		"20c570c31346f703c9ac36c61c03cb64c3970d0cfc787e9b79599d273a68d2f7\
		 f69d4cc3de9d104a351689f27cf6f5951f0103f33f4f24871024d9c27773a8dd"
			.replace(' ', "")
	);
	Ok(())
}

#[test]
fn typed_hash_is_strict_and_hasher_finalize_is_idempotent() -> oa::Result<()> {
	let lower = "00112233445566778899aabbccddeefffedcba98765432100123456789abcdef";
	let parsed = Hash::from_hex(lower)?;
	assert_eq!(parsed.to_hex(), lower);
	assert_eq!(Hash::from_hex(&lower.to_ascii_uppercase())?, parsed);
	assert!(Hash::from_hex("").is_err());
	assert!(Hash::from_hex(&format!("{}g", &lower[..63])).is_err());
	assert!(Hash::from_bytes(&[0; 31]).is_err());
	assert_eq!(Hash::from_bytes(parsed.as_bytes())?, parsed);

	let mut hasher = Hasher::new();
	hasher.update(b"abc")?;
	let first = hasher.finalize();
	assert_eq!(hasher.finalize(), first);
	assert_eq!(first, hash(b"abc"));
	assert!(hasher.update(b"abc").is_err());
	hasher.reset();
	hasher.update(b"abc")?;
	assert_eq!(hasher.finalize(), first);
	Ok(())
}

#[test]
fn every_leaf_proof_verifies_for_odd_tree() -> oa::Result<()> {
	let leaves: Vec<Hash> = (0_u8..7).map(|byte| hash(&[byte])).collect();
	let tree = build_merkle_tree(&leaves);
	assert_eq!(tree.root(), merkle_root(&leaves));
	for (index, leaf) in leaves.iter().enumerate() {
		let proof = merkle_proof(&tree, index)?;
		assert!(verify_merkle_proof(leaf, &proof, &tree.root()));
		assert!(!verify_merkle_proof(&hash(&[99]), &proof, &tree.root()));
	}
	assert!(merkle_proof(&tree, leaves.len()).is_err());
	assert!(merkle_root(&[]).is_zero());
	assert!(merkle_proof(&build_merkle_tree(&[]), 0).is_err());
	Ok(())
}

#[test]
fn secure_buffer_erases_on_drop() {
	let mut bytes = [0x7b; 41];
	{
		let mut guard = SecureBuffer::new(&mut bytes);
		assert!(guard.is_valid());
		guard.as_mut_slice()[0] = 0xaa;
	}
	assert_eq!(bytes, [0; 41]);
}

#[test]
fn ml_dsa_keygen_sign_verify_and_serialization() {
	let keypair = generate_keypair().expect("key generation failed");
	let message = b"OA ML-DSA-65 conformance";
	let signature = sign(message, &keypair.secret_key).expect("signing failed");

	assert_eq!(keypair.public_key.as_bytes().len(), PUBLIC_KEY_SIZE);
	assert_eq!(signature.as_bytes().len(), SIGNATURE_SIZE);
	assert!(verify(message, &signature, &keypair.public_key));
	assert!(!verify(b"tampered", &signature, &keypair.public_key));
	assert!(!keypair.public_key.is_zero());
	assert!(!signature.is_zero());
	assert!(oa::cryptography::pqc::PublicKey::default().is_zero());
	assert!(Signature::default().is_zero());
	assert_eq!(
		keypair.public_key.to_bytes(),
		*keypair.public_key.as_bytes()
	);
	assert_eq!(signature.to_bytes(), *signature.as_bytes());

	let parsed_key = oa::cryptography::pqc::PublicKey::from_bytes(keypair.public_key.as_bytes())
		.expect("public key parse failed");
	let parsed_signature =
		Signature::from_bytes(signature.as_bytes()).expect("signature parse failed");
	assert_eq!(parsed_key, keypair.public_key);
	assert_eq!(parsed_signature, signature);
}

#[test]
fn ml_dsa_hash_signatures_and_negative_cases() {
	let first = generate_keypair().expect("first key generation failed");
	let second = generate_keypair().expect("second key generation failed");
	let digest = hash(b"signed digest");
	let signature = sign_hash(&digest, &first.secret_key).expect("hash signing failed");
	assert!(verify_hash(&digest, &signature, &first.public_key));
	assert!(!verify_hash(&digest, &signature, &second.public_key));

	let mut tampered = signature.as_bytes().to_vec();
	tampered[19] ^= 1;
	let tampered = Signature::from_bytes(&tampered).expect("tampered signature length changed");
	assert!(!verify_hash(&digest, &tampered, &first.public_key));
}

#[test]
fn ml_dsa_parsing_rejects_wrong_lengths() {
	assert!(oa::cryptography::pqc::PublicKey::from_bytes(&[0; PUBLIC_KEY_SIZE - 1]).is_err());
	assert!(Signature::from_bytes(&[0; SIGNATURE_SIZE - 1]).is_err());
}

#[test]
fn root_exports_are_cryptography_value_identities() {
	fn root_public_key(_: oa::PublicKey) {}
	fn root_signature(_: oa::Signature) {}
	fn root_keypair(_: oa::Keypair) {}
	let keypair: oa::cryptography::pqc::Keypair =
		generate_keypair().expect("key generation failed");
	let signature: oa::cryptography::pqc::Signature =
		sign(b"identity", &keypair.secret_key).expect("signing failed");
	root_signature(signature);
	root_public_key(keypair.public_key.clone());
	root_keypair(keypair);
}

test_vk!(batch_shake_matches_cpu_across_rate_boundaries, engine, {
	let empty = oa::Matrix::from_slice(&engine, [2, 0], &[] as &[u8])?;
	assert_eq!(
		oa::cryptography::hash::shake128(&empty, 0)?.shape(),
		[2, 16]
	);
	assert_eq!(
		oa::cryptography::hash::shake256(&empty, 0)?.shape(),
		[2, 32]
	);
	for (message_len, output_len) in [(0, 1), (3, 32), (135, 137), (136, 64), (169, 17)] {
		let rows = 3;
		let values = (0_usize..rows * message_len)
			.map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
			.collect::<Vec<_>>();
		let input = oa::Matrix::from_slice(&engine, [rows, message_len], &values)?;

		let output256 = oa::cryptography::hash::shake256(&input, output_len)?;
		let padded_len = output_len.div_ceil(8) * 8;
		assert_eq!(output256.shape(), [rows, padded_len]);
		let actual256 = output256.read::<u8>()?;
		for row in 0..rows {
			let mut expected = vec![0_u8; padded_len];
			shake256(
				&values[row * message_len..(row + 1) * message_len],
				&mut expected,
			);
			assert_eq!(
				&actual256[row * padded_len..(row + 1) * padded_len],
				expected
			);
		}

		let output128 = oa::cryptography::hash::shake128(&input, output_len)?;
		let actual128 = output128.read::<u8>()?;
		for row in 0..rows {
			let mut expected = vec![0_u8; padded_len];
			shake128(
				&values[row * message_len..(row + 1) * message_len],
				&mut expected,
			);
			assert_eq!(
				&actual128[row * padded_len..(row + 1) * padded_len],
				expected
			);
		}
	}
	Ok(())
});

test_vk!(batch_keccak_matches_cpu_permutation, engine, {
	let rows = 5;
	let values = (0_usize..rows * 200)
		.map(|index| (index.wrapping_mul(29).wrapping_add(7) & 0xff) as u8)
		.collect::<Vec<_>>();
	let input = oa::Matrix::from_slice(&engine, [rows, 200], &values)?;
	let output = oa::cryptography::hash::keccak_f1600(&input)?;
	let actual = output.read::<u8>()?;
	let mut expected = Vec::with_capacity(values.len());
	for row in values.as_chunks::<200>().0 {
		let mut state = [0_u64; 25];
		for (lane, bytes) in state.iter_mut().zip(row.as_chunks::<8>().0) {
			*lane = u64::from_le_bytes(*bytes);
		}
		keccak_f1600(&mut state);
		expected.extend(state.into_iter().flat_map(u64::to_le_bytes));
	}
	assert_eq!(actual, expected);
	Ok(())
});

test_vk!(
	batch_merkle_matches_cpu_and_supports_deferred_hashes,
	engine,
	{
		for leaf_count in [1, 2, 4, 16, 1024] {
			let leaves = (0_usize..leaf_count)
				.map(|index| hash(&index.to_le_bytes()))
				.collect::<Vec<_>>();
			let bytes = leaves
				.iter()
				.flat_map(|leaf| leaf.as_bytes().iter().copied())
				.collect::<Vec<_>>();
			let input = oa::Matrix::from_slice(&engine, [leaf_count, 32], &bytes)?;
			let output = oa::cryptography::hash::merkle_root(&input)?;
			assert_eq!(output.shape(), [1, 32]);
			assert_eq!(output.read::<u8>()?, merkle_root(&leaves).as_bytes());
		}

		let messages = oa::Matrix::from_slice(&engine, [4, 3], b"abcdefghijkl")?;
		let hashed = oa::cryptography::hash::shake256(&messages, 32)?;
		let root = oa::cryptography::hash::merkle_root(&hashed)?;
		let leaves = b"abcdefghijkl"
			.as_chunks::<3>()
			.0
			.iter()
			.map(|message| hash(message))
			.collect::<Vec<_>>();
		assert_eq!(root.read::<u8>()?, merkle_root(&leaves).as_bytes());
		Ok(())
	}
);

test_vk!(multi_level_merkle_has_one_semantic_identity, engine, {
	let input = oa::Matrix::from_slice(&engine, [4, 32], &[0_u8; 128])?;
	let (plan, _root) = engine.capture(|| oa::cryptography::hash::merkle_root(&input))?;
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::cryptography::hash::merkle_root"
	);
	assert_eq!(plan.semantic_lowering().maximum_nodes_per_op(), 2);
	Ok(())
});

test_vk!(batch_hash_rejects_invalid_contracts, engine, {
	let f32_input = oa::Matrix::from_f32(&engine, [1, 32], &[0.0; 32])?;
	let rank_one = oa::Matrix::from_slice(&engine, [32], &[0_u8; 32])?;
	let wrong_state = oa::Matrix::from_slice(&engine, [1, 199], &[0_u8; 199])?;
	let odd_leaves = oa::Matrix::from_slice(&engine, [3, 32], &[0_u8; 96])?;

	for error in [
		oa::cryptography::hash::shake256(&f32_input, 32)
			.err()
			.expect("F32 SHAKE input was accepted"),
		oa::cryptography::hash::shake128(&rank_one, 16)
			.err()
			.expect("rank-one SHAKE input was accepted"),
		oa::cryptography::hash::keccak_f1600(&wrong_state)
			.err()
			.expect("short Keccak state was accepted"),
		oa::cryptography::hash::merkle_root(&odd_leaves)
			.err()
			.expect("odd Merkle level was accepted"),
	] {
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}
	Ok(())
});
