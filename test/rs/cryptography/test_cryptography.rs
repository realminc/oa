use oa::cryptography::{
	Hash, Hasher, Shake128, Shake256, build_merkle_tree, hash, keccak_f1600, kmac256, merkle_proof,
	merkle_root, shake128, shake256, verify_merkle_proof,
};
use oa::cryptography::{
	SecureBuffer,
	pqc::{
		PUBLIC_KEY_SIZE, SIGNATURE_SIZE, Signature, generate_keypair, sign, sign_hash, verify,
		verify_batch, verify_hash,
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
	let keypair: oa::cryptography::pqc::Keypair = generate_keypair().expect("key generation failed");
	let signature: oa::cryptography::pqc::Signature =
		sign(b"identity", &keypair.secret_key).expect("signing failed");
	root_signature(signature);
	root_public_key(keypair.public_key.clone());
	root_keypair(keypair);
}

// Keep the first hardware probe independent of earlier pending hashes, buffer
// recycling, rate-boundary loops and Merkle. A reset-affected device cannot
// qualify this test or the larger suite.
test_vk!(
	batch_shake128_empty_message_isolated,
	engine,
	"requires a recovered hardware Vulkan compute device",
	{
		let input = oa::Matrix::from_slice(&engine, [1, 0], &[] as &[u8])?;
		let output = oa::cryptography::hash::shake128(&input, 16)?;
		let mut expected = [0_u8; 16];
		shake128(b"", &mut expected);
		assert_eq!(output.read::<u8>()?, expected);
		Ok(())
	}
);

test_vk!(
	batch_shake_variant_transition,
	engine,
	"requires a recovered hardware Vulkan compute device",
	{
		let input = oa::Matrix::from_slice(&engine, [3, 0], &[] as &[u8])?;
		let first = oa::cryptography::hash::shake256(&input, 8)?;
		let mut first_row = [0_u8; 8];
		shake256(b"", &mut first_row);
		assert_eq!(first.read::<u8>()?, first_row.repeat(3));

		let second = oa::cryptography::hash::shake128(&input, 8)?;
		let witness = oa::cryptography::hash::shake256(&second, 8)?;
		let mut second_row = [0_u8; 8];
		shake128(b"", &mut second_row);
		let mut witness_row = [0_u8; 8];
		shake256(&second_row, &mut witness_row);
		// A device consumer distinguishes wrong producer bytes from host
		// observation failure, while exercising an actual RAW dependency.
		assert_eq!(witness.read::<u8>()?, witness_row.repeat(3));
		assert_eq!(second.read::<u8>()?, second_row.repeat(3));
		Ok(())
	}
);

test_vk!(
	batch_shake256_short_message_isolated,
	engine,
	"requires a recovered hardware Vulkan compute device",
	{
		let values = [11_u8, 48, 85, 122, 159, 196, 233, 14, 51];
		let input = oa::Matrix::from_slice(&engine, [3, 3], &values)?;
		let output = oa::cryptography::hash::shake256(&input, 32)?;
		let actual = output.read::<u8>()?;
		for row in 0..3 {
			let mut expected = [0_u8; 32];
			shake256(&values[row * 3..(row + 1) * 3], &mut expected);
			assert_eq!(&actual[row * 32..(row + 1) * 32], expected);
		}
		Ok(())
	}
);

test_vk!(batch_shake_matches_cpu_across_rate_boundaries, engine, {
	eprintln!("SHAKE: recording default-length outputs");
	let empty = oa::Matrix::from_slice(&engine, [2, 0], &[] as &[u8])?;
	assert_eq!(
		oa::cryptography::hash::shake128(&empty, 0)?.shape(),
		[2, 16]
	);
	assert_eq!(
		oa::cryptography::hash::shake256(&empty, 0)?.shape(),
		[2, 32]
	);
	for (message_len, output_len) in [
		(0, 1),
		(3, 32),
		(135, 137),
		(136, 64),
		(137, 17),
		(167, 169),
		(168, 168),
		(169, 17),
		(336, 337),
	] {
		let rows = 3;
		let values = (0_usize..rows * message_len)
			.map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
			.collect::<Vec<_>>();
		let input = oa::Matrix::from_slice(&engine, [rows, message_len], &values)?;

		let output256 = oa::cryptography::hash::shake256(&input, output_len)?;
		eprintln!("SHAKE256: observing message_len={message_len}, output_len={output_len}");
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
		eprintln!("SHAKE128: observing message_len={message_len}, output_len={output_len}");
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

test_vk!(
	batch_hash_chain_replays_and_rebinds_without_host_intermediates,
	engine,
	{
		let input = oa::Matrix::from_slice(&engine, [4, 3], b"abcdefghijkl")?;
		let (mut plan, root) = engine.capture(|| {
			let leaves = oa::cryptography::hash::shake256(&input, 32)?;
			oa::cryptography::hash::merkle_root(&leaves)
		})?;
		assert_eq!(
			root.try_read::<u8>().unwrap_err().kind(),
			oa::ErrorKind::NotReady
		);
		assert_eq!(plan.diagnostics().node_count(), 3);
		assert_eq!(plan.diagnostics().barrier_count(), 2);
		assert_eq!(plan.semantic_graph().operations().len(), 2);

		for messages in [b"abcdefghijkl", b"mnopqrstuvwx"] {
			if messages != b"abcdefghijkl" {
				let replacement = oa::Matrix::from_slice(&engine, [4, 3], messages)?;
				plan.bind_matrix_input(&input, &replacement)?;
				// The immutable plan retains this replacement after its local handle drops.
			}
			let first = engine.submit(&plan)?;
			let second = engine.submit(&plan)?;
			first.wait()?;
			second.wait()?;
			let leaves = messages
				.as_chunks::<3>()
				.0
				.iter()
				.map(|bytes| hash(bytes))
				.collect::<Vec<_>>();
			assert_eq!(root.read::<u8>()?, merkle_root(&leaves).as_bytes());
		}
		assert_eq!(plan.diagnostics().command_recording_count(), 2);
		assert_eq!(plan.diagnostics().command_cache_hit_count(), 2);
		assert_eq!(plan.diagnostics().submission_count(), 4);
		assert_eq!(plan.diagnostics().input_rebinding_count(), 1);
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

// The pinned sigGen corpus contains expanded test keys, not public keys or
// seeds. Import only these trusted fixture bytes through the independent
// RustCrypto oracle; this deprecated decoder is never a production API.
#[allow(deprecated)]
fn nist_public_key<P: ml_dsa::MlDsaParams>(
	key: &[u8],
	message: &[u8],
	signature: &[u8],
	context: &[u8],
) -> Vec<u8> {
	let key = ml_dsa::ExpandedSigningKeyBytes::<P>::try_from(key).unwrap();
	let key = ml_dsa::ExpandedSigningKey::<P>::from_expanded(&key).verifying_key();
	let signature = ml_dsa::EncodedSignature::<P>::try_from(signature).unwrap();
	let signature = ml_dsa::Signature::<P>::decode(&signature).unwrap();
	assert!(key.verify_with_context(message, context, &signature));
	key.encode().to_vec()
}

test_vk!(ml_dsa_all_parameters_nist_batch_capture, engine, {
	use oa::cryptography::pqc::{MlDsaParameters, verify_batch_with_parameters};

	let prompt: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/sigGen-prompt.json"
	))
	.unwrap();
	let answers: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/sigGen-expectedResults.json"
	))
	.unwrap();
	for (parameter, name, expected_cases) in [
		(MlDsaParameters::MlDsa44, "ML-DSA-44", 3),
		(MlDsaParameters::MlDsa65, "ML-DSA-65", 2),
		(MlDsaParameters::MlDsa87, "ML-DSA-87", 3),
	] {
		let mut messages = vec![0xa5_u8]; // Deliberately unaligned first message.
		let mut offsets = Vec::new();
		let mut lengths = Vec::new();
		let mut signatures = Vec::new();
		let mut keys = Vec::new();
		let mut expected = Vec::new();
		let mut cases = 0;
		for group in prompt["testGroups"].as_array().unwrap() {
			if group["parameterSet"] != name
				|| group["signatureInterface"] != "external"
				|| group["preHash"] != "pure"
			{
				continue;
			}
			let answer_group = answers["testGroups"]
				.as_array()
				.unwrap()
				.iter()
				.find(|answer| answer["tgId"] == group["tgId"])
				.unwrap();
			for case in group["tests"].as_array().unwrap() {
				if case["context"] != "" {
					continue;
				}
				let answer = answer_group["tests"]
					.as_array()
					.unwrap()
					.iter()
					.find(|answer| answer["tcId"] == case["tcId"])
					.unwrap();
				let message = parse_hex(case["message"].as_str().unwrap());
				let signature = parse_hex(answer["signature"].as_str().unwrap());
				let fixture_key = parse_hex(case["sk"].as_str().unwrap());
				let key = match parameter {
					MlDsaParameters::MlDsa44 => {
						nist_public_key::<ml_dsa::MlDsa44>(&fixture_key, &message, &signature, &[])
					}
					MlDsaParameters::MlDsa65 => {
						nist_public_key::<ml_dsa::MlDsa65>(&fixture_key, &message, &signature, &[])
					}
					MlDsaParameters::MlDsa87 => {
						nist_public_key::<ml_dsa::MlDsa87>(&fixture_key, &message, &signature, &[])
					}
				};
				assert_eq!(key.len(), parameter.public_key_size());
				assert_eq!(signature.len(), parameter.signature_size());
				let offset = u32::try_from(messages.len()).unwrap();
				messages.extend_from_slice(&message);
				// Exact NIST result, tampered challenge, noncanonical encoding,
				// and overflowing message range all share one odd-stride batch.
				for kind in 0..4 {
					offsets.push(if kind == 3 { u32::MAX } else { offset });
					lengths.push(u32::try_from(message.len()).unwrap());
					keys.extend_from_slice(&key);
					let start = signatures.len();
					signatures.extend_from_slice(&signature);
					if kind == 1 {
						signatures[start] ^= 1;
					}
					if kind == 2 {
						signatures[start..].fill(0xff);
					}
					expected.push(u32::from(kind == 0));
				}
				cases += 1;
			}
		}
		assert_eq!(cases, expected_cases, "{name} fixture coverage");
		let messages = oa::Matrix::from_slice(&engine, [messages.len()], &messages)?;
		let offsets = oa::Matrix::from_slice(&engine, [offsets.len()], &offsets)?;
		let lengths = oa::Matrix::from_slice(&engine, [lengths.len()], &lengths)?;
		let signatures = oa::Matrix::from_slice(&engine, [signatures.len()], &signatures)?;
		let keys = oa::Matrix::from_slice(&engine, [keys.len()], &keys)?;
		let (plan, results) = engine.capture(|| {
			verify_batch_with_parameters(parameter, &messages, &offsets, &lengths, &signatures, &keys)
		})?;
		assert_eq!(plan.diagnostics().semantic_operation_count(), 1);
		assert_eq!(
			results.try_read::<u32>().unwrap_err().kind(),
			oa::ErrorKind::NotReady
		);
		for _ in 0..2 {
			engine.submit(&plan)?.wait()?;
			assert_eq!(results.read::<u32>()?, expected, "{name} NIST batch");
		}
		let wrong_parameter = if parameter == MlDsaParameters::MlDsa44 {
			MlDsaParameters::MlDsa87
		} else {
			MlDsaParameters::MlDsa44
		};
		assert_eq!(
			verify_batch_with_parameters(
				wrong_parameter,
				&messages,
				&offsets,
				&lengths,
				&signatures,
				&keys,
			)
			.err()
			.expect("cross-parameter buffers must be rejected")
			.kind(),
			oa::ErrorKind::InvalidArgument
		);
		let empty_bytes = oa::Matrix::from_slice(&engine, [0], &[] as &[u8])?;
		let empty_indices = oa::Matrix::from_slice(&engine, [0], &[] as &[u32])?;
		let empty = verify_batch_with_parameters(
			parameter,
			&empty_bytes,
			&empty_indices,
			&empty_indices,
			&empty_bytes,
			&empty_bytes,
		)?;
		assert_eq!(empty.shape(), [0]);
		assert!(empty.try_read::<u32>()?.is_empty());
	}
	Ok(())
});

test_vk!(ml_dsa_all_parameters_context_capture, engine, {
	use oa::cryptography::pqc::{MlDsaParameters, verify_batch_with_context};
	let prompt: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/sigGen-prompt.json"
	))
	.unwrap();
	let answers: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/sigGen-expectedResults.json"
	))
	.unwrap();
	for (parameter, name) in [
		(MlDsaParameters::MlDsa44, "ML-DSA-44"),
		(MlDsaParameters::MlDsa65, "ML-DSA-65"),
		(MlDsaParameters::MlDsa87, "ML-DSA-87"),
	] {
		let group = prompt["testGroups"]
			.as_array()
			.unwrap()
			.iter()
			.find(|group| {
				group["parameterSet"] == name
					&& group["signatureInterface"] == "external"
					&& group["preHash"] == "pure"
			})
			.unwrap();
		let cases = group["tests"].as_array().unwrap();
		let maximum = cases
			.iter()
			.find(|case| case["context"].as_str().unwrap().len() == 510)
			.unwrap();
		for case in [&cases[0], maximum] {
			let answer_group = answers["testGroups"]
				.as_array()
				.unwrap()
				.iter()
				.find(|answer| answer["tgId"] == group["tgId"])
				.unwrap();
			let answer = answer_group["tests"]
				.as_array()
				.unwrap()
				.iter()
				.find(|answer| answer["tcId"] == case["tcId"])
				.unwrap();
			let context = parse_hex(case["context"].as_str().unwrap());
			let message = parse_hex(case["message"].as_str().unwrap());
			let signature = parse_hex(answer["signature"].as_str().unwrap());
			let fixture_key = parse_hex(case["sk"].as_str().unwrap());
			let key = match parameter {
				MlDsaParameters::MlDsa44 => {
					nist_public_key::<ml_dsa::MlDsa44>(&fixture_key, &message, &signature, &context)
				}
				MlDsaParameters::MlDsa65 => {
					nist_public_key::<ml_dsa::MlDsa65>(&fixture_key, &message, &signature, &context)
				}
				MlDsaParameters::MlDsa87 => {
					nist_public_key::<ml_dsa::MlDsa87>(&fixture_key, &message, &signature, &context)
				}
			};
			// Odd shared-context start and message start exercise packed-byte reads.
			let mut bytes = vec![0xa5];
			bytes.extend_from_slice(&context);
			bytes.push(0xa5);
			let offset = u32::try_from(bytes.len()).unwrap();
			bytes.extend_from_slice(&message);
			let messages = oa::Matrix::from_slice(&engine, [bytes.len()], &bytes)?;
			let offsets = oa::Matrix::from_slice(&engine, [1], &[offset])?;
			let lengths = oa::Matrix::from_slice(&engine, [1], &[u32::try_from(message.len()).unwrap()])?;
			let signatures = oa::Matrix::from_slice(&engine, [signature.len()], &signature)?;
			let keys = oa::Matrix::from_slice(&engine, [key.len()], &key)?;
			let (mut plan, result) = engine.capture(|| {
				verify_batch_with_context(
					parameter,
					&messages,
					&offsets,
					&lengths,
					&signatures,
					&keys,
					1..1 + context.len(),
				)
			})?;
			assert_eq!(
				result.try_read::<u32>().unwrap_err().kind(),
				oa::ErrorKind::NotReady
			);
			for _ in 0..2 {
				engine.submit(&plan)?.wait()?;
				assert_eq!(result.read::<u32>()?, [1], "{name} exact NIST context");
			}
			// The shared context is retained and rebound with the semantic message input.
			bytes[1] ^= 1;
			let changed = oa::Matrix::from_slice(&engine, [bytes.len()], &bytes)?;
			plan.bind_matrix_input(&messages, &changed)?;
			engine.submit(&plan)?.wait()?;
			assert_eq!(result.read::<u32>()?, [0], "{name} changed context");
			let wrong_context = verify_batch_with_context(
				parameter,
				&messages,
				&offsets,
				&lengths,
				&signatures,
				&keys,
				0..0,
			)?;
			assert_eq!(wrong_context.read::<u32>()?, [0], "{name} context omitted");
			for range in [
				0..256,
				bytes.len()..bytes.len() + 1,
				std::ops::Range { start: 2, end: 1 },
			] {
				assert_eq!(
					verify_batch_with_context(
						parameter,
						&messages,
						&offsets,
						&lengths,
						&signatures,
						&keys,
						range,
					)
					.err()
					.expect("invalid context must fail")
					.kind(),
					oa::ErrorKind::InvalidArgument
				);
			}
			let empty_bytes = oa::Matrix::from_slice(&engine, [0], &[] as &[u8])?;
			let empty_indices = oa::Matrix::from_slice(&engine, [0], &[] as &[u32])?;
			assert!(
				verify_batch_with_context(
					parameter,
					&messages,
					&empty_indices,
					&empty_indices,
					&empty_bytes,
					&empty_bytes,
					1..1 + context.len()
				)?
				.try_read::<u32>()?
				.is_empty()
			);
			assert_eq!(
				verify_batch_with_context(
					parameter,
					&empty_bytes,
					&empty_indices,
					&empty_indices,
					&empty_bytes,
					&empty_bytes,
					0..1
				)
				.err()
				.expect("empty batch must validate context")
				.kind(),
				oa::ErrorKind::InvalidArgument
			);
		}
	}
	Ok(())
});

test_vk!(ml_dsa_65_gpu_batch_verify_matches_cpu_oracle, engine, {
	// Odd batch: four valid signatures and one tampered signature.
	let batch_size: usize = 5;
	let messages: Vec<&[u8]> = vec![
		b"OA vkPQC batch verify message 0",
		b"OA vkPQC batch verify message 1",
		b"OA vkPQC batch verify message 2",
		b"OA vkPQC batch verify message 3",
		b"OA vkPQC batch verify message 4",
	];

	// Generate two keypairs; first three use kp0, last two use kp1.
	let kp0 = generate_keypair().expect("keypair 0 generation failed");
	let kp1 = generate_keypair().expect("keypair 1 generation failed");
	let keypairs = [&kp0, &kp0, &kp0, &kp1, &kp1];

	// Sign all messages with their respective keypairs.
	let mut signatures_bytes = Vec::with_capacity(batch_size * SIGNATURE_SIZE);
	let mut public_keys_bytes = Vec::with_capacity(batch_size * PUBLIC_KEY_SIZE);
	for (message, kp) in messages.iter().zip(&keypairs) {
		let sig = sign(message, &kp.secret_key).expect("signing failed");
		signatures_bytes.extend_from_slice(sig.as_bytes());
		public_keys_bytes.extend_from_slice(kp.public_key.as_bytes());
	}

	// Build the flat message buffer and per-message offsets/lengths.
	let mut message_bytes = Vec::new();
	let mut offsets = Vec::<u32>::with_capacity(batch_size);
	let mut lengths = Vec::<u32>::with_capacity(batch_size);
	for msg in &messages {
		offsets.push(u32::try_from(message_bytes.len()).expect("offset fits u32"));
		lengths.push(u32::try_from(msg.len()).expect("length fits u32"));
		message_bytes.extend_from_slice(msg);
	}

	// Tamper with signature 3.
	// (Index 3 in signatures_bytes starts at byte 3 * SIGNATURE_SIZE.)
	signatures_bytes[3 * SIGNATURE_SIZE + 42] ^= 0x01;

	// Upload buffers.
	let msg_buf = oa::Matrix::from_slice(&engine, [message_bytes.len()], &message_bytes)?;
	let off_buf = oa::Matrix::from_slice(&engine, [batch_size], &offsets)?;
	let len_buf = oa::Matrix::from_slice(&engine, [batch_size], &lengths)?;
	let sig_buf = oa::Matrix::from_slice(&engine, [batch_size * SIGNATURE_SIZE], &signatures_bytes)?;
	let pk_buf = oa::Matrix::from_slice(&engine, [batch_size * PUBLIC_KEY_SIZE], &public_keys_bytes)?;

	// Run GPU batch verify.
	let results = verify_batch(&msg_buf, &off_buf, &len_buf, &sig_buf, &pk_buf)?;
	assert_eq!(results.shape(), [batch_size]);
	let result_vals = results.read::<u32>()?;

	// Entries 0, 1, 2, 4 must be valid (1); entry 3 tampered → 0.
	// Also cross-check against the CPU oracle.
	let expected_cpu: Vec<u32> = (0..batch_size)
		.map(|i| {
			// Reconstruct the (possibly tampered) signature for entry i.
			let sig_slice = &signatures_bytes[i * SIGNATURE_SIZE..(i + 1) * SIGNATURE_SIZE];
			let sig = Signature::from_bytes(sig_slice).expect("signature bytes");
			let pk = keypairs[i].public_key.clone();
			if verify(messages[i], &sig, &pk) { 1 } else { 0 }
		})
		.collect();

	assert_eq!(
		result_vals, expected_cpu,
		"GPU verify_batch result does not match CPU oracle"
	);
	// Entry 3 must definitely be 0 (tampered).
	assert_eq!(result_vals[3], 0, "tampered signature must be rejected");
	// Entries 0–2 and 4 must be 1.
	for i in [0, 1, 2, 4] {
		assert_eq!(result_vals[i], 1, "valid signature {i} must be accepted");
	}
	Ok(())
});

test_vk!(
	ml_dsa_65_batch_capture_checks_logical_ranges_and_rebinds,
	engine,
	{
		let kp = generate_keypair()?;
		let mut bytes = vec![0xA5_u8]; // force non-word-aligned message starts
		let mut offsets = Vec::new();
		let mut lengths = Vec::new();
		let mut signatures = Vec::new();
		let mut keys = Vec::new();
		for length in [0_usize, 1, 3, 135, 136, 137] {
			let message: Vec<u8> = (0..length)
				.map(|i| u8::try_from(i % 251).unwrap())
				.collect();
			offsets.push(u32::try_from(bytes.len()).unwrap());
			lengths.push(u32::try_from(length).unwrap());
			bytes.extend_from_slice(&message);
			bytes.push(0xA5);
			signatures.extend_from_slice(sign(&message, &kp.secret_key)?.as_bytes());
			keys.extend_from_slice(kp.public_key.as_bytes());
		}
		let empty_signature = sign(b"", &kp.secret_key)?;
		// Valid empty message at the logical end, followed by overflow/out-of-range
		// cases that must fail closed rather than use allocation padding as input.
		let logical_end = u32::try_from(bytes.len()).unwrap();
		assert_ne!(logical_end % 4, 0);
		for (offset, length) in [
			(logical_end, 0),
			(logical_end + 1, 0),
			(u32::MAX, 0),
			(0, u32::MAX),
			(logical_end, 1),
		] {
			offsets.push(offset);
			lengths.push(length);
			signatures.extend_from_slice(empty_signature.as_bytes());
			keys.extend_from_slice(kp.public_key.as_bytes());
		}
		let count = offsets.len();
		let messages = oa::Matrix::from_slice(&engine, [bytes.len()], &bytes)?;
		let offsets = oa::Matrix::from_slice(&engine, [count], &offsets)?;
		let lengths = oa::Matrix::from_slice(&engine, [count], &lengths)?;
		let signatures = oa::Matrix::from_slice(&engine, [signatures.len()], &signatures)?;
		let keys = oa::Matrix::from_slice(&engine, [keys.len()], &keys)?;
		let (mut plan, output) =
			engine.capture(|| verify_batch(&messages, &offsets, &lengths, &signatures, &keys))?;
		assert_eq!(
			output.try_read::<u32>().unwrap_err().kind(),
			oa::ErrorKind::NotReady
		);
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		let expected = [1_u32, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
		for _ in 0..2 {
			engine.submit(&plan)?.wait()?;
			assert_eq!(output.read::<u32>()?, expected);
		}
		let replacement = oa::Matrix::from_slice(
			&engine,
			[count * SIGNATURE_SIZE],
			&vec![0_u8; count * SIGNATURE_SIZE],
		)?;
		plan.bind_matrix_input(&signatures, &replacement)?;
		engine.submit(&plan)?.wait()?;
		assert_eq!(output.read::<u32>()?, vec![0_u32; count]);
		assert_eq!(plan.diagnostics().input_rebinding_count(), 1);
		// Empty batch has a typed empty result and no shader invocation.
		let empty_bytes = oa::Matrix::from_slice(&engine, [0], &[] as &[u8])?;
		let empty_indices = oa::Matrix::from_slice(&engine, [0], &[] as &[u32])?;
		assert!(
			verify_batch(
				&empty_bytes,
				&empty_indices,
				&empty_indices,
				&empty_bytes,
				&empty_bytes
			)?
			.read::<u32>()?
			.is_empty()
		);
		// A nonempty batch can verify an empty logical message buffer. Its U8
		// allocation padding must neither become message data nor a host fallback.
		let single_offset = oa::Matrix::from_slice(&engine, [1], &[0_u32])?;
		let single_signature =
			oa::Matrix::from_slice(&engine, [SIGNATURE_SIZE], empty_signature.as_bytes())?;
		let single_key = oa::Matrix::from_slice(&engine, [PUBLIC_KEY_SIZE], kp.public_key.as_bytes())?;
		assert_eq!(
			verify_batch(
				&empty_bytes,
				&single_offset,
				&single_offset,
				&single_signature,
				&single_key
			)?
			.read::<u32>()?,
			[1]
		);
		Ok(())
	}
);

test_vk!(ml_dsa_65_gpu_verify_batch_rejects_bad_inputs, engine, {
	let empty_u8 = oa::Matrix::from_slice(&engine, [0_usize], &[] as &[u8])?;
	let offsets_u32 = oa::Matrix::from_slice(&engine, [1_usize], &[0_u32])?;
	let lengths_u32 = oa::Matrix::from_slice(&engine, [1_usize], &[0_u32])?;
	let sigs_u8 = oa::Matrix::from_slice(&engine, [SIGNATURE_SIZE], &vec![0_u8; SIGNATURE_SIZE])?;
	let pks_u8 = oa::Matrix::from_slice(&engine, [PUBLIC_KEY_SIZE], &vec![0_u8; PUBLIC_KEY_SIZE])?;

	// Wrong dtype for messages (F32 instead of U8).
	let f32_msg = oa::Matrix::from_f32(&engine, [1], &[0.0_f32])?;
	assert_eq!(
		verify_batch(&f32_msg, &offsets_u32, &lengths_u32, &sigs_u8, &pks_u8)
			.err()
			.expect("F32 messages accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);

	// Mismatched batch size between offsets and lengths.
	let two_offsets = oa::Matrix::from_slice(&engine, [2_usize], &[0_u32, 0_u32])?;
	assert_eq!(
		verify_batch(&empty_u8, &two_offsets, &lengths_u32, &sigs_u8, &pks_u8)
			.err()
			.expect("batch-size mismatch accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);

	// Signature buffer size mismatch.
	let bad_sigs = oa::Matrix::from_slice(&engine, [17_usize], &[0_u8; 17])?;
	assert_eq!(
		verify_batch(&empty_u8, &offsets_u32, &lengths_u32, &bad_sigs, &pks_u8)
			.err()
			.expect("wrong sig size accepted")
			.kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
});

/// Official external-interface cases qualify the independent host oracle used
/// for the device implementation, including nonempty contexts and malformed
/// signatures. This does not claim OA's empty-context API supports contexts.
#[test]
fn ml_dsa_65_host_oracle_matches_nist_acvp_external_pure_vectors() {
	use ml_dsa::{MlDsa65, Signature as OracleSignature, VerifyingKey};

	let fixture: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/acvp_mldsa65_sigver.json"
	))
	.unwrap();
	assert_eq!(fixture["parameterSet"], "ML-DSA-65");
	assert_eq!(fixture["signatureInterface"], "external");
	assert_eq!(fixture["preHash"], "pure");
	let tests = fixture["tests"].as_array().unwrap();
	assert_eq!(tests.len(), 15);
	let mut accepted = 0;
	for test in tests {
		let pk = parse_hex(test["pk"].as_str().unwrap());
		let signature = parse_hex(test["signature"].as_str().unwrap());
		let message = parse_hex(test["message"].as_str().unwrap());
		let context = parse_hex(test["context"].as_str().unwrap());
		let expected = test["testPassed"].as_bool().unwrap();
		let key_bytes: [u8; PUBLIC_KEY_SIZE] = pk.try_into().unwrap();
		let sig_bytes: [u8; SIGNATURE_SIZE] = signature.try_into().unwrap();
		let key = VerifyingKey::<MlDsa65>::decode((&key_bytes).into());
		let actual = OracleSignature::<MlDsa65>::decode((&sig_bytes).into())
			.is_some_and(|signature| key.verify_with_context(&message, &context, &signature));
		assert_eq!(actual, expected, "NIST ACVP tcId {}", test["tcId"]);
		if actual {
			accepted += 1;
		}
		if context.is_empty() {
			assert_eq!(
				verify(
					&message,
					&Signature::from_bytes(&sig_bytes).unwrap(),
					&oa::cryptography::pqc::PublicKey::from_bytes(&key_bytes).unwrap(),
				),
				expected,
				"OA empty-context API tcId {}",
				test["tcId"],
			);
		}
	}
	assert_eq!(accepted, 3);
}
