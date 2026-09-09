use oa::crypto::{
	Hash, Hasher, Shake128, Shake256, build_merkle_tree, hash, keccak_f1600, kmac256, merkle_proof,
	merkle_root, shake128, shake256, verify_merkle_proof,
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
