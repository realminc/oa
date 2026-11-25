//! General multi-algorithm device Merkle coverage with independent host oracles.
//! Rust port of test/py/crypto/test_merkle_general.py; the Python-side
//! Poseidon2 permute helper is reimplemented from the schema profiles.

use oa::Matrix;
use oa::crypto::merkle::{self, MerkleAlgorithm};

use crate::oracles::*;

const ALGORITHMS: [MerkleAlgorithm; 10] = [
	MerkleAlgorithm::Sha256,
	MerkleAlgorithm::Sha512,
	MerkleAlgorithm::Sha3_256,
	MerkleAlgorithm::Sha3_512,
	MerkleAlgorithm::Shake128,
	MerkleAlgorithm::Shake256,
	MerkleAlgorithm::Poseidon2BabyBear16,
	MerkleAlgorithm::Poseidon2BabyBear24,
	MerkleAlgorithm::Poseidon2KoalaBear16,
	MerkleAlgorithm::Poseidon2KoalaBear24,
];

fn is_poseidon(algorithm: MerkleAlgorithm) -> bool {
	matches!(
		algorithm,
		MerkleAlgorithm::Poseidon2BabyBear16
			| MerkleAlgorithm::Poseidon2BabyBear24
			| MerkleAlgorithm::Poseidon2KoalaBear16
			| MerkleAlgorithm::Poseidon2KoalaBear24
	)
}

fn parent(
	left: &[u8],
	right: &[u8],
	algorithm: MerkleAlgorithm,
	profiles: &serde_json::Value,
) -> Vec<u8> {
	if is_poseidon(algorithm) {
		let profile = poseidon_profile(profiles, algorithm);
		let to_words = |bytes: &[u8]| -> [u64; 8] {
			let mut words = [0_u64; 8];
			for (slot, word) in words.iter_mut().zip(bytes.as_chunks::<4>().0) {
				*slot = u64::from(u32::from_le_bytes(*word));
			}
			words
		};
		return poseidon2_parent(&to_words(left), &to_words(right), profile);
	}
	let joined: Vec<u8> = left.iter().chain(right.iter()).copied().collect();
	match algorithm {
		MerkleAlgorithm::Sha256 => sha256(&joined),
		MerkleAlgorithm::Sha512 => sha512(&joined),
		MerkleAlgorithm::Sha3_256 => keccak_sponge(136, 0x06, &joined, 32),
		MerkleAlgorithm::Sha3_512 => keccak_sponge(72, 0x06, &joined, 64),
		MerkleAlgorithm::Shake128 => keccak_sponge(168, 0x1f, &joined, 32),
		MerkleAlgorithm::Shake256 => keccak_sponge(136, 0x1f, &joined, 32),
		_ => unreachable!(),
	}
}

fn levels_of(leaves: &[Vec<u8>], algorithm: MerkleAlgorithm) -> Vec<Vec<Vec<u8>>> {
	let profiles = poseidon_profiles();
	let mut levels = vec![leaves.to_vec()];
	while levels.last().unwrap().len() > 1 {
		let row = levels.last().unwrap();
		levels.push(
			(0..row.len())
				.step_by(2)
				.map(|i| {
					parent(
						&row[i],
						&row[(i + 1).min(row.len() - 1)],
						algorithm,
						&profiles,
					)
				})
				.collect(),
		);
	}
	levels
}

fn leaf_bytes(algorithm: MerkleAlgorithm) -> usize {
	match algorithm {
		MerkleAlgorithm::Sha512 | MerkleAlgorithm::Sha3_512 => 64,
		_ => 32,
	}
}

fn make_leaves(count: usize, algorithm: MerkleAlgorithm) -> Vec<Vec<u8>> {
	if is_poseidon(algorithm) {
		let profiles = poseidon_profiles();
		let profile = poseidon_profile(&profiles, algorithm);
		let prime = profile["prime"].as_u64().unwrap();
		return (0..count)
			.map(|i| {
				(0..8)
					.map(|j| (((i * 937 + j * 91) as u64) % prime) as u32)
					.flat_map(u32::to_le_bytes)
					.collect()
			})
			.collect();
	}
	let bytes = leaf_bytes(algorithm);
	(0..count)
		.map(|i| (0..bytes).map(|j| ((i * 13 + j * 7) % 256) as u8).collect())
		.collect()
}

fn leaf_matrix(
	_runtime: &oa::Runtime,
	leaves: &[Vec<u8>],
	algorithm: MerkleAlgorithm,
) -> oa::Result<Matrix> {
	if is_poseidon(algorithm) {
		let words: Vec<u32> = leaves
			.iter()
			.flat_map(|row| {
				row
					.as_chunks::<4>()
					.0
					.iter()
					.map(|bytes| u32::from_le_bytes(*bytes))
			})
			.collect();
		Matrix::new_from_slice([leaves.len(), 8], &words)
	} else {
		let bytes: Vec<u8> = leaves.iter().flatten().copied().collect();
		Matrix::new_from_slice([leaves.len(), leaf_bytes(algorithm)], &bytes)
	}
}

fn read_digest(matrix: &Matrix, algorithm: MerkleAlgorithm) -> oa::Result<Vec<u8>> {
	if is_poseidon(algorithm) {
		Ok(
			matrix
				.read::<u32>()?
				.iter()
				.flat_map(|word| word.to_le_bytes())
				.collect(),
		)
	} else {
		matrix.read::<u8>()
	}
}

// ---------- Device tests ----------

test_vk!(
	merkle_general_roots_frames_and_fail_closed_verification,
	runtime,
	{
		for algorithm in ALGORITHMS {
			let bytes = leaf_bytes(algorithm);
			for count in [1_usize, 2, 5, 33] {
				let leaves = make_leaves(count, algorithm);
				let levels = levels_of(&leaves, algorithm);
				let input = leaf_matrix(&runtime, &leaves, algorithm)?;
				let tree = merkle::build(&input, algorithm)?;
				assert_eq!(
					read_digest(tree.root(), algorithm)?,
					levels.last().unwrap()[0],
					"root mismatch for {algorithm:?} count {count}"
				);
				let mut indices: Vec<u32> = (0..count as u32).collect();
				indices.extend([count as u32, u32::MAX]);
				let mut selected = leaves.clone();
				selected.push(leaves[0].clone());
				selected.push(leaves[0].clone());
				let selected_matrix = leaf_matrix(&runtime, &selected, algorithm)?;
				let index_matrix = Matrix::new_from_slice([indices.len()], &indices)?;
				let proofs = merkle::proofs(&tree, &index_matrix)?;
				let mut expected = Vec::new();
				for &index in &indices {
					let valid = index < count as u32;
					expected.extend_from_slice(&2_u32.to_le_bytes());
					expected.extend_from_slice(&(count as u32).to_le_bytes());
					expected.extend_from_slice(&index.to_le_bytes());
					expected.extend_from_slice(&(algorithm as u32).to_le_bytes());
					expected.extend_from_slice(&(valid as u32).to_le_bytes());
					let mut position = index as usize;
					for row in &levels[..levels.len() - 1] {
						if valid {
							expected.extend_from_slice(&row[(position ^ 1).min(row.len() - 1)]);
						} else {
							expected.extend(std::iter::repeat_n(0_u8, bytes));
						}
						position /= 2;
					}
				}
				assert_eq!(
					proofs.read::<u8>()?,
					expected,
					"proof frames mismatch for {algorithm:?} count {count}"
				);
				let flags = merkle::verify(
					&selected_matrix,
					&proofs,
					tree.root(),
					algorithm,
					count as u32,
				)?;
				let mut expected_flags = vec![1_u32; count];
				expected_flags.extend([0, 0]);
				assert_eq!(
					flags.read::<u32>()?,
					expected_flags,
					"verify flags mismatch for {algorithm:?} count {count}"
				);
				assert_eq!(
					read_digest(&input, algorithm)?,
					leaves.iter().flatten().copied().collect::<Vec<u8>>(),
					"input mutated for {algorithm:?}"
				);
				if count == 5 {
					let wrong_count = merkle::verify(&selected_matrix, &proofs, tree.root(), algorithm, 6)?;
					assert_eq!(
						wrong_count.read::<u32>()?,
						vec![0_u32; indices.len()],
						"wrong count must fail closed for {algorithm:?}"
					);
					let frame_bytes = expected.len() / indices.len();
					for position in [0_usize, 4, 8, 12, 16, 20] {
						let mut bad = expected.clone();
						bad[position] ^= 1;
						let bad_frames = Matrix::new_from_slice([indices.len(), frame_bytes], &bad)?;
						let result = merkle::verify(
							&selected_matrix,
							&bad_frames,
							tree.root(),
							algorithm,
							count as u32,
						)?;
						assert_eq!(
							result.read::<u32>()?[0],
							0,
							"corrupt frame at {position} must fail for {algorithm:?}"
						);
					}
				}
			}
		}
		Ok(())
	}
);

test_vk!(
	merkle_general_hierarchical_subtrees_preserve_the_root,
	runtime,
	{
		for algorithm in ALGORITHMS {
			let bytes = leaf_bytes(algorithm);
			let leaves: Vec<Vec<u8>> = if is_poseidon(algorithm) {
				(0..16)
					.map(|i| {
						(0..8)
							.flat_map(|j| ((i * 8 + j) as u32).to_le_bytes())
							.collect()
					})
					.collect()
			} else {
				(0..16)
					.map(|i| (0..bytes).map(|j| ((i + j) % 256) as u8).collect())
					.collect()
			};
			let input = leaf_matrix(&runtime, &leaves, algorithm)?;
			for width in [1_usize, 2, 4, 16] {
				let subtrees = merkle::subtree_roots(&input, algorithm, width)?;
				let composed = merkle::build(&subtrees, algorithm)?;
				assert_eq!(
					read_digest(composed.root(), algorithm)?,
					levels_of(&leaves, algorithm).last().unwrap()[0],
					"subtree width {width} broke root for {algorithm:?}"
				);
			}
		}
		Ok(())
	}
);

// ---------- Poseidon2 permutation and sponge coverage ----------

fn poseidon2_case_fixture() -> serde_json::Value {
	serde_json::from_str(include_str!(
		"../../fixtures/crypto/poseidon2/reference.json"
	))
	.unwrap()
}

#[test]
fn poseidon2_pinned_permutation_vectors_match_schema_oracle() {
	let profiles = poseidon_profiles();
	for case in poseidon2_case_fixture()["cases"].as_array().unwrap() {
		let profile = profiles
			.as_array()
			.unwrap()
			.iter()
			.find(|profile| profile["name"] == case["profile"])
			.unwrap();
		let input: Vec<u64> = case["input"]
			.as_array()
			.unwrap()
			.iter()
			.map(|v| v.as_u64().unwrap())
			.collect();
		let expected: Vec<u64> = case["output"]
			.as_array()
			.unwrap()
			.iter()
			.map(|v| v.as_u64().unwrap())
			.collect();
		assert_eq!(
			poseidon_permute(&input, profile),
			expected,
			"permute mismatch for {}",
			case["profile"]
		);
	}
}

test_vk!(poseidon2_permute_and_framed_sponge_match_oracle, runtime, {
	let profiles = poseidon_profiles();
	for case in poseidon2_case_fixture()["cases"].as_array().unwrap() {
		let values: Vec<u32> = case["input"]
			.as_array()
			.unwrap()
			.iter()
			.map(|v| v.as_u64().unwrap() as u32)
			.collect();
		let expected: Vec<u32> = case["output"]
			.as_array()
			.unwrap()
			.iter()
			.map(|v| v.as_u64().unwrap() as u32)
			.collect();
		let input = Matrix::new_from_slice([1, values.len()], &values)?;
		let permute = match case["profile"].as_str().unwrap() {
			"babybear16" => oa::crypto::hash::poseidon2_babybear16_permute(&input)?,
			"babybear24" => oa::crypto::hash::poseidon2_babybear24_permute(&input)?,
			"koalabear16" => oa::crypto::hash::poseidon2_koalabear16_permute(&input)?,
			_ => oa::crypto::hash::poseidon2_koalabear24_permute(&input)?,
		};
		assert_eq!(permute.read::<u32>()?, expected, "permute mismatch");
	}
	for profile in profiles.as_array().unwrap() {
		let name = profile["name"].as_str().unwrap();
		let width = profile["width"].as_u64().unwrap() as usize;
		let rate = width - profile["capacity"].as_u64().unwrap() as usize;
		for length in [0_usize, 1, rate - 1, rate, rate + 1, 2 * rate] {
			let rows = if length == rate + 1 { 33 } else { 1 };
			let values: Vec<u32> = (0..length * rows)
				.map(|i| (i as u32).wrapping_mul(2654435761))
				.collect();
			let input = Matrix::new_from_slice([rows, length], &values)?;
			for output_length in [8_usize, rate + 3] {
				let mut expected = Vec::new();
				for r in 0..rows {
					let row: Vec<u64> = values[r * length..(r + 1) * length]
						.iter()
						.map(|v| u64::from(*v))
						.collect();
					expected.extend(
						poseidon2_sponge(&row, profile, output_length)
							.iter()
							.map(|v| *v as u32),
					);
				}
				let output = match name {
					"babybear16" => oa::crypto::hash::poseidon2_babybear16(&input, output_length)?,
					"babybear24" => oa::crypto::hash::poseidon2_babybear24(&input, output_length)?,
					"koalabear16" => oa::crypto::hash::poseidon2_koalabear16(&input, output_length)?,
					_ => oa::crypto::hash::poseidon2_koalabear24(&input, output_length)?,
				};
				assert_eq!(
					output.read::<u32>()?,
					expected,
					"sponge mismatch for {name} length {length} out {output_length}"
				);
			}
			assert_eq!(input.read::<u32>()?, values, "input mutated for {name}");
		}
	}
	Ok(())
});
