use oa::crypto::ntt::{NttConfig, NttRepresentation};

#[test]
fn rejects_invalid_field_root_and_length() {
	for (prime, root, length) in [
		(15, 2, 4),
		(17, 1, 4),
		(17, 3, 4),
		(17, 4, 3),
		(1_u64 << 62, 4, 4),
		(17, 4, 0),
	] {
		assert!(NttConfig::new(prime, root, length).is_err());
	}
	let config = NttConfig::new(17, 4, 4)
		.unwrap()
		.with_representation(NttRepresentation::Montgomery);
	assert_eq!(config.modulus(), 17);
	assert_eq!(config.root(), 4);
	assert_eq!(config.length(), 4);
	assert_eq!(config.representation(), NttRepresentation::Montgomery);
}

fn power_mod(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
	let mut result = 1_u64;
	base %= modulus;
	while exponent != 0 {
		if exponent & 1 == 1 {
			result = (u128::from(result) * u128::from(base) % u128::from(modulus)) as u64;
		}
		base = (u128::from(base) * u128::from(base) % u128::from(modulus)) as u64;
		exponent >>= 1;
	}
	result
}

fn primitive_root(modulus: u64) -> u64 {
	(2..modulus)
		.find(|&candidate| power_mod(candidate, (modulus - 1) / 2, modulus) == modulus - 1)
		.unwrap()
}

fn host_ntt(input: &[u64], root: u64, modulus: u64) -> Vec<u64> {
	let length = input.len();
	let mut output = input.to_vec();
	for index in 0..length {
		let reversed = index.reverse_bits() >> (64 - length.trailing_zeros());
		if reversed > index {
			output.swap(index, reversed);
		}
	}
	let mut span = 2_usize;
	while span <= length {
		let half = span / 2;
		for block in 0..length / span {
			for offset in 0..half {
				let left = block * span + offset;
				let right = left + half;
				let twiddle = power_mod(root, (offset * (length / span)) as u64, modulus);
				let b = (u128::from(output[right]) * u128::from(twiddle) % u128::from(modulus)) as u64;
				let a = output[left];
				output[left] = if a + b >= modulus {
					a + b - modulus
				} else {
					a + b
				};
				output[right] = if a >= b { a - b } else { modulus - (b - a) };
			}
		}
		span *= 2;
	}
	output
}

fn ntt_root(modulus: u64, length: usize) -> u64 {
	power_mod(
		primitive_root(modulus),
		(modulus - 1) / length as u64,
		modulus,
	)
}

test_vk!(ntt_forward_matches_host_oracle_and_roundtrips, engine, {
	use oa::crypto::ntt;
	let modulus = 786_433_u64; // 3 * 2^18 + 1
	for length in [16_usize, 256, 1024] {
		let config = NttConfig::new(modulus, ntt_root(modulus, length), length).unwrap();
		let parameters = ntt::prepare(&engine, &config)?;
		let rows: Vec<Vec<u64>> = (0..2)
			.map(|row| {
				(0..length)
					.map(|i| ((row * 97 + i * 31 + 7) as u64 * 6_553 + 13) % modulus)
					.collect()
			})
			.collect();
		let pair_words: Vec<u32> = rows
			.iter()
			.flatten()
			.flat_map(|value| [*value as u32, 0_u32])
			.collect();
		let pair = oa::Matrix::new_from_slice(&engine, [2, length, 2], &pair_words)?;
		let transformed = ntt::forward(&pair, &parameters)?;
		let recovered = ntt::inverse(&transformed, &parameters)?;
		let device: Vec<u32> = transformed.read()?;
		let restored: Vec<u32> = recovered.read()?;
		for (row_index, row) in rows.iter().enumerate() {
			let oracle = host_ntt(row, ntt_root(modulus, length), modulus);
			for i in 0..length {
				assert_eq!(
					device[row_index * length * 2 + i * 2] as u64,
					oracle[i],
					"forward mismatch at row {row_index} index {i} length {length}"
				);
				assert_eq!(
					restored[row_index * length * 2 + i * 2] as u64,
					row[i],
					"roundtrip mismatch at row {row_index} index {i} length {length}"
				);
			}
		}
	}
	Ok(())
});

test_vk!(ntt_scalar_storage_matches_pair_transform, engine, {
	use oa::crypto::ntt;
	let modulus = 786_433_u64;
	let length = 256_usize;
	let config = NttConfig::new(modulus, ntt_root(modulus, length), length).unwrap();
	let parameters = ntt::prepare(&engine, &config)?;
	let row: Vec<u64> = (0..length)
		.map(|i| ((i * 41 + 5) as u64 * 999_331 + 3) % modulus)
		.collect();
	let scalar_words: Vec<u32> = row.iter().map(|v| *v as u32).collect();
	let scalar = oa::Matrix::new_from_slice(&engine, [1, length], &scalar_words)?;
	let transformed = ntt::forward(&scalar, &parameters)?;
	let restored = ntt::inverse(&transformed, &parameters)?;
	let device: Vec<u32> = transformed.read()?;
	let oracle = host_ntt(&row, ntt_root(modulus, length), modulus);
	for (i, expected) in oracle.iter().enumerate() {
		assert_eq!(
			device[i] as u64, *expected,
			"scalar forward mismatch at {i}"
		);
	}
	assert_eq!(restored.read::<u32>()?, scalar_words);
	// Scalar and paired storage compute the same transform.
	let pair_words: Vec<u32> = row.iter().flat_map(|v| [*v as u32, 0_u32]).collect();
	let pair = oa::Matrix::new_from_slice(&engine, [1, length, 2], &pair_words)?;
	let pair_transformed = ntt::forward(&pair, &parameters)?.read::<u32>()?;
	for i in 0..length {
		assert_eq!(
			pair_transformed[i * 2],
			device[i],
			"pair/scalar mismatch at {i}"
		);
	}
	Ok(())
});

test_vk!(
	ntt_large_transform_roundtrips_through_staged_fused_route,
	engine,
	{
		use oa::crypto::ntt;
		let modulus = 786_433_u64;
		let length = 16_384_usize;
		let config = NttConfig::new(modulus, ntt_root(modulus, length), length).unwrap();
		let parameters = ntt::prepare(&engine, &config)?;
		let row: Vec<u64> = (0..length)
			.map(|i| ((i * 13 + 11) as u64 * 8_191 + 1) % modulus)
			.collect();
		let pair_words: Vec<u32> = row.iter().flat_map(|v| [*v as u32, 0_u32]).collect();
		let pair = oa::Matrix::new_from_slice(&engine, [1, length, 2], &pair_words)?;
		let transformed = ntt::forward(&pair, &parameters)?;
		let restored = ntt::inverse(&transformed, &parameters)?;
		assert_eq!(restored.read::<u32>()?, pair_words);
		// Full-length oracle spot check on selected coefficients.
		let device = transformed.read::<u32>()?;
		let oracle = host_ntt(&row, ntt_root(modulus, length), modulus);
		for &i in &[0_usize, 1, 37, 1024, 8191, 8192, 16383] {
			assert_eq!(
				device[i * 2] as u64,
				oracle[i],
				"fused-route mismatch at {i}"
			);
		}
		Ok(())
	}
);
