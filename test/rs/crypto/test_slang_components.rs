//! Production Slang component oracles executed on Slang's host target.
//!
//! Host execution is a test laboratory, not an OA CPU fallback or GPU
//! qualification; the harness additionally pins both Vulkan SPIR-V targets.
//! Rust port of `test/py/crypto/kernel/*`.

use oa::crypto::keccak_f1600;

use crate::oracles::*;
use crate::slang_host::execute;

const POISON: u32 = 0xA5A5A5A5;
const Q: u64 = 8380417;
const N: usize = 256;
const MASK: u64 = 0xFFFF_FFFF;

fn words(data: &[u8]) -> Vec<u32> {
	data
		.as_chunks::<4>()
		.0
		.iter()
		.map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
		.collect()
}

/// Deterministic byte generator matching the Python tests' role: values are
/// arbitrary but reproducible, not entropy.
fn deterministic_bytes(seed: u64, count: usize) -> Vec<u8> {
	let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
	(0..count)
		.map(|_| {
			state ^= state << 13;
			state ^= state >> 7;
			state ^= state << 17;
			(state >> 24) as u8
		})
		.collect()
}

// ---------- Keccak ----------

#[test]
fn slang_keccak_permutation_matches_host_oracle() {
	let mut states = vec![vec![0_u8; 200], vec![255_u8; 200], (0..200).collect()];
	for bit in 0..1600_usize {
		let mut state = vec![0_u8; 200];
		state[bit / 8] = 1 << (bit % 8);
		states.push(state);
	}
	for round in 0..64_u64 {
		states.push(deterministic_bytes(20261003 + round, 200));
	}
	let mut inputs = Vec::new();
	let mut expected = Vec::new();
	for state in &states {
		inputs.extend(words(state));
		let mut lanes = [0_u64; 25];
		for (lane, chunk) in state.as_chunks::<8>().0.iter().enumerate() {
			lanes[lane] = u64::from_le_bytes(*chunk);
		}
		keccak_f1600(&mut lanes);
		for lane in lanes {
			expected.extend_from_slice(&[lane as u32, (lane >> 32) as u32]);
		}
	}
	assert_eq!(
		execute("keccak_components", &inputs, states.len() as u32, 50),
		expected
	);
}

// ---------- SHA-2 / SHA-3 digest framing ----------

/// Record shape shared by the sha2/sha3 digest components: a triple
/// (selector, base, length), alignment filler, the message, word padding and
/// optional trailing poison.
fn digest_record(selector: u32, message: &[u8], alignment: usize, exact_end: bool) -> Vec<u32> {
	let mut bytes = selector.to_le_bytes().to_vec();
	bytes.extend_from_slice(&(12 + alignment as u32).to_le_bytes());
	bytes.extend_from_slice(&(message.len() as u32).to_le_bytes());
	bytes.extend_from_slice(&vec![0xA5; alignment]);
	bytes.extend_from_slice(message);
	bytes.extend_from_slice(&vec![0xA5; (4 - bytes.len() % 4) % 4]);
	if !exact_end {
		bytes.extend_from_slice(&[0xA5; 16]);
	}
	words(&bytes)
}

fn sha2_assert_digest(bits: u32, message: &[u8], alignment: usize, exact_end: bool) {
	let expected = match bits {
		224 => sha224(message),
		_ => sha256(message),
	};
	let actual = execute(
		"sha2_u32_components",
		&digest_record(bits, message, alignment, exact_end),
		1,
		9,
	);
	let mut want = vec![1_u32];
	want.extend(words(&expected));
	want.extend(std::iter::repeat_n(POISON, 8 - (bits / 32) as usize));
	assert_eq!(actual, want, "sha{bits} alignment {alignment}");
}

#[test]
fn slang_sha2_digests_match_oracles() {
	for bits in [224_u32, 256] {
		for message in [b"".as_slice(), b"abc"] {
			sha2_assert_digest(bits, message, 0, true);
		}
		sha2_assert_digest(
			bits,
			b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
			1,
			false,
		);
		let lengths = [
			1, 3, 4, 7, 8, 54, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 191, 192, 1021,
		];
		for (index, length) in lengths.iter().enumerate() {
			let message = deterministic_bytes(20261004 + index as u64, *length);
			for alignment in 0..4 {
				sha2_assert_digest(bits, &message, alignment, false);
			}
		}
		for alignment in 0..4 {
			for length in [0_usize, 56 - alignment, 64 - alignment, 128 - alignment] {
				let message = vec![0x93; length];
				sha2_assert_digest(bits, &message, alignment, true);
			}
		}
	}
}

fn sha2_pair_assert_digest(bits: u32, message: &[u8], alignment: usize, exact_end: bool) {
	let expected = match bits {
		224 => sha512_224(message),
		256 => sha512_256(message),
		384 => sha384(message),
		_ => sha512(message),
	};
	let actual = execute(
		"sha2_u32_pair_components",
		&digest_record(bits, message, alignment, exact_end),
		1,
		17,
	);
	let mut want = vec![1_u32];
	want.extend(words(&expected));
	want.extend(std::iter::repeat_n(POISON, 16 - (bits / 32) as usize));
	assert_eq!(actual, want, "sha512-{bits} alignment {alignment}");
}

#[test]
fn slang_sha2_pair_digests_match_oracles() {
	for bits in [224_u32, 256, 384, 512] {
		for message in [b"".as_slice(), b"abc"] {
			sha2_pair_assert_digest(bits, message, 0, true);
		}
		sha2_pair_assert_digest(
			bits,
			b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
			1,
			false,
		);
		let rates = [64_usize; 4];
		let _ = rates;
		for (index, length) in [
			1_usize, 3, 55, 56, 63, 64, 65, 111, 112, 127, 128, 129, 255, 256, 1021,
		]
		.iter()
		.enumerate()
		{
			let message = deterministic_bytes(7777 + index as u64, *length);
			for alignment in 0..4 {
				sha2_pair_assert_digest(bits, &message, alignment, false);
			}
		}
	}
}

fn sha3_assert_digest(bits: u32, message: &[u8], alignment: usize, exact_end: bool) {
	let actual = execute(
		"sha3_components",
		&digest_record(bits, message, alignment, exact_end),
		1,
		17,
	);
	let mut want = vec![1_u32];
	want.extend(words(&sha3(bits as usize, message)));
	want.extend(std::iter::repeat_n(POISON, 16 - (bits / 32) as usize));
	assert_eq!(
		actual,
		want,
		"sha3-{bits} len {} alignment {alignment} exact {exact_end}",
		message.len()
	);
}

#[test]
fn slang_sha3_digests_match_oracles() {
	for bits in [224_u32, 256, 384, 512] {
		for message in [b"".as_slice(), b"abc"] {
			sha3_assert_digest(bits, message, 0, true);
			sha3_assert_digest(bits, message, 1, true);
		}
		let rate = 200 - 2 * (bits / 8) as usize;
		for (index, length) in [
			1,
			3,
			4,
			7,
			8,
			rate - 1,
			rate,
			rate + 1,
			2 * rate - 1,
			2 * rate,
			2 * rate + 1,
			3 * rate,
			1021,
		]
		.iter()
		.enumerate()
		{
			let message = deterministic_bytes(20201004 + index as u64, *length);
			for alignment in 0..4 {
				sha3_assert_digest(bits, &message, alignment, false);
			}
		}
		for alignment in 0..4 {
			for length in [rate - alignment, 2 * rate - alignment, 0] {
				let message = vec![0x93; length];
				sha3_assert_digest(bits, &message, alignment, true);
			}
		}
	}
}

#[test]
fn slang_sha3_invalid_ranges_fail_closed() {
	for bits in [224_u32, 256, 384, 512] {
		for (base, length) in [
			(17_u32, 0_u32),
			(16, 1),
			(12, 5),
			(0xFFFFFFFF, 4),
			(12, 0xFFFFFFFF),
		] {
			let actual = execute("sha3_components", &[bits, base, length, POISON], 1, 17);
			let mut want = vec![0_u32];
			want.extend(std::iter::repeat_n(POISON, 16));
			assert_eq!(actual, want);
		}
	}
	let mut cases: Vec<(u32, u32, u32)> = [0, 1, 8, 64, 137, 200, 0xFFFFFFFF]
		.iter()
		.map(|rate| (*rate, 12, 1))
		.collect();
	for rate in [144_u32, 136, 104, 72] {
		for (base, length) in [(17, 0), (16, 1), (12, 5), (0xFFFFFFFF, 1), (12, 0xFFFFFFFF)] {
			cases.push((rate, base, length));
		}
	}
	for (rate, base, length) in cases {
		let actual = execute(
			"sha3_range_components",
			&[rate, base, length, POISON],
			1,
			51,
		);
		let mut want = vec![0_u32];
		want.extend(std::iter::repeat_n(POISON, 50));
		assert_eq!(actual, want, "rate {rate} base {base} length {length}");
	}
}

// ---------- SHAKE streaming ----------

#[test]
fn slang_streaming_shake_matches_oracle() {
	let lengths = [0_usize, 1, 63, 64, 66, 135, 136, 137, 271, 272, 511, 1021];
	let mut records = Vec::new();
	let mut expected = Vec::new();
	for (i, first_length) in lengths.iter().enumerate() {
		for (j, second_length) in lengths.iter().enumerate() {
			let first = deterministic_bytes(202204 + i as u64, *first_length);
			let second = deterministic_bytes(902204 + j as u64, *second_length);
			let mut record = vec![0xA5_u8; 2056];
			record[..4].copy_from_slice(&(*first_length as u32).to_le_bytes());
			record[4..8].copy_from_slice(&(*second_length as u32).to_le_bytes());
			record[9..9 + first_length].copy_from_slice(&first);
			record[1035..1035 + second_length].copy_from_slice(&second);
			records.extend(words(&record));
			let mut message = first;
			message.extend_from_slice(&second);
			let mut digest = vec![0_u8; 64];
			oa::crypto::shake256(&message, &mut digest);
			expected.extend(words(&digest));
		}
	}
	assert_eq!(
		execute(
			"streaming_components",
			&records,
			(lengths.len() * lengths.len()) as u32,
			16
		),
		expected
	);
}

// ---------- ML-KEM field, compression, NTT ----------

#[test]
fn slang_mlkem_field_matches_modular_oracle() {
	const MODULUS: u64 = 3329;
	let mut pairs: Vec<(u64, u64)> = Vec::new();
	for a in [0_u64, 1, 2, 1664, 1665, 3327, 3328] {
		for b in [0_u64, 1, 2, 1664, 1665, 3327, 3328] {
			pairs.push((a, b));
		}
	}
	for a in 0..MODULUS {
		pairs.push((a, (a.wrapping_mul(1103515245) + 2033329) % MODULUS));
	}
	let inputs: Vec<u32> = pairs
		.iter()
		.flat_map(|(a, b)| [*a as u32, *b as u32])
		.collect();
	let expected: Vec<u32> = pairs
		.iter()
		.flat_map(|(a, b)| {
			let centered = if *a <= MODULUS / 2 {
				*a as i64
			} else {
				*a as i64 - MODULUS as i64
			};
			[
				((a + b) % MODULUS) as u32,
				((*a + MODULUS - b % MODULUS) % MODULUS) as u32,
				(a * b % MODULUS) as u32,
				((MODULUS - a % MODULUS) % MODULUS) as u32,
				(centered & MASK as i64) as u32,
			]
		})
		.collect();
	assert_eq!(
		execute("mlkem_field_components", &inputs, pairs.len() as u32, 5),
		expected
	);
}

#[test]
fn slang_mlkem_compression_matches_rational_oracle() {
	const MODULUS: u64 = 3329;
	fn round_ratio(numerator: u64, denominator: u64) -> u64 {
		numerator / denominator + u64::from(2 * (numerator % denominator) >= denominator)
	}
	let mut records = Vec::new();
	let mut expected = Vec::new();
	for bits in 1..12_u64 {
		let scale = 1_u64 << bits;
		for coefficient in 0..MODULUS {
			let encoded = coefficient % scale;
			let compressed = round_ratio(coefficient * scale, MODULUS) % scale;
			let decompressed = round_ratio(encoded * MODULUS, scale);
			let restored = round_ratio(compressed * MODULUS, scale);
			records.extend_from_slice(&[coefficient as u32, bits as u32, encoded as u32]);
			expected.extend_from_slice(&[
				compressed as u32,
				decompressed as u32,
				restored as u32,
				encoded as u32,
			]);
		}
	}
	assert_eq!(
		execute(
			"mlkem_compression_components",
			&records,
			(records.len() / 3) as u32,
			4
		),
		expected
	);
}

#[test]
fn slang_mlkem_ntt_matches_evaluation_and_ring_product() {
	const MODULUS: u64 = 3329;
	let zero = vec![0_u64; 256];
	let one = {
		let mut v = vec![0_u64; 256];
		v[0] = 1;
		v
	};
	let highest = {
		let mut v = vec![0_u64; 256];
		v[255] = 1;
		v
	};
	let mut pairs = vec![
		(zero.clone(), zero.clone()),
		(one.clone(), highest.clone()),
		(highest.clone(), highest.clone()),
		(vec![MODULUS - 1; 256], vec![MODULUS - 1; 256]),
	];
	for round in 0..12_u64 {
		let make = |seed: u64| -> Vec<u64> {
			let mut state = seed + 20317;
			(0..256)
				.map(|_| {
					state ^= state << 13;
					state ^= state >> 7;
					state ^= state << 17;
					state % MODULUS
				})
				.collect()
		};
		pairs.push((make(round * 2), make(round * 2 + 1)));
	}
	let mut records = Vec::new();
	let mut expected = Vec::new();
	for (left, right) in &pairs {
		let mut evaluations = Vec::with_capacity(256);
		for pair in 0..128_u64 {
			let reversed = pair.reverse_bits() >> (64 - 7);
			let mut gamma = 1_u64;
			let mut base = 17_u64;
			let mut exp = 2 * reversed + 1;
			while exp > 0 {
				if exp & 1 == 1 {
					gamma = gamma * base % MODULUS;
				}
				base = base * base % MODULUS;
				exp >>= 1;
			}
			for parity in 0..2_usize {
				let mut value = 0_u64;
				for index in (0..128).rev() {
					value = (value * gamma + left[2 * index + parity]) % MODULUS;
				}
				evaluations.push(value);
			}
		}
		let mut product = vec![0_i64; 256];
		for (i, a) in left.iter().enumerate() {
			for (j, b) in right.iter().enumerate() {
				product[(i + j) % 256] += (*a * *b) as i64 * if i + j < 256 { 1 } else { -1 };
			}
		}
		records.extend(left.iter().chain(right.iter()).map(|v| *v as u32));
		expected.extend(evaluations.iter().map(|v| *v as u32));
		expected.extend(left.iter().map(|v| *v as u32));
		expected.extend(product.iter().map(|v| v.rem_euclid(MODULUS as i64) as u32));
	}
	assert_eq!(
		execute(
			"mlkem_polynomial_components",
			&records,
			pairs.len() as u32,
			768
		),
		expected
	);
}

// ---------- ML-KEM packing / sampling ----------

#[test]
fn slang_mlkem_packing_matches_bit_oracle() {
	let mut values = Vec::new();
	let mut expected = Vec::new();
	let mut random = 20356_u64;
	let mut next = |modulus: u64| -> u64 {
		random ^= random << 13;
		random ^= random >> 7;
		random ^= random << 17;
		random % modulus
	};
	for bits in 1..13_usize {
		let modulus = if bits == 12 { 3329_u64 } else { 1 << bits };
		for case in 0..6_usize {
			let poly: Vec<u64> = match case {
				0 => vec![0; 256],
				1 => vec![modulus - 1; 256],
				_ => (0..256).map(|_| next(modulus)).collect(),
			};
			let mut encoded = vec![0_u8; 32 * bits];
			for (i, value) in poly.iter().enumerate() {
				for k in 0..bits {
					if (*value >> k) & 1 == 1 {
						encoded[(i * bits + k) / 8] |= 1 << ((i * bits + k) % 8);
					}
				}
			}
			let raw: Vec<u8> = if case < 3 {
				encoded.clone()
			} else if bits == 12 && case == 5 {
				let alternating: Vec<u64> = (0..256)
					.map(|i| if i % 2 == 0 { 3329 } else { 4095 })
					.collect();
				let mut packed = vec![0_u8; 384];
				for (i, value) in alternating.iter().enumerate() {
					for k in 0..12_usize {
						if (*value >> k) & 1 == 1 {
							packed[(i * 12 + k) / 8] |= 1 << ((i * 12 + k) % 8);
						}
					}
				}
				packed
			} else {
				(0..32 * bits).map(|_| next(256) as u8).collect()
			};
			let decoded: Vec<u64> = (0..256)
				.map(|i| {
					let mut value = 0_u64;
					for k in 0..bits {
						let bit = i * bits + k;
						if bit / 8 < raw.len() && (raw[bit / 8] >> (bit % 8)) & 1 == 1 {
							value |= 1 << k;
						}
					}
					value
				})
				.collect();
			let canonical = u32::from(bits < 12 || decoded.iter().all(|x| *x < 3329));
			values.push(bits as u32);
			values.extend(poly.iter().map(|v| *v as u32));
			values.extend(raw.iter().map(|b| *b as u32));
			values.extend(std::iter::repeat_n(0xCC_u32, 384 - raw.len()));
			expected.extend(encoded.iter().map(|b| *b as u32));
			expected.extend(std::iter::repeat_n(0xA5_u32, 384 - encoded.len()));
			expected.extend(decoded.iter().map(|x| (*x % modulus) as u32));
			expected.push(canonical);
		}
	}
	for offset in (0_u64..4096).step_by(256) {
		let mut raw = vec![0_u8; 384];
		for i in 0..256_usize {
			for k in 0..12_usize {
				if ((offset + i as u64) >> k) & 1 == 1 {
					raw[(i * 12 + k) / 8] |= 1 << ((i * 12 + k) % 8);
				}
			}
		}
		values.push(12);
		values.extend(std::iter::repeat_n(0_u32, 256));
		values.extend(raw.iter().map(|b| *b as u32));
		expected.extend(std::iter::repeat_n(0_u32, 384));
		expected.extend((0..256).map(|i| ((offset + i as u64) % 3329) as u32));
		expected.push(u32::from(offset + 255 < 3329));
	}
	assert_eq!(
		execute(
			"mlkem_packing_components",
			&values,
			(values.len() / 641) as u32,
			641
		),
		expected
	);
}

#[test]
fn slang_mlkem_sampling_and_rejection_match_stream_oracles() {
	const MODULUS: u64 = 3329;
	let mut values = Vec::new();
	let mut expected = Vec::new();
	for case in 0..32_u32 {
		let seed = match case {
			0 => vec![0_u8; 32],
			1 => vec![255_u8; 32],
			_ => deterministic_bytes(20378 + case as u64, 32),
		};
		let (mut first, mut second, mut nonce) = (
			((case * 13) % 256) as u8,
			((case * 29) % 256) as u8,
			((case * 31) % 256) as u8,
		);
		if case == 1 {
			first = 255;
			second = 255;
			nonce = 255;
		}
		let eta = 2 + (case % 2) as usize;
		let mut seed_ext = seed.clone();
		seed_ext.extend_from_slice(&[first, second]);
		let mut stream = vec![0_u8; 4096];
		oa::crypto::shake128(&seed_ext, &mut stream);
		let mut matrix = Vec::new();
		for chunk in stream.as_chunks::<3>().0 {
			let pair = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0]);
			for value in [pair % 4096, pair / 4096] {
				if (value as u64) < MODULUS {
					matrix.push(value);
				}
			}
			if matrix.len() >= 256 {
				break;
			}
		}
		let mut noise_input = seed.clone();
		noise_input.push(nonce);
		let mut noise_bytes = vec![0_u8; 64 * eta];
		oa::crypto::shake256(&noise_input, &mut noise_bytes);
		let mask = (1_u64 << eta) - 1;
		let noise: Vec<u64> = (0..256)
			.map(|i| {
				let bit = 2 * eta * i;
				let byte = bit / 8;
				let mut chunk = 0_u64;
				for k in 0..(2 * eta).div_ceil(8) + 1 {
					if byte + k < noise_bytes.len() {
						chunk |= (noise_bytes[byte + k] as u64) << (8 * k);
					}
				}
				chunk >>= bit % 8;
				((chunk & mask).count_ones() as i64 - ((chunk >> eta) & mask).count_ones() as i64)
					.rem_euclid(MODULUS as i64) as u64
			})
			.collect();
		values.extend(words(&seed));
		values.extend_from_slice(&[first as u32, second as u32, eta as u32, nonce as u32]);
		expected.extend_from_slice(&matrix[..256]);
		expected.extend(noise.iter().map(|v| *v as u32));
	}
	assert_eq!(
		execute("mlkem_sampling_components", &values, 32, 512),
		expected
	);

	let mut values = Vec::new();
	let mut expected = Vec::new();
	for first in [0_u32, 1, 3328, 3329, 4095] {
		for second in [0_u32, 1, 3328, 3329, 4095] {
			for count in [0_u32, 254, 255, 256] {
				values.extend_from_slice(&[first, second, count]);
				let mut poly = vec![POISON; 256];
				let mut cursor = count;
				for value in [first, second] {
					if value < 3329 && cursor < 256 {
						poly[cursor as usize] = value;
						cursor += 1;
					}
				}
				expected.extend(poly);
				expected.push(cursor);
			}
		}
	}
	assert_eq!(
		execute("mlkem_rejection_components", &values, 100, 257),
		expected
	);
}

// ---------- ML-DSA field / NTT ----------

fn decompose(value: u64, gamma2: u64) -> (u64, i64) {
	let high = (value + gamma2 - 1) / (2 * gamma2);
	let low = value as i64 - (high * 2 * gamma2) as i64;
	if high * 2 * gamma2 == Q - 1 {
		(0, low - 1)
	} else {
		(high, low)
	}
}

#[test]
fn slang_mldsa_wide_multiply_and_field() {
	let boundaries = [0_u64, 1, 65535, 65536, 0x7FFFFFFF, 0x80000000, MASK];
	let mut pairs: Vec<(u64, u64)> = Vec::new();
	for a in boundaries {
		for b in boundaries {
			pairs.push((a, b));
		}
	}
	let mut random = 204_u64;
	for _ in 0..4096 {
		let mut next = || {
			random ^= random << 13;
			random ^= random >> 7;
			random ^= random << 17;
			random & MASK
		};
		pairs.push((next(), next()));
	}
	let inputs: Vec<u32> = pairs
		.iter()
		.flat_map(|(a, b)| [*a as u32, *b as u32])
		.collect();
	let expected: Vec<u32> = pairs
		.iter()
		.flat_map(|(a, b)| [((a * b) & MASK) as u32, ((a * b) >> 32) as u32])
		.collect();
	assert_eq!(
		execute("wide_components", &inputs, pairs.len() as u32, 2),
		expected
	);

	let mut random = 203204_u64;
	let mut next = |modulus: u64| {
		random ^= random << 13;
		random ^= random >> 7;
		random ^= random << 17;
		random % modulus
	};
	let mut cases = Vec::new();
	for gamma2 in [(Q - 1) / 88, (Q - 1) / 32] {
		let mut values =
			std::collections::BTreeSet::from([0_u64, 1, 4095, 4096, 4097, Q / 2, Q - 2, Q - 1]);
		for high in 0..=(Q - 1) / (2 * gamma2) {
			for offset in [
				-(gamma2 as i64) - 1,
				-(gamma2 as i64),
				-(gamma2 as i64) + 1,
				-1,
				0,
				1,
				gamma2 as i64 - 1,
				gamma2 as i64,
				gamma2 as i64 + 1,
			] {
				values.insert(((high * 2 * gamma2) as i64 + offset).rem_euclid(Q as i64) as u64);
			}
		}
		for a in values {
			for b in [0_u64, 1, Q - 1, next(Q)] {
				for hint in 0_u64..2 {
					cases.push((a, b, gamma2, hint));
				}
			}
		}
		for _ in 0..4096 {
			cases.push((next(Q), next(Q), gamma2, next(2)));
		}
	}
	let inputs: Vec<u32> = cases
		.iter()
		.flat_map(|(a, b, g, h)| [*a as u32, *b as u32, *g as u32, *h as u32])
		.collect();
	let expected: Vec<u32> = cases
		.iter()
		.flat_map(|(a, b, gamma2, hint)| {
			let (high, low) = decompose(*a, *gamma2);
			let round_high = (a + 4095) / 8192;
			let (other_high, _) = decompose((a + b) % Q, *gamma2);
			let use_hint = if *hint != 0 {
				(high as i64 + if low > 0 { 1 } else { -1 }).rem_euclid(((Q - 1) / (2 * gamma2)) as i64)
					as u64
			} else {
				high
			};
			[
				(a * b % Q) as u32,
				*a as u32,
				((a << 32) % Q) as u32,
				((a + b) % Q) as u32,
				((*a + Q - b % Q) % Q) as u32,
				((Q - a % Q) % Q) as u32,
				(if *a <= Q / 2 {
					*a as i64
				} else {
					*a as i64 - Q as i64
				} & MASK as i64) as u32,
				round_high as u32,
				((*a as i64 - (round_high * 8192) as i64) & MASK as i64) as u32,
				high as u32,
				(low & MASK as i64) as u32,
				u32::from(high != other_high),
				use_hint as u32,
				*b as u32,
			]
		})
		.collect();
	assert_eq!(
		execute("field_components", &inputs, cases.len() as u32, 14),
		expected
	);
}

fn evaluate_ntt(poly: &[u64]) -> Vec<u64> {
	(0..N)
		.map(|i| {
			let reversed = (i as u32).reverse_bits() >> 24;
			let mut root = 1_u64;
			let mut base = 1753_u64;
			let mut exp = 2 * reversed as u64 + 1;
			while exp > 0 {
				if exp & 1 == 1 {
					root = root * base % Q;
				}
				base = base * base % Q;
				exp >>= 1;
			}
			poly.iter().rev().fold(0_u64, |v, c| (v * root + c) % Q)
		})
		.collect()
}

fn negacyclic_product(a: &[u64], b: &[u64]) -> Vec<u64> {
	let mut result = vec![0_i64; N];
	for (i, left) in a.iter().enumerate() {
		for (j, right) in b.iter().enumerate() {
			result[(i + j) % N] += (*left * *right) as i64 * if i + j < N { 1 } else { -1 };
		}
	}
	result
		.iter()
		.map(|v| v.rem_euclid(Q as i64) as u64)
		.collect()
}

#[test]
fn slang_mldsa_ntt_matches_evaluation_oracle() {
	let mut random = 1753_u64;
	let mut next = |modulus: u64| {
		random ^= random << 13;
		random ^= random >> 7;
		random ^= random << 17;
		random % modulus
	};
	let mut polynomials = vec![
		vec![0_u64; N],
		vec![Q - 1; N],
		{
			let mut v = vec![0_u64; N];
			v[0] = 1;
			v
		},
		{
			let mut v = vec![0_u64; N];
			v[N - 1] = 1;
			v
		},
		(0..N).map(|i| (i % 2) as u64 * (Q - 1)).collect(),
	];
	for _ in 0..8 {
		polynomials.push((0..N).map(|_| next(Q)).collect());
	}
	for (i, a) in polynomials.iter().enumerate() {
		let b = &polynomials[(i + 1) % polynomials.len()];
		let inputs: Vec<u32> = a.iter().chain(b.iter()).map(|v| *v as u32).collect();
		let actual = execute("polynomial_components", &inputs, 1, 1024);
		let transformed = evaluate_ntt(a);
		assert_eq!(
			actual[..N],
			transformed.iter().map(|v| *v as u32).collect::<Vec<_>>()[..]
		);
		assert_eq!(
			actual[N..2 * N],
			a.iter().map(|v| *v as u32).collect::<Vec<_>>()[..]
		);
		assert_eq!(
			actual[2 * N..3 * N],
			negacyclic_product(a, b)
				.iter()
				.map(|v| *v as u32)
				.collect::<Vec<_>>()[..]
		);
		assert_eq!(
			actual[3 * N..],
			transformed
				.iter()
				.map(|v| (8192 * v % Q) as u32)
				.collect::<Vec<_>>()[..]
		);
	}
}

// ---------- ML-DSA bit packing + signature decoding ----------

fn unpack_bits(data: &[u8], bits: usize) -> Vec<u64> {
	(0..256)
		.map(|i| {
			let mut value = 0_u64;
			for k in 0..bits {
				let bit = i * bits + k;
				if bit / 8 < data.len() && (data[bit / 8] >> (bit % 8)) & 1 == 1 {
					value |= 1 << k;
				}
			}
			value
		})
		.collect()
}

#[test]
fn slang_mldsa_bit_packing_matches_oracle() {
	let mut records = Vec::new();
	let mut expected = Vec::new();
	let mut random = 20423_u64;
	for bits in 1..21_usize {
		for offset in 0..4_usize {
			let mut coefficients = vec![0_u64, (1 << bits) - 1];
			for _ in 0..254 {
				random ^= random << 13;
				random ^= random >> 7;
				random ^= random << 17;
				coefficients.push(random % (1 << bits));
			}
			let mut payload = vec![0_u8; 32 * bits];
			for (i, value) in coefficients.iter().enumerate() {
				let packed = (*value as u128) << ((i * bits) % 8);
				for (k, byte) in packed.to_le_bytes().iter().enumerate() {
					if (i * bits) / 8 + k < payload.len() {
						payload[(i * bits) / 8 + k] |= byte;
					}
				}
			}
			let mut record = vec![0xA5_u8; 652];
			record[..4].copy_from_slice(&(bits as u32).to_le_bytes());
			record[4..8].copy_from_slice(&(offset as u32).to_le_bytes());
			record[8 + offset..8 + offset + payload.len()].copy_from_slice(&payload);
			records.extend(words(&record));
			expected.extend(coefficients.iter().map(|v| *v as u32));
		}
	}
	assert_eq!(
		execute("bit_packing_components", &records, 80, 256),
		expected
	);
}

#[test]
fn slang_mldsa_nist_decoding_and_malformed_hints() {
	let fixture: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/crypto/acvp_mldsa65_sigver.json"
	))
	.unwrap();
	let mut cases: Vec<(Vec<u8>, Vec<u8>)> = fixture["tests"]
		.as_array()
		.unwrap()
		.iter()
		.map(|case| {
			(
				hex_decode(case["pk"].as_str().unwrap()),
				hex_decode(case["signature"].as_str().unwrap()),
			)
		})
		.collect();
	let pk = cases[0].0.clone();
	let hints: Vec<Vec<u8>> = vec![
		vec![0; 61],
		[vec![255, 0], vec![0; 53], vec![1, 2, 2, 2, 2, 2]].concat(),
		[vec![7, 7], vec![0; 53], vec![2; 6]].concat(),
		[vec![9, 8], vec![0; 53], vec![2; 6]].concat(),
		[vec![0; 55], vec![1, 0, 0, 0, 0, 0]].concat(),
		[vec![0; 55], vec![56; 6]].concat(),
		[vec![1], vec![0; 60]].concat(),
		(0..55).chain([55; 6]).collect(),
	];
	for hint in hints {
		cases.push((pk.clone(), [vec![0; 3248], hint].concat()));
	}
	for (case_index, (pk, signature)) in cases.iter().enumerate() {
		let rho: Vec<u32> = words(&pk[..32]);
		let t1: Vec<u64> = (0..6)
			.flat_map(|k| unpack_bits(&pk[32 + k * 320..32 + (k + 1) * 320], 10))
			.collect();
		let challenge: Vec<u32> = words(&signature[..48]);
		let z: Vec<u64> = (0..5)
			.flat_map(|l| unpack_bits(&signature[48 + l * 640..48 + (l + 1) * 640], 20))
			.map(|v| (Q + 524288 - (v % Q)) % Q)
			.collect();
		let hint = &signature[3248..];
		let indices = &hint[..55];
		let delimiters = &hint[55..];
		let mut decoded = Vec::new();
		let mut previous = 0_usize;
		let mut valid = true;
		for end in delimiters {
			let end = *end as usize;
			let positions: Vec<u8> = if end >= previous {
				indices[previous.min(55)..end.min(55)].to_vec()
			} else {
				Vec::new()
			};
			let unique_sorted = {
				let mut sorted = positions.clone();
				sorted.sort_unstable();
				sorted.dedup();
				sorted == positions
			};
			valid &= previous <= end && end <= 55 && unique_sorted;
			for i in 0..256_usize {
				decoded.push(u32::from(positions.contains(&(i as u8))));
			}
			previous = end;
		}
		if previous <= 55 {
			valid &= !indices[previous..].iter().any(|v| *v != 0);
		}
		let norm = z.iter().all(|v| (*v).min(Q - v) < 524092);
		for offset in 0..4_usize {
			let mut record = vec![0xA5_u8; 5268];
			record[..4].copy_from_slice(&(offset as u32).to_le_bytes());
			record[4 + offset..4 + offset + 5261]
				.copy_from_slice(&[pk.as_slice(), signature.as_slice()].concat());
			let actual = execute("packing_components", &words(&record), 1, 4374);
			let want_head: Vec<u32> = rho
				.iter()
				.copied()
				.chain(t1.iter().map(|v| *v as u32))
				.chain(challenge.iter().copied())
				.chain(z.iter().map(|v| *v as u32))
				.collect();
			assert_eq!(
				actual[..2836],
				want_head[..],
				"case {case_index} offset {offset} header"
			);
			let expected_flags = [u32::from(valid), u32::from(norm)];
			assert_eq!(
				actual[4374 - 2..],
				expected_flags,
				"case {case_index} offset {offset} flags"
			);
			if valid {
				assert_eq!(
					actual[2836..4374 - 2],
					decoded[..],
					"case {case_index} offset {offset} hints"
				);
			} else {
				assert!(
					actual[2836..4374 - 2].iter().all(|v| *v <= 1),
					"case {case_index} exposed uninitialized coefficients"
				);
			}
		}
	}
}

fn hex_decode(text: &str) -> Vec<u8> {
	(0..text.len() / 2)
		.map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap())
		.collect()
}

// ---------- ML-DSA sampling (ball + uniform) ----------

#[test]
fn slang_mldsa_sampling_matches_stream_oracles() {
	let fixture: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/crypto/acvp_mldsa65_sigver.json"
	))
	.unwrap();
	let mut seeds: Vec<(Vec<u8>, Vec<u8>)> = fixture["tests"]
		.as_array()
		.unwrap()
		.iter()
		.map(|case| {
			(
				hex_decode(&case["signature"].as_str().unwrap()[..96]),
				hex_decode(&case["pk"].as_str().unwrap()[..64]),
			)
		})
		.collect();
	seeds.push((vec![0; 48], vec![0; 32]));
	seeds.push((vec![255; 48], vec![255; 32]));
	for case in 0..32_u64 {
		seeds.push((
			deterministic_bytes(2042932 + case * 2, 48),
			deterministic_bytes(3042932 + case * 2, 32),
		));
	}
	for (case_index, (challenge, rho)) in seeds.iter().enumerate() {
		let mut stream = vec![0_u8; 4096];
		oa::crypto::shake256(challenge, &mut stream);
		let signs = u64::from_le_bytes(stream[..8].try_into().unwrap());
		let mut c = vec![0_u64; 256];
		let mut pos = 8_usize;
		for i in 207..256 {
			while stream[pos] as usize > i {
				pos += 1;
			}
			let j = stream[pos] as usize;
			pos += 1;
			c[i] = c[j];
			c[j] = if (signs >> (i - 207)) & 1 == 1 {
				Q - 1
			} else {
				1
			};
		}
		let (row, column) = (case_index % 6, case_index % 5);
		let mut input = rho.clone();
		input.extend_from_slice(&[column as u8, row as u8]);
		oa::crypto::shake128(&input, &mut stream);
		let mut a = Vec::new();
		for chunk in stream.as_chunks::<3>().0 {
			let coefficient = u64::from(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0]) & 0x7FFFFF);
			if coefficient < Q {
				a.push(coefficient);
			}
			if a.len() == 256 {
				break;
			}
		}
		assert_eq!(a.len(), 256);
		let values: Vec<u32> = words(challenge)
			.into_iter()
			.chain(words(rho))
			.chain([row as u32, column as u32])
			.collect();
		let actual = execute("sampling_components", &values, 1, 512);
		assert_eq!(
			actual[..256],
			c.iter().map(|v| *v as u32).collect::<Vec<_>>()[..]
		);
		assert_eq!(
			actual[256..],
			a.iter().map(|v| *v as u32).collect::<Vec<_>>()[..]
		);
		assert_eq!(actual[..256].iter().filter(|v| **v != 0).count(), 49);
	}
}

// ---------- ML-DSA prehash digest composition ----------

fn prehash_digest(algorithm: u32, message: &[u8]) -> Vec<u8> {
	match algorithm {
		1 => sha256(message),
		2 => sha384(message),
		3 => sha512(message),
		4 => sha224(message),
		5 => sha512_224(message),
		6 => sha512_256(message),
		7 => sha3(224, message),
		8 => sha3(256, message),
		9 => sha3(384, message),
		10 => sha3(512, message),
		11 => shake(128, message, 32),
		_ => shake(256, message, 64),
	}
}

fn prehash_check(algorithm: u32, message: &[u8], context: &[u8], alignment: usize) {
	let tr: Vec<u8> = (0..64).collect();
	let context_base = 24 + tr.len() + alignment;
	let base = context_base + context.len();
	let mut record = Vec::new();
	for value in [
		algorithm,
		base as u32,
		message.len() as u32,
		24,
		context_base as u32,
		context.len() as u32,
	] {
		record.extend_from_slice(&value.to_le_bytes());
	}
	record.extend_from_slice(&tr);
	record.extend_from_slice(&vec![0xA5; alignment]);
	record.extend_from_slice(context);
	record.extend_from_slice(message);
	record.extend_from_slice(&vec![0xA5; (4 - record.len() % 4) % 4]);
	let digest = prehash_digest(algorithm, message);
	let oid = [6, 9, 96, 134, 72, 1, 101, 3, 4, 2, algorithm as u8];
	let mut mu_input = tr.clone();
	mu_input.extend_from_slice(&[1, context.len() as u8]);
	mu_input.extend_from_slice(context);
	mu_input.extend_from_slice(&oid);
	mu_input.extend_from_slice(&digest);
	let mut mu = vec![0_u8; 64];
	oa::crypto::shake256(&mu_input, &mut mu);
	let actual = execute("mldsa_prehash_digest_components", &words(&record), 1, 34);
	let mut want = vec![1_u32, 1];
	want.extend(words(&digest));
	want.extend(std::iter::repeat_n(POISON, 16 - digest.len() / 4));
	want.extend(words(&mu));
	assert_eq!(actual, want, "algorithm {algorithm}");
}

#[test]
fn slang_mldsa_prehash_digest_composition() {
	for algorithm in 1..=12_u32 {
		for (index, size) in [
			0_usize, 1, 55, 56, 63, 64, 111, 112, 127, 128, 135, 136, 143, 144, 167, 168, 169, 336, 1021,
		]
		.iter()
		.enumerate()
		{
			let message = deterministic_bytes(20261006 + index as u64, *size);
			for alignment in 0..4 {
				prehash_check(algorithm, &message, b"", alignment);
			}
		}
		for size in [1_usize, 127, 128, 255] {
			let context: Vec<u8> = (0..size).map(|i| (i & 255) as u8).collect();
			prehash_check(algorithm, b"message\x00\xff", &context, 3);
		}
	}
}

#[test]
fn slang_mldsa_prehash_invalid_ranges_fail_closed() {
	for algorithm in 1..=12_u32 {
		for (base, length) in [
			(101_u32, 0_u32),
			(100, 1),
			(96, 5),
			(0xFFFFFFFF, 4),
			(96, 0xFFFFFFFF),
		] {
			let record: Vec<u32> = [algorithm, base, length, 24, 88, 0]
				.into_iter()
				.chain(std::iter::repeat_n(POISON, 19))
				.collect();
			let actual = execute("mldsa_prehash_digest_components", &record, 1, 34);
			let mut want = vec![0_u32, 0];
			want.extend(std::iter::repeat_n(POISON, 32));
			assert_eq!(actual, want);
		}
	}
	for algorithm in [0_u32, 13, 255, 0xFFFFFFFF] {
		let record: Vec<u32> = [algorithm, 88, 0, 24, 88, 0]
			.into_iter()
			.chain(std::iter::repeat_n(POISON, 16))
			.collect();
		let actual = execute("mldsa_prehash_digest_components", &record, 1, 34);
		let mut want = vec![0_u32, 0];
		want.extend(std::iter::repeat_n(POISON, 32));
		assert_eq!(actual, want);
	}
	for algorithm in 1..=12_u32 {
		let digest = prehash_digest(algorithm, b"");
		for (tr_base, context_base, context_length) in [
			(37_u32, 88_u32, 0_u32),
			(0xFFFFFFFF, 88, 0),
			(24, 101, 0),
			(24, 100, 1),
			(24, 96, 5),
			(24, 88, 256),
			(24, 88, 0xFFFFFFFF),
		] {
			let record: Vec<u32> = [algorithm, 100, 0, tr_base, context_base, context_length]
				.into_iter()
				.chain(std::iter::repeat_n(POISON, 19))
				.collect();
			let actual = execute("mldsa_prehash_digest_components", &record, 1, 34);
			let mut want = vec![1_u32, 0];
			want.extend(words(&digest));
			want.extend(std::iter::repeat_n(POISON, 16 - digest.len() / 4));
			want.extend(std::iter::repeat_n(POISON, 16));
			assert_eq!(actual, want);
		}
	}
}

// ---------- Bigint ----------

/// Fixed 128-limb little-endian oracle helpers for the bigint components.
fn limbs_of(value: &[u32]) -> Vec<u32> {
	let mut out = vec![0_u32; 128];
	out[..value.len().min(128)].copy_from_slice(&value[..value.len().min(128)]);
	out
}

fn big_leading_zeros(limbs: &[u32], word_count: usize) -> usize {
	let mut zeros = 0;
	for index in (0..word_count).rev() {
		if limbs[index] == 0 {
			zeros += 32;
		} else {
			return zeros + limbs[index].leading_zeros() as usize;
		}
	}
	zeros
}

fn big_cmp(a: &[u32], b: &[u32], word_count: usize) -> std::cmp::Ordering {
	for index in (0..word_count).rev() {
		match a[index].cmp(&b[index]) {
			std::cmp::Ordering::Equal => continue,
			order => return order,
		}
	}
	std::cmp::Ordering::Equal
}

fn limb(value: &[u32], index: usize) -> u32 {
	value.get(index).copied().unwrap_or(0)
}

fn big_add(a: &[u32], b: &[u32], word_count: usize) -> (Vec<u32>, u32) {
	let mut out = vec![0_u32; word_count + 1];
	let mut carry = 0_u64;
	for (i, word) in out.iter_mut().take(word_count).enumerate() {
		let sum = limb(a, i) as u64 + limb(b, i) as u64 + carry;
		*word = sum as u32;
		carry = sum >> 32;
	}
	out[word_count] = carry as u32;
	(out, carry as u32)
}

fn big_sub_mod(a: &[u32], b: &[u32], word_count: usize) -> Vec<u32> {
	let mut out = vec![0_u32; word_count];
	let mut borrow = 0_i64;
	for (i, word) in out.iter_mut().enumerate() {
		let diff = limb(a, i) as i64 - limb(b, i) as i64 - borrow;
		*word = diff as u32;
		borrow = i64::from(diff < 0);
	}
	out
}

fn big_mul(a: &[u32], b: &[u32], word_count: usize) -> Vec<u32> {
	let mut out = vec![0_u64; 2 * word_count + 1];
	for i in 0..word_count {
		let mut carry = 0_u64;
		for j in 0..word_count {
			let total = limb(a, i) as u64 * limb(b, j) as u64 + out[i + j] + carry;
			out[i + j] = total & 0xFFFF_FFFF;
			carry = total >> 32;
		}
		let mut index = i + word_count;
		while carry > 0 {
			let total = out[index] + carry;
			out[index] = total & 0xFFFF_FFFF;
			carry = total >> 32;
			index += 1;
		}
	}
	let mut result = vec![0_u32; 256];
	for (i, word) in out.iter().enumerate().take(256) {
		result[i] = *word as u32;
	}
	result
}

/// Bitwise long division: (quotient, remainder) over `word_count` limbs.
fn big_divmod(a: &[u32], b: &[u32], word_count: usize) -> (Vec<u32>, Vec<u32>) {
	let mut quotient = vec![0_u32; word_count];
	let mut remainder = vec![0_u32; word_count];
	for bit in (0..word_count * 32).rev() {
		// remainder <<= 1 | a[bit]
		let mut carry = (a[bit / 32] >> (bit % 32)) & 1;
		for slot in remainder.iter_mut().take(word_count) {
			let next = *slot >> 31;
			*slot = (*slot << 1) | carry;
			carry = next;
		}
		if big_cmp(&remainder, b, word_count) != std::cmp::Ordering::Less {
			remainder = big_sub_mod(&remainder, b, word_count);
			quotient[bit / 32] |= 1 << (bit % 32);
		}
	}
	(quotient, remainder)
}

fn next32(random: &mut u64) -> u32 {
	*random ^= *random << 13;
	*random ^= *random >> 7;
	*random ^= *random << 17;
	(*random >> 16) as u32
}

fn random_limbs(count: usize, random: &mut u64) -> Vec<u32> {
	(0..count).map(|_| next32(random)).collect()
}

#[test]
fn slang_bigint_bits_leading_zeros() {
	let mut inputs = Vec::new();
	let mut expected = Vec::new();
	let mut random = 2026101401_u64;
	let mut next32 = || {
		random ^= random << 13;
		random ^= random >> 7;
		random ^= random << 17;
		(random >> 16) as u32
	};
	for word_count in 1..129_usize {
		let mut values: Vec<Vec<u32>> = vec![
			vec![0; 128],
			{
				let mut v = vec![0; 128];
				v[0] = 1;
				v
			},
			{
				let mut v = vec![0; 128];
				v[0] = 0x80000000;
				v
			},
			{
				let mut v = vec![0; 128];
				v[..word_count].fill(u32::MAX);
				v
			},
			{
				let mut v = vec![0; 128];
				for slot in v.iter_mut().take(word_count) {
					*slot = next32();
				}
				v
			},
		];
		for bit in 0..32 {
			let mut v = vec![0_u32; 128];
			v[word_count - 1] = 1 << bit;
			values.push(v);
		}
		for value in values {
			inputs.push(word_count as u32);
			inputs.extend_from_slice(&value);
			expected.push(big_leading_zeros(&value, word_count) as u32);
		}
	}
	assert_eq!(
		execute("bigint_bit_components", &inputs, expected.len() as u32, 1),
		expected
	);
}

#[test]
fn slang_bigint_arithmetic_matches_limb_oracle() {
	let mut random = 2032044096_u64;
	let mut cases: Vec<(usize, Vec<u32>, Vec<u32>)> = Vec::new();
	for word_count in 1..129_usize {
		let mut full = vec![0_u32; word_count];
		full.fill(u32::MAX);
		let top = {
			let mut v = vec![0_u32; word_count];
			v[word_count - 1] = 0x80000000;
			v
		};
		let top_plus = {
			let mut v = top.clone();
			v[0] = 1;
			v
		};
		let zero = vec![0_u32; word_count];
		let one = {
			let mut v = vec![0_u32; word_count];
			v[0] = 1;
			v
		};
		for (a, b) in [
			(zero.clone(), zero.clone()),
			(full.clone(), full.clone()),
			(full.clone(), one.clone()),
			(one.clone(), full.clone()),
			(top.clone(), top_plus),
			(
				random_limbs(word_count, &mut random),
				random_limbs(word_count, &mut random),
			),
		] {
			cases.push((word_count, a, b));
		}
	}
	for word_count in [1_usize, 2, 3, 4, 7, 8, 9, 16, 31, 32, 33, 64, 127, 128] {
		for _ in 0..12 {
			let a = random_limbs(word_count, &mut random);
			let b_words = (next32(&mut random) as usize % word_count) + 1;
			cases.push((word_count, a, random_limbs(b_words, &mut random)));
		}
	}
	let mut inputs = Vec::new();
	let mut expected = Vec::new();
	for (word_count, a, b) in &cases {
		inputs.push(*word_count as u32);
		inputs.extend(limbs_of(a));
		inputs.extend(limbs_of(b));
		let (sum, carry) = big_add(a, b, *word_count);
		expected.extend(limbs_of(&sum[..*word_count]));
		let _ = carry;
		expected.extend(limbs_of(&big_sub_mod(a, b, *word_count)));
		expected.extend(big_mul(a, b, *word_count));
		let (quotient, remainder) = if b.iter().any(|v| *v != 0) {
			big_divmod(
				&limbs_of(a)[..*word_count],
				&limbs_of(b)[..*word_count],
				*word_count,
			)
		} else {
			(
				vec![0_u32; *word_count],
				limbs_of(a)[..*word_count].to_vec(),
			)
		};
		expected.extend(limbs_of(&quotient));
		expected.extend(limbs_of(&remainder));
		let order = big_cmp(
			&limbs_of(a)[..*word_count],
			&limbs_of(b)[..*word_count],
			*word_count,
		);
		expected.push(u32::from(
			limbs_of(&sum[..*word_count]) != limbs_of(&big_add(a, b, *word_count).0[..*word_count])
				|| sum[*word_count] != 0,
		));
		expected.push(u32::from(order == std::cmp::Ordering::Less));
		expected.push(
			(order != std::cmp::Ordering::Less) as u32 + u32::from(order == std::cmp::Ordering::Greater),
		);
		expected.push(u32::from(b.iter().any(|v| *v != 0)));
	}
	assert_eq!(
		execute(
			"bigint_arithmetic_components",
			&inputs,
			cases.len() as u32,
			772
		),
		expected
	);
}

// ---------- Dispatch smoke ----------

#[test]
fn slang_mlkem_empty_dispatch() {
	for (entry, width) in [
		("mlkem_field_components", 5_usize),
		("mlkem_compression_components", 4),
		("mlkem_polynomial_components", 768),
		("mlkem_packing_components", 641),
		("mlkem_sampling_components", 512),
		("mlkem_rejection_components", 257),
		("mlkem_kpke_components", 4705),
		("mlkem_kpke_decrypt_components", 32),
		("mlkem_kem_components", 6369),
		("mlkem_kem_decaps_components", 32),
		("mlkem_keygen_components", 4736),
		("mlkem_encaps_components", 1600),
		("mlkem_key_check_components", 3),
	] {
		assert!(execute(entry, &[], 0, width).is_empty(), "{entry}");
	}
}
