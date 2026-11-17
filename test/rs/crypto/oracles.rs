//! Independent host oracles shared by crypto tests: SHA-2, SHA-3/SHAKE, and
//! schema-driven Poseidon2. These are slow, obvious implementations used only
//! to check production device/slang paths bit-for-bit.

use oa::crypto::keccak_f1600;
use oa::crypto::merkle::MerkleAlgorithm;

// ---------- SHA-256 / SHA-512 host oracles ----------

pub(crate) fn sha256_compress(state: &mut [u32; 8], block: &[u8]) {
	const K: [u32; 64] = [
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
	];
	let mut w = [0_u32; 64];
	for (i, word) in w.iter_mut().take(16).enumerate() {
		*word = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
	}
	for i in 16..64 {
		let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
		let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16]
			.wrapping_add(s0)
			.wrapping_add(w[i - 7])
			.wrapping_add(s1);
	}
	let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
	for i in 0..64 {
		let big1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
		let ch = (e & f) ^ (!e & g);
		let t1 = h
			.wrapping_add(big1)
			.wrapping_add(ch)
			.wrapping_add(K[i])
			.wrapping_add(w[i]);
		let big0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
		let maj = (a & b) ^ (a & c) ^ (b & c);
		let t2 = big0.wrapping_add(maj);
		h = g;
		g = f;
		f = e;
		e = d.wrapping_add(t1);
		d = c;
		c = b;
		b = a;
		a = t1.wrapping_add(t2);
	}
	for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
		*slot = slot.wrapping_add(value);
	}
}

pub(crate) fn sha256(data: &[u8]) -> Vec<u8> {
	let mut state: [u32; 8] = [
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
	];
	let mut padded = data.to_vec();
	padded.push(0x80);
	while padded.len() % 64 != 56 {
		padded.push(0);
	}
	padded.extend_from_slice(&(8 * data.len() as u64).to_be_bytes());
	for block in padded.as_chunks::<64>().0 {
		sha256_compress(&mut state, block);
	}
	state.iter().flat_map(|word| word.to_be_bytes()).collect()
}

pub(crate) fn sha512_compress(state: &mut [u64; 8], block: &[u8]) {
	const K: [u64; 80] = [
		0x428a2f98d728ae22,
		0x7137449123ef65cd,
		0xb5c0fbcfec4d3b2f,
		0xe9b5dba58189dbbc,
		0x3956c25bf348b538,
		0x59f111f1b605d019,
		0x923f82a4af194f9b,
		0xab1c5ed5da6d8118,
		0xd807aa98a3030242,
		0x12835b0145706fbe,
		0x243185be4ee4b28c,
		0x550c7dc3d5ffb4e2,
		0x72be5d74f27b896f,
		0x80deb1fe3b1696b1,
		0x9bdc06a725c71235,
		0xc19bf174cf692694,
		0xe49b69c19ef14ad2,
		0xefbe4786384f25e3,
		0x0fc19dc68b8cd5b5,
		0x240ca1cc77ac9c65,
		0x2de92c6f592b0275,
		0x4a7484aa6ea6e483,
		0x5cb0a9dcbd41fbd4,
		0x76f988da831153b5,
		0x983e5152ee66dfab,
		0xa831c66d2db43210,
		0xb00327c898fb213f,
		0xbf597fc7beef0ee4,
		0xc6e00bf33da88fc2,
		0xd5a79147930aa725,
		0x06ca6351e003826f,
		0x142929670a0e6e70,
		0x27b70a8546d22ffc,
		0x2e1b21385c26c926,
		0x4d2c6dfc5ac42aed,
		0x53380d139d95b3df,
		0x650a73548baf63de,
		0x766a0abb3c77b2a8,
		0x81c2c92e47edaee6,
		0x92722c851482353b,
		0xa2bfe8a14cf10364,
		0xa81a664bbc423001,
		0xc24b8b70d0f89791,
		0xc76c51a30654be30,
		0xd192e819d6ef5218,
		0xd69906245565a910,
		0xf40e35855771202a,
		0x106aa07032bbd1b8,
		0x19a4c116b8d2d0c8,
		0x1e376c085141ab53,
		0x2748774cdf8eeb99,
		0x34b0bcb5e19b48a8,
		0x391c0cb3c5c95a63,
		0x4ed8aa4ae3418acb,
		0x5b9cca4f7763e373,
		0x682e6ff3d6b2b8a3,
		0x748f82ee5defb2fc,
		0x78a5636f43172f60,
		0x84c87814a1f0ab72,
		0x8cc702081a6439ec,
		0x90befffa23631e28,
		0xa4506cebde82bde9,
		0xbef9a3f7b2c67915,
		0xc67178f2e372532b,
		0xca273eceea26619c,
		0xd186b8c721c0c207,
		0xeada7dd6cde0eb1e,
		0xf57d4f7fee6ed178,
		0x06f067aa72176fba,
		0x0a637dc5a2c898a6,
		0x113f9804bef90dae,
		0x1b710b35131c471b,
		0x28db77f523047d84,
		0x32caab7b40c72493,
		0x3c9ebe0a15c9bebc,
		0x431d67c49c100d4c,
		0x4cc5d4becb3e42b6,
		0x597f299cfc657e2a,
		0x5fcb6fab3ad6faec,
		0x6c44198c4a475817,
	];
	let mut w = [0_u64; 80];
	for (i, word) in w.iter_mut().take(16).enumerate() {
		*word = u64::from_be_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
	}
	for i in 16..80 {
		let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
		let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
		w[i] = w[i - 16]
			.wrapping_add(s0)
			.wrapping_add(w[i - 7])
			.wrapping_add(s1);
	}
	let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
	for i in 0..80 {
		let big1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
		let ch = (e & f) ^ (!e & g);
		let t1 = h
			.wrapping_add(big1)
			.wrapping_add(ch)
			.wrapping_add(K[i])
			.wrapping_add(w[i]);
		let big0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
		let maj = (a & b) ^ (a & c) ^ (b & c);
		let t2 = big0.wrapping_add(maj);
		h = g;
		g = f;
		f = e;
		e = d.wrapping_add(t1);
		d = c;
		c = b;
		b = a;
		a = t1.wrapping_add(t2);
	}
	for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
		*slot = slot.wrapping_add(value);
	}
}

pub(crate) fn sha512(data: &[u8]) -> Vec<u8> {
	let mut state: [u64; 8] = [
		0x6a09e667f3bcc908,
		0xbb67ae8584caa73b,
		0x3c6ef372fe94f82b,
		0xa54ff53a5f1d36f1,
		0x510e527fade682d1,
		0x9b05688c2b3e6c1f,
		0x1f83d9abfb41bd6b,
		0x5be0cd19137e2179,
	];
	let mut padded = data.to_vec();
	padded.push(0x80);
	while padded.len() % 128 != 112 {
		padded.push(0);
	}
	padded.extend_from_slice(&(8 * data.len() as u64 as u128).to_be_bytes());
	for block in padded.as_chunks::<128>().0 {
		sha512_compress(&mut state, block);
	}
	state.iter().flat_map(|word| word.to_be_bytes()).collect()
}

// ---------- Keccak host oracle (SHA-3 / SHAKE families) ----------

pub(crate) fn keccak_sponge(rate: usize, domain: u8, input: &[u8], output_len: usize) -> Vec<u8> {
	let mut state = [0_u64; 25];
	let mut offset = 0;
	while offset + rate <= input.len() {
		for (lane, word) in input[offset..offset + rate]
			.as_chunks::<8>()
			.0
			.iter()
			.enumerate()
		{
			state[lane] ^= u64::from_le_bytes(*word);
		}
		keccak_f1600(&mut state);
		offset += rate;
	}
	// pad10*1: the domain byte and the trailing 0x80 share one rate block;
	// when the domain lands at rate-1 the byte becomes domain|0x80.
	let mut tail = input[offset..].to_vec();
	tail.push(domain);
	tail.resize(rate, 0);
	tail[rate - 1] |= 0x80;
	for (lane, word) in tail.as_chunks::<8>().0.iter().enumerate() {
		state[lane] ^= u64::from_le_bytes(*word);
	}
	keccak_f1600(&mut state);
	let mut out = Vec::with_capacity(output_len);
	while out.len() < output_len {
		for lane in state.iter().take(rate / 8) {
			out.extend_from_slice(&lane.to_le_bytes());
		}
		if out.len() < output_len {
			keccak_f1600(&mut state);
		}
	}
	out.truncate(output_len);
	out
}

// ---------- Poseidon2 host oracle (schema-profile driven) ----------

pub(crate) fn poseidon_profiles() -> serde_json::Value {
	let schema: serde_json::Value = serde_json::from_str(include_str!(
		"../../../tool/gen/fn/schema/crypto/crypto_hash.json"
	))
	.unwrap();
	schema["poseidon2_profiles"].clone()
}

pub(crate) fn poseidon_profile(
	profiles: &serde_json::Value,
	algorithm: MerkleAlgorithm,
) -> &serde_json::Value {
	let name = match algorithm {
		MerkleAlgorithm::Poseidon2BabyBear16 => "babybear16",
		MerkleAlgorithm::Poseidon2BabyBear24 => "babybear24",
		MerkleAlgorithm::Poseidon2KoalaBear16 => "koalabear16",
		MerkleAlgorithm::Poseidon2KoalaBear24 => "koalabear24",
		_ => unreachable!("poseidon profile requested for non-poseidon algorithm"),
	};
	profiles
		.as_array()
		.unwrap()
		.iter()
		.find(|profile| profile["name"] == name)
		.unwrap()
}

pub(crate) fn profile_u32s(profile: &serde_json::Value, key: &str) -> Vec<u64> {
	profile[key]
		.as_array()
		.unwrap()
		.iter()
		.map(|v| v.as_u64().unwrap())
		.collect()
}

pub(crate) fn profile_rounds(profile: &serde_json::Value, key: &str) -> Vec<Vec<u64>> {
	profile[key]
		.as_array()
		.unwrap()
		.iter()
		.map(|row| {
			row
				.as_array()
				.unwrap()
				.iter()
				.map(|v| v.as_u64().unwrap())
				.collect()
		})
		.collect()
}

pub(crate) fn poseidon_pow(mut base: u64, mut degree: u64, prime: u64) -> u64 {
	let mut result = 1_u64;
	base %= prime;
	while degree > 0 {
		if degree & 1 == 1 {
			result = result * base % prime;
		}
		base = base * base % prime;
		degree >>= 1;
	}
	result
}

pub(crate) fn poseidon_external(state: &mut [u64], prime: u64) {
	const MDS: [[u64; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
	for chunk in state.as_chunks_mut::<4>().0 {
		let x = *chunk;
		for (slot, row) in chunk.iter_mut().zip(MDS.iter()) {
			*slot = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum::<u64>() % prime;
		}
	}
	let sums: Vec<u64> = (0..4)
		.map(|i| (0..state.len() / 4).map(|j| state[4 * j + i]).sum::<u64>() % prime)
		.collect();
	for (i, x) in state.iter_mut().enumerate() {
		*x = (*x + sums[i % 4]) % prime;
	}
}

pub(crate) fn poseidon_permute(values: &[u64], profile: &serde_json::Value) -> Vec<u64> {
	let prime = profile["prime"].as_u64().unwrap();
	let degree = profile["sbox_degree"].as_u64().unwrap();
	let width = profile["width"].as_u64().unwrap() as usize;
	let mut state: Vec<u64> = values.iter().map(|x| x % prime).collect();
	state.resize(width, 0);
	poseidon_external(&mut state, prime);
	for constants in profile_rounds(profile, "initial_constants") {
		for (x, c) in state.iter_mut().zip(constants.iter()) {
			*x = poseidon_pow((*x + c) % prime, degree, prime);
		}
		poseidon_external(&mut state, prime);
	}
	let diagonal = profile_u32s(profile, "diagonal");
	for constant in profile_u32s(profile, "internal_constants") {
		state[0] = poseidon_pow((state[0] + constant) % prime, degree, prime);
		let total: u64 = state.iter().sum::<u64>() % prime;
		for (x, d) in state.iter_mut().zip(diagonal.iter()) {
			*x = (total + *d * *x) % prime;
		}
	}
	for constants in profile_rounds(profile, "final_constants") {
		for (x, c) in state.iter_mut().zip(constants.iter()) {
			*x = poseidon_pow((*x + c) % prime, degree, prime);
		}
		poseidon_external(&mut state, prime);
	}
	state
}

/// Merkle parent hash for one Poseidon2 profile: absorb two 8-word digests
/// under the OAM2 frame, squeeze the first 8 field elements, pack as bytes.
pub(crate) fn poseidon2_parent(
	left: &[u64; 8],
	right: &[u64; 8],
	profile: &serde_json::Value,
) -> Vec<u8> {
	let prime = profile["prime"].as_u64().unwrap();
	let width = profile["width"].as_u64().unwrap() as usize;
	let rate = width - 8;
	let mut state = vec![0_u64; width];
	state[rate] = 0x4F414D32;
	state[rate + 1] = 16;
	let mut cursor = 0;
	for value in left.iter().chain(right.iter()) {
		state[cursor] = (state[cursor] + value) % prime;
		cursor += 1;
		if cursor == rate {
			state = poseidon_permute(&state, profile);
			cursor = 0;
		}
	}
	state[cursor] = (state[cursor] + 1) % prime;
	state = poseidon_permute(&state, profile);
	state[..8]
		.iter()
		.flat_map(|x| (*x as u32).to_le_bytes())
		.collect()
}

/// OA-framed Poseidon2 sponge for the hash operations: 0x4F415032 tag and
/// input-length trailer under the same absorb/squeeze schedule as device.
pub(crate) fn poseidon2_sponge(
	values: &[u64],
	profile: &serde_json::Value,
	output_length: usize,
) -> Vec<u64> {
	let prime = profile["prime"].as_u64().unwrap();
	let width = profile["width"].as_u64().unwrap() as usize;
	let rate = width - profile["capacity"].as_u64().unwrap() as usize;
	let mut state = vec![0_u64; width];
	state[rate] = 0x4F415032;
	state[rate + 1] = values.len() as u64;
	let mut cursor = 0;
	for value in values {
		state[cursor] = (state[cursor] + value) % prime;
		cursor += 1;
		if cursor == rate {
			state = poseidon_permute(&state, profile);
			cursor = 0;
		}
	}
	state[cursor] = (state[cursor] + 1) % prime;
	state = poseidon_permute(&state, profile);
	cursor = 0;
	let mut result = Vec::with_capacity(output_length);
	while result.len() < output_length {
		if cursor == rate {
			state = poseidon_permute(&state, profile);
			cursor = 0;
		}
		result.push(state[cursor]);
		cursor += 1;
	}
	result
}

// ---------- Shared Merkle host oracle ----------

pub(crate) fn sha256_variant(data: &[u8], iv: [u32; 8]) -> Vec<u8> {
	let mut state = iv;
	let mut padded = data.to_vec();
	padded.push(0x80);
	while padded.len() % 64 != 56 {
		padded.push(0);
	}
	padded.extend_from_slice(&(8 * data.len() as u64).to_be_bytes());
	for block in padded.as_chunks::<64>().0 {
		sha256_compress(&mut state, block);
	}
	state.iter().flat_map(|word| word.to_be_bytes()).collect()
}

pub(crate) fn sha224(data: &[u8]) -> Vec<u8> {
	sha256_variant(
		data,
		[
			0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939, 0xffc00b31, 0x68581511, 0x64f98fa7,
			0xbefa4fa4,
		],
	)[..28]
		.to_vec()
}

pub(crate) fn sha512_variant(data: &[u8], iv: [u64; 8]) -> Vec<u8> {
	let mut state = iv;
	let mut padded = data.to_vec();
	padded.push(0x80);
	while padded.len() % 128 != 112 {
		padded.push(0);
	}
	padded.extend_from_slice(&(8 * data.len() as u64 as u128).to_be_bytes());
	for block in padded.as_chunks::<128>().0 {
		sha512_compress(&mut state, block);
	}
	state.iter().flat_map(|word| word.to_be_bytes()).collect()
}

pub(crate) fn sha384(data: &[u8]) -> Vec<u8> {
	sha512_variant(
		data,
		[
			0xcbbb9d5dc1059ed8,
			0x629a292a367cd507,
			0x9159015a3070dd17,
			0x152fecd8f70e5939,
			0x67332667ffc00b31,
			0x8eb44a8768581511,
			0xdb0c2e0d64f98fa7,
			0x47b5481dbefa4fa4,
		],
	)[..48]
		.to_vec()
}

pub(crate) fn sha512_224(data: &[u8]) -> Vec<u8> {
	sha512_variant(
		data,
		[
			0x8c3d37c819544da2,
			0x73e1996689dcd4d6,
			0x1dfab7ae32ff9c82,
			0x679dd514582f9fcf,
			0x0f6d2b697bd44da8,
			0x77e36f7304c48942,
			0x3f9d85a86a1d36c8,
			0x1112e6ad91d692a1,
		],
	)[..28]
		.to_vec()
}

pub(crate) fn sha512_256(data: &[u8]) -> Vec<u8> {
	sha512_variant(
		data,
		[
			0x22312194fc2bf72c,
			0x9f555fa3c84c64c2,
			0x2393b86b6f53b151,
			0x963877195940eabd,
			0x96283ee2a88effe3,
			0xbe5e1e2553863992,
			0x2b0199fc2c85b8aa,
			0x0eb72ddc81c52ca2,
		],
	)[..32]
		.to_vec()
}

pub(crate) fn sha3(bits: usize, data: &[u8]) -> Vec<u8> {
	keccak_sponge(200 - 2 * (bits / 8), 0x06, data, bits / 8)
}

pub(crate) fn shake(bits: usize, data: &[u8], output_len: usize) -> Vec<u8> {
	keccak_sponge(200 - 2 * (bits / 8), 0x1f, data, output_len)
}
