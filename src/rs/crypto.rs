//! CPU cryptographic primitives and typed hash/Merkle values.
//!
//! This is a direct algorithmic port of OA's FIPS 202 Keccak/SHAKE and SP
//! 800-185 KMAC implementation. Secret-key operations remain on the CPU; no
//! secret material is placed in generic device storage.

use crate::{Error, Result, core::memory::zero_secure};

const SHAKE128_RATE: usize = 168;
const SHAKE256_RATE: usize = 136;
const SHAKE_DOMAIN: u8 = 0x1f;
const CSHAKE_DOMAIN: u8 = 0x04;

const ROUND_CONSTANTS: [u64; 24] = [
	0x0000_0000_0000_0001,
	0x0000_0000_0000_8082,
	0x8000_0000_0000_808a,
	0x8000_0000_8000_8000,
	0x0000_0000_0000_808b,
	0x0000_0000_8000_0001,
	0x8000_0000_8000_8081,
	0x8000_0000_0000_8009,
	0x0000_0000_0000_008a,
	0x0000_0000_0000_0088,
	0x0000_0000_8000_8009,
	0x0000_0000_8000_000a,
	0x0000_0000_8000_808b,
	0x8000_0000_0000_008b,
	0x8000_0000_0000_8089,
	0x8000_0000_0000_8003,
	0x8000_0000_0000_8002,
	0x8000_0000_0000_0080,
	0x0000_0000_0000_800a,
	0x8000_0000_8000_000a,
	0x8000_0000_8000_8081,
	0x8000_0000_0000_8080,
	0x0000_0000_8000_0001,
	0x8000_0000_8000_8008,
];

const ROTATION_OFFSETS: [u32; 25] = [
	0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// A fixed 32-byte SHAKE-256 digest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Hash([u8; Self::SIZE]);

impl Hash {
	/// Hash storage size in bytes.
	pub const SIZE: usize = 32;

	/// Return the all-zero sentinel hash.
	#[must_use]
	pub const fn zero() -> Self {
		Self([0; Self::SIZE])
	}

	/// Parse exactly 32 bytes.
	///
	/// # Errors
	///
	/// Returns an error when `bytes` is not exactly 32 bytes long.
	pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
		let bytes = <[u8; Self::SIZE]>::try_from(bytes)
			.map_err(|_| Error::invalid_argument("Hash requires exactly 32 bytes"))?;
		Ok(Self(bytes))
	}

	/// Parse exactly 64 hexadecimal ASCII characters.
	///
	/// # Errors
	///
	/// Returns an error for the wrong length or a non-hexadecimal character.
	pub fn from_hex(hex: &str) -> Result<Self> {
		if hex.len() != Self::SIZE * 2 {
			return Err(Error::invalid_argument(
				"Hash hexadecimal text must contain exactly 64 characters",
			));
		}
		let mut bytes = [0_u8; Self::SIZE];
		let (pairs, remainder) = hex.as_bytes().as_chunks::<2>();
		debug_assert!(remainder.is_empty());
		for (index, pair) in pairs.iter().enumerate() {
			let high = hex_nibble(pair[0]).ok_or_else(|| {
				Error::invalid_argument(
					"Hash hexadecimal text contains a non-hexadecimal character",
				)
			})?;
			let low = hex_nibble(pair[1]).ok_or_else(|| {
				Error::invalid_argument(
					"Hash hexadecimal text contains a non-hexadecimal character",
				)
			})?;
			bytes[index] = (high << 4) | low;
		}
		Ok(Self(bytes))
	}

	/// Return the digest bytes.
	#[must_use]
	pub const fn as_bytes(&self) -> &[u8; Self::SIZE] {
		&self.0
	}

	/// Return whether this is the all-zero sentinel.
	#[must_use]
	pub fn is_zero(&self) -> bool {
		self.0.iter().all(|byte| *byte == 0)
	}

	/// Format all 32 bytes as lowercase hexadecimal text.
	#[must_use]
	pub fn to_hex(self) -> String {
		const HEX: &[u8; 16] = b"0123456789abcdef";
		let mut output = String::with_capacity(Self::SIZE * 2);
		for byte in self.0 {
			output.push(char::from(HEX[usize::from(byte >> 4)]));
			output.push(char::from(HEX[usize::from(byte & 0x0f)]));
		}
		output
	}

	/// Format the first eight bytes as lowercase hexadecimal text.
	#[must_use]
	pub fn to_short_hex(self) -> String {
		self.to_hex()[..16].to_owned()
	}
}

/// Apply the 24-round Keccak-f[1600] permutation in place.
pub fn keccak_f1600(state: &mut [u64; 25]) {
	for round_constant in ROUND_CONSTANTS {
		let mut columns = [0_u64; 5];
		for index in 0..5 {
			columns[index] = state[index]
				^ state[5 + index]
				^ state[10 + index]
				^ state[15 + index]
				^ state[20 + index];
		}
		let mut differences = [0_u64; 5];
		for index in 0..5 {
			differences[index] = columns[(index + 4) % 5] ^ columns[(index + 1) % 5].rotate_left(1);
		}
		for index in 0..25 {
			state[index] ^= differences[index % 5];
		}

		let mut rotated = [0_u64; 25];
		for index in 0..25 {
			let x = index % 5;
			let y = index / 5;
			let next_x = y;
			let next_y = (2 * x + 3 * y) % 5;
			rotated[5 * next_y + next_x] = state[index].rotate_left(ROTATION_OFFSETS[index]);
		}

		for y in 0..5 {
			let base = y * 5;
			state[base] = rotated[base] ^ (!rotated[base + 1] & rotated[base + 2]);
			state[base + 1] = rotated[base + 1] ^ (!rotated[base + 2] & rotated[base + 3]);
			state[base + 2] = rotated[base + 2] ^ (!rotated[base + 3] & rotated[base + 4]);
			state[base + 3] = rotated[base + 3] ^ (!rotated[base + 4] & rotated[base]);
			state[base + 4] = rotated[base + 4] ^ (!rotated[base] & rotated[base + 1]);
		}

		state[0] ^= round_constant;
	}
}

/// Produce SHAKE-128 output of exactly `output.len()` bytes.
pub fn shake128(input: &[u8], output: &mut [u8]) {
	let mut sponge = Sponge::new(SHAKE128_RATE, SHAKE_DOMAIN);
	sponge.absorb(input);
	sponge.squeeze(output);
}

/// Produce SHAKE-256 output of exactly `output.len()` bytes.
pub fn shake256(input: &[u8], output: &mut [u8]) {
	let mut sponge = Sponge::new(SHAKE256_RATE, SHAKE_DOMAIN);
	sponge.absorb(input);
	sponge.squeeze(output);
}

/// Compute the canonical 32-byte SHAKE-256 digest.
#[must_use]
pub fn hash(input: &[u8]) -> Hash {
	let mut output = [0_u8; Hash::SIZE];
	shake256(input, &mut output);
	Hash(output)
}

/// Incremental SHAKE-128 context.
pub struct Shake128(Sponge);

impl Shake128 {
	/// Create an empty SHAKE-128 context.
	#[must_use]
	pub fn new() -> Self {
		Self(Sponge::new(SHAKE128_RATE, SHAKE_DOMAIN))
	}

	/// Absorb another input chunk.
	///
	/// # Errors
	///
	/// Returns an error after the first squeeze operation.
	pub fn update(&mut self, input: &[u8]) -> Result<()> {
		self.0.update(input)
	}

	/// Squeeze the next output bytes, finalizing on the first call.
	pub fn squeeze(&mut self, output: &mut [u8]) {
		self.0.squeeze(output);
	}

	/// Reset to a fresh empty context.
	pub fn reset(&mut self) {
		self.0.reset();
	}
}

impl Default for Shake128 {
	fn default() -> Self {
		Self::new()
	}
}

/// Incremental SHAKE-256 context.
pub struct Shake256(Sponge);

impl Shake256 {
	/// Create an empty SHAKE-256 context.
	#[must_use]
	pub fn new() -> Self {
		Self(Sponge::new(SHAKE256_RATE, SHAKE_DOMAIN))
	}

	/// Absorb another input chunk.
	///
	/// # Errors
	///
	/// Returns an error after the first squeeze operation.
	pub fn update(&mut self, input: &[u8]) -> Result<()> {
		self.0.update(input)
	}

	/// Squeeze the next output bytes, finalizing on the first call.
	pub fn squeeze(&mut self, output: &mut [u8]) {
		self.0.squeeze(output);
	}

	/// Reset to a fresh empty context.
	pub fn reset(&mut self) {
		self.0.reset();
	}
}

impl Default for Shake256 {
	fn default() -> Self {
		Self::new()
	}
}

/// Incremental canonical 32-byte SHAKE-256 hasher.
#[derive(Default)]
pub struct Hasher {
	context: Shake256,
	digest: Option<Hash>,
}

impl Hasher {
	/// Create an empty hasher.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Absorb another input chunk.
	///
	/// # Errors
	///
	/// Returns an error after [`Self::finalize`] until [`Self::reset`] is called.
	pub fn update(&mut self, input: &[u8]) -> Result<()> {
		if self.digest.is_some() {
			return Err(Error::failed_precondition(
				"Hasher cannot absorb after finalize; call reset first",
			));
		}
		self.context.update(input)
	}

	/// Finalize and return the digest. Repeated calls return the same value.
	pub fn finalize(&mut self) -> Hash {
		if let Some(digest) = self.digest {
			return digest;
		}
		let mut bytes = [0_u8; Hash::SIZE];
		self.context.squeeze(&mut bytes);
		let digest = Hash(bytes);
		self.digest = Some(digest);
		digest
	}

	/// Reset to the empty-message state.
	pub fn reset(&mut self) {
		self.context.reset();
		self.digest = None;
	}
}

/// Compute KMAC-256 according to NIST SP 800-185.
///
/// # Errors
///
/// Returns an error when a bit length, encoded length, or allocation size
/// overflows the platform or SP 800-185 `u64` encoding.
pub fn kmac256(
	key: &[u8],
	data: &[u8],
	customization: &[u8],
	output_len: usize,
) -> Result<Vec<u8>> {
	let key_bits = bit_length(key.len(), "KMAC key")?;
	let custom_bits = bit_length(customization.len(), "KMAC customization")?;
	let output_bits = bit_length(output_len, "KMAC output")?;

	let mut prefix = SecureBytes::new(bytepad_len(
		left_encode_len(SHAKE256_RATE as u64)
			.checked_add(encoded_string_len(4)?)
			.and_then(|length| length.checked_add(encoded_string_len(customization.len()).ok()?))
			.ok_or_else(|| Error::out_of_range("KMAC customization encoding is too large"))?,
		SHAKE256_RATE,
	)?)?;
	append_left_encode(&mut prefix.bytes, SHAKE256_RATE as u64);
	append_encode_string(&mut prefix.bytes, b"KMAC", 32);
	append_encode_string(&mut prefix.bytes, customization, custom_bits);
	prefix.pad_to_capacity();

	let mut key_block = SecureBytes::new(bytepad_len(
		left_encode_len(SHAKE256_RATE as u64)
			.checked_add(left_encode_len(key_bits))
			.and_then(|length| length.checked_add(key.len()))
			.ok_or_else(|| Error::out_of_range("KMAC key encoding is too large"))?,
		SHAKE256_RATE,
	)?)?;
	append_left_encode(&mut key_block.bytes, SHAKE256_RATE as u64);
	append_encode_string(&mut key_block.bytes, key, key_bits);
	key_block.pad_to_capacity();

	let mut right_encoded = [0_u8; 9];
	let right_len = write_right_encode(output_bits, &mut right_encoded);
	let mut sponge = Sponge::new(SHAKE256_RATE, CSHAKE_DOMAIN);
	sponge.absorb(&prefix.bytes);
	sponge.absorb(&key_block.bytes);
	sponge.absorb(data);
	sponge.absorb(&right_encoded[..right_len]);

	let mut output = Vec::new();
	output
		.try_reserve_exact(output_len)
		.map_err(|_| Error::resource_exhausted("KMAC output allocation failed"))?;
	output.resize(output_len, 0);
	sponge.squeeze(&mut output);
	zero_secure(&mut right_encoded);
	Ok(output)
}

/// Combine two Merkle nodes as SHAKE-256(`left || right`).
#[must_use]
pub fn hash_combine(left: &Hash, right: &Hash) -> Hash {
	let mut pair = [0_u8; Hash::SIZE * 2];
	pair[..Hash::SIZE].copy_from_slice(left.as_bytes());
	pair[Hash::SIZE..].copy_from_slice(right.as_bytes());
	hash(&pair)
}

/// Compute the Merkle root, duplicating the final node of each odd level.
#[must_use]
pub fn merkle_root(leaves: &[Hash]) -> Hash {
	if leaves.is_empty() {
		return Hash::zero();
	}
	let mut level = leaves.to_vec();
	while level.len() > 1 {
		let mut next = Vec::with_capacity(level.len().div_ceil(2));
		for pair in level.chunks(2) {
			let right = pair.get(1).unwrap_or(&pair[0]);
			next.push(hash_combine(&pair[0], right));
		}
		level = next;
	}
	level[0]
}

/// Fully materialized CPU Merkle tree, with leaves at level zero.
pub struct MerkleTree {
	levels: Vec<Vec<Hash>>,
	root: Hash,
}

impl MerkleTree {
	/// Return all levels from leaves through the root.
	#[must_use]
	pub fn levels(&self) -> &[Vec<Hash>] {
		&self.levels
	}

	/// Return the tree root.
	#[must_use]
	pub const fn root(&self) -> Hash {
		self.root
	}
}

/// Build a materialized Merkle tree.
#[must_use]
pub fn build_merkle_tree(leaves: &[Hash]) -> MerkleTree {
	if leaves.is_empty() {
		return MerkleTree {
			levels: Vec::new(),
			root: Hash::zero(),
		};
	}
	let mut levels = vec![leaves.to_vec()];
	while levels.last().is_some_and(|level| level.len() > 1) {
		let previous = &levels[levels.len() - 1];
		let mut next = Vec::with_capacity(previous.len().div_ceil(2));
		for pair in previous.chunks(2) {
			let right = pair.get(1).unwrap_or(&pair[0]);
			next.push(hash_combine(&pair[0], right));
		}
		levels.push(next);
	}
	let root = levels[levels.len() - 1][0];
	MerkleTree { levels, root }
}

/// Sibling nodes and directions for one Merkle inclusion proof.
pub struct MerkleProof {
	siblings: Vec<Hash>,
	is_left: Vec<bool>,
}

impl MerkleProof {
	/// Construct a checked proof representation.
	///
	/// # Errors
	///
	/// Returns an error unless each sibling has one matching direction flag.
	pub fn from_parts(siblings: Vec<Hash>, is_left: Vec<bool>) -> Result<Self> {
		if siblings.len() != is_left.len() {
			return Err(Error::invalid_argument(
				"Merkle proof sibling and direction counts differ",
			));
		}
		Ok(Self { siblings, is_left })
	}

	/// Return sibling nodes from leaf level upward.
	#[must_use]
	pub fn siblings(&self) -> &[Hash] {
		&self.siblings
	}

	/// Return whether each sibling appears to the left of the running hash.
	#[must_use]
	pub fn sibling_is_left(&self) -> &[bool] {
		&self.is_left
	}
}

/// Construct the inclusion proof for one leaf.
///
/// # Errors
///
/// Returns an error for an empty tree or an out-of-range leaf index.
pub fn merkle_proof(tree: &MerkleTree, leaf_index: usize) -> Result<MerkleProof> {
	let leaves = tree.levels.first().ok_or_else(|| {
		Error::invalid_argument("cannot prove membership in an empty Merkle tree")
	})?;
	if leaf_index >= leaves.len() {
		return Err(Error::out_of_range("Merkle leaf index is out of range"));
	}
	let mut index = leaf_index;
	let mut siblings = Vec::with_capacity(tree.levels.len().saturating_sub(1));
	let mut is_left = Vec::with_capacity(tree.levels.len().saturating_sub(1));
	for nodes in tree.levels.iter().take(tree.levels.len() - 1) {
		let sibling_index = if index.is_multiple_of(2) {
			index + 1
		} else {
			index - 1
		};
		siblings.push(*nodes.get(sibling_index).unwrap_or(&nodes[index]));
		is_left.push(!index.is_multiple_of(2));
		index /= 2;
	}
	MerkleProof::from_parts(siblings, is_left)
}

/// Verify one Merkle inclusion proof against `root`.
#[must_use]
pub fn verify_merkle_proof(leaf: &Hash, proof: &MerkleProof, root: &Hash) -> bool {
	if proof.siblings.len() != proof.is_left.len() {
		return false;
	}
	let mut current = *leaf;
	for (sibling, is_left) in proof.siblings.iter().zip(&proof.is_left) {
		current = if *is_left {
			hash_combine(sibling, &current)
		} else {
			hash_combine(&current, sibling)
		};
	}
	current == *root
}

struct Sponge {
	state: [u64; 25],
	buffer: [u8; SHAKE128_RATE],
	buffer_len: usize,
	rate: usize,
	domain: u8,
	squeezing: bool,
	squeeze_offset: usize,
}

impl Sponge {
	const fn new(rate: usize, domain: u8) -> Self {
		Self {
			state: [0; 25],
			buffer: [0; SHAKE128_RATE],
			buffer_len: 0,
			rate,
			domain,
			squeezing: false,
			squeeze_offset: 0,
		}
	}

	fn update(&mut self, input: &[u8]) -> Result<()> {
		if self.squeezing {
			return Err(Error::failed_precondition(
				"SHAKE cannot absorb after squeezing; reset the context first",
			));
		}
		self.absorb(input);
		Ok(())
	}

	fn absorb(&mut self, mut input: &[u8]) {
		debug_assert!(!self.squeezing);
		while !input.is_empty() {
			let chunk = input.len().min(self.rate - self.buffer_len);
			self.buffer[self.buffer_len..self.buffer_len + chunk].copy_from_slice(&input[..chunk]);
			self.buffer_len += chunk;
			input = &input[chunk..];
			if self.buffer_len == self.rate {
				xor_bytes_into_state(&mut self.state, &self.buffer[..self.rate]);
				keccak_f1600(&mut self.state);
				self.buffer[..self.rate].fill(0);
				self.buffer_len = 0;
			}
		}
	}

	fn squeeze(&mut self, mut output: &mut [u8]) {
		if !self.squeezing {
			self.buffer[self.buffer_len] = self.domain;
			self.buffer[self.buffer_len + 1..self.rate].fill(0);
			self.buffer[self.rate - 1] |= 0x80;
			xor_bytes_into_state(&mut self.state, &self.buffer[..self.rate]);
			keccak_f1600(&mut self.state);
			self.buffer[..self.rate].fill(0);
			self.buffer_len = 0;
			self.squeezing = true;
			self.squeeze_offset = 0;
		}

		while !output.is_empty() {
			if self.squeeze_offset == self.rate {
				keccak_f1600(&mut self.state);
				self.squeeze_offset = 0;
			}
			let chunk = output.len().min(self.rate - self.squeeze_offset);
			extract_bytes(&self.state, self.squeeze_offset, &mut output[..chunk]);
			self.squeeze_offset += chunk;
			output = &mut output[chunk..];
		}
	}

	fn reset(&mut self) {
		zeroize_lanes(&mut self.state);
		zero_secure(&mut self.buffer);
		self.buffer_len = 0;
		self.squeezing = false;
		self.squeeze_offset = 0;
	}
}

impl Drop for Sponge {
	fn drop(&mut self) {
		zeroize_lanes(&mut self.state);
		zero_secure(&mut self.buffer);
	}
}

struct SecureBytes {
	bytes: Vec<u8>,
	capacity: usize,
}

impl SecureBytes {
	fn new(capacity: usize) -> Result<Self> {
		let mut bytes = Vec::new();
		bytes
			.try_reserve_exact(capacity)
			.map_err(|_| Error::resource_exhausted("KMAC temporary allocation failed"))?;
		Ok(Self { bytes, capacity })
	}

	fn pad_to_capacity(&mut self) {
		self.bytes.resize(self.capacity, 0);
	}
}

impl Drop for SecureBytes {
	fn drop(&mut self) {
		zero_secure(&mut self.bytes);
	}
}

fn xor_bytes_into_state(state: &mut [u64; 25], bytes: &[u8]) {
	for (index, byte) in bytes.iter().copied().enumerate() {
		state[index / 8] ^= u64::from(byte) << ((index % 8) * 8);
	}
}

fn extract_bytes(state: &[u64; 25], offset: usize, output: &mut [u8]) {
	for (output_index, byte) in output.iter_mut().enumerate() {
		let state_index = offset + output_index;
		*byte = (state[state_index / 8] >> ((state_index % 8) * 8)) as u8;
	}
}

fn zeroize_lanes(state: &mut [u64; 25]) {
	let byte_len = size_of_val(state);
	// SAFETY: `state` is a live, exclusively borrowed contiguous array. A byte
	// slice may view every initialized byte of any value, and its exact extent is
	// `size_of_val(state)`. The slice does not outlive the borrow.
	let bytes = unsafe { std::slice::from_raw_parts_mut(state.as_mut_ptr().cast(), byte_len) };
	zero_secure(bytes);
}

fn bit_length(byte_length: usize, name: &'static str) -> Result<u64> {
	let bytes = u64::try_from(byte_length)
		.map_err(|_| Error::out_of_range(format!("{name} length exceeds u64")))?;
	bytes
		.checked_mul(8)
		.ok_or_else(|| Error::out_of_range(format!("{name} bit length exceeds u64")))
}

fn left_encode_len(value: u64) -> usize {
	if value == 0 {
		2
	} else {
		1 + ((u64::BITS - value.leading_zeros()) as usize).div_ceil(8)
	}
}

fn encoded_string_len(byte_length: usize) -> Result<usize> {
	let bits = bit_length(byte_length, "encoded string")?;
	left_encode_len(bits)
		.checked_add(byte_length)
		.ok_or_else(|| Error::out_of_range("encoded string length overflows usize"))
}

fn bytepad_len(raw_length: usize, rate: usize) -> Result<usize> {
	raw_length
		.checked_add(rate - 1)
		.map(|length| length / rate * rate)
		.ok_or_else(|| Error::out_of_range("KMAC bytepad length overflows usize"))
}

fn append_left_encode(output: &mut Vec<u8>, value: u64) {
	let mut encoded = [0_u8; 9];
	let length = write_left_encode(value, &mut encoded);
	output.extend_from_slice(&encoded[..length]);
}

fn append_encode_string(output: &mut Vec<u8>, input: &[u8], bit_length: u64) {
	append_left_encode(output, bit_length);
	output.extend_from_slice(input);
}

fn write_left_encode(value: u64, output: &mut [u8; 9]) -> usize {
	let bytes = if value == 0 {
		1
	} else {
		((u64::BITS - value.leading_zeros()) as usize).div_ceil(8)
	};
	output[0] = bytes as u8;
	for (index, byte) in output[1..=bytes].iter_mut().enumerate() {
		*byte = (value >> (8 * (bytes - 1 - index))) as u8;
	}
	bytes + 1
}

fn write_right_encode(value: u64, output: &mut [u8; 9]) -> usize {
	let bytes = if value == 0 {
		1
	} else {
		((u64::BITS - value.leading_zeros()) as usize).div_ceil(8)
	};
	for (index, byte) in output[..bytes].iter_mut().enumerate() {
		*byte = (value >> (8 * (bytes - 1 - index))) as u8;
	}
	output[bytes] = bytes as u8;
	bytes + 1
}

const fn hex_nibble(byte: u8) -> Option<u8> {
	match byte {
		b'0'..=b'9' => Some(byte - b'0'),
		b'a'..=b'f' => Some(byte - b'a' + 10),
		b'A'..=b'F' => Some(byte - b'A' + 10),
		_ => None,
	}
}
