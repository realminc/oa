//! Deterministic byte-pair tokenization for language-model inputs.

use std::{collections::BTreeMap, fs, path::Path};

use crate::{Error, Result};

const BPE_MAGIC: &str = "oa_bpe_v1";
const BASE_VOCAB_SIZE: usize = 256;
const MAX_PERSISTED_MERGES: usize = 1_000_000;

/// One learned byte-pair merge. Merge rank `i` creates token `256 + i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BpeMerge {
	/// Left byte or previously learned token.
	pub left: u32,
	/// Right byte or previously learned token.
	pub right: u32,
}

/// Deterministic byte-pair encoder compatible with OA's `oa_bpe_v1` files.
pub struct BpeTokenizer {
	target_vocab: usize,
	merges: Vec<BpeMerge>,
}

impl BpeTokenizer {
	/// Construct an untrained tokenizer, clamping the target to the 256 base bytes.
	pub fn new(target_vocab: usize) -> Self {
		Self {
			target_vocab: target_vocab.clamp(BASE_VOCAB_SIZE, u32::MAX as usize),
			merges: Vec::new(),
		}
	}

	/// Learn at most `num_merges` deterministic pair ranks from raw bytes.
	pub fn train(&mut self, bytes: &[u8], num_merges: usize) {
		self.merges.clear();
		let mut ids = bytes
			.iter()
			.map(|byte| u32::from(*byte))
			.collect::<Vec<_>>();
		let limit = num_merges.min(self.target_vocab - BASE_VOCAB_SIZE);
		for _ in 0..limit {
			if ids.len() < 2 {
				break;
			}
			let mut counts = BTreeMap::<(u32, u32), usize>::new();
			for pair in ids.windows(2) {
				*counts.entry((pair[0], pair[1])).or_default() += 1;
			}
			let best = counts.into_iter().filter(|(_, count)| *count >= 2).max_by(
				|(left_pair, left_count), (right_pair, right_count)| {
					left_count
						.cmp(right_count)
						.then_with(|| right_pair.cmp(left_pair))
				},
			);
			let Some(((left, right), _)) = best else {
				break;
			};
			let merge = BpeMerge { left, right };
			let new_token = u32::try_from(BASE_VOCAB_SIZE + self.merges.len())
				.expect("BPE target vocabulary cannot exceed usize before u32");
			self.merges.push(merge);
			ids = apply_merge(&ids, merge, new_token);
		}
	}

	/// Learn merges from UTF-8 text without changing its byte representation.
	pub fn train_text(&mut self, text: &str, num_merges: usize) {
		self.train(text.as_bytes(), num_merges);
	}

	/// Apply learned ranks to raw bytes.
	pub fn encode(&self, bytes: &[u8]) -> Vec<u32> {
		let mut tokens = bytes
			.iter()
			.map(|byte| u32::from(*byte))
			.collect::<Vec<_>>();
		for (rank, merge) in self.merges.iter().copied().enumerate() {
			let new_token =
				u32::try_from(BASE_VOCAB_SIZE + rank).expect("validated BPE rank must fit u32");
			tokens = apply_merge(&tokens, merge, new_token);
		}
		tokens
	}

	/// Apply learned ranks to UTF-8 text.
	pub fn encode_text(&self, text: &str) -> Vec<u32> {
		self.encode(text.as_bytes())
	}

	/// Expand valid tokens recursively to their exact original bytes.
	///
	/// Negative C++ token IDs cannot occur in this Rust spelling. Unknown token
	/// IDs are ignored, matching the donor decoder's failure behavior.
	///
	/// # Errors
	///
	/// Returns [`crate::ErrorKind::ResourceExhausted`] when expansion storage
	/// cannot be reserved. Expansion is iterative, so nested persisted ranks do
	/// not consume the native call stack.
	pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>> {
		let mut output = Vec::new();
		let mut pending = Vec::new();
		pending
			.try_reserve(tokens.len())
			.map_err(|_| Error::resource_exhausted("BPE decode stack allocation failed"))?;
		pending.extend(tokens.iter().rev().copied());
		while let Some(token) = pending.pop() {
			if token < BASE_VOCAB_SIZE as u32 {
				output.try_reserve(1).map_err(|_| {
					Error::resource_exhausted("BPE decoded output allocation failed")
				})?;
				output.push(token as u8);
				continue;
			}
			let Ok(rank) = usize::try_from(token - BASE_VOCAB_SIZE as u32) else {
				continue;
			};
			let Some(merge) = self.merges.get(rank) else {
				continue;
			};
			pending
				.try_reserve(2)
				.map_err(|_| Error::resource_exhausted("BPE decode stack allocation failed"))?;
			pending.push(merge.right);
			pending.push(merge.left);
		}
		Ok(output)
	}

	/// Decode tokens as UTF-8 text.
	///
	/// # Errors
	///
	/// Returns [`crate::ErrorKind::DataLoss`] when the exact decoded bytes are not
	/// valid UTF-8.
	pub fn decode_text(&self, tokens: &[u32]) -> Result<String> {
		String::from_utf8(self.decode(tokens)?)
			.map_err(|_| Error::data_loss("BPE tokens do not decode to valid UTF-8"))
	}

	/// Right-align an encoded prompt in a zero-padded fixed context.
	pub fn encode_prompt(&self, prompt: &[u8], context_length: usize) -> Vec<u32> {
		let tokens = self.encode(prompt);
		let copied = tokens.len().min(context_length);
		let mut output = vec![0_u32; context_length];
		output[context_length - copied..].copy_from_slice(&tokens[tokens.len() - copied..]);
		output
	}

	/// Persist exact merge ranks in OA's architecture-independent text format.
	///
	/// # Errors
	///
	/// Returns an I/O error when the file cannot be written.
	pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
		let mut text = format!("{BPE_MAGIC}\n{}\n", self.merges.len());
		for merge in &self.merges {
			text.push_str(&format!("{} {}\n", merge.left, merge.right));
		}
		fs::write(path, text).map_err(|source| Error::io("BPE vocabulary write", source))
	}

	/// Load exact merge ranks transactionally from an OA `oa_bpe_v1` file.
	///
	/// # Errors
	///
	/// Returns an I/O error when reading fails or [`crate::ErrorKind::DataLoss`]
	/// when the header, count, integer syntax, or merge dependency order is invalid.
	pub fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
		let bytes = fs::read(path).map_err(|source| Error::io("BPE vocabulary read", source))?;
		let text = std::str::from_utf8(&bytes)
			.map_err(|_| Error::data_loss("BPE vocabulary is not UTF-8 text"))?;
		let mut fields = text.split_ascii_whitespace();
		if fields.next() != Some(BPE_MAGIC) {
			return Err(Error::data_loss("BPE vocabulary has an invalid header"));
		}
		let count = parse_usize(fields.next(), "merge count")?;
		if count > MAX_PERSISTED_MERGES {
			return Err(Error::data_loss("BPE merge count exceeds the format limit"));
		}
		let mut merges = Vec::with_capacity(count);
		for rank in 0..count {
			let left = parse_u32(fields.next(), "left merge token")?;
			let right = parse_u32(fields.next(), "right merge token")?;
			let next_token = u32::try_from(BASE_VOCAB_SIZE + rank)
				.map_err(|_| Error::data_loss("BPE token rank exceeds u32"))?;
			if left >= next_token || right >= next_token {
				return Err(Error::data_loss(
					"BPE merge references its own or a later token",
				));
			}
			merges.push(BpeMerge { left, right });
		}
		if fields.next().is_some() {
			return Err(Error::data_loss("BPE vocabulary has trailing fields"));
		}
		self.target_vocab = BASE_VOCAB_SIZE + merges.len();
		self.merges = merges;
		Ok(())
	}

	/// Current vocabulary size, including the 256 base bytes.
	pub const fn vocab_size(&self) -> usize {
		BASE_VOCAB_SIZE + self.merges.len()
	}

	/// Number of learned merge ranks.
	pub const fn num_merges(&self) -> usize {
		self.merges.len()
	}

	/// Borrow the deterministic learned merge sequence.
	pub fn merges(&self) -> &[BpeMerge] {
		&self.merges
	}

	/// Return the exact decoded byte width of one valid token.
	///
	/// Unknown tokens have width zero, matching decode's skip behavior.
	///
	/// # Errors
	///
	/// Returns [`crate::ErrorKind::ResourceExhausted`] when the expansion width
	/// or temporary traversal storage cannot be represented.
	pub fn token_byte_len(&self, token: u32) -> Result<usize> {
		let mut length = 0_usize;
		let mut pending = vec![token];
		while let Some(token) = pending.pop() {
			if token < BASE_VOCAB_SIZE as u32 {
				length = length.checked_add(1).ok_or_else(|| {
					Error::resource_exhausted("BPE token byte length overflows usize")
				})?;
				continue;
			}
			let Ok(rank) = usize::try_from(token - BASE_VOCAB_SIZE as u32) else {
				continue;
			};
			let Some(merge) = self.merges.get(rank) else {
				continue;
			};
			pending
				.try_reserve(2)
				.map_err(|_| Error::resource_exhausted("BPE token traversal allocation failed"))?;
			pending.push(merge.right);
			pending.push(merge.left);
		}
		Ok(length)
	}
}

impl Default for BpeTokenizer {
	fn default() -> Self {
		Self::new(512)
	}
}

fn apply_merge(ids: &[u32], merge: BpeMerge, new_token: u32) -> Vec<u32> {
	let mut output = Vec::with_capacity(ids.len());
	let mut index = 0;
	while index < ids.len() {
		if index + 1 < ids.len() && ids[index] == merge.left && ids[index + 1] == merge.right {
			output.push(new_token);
			index += 2;
		} else {
			output.push(ids[index]);
			index += 1;
		}
	}
	output
}

fn parse_usize(field: Option<&str>, label: &str) -> Result<usize> {
	field
		.and_then(|field| field.parse().ok())
		.ok_or_else(|| Error::data_loss(format!("BPE vocabulary has an invalid {label}")))
}

fn parse_u32(field: Option<&str>, label: &str) -> Result<u32> {
	field
		.and_then(|field| field.parse().ok())
		.ok_or_else(|| Error::data_loss(format!("BPE vocabulary has an invalid {label}")))
}
