use std::{cell::RefCell, collections::HashMap, fs, path::Path};

use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization as _;

use crate::{Error, Result};

const MERGE_COUNT: usize = 48_894;
const VOCAB_SIZE: usize = 49_408;
const BOS_TOKEN: i32 = 49_406;
const EOS_TOKEN: i32 = 49_407;

/// Host token batch consumed by [`super::ClipText::forward_tokens`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipTokenBatch {
	/// Row-major token IDs with shape `[batch, context_length]`.
	pub token_ids: Vec<i32>,
	/// Flattened position of the effective EOS token in every row.
	pub flat_eos_rows: Vec<i32>,
	/// Number of prompt rows.
	pub batch: usize,
	/// Fixed padded row width.
	pub context_length: usize,
}

/// Native OpenAI CLIP byte-BPE tokenizer.
///
/// The merges input is the canonical `bpe_simple_vocab_16e6` text format. It
/// remains an explicit host asset rather than being hidden inside model state.
pub struct ClipTokenizer {
	byte_map: Vec<String>,
	ranks: HashMap<(String, String), usize>,
	encoder: HashMap<String, i32>,
	cache: RefCell<HashMap<String, Vec<String>>>,
	loaded: bool,
}

impl Default for ClipTokenizer {
	fn default() -> Self {
		Self::new()
	}
}

impl ClipTokenizer {
	/// Construct an unloaded tokenizer.
	pub fn new() -> Self {
		Self {
			byte_map: Vec::new(),
			ranks: HashMap::new(),
			encoder: HashMap::new(),
			cache: RefCell::new(HashMap::new()),
			loaded: false,
		}
	}

	/// Load the canonical CLIP merge table from a file.
	///
	/// # Errors
	///
	/// Returns an error for filesystem failure or malformed merge content.
	pub fn load_merges_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
		let bytes = fs::read(path).map_err(|source| Error::io("read CLIP merges", source))?;
		self.load_merges(&bytes)
	}

	/// Load the canonical CLIP merge table from bytes.
	///
	/// # Errors
	///
	/// Returns an error unless the input is UTF-8 with exactly 48,894 merge
	/// entries following its header.
	pub fn load_merges(&mut self, bytes: &[u8]) -> Result<()> {
		*self = Self::new();
		let text = std::str::from_utf8(bytes)
			.map_err(|_| Error::invalid_argument("CLIP merges must be valid UTF-8"))?;
		let byte_map = byte_encoder();
		let order = byte_order();
		let mut vocab = Vec::with_capacity(VOCAB_SIZE);
		for &byte in &order {
			vocab.push(byte_map[byte as usize].clone());
		}
		for &byte in &order {
			vocab.push(format!("{}</w>", byte_map[byte as usize]));
		}

		let mut lines = text.lines();
		lines
			.next()
			.ok_or_else(|| Error::invalid_argument("CLIP merges are missing the format header"))?;
		let mut ranks = HashMap::with_capacity(MERGE_COUNT);
		let mut rank = 0_usize;
		for line in lines {
			if line.is_empty() {
				continue;
			}
			if rank == MERGE_COUNT {
				break;
			}
			let (left, right) = line
				.split_once(' ')
				.ok_or_else(|| Error::invalid_argument("malformed CLIP merge entry"))?;
			if left.is_empty() || right.is_empty() || right.contains(' ') {
				return Err(Error::invalid_argument("malformed CLIP merge entry"));
			}
			ranks
				.entry((left.to_owned(), right.to_owned()))
				.or_insert(rank);
			vocab.push(format!("{left}{right}"));
			rank += 1;
		}
		if rank != MERGE_COUNT {
			return Err(Error::invalid_argument(
				"CLIP merges must contain 48,894 entries",
			));
		}
		vocab.push("<|startoftext|>".to_owned());
		vocab.push("<|endoftext|>".to_owned());
		if vocab.len() != VOCAB_SIZE {
			return Err(Error::invalid_argument("CLIP vocabulary size mismatch"));
		}
		let mut encoder = HashMap::with_capacity(VOCAB_SIZE);
		for (index, token) in vocab.into_iter().enumerate() {
			encoder.entry(token).or_insert(
				i32::try_from(index)
					.map_err(|_| Error::resource_exhausted("CLIP vocabulary index exceeds I32"))?,
			);
		}

		self.byte_map = byte_map;
		self.ranks = ranks;
		self.encoder = encoder;
		self.loaded = true;
		Ok(())
	}

	/// Encode and EOS-pad one or more prompts.
	///
	/// # Errors
	///
	/// Returns an error when no merge table is loaded, batch geometry is empty,
	/// a BPE piece is unknown, arithmetic overflows, or a prompt exceeds the
	/// context while `truncate` is false.
	pub fn encode<S: AsRef<str>>(
		&self,
		prompts: &[S],
		context_length: usize,
		truncate: bool,
	) -> Result<ClipTokenBatch> {
		if !self.loaded {
			return Err(Error::failed_precondition("CLIP tokenizer is not loaded"));
		}
		if prompts.is_empty() || context_length < 2 {
			return Err(Error::invalid_argument("invalid CLIP token batch shape"));
		}
		let count = prompts
			.len()
			.checked_mul(context_length)
			.ok_or_else(|| Error::resource_exhausted("CLIP token batch size overflows usize"))?;
		let mut token_ids = vec![EOS_TOKEN; count];
		let mut flat_eos_rows = Vec::with_capacity(prompts.len());
		for (batch_index, prompt) in prompts.iter().enumerate() {
			let mut ids = vec![BOS_TOKEN];
			for token in pretokenize(prompt.as_ref()) {
				let mut encoded = String::new();
				for byte in token.bytes() {
					encoded.push_str(&self.byte_map[byte as usize]);
				}
				for piece in self.bpe(&encoded) {
					ids.push(
						*self
							.encoder
							.get(&piece)
							.ok_or_else(|| Error::failed_precondition("CLIP BPE emitted an unknown piece"))?,
					);
				}
			}
			ids.push(EOS_TOKEN);
			if ids.len() > context_length {
				if !truncate {
					return Err(Error::invalid_argument(
						"CLIP prompt exceeds context length",
					));
				}
				ids.truncate(context_length);
				ids[context_length - 1] = EOS_TOKEN;
			}
			let row = batch_index * context_length;
			token_ids[row..row + ids.len()].copy_from_slice(&ids);
			flat_eos_rows.push(
				i32::try_from(row + ids.len() - 1)
					.map_err(|_| Error::resource_exhausted("CLIP EOS row exceeds I32"))?,
			);
		}
		Ok(ClipTokenBatch {
			token_ids,
			flat_eos_rows,
			batch: prompts.len(),
			context_length,
		})
	}

	/// Whether a valid canonical merge table is loaded.
	pub const fn is_loaded(&self) -> bool {
		self.loaded
	}

	/// Return 49,408 when loaded and zero otherwise.
	pub const fn vocab_size(&self) -> usize {
		if self.loaded { VOCAB_SIZE } else { 0 }
	}

	/// Return the canonical BOS token, or -1 while unloaded.
	pub const fn bos_token(&self) -> i32 {
		if self.loaded { BOS_TOKEN } else { -1 }
	}

	/// Return the canonical EOS/padding token, or -1 while unloaded.
	pub const fn eos_token(&self) -> i32 {
		if self.loaded { EOS_TOKEN } else { -1 }
	}

	fn bpe(&self, token: &str) -> Vec<String> {
		if let Some(cached) = self.cache.borrow().get(token) {
			return cached.clone();
		}
		let mut word = token
			.chars()
			.map(|value| value.to_string())
			.collect::<Vec<_>>();
		if word.is_empty() {
			return word;
		}
		word.last_mut().expect("nonempty word").push_str("</w>");
		while word.len() > 1 {
			let mut best: Option<(usize, String, String)> = None;
			for pair in word.windows(2) {
				if let Some(&rank) = self.ranks.get(&(pair[0].clone(), pair[1].clone()))
					&& best.as_ref().is_none_or(|candidate| rank < candidate.0)
				{
					best = Some((rank, pair[0].clone(), pair[1].clone()));
				}
			}
			let Some((_, left, right)) = best else { break };
			let mut merged = Vec::with_capacity(word.len());
			let mut index = 0;
			while index < word.len() {
				if index + 1 < word.len() && word[index] == left && word[index + 1] == right {
					merged.push(format!("{}{}", word[index], word[index + 1]));
					index += 2;
				} else {
					merged.push(word[index].clone());
					index += 1;
				}
			}
			word = merged;
		}
		self
			.cache
			.borrow_mut()
			.insert(token.to_owned(), word.clone());
		word
	}
}

fn byte_order() -> Vec<u16> {
	let mut bytes = (b'!'..=b'~').map(u16::from).collect::<Vec<_>>();
	bytes.extend(0xA1_u16..=0xAC);
	bytes.extend(0xAE_u16..=0xFF);
	for byte in 0_u16..=255 {
		if !bytes.contains(&byte) {
			bytes.push(byte);
		}
	}
	bytes
}

fn byte_encoder() -> Vec<String> {
	let bytes = byte_order();
	let mut codepoints = bytes.clone();
	for (index, codepoint) in codepoints.iter_mut().enumerate().skip(188) {
		*codepoint = 256 + u16::try_from(index - 188).expect("byte alphabet is bounded");
	}
	let mut result = vec![String::new(); 256];
	for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
		result[byte as usize] = char::from_u32(u32::from(codepoint))
			.expect("CLIP byte alphabet codepoint is valid")
			.to_string();
	}
	result
}

fn pretokenize(text: &str) -> Vec<String> {
	let units = text
		.nfc()
		.map(|value| value.to_lowercase().next().unwrap_or(value))
		.collect::<Vec<_>>();
	let mut tokens = Vec::new();
	let mut index = 0;
	while index < units.len() {
		if is_whitespace(units[index]) {
			index += 1;
			continue;
		}
		if units[index] == '\'' {
			let suffixes = ["s", "t", "re", "ve", "m", "ll", "d"];
			if let Some(suffix) = suffixes.iter().find(|suffix| {
				let end = index + 1 + suffix.chars().count();
				end <= units.len() && units[index + 1..end].iter().copied().eq(suffix.chars())
			}) {
				tokens.push(format!("'{suffix}"));
				index += 1 + suffix.chars().count();
				continue;
			}
		}
		let letter = is_letter(units[index]);
		let number = is_number(units[index]);
		let mut token = units[index].to_string();
		index += 1;
		if letter {
			while index < units.len() && is_letter(units[index]) {
				token.push(units[index]);
				index += 1;
			}
		} else if !number {
			while index < units.len()
				&& !is_whitespace(units[index])
				&& !is_letter(units[index])
				&& !is_number(units[index])
			{
				token.push(units[index]);
				index += 1;
			}
		}
		tokens.push(token);
	}
	tokens
}

fn is_letter(value: char) -> bool {
	matches!(
		get_general_category(value),
		GeneralCategory::UppercaseLetter
			| GeneralCategory::LowercaseLetter
			| GeneralCategory::TitlecaseLetter
			| GeneralCategory::ModifierLetter
			| GeneralCategory::OtherLetter
	)
}

fn is_whitespace(value: char) -> bool {
	matches!(value, '\t' | '\n' | '\r' | '\u{000C}' | '\u{000B}')
		|| matches!(
			get_general_category(value),
			GeneralCategory::SpaceSeparator
				| GeneralCategory::LineSeparator
				| GeneralCategory::ParagraphSeparator
		)
}

fn is_number(value: char) -> bool {
	matches!(
		get_general_category(value),
		GeneralCategory::DecimalNumber | GeneralCategory::LetterNumber | GeneralCategory::OtherNumber
	)
}
