use pyo3::prelude::*;

use crate::error::python_error;

/// Deterministic byte-pair tokenizer (BPE) compatible with `oa_bpe_v1`.
#[pyclass(name = "BpeTokenizer", unsendable)]
pub(crate) struct PythonBpeTokenizer {
	pub(crate) inner: oa::ml::BpeTokenizer,
}

#[pymethods]
impl PythonBpeTokenizer {
	/// Construct an untrained tokenizer targeting `target_vocab` tokens.
	#[new]
	#[pyo3(signature = (target_vocab = 512))]
	pub fn new(target_vocab: usize) -> Self {
		Self {
			inner: oa::ml::BpeTokenizer::new(target_vocab),
		}
	}

	/// Learn up to `num_merges` pair ranks from raw bytes.
	pub fn train(&mut self, bytes: &[u8], num_merges: usize) {
		self.inner.train(bytes, num_merges);
	}

	/// Learn up to `num_merges` pair ranks from UTF-8 text.
	pub fn train_text(&mut self, text: &str, num_merges: usize) {
		self.inner.train_text(text, num_merges);
	}

	/// Encode raw bytes to a list of token IDs.
	pub fn encode(&self, bytes: &[u8]) -> Vec<u32> {
		self.inner.encode(bytes)
	}

	/// Encode UTF-8 text to a list of token IDs.
	pub fn encode_text(&self, text: &str) -> Vec<u32> {
		self.inner.encode_text(text)
	}

	/// Right-align an encoded prompt in a zero-padded fixed context.
	pub fn encode_prompt(&self, bytes: &[u8], context_length: usize) -> Vec<u32> {
		self.inner.encode_prompt(bytes, context_length)
	}

	/// Decode token IDs back to raw bytes.
	pub fn decode(&self, tokens: Vec<u32>) -> PyResult<Vec<u8>> {
		self.inner.decode(&tokens).map_err(python_error)
	}

	/// Decode token IDs back to UTF-8 text.
	pub fn decode_text(&self, tokens: Vec<u32>) -> PyResult<String> {
		self.inner.decode_text(&tokens).map_err(python_error)
	}

	/// Return the number of learned merges.
	pub fn num_merges(&self) -> usize {
		self.inner.num_merges()
	}

	/// Return the effective vocabulary size (256 + num_merges).
	pub fn vocab_size(&self) -> usize {
		self.inner.vocab_size()
	}

	/// Save the vocabulary in `oa_bpe_v1` format.
	pub fn save(&self, path: &str) -> PyResult<()> {
		self.inner.save(path).map_err(python_error)
	}

	/// Load vocabulary from an `oa_bpe_v1` file.
	pub fn load(&mut self, path: &str) -> PyResult<()> {
		self.inner.load(path).map_err(python_error)
	}

	pub fn __repr__(&self) -> String {
		format!(
			"BpeTokenizer(vocab_size={}, num_merges={})",
			self.inner.vocab_size(),
			self.inner.num_merges()
		)
	}
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonBpeTokenizer>()?;
	Ok(())
}
