use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix, runtime::PythonEngine};

use super::autograd::PythonParameter;

// ── helpers ───────────────────────────────────────────────────────────────────

fn all_parameters_of(module: &impl oa::ml::Module) -> PyResult<Vec<PythonParameter>> {
	module
		.all_parameters()
		.map(|params| {
			params
				.into_iter()
				.map(|inner| PythonParameter { inner })
				.collect()
		})
		.map_err(python_error)
}

// ── CharSampler ───────────────────────────────────────────────────────────────

/// Deterministic dense all-position next-character sampler over the canonical corpus.
#[pyclass(name = "CharSampler", unsendable)]
pub(crate) struct PythonCharSampler {
	pub(crate) inner: oa::sdk::ml::nlp::CharSampler,
}

#[pymethods]
impl PythonCharSampler {
	/// Construct a sampler over the canonical corpus with the given batch size.
	#[new]
	pub fn new(batch_size: usize) -> PyResult<Self> {
		oa::sdk::ml::nlp::CharSampler::new(batch_size)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Upload the next `[batch, 16]` U32 input and shifted-target pair.
	pub fn next(&mut self, engine: &PythonEngine) -> PyResult<(PythonMatrix, PythonMatrix)> {
		self
			.inner
			.next(&engine.inner)
			.map(|(a, b)| (PythonMatrix::wrap(a), PythonMatrix::wrap(b)))
			.map_err(python_error)
	}

	/// Produce the next host-side input and shifted-target token arrays without uploading.
	pub fn next_values(&mut self) -> PyResult<(Vec<u32>, Vec<u32>)> {
		self.inner.next_values().map_err(python_error)
	}

	pub fn __repr__(&self) -> &str {
		"CharSampler"
	}
}

// ── ByteSampler ───────────────────────────────────────────────────────────────

/// Deterministic dense all-position next-byte sampler over the canonical corpus.
#[pyclass(name = "ByteSampler", unsendable)]
pub(crate) struct PythonByteSampler {
	pub(crate) inner: oa::sdk::ml::nlp::ByteSampler,
}

#[pymethods]
impl PythonByteSampler {
	/// Construct the canonical raw-byte sampler with the given batch size.
	#[new]
	pub fn new(batch_size: usize) -> PyResult<Self> {
		oa::sdk::ml::nlp::ByteSampler::new(batch_size)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Upload the next packed-U8 input and U32 shifted-target pair.
	pub fn next(&mut self, engine: &PythonEngine) -> PyResult<(PythonMatrix, PythonMatrix)> {
		self
			.inner
			.next(&engine.inner)
			.map(|(a, b)| (PythonMatrix::wrap(a), PythonMatrix::wrap(b)))
			.map_err(python_error)
	}

	/// Produce the next host values for stable captured-program input slots.
	pub fn next_values(&mut self) -> PyResult<(Vec<u8>, Vec<u32>)> {
		self.inner.next_values().map_err(python_error)
	}

	pub fn __repr__(&self) -> &str {
		"ByteSampler"
	}
}

// ── BpeSampler ────────────────────────────────────────────────────────────────

/// Deterministic dense all-position sampler over canonical BPE tokens.
///
/// Borrows the tokenizer from the Python `BpeTokenizer` object.
///
/// The Python object is kept alive via a `Py<PythonBpeTokenizer>` handle so
/// the raw pointer remains valid for the sampler's lifetime.
#[pyclass(name = "BpeSampler", unsendable)]
pub(crate) struct PythonBpeSampler {
	// Keep the Python tokenizer object alive so the raw ptr inside `inner`
	// stays valid. Pyo3 pyclass objects are heap-allocated and never moved.
	_tokenizer: Py<super::tokenizer::PythonBpeTokenizer>,
	pub(crate) inner: oa::sdk::ml::nlp::BpeSampler<'static>,
}

#[pymethods]
impl PythonBpeSampler {
	/// Construct a BPE sampler. `tokenizer` must already be trained.
	///
	/// The sampler borrows the tokenizer; subsequent changes to the Python-side
	/// tokenizer via `train` will affect this sampler.
	#[new]
	pub fn new(
		py: Python<'_>,
		batch_size: usize,
		tokenizer: Py<super::tokenizer::PythonBpeTokenizer>,
	) -> PyResult<Self> {
		let tok_ptr: *const oa::ml::BpeTokenizer = {
			let borrowed = tokenizer.bind(py).borrow();
			// SAFETY: the tokenizer is pinned by the pyo3 heap allocation and kept
			// alive by `_tokenizer`.  The raw pointer is valid for as long as
			// `_tokenizer` is held (i.e. the lifetime of this struct).
			&borrowed.inner as *const oa::ml::BpeTokenizer
		};
		let tok_ref: &'static oa::ml::BpeTokenizer = unsafe { &*tok_ptr };
		let inner = oa::sdk::ml::nlp::BpeSampler::new(batch_size, tok_ref).map_err(python_error)?;
		Ok(Self {
			_tokenizer: tokenizer,
			inner,
		})
	}

	/// Produce and upload the next U32 BPE input/target pair.
	pub fn next(&mut self, engine: &PythonEngine) -> PyResult<(PythonMatrix, PythonMatrix)> {
		self
			.inner
			.next(&engine.inner)
			.map(|(a, b)| (PythonMatrix::wrap(a), PythonMatrix::wrap(b)))
			.map_err(python_error)
	}

	/// Produce the next host values for captured-program input slots.
	pub fn next_values(&mut self) -> PyResult<(Vec<u32>, Vec<u32>)> {
		self.inner.next_values().map_err(python_error)
	}

	/// Exact source bytes represented by the most recent shifted targets.
	#[getter]
	pub fn last_batch_bytes(&self) -> usize {
		self.inner.last_batch_bytes()
	}

	/// Mean source bytes per target token in the last batch.
	pub fn last_batch_bytes_per_token(&self) -> f64 {
		self.inner.last_batch_bytes_per_token()
	}

	/// Return exact source bytes represented by the next target batch without
	/// advancing the sampler.
	pub fn next_batch_bytes(&self) -> PyResult<usize> {
		self.inner.next_batch_bytes().map_err(python_error)
	}

	pub fn __repr__(&self) -> &'static str {
		"BpeSampler()"
	}
}

// ── CharRnn ───────────────────────────────────────────────────────────────────

/// Canonical character language model: Embedding → Elman RNN → Linear.
#[pyclass(name = "CharRnn", unsendable)]
pub(crate) struct PythonCharRnn {
	pub(crate) inner: oa::sdk::ml::nlp::CharRnn,
}

#[pymethods]
impl PythonCharRnn {
	/// Construct the canonical deterministic 8,891-parameter model.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::CharRnn::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-character logits for every token position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"CharRnn"
	}
}

// ── CharGru ───────────────────────────────────────────────────────────────────

/// Canonical character language model: Embedding → GRU → Linear.
#[pyclass(name = "CharGru", unsendable)]
pub(crate) struct PythonCharGru {
	pub(crate) inner: oa::sdk::ml::nlp::CharGru,
}

#[pymethods]
impl PythonCharGru {
	/// Construct the deterministic 21,435-parameter model.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::CharGru::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-character logits for every token position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"CharGru"
	}
}

// ── CharTransformer ───────────────────────────────────────────────────────────

/// Canonical 10,875-parameter character Transformer language model.
#[pyclass(name = "CharTransformer", unsendable)]
pub(crate) struct PythonCharTransformer {
	pub(crate) inner: oa::sdk::ml::nlp::CharTransformer,
}

#[pymethods]
impl PythonCharTransformer {
	/// Construct the exact one-block one-head OA C++ tutorial architecture.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::CharTransformer::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate all-position next-character logits for `[B, 16]` tokens.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"CharTransformer"
	}
}

// ── CharMoeTransformer ────────────────────────────────────────────────────────

/// Canonical character Transformer with four sparse top-two experts.
#[pyclass(name = "CharMoeTransformer", unsendable)]
pub(crate) struct PythonCharMoeTransformer {
	pub(crate) inner: oa::sdk::ml::nlp::CharMoeTransformer,
}

#[pymethods]
impl PythonCharMoeTransformer {
	/// Construct the donor one-block one-head MoE Transformer recipe.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::CharMoeTransformer::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate all-position next-character logits for `[B, 16]` tokens.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"CharMoeTransformer"
	}
}

// ── CharMamba3 ────────────────────────────────────────────────────────────────

/// Canonical character Mamba-3 language model with a flat residual head.
#[pyclass(name = "CharMamba3", unsendable)]
pub(crate) struct PythonCharMamba3 {
	pub(crate) inner: oa::sdk::ml::nlp::CharMamba3,
}

#[pymethods]
impl PythonCharMamba3 {
	/// Construct the donor `32/state=32/expand=2/head=16` SISO recipe.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::CharMamba3::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-character logits for every token position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"CharMamba3"
	}
}

// ── ByteRnn ───────────────────────────────────────────────────────────────────

/// Canonical byte language model: ByteEmbedding → Elman RNN → ByteHead.
#[pyclass(name = "ByteRnn", unsendable)]
pub(crate) struct PythonByteRnn {
	pub(crate) inner: oa::sdk::ml::nlp::ByteRnn,
}

#[pymethods]
impl PythonByteRnn {
	/// Construct the deterministic donor-shaped 31,104-parameter model.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::ByteRnn::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-byte logits for every token position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"ByteRnn"
	}
}

// ── ByteGru ───────────────────────────────────────────────────────────────────

/// Canonical byte language model: ByteEmbedding → GRU → ByteHead.
#[pyclass(name = "ByteGru", unsendable)]
pub(crate) struct PythonByteGru {
	pub(crate) inner: oa::sdk::ml::nlp::ByteGru,
}

#[pymethods]
impl PythonByteGru {
	/// Construct the deterministic donor-shaped 43,648-parameter model.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::ByteGru::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-byte logits for every token position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"ByteGru"
	}
}

// ── ByteTransformer ───────────────────────────────────────────────────────────

/// Canonical byte Transformer language model.
#[pyclass(name = "ByteTransformer", unsendable)]
pub(crate) struct PythonByteTransformer {
	pub(crate) inner: oa::sdk::ml::nlp::ByteTransformer,
}

#[pymethods]
impl PythonByteTransformer {
	/// Construct the deterministic donor one-block one-head byte model.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::ByteTransformer::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate all-position logits for packed-U8 or U32 byte tokens.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"ByteTransformer"
	}
}

// ── ByteMoeTransformer ────────────────────────────────────────────────────────

/// Canonical byte Transformer with four sparse top-two experts.
#[pyclass(name = "ByteMoeTransformer", unsendable)]
pub(crate) struct PythonByteMoeTransformer {
	pub(crate) inner: oa::sdk::ml::nlp::ByteMoeTransformer,
}

#[pymethods]
impl PythonByteMoeTransformer {
	/// Construct the donor one-block one-head byte MoE recipe.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::ByteMoeTransformer::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate all-position logits for packed-U8 or U32 byte tokens.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"ByteMoeTransformer"
	}
}

// ── ByteMamba3 ────────────────────────────────────────────────────────────────

/// Canonical raw-byte Mamba-3 language model with a flat residual head.
#[pyclass(name = "ByteMamba3", unsendable)]
pub(crate) struct PythonByteMamba3 {
	pub(crate) inner: oa::sdk::ml::nlp::ByteMamba3,
}

#[pymethods]
impl PythonByteMamba3 {
	/// Construct the donor `32/state=32/expand=2/head=16` byte recipe.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::ByteMamba3::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-byte logits for every token position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"ByteMamba3"
	}
}

// ── ByteEmpyrealm ─────────────────────────────────────────────────────────────

/// Canonical byte language model using the donor Empyrealm-Core ownership tree.
#[pyclass(name = "ByteEmpyrealm", unsendable)]
pub(crate) struct PythonByteEmpyrealm {
	pub(crate) inner: oa::sdk::ml::nlp::ByteEmpyrealm,
}

#[pymethods]
impl PythonByteEmpyrealm {
	/// Construct the deterministic donor fidelity model.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::ByteEmpyrealm::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-byte logits for every token position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"ByteEmpyrealm"
	}
}

// ── BpeRnn ────────────────────────────────────────────────────────────────────

/// Canonical BPE language model: Embedding → Elman RNN → Linear.
#[pyclass(name = "BpeRnn", unsendable)]
pub(crate) struct PythonBpeRnn {
	pub(crate) inner: oa::sdk::ml::nlp::BpeRnn,
}

#[pymethods]
impl PythonBpeRnn {
	/// Construct the deterministic donor-shaped 37,312-parameter model.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::BpeRnn::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-token logits for every BPE position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"BpeRnn"
	}
}

// ── BpeGru ────────────────────────────────────────────────────────────────────

/// Canonical BPE language model: Embedding → GRU → Linear.
#[pyclass(name = "BpeGru", unsendable)]
pub(crate) struct PythonBpeGru {
	pub(crate) inner: oa::sdk::ml::nlp::BpeGru,
}

#[pymethods]
impl PythonBpeGru {
	/// Construct the deterministic donor-shaped 49,856-parameter model.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::BpeGru::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-token logits for every BPE position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"BpeGru"
	}
}

// ── BpeTransformer ────────────────────────────────────────────────────────────

/// Canonical BPE Transformer language model.
#[pyclass(name = "BpeTransformer", unsendable)]
pub(crate) struct PythonBpeTransformer {
	pub(crate) inner: oa::sdk::ml::nlp::BpeTransformer,
}

#[pymethods]
impl PythonBpeTransformer {
	/// Construct the deterministic donor one-block one-head BPE model.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::BpeTransformer::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate all-position logits for U32 BPE tokens.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"BpeTransformer"
	}
}

// ── BpeMoeTransformer ─────────────────────────────────────────────────────────

/// Canonical BPE Transformer with four sparse top-two experts.
#[pyclass(name = "BpeMoeTransformer", unsendable)]
pub(crate) struct PythonBpeMoeTransformer {
	pub(crate) inner: oa::sdk::ml::nlp::BpeMoeTransformer,
}

#[pymethods]
impl PythonBpeMoeTransformer {
	/// Construct the donor one-block one-head BPE MoE recipe.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::BpeMoeTransformer::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate all-position logits for U32 BPE tokens.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"BpeMoeTransformer"
	}
}

// ── BpeMamba3 ─────────────────────────────────────────────────────────────────

/// Canonical BPE Mamba-3 language model with a flat residual head.
#[pyclass(name = "BpeMamba3", unsendable)]
pub(crate) struct PythonBpeMamba3 {
	pub(crate) inner: oa::sdk::ml::nlp::BpeMamba3,
}

#[pymethods]
impl PythonBpeMamba3 {
	/// Construct the donor `32/state=32/expand=2/head=16` BPE recipe.
	#[new]
	pub fn new(engine: &PythonEngine) -> PyResult<Self> {
		oa::sdk::ml::nlp::BpeMamba3::new(&engine.inner)
			.map(|inner| Self { inner })
			.map_err(python_error)
	}

	/// Evaluate dense next-token logits for every BPE position.
	pub fn forward(&self, tokens: &PythonMatrix) -> PyResult<PythonMatrix> {
		self
			.inner
			.forward(&tokens.inner)
			.map(PythonMatrix::wrap)
			.map_err(python_error)
	}

	pub fn all_parameters(&self) -> PyResult<Vec<PythonParameter>> {
		all_parameters_of(&self.inner)
	}

	pub fn __repr__(&self) -> &str {
		"BpeMamba3"
	}
}

// ── generate_greedy ───────────────────────────────────────────────────────────

/// Greedily generate characters with the canonical left-filled/sliding window.
///
/// `model` must be one of `CharRnn`, `CharGru`, `CharTransformer`,
/// `CharMoeTransformer`, or `CharMamba3`.
#[pyfunction]
pub(crate) fn ml_nlp_generate_greedy(
	engine: &PythonEngine,
	model: &Bound<'_, PyAny>,
	prompt: &str,
	count: usize,
) -> PyResult<String> {
	// generate_greedy takes &impl Module (Sized), so dispatch per concrete type.
	macro_rules! try_char_gen {
		($($ty:ty),+ $(,)?) => {
			$(if let Ok(m) = model.extract::<PyRef<$ty>>() {
				return oa::sdk::ml::nlp::generate_greedy(&engine.inner, &m.inner, prompt, count)
					.map_err(python_error);
			})+
		};
	}
	try_char_gen!(
		PythonCharRnn,
		PythonCharGru,
		PythonCharTransformer,
		PythonCharMoeTransformer,
		PythonCharMamba3,
	);
	Err(pyo3::exceptions::PyTypeError::new_err(
		"model must be one of CharRnn, CharGru, CharTransformer, CharMoeTransformer, or CharMamba3",
	))
}

/// Greedily generate exact bytes with the canonical left-filled/sliding window.
///
/// `model` must be one of `ByteRnn`, `ByteGru`, `ByteTransformer`,
/// `ByteMoeTransformer`, `ByteMamba3`, or `ByteEmpyrealm`.
#[pyfunction]
pub(crate) fn ml_nlp_generate_bytes_greedy(
	engine: &PythonEngine,
	model: &Bound<'_, PyAny>,
	prompt: Vec<u8>,
	count: usize,
) -> PyResult<Vec<u8>> {
	macro_rules! try_byte_gen {
		($($ty:ty),+ $(,)?) => {
			$(if let Ok(m) = model.extract::<PyRef<$ty>>() {
				return oa::sdk::ml::nlp::generate_bytes_greedy(&engine.inner, &m.inner, &prompt, count)
					.map_err(python_error);
			})+
		};
	}
	try_byte_gen!(
		PythonByteRnn,
		PythonByteGru,
		PythonByteTransformer,
		PythonByteMoeTransformer,
		PythonByteMamba3,
		PythonByteEmpyrealm,
	);
	Err(pyo3::exceptions::PyTypeError::new_err(
		"model must be one of ByteRnn, ByteGru, ByteTransformer, \
		 ByteMoeTransformer, ByteMamba3, or ByteEmpyrealm",
	))
}

/// Greedily generate BPE tokens until `byte_count` decoded source bytes.
///
/// `model` must be one of `BpeRnn`, `BpeGru`, `BpeTransformer`,
/// `BpeMoeTransformer`, or `BpeMamba3`.
#[pyfunction]
pub(crate) fn ml_nlp_generate_bpe_greedy(
	engine: &PythonEngine,
	model: &Bound<'_, PyAny>,
	tokenizer: &super::tokenizer::PythonBpeTokenizer,
	prompt: Vec<u8>,
	byte_count: usize,
) -> PyResult<Vec<u8>> {
	macro_rules! try_bpe_gen {
		($($ty:ty),+ $(,)?) => {
			$(if let Ok(m) = model.extract::<PyRef<$ty>>() {
				return oa::sdk::ml::nlp::generate_bpe_greedy(
					&engine.inner, &m.inner, &tokenizer.inner, &prompt, byte_count,
				).map_err(python_error);
			})+
		};
	}
	try_bpe_gen!(
		PythonBpeRnn,
		PythonBpeGru,
		PythonBpeTransformer,
		PythonBpeMoeTransformer,
		PythonBpeMamba3,
	);
	Err(pyo3::exceptions::PyTypeError::new_err(
		"model must be one of BpeRnn, BpeGru, BpeTransformer, BpeMoeTransformer, or BpeMamba3",
	))
}

// ── register ──────────────────────────────────────────────────────────────────

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	// samplers
	module.add_class::<PythonCharSampler>()?;
	module.add_class::<PythonByteSampler>()?;
	module.add_class::<PythonBpeSampler>()?;
	// char model presets
	module.add_class::<PythonCharRnn>()?;
	module.add_class::<PythonCharGru>()?;
	module.add_class::<PythonCharTransformer>()?;
	module.add_class::<PythonCharMoeTransformer>()?;
	module.add_class::<PythonCharMamba3>()?;
	// byte model presets
	module.add_class::<PythonByteRnn>()?;
	module.add_class::<PythonByteGru>()?;
	module.add_class::<PythonByteTransformer>()?;
	module.add_class::<PythonByteMoeTransformer>()?;
	module.add_class::<PythonByteMamba3>()?;
	module.add_class::<PythonByteEmpyrealm>()?;
	// BPE model presets
	module.add_class::<PythonBpeRnn>()?;
	module.add_class::<PythonBpeGru>()?;
	module.add_class::<PythonBpeTransformer>()?;
	module.add_class::<PythonBpeMoeTransformer>()?;
	module.add_class::<PythonBpeMamba3>()?;
	// generation functions
	module.add_function(wrap_pyfunction!(ml_nlp_generate_greedy, module)?)?;
	module.add_function(wrap_pyfunction!(ml_nlp_generate_bytes_greedy, module)?)?;
	module.add_function(wrap_pyfunction!(ml_nlp_generate_bpe_greedy, module)?)?;
	Ok(())
}
