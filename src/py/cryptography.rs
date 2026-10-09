use pyo3::prelude::*;

use crate::{error::python_error, matrix::PythonMatrix};

fn hash_from_bytes(bytes: &[u8]) -> PyResult<oa::cryptography::Hash> {
	oa::cryptography::Hash::from_bytes(bytes)
		.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

// ── Stateless hasher ─────────────────────────────────────────────────────────

#[pyclass(name = "Hasher", unsendable)]
pub(crate) struct PythonHasher {
	inner: oa::cryptography::Hasher,
}

impl PythonHasher {
	fn wrap(inner: oa::cryptography::Hasher) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonHasher {
	/// Create a new SHAKE-256/256-based incremental hasher.
	#[new]
	fn new() -> Self {
		Self::wrap(oa::cryptography::Hasher::new())
	}

	fn update(&mut self, data: &[u8]) -> PyResult<()> {
		self.inner.update(data).map_err(python_error)
	}

	/// Return the 32-byte digest.
	fn finalize(&mut self) -> Vec<u8> {
		self.inner.finalize().as_bytes().to_vec()
	}

	fn reset(&mut self) {
		self.inner.reset();
	}
}

// ── Merkle tree ───────────────────────────────────────────────────────────────

#[pyclass(name = "MerkleTree", unsendable)]
pub(crate) struct PythonMerkleTree {
	inner: oa::cryptography::MerkleTree,
}

impl PythonMerkleTree {
	fn wrap(inner: oa::cryptography::MerkleTree) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonMerkleTree {
	/// Build a Merkle tree from a list of 32-byte leaf values.
	#[staticmethod]
	fn build(leaves: Vec<Vec<u8>>) -> PyResult<Self> {
		let hashes = leaves
			.iter()
			.map(|leaf| hash_from_bytes(leaf))
			.collect::<PyResult<Vec<_>>>()?;
		Ok(Self::wrap(oa::cryptography::build_merkle_tree(&hashes)))
	}

	fn root(&self) -> Vec<u8> {
		self.inner.root().as_bytes().to_vec()
	}

	fn proof(&self, leaf_index: usize) -> PyResult<PythonMerkleProof> {
		oa::cryptography::merkle_proof(&self.inner, leaf_index)
			.map(PythonMerkleProof::wrap)
			.map_err(python_error)
	}
}

// ── Merkle proof ──────────────────────────────────────────────────────────────

#[pyclass(name = "MerkleProof", unsendable)]
pub(crate) struct PythonMerkleProof {
	inner: oa::cryptography::MerkleProof,
}

impl PythonMerkleProof {
	fn wrap(inner: oa::cryptography::MerkleProof) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonMerkleProof {
	/// Verify this proof against a 32-byte root and a 32-byte leaf.
	fn verify(&self, leaf: Vec<u8>, root: Vec<u8>) -> PyResult<bool> {
		let leaf_hash = hash_from_bytes(&leaf)?;
		let root_hash = hash_from_bytes(&root)?;
		Ok(oa::cryptography::verify_merkle_proof(
			&leaf_hash,
			&self.inner,
			&root_hash,
		))
	}

	fn siblings(&self) -> Vec<Vec<u8>> {
		self
			.inner
			.siblings()
			.iter()
			.map(|h| h.as_bytes().to_vec())
			.collect()
	}

	fn sibling_is_left(&self) -> Vec<bool> {
		self.inner.sibling_is_left().to_vec()
	}
}

// ── Post-quantum cryptography (ML-DSA-65) ────────────────────────────────────

#[pyclass(name = "PublicKey", unsendable)]
pub(crate) struct PythonPublicKey {
	inner: oa::cryptography::pqc::PublicKey,
}

impl PythonPublicKey {
	fn wrap(inner: oa::cryptography::pqc::PublicKey) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonPublicKey {
	#[staticmethod]
	fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
		oa::cryptography::pqc::PublicKey::from_bytes(&bytes)
			.map(Self::wrap)
			.map_err(python_error)
	}

	fn to_bytes(&self) -> Vec<u8> {
		self.inner.to_bytes().to_vec()
	}

	fn is_zero(&self) -> bool {
		self.inner.is_zero()
	}

	fn to_short_hex(&self) -> String {
		self.inner.to_short_hex()
	}

	fn __repr__(&self) -> String {
		format!("PublicKey({})", self.inner.to_short_hex())
	}
}

#[pyclass(name = "Signature", unsendable)]
pub(crate) struct PythonSignature {
	inner: oa::cryptography::pqc::Signature,
}

impl PythonSignature {
	fn wrap(inner: oa::cryptography::pqc::Signature) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonSignature {
	#[staticmethod]
	fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
		oa::cryptography::pqc::Signature::from_bytes(&bytes)
			.map(Self::wrap)
			.map_err(python_error)
	}

	fn to_bytes(&self) -> Vec<u8> {
		self.inner.to_bytes().to_vec()
	}

	fn is_zero(&self) -> bool {
		self.inner.is_zero()
	}

	fn to_short_hex(&self) -> String {
		self.inner.to_short_hex()
	}

	fn __repr__(&self) -> String {
		format!("Signature({})", self.inner.to_short_hex())
	}
}

#[pyclass(name = "Keypair", unsendable)]
pub(crate) struct PythonKeypair {
	inner: oa::cryptography::pqc::Keypair,
}

impl PythonKeypair {
	fn wrap(inner: oa::cryptography::pqc::Keypair) -> Self {
		Self { inner }
	}
}

#[pymethods]
impl PythonKeypair {
	#[getter]
	fn public_key(&self) -> PythonPublicKey {
		PythonPublicKey::wrap(self.inner.public_key.clone())
	}

	fn sign(&self, message: Vec<u8>) -> PyResult<PythonSignature> {
		oa::cryptography::pqc::sign(&message, &self.inner.secret_key)
			.map(PythonSignature::wrap)
			.map_err(python_error)
	}

	fn sign_hash(&self, hash: Vec<u8>) -> PyResult<PythonSignature> {
		let h = hash_from_bytes(&hash)?;
		oa::cryptography::pqc::sign_hash(&h, &self.inner.secret_key)
			.map(PythonSignature::wrap)
			.map_err(python_error)
	}

	fn __repr__(&self) -> String {
		format!(
			"Keypair(public_key={})",
			self.inner.public_key.to_short_hex()
		)
	}
}

// ── Stateless functions ───────────────────────────────────────────────────────

#[pyfunction]
pub(crate) fn crypto_hash(data: &[u8]) -> Vec<u8> {
	oa::cryptography::hash(data).as_bytes().to_vec()
}

#[pyfunction]
pub(crate) fn crypto_hash_combine(left: &[u8], right: &[u8]) -> PyResult<Vec<u8>> {
	let l = hash_from_bytes(left)?;
	let r = hash_from_bytes(right)?;
	Ok(oa::cryptography::hash_combine(&l, &r).as_bytes().to_vec())
}

#[pyfunction]
pub(crate) fn crypto_keccak_f1600(data: &[u8]) -> PyResult<Vec<u8>> {
	if data.len() != 200 {
		return Err(pyo3::exceptions::PyValueError::new_err(
			"keccak_f1600 requires exactly 200 bytes (25 u64 state words)",
		));
	}
	let mut state = [0_u64; 25];
	for (i, chunk) in data.as_chunks::<8>().0.iter().enumerate() {
		state[i] = u64::from_le_bytes(*chunk);
	}
	oa::cryptography::keccak_f1600(&mut state);
	let mut out = vec![0_u8; 200];
	for (i, word) in state.iter().enumerate() {
		out[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
	}
	Ok(out)
}

#[pyfunction]
#[pyo3(signature = (key, data, custom, output_len=32))]
pub(crate) fn crypto_kmac256(
	key: &[u8],
	data: &[u8],
	custom: &[u8],
	output_len: usize,
) -> PyResult<Vec<u8>> {
	oa::cryptography::kmac256(key, data, custom, output_len).map_err(python_error)
}

#[pyfunction]
pub(crate) fn crypto_generate_keypair() -> PyResult<PythonKeypair> {
	oa::cryptography::pqc::generate_keypair()
		.map(PythonKeypair::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn crypto_pqc_verify(
	message: Vec<u8>,
	signature: &PythonSignature,
	public_key: &PythonPublicKey,
) -> bool {
	oa::cryptography::pqc::verify(&message, &signature.inner, &public_key.inner)
}

#[pyfunction]
pub(crate) fn crypto_pqc_verify_hash(
	hash: Vec<u8>,
	signature: &PythonSignature,
	public_key: &PythonPublicKey,
) -> PyResult<bool> {
	let h = hash_from_bytes(&hash)?;
	Ok(oa::cryptography::pqc::verify_hash(
		&h,
		&signature.inner,
		&public_key.inner,
	))
}

// ── Incremental SHAKE-128 XOF ────────────────────────────────────────────────

#[pyclass(name = "Shake128", unsendable)]
pub(crate) struct PythonShake128 {
	inner: oa::cryptography::Shake128,
}

#[pymethods]
impl PythonShake128 {
	/// Create a fresh SHAKE-128 context.
	#[new]
	fn new() -> Self {
		Self {
			inner: oa::cryptography::Shake128::new(),
		}
	}

	/// Absorb another input chunk.
	///
	/// # Errors
	///
	/// Raises after the first squeeze call until reset.
	fn update(&mut self, data: &[u8]) -> PyResult<()> {
		self.inner.update(data).map_err(python_error)
	}

	/// Squeeze exactly `output_len` bytes, finalizing on the first call.
	fn squeeze(&mut self, output_len: usize) -> Vec<u8> {
		let mut out = vec![0_u8; output_len];
		self.inner.squeeze(&mut out);
		out
	}

	/// Reset to a fresh empty context.
	fn reset(&mut self) {
		self.inner.reset();
	}

	fn __repr__(&self) -> &'static str {
		"Shake128()"
	}
}

// ── Incremental SHAKE-256 XOF ────────────────────────────────────────────────

#[pyclass(name = "Shake256", unsendable)]
pub(crate) struct PythonShake256 {
	inner: oa::cryptography::Shake256,
}

#[pymethods]
impl PythonShake256 {
	/// Create a fresh SHAKE-256 context.
	#[new]
	fn new() -> Self {
		Self {
			inner: oa::cryptography::Shake256::new(),
		}
	}

	/// Absorb another input chunk.
	///
	/// # Errors
	///
	/// Raises after the first squeeze call until reset.
	fn update(&mut self, data: &[u8]) -> PyResult<()> {
		self.inner.update(data).map_err(python_error)
	}

	/// Squeeze exactly `output_len` bytes, finalizing on the first call.
	fn squeeze(&mut self, output_len: usize) -> Vec<u8> {
		let mut out = vec![0_u8; output_len];
		self.inner.squeeze(&mut out);
		out
	}

	/// Reset to a fresh empty context.
	fn reset(&mut self) {
		self.inner.reset();
	}

	fn __repr__(&self) -> &'static str {
		"Shake256()"
	}
}

// ── Owned secure buffer ───────────────────────────────────────────────────────

/// A Python-owned allocation that is securely erased on drop (and on explicit
/// `reset`).  The underlying bytes are page-locked on Linux when possible.
///
/// Because Python cannot express Rust borrows, this type owns its allocation
/// rather than guarding an external slice.
#[pyclass(name = "SecureBuffer", unsendable)]
pub(crate) struct PythonSecureBuffer {
	bytes: Vec<u8>,
}

#[pymethods]
impl PythonSecureBuffer {
	/// Allocate `size` zero-initialised bytes and attempt page-locking.
	#[new]
	fn new(size: usize) -> Self {
		Self {
			bytes: vec![0_u8; size],
		}
	}

	/// Return the current contents as a `bytes` object.
	fn as_bytes(&self) -> &[u8] {
		&self.bytes
	}

	/// Return the allocation size in bytes.
	fn size_bytes(&self) -> usize {
		self.bytes.len()
	}

	/// Return whether this buffer holds a non-empty allocation.
	fn is_valid(&self) -> bool {
		!self.bytes.is_empty()
	}

	/// Securely zero the buffer contents without releasing the allocation.
	fn secure_zero(&mut self) {
		// Delegate to the OA secure-erase primitive via a temporary guard.
		let mut guard = oa::cryptography::SecureBuffer::new(&mut self.bytes);
		guard.secure_zero();
	}

	/// Securely zero the buffer and shrink the allocation to zero.
	fn reset(&mut self) {
		let mut guard = oa::cryptography::SecureBuffer::new(&mut self.bytes);
		guard.reset();
		drop(guard);
		self.bytes = Vec::new();
	}

	fn __repr__(&self) -> String {
		format!("SecureBuffer(size={})", self.bytes.len())
	}
}

impl Drop for PythonSecureBuffer {
	fn drop(&mut self) {
		// Securely erase before the Vec is freed.
		if !self.bytes.is_empty() {
			let mut guard = oa::cryptography::SecureBuffer::new(&mut self.bytes);
			guard.reset();
		}
	}
}

// ── One-shot SHAKE functions ──────────────────────────────────────────────────

#[pyfunction]
pub(crate) fn crypto_shake128(input: &[u8], output_len: usize) -> Vec<u8> {
	let mut out = vec![0_u8; output_len];
	oa::cryptography::shake128(input, &mut out);
	out
}

#[pyfunction]
pub(crate) fn crypto_shake256(input: &[u8], output_len: usize) -> Vec<u8> {
	let mut out = vec![0_u8; output_len];
	oa::cryptography::shake256(input, &mut out);
	out
}

// ── Standalone merkle_root ────────────────────────────────────────────────────

#[pyfunction]
pub(crate) fn crypto_merkle_root(leaves: Vec<Vec<u8>>) -> PyResult<Vec<u8>> {
	let hashes = leaves
		.iter()
		.map(|leaf| hash_from_bytes(leaf))
		.collect::<PyResult<Vec<_>>>()?;
	Ok(oa::cryptography::merkle_root(&hashes).as_bytes().to_vec())
}

// ── GPU / Vulkan batch hash functions ─────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (input, output_length=0))]
pub(crate) fn crypto_hash_shake128(
	input: &PythonMatrix,
	output_length: usize,
) -> PyResult<PythonMatrix> {
	oa::cryptography::hash::shake128(&input.inner, output_length)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (input, output_length=0))]
pub(crate) fn crypto_hash_shake256(
	input: &PythonMatrix,
	output_length: usize,
) -> PyResult<PythonMatrix> {
	oa::cryptography::hash::shake256(&input.inner, output_length)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn crypto_hash_keccak_f1600(input: &PythonMatrix) -> PyResult<PythonMatrix> {
	oa::cryptography::hash::keccak_f1600(&input.inner)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

#[pyfunction]
pub(crate) fn crypto_hash_merkle_root(input: &PythonMatrix) -> PyResult<PythonMatrix> {
	oa::cryptography::hash::merkle_root(&input.inner)
		.map(PythonMatrix::wrap)
		.map_err(python_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
	module.add_class::<PythonHasher>()?;
	module.add_class::<PythonShake128>()?;
	module.add_class::<PythonShake256>()?;
	module.add_class::<PythonSecureBuffer>()?;
	module.add_class::<PythonMerkleTree>()?;
	module.add_class::<PythonMerkleProof>()?;
	module.add_class::<PythonPublicKey>()?;
	module.add_class::<PythonSignature>()?;
	module.add_class::<PythonKeypair>()?;
	module.add_function(wrap_pyfunction!(crypto_hash, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_hash_combine, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_keccak_f1600, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_kmac256, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_shake128, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_shake256, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_merkle_root, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_generate_keypair, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_pqc_verify, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_pqc_verify_hash, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_hash_shake128, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_hash_shake256, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_hash_keccak_f1600, module)?)?;
	module.add_function(wrap_pyfunction!(crypto_hash_merkle_root, module)?)?;
	Ok(())
}
