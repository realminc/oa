//! Raw-byte model inputs and logit decoding.

use crate::{DType, Engine, Error, Matrix, Result};

/// Fixed vocabulary cardinality for one token per byte value.
pub const VOCAB_SIZE: usize = 256;
/// Optional zero-valued padding byte.
pub const PAD: u8 = 0x00;
/// Optional beginning-of-sequence byte.
pub const BOS: u8 = 0x01;
/// Optional end-of-sequence byte.
pub const EOS: u8 = 0x02;
/// Optional sequence-separator byte.
pub const SEP: u8 = 0x03;

/// Upload raw bytes as a rank-one U8 Matrix.
///
/// # Errors
///
/// Returns an error when allocation or upload fails.
pub fn encode(engine: &Engine, bytes: &[u8]) -> Result<Matrix> {
	Matrix::from_slice(engine, [bytes.len()], bytes)
}

/// Upload raw bytes as one batched U8 sequence `[1, S]`.
///
/// # Errors
///
/// Returns an error when allocation or upload fails.
pub fn encode_batched(engine: &Engine, bytes: &[u8]) -> Result<Matrix> {
	Matrix::from_slice(engine, [1, bytes.len()], bytes)
}

/// Upload UTF-8 text without tokenization or normalization.
///
/// # Errors
///
/// Returns an error when allocation or upload fails.
pub fn encode_text(engine: &Engine, text: &str) -> Result<Matrix> {
	encode(engine, text.as_bytes())
}

/// Greedily decode one byte from each FP32 logit row.
///
/// This is an explicit submission, completion, and compact host-read boundary.
///
/// # Errors
///
/// Returns an error unless logits are rank-one `[256]` or rank-two `[N, 256]`
/// FP32, or execution/readback fails.
pub fn decode(logits: &Matrix) -> Result<Vec<u8>> {
	decode_sampled(logits, 0.0, 0, 1.0, 0)
}

/// Sample one byte from each FP32 logit row.
///
/// # Errors
///
/// Returns an error for an invalid 256-class logit contract, sampling contract,
/// execution, or readback failure.
pub fn sample(
	logits: &Matrix,
	temperature: f32,
	top_k: i32,
	top_p: f32,
	seed: u64,
) -> Result<Vec<u8>> {
	decode_sampled(logits, temperature, top_k, top_p, seed)
}

/// Greedily decode logits and require exact UTF-8 output.
///
/// # Errors
///
/// Returns the errors from [`decode`] or [`crate::ErrorKind::DataLoss`] when the
/// decoded byte stream is not UTF-8.
pub fn decode_text(logits: &Matrix) -> Result<String> {
	String::from_utf8(decode(logits)?)
		.map_err(|_| Error::data_loss("byte logits do not decode to valid UTF-8"))
}

fn decode_sampled(
	logits: &Matrix,
	temperature: f32,
	top_k: i32,
	top_p: f32,
	seed: u64,
) -> Result<Vec<u8>> {
	if logits.dtype() != DType::F32
		|| !(1..=2).contains(&logits.shape().len())
		|| logits.shape().last() != Some(&VOCAB_SIZE)
	{
		return Err(Error::invalid_argument(
			"byte decoding requires rank-one or rank-two FP32 logits with 256 classes",
		));
	}
	let ids = crate::matrix::sample_logits(logits, temperature, top_k, top_p, seed)?;
	ids
		.read::<i32>()?
		.into_iter()
		.map(|id| {
			u8::try_from(id)
				.map_err(|_| Error::internal("256-class byte sampling produced an out-of-range token"))
		})
		.collect()
}
