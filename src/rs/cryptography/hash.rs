//! Vulkan batch hashing operations.

mod lowering;

use crate::{Matrix, Result};

/// Hash every row with SHAKE-128.
///
/// An `output_length` of zero selects 16 bytes. Output rows are padded to an
/// eight-byte boundary, matching the OA donor ABI.
pub fn shake128(input: &Matrix, output_length: usize) -> Result<Matrix> {
	lowering::shake(input, output_length, lowering::ShakeKind::Shake128)
}

/// Hash every row with SHAKE-256.
///
/// An `output_length` of zero selects 32 bytes. Output rows are padded to an
/// eight-byte boundary, matching the OA donor ABI.
pub fn shake256(input: &Matrix, output_length: usize) -> Result<Matrix> {
	lowering::shake(input, output_length, lowering::ShakeKind::Shake256)
}

/// Apply Keccak-f\[1600\] independently to every 200-byte row.
pub fn keccak_f1600(input: &Matrix) -> Result<Matrix> {
	lowering::keccak_f1600(input)
}

/// Reduce 32-byte leaf hashes to one Merkle root.
pub fn merkle_root(input: &Matrix) -> Result<Matrix> {
	lowering::merkle_root(input)
}
