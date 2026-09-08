//! Private target-independent shader artifacts and metadata.

use crate::{Error, Result};

const SPIRV_MAGIC: u32 = 0x0723_0203;

/// A build-validated shader artifact embedded into the OA library.
pub struct ShaderArtifact {
	bytes: &'static [u8],
	pub workgroup_size: [u32; 3],
	pub dispatch_tile_size: [u32; 3],
	pub push_constant_size: u32,
}

impl ShaderArtifact {
	/// Decode the embedded little-endian SPIR-V bytes for Vulkan module creation.
	pub fn spirv_words(&self) -> Result<Vec<u32>> {
		decode_spirv(self.bytes)
	}
}

#[path = "shader/generated.rs"]
mod generated;

pub(crate) use generated::KernelId;

fn decode_spirv(bytes: &[u8]) -> Result<Vec<u32>> {
	if bytes.len() < size_of::<u32>() || !bytes.len().is_multiple_of(size_of::<u32>()) {
		return Err(invalid_artifact(
			"SPIR-V byte length is not a non-empty word sequence",
		));
	}
	let (word_bytes, remainder) = bytes.as_chunks::<4>();
	debug_assert!(remainder.is_empty());
	let words = word_bytes
		.iter()
		.map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
		.collect::<Vec<_>>();
	if words[0] != SPIRV_MAGIC {
		return Err(invalid_artifact("SPIR-V magic number is invalid"));
	}
	Ok(words)
}

fn invalid_artifact(message: &'static str) -> Error {
	Error::backend_failure(
		"Slang",
		"shader-artifact validation",
		std::io::Error::other(message),
	)
}

#[cfg(test)]
mod tests {
	use super::{KernelId, decode_spirv};

	#[test]
	fn embedded_matrix_artifacts_match_their_validated_abi() -> crate::Result<()> {
		for kernel in KernelId::ALL {
			let artifact = kernel.artifact();
			assert!(!artifact.spirv_words()?.is_empty());
			match kernel {
				KernelId::MatrixMatMulNtTiledF32 => {
					assert_eq!(artifact.workgroup_size, [256, 1, 1]);
					assert_eq!(artifact.dispatch_tile_size, [64, 64, 1]);
					assert_eq!(artifact.push_constant_size, 24);
				}
				_ => {
					assert_eq!(artifact.workgroup_size, [256, 1, 1]);
					assert_eq!(artifact.dispatch_tile_size, [256, 1, 1]);
					assert!(matches!(artifact.push_constant_size, 12 | 16));
				}
			}
		}
		Ok(())
	}

	#[test]
	fn malformed_spirv_is_rejected() {
		assert!(decode_spirv(&[]).is_err());
		assert!(decode_spirv(&[0, 0, 0, 0]).is_err());
		assert!(decode_spirv(&[3, 2, 35]).is_err());
	}
}
