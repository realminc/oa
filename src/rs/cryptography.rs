//! Cryptographic primitives, secure host memory, post-quantum signatures, and
//! schema-owned Vulkan batch hashing.
//!
//! General-purpose primitives remain at this module root. Device batch
//! operations live under [`mod@hash`], and post-quantum algorithms live under
//! [`pqc`]. Secret keys never enter generic [`crate::Matrix`] storage.

#[cfg_attr(
	not(test),
	expect(
		dead_code,
		reason = "private entropy boundary; GPU secret operations are not admitted"
	)
)]
pub(crate) mod entropy;
pub mod hash;
pub mod pqc;
mod primitives;
mod secure_buffer;

pub use primitives::{
	Hash, Hasher, MerkleProof, MerkleTree, Shake128, Shake256, build_merkle_tree, hash, hash_combine,
	keccak_f1600, kmac256, merkle_proof, merkle_root, shake128, shake256, verify_merkle_proof,
};
pub use secure_buffer::SecureBuffer;
