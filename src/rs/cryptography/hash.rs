//! Vulkan batch hashing operations.

#[path = "hash/batch.gen.rs"]
mod batch;

pub use batch::{keccak_f1600, merkle_root, shake128, shake256};
