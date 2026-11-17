//! Selected-hash device Merkle construction, subtree roots and inclusion frames.
#[path = "merkle/operations.gen.rs"]
mod operations;
#[path = "merkle/parity.gen.rs"]
mod parity;

pub use operations::{DeviceMerkleTree, MerkleAlgorithm, build, proofs, subtree_roots, verify};
