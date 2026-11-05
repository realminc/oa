//! Post-quantum cryptography.
//!
//! This module exposes host ML-DSA-65 and Vulkan batch verification for
//! ML-DSA-44/65/87 (FIPS 204). Private key
//! material stays in a purpose-specific host type and is zeroized on drop.
//!
//! CPU signing and verification use the `ml-dsa` crate for correctness-gated
//! host operations. Device batch verification is Experimental; software Vulkan
//! correctness evidence and named physical-device qualification are separate.

pub mod device;

#[path = "pqc/host.rs"]
mod host;
#[path = "pqc/parity.gen.rs"]
mod parity;
#[path = "pqc/verify.gen.rs"]
pub(crate) mod verification;

pub use host::{
	Keypair, MlDsaParameters, PUBLIC_KEY_SIZE, PublicKey, SECRET_KEY_SIZE, SIGNATURE_SIZE, SecretKey,
	Signature, generate_keypair, sign, sign_hash, verify, verify_hash,
};
pub use verification::{verify_batch, verify_batch_with_context, verify_batch_with_parameters};
