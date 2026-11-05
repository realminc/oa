//! CPU cryptography conformance and value-contract tests.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "crypto/test_crypto.rs"]
mod crypto;
#[path = "crypto/test_secure_buffer.rs"]
mod secure_buffer;

#[path = "crypto/test_hash_batch.rs"]
mod hash_batch;

#[path = "crypto/test_merkle_proofs.rs"]
mod merkle_proofs;

#[path = "crypto/test_bigint.rs"]
mod bigint;

#[path = "crypto/test_bigint_modular.rs"]
mod bigint_modular;

#[path = "crypto/test_bigint_breadth.rs"]
mod bigint_breadth;

#[path = "crypto/test_ntt.rs"]
mod ntt;

#[path = "crypto/test_pqc_device.rs"]
mod pqc_device;
