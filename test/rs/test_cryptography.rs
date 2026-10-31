//! CPU cryptography conformance and value-contract tests.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "cryptography/test_cryptography.rs"]
mod cryptography;
#[path = "cryptography/test_secure_buffer.rs"]
mod secure_buffer;

#[path = "cryptography/test_hash_batch.rs"]
mod hash_batch;

#[path = "cryptography/test_merkle_proofs.rs"]
mod merkle_proofs;

#[path = "cryptography/test_bigint.rs"]
mod bigint;

#[path = "cryptography/test_bigint_modular.rs"]
mod bigint_modular;

#[path = "cryptography/test_bigint_breadth.rs"]
mod bigint_breadth;

#[path = "cryptography/test_ntt.rs"]
mod ntt;

#[path = "cryptography/test_pqc_device.rs"]
mod pqc_device;
