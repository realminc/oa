//! CPU cryptography conformance and value-contract tests.

#[macro_use]
#[path = "support/test_vk.rs"]
mod test_vk_fixture;

#[path = "cryptography/test_cryptography.rs"]
mod cryptography;
#[path = "cryptography/test_secure_buffer.rs"]
mod secure_buffer;
