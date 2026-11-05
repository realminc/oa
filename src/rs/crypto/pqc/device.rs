//! Experimental ML-DSA/ML-KEM operations with unique, opaque device secrets.
//!
//! Public messages and results use ordinary Matrix values. Keys and shared
//! secrets have no mapping, export, Clone, or Debug, and require consuming
//! erasure. Abandonment quarantines their allocations without GPU work in Drop.
//! Software correctness and production security admission remain distinct.
#[path = "device/operations.gen.rs"]
mod operations;
#[path = "device/parity.gen.rs"]
mod parity;

pub use operations::{
	DeviceDecapsulation, DeviceDerivation, DeviceEncapsulation, DeviceErasure, DeviceMlDsaKey,
	DeviceMlKemKey, DeviceSharedSecret, DeviceSignature, MlDsaPrehash, MlKemParameters,
	PqcSecretScope, SigningMode, begin_secret_scope, mldsa_import_key, mldsa_keygen, mldsa_sign,
	mldsa_sign_hash_message, mldsa_sign_prehashed, mldsa_verify_hash_message, mldsa_verify_prehashed,
	mlkem_decapsulate, mlkem_encapsulate, mlkem_import_key, mlkem_keygen, mlkem_shared_derive,
};
