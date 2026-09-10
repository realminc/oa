//! Post-quantum cryptography.
//!
//! This module exposes ML-DSA-65 (FIPS 204, security category 3). Private key
//! material stays in a purpose-specific host type and is zeroized on drop.

use ml_dsa::{
	Generate, Keypair as _, MlDsa65, Signature as MlSignature, Signer as _, SigningKey,
	Verifier as _, VerifyingKey,
};

use crate::{Error, Hash, Result};

/// Encoded ML-DSA-65 public-key size.
pub const PUBLIC_KEY_SIZE: usize = 1_952;
/// Expanded ML-DSA-65 secret-key size used by the OA donor ABI.
pub const SECRET_KEY_SIZE: usize = 4_032;
/// Encoded ML-DSA-65 signature size.
pub const SIGNATURE_SIZE: usize = 3_309;

/// An encoded ML-DSA-65 public key.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PublicKey([u8; PUBLIC_KEY_SIZE]);

impl Default for PublicKey {
	fn default() -> Self {
		Self([0; PUBLIC_KEY_SIZE])
	}
}

impl PublicKey {
	/// Parse exactly [`PUBLIC_KEY_SIZE`] bytes.
	pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
		Ok(Self(copy_array(bytes, "ML-DSA-65 public key")?))
	}

	/// Return the fixed-size encoded key.
	#[must_use]
	pub const fn as_bytes(&self) -> &[u8; PUBLIC_KEY_SIZE] {
		&self.0
	}

	/// Serialize the key to its fixed-size wire representation.
	#[must_use]
	pub fn to_bytes(&self) -> [u8; PUBLIC_KEY_SIZE] {
		self.0
	}

	/// Return whether every encoded byte is zero.
	#[must_use]
	pub fn is_zero(&self) -> bool {
		self.0.iter().all(|byte| *byte == 0)
	}

	/// Format the first 16 bytes as lowercase hexadecimal text.
	#[must_use]
	pub fn to_short_hex(&self) -> String {
		short_hex(&self.0)
	}
}

impl std::fmt::Debug for PublicKey {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_tuple("PublicKey")
			.field(&self.to_short_hex())
			.finish()
	}
}

/// An ML-DSA-65 secret signing key.
///
/// The implementation stores the standard 32-byte seed plus its expanded
/// signing state; both are zeroized by the RustCrypto dependency on drop.
pub struct SecretKey(SigningKey<MlDsa65>);

impl std::fmt::Debug for SecretKey {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.debug_struct("SecretKey").finish_non_exhaustive()
	}
}

/// An encoded ML-DSA-65 signature.
#[derive(Clone, PartialEq, Eq)]
pub struct Signature([u8; SIGNATURE_SIZE]);

impl Default for Signature {
	fn default() -> Self {
		Self([0; SIGNATURE_SIZE])
	}
}

impl Signature {
	/// Parse exactly [`SIGNATURE_SIZE`] bytes.
	pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
		Ok(Self(copy_array(bytes, "ML-DSA-65 signature")?))
	}

	/// Return the fixed-size encoded signature.
	#[must_use]
	pub const fn as_bytes(&self) -> &[u8; SIGNATURE_SIZE] {
		&self.0
	}

	/// Serialize the signature to its fixed-size wire representation.
	#[must_use]
	pub fn to_bytes(&self) -> [u8; SIGNATURE_SIZE] {
		self.0
	}

	/// Return whether every encoded byte is zero.
	#[must_use]
	pub fn is_zero(&self) -> bool {
		self.0.iter().all(|byte| *byte == 0)
	}

	/// Format the first 16 bytes as lowercase hexadecimal text.
	#[must_use]
	pub fn to_short_hex(&self) -> String {
		short_hex(&self.0)
	}
}

impl std::fmt::Debug for Signature {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_tuple("Signature")
			.field(&self.to_short_hex())
			.finish()
	}
}

/// One generated ML-DSA-65 key pair.
pub struct Keypair {
	/// Public verification key.
	pub public_key: PublicKey,
	/// Secret signing key.
	pub secret_key: SecretKey,
}

impl std::fmt::Debug for Keypair {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("Keypair")
			.field("public_key", &self.public_key)
			.field("secret_key", &self.secret_key)
			.finish()
	}
}

/// Generate a random ML-DSA-65 key pair using the operating-system RNG.
pub fn generate_keypair() -> Result<Keypair> {
	let signing = SigningKey::<MlDsa65>::try_generate()
		.map_err(|_| Error::internal("operating-system random generation failed"))?;
	let encoded = signing.verifying_key().encode();
	let mut public_key = [0_u8; PUBLIC_KEY_SIZE];
	public_key.copy_from_slice(encoded.as_slice());
	Ok(Keypair {
		public_key: PublicKey(public_key),
		secret_key: SecretKey(signing),
	})
}

/// Sign an arbitrary message with ML-DSA-65.
pub fn sign(message: &[u8], secret_key: &SecretKey) -> Result<Signature> {
	let signature = secret_key
		.0
		.try_sign(message)
		.map_err(|_| Error::internal("ML-DSA-65 signing failed"))?;
	let encoded = signature.encode();
	let mut bytes = [0_u8; SIGNATURE_SIZE];
	bytes.copy_from_slice(encoded.as_slice());
	Ok(Signature(bytes))
}

/// Sign a canonical OA hash with ML-DSA-65.
pub fn sign_hash(hash: &Hash, secret_key: &SecretKey) -> Result<Signature> {
	sign(hash.as_bytes(), secret_key)
}

/// Verify an ML-DSA-65 message signature. Malformed encodings fail closed.
#[must_use]
pub fn verify(message: &[u8], signature: &Signature, public_key: &PublicKey) -> bool {
	let Some(signature) = MlSignature::<MlDsa65>::decode((&signature.0).into()) else {
		return false;
	};
	let verifying_key = VerifyingKey::<MlDsa65>::decode((&public_key.0).into());
	verifying_key.verify(message, &signature).is_ok()
}

/// Verify an ML-DSA-65 signature over a canonical OA hash.
#[must_use]
pub fn verify_hash(hash: &Hash, signature: &Signature, public_key: &PublicKey) -> bool {
	verify(hash.as_bytes(), signature, public_key)
}

fn copy_array<const N: usize>(bytes: &[u8], label: &str) -> Result<[u8; N]> {
	<[u8; N]>::try_from(bytes)
		.map_err(|_| Error::invalid_argument(format!("{label} must contain exactly {N} bytes")))
}

fn short_hex(bytes: &[u8]) -> String {
	const HEX: &[u8; 16] = b"0123456789abcdef";
	let mut output = String::with_capacity(32);
	for byte in bytes.iter().take(16) {
		output.push(char::from(HEX[usize::from(byte >> 4)]));
		output.push(char::from(HEX[usize::from(byte & 0x0f)]));
	}
	output
}
