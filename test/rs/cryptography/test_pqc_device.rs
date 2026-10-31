//! Deferred public-device ownership and independent ML-DSA verification proofs.
use ml_dsa::{MlDsa44, MlDsa65, MlDsa87, Signature, VerifyingKey};
use oa::{
	Matrix,
	cryptography::pqc::{MlDsaParameters, device},
};

#[test]
fn fips_parameter_extents_and_prehash_sizes_are_exact() {
	for (parameter, public, ciphertext) in [
		(device::MlKemParameters::MlKem512, 800, 768),
		(device::MlKemParameters::MlKem768, 1184, 1088),
		(device::MlKemParameters::MlKem1024, 1568, 1568),
	] {
		assert_eq!(parameter.public_key_size(), public);
		assert_eq!(parameter.ciphertext_size(), ciphertext);
	}
	assert_eq!(device::MlDsaPrehash::Sha256.digest_bytes(), 32);
	assert_eq!(device::MlDsaPrehash::Sha512.digest_bytes(), 64);
	assert_eq!(device::MlDsaPrehash::Shake128.digest_bytes(), 32);
	assert_eq!(device::MlDsaPrehash::Shake256.digest_bytes(), 64);
}

test_vk_slow!(
	device_mldsa_public_signing_matches_independent_verifier,
	engine,
	{
		for parameter in [
			MlDsaParameters::MlDsa44,
			MlDsaParameters::MlDsa65,
			MlDsaParameters::MlDsa87,
		] {
			let key = device::mldsa_keygen(&engine, parameter)?;
			let public = key.public_key().clone();
			let message_bytes = b"OA public device-key integration";
			let context_bytes = b"ctx";
			let message = Matrix::new_from_slice(&engine, [message_bytes.len()], message_bytes)?;
			let context = Matrix::new_from_slice(&engine, [context_bytes.len()], context_bytes)?;
			let first = device::mldsa_sign(&key, &message, &context, device::SigningMode::Deterministic)?;
			let second =
				device::mldsa_sign(&key, &message, &context, device::SigningMode::Deterministic)?;
			// Consuming erasure is queued after both consumers, with no host wait first.
			key.erase()?.wait()?;
			first.cleanup.wait()?;
			second.cleanup.wait()?;
			assert_eq!(first.status.read::<u32>()?, [1]);
			assert_eq!(second.status.read::<u32>()?, [1]);
			let signature_bytes = first.signature.read::<u8>()?;
			assert_eq!(signature_bytes, second.signature.read::<u8>()?);
			assert_eq!(signature_bytes.len(), parameter.signature_size());
			let public_bytes = public.read::<u8>()?;
			macro_rules! independent_verify {
				($set:ty, $pk_len:expr, $sig_len:expr) => {{
					let pk: &[u8; $pk_len] = public_bytes.as_slice().try_into().unwrap();
					let sig: &[u8; $sig_len] = signature_bytes.as_slice().try_into().unwrap();
					let verifier = VerifyingKey::<$set>::decode(pk.into());
					let signature = Signature::<$set>::decode(sig.into()).expect("canonical signature");
					assert!(verifier.verify_with_context(message_bytes, context_bytes, &signature));
					assert!(!verifier.verify_with_context(b"wrong", context_bytes, &signature));
				}};
			}
			match parameter {
				MlDsaParameters::MlDsa44 => independent_verify!(MlDsa44, 1312, 2420),
				MlDsaParameters::MlDsa65 => independent_verify!(MlDsa65, 1952, 3309),
				MlDsaParameters::MlDsa87 => independent_verify!(MlDsa87, 2592, 4627),
			}
		}
		Ok(())
	}
);

test_vk_slow!(
	device_mlkem_public_lifecycle_and_implicit_rejection,
	engine,
	{
		for parameter in [
			device::MlKemParameters::MlKem512,
			device::MlKemParameters::MlKem768,
			device::MlKemParameters::MlKem1024,
		] {
			let key = device::mlkem_keygen(&engine, parameter)?;
			let encapsulated = device::mlkem_encapsulate(key.public_key(), parameter)?;
			let decapsulated = device::mlkem_decapsulate(&key, &encapsulated.ciphertext)?;
			let malformed = Matrix::new_from_slice(
				&engine,
				[parameter.ciphertext_size()],
				&vec![0_u8; parameter.ciphertext_size()],
			)?;
			let rejected = device::mlkem_decapsulate(&key, &malformed)?;
			key.erase()?.wait()?;
			assert_eq!(encapsulated.status.read::<u32>()?, [1]);
			assert_eq!(
				encapsulated.ciphertext.num_elements(),
				parameter.ciphertext_size()
			);
			encapsulated.cleanup.wait()?;
			decapsulated.cleanup.wait()?;
			rejected.cleanup.wait()?;
			encapsulated.shared_secret.erase()?.wait()?;
			decapsulated.shared_secret.erase()?.wait()?;
			rejected.shared_secret.erase()?.wait()?;
		}
		Ok(())
	}
);

test_vk_slow!(
	device_secrets_reject_capture_and_bad_public_metadata,
	engine,
	{
		let result =
			engine.capture(|| device::mlkem_keygen(&engine, device::MlKemParameters::MlKem512));
		assert!(matches!(result, Err(ref error) if error.kind() == oa::ErrorKind::FailedPrecondition));
		let key = device::mldsa_keygen(&engine, MlDsaParameters::MlDsa44)?;
		let message = Matrix::new_from_slice(&engine, [1], &[1_u8])?;
		let oversized_context = Matrix::new_from_slice(&engine, [256], &[0_u8; 256])?;
		assert!(
			matches!(device::mldsa_sign(&key, &message, &oversized_context, device::SigningMode::Randomized), Err(ref error) if error.kind() == oa::ErrorKind::InvalidArgument)
		);
		let wrong_digest = Matrix::new_from_slice(&engine, [31], &[0_u8; 31])?;
		assert!(
			device::mldsa_sign_prehashed(
				&key,
				&wrong_digest,
				&message,
				device::MlDsaPrehash::Sha256,
				device::SigningMode::Deterministic
			)
			.is_err()
		);
		key.erase()?.wait()?;
		Ok(())
	}
);
