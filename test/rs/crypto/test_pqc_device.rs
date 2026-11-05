//! Deferred public-device ownership and independent ML-DSA verification proofs.
use ml_dsa::{MlDsa44, MlDsa65, MlDsa87, Signature, VerifyingKey};
use oa::{
	Matrix,
	crypto::pqc::{MlDsaParameters, device},
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

fn unhex(text: &str) -> Vec<u8> {
	(0..text.len())
		.step_by(2)
		.map(|index| u8::from_str_radix(&text[index..index + 2], 16).unwrap())
		.collect()
}

test_vk_slow!(
	device_mlkem_import_decapsulates_and_derives_public_bytes,
	engine,
	{
		let prompt: serde_json::Value = serde_json::from_str(include_str!(
			"../../fixtures/crypto/mlkem/encapDecap-prompt.json"
		))
		.unwrap();
		let expected: serde_json::Value = serde_json::from_str(include_str!(
			"../../fixtures/crypto/mlkem/encapDecap-expectedResults.json"
		))
		.unwrap();
		let decap_prompts: Vec<&serde_json::Value> = prompt["testGroups"]
			.as_array()
			.unwrap()
			.iter()
			.filter(|group| group["function"] == "decapsulation")
			.collect();
		// Expected groups 4-6 answer decapsulation prompts in parameter order.
		let decap_expected = &expected["testGroups"].as_array().unwrap()[3..6];
		for (group, answers) in decap_prompts.iter().zip(decap_expected.iter()) {
			let parameter = match group["parameterSet"].as_str().unwrap() {
				"ML-KEM-512" => device::MlKemParameters::MlKem512,
				"ML-KEM-768" => device::MlKemParameters::MlKem768,
				_ => device::MlKemParameters::MlKem1024,
			};
			let case = &group["tests"].as_array().unwrap()[0];
			let answer = answers["tests"]
				.as_array()
				.unwrap()
				.iter()
				.find(|test| test["tcId"] == case["tcId"])
				.unwrap();
			let mut dk = unhex(case["dk"].as_str().unwrap());
			let ciphertext = unhex(case["c"].as_str().unwrap());
			let shared = unhex(answer["k"].as_str().unwrap());
			// The embedded ek is the retained public transport: dk = dkPKE || ek || H(ek) || z.
			let k = match parameter {
				device::MlKemParameters::MlKem512 => 2_usize,
				device::MlKemParameters::MlKem768 => 3,
				_ => 4,
			};
			let expected_ek = dk[384 * k..384 * k + parameter.public_key_size()].to_vec();
			let key = device::mlkem_import_key(&engine, parameter, &mut dk)?;
			let public = key.public_key().read::<u8>()?;
			assert_eq!(public, expected_ek);
			let ct = Matrix::new_from_slice(&engine, [ciphertext.len()], &ciphertext)?;
			let decapsulated = device::mlkem_decapsulate(&key, &ct)?;
			let info_bytes = b"OA keyed derivation context";
			let info = Matrix::new_from_slice(&engine, [info_bytes.len()], info_bytes)?;
			let derivation = device::mlkem_shared_derive(&decapsulated.shared_secret, &info, 64)?;
			let derived = derivation.derived.read::<u8>()?;
			let mut oracle = Vec::with_capacity(shared.len() + info_bytes.len());
			oracle.extend_from_slice(&shared);
			oracle.extend_from_slice(info_bytes);
			let mut expected_derived = [0_u8; 64];
			oa::crypto::shake256(&oracle, &mut expected_derived);
			assert_eq!(derived, expected_derived);
			key.cleanup().wait()?;
			decapsulated.cleanup.wait()?;
			derivation.cleanup.wait()?;
			decapsulated.shared_secret.erase()?.wait()?;
			key.erase()?.wait()?;
		}
		Ok(())
	}
);

test_vk_slow!(
	device_mldsa_import_signs_for_independent_verifier,
	engine,
	{
		let expected: serde_json::Value = serde_json::from_str(include_str!(
			"../../fixtures/crypto/mldsa/keyGen-expectedResults.json"
		))
		.unwrap();
		for group in expected["testGroups"].as_array().unwrap() {
			let case = &group["tests"].as_array().unwrap()[0];
			let pk = unhex(case["pk"].as_str().unwrap());
			let mut sk = unhex(case["sk"].as_str().unwrap());
			let parameter = match group["parameterSet"].as_str().unwrap() {
				"ML-DSA-44" => MlDsaParameters::MlDsa44,
				"ML-DSA-65" => MlDsaParameters::MlDsa65,
				_ => MlDsaParameters::MlDsa87,
			};
			let public = Matrix::new_from_slice(&engine, [pk.len()], &pk)?;
			let key = device::mldsa_import_key(&engine, parameter, &mut sk, &public)?;
			assert_eq!(key.public_key().read::<u8>()?, pk);
			let message_bytes = b"OA imported-key signing";
			let context_bytes = b"import-ctx";
			let message = Matrix::new_from_slice(&engine, [message_bytes.len()], message_bytes)?;
			let context = Matrix::new_from_slice(&engine, [context_bytes.len()], context_bytes)?;
			let signed =
				device::mldsa_sign(&key, &message, &context, device::SigningMode::Deterministic)?;
			assert_eq!(signed.status.read::<u32>()?, [1]);
			let signature_bytes = signed.signature.read::<u8>()?;
			macro_rules! independent_verify {
				($set:ty, $pk_len:expr, $sig_len:expr) => {{
					let pk_array: &[u8; $pk_len] = pk.as_slice().try_into().unwrap();
					let sig: &[u8; $sig_len] = signature_bytes.as_slice().try_into().unwrap();
					let verifier = VerifyingKey::<$set>::decode(pk_array.into());
					let signature = Signature::<$set>::decode(sig.into()).expect("canonical signature");
					assert!(verifier.verify_with_context(message_bytes, context_bytes, &signature));
				}};
			}
			match parameter {
				MlDsaParameters::MlDsa44 => independent_verify!(MlDsa44, 1312, 2420),
				MlDsaParameters::MlDsa65 => independent_verify!(MlDsa65, 1952, 3309),
				MlDsaParameters::MlDsa87 => independent_verify!(MlDsa87, 2592, 4627),
			}
			key.cleanup().wait()?;
			signed.cleanup.wait()?;
			key.erase()?.wait()?;
		}
		Ok(())
	}
);

test_vk_slow!(device_secret_scope_defers_classifies_and_commits, engine, {
	let scope = device::begin_secret_scope(&engine)?;
	let key = device::mlkem_keygen(&engine, device::MlKemParameters::MlKem512)?;
	let encapsulated =
		device::mlkem_encapsulate(key.public_key(), device::MlKemParameters::MlKem512)?;
	let nodes = scope.nodes();
	assert_eq!(nodes.len(), 2);
	assert!(nodes.iter().all(|node| node.uses_secret));
	// Deferred events stay pending until the batch submits.
	assert!(!key.cleanup().try_is_complete()?);
	scope.commit()?;
	key.cleanup().wait()?;
	encapsulated.cleanup.wait()?;
	assert_eq!(encapsulated.status.read::<u32>()?, [1]);
	encapsulated.shared_secret.erase()?.wait()?;
	key.erase()?.wait()?;
	assert!(device::begin_secret_scope(&engine).is_ok());
	Ok(())
});

test_vk_slow!(device_secret_scope_abandons_without_submission, engine, {
	let key;
	{
		let _scope = device::begin_secret_scope(&engine)?;
		key = device::mlkem_keygen(&engine, device::MlKemParameters::MlKem512)?;
		// Dropping the scope abandons the batch: no submission, no wait.
	}
	assert!(key.cleanup().wait().is_err());
	// The engine remains usable; the abandoned products stay quarantined.
	let fresh = device::mlkem_keygen(&engine, device::MlKemParameters::MlKem512)?;
	fresh.cleanup().wait()?;
	fresh.erase()?.wait()?;
	Ok(())
});
