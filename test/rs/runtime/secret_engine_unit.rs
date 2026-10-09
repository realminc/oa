use super::*;

// This adapter affects tests only; software Vulkan remains an explicit choice.
fn secret_test_engine() -> Result<Engine> {
	let selection = match std::env::var("OA_TEST_DEVICE_INDEX") {
		Ok(index) => DeviceSelection::Index(index.parse::<usize>().map_err(|_| {
			Error::invalid_argument("OA_TEST_DEVICE_INDEX must be a nonnegative integer")
		})?),
		Err(std::env::VarError::NotPresent) => DeviceSelection::Automatic,
		Err(_) => {
			return Err(Error::invalid_argument(
				"OA_TEST_DEVICE_INDEX must be Unicode",
			));
		}
	};
	Engine::builder().devices(selection).build()
}

#[test]
#[ignore = "requires a Vulkan device; disposable entropy-upload validation"]
fn secret_entropy_upload_reaches_consumer_before_both_wipes() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	for size in [32, 64] {
		let bytes: Vec<u8> = (0..size).map(|index| (index * 7 + 3) as u8).collect();
		let probe = Buffer::host_visible_storage(&device, size)?;
		probe.write(0, &vec![0x5a; size])?;
		// Only this disposable constructor grants TRANSFER_SRC to device secrets.
		let secret = SecretBuffer::test_new(&device, size)?;
		let (command, witness) = secret.test_record_entropy_readback(&bytes, &probe)?;
		let event = submit_recorded(&mut handle.state.borrow_mut(), command)?;
		let completion = SecretErasure::new(event, witness);
		completion.wait()?;
		assert!(completion.try_is_complete()?);
		let mut result = vec![0; size];
		probe.read(0, &mut result)?;
		assert_eq!(result, bytes);
	}
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; disposable entropy-upload validation"]
fn secret_entropy_engine_consumes_and_rejects_capture() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let entropy = Entropy::<32>::generate()?;
	let completion = handle.consume_secret_entropy(entropy, |_, _, _| Ok(()))?;
	completion.wait()?;
	assert!(completion.try_is_complete()?);
	drop(handle.consume_secret_entropy(Entropy::<64>::generate()?, |_, _, _| Ok(()))?);
	engine.checkpoint()?.wait()?;
	let error = match engine.capture(|| {
		handle
			.consume_secret_entropy(Entropy::<32>::generate()?, |_, _, _| {
				panic!("capture invoked secret consumer")
			})
			.map(|_| ())
	}) {
		Ok(_) => panic!("secret entropy entered capture"),
		Err(error) => error,
	};
	assert_eq!(error.kind(), crate::ErrorKind::FailedPrecondition);
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; disposable entropy-upload validation"]
fn secret_entropy_recording_failure_does_not_submit() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let before = handle.state.borrow().next_epoch;
	let result = handle.consume_secret_entropy(Entropy::<32>::generate()?, |_, _, _| {
		Err(Error::invalid_argument("test consumer recording failure"))
	});
	assert!(matches!(result, Err(error) if error.message() == "test consumer recording failure"));
	assert_eq!(handle.state.borrow().next_epoch, before);
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; disposable secret-allocation validation"]
fn secret_erasure_overwrites_poison_and_padding_through_engine() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	for bytes in [1_usize, 3, 4, 19, 4097] {
		let padded = (bytes + 3) & !3;
		let probe = Buffer::host_visible_storage(&device, padded)?;
		probe.write(0, &vec![0x5a; padded])?;
		let consumer = engine.checkpoint()?;
		let completion =
			handle.erase_secret_buffer(handle.allocate_secret_buffer(bytes)?, &consumer)?;
		completion.wait()?;
		let secret = SecretBuffer::test_new(&device, bytes)?;
		let (command, witness) = secret.test_record_erasure(&probe)?;
		let event = submit_recorded(&mut handle.state.borrow_mut(), command)?;
		let completion = SecretErasure::new(event, witness);
		completion.wait()?;
		assert!(completion.try_is_complete()?);
		let mut output = vec![0x5a; padded];
		probe.read(0, &mut output)?;
		assert_eq!(output, vec![0; padded]);
	}
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; disposable secret-allocation validation"]
fn dropped_erasure_ticket_does_not_release_pending_resources() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let probe = Buffer::host_visible_storage(&device, 20)?;
	probe.write(0, &[0x5a; 20])?;
	let secret = SecretBuffer::test_new(&device, 19)?;
	let (command, witness) = secret.test_record_erasure(&probe)?;
	let event = submit_recorded(&mut handle.state.borrow_mut(), command)?;
	drop(SecretErasure::new(event, witness));
	// A later retirement completes after the wipe, independently of its ticket.
	engine.checkpoint()?.wait()?;
	let mut output = [0x5a; 20];
	probe.read(0, &mut output)?;
	assert_eq!(output, [0; 20]);
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; disposable secret-allocation validation"]
fn abandoned_secret_is_quarantined_instead_of_recycled() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let secret = handle.allocate_secret_buffer(32)?;
	let (command, witness) = secret.record_erasure()?;
	assert!(!witness.is_confirmed());
	assert!(!witness.is_unconfirmed());
	// Pre-submit failure/cancellation: command release must not confirm erasure.
	// This intentionally quarantines one disposable 32-byte dedicated allocation.
	let device = handle.state.borrow().device.clone();
	device.free(command);
	assert!(witness.is_unconfirmed());
	assert!(!witness.is_confirmed());
	assert!(handle.state.borrow_mut().retirement.prepare().is_err());
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; disposable secret-allocation validation"]
fn unavailable_retirement_marks_erasure_unconfirmed() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let (command, witness) = handle.allocate_secret_buffer(32)?.record_erasure()?;
	// Inject the unavailable-worker branch without submitting real GPU work.
	assert!(handle.state.borrow().retirement.retire(command, 1).is_err());
	assert!(witness.is_unconfirmed());
	assert!(!witness.is_confirmed());
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; disposable secret-allocation validation"]
fn secret_storage_rejects_capture_and_foreign_completion() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let error = match engine.capture(|| handle.allocate_secret_buffer(32).map(|_| ())) {
		Ok(_) => panic!("secret allocation entered capture"),
		Err(error) => error,
	};
	assert_eq!(error.kind(), crate::ErrorKind::FailedPrecondition);
	let other = secret_test_engine()?;
	let foreign = other.checkpoint()?;
	let secret = handle.allocate_secret_buffer(32)?;
	let error = match handle.erase_secret_buffer(secret, &foreign) {
		Ok(_) => panic!("foreign completion was accepted"),
		Err(error) => error,
	};
	assert_eq!(error.kind(), crate::ErrorKind::InvalidArgument);
	foreign.wait()?;
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; disposable staging-retirement validation"]
fn secret_staging_witness_requires_host_cache_cleanup() -> Result<()> {
	let engine = secret_test_engine()?;
	let device = engine.handle().state.borrow().device.clone();
	SecretBuffer::test_staging_witness(&device)
}

#[test]
#[ignore = "requires a Vulkan device; disposable secret-descriptor validation"]
fn secret_descriptor_reuse_requires_confirmed_retirement() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let secret = handle.allocate_secret_buffer(19)?;
	let index = {
		let binding = secret.binding()?;
		assert_eq!(binding.byte_len(), 20);
		assert_ne!(binding.raw(), ash::vk::Buffer::null());
		binding.descriptor_index()
	};
	let (command, witness) = secret.record_erasure()?;
	// The owner has moved into the unsubmitted command. Its slot cannot be reused.
	let other = handle.allocate_secret_buffer(32)?;
	assert_ne!(other.binding()?.descriptor_index(), index);
	let event = submit_recorded(&mut handle.state.borrow_mut(), command)?;
	SecretErasure::new(event, witness).wait()?;
	let replacement = handle.allocate_secret_buffer(32)?;
	assert_eq!(replacement.binding()?.descriptor_index(), index);
	let final_consumer = engine.checkpoint()?;
	handle.erase_secret_buffer(other, &final_consumer)?.wait()?;
	handle
		.erase_secret_buffer(replacement, &final_consumer)?
		.wait()?;
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; disposable secret-descriptor validation"]
fn secret_descriptor_cancellation_quarantines_its_slot() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let secret = handle.allocate_secret_buffer(32)?;
	let index = secret.binding()?.descriptor_index();
	let (command, witness) = secret.record_erasure()?;
	device.free(command);
	assert!(witness.is_unconfirmed());
	let replacement = handle.allocate_secret_buffer(32)?;
	assert_ne!(replacement.binding()?.descriptor_index(), index);
	let final_consumer = engine.checkpoint()?;
	handle
		.erase_secret_buffer(replacement, &final_consumer)?
		.wait()?;
	Ok(())
}

#[test]
#[ignore = "requires a Vulkan device; secret-descriptor boundary validation"]
fn secret_descriptor_rejects_extent_above_queried_limit() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let limit = handle
		.state
		.borrow()
		.device
		.physical()
		.limits
		.max_storage_buffer_range;
	let bytes = usize::try_from(u64::from(limit) + 1).unwrap();
	let before = handle.state.borrow().next_epoch;
	let error = handle
		.allocate_secret_buffer(bytes)
		.err()
		.expect("oversized descriptor admitted");
	assert_eq!(error.kind(), crate::ErrorKind::InvalidArgument);
	assert_eq!(handle.state.borrow().next_epoch, before);
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; NIST public-key/secret-erasure proof"]
fn secret_mlkem_keygen_nist_all_parameters() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let prompt: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mlkem/keyGen-prompt.json"
	))
	.expect("pinned NIST prompt");
	let expected: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mlkem/keyGen-expectedResults.json"
	))
	.expect("pinned NIST results");
	fn decode(value: &serde_json::Value) -> Vec<u8> {
		let text = value.as_str().expect("fixture hex");
		text
			.as_bytes()
			.as_chunks::<2>()
			.0
			.iter()
			.map(|pair| {
				u8::from_str_radix(std::str::from_utf8(pair).expect("ASCII"), 16).expect("hex byte")
			})
			.collect()
	}
	let mut cases = 0;
	for group in prompt["testGroups"].as_array().expect("groups") {
		let k = match group["parameterSet"].as_str().expect("parameter") {
			"ML-KEM-512" => 2,
			"ML-KEM-768" => 3,
			"ML-KEM-1024" => 4,
			_ => panic!("unknown fixture parameter"),
		};
		let answers = expected["testGroups"]
			.as_array()
			.expect("results")
			.iter()
			.find(|answer| answer["tgId"] == group["tgId"])
			.expect("result group");
		// Every unchanged ACVP key-generation vector, without secret readback.
		for case in group["tests"].as_array().expect("cases") {
			let answer = answers["tests"]
				.as_array()
				.expect("answers")
				.iter()
				.find(|answer| answer["tcId"] == case["tcId"])
				.expect("answer");
			let mut seed = [0; 64];
			seed[..32].copy_from_slice(&decode(&case["d"]));
			seed[32..].copy_from_slice(&decode(&case["z"]));
			let entropy = Entropy::test_from_bytes(&seed)?;
			crate::core::memory::zero_secure(&mut seed);
			let (public, erasure) = handle.prove_mlkem_keygen(k, entropy)?;
			erasure.wait()?;
			assert!(erasure.try_is_complete()?);
			let mut bytes = vec![0; public.byte_len()];
			public.read(0, &mut bytes)?;
			assert_eq!(bytes, decode(&answer["ek"]), "NIST tcId {}", case["tcId"]);
			cases += 1;
		}
	}
	assert_eq!(cases, 75);
	let before = handle.state.borrow().next_epoch;
	assert!(
		matches!(handle.prove_mlkem_keygen(1, Entropy::generate()?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert_eq!(handle.state.borrow().next_epoch, before);
	assert!(
		matches!(engine.capture(|| handle.prove_mlkem_keygen(2, Entropy::generate()?).map(|_| ())), Err(error) if error.kind() == crate::ErrorKind::FailedPrecondition)
	);
	Ok(())
}

#[test]
#[ignore = "requires Vulkan; secret-dispatch preflight validation"]
fn secret_mlkem_keygen_rejects_alias_extent_and_foreign_storage() -> Result<()> {
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let seed = SecretBuffer::new(&device, 64)?;
	let private = SecretBuffer::new(&device, 1632)?;
	let public = Buffer::host_visible_storage(&device, 800)?;
	let wrong = Buffer::host_visible_storage(&device, 804)?;
	let before = handle.state.borrow().next_epoch;
	assert!(
		matches!(device.prepare_mlkem_keygen(seed.binding()?, seed.binding()?, &public, 2),
		Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert!(
		matches!(device.prepare_mlkem_keygen(seed.binding()?, private.binding()?, &wrong, 2),
		Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert!(
		matches!(device.prepare_mlkem_keygen(seed.binding()?, private.binding()?, &public, 4),
		Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	let foreign = secret_test_engine()?;
	let foreign_handle = foreign.handle();
	let foreign_device = foreign_handle.state.borrow().device.clone();
	let foreign_public = Buffer::host_visible_storage(&foreign_device, 800)?;
	assert!(
		matches!(device.prepare_mlkem_keygen(seed.binding()?, private.binding()?, &foreign_public, 2),
		Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert_eq!(handle.state.borrow().next_epoch, before);
	let final_consumer = engine.checkpoint()?;
	handle.erase_secret_buffer(seed, &final_consumer)?.wait()?;
	handle
		.erase_secret_buffer(private, &final_consumer)?
		.wait()?;
	Ok(())
}

#[test]
#[ignore = "requires Vulkan; retained ML-KEM key lifetime validation"]
fn secret_mlkem_retained_key_erases_without_waiting_for_generation() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	for k in [2, 3, 4] {
		let (key, seed_erasure) = handle.generate_mlkem_key(k, Entropy::generate()?)?;
		assert_eq!(key.k, k);
		assert_eq!(key.private.binding()?.byte_len(), u64::from(768 * k + 96));
		let public = key.public.clone();
		let ready = key.ready.clone();
		// Queue the consuming wipe immediately. Both commands retain storage;
		// the caller has not waited for generation or seed erasure.
		let key_erasure = handle.erase_mlkem_key(key)?;
		drop(seed_erasure);
		key_erasure.wait()?;
		assert!(key_erasure.try_is_complete()?);
		assert!(ready.is_complete()?);
		let mut bytes = vec![0; public.byte_len()];
		public.read(0, &mut bytes)?;
		assert!(bytes.iter().any(|byte| *byte != 0));
	}
	let before = handle.state.borrow().next_epoch;
	assert!(matches!(handle.generate_mlkem_key(1, Entropy::generate()?),
		Err(error) if error.kind() == crate::ErrorKind::InvalidArgument));
	assert_eq!(handle.state.borrow().next_epoch, before);
	assert!(
		matches!(engine.capture(|| handle.generate_mlkem_key(2, Entropy::generate()?).map(|_| ())),
		Err(error) if error.kind() == crate::ErrorKind::FailedPrecondition)
	);
	Ok(())
}

#[test]
#[ignore = "requires Vulkan; abandoned retained-key quarantine validation"]
fn secret_mlkem_retained_key_drop_never_submits_or_erases() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let (key, seed_erasure) = handle.generate_mlkem_key(2, Entropy::generate()?)?;
	let before = handle.state.borrow().next_epoch;
	let public = key.public.clone();
	drop(key);
	assert_eq!(handle.state.borrow().next_epoch, before);
	// A command-held reference survives abandoning the value. Only entropy
	// erasure completes; the private-key allocation is quarantined, not freed.
	seed_erasure.wait()?;
	assert!(seed_erasure.try_is_complete()?);
	assert_eq!(handle.state.borrow().next_epoch, before);
	let mut bytes = vec![0; public.byte_len()];
	public.read(0, &mut bytes)?;
	assert!(bytes.iter().any(|byte| *byte != 0));
	Ok(())
}

#[test]
#[ignore = "requires explicit Vulkan selection; private KEM ownership integration"]
fn secret_mlkem_kem_chains_without_host_waits_and_erases_all_owners() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	for k in [2, 3, 4] {
		let (key, seed_erasure) = handle.generate_mlkem_key(k, Entropy::generate()?)?;
		let (encapsulated, ciphertext, status, message_erasure) =
			handle.encapsulate_mlkem(&key, Entropy::generate()?)?;
		let decapsulated = handle.decapsulate_mlkem(&key, &ciphertext)?;
		// No waits between producers, consumers or consuming erasures.
		let first = handle.erase_mlkem_shared_secret(encapsulated)?;
		let second = handle.erase_mlkem_shared_secret(decapsulated)?;
		let key_erasure = handle.erase_mlkem_key(key)?;
		key_erasure.wait()?;
		for ticket in [seed_erasure, message_erasure, first, second, key_erasure] {
			assert!(ticket.try_is_complete()?);
		}
		let mut valid = [0; 4];
		status.read(0, &mut valid)?;
		assert_eq!(u32::from_le_bytes(valid), 1);
		let mut encoded = vec![0; ciphertext.size() as usize];
		ciphertext.read(0, &mut encoded)?;
		assert!(encoded.iter().any(|byte| *byte != 0));
	}
	Ok(())
}

#[test]
#[ignore = "requires explicit Vulkan selection; private KEM validation"]
fn secret_mlkem_kem_rejects_capture_and_wrong_ciphertext_before_submission() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let (key, erasure) = handle.generate_mlkem_key(2, Entropy::generate()?)?;
	let device = handle.state.borrow().device.clone();
	let wrong = Buffer::host_visible_storage(&device, 4)?;
	let epoch = handle.state.borrow().next_epoch;
	assert!(matches!(handle.decapsulate_mlkem(&key, &wrong), Err(error)
		if error.kind() == crate::ErrorKind::InvalidArgument));
	assert_eq!(handle.state.borrow().next_epoch, epoch);
	erasure.wait()?;
	assert!(
		matches!(engine.capture(|| handle.encapsulate_mlkem(&key, Entropy::generate()?).map(|_| ())),
		Err(error) if error.kind() == crate::ErrorKind::FailedPrecondition)
	);
	assert_eq!(handle.state.borrow().next_epoch, epoch);
	handle.erase_mlkem_key(key)?.wait()?;
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; NIST public-key/secret-erasure proof"]
fn secret_mldsa_keygen_nist_public_keys_and_retained_erasure() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let prompt: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/keyGen-prompt.json"
	))
	.expect("pinned NIST prompt");
	let expected: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/keyGen-expectedResults.json"
	))
	.expect("pinned NIST results");
	fn decode(value: &serde_json::Value) -> Vec<u8> {
		let text = value.as_str().expect("fixture hex");
		text
			.as_bytes()
			.as_chunks::<2>()
			.0
			.iter()
			.map(|pair| {
				u8::from_str_radix(std::str::from_utf8(pair).expect("ASCII"), 16).expect("hex byte")
			})
			.collect()
	}
	let mut cases = 0;
	for group in prompt["testGroups"].as_array().expect("groups") {
		let parameter = match group["parameterSet"].as_str().expect("parameter") {
			"ML-DSA-44" => 44,
			"ML-DSA-65" => 65,
			"ML-DSA-87" => 87,
			_ => panic!("unknown fixture parameter"),
		};
		let answers = expected["testGroups"]
			.as_array()
			.expect("results")
			.iter()
			.find(|answer| answer["tgId"] == group["tgId"])
			.expect("result group");
		// Every unchanged ACVP key-generation vector, without secret readback.
		for case in group["tests"].as_array().expect("cases") {
			let answer = answers["tests"]
				.as_array()
				.expect("answers")
				.iter()
				.find(|answer| answer["tcId"] == case["tcId"])
				.expect("answer");
			let mut seed = [0; 32];
			seed.copy_from_slice(&decode(&case["seed"]));
			let entropy = Entropy::test_from_bytes(&seed)?;
			crate::core::memory::zero_secure(&mut seed);
			let (key, seed_erasure) = handle.generate_mldsa_key(parameter, entropy)?;
			assert_eq!(key.parameter, parameter);
			let public = key.public.clone();
			let ready = key.ready.clone();
			let erasure = handle.erase_mldsa_key(key)?;
			erasure.wait()?;
			assert!(erasure.try_is_complete()?);
			assert!(seed_erasure.try_is_complete()?);
			assert!(ready.is_complete()?);
			let mut bytes = vec![0; public.byte_len()];
			public.read(0, &mut bytes)?;
			assert_eq!(bytes, decode(&answer["pk"]), "NIST tcId {}", case["tcId"]);
			cases += 1;
		}
	}
	assert_eq!(cases, 75);
	Ok(())
}

#[test]
#[ignore = "requires explicit Vulkan selection; ML-DSA ownership preflight"]
fn secret_mldsa_keygen_rejects_invalid_parameter_and_capture() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let before = handle.state.borrow().next_epoch;
	for parameter in [0, 43, 64, 86, u32::MAX] {
		assert!(
			matches!(handle.generate_mldsa_key(parameter, Entropy::generate()?),
			Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
		);
	}
	assert_eq!(handle.state.borrow().next_epoch, before);
	assert!(
		matches!(engine.capture(|| handle.generate_mldsa_key(44, Entropy::generate()?).map(|_| ())),
		Err(error) if error.kind() == crate::ErrorKind::FailedPrecondition)
	);
	assert_eq!(handle.state.borrow().next_epoch, before);
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; private ML-DSA signing integration"]
fn secret_mldsa_sign_mu_all_parameters_and_immediate_key_erasure() -> Result<()> {
	use crate::cryptography::{entropy::Entropy, shake256};
	use ml_dsa::{MlDsa44, MlDsa65, MlDsa87, Signature, Verifier as _, VerifyingKey};
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let prompt: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/keyGen-prompt.json"
	))
	.expect("NIST prompt");
	let expected: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/keyGen-expectedResults.json"
	))
	.expect("NIST results");
	fn decode(value: &serde_json::Value) -> Vec<u8> {
		let text = value.as_str().expect("hex");
		assert_eq!(text.len() % 2, 0);
		text
			.as_bytes()
			.as_chunks::<2>()
			.0
			.iter()
			.map(|pair| {
				u8::from_str_radix(std::str::from_utf8(pair).expect("ASCII"), 16).expect("hex byte")
			})
			.collect()
	}
	for (index, parameter) in [44, 65, 87].into_iter().enumerate() {
		let case = &prompt["testGroups"][index]["tests"][0];
		let answer = &expected["testGroups"][index]["tests"][0];
		assert_eq!(case["tcId"], answer["tcId"]);
		let mut seed: [u8; 32] = decode(&case["seed"]).try_into().expect("seed size");
		let pk = decode(&answer["pk"]);
		let entropy = Entropy::test_from_bytes(&seed)?;
		crate::core::memory::zero_secure(&mut seed);
		let (key, key_seed_erasure) = handle.generate_mldsa_key(parameter, entropy)?;
		// Derive mu using only public NIST bytes. No host wait for key generation:
		// the consumer must provide the actual device producer dependency.
		let message = b"OA private signing lifecycle proof";
		let mut tr = [0u8; 64];
		shake256(&pk, &mut tr);
		let mut framed = tr.to_vec();
		framed.extend_from_slice(&[0, 0]);
		framed.extend_from_slice(message);
		let mut mu_bytes = [0u8; 64];
		shake256(&framed, &mut mu_bytes);
		let mu = Buffer::host_visible_storage(&device, 64)?;
		mu.write(0, &mu_bytes)?;
		let mut signatures = Vec::new();
		for rnd in [[0u8; 32], [0u8; 32], [0x33u8; 32]] {
			signatures.push(handle.sign_mldsa_mu(&key, &mu, Entropy::test_from_bytes(&rnd)?)?);
		}
		// Consuming key erasure immediately after all consumers, without waits.
		let erased = handle.erase_mldsa_key(key)?;
		erased.wait()?;
		assert!(key_seed_erasure.try_is_complete()?);
		let (_, _, _, wire_len) =
			crate::runtime::shader::KernelId::mldsa_sign_layout(parameter).expect("layout");
		let mut encodings = Vec::new();
		for (signature, status, entropy_erasure) in signatures {
			// Device entropy, locked staging and the secret signing cache share
			// this exact completion witness; no allocation can retire early.
			assert_eq!(entropy_erasure.test_retained_allocation_count(), 3);
			assert!(entropy_erasure.try_is_complete()?);
			let mut result = [0u8; 4];
			status.read(0, &mut result)?;
			assert_eq!(u32::from_le_bytes(result), 1);
			let mut bytes = vec![0; signature.size() as usize];
			signature.read(0, &mut bytes)?;
			assert!(bytes[wire_len as usize..].iter().all(|&byte| byte == 0));
			bytes.truncate(wire_len as usize);
			macro_rules! verify {
				($set:ty, $pk_len:expr, $sig_len:expr) => {{
					let encoded_pk: &[u8; $pk_len] = pk.as_slice().try_into().expect("pk size");
					let encoded_sig: &[u8; $sig_len] = bytes.as_slice().try_into().expect("signature size");
					let vk = VerifyingKey::<$set>::decode(encoded_pk.into());
					let sig = Signature::<$set>::decode(encoded_sig.into()).expect("canonical signature");
					assert!(vk.verify(message, &sig).is_ok());
					assert!(vk.verify(b"wrong message", &sig).is_err());
				}};
			}
			match parameter {
				44 => verify!(MlDsa44, 1312, 2420),
				65 => verify!(MlDsa65, 1952, 3309),
				87 => verify!(MlDsa87, 2592, 4627),
				_ => unreachable!(),
			}
			encodings.push(bytes);
		}
		assert_eq!(encodings[0], encodings[1]);
		assert_ne!(encodings[0], encodings[2]);
	}
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; signing preflight rejection"]
fn secret_mldsa_sign_rejects_extent_foreign_engine_and_capture() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let (key, seed_erasure) = handle.generate_mldsa_key(44, Entropy::generate()?)?;
	let mu = Buffer::host_visible_storage(&device, 64)?;
	let short_mu = Buffer::host_visible_storage(&device, 63)?;
	let before = handle.state.borrow().next_epoch;
	assert!(
		matches!(handle.sign_mldsa_mu(&key, &short_mu, Entropy::generate()?),
		Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert!(
		matches!(engine.capture(|| handle.sign_mldsa_mu(&key, &mu, Entropy::generate()?).map(|_| ())),
		Err(error) if error.kind() == crate::ErrorKind::FailedPrecondition)
	);
	assert_eq!(handle.state.borrow().next_epoch, before);
	let foreign = secret_test_engine()?;
	let foreign_device = foreign.handle().state.borrow().device.clone();
	let foreign_mu = Buffer::host_visible_storage(&foreign_device, 64)?;
	assert!(
		matches!(handle.sign_mldsa_mu(&key, &foreign_mu, Entropy::generate()?),
		Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert_eq!(handle.state.borrow().next_epoch, before);
	assert!(
		matches!(foreign.handle().sign_mldsa_mu(&key, &mu, Entropy::generate()?),
		Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	handle.erase_mldsa_key(key)?.wait()?;
	assert!(seed_erasure.try_is_complete()?);
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; pure ML-DSA framing and input retention"]
fn secret_mldsa_message_context_all_parameters() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	use ml_dsa::{MlDsa44, MlDsa65, MlDsa87, Signature, VerifyingKey};
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	for parameter in [44, 65, 87] {
		let (key, key_seed) =
			handle.generate_mldsa_key(parameter, Entropy::test_from_bytes(&[0x42; 32])?)?;
		let public = key.public.clone();
		let mut cases = Vec::new();
		for (message, context) in [
			(Vec::new(), Vec::new()),
			(vec![0x80, 0, 0xff], vec![0, 0xff, 0x80]),
			(vec![0x55; 137], (0..255).map(|n| n as u8).collect()),
		] {
			let message_buffer =
				Buffer::host_visible_storage(&device, ((message.len() + 3) & !3).max(4))?;
			let context_buffer =
				Buffer::host_visible_storage(&device, ((context.len() + 3) & !3).max(4))?;
			// Poison physical padding: only logical bytes may enter the hash.
			let mut message_storage = vec![0xa5; message_buffer.size() as usize];
			message_storage[..message.len()].copy_from_slice(&message);
			message_buffer.write(0, &message_storage)?;
			let mut context_storage = vec![0x5a; context_buffer.size() as usize];
			context_storage[..context.len()].copy_from_slice(&context);
			context_buffer.write(0, &context_storage)?;
			let result = handle.sign_mldsa_message(
				&key,
				&message_buffer,
				&context_buffer,
				[message.len() as u32, context.len() as u32],
				Entropy::test_from_bytes(&[0; 32])?,
			)?;
			cases.push((message, context, result));
			// Input handles die before completion; commands must retain both.
		}
		handle.erase_mldsa_key(key)?.wait()?;
		assert!(key_seed.try_is_complete()?);
		let mut pk = vec![0; public.size() as usize];
		public.read(0, &mut pk)?;
		let (_, _, _, wire_len) =
			crate::runtime::shader::KernelId::mldsa_sign_layout(parameter).expect("layout");
		for (message, context, (signature, status, entropy_erasure)) in cases {
			assert_eq!(entropy_erasure.test_retained_allocation_count(), 3);
			assert!(entropy_erasure.try_is_complete()?);
			let mut result = [0; 4];
			status.read(0, &mut result)?;
			assert_eq!(u32::from_le_bytes(result), 1);
			let mut bytes = vec![0; signature.size() as usize];
			signature.read(0, &mut bytes)?;
			assert!(bytes[wire_len as usize..].iter().all(|&n| n == 0));
			bytes.truncate(wire_len as usize);
			macro_rules! verify {
				($set:ty, $pk_len:expr, $sig_len:expr) => {{
					let pk: &[u8; $pk_len] = pk.as_slice().try_into().expect("pk size");
					let sig: &[u8; $sig_len] = bytes.as_slice().try_into().expect("signature size");
					let verifier = VerifyingKey::<$set>::decode(pk.into());
					let signature = Signature::<$set>::decode(sig.into()).expect("canonical signature");
					assert!(verifier.verify_with_context(&message, &context, &signature));
					assert!(!verifier.verify_with_context(b"wrong message", &context, &signature));
					assert!(!verifier.verify_with_context(&message, b"wrong context", &signature));
				}};
			}
			match parameter {
				44 => verify!(MlDsa44, 1312, 2420),
				65 => verify!(MlDsa65, 1952, 3309),
				87 => verify!(MlDsa87, 2592, 4627),
				_ => unreachable!(),
			}
		}
	}
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; pure signing preflight rejection"]
fn secret_mldsa_message_rejects_invalid_inputs() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let (key, key_seed) = handle.generate_mldsa_key(44, Entropy::test_from_bytes(&[0x42; 32])?)?;
	let message = Buffer::host_visible_storage(&device, 4)?;
	let context = Buffer::host_visible_storage(&device, 4)?;
	let before = handle.state.borrow().next_epoch;
	for lengths in [[0, 256], [u32::MAX, 0], [5, 0], [0, 5]] {
		assert!(
			matches!(handle.sign_mldsa_message(&key, &message, &context, lengths,
			Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
		);
	}
	assert!(
		matches!(handle.sign_mldsa_message(&key, &message, &message, [0, 0],
		Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert!(
		matches!(engine.capture(|| handle.sign_mldsa_message(&key, &message, &context,
		[0, 0], Entropy::test_from_bytes(&[0; 32])?).map(|_| ())),
		Err(error) if error.kind() == crate::ErrorKind::FailedPrecondition)
	);
	let foreign = secret_test_engine()?;
	let foreign_device = foreign.handle().state.borrow().device.clone();
	let foreign_input = Buffer::host_visible_storage(&foreign_device, 4)?;
	for (message, context) in [(&foreign_input, &context), (&message, &foreign_input)] {
		assert!(
			matches!(handle.sign_mldsa_message(&key, message, context, [0, 0],
			Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
		);
	}
	assert!(
		matches!(foreign.handle().sign_mldsa_message(&key, &message, &context, [0, 0],
		Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert_eq!(handle.state.borrow().next_epoch, before);
	handle.erase_mldsa_key(key)?.wait()?;
	assert!(key_seed.try_is_complete()?);
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; HashML-DSA signing preflight rejection"]
fn secret_mldsa_hash_message_rejects_invalid_inputs() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let (key, key_seed) = handle.generate_mldsa_key(44, Entropy::test_from_bytes(&[0x42; 32])?)?;
	let message = Buffer::host_visible_storage(&device, 4)?;
	let context = Buffer::host_visible_storage(&device, 4)?;
	let before = handle.state.borrow().next_epoch;
	for algorithm in [0, 13, u32::MAX] {
		assert!(
			matches!(handle.sign_mldsa_hash_message(&key, &message, &context,
			[0, 0], algorithm, Entropy::test_from_bytes(&[0; 32])?),
			Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
		);
	}
	for lengths in [[0, 256], [u32::MAX, 0], [5, 0], [0, 5]] {
		assert!(
			matches!(handle.sign_mldsa_hash_message(&key, &message, &context, lengths, 12,
			Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
		);
	}
	assert!(
		matches!(handle.sign_mldsa_hash_message(&key, &message, &message, [0, 0], 12,
		Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert!(
		matches!(engine.capture(|| handle.sign_mldsa_hash_message(&key, &message, &context,
		[0, 0], 12, Entropy::test_from_bytes(&[0; 32])?).map(|_| ())),
		Err(error) if error.kind() == crate::ErrorKind::FailedPrecondition)
	);
	let foreign = secret_test_engine()?;
	let foreign_device = foreign.handle().state.borrow().device.clone();
	let foreign_input = Buffer::host_visible_storage(&foreign_device, 4)?;
	for (message, context) in [(&foreign_input, &context), (&message, &foreign_input)] {
		assert!(
			matches!(handle.sign_mldsa_hash_message(&key, message, context, [0, 0], 12,
			Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
		);
	}
	assert!(
		matches!(foreign.handle().sign_mldsa_hash_message(&key, &message, &context, [0, 0], 12,
		Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert_eq!(handle.state.borrow().next_epoch, before);
	handle.erase_mldsa_key(key)?.wait()?;
	assert!(key_seed.try_is_complete()?);
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; prehashed signing integration"]
fn secret_mldsa_prehashed_all_parameters() -> Result<()> {
	use crate::cryptography::{entropy::Entropy, shake256};
	use ml_dsa::{MlDsa44, MlDsa65, MlDsa87, Signature, VerifyingKey};
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let message = b"OA prehashed signing integration";
	let context = b"OA context";
	let mut hash = [0u8; 64];
	shake256(message, &mut hash);
	let mut framed = vec![1, context.len() as u8];
	framed.extend_from_slice(context);
	framed.extend_from_slice(&[6, 9, 96, 134, 72, 1, 101, 3, 4, 2, 12]);
	framed.extend_from_slice(&hash);
	for parameter in [44, 65, 87] {
		let (key, key_seed) =
			handle.generate_mldsa_key(parameter, Entropy::test_from_bytes(&[0x42; 32])?)?;
		let public = key.public.clone();
		let digest = Buffer::host_visible_storage(&device, 64)?;
		digest.write(0, &hash)?;
		let context_buffer = Buffer::host_visible_storage(&device, (context.len() + 3) & !3)?;
		let mut context_storage = vec![0xa5; context_buffer.size() as usize];
		context_storage[..context.len()].copy_from_slice(context);
		context_buffer.write(0, &context_storage)?;
		let (signature, status, entropy_erasure) = handle.sign_mldsa_prehashed(
			&key,
			&digest,
			&context_buffer,
			12,
			context.len() as u32,
			Entropy::test_from_bytes(&[0; 32])?,
		)?;
		drop(digest);
		drop(context_buffer);
		handle.erase_mldsa_key(key)?.wait()?;
		assert!(key_seed.try_is_complete()?);
		assert_eq!(entropy_erasure.test_retained_allocation_count(), 3);
		assert!(entropy_erasure.try_is_complete()?);
		let mut pk = vec![0; public.size() as usize];
		public.read(0, &mut pk)?;
		let mut status_bytes = [0; 4];
		status.read(0, &mut status_bytes)?;
		assert_eq!(u32::from_le_bytes(status_bytes), 1);
		let mut bytes = vec![0; signature.size() as usize];
		signature.read(0, &mut bytes)?;
		let (_, _, _, wire_len) =
			crate::runtime::shader::KernelId::mldsa_sign_layout(parameter).expect("layout");
		assert!(bytes[wire_len as usize..].iter().all(|&n| n == 0));
		bytes.truncate(wire_len as usize);
		macro_rules! verify {
			($set:ty, $pk_len:expr, $sig_len:expr) => {{
				let pk: &[u8; $pk_len] = pk.as_slice().try_into().expect("pk size");
				let sig: &[u8; $sig_len] = bytes.as_slice().try_into().expect("signature size");
				let verifier = VerifyingKey::<$set>::decode(pk.into());
				let signature = Signature::<$set>::decode(sig.into()).expect("canonical signature");
				assert!(verifier.verify_internal(&framed, &signature));
				let mut wrong_oid = framed.clone();
				wrong_oid[2 + context.len() + 10] = 3;
				assert!(!verifier.verify_internal(&wrong_oid, &signature));
				assert!(!verifier.verify_with_context(message, context, &signature));
			}};
		}
		match parameter {
			44 => verify!(MlDsa44, 1312, 2420),
			65 => verify!(MlDsa65, 1952, 3309),
			87 => verify!(MlDsa87, 2592, 4627),
			_ => unreachable!(),
		}
	}
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; message-taking HashML-DSA signing integration"]
fn secret_mldsa_hash_message_all_parameters() -> Result<()> {
	use crate::cryptography::{entropy::Entropy, shake256};
	use ml_dsa::{MlDsa44, MlDsa65, MlDsa87, Signature, VerifyingKey};
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let message = b"OA message-taking HashML-DSA signing integration";
	let context = b"OA context";
	let mut hash = [0u8; 64];
	shake256(message, &mut hash);
	let mut framed = vec![1, context.len() as u8];
	framed.extend_from_slice(context);
	framed.extend_from_slice(&[6, 9, 96, 134, 72, 1, 101, 3, 4, 2, 12]);
	framed.extend_from_slice(&hash);
	for parameter in [44, 65, 87] {
		let (key, key_seed) =
			handle.generate_mldsa_key(parameter, Entropy::test_from_bytes(&[0x42; 32])?)?;
		let public = key.public.clone();
		let input = Buffer::host_visible_storage(&device, (message.len() + 3) & !3)?;
		let mut storage = vec![0xa5; input.size() as usize];
		storage[..message.len()].copy_from_slice(message);
		input.write(0, &storage)?;
		let context_buffer = Buffer::host_visible_storage(&device, (context.len() + 3) & !3)?;
		let mut context_storage = vec![0xa5; context_buffer.size() as usize];
		context_storage[..context.len()].copy_from_slice(context);
		context_buffer.write(0, &context_storage)?;
		let (signature, status, entropy_erasure) = handle.sign_mldsa_hash_message(
			&key,
			&input,
			&context_buffer,
			[message.len() as u32, context.len() as u32],
			12,
			Entropy::test_from_bytes(&[0; 32])?,
		)?;
		drop(input);
		drop(context_buffer);
		handle.erase_mldsa_key(key)?.wait()?;
		assert!(key_seed.try_is_complete()?);
		assert_eq!(entropy_erasure.test_retained_allocation_count(), 3);
		assert!(entropy_erasure.try_is_complete()?);
		let mut pk = vec![0; public.size() as usize];
		public.read(0, &mut pk)?;
		let mut status_bytes = [0; 4];
		status.read(0, &mut status_bytes)?;
		assert_eq!(u32::from_le_bytes(status_bytes), 1);
		let mut bytes = vec![0; signature.size() as usize];
		signature.read(0, &mut bytes)?;
		let (_, _, _, wire_len) =
			crate::runtime::shader::KernelId::mldsa_sign_layout(parameter).expect("layout");
		assert!(bytes[wire_len as usize..].iter().all(|&n| n == 0));
		bytes.truncate(wire_len as usize);
		macro_rules! verify {
			($set:ty, $pk_len:expr, $sig_len:expr) => {{
				let pk: &[u8; $pk_len] = pk.as_slice().try_into().expect("pk size");
				let sig: &[u8; $sig_len] = bytes.as_slice().try_into().expect("signature size");
				let verifier = VerifyingKey::<$set>::decode(pk.into());
				let signature = Signature::<$set>::decode(sig.into()).expect("canonical signature");
				assert!(verifier.verify_internal(&framed, &signature));
				let mut wrong_oid = framed.clone();
				wrong_oid[2 + context.len() + 10] = 3;
				assert!(!verifier.verify_internal(&wrong_oid, &signature));
				assert!(!verifier.verify_with_context(message, context, &signature));
			}};
		}
		match parameter {
			44 => verify!(MlDsa44, 1312, 2420),
			65 => verify!(MlDsa65, 1952, 3309),
			87 => verify!(MlDsa87, 2592, 4627),
			_ => unreachable!(),
		}
	}
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; prehashed signing rejection"]
fn secret_mldsa_prehashed_rejects_identity_extent_and_context() -> Result<()> {
	use crate::cryptography::entropy::Entropy;
	let engine = secret_test_engine()?;
	let handle = engine.handle();
	let device = handle.state.borrow().device.clone();
	let (key, key_seed) = handle.generate_mldsa_key(44, Entropy::test_from_bytes(&[0x42; 32])?)?;
	let digest = Buffer::host_visible_storage(&device, 64)?;
	let context = Buffer::host_visible_storage(&device, 4)?;
	let before = handle.state.borrow().next_epoch;
	for (algorithm, length) in [(0, 0), (13, 0), (1, 0), (12, 256), (12, u32::MAX), (12, 5)] {
		assert!(
			matches!(handle.sign_mldsa_prehashed(&key, &digest, &context, algorithm, length,
			Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
		);
	}
	assert!(
		matches!(handle.sign_mldsa_prehashed(&key, &digest, &digest, 12, 64,
		Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert!(
		matches!(engine.capture(|| handle.sign_mldsa_prehashed(&key, &digest, &context,
		12, 0, Entropy::test_from_bytes(&[0; 32])?).map(|_| ())),
		Err(error) if error.kind() == crate::ErrorKind::FailedPrecondition)
	);
	let foreign = secret_test_engine()?;
	let foreign_digest = Buffer::host_visible_storage(&foreign.handle().state.borrow().device, 64)?;
	assert!(
		matches!(handle.sign_mldsa_prehashed(&key, &foreign_digest, &context, 12, 0,
		Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	let foreign_context = Buffer::host_visible_storage(&foreign.handle().state.borrow().device, 4)?;
	assert!(
		matches!(handle.sign_mldsa_prehashed(&key, &digest, &foreign_context, 12, 0,
		Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert!(
		matches!(foreign.handle().sign_mldsa_prehashed(&key, &digest, &context, 12, 0,
		Entropy::test_from_bytes(&[0; 32])?), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
	);
	assert_eq!(handle.state.borrow().next_epoch, before);
	handle.erase_mldsa_key(key)?.wait()?;
	assert!(key_seed.try_is_complete()?);
	Ok(())
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; prehashed verification capture and rebinding"]
fn mldsa_prehashed_verifier_nist_capture() -> Result<()> {
	mldsa_hash_verifier_nist_capture(false)
}

#[test]
#[ignore = "requires explicitly selected Vulkan device; message-taking verification capture and rebinding"]
fn mldsa_hash_message_verifier_nist_capture() -> Result<()> {
	mldsa_hash_verifier_nist_capture(true)
}

// Trusted NIST fixtures use expanded keys; this decoder is test-only.
#[allow(
	deprecated,
	reason = "pinned NIST expanded test keys, not production key admission"
)]
fn mldsa_hash_verifier_nist_capture(hash_message: bool) -> Result<()> {
	use crate::Matrix;
	use crate::cryptography::{
		pqc::{MlDsaParameters, verify_hash_message_batch, verify_prehashed_batch},
		shake256,
	};
	if std::env::var_os("OA_TEST_DEVICE_INDEX").is_none() {
		return Err(Error::invalid_argument(
			"hash verification qualification requires an explicit OA_TEST_DEVICE_INDEX",
		));
	}
	let engine = secret_test_engine()?;
	let verify = if hash_message {
		verify_hash_message_batch
	} else {
		verify_prehashed_batch
	};
	let prompt: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/sigGen-prompt.json"
	))
	.expect("pinned prompt");
	let answers: serde_json::Value = serde_json::from_str(include_str!(
		"../../fixtures/cryptography/mldsa/sigGen-expectedResults.json"
	))
	.expect("pinned answers");
	let decode = |value: &serde_json::Value| -> Vec<u8> {
		let hex = value.as_str().expect("hex string");
		(0..hex.len())
			.step_by(2)
			.map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte"))
			.collect()
	};
	for (parameters, name) in [
		(MlDsaParameters::MlDsa44, "ML-DSA-44"),
		(MlDsaParameters::MlDsa65, "ML-DSA-65"),
		(MlDsaParameters::MlDsa87, "ML-DSA-87"),
	] {
		let group = prompt["testGroups"]
			.as_array()
			.expect("groups")
			.iter()
			.find(|g| {
				g["parameterSet"] == name && g["preHash"] == "preHash" && g["deterministic"] == true
			})
			.expect("prehash group");
		let case = group["tests"]
			.as_array()
			.expect("cases")
			.iter()
			.find(|c| c["hashAlg"] == "SHAKE-256")
			.expect("SHAKE fixture");
		let answer_group = answers["testGroups"]
			.as_array()
			.expect("answers")
			.iter()
			.find(|g| g["tgId"] == group["tgId"])
			.expect("answer group");
		let answer = answer_group["tests"]
			.as_array()
			.expect("answers")
			.iter()
			.find(|c| c["tcId"] == case["tcId"])
			.expect("answer");
		let sk = decode(&case["sk"]);
		macro_rules! public_key {
			($parameter:ty) => {{
				let key = ml_dsa::ExpandedSigningKeyBytes::<$parameter>::try_from(sk.as_slice())
					.expect("key length");
				ml_dsa::ExpandedSigningKey::<$parameter>::from_expanded(&key)
					.verifying_key()
					.encode()
					.to_vec()
			}};
		}
		let pk = match parameters {
			MlDsaParameters::MlDsa44 => public_key!(ml_dsa::MlDsa44),
			MlDsaParameters::MlDsa65 => public_key!(ml_dsa::MlDsa65),
			MlDsaParameters::MlDsa87 => public_key!(ml_dsa::MlDsa87),
		};
		let sig = decode(&answer["signature"]);
		let context = decode(&case["context"]);
		let mut digest = [0_u8; 64];
		shake256(&decode(&case["message"]), &mut digest);
		let mut bytes = vec![0xa5];
		bytes.extend_from_slice(&context);
		bytes.push(0xa5);
		let offset = u32::try_from(bytes.len()).expect("offset");
		let message = decode(&case["message"]);
		let payload = if hash_message {
			message.as_slice()
		} else {
			digest.as_slice()
		};
		bytes.extend_from_slice(payload);
		let payload_length = u32::try_from(payload.len()).expect("payload length");
		let inputs = Matrix::from_slice(&engine, [bytes.len()], &bytes)?;
		let offsets = Matrix::from_slice(&engine, [1], &[offset])?;
		let lengths = Matrix::from_slice(&engine, [1], &[payload_length])?;
		let signatures = Matrix::from_slice(&engine, [sig.len()], &sig)?;
		let keys = Matrix::from_slice(&engine, [pk.len()], &pk)?;
		let ctx = 1..1 + context.len();
		let (mut plan, result) = engine.capture(|| {
			verify(
				parameters,
				12,
				&inputs,
				&offsets,
				&lengths,
				&signatures,
				&keys,
				ctx.clone(),
			)
		})?;
		assert_eq!(
			result
				.try_read::<u32>()
				.expect_err("unsubmitted result")
				.kind(),
			crate::ErrorKind::NotReady
		);
		assert!(plan.debug_report_json("prehash").contains(if hash_message {
			"verify_hash_message_batch"
		} else {
			"verify_prehashed_batch"
		}));
		for _ in 0..2 {
			engine.submit(&plan)?.wait()?;
			assert_eq!(result.read::<u32>()?, [1], "{name} NIST SHAKE-256");
		}
		bytes[offset as usize] ^= 1;
		let changed = Matrix::from_slice(&engine, [bytes.len()], &bytes)?;
		plan.bind_matrix_input(&inputs, &changed)?;
		drop(changed);
		engine.submit(&plan)?.wait()?;
		assert_eq!(result.read::<u32>()?, [0], "{name} changed payload");
		for algorithm in [3, 11] {
			assert_eq!(
				verify(
					parameters,
					algorithm,
					&inputs,
					&offsets,
					&lengths,
					&signatures,
					&keys,
					ctx.clone()
				)?
				.read::<u32>()?,
				[0],
				"wrong hash identity"
			);
		}
		for row in [
			[offset, payload_length.saturating_sub(1)],
			[u32::MAX, payload_length],
			[offset, u32::MAX],
		] {
			let offsets = Matrix::from_slice(&engine, [1], &row[..1])?;
			let lengths = Matrix::from_slice(&engine, [1], &row[1..])?;
			assert_eq!(
				verify(
					parameters,
					12,
					&inputs,
					&offsets,
					&lengths,
					&signatures,
					&keys,
					ctx.clone()
				)?
				.read::<u32>()?,
				[0]
			);
		}
		for algorithm in [0, 13, u32::MAX] {
			assert!(
				matches!(verify(parameters, algorithm, &inputs, &offsets, &lengths, &signatures, &keys, ctx.clone()), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
			);
		}
		assert_eq!(
			crate::cryptography::pqc::verify_batch_with_context(
				parameters,
				&inputs,
				&offsets,
				&lengths,
				&signatures,
				&keys,
				ctx.clone()
			)?
			.read::<u32>()?,
			[0],
			"prehash signature must reject pure framing"
		);
		let empty = Matrix::from_slice(&engine, [0], &[] as &[u8])?;
		let empty_rows = Matrix::from_slice(&engine, [0], &[] as &[u32])?;
		assert_eq!(
			verify(
				parameters,
				12,
				&empty,
				&empty_rows,
				&empty_rows,
				&empty,
				&empty,
				0..0
			)?
			.shape(),
			[0]
		);
		for ctx in [
			0..bytes.len() + 1,
			0..256,
			std::ops::Range { start: 2, end: 1 },
		] {
			assert!(
				matches!(verify(parameters, 12, &empty, &empty_rows, &empty_rows, &empty, &empty, ctx), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
			);
		}
		let foreign = secret_test_engine()?;
		let foreign_offsets = Matrix::from_slice(&foreign, [1], &[offset])?;
		assert!(
			matches!(verify(parameters, 12, &inputs, &foreign_offsets, &lengths, &signatures, &keys, ctx), Err(error) if error.kind() == crate::ErrorKind::InvalidArgument)
		);
	}
	Ok(())
}
