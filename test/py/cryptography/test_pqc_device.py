"""Deferred public binding integration; private formula KATs stay per kernel."""
import hashlib
import os
import unittest

import oa


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit software Vulkan qualification required")
class PqcDeviceVulkanTest(unittest.TestCase):
	def test_mldsa_public_routes_and_consumed_key(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		message_bytes, context_bytes = b"public input", b"OA"
		message = oa.Matrix.new_from_u8(engine, [len(message_bytes)], list(message_bytes))
		context = oa.Matrix.new_from_u8(engine, [len(context_bytes)], list(context_bytes))
		combined = oa.Matrix.new_from_u8(engine, [len(message_bytes) + len(context_bytes)], list(message_bytes + context_bytes))
		offsets = oa.Matrix.new_from_u32(engine, [1], [0])
		lengths = oa.Matrix.new_from_u32(engine, [1], [len(message_bytes)])
		for parameter, public_size, signature_size in ((44, 1312, 2420), (65, 1952, 3309), (87, 2592, 4627)):
			key = oa.cryptography.device_mldsa_keygen(engine, parameter)
			public = key.public_key
			self.assertEqual(len(public.read_u8()), public_size)
			first = oa.cryptography.device_mldsa_sign(key, message, context, True)
			second = oa.cryptography.device_mldsa_sign(key, message, context, True)
			first.wait_cleanup()
			second.wait_cleanup()
			self.assertEqual(first.status.read_u32(), [1])
			self.assertEqual(first.signature.read_u8(), second.signature.read_u8())
			self.assertEqual(len(first.signature.read_u8()), signature_size)
			packed_signature = first.signature.reshape([signature_size])
			packed_public = public.reshape([public_size])
			flags = oa.cryptography.device_mldsa_verify_message(parameter, combined, offsets, lengths,
				packed_signature, packed_public, len(message_bytes), len(message_bytes) + len(context_bytes))
			self.assertEqual(flags.read_u32(), [1])
			for name, digest_bytes in (("SHA2-256", hashlib.sha256(message_bytes).digest()),
				("SHAKE-256", hashlib.shake_256(message_bytes).digest(64))):
				digest = oa.Matrix.new_from_u8(engine, [len(digest_bytes)], list(digest_bytes))
				hashed = oa.cryptography.device_mldsa_sign_hash_message(key, message, context, name, True)
				prehashed = oa.cryptography.device_mldsa_sign_prehashed(key, digest, context, name, True)
				hashed.wait_cleanup()
				prehashed.wait_cleanup()
				self.assertEqual(hashed.status.read_u32(), [1])
				self.assertEqual(hashed.signature.read_u8(), prehashed.signature.read_u8())
				flags = oa.cryptography.device_mldsa_verify_hash_message(parameter, name, combined, offsets, lengths,
					hashed.signature.reshape([signature_size]), packed_public, len(message_bytes), len(message_bytes) + len(context_bytes))
				self.assertEqual(flags.read_u32(), [1])
			key.erase().wait()
			with self.assertRaises(Exception):
				key.erase()
			with self.assertRaises(Exception):
				oa.cryptography.device_mldsa_sign(key, message, context)

	def test_mlkem_public_lifecycle_and_no_secret_export(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for parameter, public_size, ciphertext_size in ((512, 800, 768), (768, 1184, 1088), (1024, 1568, 1568)):
			key = oa.cryptography.device_mlkem_keygen(engine, parameter)
			encapsulation = oa.cryptography.device_mlkem_encapsulate(key.public_key, parameter)
			decapsulation = oa.cryptography.device_mlkem_decapsulate(key, encapsulation.ciphertext)
			key.erase().wait()
			self.assertEqual(encapsulation.status.read_u32(), [1])
			self.assertEqual(len(encapsulation.ciphertext.read_u8()), ciphertext_size)
			for operation in (encapsulation, decapsulation):
				operation.wait_cleanup()
				secret = operation.take_shared_secret()
				self.assertFalse(hasattr(secret, "read_u8"))
				self.assertFalse(hasattr(secret, "to_bytes"))
				secret.erase().wait()
				with self.assertRaises(Exception):
					secret.erase()
				with self.assertRaises(Exception):
					operation.take_shared_secret()
			bad_key = oa.Matrix.new_from_u8(engine, [public_size], [255] * public_size)
			invalid = oa.cryptography.device_mlkem_encapsulate(bad_key, parameter)
			self.assertEqual(invalid.status.read_u32(), [0])
			self.assertEqual(invalid.ciphertext.read_u8(), [0] * ciphertext_size)
			invalid.wait_cleanup()
			invalid.take_shared_secret().erase().wait()
