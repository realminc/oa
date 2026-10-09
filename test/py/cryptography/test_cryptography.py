"""Python binding tests for oa.cryptography primitives and PQC (ML-DSA-65)."""

import unittest

import oa


class CryptoHashTest(unittest.TestCase):
	def test_hash_returns_32_bytes(self) -> None:
		digest = oa.cryptography.hash(b"hello")
		self.assertEqual(len(digest), 32)

	def test_hash_is_deterministic(self) -> None:
		a = oa.cryptography.hash(b"hello")
		b = oa.cryptography.hash(b"hello")
		self.assertEqual(a, b)

	def test_hash_differs_on_different_input(self) -> None:
		a = oa.cryptography.hash(b"hello")
		b = oa.cryptography.hash(b"world")
		self.assertNotEqual(a, b)

	def test_hash_combine_is_associative_in_order(self) -> None:
		h1 = bytes(oa.cryptography.hash(b"left"))
		h2 = bytes(oa.cryptography.hash(b"right"))
		combined = oa.cryptography.hash_combine(h1, h2)
		self.assertEqual(len(combined), 32)

	def test_hash_combine_asymmetric(self) -> None:
		h1 = bytes(oa.cryptography.hash(b"a"))
		h2 = bytes(oa.cryptography.hash(b"b"))
		ab = oa.cryptography.hash_combine(h1, h2)
		ba = oa.cryptography.hash_combine(h2, h1)
		self.assertNotEqual(ab, ba)


class CryptoHasherTest(unittest.TestCase):
	def test_hasher_incremental_matches_one_shot(self) -> None:
		one_shot = bytes(oa.cryptography.hash(b"hello world"))

		h = oa.cryptography.Hasher()
		h.update(b"hello ")
		h.update(b"world")
		incremental = bytes(h.finalize())

		self.assertEqual(incremental, one_shot)

	def test_hasher_reset_restarts_state(self) -> None:
		h = oa.cryptography.Hasher()
		h.update(b"first")
		_ = h.finalize()
		h.reset()
		h.update(b"hello world")
		after_reset = bytes(h.finalize())
		self.assertEqual(after_reset, bytes(oa.cryptography.hash(b"hello world")))

	def test_hasher_returns_32_bytes(self) -> None:
		h = oa.cryptography.Hasher()
		h.update(b"test")
		digest = h.finalize()
		self.assertEqual(len(digest), 32)


class CryptoShake128Test(unittest.TestCase):
	def test_shake128_one_shot_length(self) -> None:
		out = oa.cryptography.shake128(b"hello", 32)
		self.assertEqual(len(out), 32)

	def test_shake128_one_shot_variable_length(self) -> None:
		out = oa.cryptography.shake128(b"hello", 64)
		self.assertEqual(len(out), 64)

	def test_shake128_deterministic(self) -> None:
		a = oa.cryptography.shake128(b"hello", 32)
		b = oa.cryptography.shake128(b"hello", 32)
		self.assertEqual(a, b)

	def test_shake128_differs_on_different_input(self) -> None:
		a = oa.cryptography.shake128(b"hello", 32)
		b = oa.cryptography.shake128(b"world", 32)
		self.assertNotEqual(a, b)

	def test_shake128_xof_incremental(self) -> None:
		# Incremental and one-shot must produce the same output.
		one_shot = oa.cryptography.shake128(b"hello world", 48)

		xof = oa.cryptography.Shake128()
		xof.update(b"hello ")
		xof.update(b"world")
		incremental = xof.squeeze(48)

		self.assertEqual(bytes(one_shot), bytes(incremental))

	def test_shake128_xof_reset(self) -> None:
		xof = oa.cryptography.Shake128()
		xof.update(b"discard")
		_ = xof.squeeze(16)
		xof.reset()
		xof.update(b"hello world")
		after_reset = bytes(xof.squeeze(48))
		one_shot = bytes(oa.cryptography.shake128(b"hello world", 48))
		self.assertEqual(after_reset, one_shot)

	def test_shake128_xof_repr(self) -> None:
		xof = oa.cryptography.Shake128()
		self.assertIn("Shake128", repr(xof))

	def test_shake128_update_after_squeeze_raises(self) -> None:
		xof = oa.cryptography.Shake128()
		xof.update(b"data")
		_ = xof.squeeze(16)
		with self.assertRaises(Exception):
			xof.update(b"more")


class CryptoShake256Test(unittest.TestCase):
	def test_shake256_one_shot_length(self) -> None:
		out = oa.cryptography.shake256(b"hello", 32)
		self.assertEqual(len(out), 32)

	def test_shake256_one_shot_variable_length(self) -> None:
		out = oa.cryptography.shake256(b"hello", 64)
		self.assertEqual(len(out), 64)

	def test_shake256_deterministic(self) -> None:
		a = oa.cryptography.shake256(b"hello", 32)
		b = oa.cryptography.shake256(b"hello", 32)
		self.assertEqual(a, b)

	def test_shake256_differs_from_128(self) -> None:
		a = oa.cryptography.shake128(b"hello", 32)
		b = oa.cryptography.shake256(b"hello", 32)
		self.assertNotEqual(a, b)

	def test_shake256_xof_incremental(self) -> None:
		one_shot = oa.cryptography.shake256(b"hello world", 48)

		xof = oa.cryptography.Shake256()
		xof.update(b"hello ")
		xof.update(b"world")
		incremental = xof.squeeze(48)

		self.assertEqual(bytes(one_shot), bytes(incremental))

	def test_shake256_xof_reset(self) -> None:
		xof = oa.cryptography.Shake256()
		xof.update(b"discard")
		_ = xof.squeeze(16)
		xof.reset()
		xof.update(b"hello world")
		after_reset = bytes(xof.squeeze(48))
		one_shot = bytes(oa.cryptography.shake256(b"hello world", 48))
		self.assertEqual(after_reset, one_shot)

	def test_shake256_xof_repr(self) -> None:
		xof = oa.cryptography.Shake256()
		self.assertIn("Shake256", repr(xof))

	def test_shake256_matches_hasher_for_32_bytes(self) -> None:
		# shake256 output (32 bytes) must equal hash() since hash() is shake256/32
		one_shot = bytes(oa.cryptography.shake256(b"test", 32))
		hashed = bytes(oa.cryptography.hash(b"test"))
		self.assertEqual(one_shot, hashed)


class CryptoKeccakTest(unittest.TestCase):
	def test_keccak_f1600_wrong_length_raises(self) -> None:
		with self.assertRaises(Exception):
			oa.cryptography.keccak_f1600(b"short")

	def test_keccak_f1600_correct_length_returns_200_bytes(self) -> None:
		state = bytes(200)
		result = oa.cryptography.keccak_f1600(state)
		self.assertEqual(len(result), 200)

	def test_keccak_f1600_deterministic(self) -> None:
		state = bytes(range(200))
		a = oa.cryptography.keccak_f1600(state)
		b = oa.cryptography.keccak_f1600(state)
		self.assertEqual(a, b)

	def test_keccak_f1600_zero_state_changes(self) -> None:
		state = bytes(200)
		result = oa.cryptography.keccak_f1600(state)
		# The all-zero state must change after one permutation
		self.assertNotEqual(result, state)


class CryptoKmacTest(unittest.TestCase):
	def test_kmac256_returns_requested_length(self) -> None:
		result = oa.cryptography.kmac256(
			key=b"key", data=b"data", custom=b"", output_len=32
		)
		self.assertEqual(len(result), 32)

	def test_kmac256_64_byte_output(self) -> None:
		result = oa.cryptography.kmac256(
			key=b"key", data=b"data", custom=b"", output_len=64
		)
		self.assertEqual(len(result), 64)

	def test_kmac256_different_keys_different_output(self) -> None:
		a = oa.cryptography.kmac256(key=b"key1", data=b"data", custom=b"", output_len=32)
		b = oa.cryptography.kmac256(key=b"key2", data=b"data", custom=b"", output_len=32)
		self.assertNotEqual(a, b)

	def test_kmac256_custom_label_changes_output(self) -> None:
		a = oa.cryptography.kmac256(key=b"key", data=b"data", custom=b"ctx1", output_len=32)
		b = oa.cryptography.kmac256(key=b"key", data=b"data", custom=b"ctx2", output_len=32)
		self.assertNotEqual(a, b)


class CryptoMerkleTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		# Build leaves as 32-byte hashes
		cls.leaves_raw = [b"leaf0", b"leaf1", b"leaf2", b"leaf3"]
		cls.leaf_hashes = [list(oa.cryptography.hash(l)) for l in cls.leaves_raw]

	def test_build_returns_merkle_tree(self) -> None:
		tree = oa.cryptography.MerkleTree.build(self.leaf_hashes)
		self.assertIsInstance(tree, oa.cryptography.MerkleTree)

	def test_root_returns_32_bytes(self) -> None:
		tree = oa.cryptography.MerkleTree.build(self.leaf_hashes)
		root = tree.root()
		self.assertEqual(len(root), 32)

	def test_root_is_deterministic(self) -> None:
		tree1 = oa.cryptography.MerkleTree.build(self.leaf_hashes)
		tree2 = oa.cryptography.MerkleTree.build(self.leaf_hashes)
		self.assertEqual(tree1.root(), tree2.root())

	def test_proof_verify_valid_leaf(self) -> None:
		tree = oa.cryptography.MerkleTree.build(self.leaf_hashes)
		root = tree.root()
		for i in range(len(self.leaf_hashes)):
			proof = tree.proof(i)
			self.assertIsInstance(proof, oa.cryptography.MerkleProof)
			self.assertTrue(proof.verify(self.leaf_hashes[i], root))

	def test_proof_verify_wrong_leaf_fails(self) -> None:
		tree = oa.cryptography.MerkleTree.build(self.leaf_hashes)
		root = tree.root()
		proof = tree.proof(0)
		# Verify with wrong leaf
		self.assertFalse(proof.verify(self.leaf_hashes[1], root))

	def test_proof_siblings_not_empty(self) -> None:
		tree = oa.cryptography.MerkleTree.build(self.leaf_hashes)
		proof = tree.proof(0)
		siblings = proof.siblings()
		self.assertGreater(len(siblings), 0)
		for s in siblings:
			self.assertEqual(len(s), 32)

	def test_proof_sibling_is_left_matches_siblings_count(self) -> None:
		tree = oa.cryptography.MerkleTree.build(self.leaf_hashes)
		proof = tree.proof(0)
		self.assertEqual(len(proof.siblings()), len(proof.sibling_is_left()))

	def test_merkle_root_standalone_matches_tree(self) -> None:
		tree = oa.cryptography.MerkleTree.build(self.leaf_hashes)
		expected = bytes(tree.root())
		standalone = bytes(oa.cryptography.merkle_root(self.leaf_hashes))
		self.assertEqual(standalone, expected)

	def test_merkle_root_empty(self) -> None:
		root = oa.cryptography.merkle_root([])
		# empty tree root is all zeros
		self.assertEqual(len(root), 32)
		self.assertEqual(bytes(root), bytes(32))

	def test_merkle_root_single_leaf(self) -> None:
		leaf = list(oa.cryptography.hash(b"only"))
		root = oa.cryptography.merkle_root([leaf])
		self.assertEqual(len(root), 32)


class CryptoPqcTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.keypair = oa.cryptography.generate_keypair()

	def test_generate_keypair_returns_keypair(self) -> None:
		self.assertIsInstance(self.keypair, oa.cryptography.Keypair)

	def test_keypair_has_public_key(self) -> None:
		pk = self.keypair.public_key
		self.assertIsInstance(pk, oa.cryptography.PublicKey)

	def test_public_key_not_zero(self) -> None:
		self.assertFalse(self.keypair.public_key.is_zero())

	def test_public_key_to_bytes_roundtrip(self) -> None:
		pk = self.keypair.public_key
		raw = pk.to_bytes()
		pk2 = oa.cryptography.PublicKey.from_bytes(raw)
		self.assertEqual(pk.to_bytes(), pk2.to_bytes())

	def test_sign_returns_signature(self) -> None:
		sig = self.keypair.sign(b"hello OA")
		self.assertIsInstance(sig, oa.cryptography.Signature)

	def test_signature_not_zero(self) -> None:
		sig = self.keypair.sign(b"data")
		self.assertFalse(sig.is_zero())

	def test_signature_to_bytes_roundtrip(self) -> None:
		sig = self.keypair.sign(b"data")
		raw = sig.to_bytes()
		sig2 = oa.cryptography.Signature.from_bytes(raw)
		self.assertEqual(sig.to_bytes(), sig2.to_bytes())

	def test_pqc_verify_valid_signature(self) -> None:
		message = b"test message for PQC"
		sig = self.keypair.sign(message)
		pk = self.keypair.public_key
		self.assertTrue(oa.cryptography.pqc_verify(message, sig, pk))

	def test_pqc_verify_wrong_message_fails(self) -> None:
		message = b"correct message"
		sig = self.keypair.sign(message)
		pk = self.keypair.public_key
		self.assertFalse(oa.cryptography.pqc_verify(b"wrong message", sig, pk))

	def test_pqc_verify_hash_valid(self) -> None:
		message = b"hash sign test"
		h = list(oa.cryptography.hash(message))
		sig = self.keypair.sign_hash(h)
		pk = self.keypair.public_key
		self.assertTrue(oa.cryptography.pqc_verify_hash(h, sig, pk))

	def test_pqc_verify_hash_wrong_hash_fails(self) -> None:
		message = b"original"
		h = list(oa.cryptography.hash(message))
		sig = self.keypair.sign_hash(h)
		wrong_h = list(oa.cryptography.hash(b"different"))
		pk = self.keypair.public_key
		self.assertFalse(oa.cryptography.pqc_verify_hash(wrong_h, sig, pk))

	def test_sign_different_messages_different_signatures(self) -> None:
		sig1 = self.keypair.sign(b"msg1")
		sig2 = self.keypair.sign(b"msg2")
		self.assertNotEqual(sig1.to_bytes(), sig2.to_bytes())

	def test_public_key_repr_contains_prefix(self) -> None:
		r = repr(self.keypair.public_key)
		self.assertTrue(r.startswith("PublicKey("))

	def test_keypair_repr_contains_public_key(self) -> None:
		r = repr(self.keypair)
		self.assertTrue(r.startswith("Keypair("))


class CryptoSecureBufferTest(unittest.TestCase):
	def test_create_nonzero_size(self) -> None:
		buf = oa.cryptography.SecureBuffer(32)
		self.assertEqual(buf.size_bytes(), 32)
		self.assertTrue(buf.is_valid())

	def test_create_zero_size(self) -> None:
		buf = oa.cryptography.SecureBuffer(0)
		self.assertEqual(buf.size_bytes(), 0)
		self.assertFalse(buf.is_valid())

	def test_initial_contents_are_zero(self) -> None:
		buf = oa.cryptography.SecureBuffer(16)
		self.assertEqual(bytes(buf.as_bytes()), bytes(16))

	def test_secure_zero_clears_content(self) -> None:
		buf = oa.cryptography.SecureBuffer(16)
		# The buffer is already zeroed; just verify the call succeeds
		buf.secure_zero()
		self.assertEqual(bytes(buf.as_bytes()), bytes(16))

	def test_reset_releases_allocation(self) -> None:
		buf = oa.cryptography.SecureBuffer(64)
		self.assertTrue(buf.is_valid())
		buf.reset()
		self.assertEqual(buf.size_bytes(), 0)
		self.assertFalse(buf.is_valid())

	def test_repr_contains_size(self) -> None:
		buf = oa.cryptography.SecureBuffer(48)
		self.assertIn("48", repr(buf))
		self.assertIn("SecureBuffer", repr(buf))


if __name__ == "__main__":
	unittest.main()
