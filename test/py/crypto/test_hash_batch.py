"""Independent fixed-digest fixtures and opt-in Vulkan binding proofs."""
import hashlib
import json
import os
import unittest
from pathlib import Path

import oa

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/crypto/hash/batch.json"
ALGORITHMS = (
	"sha224", "sha256", "sha384", "sha512", "sha512_224", "sha512_256",
	"sha3_224", "sha3_256", "sha3_384", "sha3_512",
)


class FixedHashFixtureTest(unittest.TestCase):
	def test_independent_oracles_and_binding_inventory(self):
		fixture = json.loads(FIXTURE.read_text())
		for name in ALGORITHMS:
			self.assertTrue(callable(getattr(oa.crypto, f"hash_{name}")))
		for case in fixture["cases"]:
			self.assertEqual(set(case["digests"]), set(ALGORITHMS))
			for row, message in enumerate(case["messages"]):
				message = bytes.fromhex(message)
				self.assertEqual(len(message), case["length"])
				for name in ALGORITHMS:
					self.assertEqual(hashlib.new(name, message).hexdigest(), case["digests"][name][row])
		for case in fixture["merkle_cases"]:
			level = [bytes.fromhex(leaf) for leaf in case["leaves"]]
			while len(level) > 1:
				level = [hashlib.shake_256(level[i] + level[min(i + 1, len(level) - 1)]).digest(32)
					for i in range(0, len(level), 2)]
			self.assertEqual(level[0].hex(), case["root"])
		for chain in fixture["chains"]:
			message = bytes.fromhex(chain["messages"])
			self.assertEqual(chain["sha256_sha3_256"], [
				hashlib.sha3_256(hashlib.sha256(message[i:i + 3]).digest()).hexdigest()
				for i in range(0, len(message), 3)
			])


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit Vulkan test device required")
class FixedHashVulkanTest(unittest.TestCase):
	def test_generated_adapters_match_hashlib(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for length in (0, 3, 112, 145):
			messages = [bytes((i + row) & 255 for i in range(length)) for row in range(3)]
			input_value = oa.Matrix.new_from_u8(engine, [3, length], list(b"".join(messages)))
			for name in ALGORITHMS:
				with self.subTest(name=name, length=length):
					output = getattr(oa.crypto, f"hash_{name}")(input_value)
					expected = b"".join(hashlib.new(name, message).digest() for message in messages)
					self.assertEqual(output.shape, [3, len(expected) // 3])
					self.assertEqual(bytes(output.to_list()), expected)

	def test_non_power_of_two_merkle_adapters(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for case in json.loads(FIXTURE.read_text())["merkle_cases"]:
			leaves = b"".join(bytes.fromhex(leaf) for leaf in case["leaves"])
			input_value = oa.Matrix.new_from_u8(engine, [len(case["leaves"]), 32], list(leaves))
			output = oa.crypto.hash_merkle_root(input_value)
			self.assertEqual(bytes(output.to_list()).hex(), case["root"])


if __name__ == "__main__":
	unittest.main()
