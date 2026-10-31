"""Indexed SHAKE Merkle frames against independent hashlib construction."""
import hashlib
import json
import os
import struct
import unittest
from pathlib import Path

import oa

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/cryptography/hash/merkle_proofs.json"


class MerkleProofFixtureTest(unittest.TestCase):
	def test_frames_match_independent_tree(self):
		for case in json.loads(FIXTURE.read_text())["cases"]:
			levels = [[bytes.fromhex(v) for v in case["leaves"]]]
			while len(levels[-1]) > 1:
				level = levels[-1]
				levels.append([hashlib.shake_256(level[i] + level[min(i + 1, len(level) - 1)]).digest(32)
					for i in range(0, len(level), 2)])
			self.assertEqual(levels[-1][0].hex(), case["root"])
			count = len(levels[0])
			for index, frame in zip(case["indices"], case["frames"]):
				valid = index < count
				expected = struct.pack("<IIII", 1, count, index, int(valid))
				position = index
				for level in levels[:-1]:
					expected += level[min(position ^ 1, len(level) - 1)] if valid else bytes(32)
					position //= 2
				self.assertEqual(expected.hex(), frame)


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit Vulkan test device required")
class MerkleProofVulkanTest(unittest.TestCase):
	def test_selected_device_frames_and_rejection(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for case in json.loads(FIXTURE.read_text())["cases"]:
			count = len(case["leaves"])
			leaves = oa.Matrix.new_from_u8(engine, [count, 32], list(b"".join(bytes.fromhex(v) for v in case["leaves"])))
			selected = oa.Matrix.new_from_u8(engine, [5, 32], list(b"".join(bytes.fromhex(v) for v in case["selected_leaves"])))
			indices = oa.Matrix.new_from_u32(engine, [5], case["indices"])
			root = oa.Matrix.new_from_u8(engine, [1, 32], list(bytes.fromhex(case["root"])))
			proofs = oa.cryptography.hash_merkle_proofs(leaves, indices)
			self.assertEqual(bytes(proofs.read_u8()).hex(), "".join(case["frames"]))
			flags = oa.cryptography.hash_verify_merkle_proofs(selected, proofs, root, count)
			self.assertEqual(flags.read_u32(), case["valid"])
			wrong_count = oa.cryptography.hash_verify_merkle_proofs(selected, proofs, root, count + 1)
			self.assertEqual(wrong_count.read_u32(), [0] * 5)


if __name__ == "__main__":
	unittest.main()
