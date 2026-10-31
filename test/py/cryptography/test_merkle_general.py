"""Deferred selected-hash tree/proof tests with independent parent hashing."""
import hashlib
import json
import os
import struct
import unittest
from pathlib import Path

import oa
from .test_poseidon2 import permute

ROOT = Path(__file__).resolve().parents[3]
HASH_SCHEMA = ROOT / "tool/gen/fn/schema/cryptography/cryptography_hash.json"
ALGORITHMS = ("sha256", "sha512", "sha3_256", "sha3_512", "shake128", "shake256",
	"poseidon2_babybear16", "poseidon2_babybear24", "poseidon2_koalabear16", "poseidon2_koalabear24")
PROFILES = {p["name"]: p for p in json.loads(HASH_SCHEMA.read_text())["poseidon2_profiles"]}


def parent(left, right, algorithm):
	if algorithm.startswith("poseidon2_"):
		profile = PROFILES[algorithm.removeprefix("poseidon2_")]
		width, prime = profile["width"], profile["prime"]
		rate, state = width - 8, [0] * width
		state[rate], state[rate + 1] = 0x4F414D32, 16
		cursor = 0
		for value in struct.unpack("<16I", left + right):
			state[cursor] = (state[cursor] + value) % prime
			cursor += 1
			if cursor == rate:
				state, cursor = permute(state, profile), 0
		state[cursor] = (state[cursor] + 1) % prime
		state = permute(state, profile)
		return struct.pack("<8I", *state[:8])
	if algorithm == "shake128":
		return hashlib.shake_128(left + right).digest(32)
	if algorithm == "shake256":
		return hashlib.shake_256(left + right).digest(32)
	return getattr(hashlib, algorithm)(left + right).digest()


def levels_of(leaves, algorithm):
	levels = [leaves]
	while len(levels[-1]) > 1:
		row = levels[-1]
		levels.append([parent(row[i], row[min(i + 1, len(row) - 1)], algorithm)
			for i in range(0, len(row), 2)])
	return levels


def matrix(engine, rows, algorithm):
	if algorithm.startswith("poseidon2_"):
		return oa.Matrix.new_from_u32(engine, [len(rows), 8], [n for row in rows for n in struct.unpack("<8I", row)])
	return oa.Matrix.new_from_u8(engine, [len(rows), len(rows[0])], list(b"".join(rows)))


def read(matrix_, algorithm):
	if algorithm.startswith("poseidon2_"):
		values = matrix_.read_u32()
		return struct.pack(f"<{len(values)}I", *values)
	return bytes(matrix_.read_u8())


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit Vulkan test device required")
class GeneralMerkleVulkanTest(unittest.TestCase):
	def test_roots_exact_frames_and_fail_closed_verification(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for algorithm_id, algorithm in enumerate(ALGORITHMS):
			bytes_ = 64 if algorithm in ("sha512", "sha3_512") else 32
			for count in (1, 2, 5, 33):
				leaves = [bytes((i * 13 + j * 7) % 256 for j in range(bytes_)) for i in range(count)]
				if algorithm.startswith("poseidon2_"):
					prime = PROFILES[algorithm.removeprefix("poseidon2_")]["prime"]
					leaves = [struct.pack("<8I", *((i * 937 + j * 91) % prime for j in range(8))) for i in range(count)]
				levels = levels_of(leaves, algorithm)
				input_ = matrix(engine, leaves, algorithm)
				tree = oa.cryptography.merkle_build(input_, algorithm)
				self.assertEqual(read(tree.root, algorithm), levels[-1][0], (algorithm, count))
				indices = list(range(count)) + [count, 0xFFFFFFFF]
				selected = matrix(engine, leaves + [leaves[0], leaves[0]], algorithm)
				index_matrix = oa.Matrix.new_from_u32(engine, [len(indices)], indices)
				proofs = oa.cryptography.merkle_proofs(tree, index_matrix)
				expected = b""
				for index in indices:
					valid = index < count
					expected += struct.pack("<5I", 2, count, index, algorithm_id, int(valid))
					position = index
					for row in levels[:-1]:
						expected += row[min(position ^ 1, len(row) - 1)] if valid else bytes(bytes_)
						position //= 2
				self.assertEqual(bytes(proofs.read_u8()), expected, (algorithm, count))
				flags = oa.cryptography.merkle_verify(selected, proofs, tree.root, algorithm, count)
				self.assertEqual(flags.read_u32(), [1] * count + [0, 0])
				self.assertEqual(read(input_, algorithm), b"".join(leaves))
				if count == 5:
					wrong_count = oa.cryptography.merkle_verify(selected, proofs, tree.root, algorithm, 6)
					self.assertEqual(wrong_count.read_u32(), [0] * len(indices))
					frame_bytes = len(expected) // len(indices)
					for position in (0, 4, 8, 12, 16, 20):
						bad = bytearray(expected)
						bad[position] ^= 1
						bad_frames = oa.Matrix.new_from_u8(engine, [len(indices), frame_bytes], list(bad))
						result = oa.cryptography.merkle_verify(selected, bad_frames, tree.root, algorithm, count)
						self.assertEqual(result.read_u32()[0], 0, (algorithm, position))

	def test_hierarchical_subtrees_preserve_the_root(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for algorithm in ALGORITHMS:
			bytes_ = 64 if algorithm in ("sha512", "sha3_512") else 32
			leaves = [bytes((i + j) % 256 for j in range(bytes_)) for i in range(16)]
			if algorithm.startswith("poseidon2_"):
				leaves = [struct.pack("<8I", *(i * 8 + j for j in range(8))) for i in range(16)]
			input_ = matrix(engine, leaves, algorithm)
			for width in (1, 2, 4, 16):
				subtrees = oa.cryptography.merkle_subtree_roots(input_, algorithm, width)
				composed = oa.cryptography.merkle_build(subtrees, algorithm)
				self.assertEqual(read(composed.root, algorithm), levels_of(leaves, algorithm)[-1][0])
