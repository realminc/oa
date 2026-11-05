"""Pinned external permutation vectors and OA field-sponge/device contracts."""
import json
import os
import unittest
from pathlib import Path

import oa

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "tool/gen/fn/schema/crypto/crypto_hash.json"
FIXTURE = ROOT / "test/fixtures/crypto/poseidon2/reference.json"


def permute(values, profile):
	"""Independent ordinary-integer reference; GPU uses Montgomery arithmetic."""
	p, width = profile["prime"], profile["width"]
	state = [x % p for x in values]

	def external():
		for i in range(0, width, 4):
			x = state[i:i + 4]
			matrix = ((2, 3, 1, 1), (1, 2, 3, 1), (1, 1, 2, 3), (3, 1, 1, 2))
			state[i:i + 4] = [sum(a * b for a, b in zip(row, x)) % p for row in matrix]
		sums = [sum(state[i::4]) % p for i in range(4)]
		state[:] = [(x + sums[i % 4]) % p for i, x in enumerate(state)]

	external()
	for constants in profile["initial_constants"]:
		state[:] = [pow((x + c) % p, profile["sbox_degree"], p) for x, c in zip(state, constants)]
		external()
	for constant in profile["internal_constants"]:
		state[0] = pow((state[0] + constant) % p, profile["sbox_degree"], p)
		total = sum(state) % p
		state[:] = [(total + d * x) % p for d, x in zip(profile["diagonal"], state)]
	for constants in profile["final_constants"]:
		state[:] = [pow((x + c) % p, profile["sbox_degree"], p) for x, c in zip(state, constants)]
		external()
	return state


def sponge(values, profile, output_length):
	p, width = profile["prime"], profile["width"]
	rate = width - profile["capacity"]
	state = [0] * width
	state[rate], state[rate + 1] = 0x4F415032, len(values)
	cursor = 0
	for value in values:
		state[cursor] = (state[cursor] + value) % p
		cursor += 1
		if cursor == rate:
			state, cursor = permute(state, profile), 0
	state[cursor] = (state[cursor] + 1) % p
	state, cursor, result = permute(state, profile), 0, []
	for _ in range(output_length):
		if cursor == rate:
			state, cursor = permute(state, profile), 0
		result.append(state[cursor])
		cursor += 1
	return result


class Poseidon2ReferenceTest(unittest.TestCase):
	def test_pinned_permutation_vectors(self):
		profiles = {p["name"]: p for p in json.loads(SCHEMA.read_text())["poseidon2_profiles"]}
		for case in json.loads(FIXTURE.read_text())["cases"]:
			self.assertEqual(permute(case["input"], profiles[case["profile"]]), case["output"], case["profile"])


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit Vulkan device required")
class Poseidon2VulkanTest(unittest.TestCase):
	def test_all_profiles_permutation_and_framed_sponge(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		profiles = {p["name"]: p for p in json.loads(SCHEMA.read_text())["poseidon2_profiles"]}
		for case in json.loads(FIXTURE.read_text())["cases"]:
			name, values = case["profile"], case["input"]
			input_ = oa.Matrix.new_from_u32(engine, [1, len(values)], values)
			self.assertEqual(getattr(oa.crypto, f"hash_poseidon2_{name}_permute")(input_).read_u32(), case["output"])
		for name, profile in profiles.items():
			function = getattr(oa.crypto, f"hash_poseidon2_{name}")
			rate = profile["width"] - profile["capacity"]
			for length in (0, 1, rate - 1, rate, rate + 1, 2 * rate):
				rows = 33 if length == rate + 1 else 1
				values = [(i * 2654435761) & 0xFFFFFFFF for i in range(length * rows)]
				input_ = oa.Matrix.new_from_u32(engine, [rows, length], values)
				for output_length in (8, rate + 3):
					expected = sum((sponge(values[r * length:(r + 1) * length], profile, output_length) for r in range(rows)), [])
					self.assertEqual(function(input_, output_length).read_u32(), expected, (name, length, output_length))
				self.assertEqual(input_.read_u32(), values)
