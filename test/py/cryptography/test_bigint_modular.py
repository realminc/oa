"""Modular/Montgomery/Barrett contracts against Python integer identities."""
import json
import os
import unittest
from pathlib import Path

import oa

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/cryptography/bigint/modular.json"
UNARY = ("reduce_mod", "square_mod", "inverse_mod", "to_montgomery", "from_montgomery")
BINARY = ("add_mod", "subtract_mod", "multiply_mod", "pow_mod", "montgomery_multiply", "barrett_multiply")


def integer(words):
	return sum(word << (i * 32) for i, word in enumerate(words))


def oracle(name, a, b, m, width):
	if m == 0 or (name in ("to_montgomery", "from_montgomery", "montgomery_multiply") and not m & 1):
		return 0, 0
	if name == "inverse_mod":
		try:
			return pow(a, -1, m), 1
		except ValueError:
			return 0, 0
	if name == "reduce_mod": return a % m, 1
	if name == "add_mod": return (a + b) % m, 1
	if name == "subtract_mod": return (a - b) % m, 1
	if name in ("multiply_mod", "barrett_multiply"): return a * b % m, 1
	if name == "square_mod": return a * a % m, 1
	if name == "pow_mod": return pow(a, b, m), 1
	if name == "to_montgomery": return a * (1 << width) % m, 1
	if name == "from_montgomery": return a * pow(1 << width, -1, m) % m, 1
	if name == "montgomery_multiply": return a * b * pow(1 << width, -1, m) % m, 1
	raise AssertionError(name)


class BigIntModularFixtureTest(unittest.TestCase):
	def test_fixture_against_unbounded_integer_oracle(self):
		for case in json.loads(FIXTURE.read_text())["cases"]:
			words, rows = case["words"], case["rows"]
			for row in range(rows):
				start, end = row * words, (row + 1) * words
				a, b, m = [integer(case[key][start:end]) for key in ("left", "right", "modulus")]
				for name in UNARY + BINARY:
					value, valid = oracle(name, a, b, m, words * 32)
					self.assertEqual(integer(case["results"][name]["output"][start:end]), value)
					self.assertEqual(case["results"][name]["status"][row], valid)


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit Vulkan test device required")
class BigIntModularVulkanTest(unittest.TestCase):
	def test_selected_device_values_and_status(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for case in json.loads(FIXTURE.read_text())["cases"]:
			shape = [case["rows"], case["words"]]
			left = oa.Matrix.new_from_u32(engine, shape, case["left"])
			right = oa.Matrix.new_from_u32(engine, shape, case["right"])
			modulus = oa.Matrix.new_from_u32(engine, shape, case["modulus"])
			for name in UNARY + BINARY:
				op = getattr(oa.cryptography, f"bigint_{name}")
				output, status = op(left, modulus) if name in UNARY else op(left, right, modulus)
				self.assertEqual(output.read_u32(), case["results"][name]["output"], name)
				self.assertEqual(status.read_u32(), case["results"][name]["status"], name)


if __name__ == "__main__":
	unittest.main()
