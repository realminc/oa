"""Fixed-width device arithmetic against independent Python integers."""
import json
import os
import unittest
from pathlib import Path

import oa

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/cryptography/bigint/arithmetic.json"


def integer(words):
	return sum(word << (32 * i) for i, word in enumerate(words))


class BigIntFixtureTest(unittest.TestCase):
	def test_fixture_matches_integer_identities(self):
		for case in json.loads(FIXTURE.read_text())["cases"]:
			width, rows = case["words"], case["rows"]
			mask = (1 << (width * 32)) - 1
			for row in range(rows):
				a = integer(case["left"][row * width:(row + 1) * width])
				b = integer(case["right"][row * width:(row + 1) * width])
				q, r = divmod(a, b) if b else (0, a)
				for name, expected in (("add", (a + b) & mask), ("subtract", (a - b) & mask),
					("multiply_low", (a * b) & mask), ("multiply_wide", a * b),
					("bit_and", a & b), ("bit_or", a | b), ("bit_xor", a ^ b),
					("quotient", q), ("remainder", r)):
					words = 2 * width if name == "multiply_wide" else width
					actual = integer(case["results"][name][row * words:(row + 1) * words])
					self.assertEqual(actual, expected, (width, row, name))
				self.assertEqual(case["results"]["status"][row], int(b != 0))
				self.assertEqual(case["results"]["compare"][row], (a > b) - (a < b) + 1)
				for shift in case["shifts"]:
					amount = shift["shift"]
					for name, expected in (("left", (a << amount) & mask if amount < width * 32 else 0),
						("right", a >> amount if amount < width * 32 else 0)):
						self.assertEqual(integer(shift[name][row * width:(row + 1) * width]), expected)


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit Vulkan test device required")
class BigIntVulkanTest(unittest.TestCase):
	def test_selected_device_arithmetic(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for case in json.loads(FIXTURE.read_text())["cases"]:
			shape = [case["rows"], case["words"]]
			left = oa.Matrix.new_from_u32(engine, shape, case["left"])
			right = oa.Matrix.new_from_u32(engine, shape, case["right"])
			for name in ("add", "subtract", "multiply_low", "multiply_wide", "bit_and", "bit_or", "bit_xor", "compare"):
				op = getattr(oa.cryptography, f"bigint_{name}")
				self.assertEqual(op(left, right).read_u32(), case["results"][name], name)
			for name, output in zip(("quotient", "remainder", "status"), oa.cryptography.bigint_div_rem(left, right)):
				self.assertEqual(output.read_u32(), case["results"][name], name)
			for shift in case["shifts"]:
				self.assertEqual(oa.cryptography.bigint_shift_left(left, shift["shift"]).read_u32(), shift["left"])
				self.assertEqual(oa.cryptography.bigint_shift_right(left, shift["shift"]).read_u32(), shift["right"])


if __name__ == "__main__":
	unittest.main()
