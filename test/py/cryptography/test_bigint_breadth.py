"""Independent integer identities and selected-device BigInt breadth APIs."""
import json
import os
import unittest
from pathlib import Path

import oa

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/cryptography/bigint/breadth.json"


def integer(words):
	return sum(word << (32 * i) for i, word in enumerate(words))


class BigIntBreadthFixtureTest(unittest.TestCase):
	def test_fixture_matches_integer_identities(self):
		for case in json.loads(FIXTURE.read_text())["cases"]:
			words, rows = case["words"], case["rows"]
			width = words * 32
			for row in range(rows):
				a = integer(case["left"][row * words:(row + 1) * words])
				b = integer(case["right"][row * words:(row + 1) * words])
				for name, expected, stride in (("multiply_high", (a * b) >> width, words),
					("square_wide", a * a, words * 2), ("bit_not", a ^ ((1 << width) - 1), words)):
					self.assertEqual(integer(case["results"][name][row * stride:(row + 1) * stride]), expected, (words, row, name))
				self.assertEqual(case["results"]["leading_zeros"][row], width - a.bit_length())


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit Vulkan test device required")
class BigIntBreadthVulkanTest(unittest.TestCase):
	def test_selected_device_breadth(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for case in json.loads(FIXTURE.read_text())["cases"]:
			left = oa.Matrix.new_from_u32(engine, [case["rows"], case["words"]], case["left"])
			right = oa.Matrix.new_from_u32(engine, [case["rows"], case["words"]], case["right"])
			self.assertEqual(oa.cryptography.bigint_multiply_high(left, right).read_u32(), case["results"]["multiply_high"])
			for name in ("square_wide", "bit_not", "leading_zeros"):
				self.assertEqual(getattr(oa.cryptography, f"bigint_{name}")(left).read_u32(), case["results"][name], name)


if __name__ == "__main__":
	unittest.main()
