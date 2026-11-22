"""Barrett reciprocal and multiplication against independent Python integers."""
import random
import unittest

if __package__ == "kernel":
	from _pqc_harness import SlangComponentTest
	from bigint import limbs
else:
	from .._pqc_harness import SlangComponentTest
	from .bigint import limbs


class BigIntBarrettTests(SlangComponentTest):
	def test_reciprocal_and_product_all_widths(self):
		rng = random.Random(40961442)
		cases = []
		for words in range(1, 129):
			width = words * 32
			mask = (1 << width) - 1
			cases.extend((words, a, b, m) for a, b, m in (
				(mask, mask, 0), (mask, mask, 1), (mask, mask, mask),
				(mask, mask, 1 << (width - 32)),
				(mask, mask, (1 << (width - 1)) + 1),
				(rng.getrandbits(width), rng.getrandbits(width), rng.getrandbits(width)),
				(rng.getrandbits(width), rng.getrandbits(width), 97),
			))
		inputs, expected = [], []
		for words, a, b, m in cases:
			inputs.extend([words, *limbs(a, 128), *limbs(b, 128), *limbs(m, 128)])
			k = (m.bit_length() + 31) // 32
			mu = (1 << (64 * k)) // m if m else 0
			expected.extend([*limbs(a * b % m if m else 0, 128), *limbs(mu, 130), k, int(m != 0)])
		actual = self.execute("bigint_barrett_components", inputs, len(cases), 260)
		for i, case in enumerate(cases):
			self.assertEqual(actual[i * 260:(i + 1) * 260], expected[i * 260:(i + 1) * 260], f"words={case[0]}, case={i}")


if __name__ == "__main__":
	unittest.main()
