"""Montgomery representation/product against independent Python integers."""
import random
import unittest

if __package__ == "kernel":
	from _pqc_harness import SlangComponentTest
	from bigint import limbs
else:
	from .._pqc_harness import SlangComponentTest
	from .bigint import limbs


class BigIntMontgomeryTests(SlangComponentTest):
	def test_all_widths_and_carry_boundaries(self):
		rng = random.Random(409632)
		cases = []
		for words in range(1, 129):
			width = words * 32
			mask = (1 << width) - 1
			cases.extend((words, a, b, m) for a, b, m in (
				(mask, mask, 0), (mask, mask, 1), (mask, mask, mask - 1),
				(mask - 1, mask - 1, mask), (mask, mask, (1 << (width - 1)) + 1),
				(rng.getrandbits(width), rng.getrandbits(width), rng.getrandbits(width) | 1),
			))
		inputs, expected = [], []
		for words, a, b, m in cases:
			inputs.extend([words, *limbs(a, 128), *limbs(b, 128), *limbs(m, 128)])
			valid = m & 1
			if valid:
				radix = 1 << (words * 32)
				inverse = pow(radix, -1, m)
				values = (a * radix % m, a * inverse % m, a * b * inverse % m)
			else:
				values = (0, 0, 0)
			for value in values:
				expected.extend(limbs(value, 128))
			expected.extend([0] * 128)
			# Factor is used only for odd moduli; no invalid-modulus factor claim.
			factor = (-pow(m & 0xFFFFFFFF, -1, 1 << 32)) & 0xFFFFFFFF if valid else None
			expected.extend([valid, valid, valid, factor])
		actual = self.execute("bigint_montgomery_components", inputs, len(cases), 516)
		for i, case in enumerate(cases):
			base = i * 516
			self.assertEqual(actual[base:base + 515], expected[base:base + 515], f"words={case[0]}, case={i}")
			if case[3] & 1:
				self.assertEqual(actual[base + 515], expected[base + 515])


if __name__ == "__main__":
	unittest.main()
