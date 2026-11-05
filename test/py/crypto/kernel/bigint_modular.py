"""Independent modular arithmetic/inverse/power proofs for production Slang."""
import random
import unittest

if __package__ == "kernel":
	from _pqc_harness import SlangComponentTest
	from bigint import limbs
else:
	from .._pqc_harness import SlangComponentTest
	from .bigint import limbs


class BigIntModularTests(SlangComponentTest):
	def test_modular_arithmetic_all_widths(self):
		rng = random.Random(4096203204)
		cases = []
		for words in range(1, 129):
			width = words * 32
			mask = (1 << width) - 1
			cases.extend((words, a, b, modulus) for a, b, modulus in (
				(mask, mask, 0), (mask, mask, 1), (mask, mask, mask),
				(mask, mask, (1 << (width - 1)) + 1),
				(rng.getrandbits(width), rng.getrandbits(width), rng.getrandbits(width)),
				(rng.getrandbits(width), rng.getrandbits(width), 2),
			))
		inputs, expected = [], []
		for words, a, b, modulus in cases:
			inputs.extend([words, *limbs(a, 128), *limbs(b, 128), *limbs(modulus, 128)])
			for value in (a, a + b, a - b, a * b):
				expected.extend(limbs(value % modulus if modulus else 0, 128))
			expected.extend([int(modulus != 0)] * 4)
		actual = self.execute("bigint_modular_components", inputs, len(cases), 516)
		for i, case in enumerate(cases):
			self.assertEqual(actual[i * 516:(i + 1) * 516], expected[i * 516:(i + 1) * 516], f"words={case[0]}, case={i}")

	def test_inverse_and_power_odd_even_composite_moduli(self):
		rng = random.Random(4096)
		cases = []
		for words in (1, 2, 3, 4, 8, 16):
			width = words * 32
			mask = (1 << width) - 1
			cases.extend((words, a, e, m) for a, e, m in (
				(0, 0, 0), (0, 0, 1), (0, 0, mask), (2, 17, mask - 1),
				(3, 2, 15), (mask, 0, 2), (mask - 1, 1, mask),
			))
			for _ in range(12):
				cases.append((words, rng.getrandbits(width), rng.randrange(1 << 17), rng.getrandbits(width)))
		for words in (32, 64, 127, 128):
			mask = (1 << (words * 32)) - 1
			cases.extend((words, a, e, m) for a, e, m in ((0, 0, mask), (1, 1, mask), (2, 2, mask), (3, 3, mask - 1)))
		# Exponent limbs above the least-significant word must be consumed.
		cases.append((2, 7, (1 << 33) + 1, 97))
		cases.append((8, 7, (1 << 255) + 1, 97))
		cases.append((128, 2, 65537, (1 << 4096) - 1))
		inputs, expected = [], []
		for words, a, e, m in cases:
			inputs.extend([words, *limbs(a, 128), *limbs(e, 128), *limbs(m, 128)])
			try:
				inverse = pow(a, -1, m) if m else None
			except ValueError:
				inverse = None
			expected.extend([*limbs(inverse or 0, 128), *limbs(pow(a, e, m) if m else 0, 128),
				int(inverse is not None), int(m != 0)])
		actual = self.execute("bigint_inverse_components", inputs, len(cases), 258)
		for i, case in enumerate(cases):
			self.assertEqual(actual[i * 258:(i + 1) * 258], expected[i * 258:(i + 1) * 258], f"words={case[0]}, case={i}")


if __name__ == "__main__":
	unittest.main()
