"""Independent Python-integer proofs for production unsigned limb arithmetic."""

import random
import unittest

if __package__ == "kernel":
	from _pqc_harness import SlangComponentTest
else:
	from .._pqc_harness import SlangComponentTest


def limbs(value: int, count: int) -> list[int]:
	return [(value >> (32 * i)) & 0xFFFFFFFF for i in range(count)]


class BigIntTests(SlangComponentTest):
	def test_all_widths_against_python_integers(self) -> None:
		rng = random.Random(2032044096)
		cases = []
		for words in range(1, 129):
			width = words * 32
			mask = (1 << width) - 1
			cases.extend((words, a, b) for a, b in (
				(0, 0), (mask, mask), (mask, 1), (1, mask),
				(1 << (width - 1), (1 << (width - 1)) + 1),
				(rng.getrandbits(width), rng.getrandbits(width)),
			))
		for words in (1, 2, 3, 4, 7, 8, 9, 16, 31, 32, 33, 64, 127, 128):
			width = words * 32
			for _ in range(12):
				cases.append((words, rng.getrandbits(width), rng.getrandbits(rng.randrange(1, width + 1))))
		inputs, expected = [], []
		for words, a, b in cases:
			width = words * 32
			mask = (1 << width) - 1
			quotient, remainder = divmod(a, b) if b else (0, a)
			inputs.extend([words, *limbs(a, 128), *limbs(b, 128)])
			expected.extend([
				*limbs((a + b) & mask, 128), *limbs((a - b) & mask, 128),
				*limbs(a * b, 256), *limbs(quotient, 128), *limbs(remainder, 128),
				int(a + b > mask), int(a < b), (a > b) - (a < b) + 1, int(b != 0),
			])
		actual = self.execute("bigint_arithmetic_components", inputs, len(cases), 772)
		for index, case in enumerate(cases):
			self.assertEqual(actual[index * 772:(index + 1) * 772],
				expected[index * 772:(index + 1) * 772], f"case {index}, words={case[0]}")


if __name__ == "__main__":
	unittest.main()
