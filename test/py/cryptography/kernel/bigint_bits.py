"""Leading-zero counts over every admitted width, using integer bit lengths."""
import random
import unittest

if __package__ == "kernel":
	from _pqc_harness import SlangComponentTest
else:
	from .._pqc_harness import SlangComponentTest


class BigIntBitTests(SlangComponentTest):
	def test_leading_zeros_for_all_widths_and_word_bit_positions(self):
		rng = random.Random(2026101401)
		inputs, expected = [], []
		for words in range(1, 129):
			width = words * 32
			values = [0, 1, 0x80000000, (1 << width) - 1, rng.getrandbits(width)]
			values += [1 << (width - 32 + bit) for bit in range(32)]
			for value in values:
				inputs.extend([words, *[(value >> (32 * i)) & 0xffffffff for i in range(128)]])
				expected.append(width - value.bit_length())
		self.assertEqual(self.execute("bigint_bit_components", inputs, len(expected), 1), expected)


if __name__ == "__main__":
	unittest.main()
