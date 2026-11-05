"""Deferred software-Vulkan qualification for mixed widths and prepared values."""
import os
import random
import unittest

import oa


def limbs(value, words):
	return [(value >> (32 * i)) & 0xFFFFFFFF for i in range(words)]


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit Vulkan device required")
class BigIntExtendedVulkanTest(unittest.TestCase):
	def test_mixed_division_and_scalar_operations(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		rng = random.Random(504)
		for words in (1, 2, 3, 8, 32, 128):
			width = 32 * words
			rows = 33 if words == 3 else 5
			values = [0, 1, (1 << (2 * width)) - 1] + [rng.getrandbits(2 * width) for _ in range(rows - 3)]
			divisor = (1 << (width - 1)) + 1
			left = oa.Matrix.new_from_u32(engine, [rows, 2 * words], sum((limbs(x, 2 * words) for x in values), []))
			right = oa.Matrix.new_from_u32(engine, [1, words], limbs(divisor, words))
			for name, quotient_words in (("wide", words * 2), ("narrow", words), ("mixed", words + 1)):
				function = getattr(oa.crypto, f"bigint_div_rem_{name}")
				q, r, status = function(left, right, quotient_words) if name == "mixed" else function(left, right)
				division = [divmod(v, divisor) for v in values]
				valid = [1 if a < 1 << (32 * quotient_words) else 2 for a, _ in division]
				self.assertEqual(status.read_u32(), valid)
				self.assertEqual(q.read_u32(), sum((limbs(a if flag == 1 else 0, quotient_words) for (a, _), flag in zip(division, valid)), []))
				self.assertEqual(r.read_u32(), sum((limbs(b if flag == 1 else 0, words) for (_, b), flag in zip(division, valid)), []))
			zero = oa.Matrix.new_from_u32(engine, [1, words], [0] * words)
			q, r, status = oa.crypto.bigint_div_rem_wide(left, zero)
			self.assertEqual(status.read_u32(), [0] * rows)
			self.assertFalse(any(q.read_u32() + r.read_u32()))
			values = [v & ((1 << width) - 1) for v in values]
			left = oa.Matrix.new_from_u32(engine, [rows, words], sum((limbs(v, words) for v in values), []))
			for scalar in (0, 1, 0xFFFFFFFF):
				for name, operation in (("add_scalar", lambda x: x + scalar), ("subtract_scalar", lambda x: x - scalar), ("multiply_scalar", lambda x: x * scalar)):
					self.assertEqual(getattr(oa.crypto, f"bigint_{name}")(left, scalar).read_u32(), sum((limbs(operation(v), words) for v in values), []))
				q, r, status = oa.crypto.bigint_div_rem_scalar(left, scalar)
				self.assertEqual(status.read_u32(), [int(scalar != 0)] * rows)
				self.assertEqual(q.read_u32(), sum((limbs(v // scalar if scalar else 0, words) for v in values), []))
				self.assertEqual(r.read_u32(), [v % scalar if scalar else 0 for v in values])

	def test_conversions_barrett_and_wide_redc(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		for words in (1, 2, 3, 16, 128):
			width = 32 * words
			modulus = (1 << (width - 1)) + 1
			m = oa.Matrix.new_from_u32(engine, [1, words], limbs(modulus, words))
			parameters = oa.crypto.bigint_setup_barrett(m)
			self.assertEqual(parameters.status.read_u32(), [1])
			values = [0, 1, (1 << (2 * width)) - 1]
			left = oa.Matrix.new_from_u32(engine, [3, words * 2], sum((limbs(v, words * 2) for v in values), []))
			for function, expected in ((oa.crypto.bigint_reduce_barrett, [v % modulus for v in values]), (oa.crypto.bigint_reduce_montgomery, [v * pow(1 << width, -1, modulus) % modulus for v in values])):
				output, status = function(left, parameters if function is oa.crypto.bigint_reduce_barrett else m)
				self.assertEqual(status.read_u32(), [1] * 3)
				self.assertEqual(output.read_u32(), sum((limbs(v, words) for v in expected), []))
			u32 = oa.Matrix.new_from_u32(engine, [3, 1], [0, 1, 0xFFFFFFFF])
			expanded = oa.crypto.bigint_from_u32(u32, words)
			output, status = oa.crypto.bigint_to_u32(expanded)
			self.assertEqual(output.read_u32(), [0, 1, 0xFFFFFFFF]); self.assertEqual(status.read_u32(), [1] * 3)
			output, status = oa.crypto.bigint_to_u64(expanded)
			self.assertEqual(output.read_u32(), [0, 0, 1, 0, 0xFFFFFFFF, 0])
			if words >= 2:
				wide = oa.Matrix.new_from_u32(engine, [1, 2], [0xFFFFFFFF, 0xFFFFFFFF])
				expanded = oa.crypto.bigint_from_u64(wide, words)
				output, status = oa.crypto.bigint_to_u32(expanded)
				self.assertEqual(output.read_u32(), [0]); self.assertEqual(status.read_u32(), [0])
