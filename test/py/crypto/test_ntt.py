"""Deferred independent DFT, inverse, convolution and representation proofs."""
import os
import random
import unittest

import oa


def pairs(values):
	return [word for value in values for word in (value & 0xFFFFFFFF, value >> 32)]


def integers(words):
	return [words[i] | words[i + 1] << 32 for i in range(0, len(words), 2)]


@unittest.skipUnless("OA_TEST_DEVICE_INDEX" in os.environ, "explicit Vulkan device required")
class NttVulkanTest(unittest.TestCase):
	def test_dft_inverse_and_montgomery_composition(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		rng = random.Random(527)
		for prime, generator in ((17, 3), (998244353, 3), (2013265921, 31), (2130706433, 3)):
			for length in (4, 8, 16):
				root = pow(generator, (prime - 1) // length, prime)
				values = [rng.randrange(prime) for _ in range(length * 3)]
				expected = [sum(values[row * length + i] * pow(root, i * j, prime) for i in range(length)) % prime for row in range(3) for j in range(length)]
				for representation in ("standard", "montgomery"):
					factor = pow(2, 64, prime) if representation == "montgomery" else 1
					config = oa.crypto.NttConfig(prime, root, length, representation)
					parameters = oa.crypto.ntt_prepare(engine, config)
					input_ = oa.Matrix.new_from_u32(engine, [3, length, 2], pairs([v * factor % prime for v in values]))
					transformed = oa.crypto.ntt_forward(input_, parameters)
					self.assertEqual(integers(transformed.read_u32()), [v * factor % prime for v in expected])
					restored = oa.crypto.ntt_inverse(transformed, parameters)
					self.assertEqual(restored.read_u32(), input_.read_u32())
					product = oa.crypto.ntt_pointwise_multiply(transformed, transformed, parameters)
					convolution = oa.crypto.ntt_inverse(product, parameters)
					oracle = [sum(values[row * length + i] * values[row * length + (j - i) % length] for i in range(length)) * factor % prime for row in range(3) for j in range(length)]
					self.assertEqual(integers(convolution.read_u32()), oracle)

	def test_large_staged_roundtrip(self):
		engine = oa.Engine(device=int(os.environ["OA_TEST_DEVICE_INDEX"]))
		prime, length = 2013265921, 16384
		config = oa.crypto.NttConfig(prime, pow(31, (prime - 1) // length, prime), length)
		parameters = oa.crypto.ntt_prepare(engine, config)
		values = [(i * i + 3 * i + 1) % prime for i in range(length)]
		input_ = oa.Matrix.new_from_u32(engine, [1, length, 2], pairs(values))
		transformed = oa.crypto.ntt_forward(input_, parameters)
		self.assertEqual(integers(oa.crypto.ntt_inverse(transformed, parameters).read_u32()), values)
