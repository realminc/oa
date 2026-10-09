"""Independent mlkem sampling proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import hashlib
import itertools
import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest
else:
    from .._pqc_harness import SlangComponentTest

class MlkemSamplingTests(SlangComponentTest):
    def test_mlkem_sampling(self) -> None:
        rng = random.Random(20378)
        values, expected = [], []
        for case in range(32):
            seed = (bytes(32) if case == 0 else bytes([255]) * 32 if case == 1 else
                    rng.randbytes(32))
            first, second, nonce = [(case * factor) % 256 for factor in (13, 29, 31)]
            if case == 1:
                first = second = nonce = 255
            eta = 2 + case % 2
            stream = hashlib.shake_128(seed + bytes([first, second])).digest(4096)
            matrix = []
            for i in range(0, len(stream) - 2, 3):
                pair = int.from_bytes(stream[i:i + 3], "little")
                matrix.extend(x for x in (pair % 4096, pair // 4096) if x < 3329)
                if len(matrix) >= 256:
                    break
            self.assertGreaterEqual(len(matrix), 256)
            noise_bytes = hashlib.shake_256(seed + bytes([nonce])).digest(64 * eta)
            bitstream = int.from_bytes(noise_bytes, "little")
            mask = (1 << eta) - 1
            noise = []
            for i in range(256):
                chunk = bitstream >> (2 * eta * i)
                noise.append(((chunk & mask).bit_count() -
                              ((chunk >> eta) & mask).bit_count()) % 3329)
            values += [int.from_bytes(seed[i:i + 4], "little") for i in range(0, 32, 4)]
            values += [first, second, eta, nonce]
            expected += matrix[:256] + noise
        self.assertEqual(self.execute("mlkem_sampling_components", values, 32, 512), expected)


    def test_mlkem_rejection_boundaries(self) -> None:
        values, expected = [], []
        for first, second, count in itertools.product((0, 1, 3328, 3329, 4095),
                                                       (0, 1, 3328, 3329, 4095),
                                                       (0, 254, 255, 256)):
            values += [first, second, count]
            poly = [0xA5A5A5A5] * 256
            for value in (first, second):
                if value < 3329 and count < 256:
                    poly[count] = value
                    count += 1
            expected += poly + [count]
        self.assertEqual(self.execute("mlkem_rejection_components", values, 100, 257), expected)



if __name__ == "__main__":
    unittest.main()
