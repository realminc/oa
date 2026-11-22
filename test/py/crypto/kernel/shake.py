"""Independent shake proofs for production Slang.

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

class ShakeTests(SlangComponentTest):
    def test_streaming_shake_against_hashlib(self) -> None:
        rng = random.Random(202204)
        lengths = (0, 1, 63, 64, 66, 135, 136, 137, 271, 272, 511, 1021)
        records, expected = bytearray(), []
        for first_length, second_length in itertools.product(lengths, repeat=2):
            first = rng.randbytes(first_length)
            second = rng.randbytes(second_length)
            record = bytearray([0xA5] * 2056)
            record[:4] = first_length.to_bytes(4, "little")
            record[4:8] = second_length.to_bytes(4, "little")
            record[9:9 + first_length] = first
            record[1035:1035 + second_length] = second
            records.extend(record)
            digest = hashlib.shake_256(first + second).digest(64)
            expected.extend(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 64, 4))
        words = [int.from_bytes(records[i:i + 4], "little") for i in range(0, len(records), 4)]
        self.assertEqual(self.execute("streaming_components", words, len(lengths)**2, 16), expected)



if __name__ == "__main__":
    unittest.main()
