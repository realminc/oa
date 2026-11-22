"""FIPS 202 byte-oriented SHA-3 proofs for the actual production Slang helpers.

Host-target execution and SPIR-V validation do not qualify Vulkan execution.
"""

import hashlib
import random
import struct
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest
else:
    from .._pqc_harness import SlangComponentTest


POISON = 0xA5A5A5A5
RATES = {224: 144, 256: 136, 384: 104, 512: 72}


def words(data: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(data) // 4}I", data))


class Sha3Tests(SlangComponentTest):
    def assert_digest(self, bits: int, message: bytes, alignment: int, exact_end: bool = False) -> None:
        base = 12 + alignment
        record = bytearray(struct.pack("<III", bits, base, len(message)))
        record.extend(b"\xa5" * alignment)
        record.extend(message)
        record.extend(b"\xa5" * ((-len(record)) % 4))
        if not exact_end:
            record.extend(b"\xa5" * 16)
        result = self.execute("sha3_components", words(record), 1, 17)
        digest = hashlib.new(f"sha3_{bits}", message).digest()
        self.assertEqual(result, [1] + words(digest) + [POISON] * (16 - bits // 32))

    def test_empty_and_abc_known_answers(self) -> None:
        # Independent published byte-oriented identities, not generated constants.
        expected_empty = {
            224: "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7",
            256: "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",
            384: "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2ac3713831264adb47fb6bd1e058d5f004",
            512: "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26",
        }
        for bits, expected in expected_empty.items():
            with self.subTest(bits=bits):
                self.assertEqual(hashlib.new(f"sha3_{bits}", b"").hexdigest(), expected)
                self.assert_digest(bits, b"", 0, exact_end=True)
                self.assert_digest(bits, b"abc", 1, exact_end=True)

    def test_rate_boundaries_and_unaligned_multiblock_messages(self) -> None:
        rng = random.Random(20201004)
        for bits, rate in RATES.items():
            lengths = (1, 3, 4, 7, 8, rate - 1, rate, rate + 1,
                       2 * rate - 1, 2 * rate, 2 * rate + 1, 3 * rate, 1021)
            for length in lengths:
                message = rng.randbytes(length)
                for alignment in range(4):
                    with self.subTest(bits=bits, length=length, alignment=alignment):
                        self.assert_digest(bits, message, alignment)

    def test_exact_end_ranges(self) -> None:
        for bits, rate in RATES.items():
            for alignment in range(4):
                for length in (rate - alignment, 2 * rate - alignment, 0):
                    with self.subTest(bits=bits, alignment=alignment, length=length):
                        self.assert_digest(bits, b"\x93" * length, alignment, exact_end=True)

    def test_invalid_ranges_preserve_digest(self) -> None:
        for bits in RATES:
            for base, length in ((17, 0), (16, 1), (12, 5), (0xFFFFFFFF, 4), (12, 0xFFFFFFFF)):
                with self.subTest(bits=bits, base=base, length=length):
                    result = self.execute("sha3_components", [bits, base, length, POISON], 1, 17)
                    self.assertEqual(result, [0] + [POISON] * 16)

    def test_invalid_rates_and_ranges_preserve_state(self) -> None:
        cases = [(rate, 12, 1) for rate in (0, 1, 8, 64, 136 + 1, 200, 0xFFFFFFFF)]
        cases.extend((rate, base, length) for rate in RATES.values()
                     for base, length in ((17, 0), (16, 1), (12, 5), (0xFFFFFFFF, 1), (12, 0xFFFFFFFF)))
        for rate, base, length in cases:
            with self.subTest(rate=rate, base=base, length=length):
                result = self.execute("sha3_range_components", [rate, base, length, POISON], 1, 51)
                self.assertEqual(result, [0] + [POISON] * 50)


if __name__ == "__main__":
    unittest.main()
