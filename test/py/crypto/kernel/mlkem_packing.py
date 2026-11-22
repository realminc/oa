"""Independent mlkem packing proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest
else:
    from .._pqc_harness import SlangComponentTest

class MlkemPackingTests(SlangComponentTest):
    def test_mlkem_packing(self) -> None:
        rng = random.Random(20356)
        values, expected = [], []
        for bits in range(1, 13):
            modulus = 3329 if bits == 12 else 1 << bits
            for case in range(6):
                poly = ([0] * 256 if case == 0 else
                        [modulus - 1] * 256 if case == 1 else
                        [rng.randrange(modulus) for _ in range(256)])
                packed = sum(value << (bits * i) for i, value in enumerate(poly))
                encoded = list(packed.to_bytes(32 * bits, "little"))
                raw = (encoded if case < 3 else
                       [rng.randrange(256) for _ in range(32 * bits)])
                if bits == 12 and case == 5:
                    # Every unrepresentable residue 3329..4095 is exercised
                    # below; this row includes the q and 4095 boundaries.
                    raw = list(sum((3329 if i % 2 == 0 else 4095) << (12 * i)
                                   for i in range(256)).to_bytes(384, "little"))
                integer = int.from_bytes(bytes(raw), "little")
                decoded = [(integer >> (bits * i)) & ((1 << bits) - 1)
                           for i in range(256)]
                canonical = int(bits < 12 or all(x < 3329 for x in decoded))
                values += [bits] + poly + raw + [0xCC] * (384 - len(raw))
                expected += (encoded + [0xA5] * (384 - len(encoded)) +
                             [x % modulus for x in decoded] + [canonical])
        # All 4096 possible twelve-bit words, including every rejected residue.
        for offset in range(0, 4096, 256):
            raw = list(sum((offset + i) << (12 * i) for i in range(256))
                       .to_bytes(384, "little"))
            values += [12] + [0] * 256 + raw
            expected += [0] * 384 + [(offset + i) % 3329 for i in range(256)]
            expected += [int(offset + 255 < 3329)]
        self.assertEqual(self.execute("mlkem_packing_components", values,
                                      len(values) // 641, 641), expected)



if __name__ == "__main__":
    unittest.main()
