"""Independent mlkem compression proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest
else:
    from .._pqc_harness import SlangComponentTest

class MlkemCompressionTests(SlangComponentTest):
    def test_mlkem_compression_exhaustive(self) -> None:
        # Independent nearest-rational rounding by remainder comparison.
        def round_ratio(numerator: int, denominator: int) -> int:
            quotient, remainder = divmod(numerator, denominator)
            return quotient + int(2 * remainder >= denominator)

        modulus = 3329
        records, expected = [], []
        for bits in range(1, 12):
            scale = 1 << bits
            # All q input residues and all 2^d encodings for every admitted d.
            for coefficient in range(modulus):
                encoded = coefficient % scale
                compressed = round_ratio(coefficient * scale, modulus) % scale
                decompressed = round_ratio(encoded * modulus, scale)
                restored = round_ratio(compressed * modulus, scale)
                records.extend((coefficient, bits, encoded))
                expected.extend((compressed, decompressed, restored, encoded))
                distance = min((coefficient - restored) % modulus,
                               (restored - coefficient) % modulus)
                self.assertLessEqual(distance, round_ratio(modulus, 2 * scale))
        self.assertEqual(self.execute("mlkem_compression_components", records,
                                     len(records) // 3, 4), expected)



if __name__ == "__main__":
    unittest.main()
