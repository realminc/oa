"""Independent mldsa field proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import itertools
import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest, Q, MASK, decompose
else:
    from .._pqc_harness import SlangComponentTest, Q, MASK, decompose

class MldsaFieldTests(SlangComponentTest):
    def test_wide_multiply(self) -> None:
        random_source = random.Random(204)
        boundaries = [0, 1, 65535, 65536, 0x7FFFFFFF, 0x80000000, MASK]
        pairs = list(itertools.product(boundaries, repeat=2))
        pairs += [(random_source.getrandbits(32), random_source.getrandbits(32))
                  for _ in range(4096)]
        expected = [word for a, b in pairs for word in ((a * b) & MASK, (a * b) >> 32)]
        self.assertEqual(self.execute("wide_components", [v for pair in pairs for v in pair],
                                      len(pairs), 2), expected)


    def test_field_and_rounding(self) -> None:
        random_source = random.Random(203204)
        cases = []
        for gamma2 in ((Q - 1) // 88, (Q - 1) // 32):
            values = {0, 1, 4095, 4096, 4097, Q // 2, Q - 2, Q - 1}
            for high in range((Q - 1) // (2 * gamma2) + 1):
                for offset in (-gamma2 - 1, -gamma2, -gamma2 + 1, -1, 0, 1,
                               gamma2 - 1, gamma2, gamma2 + 1):
                    values.add((high * 2 * gamma2 + offset) % Q)
            for a in sorted(values):
                for b in (0, 1, Q - 1, random_source.randrange(Q)):
                    for hint in (0, 1):
                        cases.append((a, b, gamma2, hint))
            cases += [(random_source.randrange(Q), random_source.randrange(Q), gamma2,
                       random_source.randrange(2)) for _ in range(4096)]
        expected = []
        for a, b, gamma2, hint in cases:
            high, low = decompose(a, gamma2)
            round_high = (a + 4095) // 8192
            expected += [(a * b) % Q, a, (a * (1 << 32)) % Q, (a + b) % Q,
                         (a - b) % Q, (-a) % Q, (a if a <= Q // 2 else a - Q) & MASK,
                         round_high, (a - round_high * 8192) & MASK, high, low & MASK,
                         int(high != decompose((a + b) % Q, gamma2)[0]),
                         (high + (1 if low > 0 else -1)) % ((Q - 1) // (2 * gamma2))
                         if hint else high, b]
        actual = self.execute("field_components", [v for case in cases for v in case], len(cases), 14)
        for i, case in enumerate(cases):
            self.assertEqual(actual[i * 14:(i + 1) * 14], expected[i * 14:(i + 1) * 14],
                             f"field case {case}")



if __name__ == "__main__":
    unittest.main()
