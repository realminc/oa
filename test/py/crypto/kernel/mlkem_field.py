"""Independent mlkem field proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import itertools
import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest, MASK
else:
    from .._pqc_harness import SlangComponentTest, MASK

class MlkemFieldTests(SlangComponentTest):
    def test_mlkem_field(self) -> None:
        modulus = 3329
        rng = random.Random(2033329)
        boundaries = (0, 1, 2, 1664, 1665, 3327, 3328)
        pairs = list(itertools.product(boundaries, repeat=2))
        pairs += [(a, rng.randrange(modulus)) for a in range(modulus)]
        expected = []
        for a, b in pairs:
            centered = a if a <= modulus // 2 else a - modulus
            expected.extend(((a + b) % modulus, (a - b) % modulus,
                             a * b % modulus, -a % modulus, centered & MASK))
        self.assertEqual(self.execute("mlkem_field_components",
                                     list(itertools.chain.from_iterable(pairs)), len(pairs), 5),
                         expected)



if __name__ == "__main__":
    unittest.main()
