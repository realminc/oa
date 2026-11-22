"""Independent mlkem ntt proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest, N
else:
    from .._pqc_harness import SlangComponentTest, N

class MlkemNttTests(SlangComponentTest):
    def test_mlkem_ntt_against_quadratic_remainders_and_ring_product(self) -> None:
        modulus = 3329
        rng = random.Random(20317)
        zero = [0] * N
        one = [1] + [0] * (N - 1)
        highest = [0] * (N - 1) + [1]
        pairs = [(zero, zero), (one, highest), (highest, highest),
                 ([modulus - 1] * N, [modulus - 1] * N)]
        pairs += [([rng.randrange(modulus) for _ in range(N)],
                   [rng.randrange(modulus) for _ in range(N)]) for _ in range(12)]
        records, expected = [], []
        for left, right in pairs:
            evaluations = []
            for pair in range(128):
                reversed_bits = int(f"{pair:07b}"[::-1], 2)
                gamma = pow(17, 2 * reversed_bits + 1, modulus)
                # Remainder modulo X^2-gamma, evaluating even/odd polynomials
                # independently. No butterfly, twiddle-table or inverse code.
                for parity in (0, 1):
                    value = 0
                    for coefficient in reversed(left[parity::2]):
                        value = (value * gamma + coefficient) % modulus
                    evaluations.append(value)
            product = [0] * N
            for i, a in enumerate(left):
                for j, b in enumerate(right):
                    product[(i + j) % N] += a * b * (1 if i + j < N else -1)
            records.extend(left + right)
            expected.extend(evaluations + left + [value % modulus for value in product])
        self.assertEqual(self.execute("mlkem_polynomial_components", records, len(pairs), 768),
                         expected)



if __name__ == "__main__":
    unittest.main()
