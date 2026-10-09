"""Independent mldsa ntt proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest, Q, N, evaluate_ntt, negacyclic_product
else:
    from .._pqc_harness import SlangComponentTest, Q, N, evaluate_ntt, negacyclic_product

class MldsaNttTests(SlangComponentTest):
    def test_ntt_against_polynomial_evaluation_and_ring_product(self) -> None:
        random_source = random.Random(1753)
        polynomials = [[0] * N, [Q - 1] * N, [1] + [0] * (N - 1),
                       [0] * (N - 1) + [1], [i % 2 * (Q - 1) for i in range(N)]]
        polynomials += [[random_source.randrange(Q) for _ in range(N)] for _ in range(8)]
        for i, a in enumerate(polynomials):
            b = polynomials[(i + 1) % len(polynomials)]
            with self.subTest(polynomial=i):
                actual = self.execute("polynomial_components", a + b, 1, 1024)
                transformed = evaluate_ntt(a)
                self.assertEqual(actual[:N], transformed)
                self.assertEqual(actual[N:2 * N], a)
                self.assertEqual(actual[2 * N:3 * N], negacyclic_product(a, b))
                self.assertEqual(actual[3 * N:], [(8192 * value) % Q for value in transformed])



if __name__ == "__main__":
    unittest.main()
