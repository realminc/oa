"""Production paired-word permutation versus OA's native 64-bit CPU oracle."""

import random
import unittest

import oa

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest
else:
    from .._pqc_harness import SlangComponentTest


class KeccakTests(SlangComponentTest):
    def test_permutation_all_input_bits_and_random_states(self) -> None:
        rng = random.Random(20261003)
        states = [bytes(200), bytes([255]) * 200, bytes(range(200))]
        states.extend((1 << bit).to_bytes(200, "little") for bit in range(1600))
        states.extend(rng.randbytes(200) for _ in range(64))
        inputs, expected = [], []
        for state in states:
            inputs.extend(int.from_bytes(state[i:i + 4], "little") for i in range(0, 200, 4))
            result = oa.cryptography.keccak_f1600(state)
            expected.extend(int.from_bytes(result[i:i + 4], "little") for i in range(0, 200, 4))
        self.assertEqual(self.execute("keccak_components", inputs, len(states), 50), expected)


if __name__ == "__main__":
    unittest.main()
