"""Independent mldsa sampling proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import hashlib
import json
import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest, ROOT, Q
else:
    from .._pqc_harness import SlangComponentTest, ROOT, Q

class MldsaSamplingTests(SlangComponentTest):
    def test_sampling_against_hashlib_streams(self) -> None:
        rng = random.Random(2042932)
        fixture = json.loads((ROOT / "test/fixtures/cryptography/acvp_mldsa65_sigver.json").read_text())
        seeds = [(bytes.fromhex(case["signature"])[:48], bytes.fromhex(case["pk"])[:32])
                 for case in fixture["tests"]]
        seeds += [(bytes(48), bytes(32)), (bytes([255]) * 48, bytes([255]) * 32)]
        seeds += [(rng.randbytes(48), rng.randbytes(32)) for _ in range(32)]
        for case, (challenge, rho) in enumerate(seeds):
            # Independent byte streams from Python's OpenSSL/hashlib SHAKE.
            stream = hashlib.shake_256(challenge).digest(4096)
            signs = int.from_bytes(stream[:8], "little")
            c, pos = [0] * 256, 8
            for i in range(207, 256):
                while stream[pos] > i:
                    pos += 1
                j = stream[pos]
                pos += 1
                c[i] = c[j]
                c[j] = Q - 1 if (signs >> (i - 207)) & 1 else 1
            row, column = case % 6, case % 5
            stream = hashlib.shake_128(rho + bytes([column, row])).digest(4096)
            a = []
            for pos in range(0, len(stream) - 2, 3):
                coefficient = int.from_bytes(stream[pos:pos + 3], "little") & 0x7FFFFF
                if coefficient < Q:
                    a.append(coefficient)
                if len(a) == 256:
                    break
            self.assertEqual(len(a), 256)
            values = [int.from_bytes(seed[i:i + 4], "little")
                      for seed in (challenge, rho) for i in range(0, len(seed), 4)] + [row, column]
            actual = self.execute("sampling_components", values, 1, 512)
            with self.subTest(seed=case):
                self.assertEqual(actual[:256], c)
                self.assertEqual(actual[256:], a)
                self.assertEqual(sum(v != 0 for v in actual[:256]), 49)



if __name__ == "__main__":
    unittest.main()
