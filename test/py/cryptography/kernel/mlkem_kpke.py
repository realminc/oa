"""Independent mlkem kpke proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest
else:
    from .._pqc_harness import SlangComponentTest

class MlkemKpkeTests(SlangComponentTest):
    def test_mlkem_kpke(self):
        if __package__ == "kernel":
            from _mlkem_oracle import kpke
        else:
            from .._mlkem_oracle import kpke
        rng = random.Random(2031315)
        values, expected = [], []
        for k in (2, 3, 4):
            for case in range(3):
                d = bytes(32) if case == 0 else rng.randbytes(32)
                r = bytes([255] * 32) if case == 0 else rng.randbytes(32)
                message = bytes([0 if case == 0 else 255] * 32) if case < 2 else rng.randbytes(32)
                values += [k] + list(int.from_bytes(d[i:i + 4], "little") for i in range(0, 32, 4))
                values += list(int.from_bytes(r[i:i + 4], "little") for i in range(0, 32, 4)) + list(message)
                ek, dk, ciphertext, recovered = kpke(d, r, message, k)
                self.assertEqual(recovered, message)
                expected += list(ek) + [0xA5] * (1568 - len(ek))
                expected += list(dk) + [0xA5] * (1536 - len(dk))
                expected += list(ciphertext) + [0xA5] * (1568 - len(ciphertext))
                expected += list(recovered) + [1]
        self.assertEqual(self.execute("mlkem_kpke_components", values, 9, 4705), expected)


    def test_mlkem_kpke_arbitrary_decrypt(self):
        if __package__ == "kernel":
            from _mlkem_oracle import decrypt
        else:
            from .._mlkem_oracle import decrypt
        rng = random.Random(20315)
        values, expected = [], []
        for k in (2, 3, 4):
            size = 32 * ((11 * k + 5) if k == 4 else (10 * k + 4))
            for case in range(3):
                # Includes noncanonical 12-bit secret encodings and arbitrary
                # ciphertexts, so a mutually cancelling roundtrip cannot pass.
                dk = bytes(384 * k) if case == 0 else (bytes([255] * (384 * k)) if case == 1 else rng.randbytes(384 * k))
                ciphertext = bytes(size) if case == 0 else (bytes([255] * size) if case == 1 else rng.randbytes(size))
                values += [k] + list(dk) + [0xA5] * (1536 - len(dk))
                values += list(ciphertext) + [0xA5] * (1568 - len(ciphertext))
                expected += list(decrypt(dk, ciphertext, k))
        self.assertEqual(self.execute("mlkem_kpke_decrypt_components", values, 9, 32), expected)



if __name__ == "__main__":
    unittest.main()
