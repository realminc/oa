"""Independent mlkem kpke proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest
    from _mlkem_oracle import kpke, decode, decompress, transform
else:
    from .._pqc_harness import SlangComponentTest
    from .._mlkem_oracle import kpke, decode, decompress, transform

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

    def test_buffered_kpke_all_parameters_and_poisoned_workspace(self):
        rng = random.Random(20313154096)
        values, expected = [], []
        for k in (2, 3, 4):
            for sample in range(2):
                d = bytes(32) if sample == 0 else rng.randbytes(32)
                r = bytes([255] * 32) if sample == 0 else rng.randbytes(32)
                message = bytes(32) if sample == 0 else rng.randbytes(32)
                values += [k, 0] + [int.from_bytes(d[i:i+4], "little") for i in range(0, 32, 4)]
                values += [int.from_bytes(r[i:i+4], "little") for i in range(0, 32, 4)] + list(message)
                ek, dk, ciphertext, recovered = kpke(d, r, message, k)
                cache = []
                du = 11 if k == 4 else 10
                for row in range(k):
                    packed = ciphertext[row * 32 * du:(row + 1) * 32 * du]
                    cache += transform([decompress(value, du) for value in decode(packed, du)])
                expected += list(ek) + [0xA5] * (1568 - len(ek))
                expected += list(dk) + [0xA5] * (1536 - len(dk))
                expected += list(ciphertext) + [0xA5] * (1568 - len(ciphertext))
                expected += list(recovered) + [1, 1, 1, 1, 0xA5A5A5A5]
                expected += cache + [0xA5A5A5A5] * (1024 - len(cache)) + [0xA5A5A5A5]
        self.assertEqual(self.execute("mlkem_kpke_workspace_components", values, 6, 5734), expected)

    def test_buffered_kpke_invalid_workspace_preserves_outputs(self):
        # Each one-row buffer has exactly 5734 words: an aligned base one word
        # beyond its final 4096-byte window must fail before any cache write.
        for k, base in ((1, 0), (5, 0), (2, 1), (3, 0xFFFFFFFF), (4, 4711 * 4)):
            with self.subTest(k=k, base=base):
                values = [k, base] + [0] * 48
                expected = [0xA5] * 4704 + [0, 0, 0, 0] + [0xA5A5A5A5] * 1026
                self.assertEqual(self.execute("mlkem_kpke_workspace_components", values, 1, 5734), expected)

    def test_buffered_kpke_exact_end_workspace(self):
        d, r, message = bytes(32), bytes(32), bytes(32)
        ek, dk, ciphertext, recovered = kpke(d, r, message, 4)
        cache = []
        for row in range(4):
            packed = ciphertext[row * 352:(row + 1) * 352]
            cache += transform([decompress(value, 11) for value in decode(packed, 11)])
        expected = list(ek) + list(dk) + list(ciphertext) + list(recovered)
        expected += [1, 1, 1, 1, 0xA5A5A5A5, 0xA5A5A5A5] + cache
        self.assertEqual(self.execute("mlkem_kpke_workspace_components", [4, 4710 * 4] + [0] * 48,
                                      1, 5734), expected)



if __name__ == "__main__":
    unittest.main()
