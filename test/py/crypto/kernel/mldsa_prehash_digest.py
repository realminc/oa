"""HashML-DSA digest/framing composition proofs, separate from Engine admission."""
import hashlib
import random
import struct
import unittest

if __package__ == "kernel":
    from _pqc_harness import MLDSA_PREHASH_HASHLIB, SlangComponentTest
else:
    from .._pqc_harness import MLDSA_PREHASH_HASHLIB, SlangComponentTest

POISON = 0xA5A5A5A5


def words(data):
    return list(struct.unpack(f"<{len(data) // 4}I", data))


class MldsaPrehashDigestTests(SlangComponentTest):
    def check_composition(self, algorithm, name, digest_length, message, context, alignment):
        tr = bytes(range(64))
        context_base = 24 + len(tr) + alignment
        base = context_base + len(context)
        record = bytearray(struct.pack("<6I", algorithm, base, len(message), 24,
                                       context_base, len(context)))
        record.extend(tr + b"\xa5" * alignment + context + message)
        record.extend(b"\xa5" * ((-len(record)) % 4))
        h = hashlib.new(name, message)
        digest = h.digest(digest_length) if name.startswith("shake_") else h.digest()
        oid = bytes((6, 9, 96, 134, 72, 1, 101, 3, 4, 2, algorithm))
        mu = hashlib.shake_256(tr + bytes((1, len(context))) + context + oid + digest).digest(64)
        self.assertEqual(self.execute("mldsa_prehash_digest_components", words(record), 1, 34),
                         [1, 1] + words(digest) + [POISON] * (16 - digest_length // 4) + words(mu))

    def test_all_hashes_padding_and_alignment(self):
        rng = random.Random(20261006)
        for algorithm, name, length in MLDSA_PREHASH_HASHLIB.values():
            for size in (0, 1, 55, 56, 63, 64, 111, 112, 127, 128,
                         135, 136, 143, 144, 167, 168, 169, 336, 1021):
                message = rng.randbytes(size)
                for alignment in range(4):
                    with self.subTest(algorithm=algorithm, size=size, alignment=alignment):
                        self.check_composition(algorithm, name, length, message, b"", alignment)

    def test_context_boundaries(self):
        for algorithm, name, length in MLDSA_PREHASH_HASHLIB.values():
            for size in (1, 127, 128, 255):
                with self.subTest(algorithm=algorithm, context_length=size):
                    self.check_composition(algorithm, name, length, b"message\x00\xff",
                                           bytes(i & 255 for i in range(size)), 3)

    def test_invalid_message_and_selector_leave_outputs_untouched(self):
        for algorithm in range(1, 13):
            for base, length in ((101, 0), (100, 1), (96, 5), (0xFFFFFFFF, 4), (96, 0xFFFFFFFF)):
                record = [algorithm, base, length, 24, 88, 0] + [POISON] * 19
                with self.subTest(algorithm=algorithm, base=base, length=length):
                    self.assertEqual(self.execute("mldsa_prehash_digest_components", record, 1, 34),
                                     [0, 0] + [POISON] * 32)
        for algorithm in (0, 13, 255, 0xFFFFFFFF):
            self.assertEqual(self.execute("mldsa_prehash_digest_components",
                                          [algorithm, 88, 0, 24, 88, 0] + [POISON] * 16, 1, 34),
                             [0, 0] + [POISON] * 32)

    def test_invalid_tr_or_context_preserves_mu(self):
        for algorithm, name, length in MLDSA_PREHASH_HASHLIB.values():
            h = hashlib.new(name, b"")
            digest = h.digest(length) if name.startswith("shake_") else h.digest()
            for tr_base, context_base, context_length in ((37, 88, 0), (0xFFFFFFFF, 88, 0),
                                                          (24, 101, 0), (24, 100, 1),
                                                          (24, 96, 5), (24, 88, 256),
                                                          (24, 88, 0xFFFFFFFF)):
                record = [algorithm, 100, 0, tr_base, context_base, context_length] + [POISON] * 19
                with self.subTest(algorithm=algorithm, tr_base=tr_base,
                                 context_base=context_base, context_length=context_length):
                    self.assertEqual(self.execute("mldsa_prehash_digest_components", record, 1, 34),
                                     [1, 0] + words(digest) + [POISON] * (16 - length // 4) + [POISON] * 16)


if __name__ == "__main__":
    unittest.main()
