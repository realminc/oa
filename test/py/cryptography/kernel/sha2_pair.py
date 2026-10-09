"""SHA-384/SHA-512/SHA-512-t production-helper proofs; no Vulkan execution claim."""

import hashlib
import random
import struct
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest
else:
    from .._pqc_harness import SlangComponentTest


POISON = 0xA5A5A5A5


def words(data: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(data) // 4}I", data))


class Sha2PairTests(SlangComponentTest):
    def assert_digest(self, bits: int, message: bytes, alignment: int, exact_end: bool = False) -> None:
        record = bytearray(struct.pack("<III", bits, 12 + alignment, len(message)))
        record.extend(b"\xa5" * alignment)
        record.extend(message)
        record.extend(b"\xa5" * ((-len(record)) % 4))
        if not exact_end:
            record.extend(b"\xa5" * 16)
        expected = hashlib.new({224: "sha512_224", 256: "sha512_256", 384: "sha384", 512: "sha512"}[bits], message).digest()
        self.assertEqual(self.execute("sha2_u32_pair_components", words(record), 1, 17),
                         [1] + words(expected) + [POISON] * (16 - bits // 32))

    def test_known_answers(self) -> None:
        published = {
            224: ("6ed0dd02806fa89e25de060c19d3ac86cabb87d6a0ddd05c333b84f4",
                  "4634270f707b6a54daae7530460842e20e37ed265ceee9a43e8924aa"),
            256: ("c672b8d1ef56ed28ab87c3622c5114069bdd3ad7b8f9737498d0c01ecef0967a",
                  "53048e2681941ef99b2e29b76b4c7dabe4c2d0c634fc6d46e0e2f13107e7af23"),
            384: ("38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b",
                  "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7"),
            512: ("cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
                  "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"),
        }
        for bits, digests in published.items():
            for message, digest in zip((b"", b"abc"), digests):
                with self.subTest(bits=bits, message=message):
                    self.assertEqual(hashlib.new({224: "sha512_224", 256: "sha512_256", 384: "sha384", 512: "sha512"}[bits], message).hexdigest(), digest)
                    self.assert_digest(bits, message, 0, exact_end=True)
            self.assert_digest(bits, b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 1)

    def test_padding_blocks_and_unaligned_ranges(self) -> None:
        rng = random.Random(20261004)
        for bits in (224, 256, 384, 512):
            for length in (1, 3, 4, 7, 8, 110, 111, 112, 113, 127, 128, 129,
                           239, 240, 255, 256, 257, 383, 384, 1021):
                message = rng.randbytes(length)
                for alignment in range(4):
                    with self.subTest(bits=bits, length=length, alignment=alignment):
                        self.assert_digest(bits, message, alignment)

    def test_exact_end_and_poison_exclusion(self) -> None:
        for bits in (224, 256, 384, 512):
            for alignment in range(4):
                for length in (0, 112 - alignment, 128 - alignment, 256 - alignment):
                    with self.subTest(bits=bits, length=length, alignment=alignment):
                        self.assert_digest(bits, b"\x93" * length, alignment, exact_end=True)

    def test_million_byte_message(self) -> None:
        for bits in (224, 256, 384, 512):
            with self.subTest(bits=bits):
                self.assert_digest(bits, b"a" * 1_000_000, 0, exact_end=True)

    def test_rejection_preserves_digest(self) -> None:
        for bits in (224, 256, 384, 512):
            for base, length in ((17, 0), (16, 1), (12, 5), (0xFFFFFFFF, 4), (12, 0xFFFFFFFF)):
                with self.subTest(bits=bits, base=base, length=length):
                    result = self.execute("sha2_u32_pair_components", [bits, base, length, POISON], 1, 17)
                    self.assertEqual(result, [0] + [POISON] * 16)
        for bits in (0, 1, 223, 225, 255, 257, 383, 385, 511, 513, 0xFFFFFFFF):
            with self.subTest(bits=bits):
                self.assertEqual(self.execute("sha2_u32_pair_components", [bits, 12, 0, POISON], 1, 17),
                                 [0] + [POISON] * 16)

    def test_paired_word_arithmetic(self) -> None:
        mask = (1 << 64) - 1
        rng = random.Random(20261005)
        cases = [(0, 0), (0xFFFFFFFF, 1), (mask, 1), (1 << 63, 1 << 63),
                 (0xFFFFFFFF00000000, 0x100000000), (0x123456789ABCDEF0, mask)]
        cases.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(6))
        for left, right in cases:
            for shift in (0, 1, 7, 8, 14, 18, 19, 28, 31, 32, 34, 39, 41, 61, 63, 64, 65):
                with self.subTest(left=left, right=right, shift=shift):
                    rotation = shift & 63
                    rotated = ((left >> rotation) | (left << ((64 - rotation) & 63))) & mask
                    expected = ((left + right) & mask, rotated, left >> shift)
                    lanes = [word for value in expected for word in (value & 0xFFFFFFFF, value >> 32)]
                    self.assertEqual(self.execute("sha2_u32_pair_arithmetic_components",
                                                  [left & 0xFFFFFFFF, left >> 32,
                                                   right & 0xFFFFFFFF, right >> 32, shift], 1, 6), lanes)


if __name__ == "__main__":
    unittest.main()
