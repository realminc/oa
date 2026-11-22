"""SHA-224/SHA-256 production-helper proofs; no Vulkan execution claim."""

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


class Sha2Tests(SlangComponentTest):
    def assert_digest(self, bits: int, message: bytes, alignment: int, exact_end: bool = False) -> None:
        record = bytearray(struct.pack("<III", bits, 12 + alignment, len(message)))
        record.extend(b"\xa5" * alignment)
        record.extend(message)
        record.extend(b"\xa5" * ((-len(record)) % 4))
        if not exact_end:
            record.extend(b"\xa5" * 16)
        expected = hashlib.new(f"sha{bits}", message).digest()
        self.assertEqual(self.execute("sha2_u32_components", words(record), 1, 9),
                         [1] + words(expected) + [POISON] * (8 - bits // 32))

    def test_known_answers(self) -> None:
        published = {
            224: ("d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f",
                  "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7"),
            256: ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        }
        for bits, digests in published.items():
            for message, digest in zip((b"", b"abc"), digests):
                with self.subTest(bits=bits, message=message):
                    self.assertEqual(hashlib.new(f"sha{bits}", message).hexdigest(), digest)
                    self.assert_digest(bits, message, 0, exact_end=True)
            self.assert_digest(bits, b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 1)

    def test_padding_blocks_and_unaligned_ranges(self) -> None:
        rng = random.Random(20261004)
        for bits in (224, 256):
            for length in (1, 3, 4, 7, 8, 54, 55, 56, 57, 63, 64, 65,
                           119, 120, 127, 128, 129, 191, 192, 1021):
                message = rng.randbytes(length)
                for alignment in range(4):
                    with self.subTest(bits=bits, length=length, alignment=alignment):
                        self.assert_digest(bits, message, alignment)

    def test_exact_end_and_poison_exclusion(self) -> None:
        for bits in (224, 256):
            for alignment in range(4):
                for length in (0, 56 - alignment, 64 - alignment, 128 - alignment):
                    with self.subTest(bits=bits, length=length, alignment=alignment):
                        self.assert_digest(bits, b"\x93" * length, alignment, exact_end=True)

    def test_million_byte_message(self) -> None:
        for bits in (224, 256):
            with self.subTest(bits=bits):
                self.assert_digest(bits, b"a" * 1_000_000, 0, exact_end=True)

    def test_rejection_preserves_digest(self) -> None:
        for bits in (224, 256):
            for base, length in ((17, 0), (16, 1), (12, 5), (0xFFFFFFFF, 4), (12, 0xFFFFFFFF)):
                with self.subTest(bits=bits, base=base, length=length):
                    result = self.execute("sha2_u32_components", [bits, base, length, POISON], 1, 9)
                    self.assertEqual(result, [0] + [POISON] * 8)
        for bits in (0, 1, 223, 225, 255, 257, 384, 512, 0xFFFFFFFF):
            with self.subTest(bits=bits):
                self.assertEqual(self.execute("sha2_u32_components", [bits, 12, 0, POISON], 1, 9),
                                 [0] + [POISON] * 8)

    def test_bit_length_carry_without_large_allocations(self) -> None:
        for length in (0, 1, 55, 56, 63, 64, (1 << 29) - 1, 1 << 29,
                       (1 << 29) + 1, (1 << 31) - 1, 1 << 31, 0xFFFFFFFF):
            with self.subTest(length=length):
                self.assertEqual(self.execute("sha2_u32_length_components", [length], 1, 2),
                                 [(length * 8) >> 32, (length * 8) & 0xFFFFFFFF])


if __name__ == "__main__":
    unittest.main()
