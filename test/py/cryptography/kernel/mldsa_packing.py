"""Independent mldsa packing proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import json
import random
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest, ROOT, Q
else:
    from .._pqc_harness import SlangComponentTest, ROOT, Q

class MldsaPackingTests(SlangComponentTest):
    def test_packed_coefficients_all_widths_and_alignments(self) -> None:
        rng = random.Random(20423)
        records, expected = bytearray(), []
        for bits in range(1, 21):
            for offset in range(4):
                coefficients = [0, (1 << bits) - 1] + [rng.randrange(1 << bits) for _ in range(254)]
                integer = sum(value << (i * bits) for i, value in enumerate(coefficients))
                payload = integer.to_bytes(32 * bits, "little")
                record = bytearray([0xA5] * 652)
                record[:4] = bits.to_bytes(4, "little")
                record[4:8] = offset.to_bytes(4, "little")
                record[8 + offset:8 + offset + len(payload)] = payload
                records.extend(record)
                expected.extend(coefficients)
        words = [int.from_bytes(records[i:i + 4], "little") for i in range(0, len(records), 4)]
        self.assertEqual(self.execute("bit_packing_components", words, 80, 256), expected)


    def test_nist_decoding_and_malformed_hints(self) -> None:
        fixture = json.loads((ROOT / "test/fixtures/cryptography/acvp_mldsa65_sigver.json").read_text())
        cases = [(bytes.fromhex(case["pk"]), bytes.fromhex(case["signature"])) for case in fixture["tests"]]
        pk = cases[0][0]
        # Canonical hint ordering restarts in each polynomial, including empty
        # rows. Duplicate/descending positions, decreasing/excess delimiters
        # and nonzero unused positions must fail closed.
        hints = [bytes(61), bytes([255, 0]) + bytes(53) + bytes([1, 2, 2, 2, 2, 2]),
                 bytes([7, 7]) + bytes(53) + bytes([2] * 6),
                 bytes([9, 8]) + bytes(53) + bytes([2] * 6),
                 bytes(55) + bytes([1, 0, 0, 0, 0, 0]),
                 bytes(55) + bytes([56] * 6),
                 bytes([1]) + bytes(60),
                 bytes(range(55)) + bytes([55] * 6)]
        cases += [(pk, bytes(3248) + hint) for hint in hints]

        def unpack(data: bytes, bits: int) -> list[int]:
            integer = int.from_bytes(data, "little")
            return [(integer >> (i * bits)) & ((1 << bits) - 1) for i in range(256)]

        for case_index, (pk, signature) in enumerate(cases):
            rho = [int.from_bytes(pk[i:i + 4], "little") for i in range(0, 32, 4)]
            t1 = [v for k in range(6) for v in unpack(pk[32 + k * 320:32 + (k + 1) * 320], 10)]
            challenge = [int.from_bytes(signature[i:i + 4], "little") for i in range(0, 48, 4)]
            z = [(524288 - v) % Q for l in range(5)
                 for v in unpack(signature[48 + l * 640:48 + (l + 1) * 640], 20)]
            hint = signature[3248:]
            indices, delimiters = hint[:55], hint[55:]
            decoded, previous, valid = [], 0, True
            for end in delimiters:
                positions = list(indices[previous:end])
                valid &= previous <= end <= 55 and positions == sorted(set(positions))
                decoded += [int(i in positions) for i in range(256)]
                previous = end
            valid &= not any(indices[previous:])
            norm = all(min(v, Q - v) < 524092 for v in z)
            for offset in range(4):
                record = bytearray([0xA5] * 5268)
                record[:4] = offset.to_bytes(4, "little")
                record[4 + offset:4 + offset + 5261] = pk + signature
                words = [int.from_bytes(record[i:i + 4], "little") for i in range(0, len(record), 4)]
                actual = self.execute("packing_components", words, 1, 4374)
                with self.subTest(case=case_index, offset=offset):
                    self.assertEqual(actual[:2836], rho + t1 + challenge + z)
                    self.assertEqual(actual[-2:], [int(valid), int(norm)])
                    if valid:
                        self.assertEqual(actual[2836:-2], decoded)
                    else:
                        self.assertTrue(all(value in (0, 1) for value in actual[2836:-2]),
                                        "rejected hint decoding exposed uninitialized coefficients")



if __name__ == "__main__":
    unittest.main()
