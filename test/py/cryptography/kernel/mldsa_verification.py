"""Independent mldsa verification proofs for production Slang.

Host execution is a test laboratory, not an OA CPU fallback or GPU qualification.
"""

import hashlib
import itertools
import struct
import json
import unittest

if __package__ == "kernel":
    from _pqc_harness import SlangComponentTest, ROOT, MLDSA_PREHASH_HASHLIB
else:
    from .._pqc_harness import SlangComponentTest, ROOT, MLDSA_PREHASH_HASHLIB

class MldsaVerificationTests(SlangComponentTest):
    def test_buffer_range_overflow_and_tails(self) -> None:
        count = 12
        size = count * 8
        ranges = [(0, 0), (0, size), (size, 0), (size - 1, 1),
                  (size - 4, 4), (size - 4, 5), (size, 1), (size + 1, 0),
                  (0xffffffff, 1), (1, 0xffffffff), (0xfffffff0, 32), (0, 0xffffffff)]
        expected = [int(base <= size and length <= size - base) for base, length in ranges]
        self.assertEqual(self.execute("range_components", list(itertools.chain.from_iterable(ranges)), count, 1),
                         expected)


    def test_complete_verifier_against_nist(self) -> None:
        fixture = json.loads((ROOT / "test/fixtures/cryptography/acvp_mldsa65_sigver.json").read_text())
        cases = []
        for case in fixture["tests"]:
            key = bytes.fromhex(case["pk"])
            signature = bytes.fromhex(case["signature"])
            context = bytes.fromhex(case["context"])
            message = bytes.fromhex(case["message"])
            cases.append((key, signature, context, message, 1952, 3309,
                          len(context), len(message), int(case["testPassed"])))
            if case["testPassed"]:
                def altered(value: bytes) -> bytes:
                    return bytes([value[0] ^ 1]) + value[1:] if value else b"x"
                cases.extend([
                    (altered(key), signature, context, message, 1952, 3309, len(context), len(message), 0),
                    (key, altered(signature), context, message, 1952, 3309, len(context), len(message), 0),
                    (key, signature, context, altered(message), 1952, 3309, len(context), len(altered(message)), 0),
                    (key, signature, altered(context), message, 1952, 3309, len(altered(context)), len(message), 0),
                    (key, signature, context, message, 1951, 3309, len(context), len(message), 0),
                    (key, signature, context, message, 1952, 3308, len(context), len(message), 0),
                    (key, signature, context, message, 1952, 3309, 256, len(message), 0),
                    (key, signature, context, message, 1952, 3309, len(context), 0xffffffff, 0),
                ])
        stride = (24 + 3 + 1952 + 3309 + 255 + max(len(c[3]) for c in cases) + 3) & ~3
        records, expected = bytearray(stride.to_bytes(4, "little")), []
        for key, signature, context, message, pk_len, sig_len, ctx_len, msg_len, valid in cases:
            for offset in range(4):
                record = bytearray([0xA5] * stride)
                fields = (pk_len, sig_len, ctx_len, msg_len, offset, 0)
                for i, value in enumerate(fields):
                    record[i * 4:i * 4 + 4] = value.to_bytes(4, "little")
                base = 24 + offset
                record[base:base + 1952] = key
                base += 1952
                record[base:base + 3309] = signature
                base += 3309
                record[base:base + len(context)] = context
                base += 255
                record[base:base + len(message)] = message
                records.extend(record)
                expected.append(valid)
        words = [int.from_bytes(records[i:i + 4], "little") for i in range(0, len(records), 4)]
        self.assertEqual(self.execute("verification_components", words, len(expected), 1), expected)


    def test_all_parameter_sets_against_nist(self) -> None:
        fixture = json.loads((ROOT / "test/fixtures/cryptography/acvp_mldsa_sigver.json").read_text())
        cases = []
        for group in fixture["testGroups"]:
            parameter = int(group["parameterSet"].rsplit("-", 1)[1])
            for case in group["tests"]:
                key, signature, context, message = (bytes.fromhex(case[name])
                                                   for name in ("pk", "signature", "context", "message"))
                row = (parameter, key, signature, context, message,
                       len(key), len(signature), len(context), len(message), int(case["testPassed"]))
                cases.append(row)
                if case["testPassed"]:
                    for index in range(1, 5):
                        changed = list(row)
                        data = changed[index]
                        changed[index] = bytes([data[0] ^ 1]) + data[1:] if data else b"x"
                        if index == 3:
                            changed[7] = len(changed[index])
                        if index == 4:
                            changed[8] = len(changed[index])
                        changed[9] = 0
                        cases.append(tuple(changed))
                    for index, value in ((0, 0), (0, 0xffffffff), (5, len(key) - 1),
                                         (6, len(signature) - 1), (7, 256), (8, 0xffffffff)):
                        changed = list(row)
                        changed[index], changed[9] = value, 0
                        cases.append(tuple(changed))
                    # Same bytes must never be accepted under another parameter set.
                    for other in (44, 65, 87):
                        if other != parameter:
                            changed = list(row)
                            changed[0], changed[9] = other, 0
                            cases.append(tuple(changed))
        stride = (28 + 3 + 2592 + 4627 + 255 + max(len(row[4]) for row in cases) + 3) & ~3
        records, expected = bytearray(stride.to_bytes(4, "little")), []
        for parameter, key, signature, context, message, pk_len, sig_len, ctx_len, msg_len, valid in cases:
            for offset in range(4):
                record = bytearray([0xA5] * stride)
                fields = (parameter, pk_len, sig_len, ctx_len, msg_len, offset, 0)
                for i, value in enumerate(fields):
                    record[i * 4:i * 4 + 4] = value.to_bytes(4, "little")
                base = 28 + offset
                record[base:base + len(key)] = key
                base += 2592
                record[base:base + len(signature)] = signature
                base += 4627
                record[base:base + len(context)] = context
                base += 255
                record[base:base + len(message)] = message
                records.extend(record)
                expected.append(valid)
        words = [int.from_bytes(records[i:i + 4], "little") for i in range(0, len(records), 4)]
        actual = self.execute("parameter_verification_components", words, len(expected), 1)
        for index, (got, wanted) in enumerate(zip(actual, expected, strict=True)):
            with self.subTest(row=index, parameter=cases[index // 4][0], alignment=index % 4):
                self.assertEqual(got, wanted)
        self.assertEqual(len(expected), 612)



class MldsaPrehashedVerificationTests(SlangComponentTest):
    def test_nist_hash_message_verification_and_rejection(self):
        self.check_nist_hash_verification(True)

    def test_nist_prehashed_verification_and_rejection(self):
        self.check_nist_hash_verification(False)

    def check_nist_hash_verification(self, hash_message):
        directory = ROOT / "test/fixtures/cryptography/mldsa"
        manifest = json.loads((directory / "manifest.json").read_text())
        for filename in ("sigVer-prompt.json", "sigVer-expectedResults.json"):
            self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                             manifest["files"][filename]["sha256"])
        prompt = json.loads((directory / "sigVer-prompt.json").read_text())
        result = json.loads((directory / "sigVer-expectedResults.json").read_text())
        results = {g["tgId"]: {c["tcId"]: c for c in g["tests"]} for g in result["testGroups"]}
        cases, corpus_count, seen = [], 0, set()
        for group in prompt["testGroups"]:
            if group.get("preHash") != "preHash":
                continue
            self.assertEqual(set(results[group["tgId"]]), {c["tcId"] for c in group["tests"]})
            parameter = int(group["parameterSet"].rsplit("-", 1)[1])
            for case in group["tests"]:
                algorithm, name, length = MLDSA_PREHASH_HASHLIB[case["hashAlg"]]
                h = hashlib.new(name, bytes.fromhex(case["message"]))
                digest = h.digest(length) if name.startswith("shake_") else h.digest()
                if hash_message:
                    digest = bytes.fromhex(case["message"])
                key, signature, context = (bytes.fromhex(case[n]) for n in ("pk", "signature", "context"))
                valid = int(results[group["tgId"]][case["tcId"]]["testPassed"])
                row = (parameter, key, signature, context, digest, algorithm,
                       len(key), len(signature), len(context), len(digest), valid)
                cases.append(row)
                corpus_count += 1
                seen.add(parameter)
                if not valid:
                    continue
                # Cryptographic mutations, independent of the preflight checks.
                for index in range(1, 5):
                    changed = list(row)
                    data = changed[index]
                    changed[index] = bytes([data[0] ^ 1]) + data[1:] if data else b"x"
                    if index == 3:
                        changed[8] = len(changed[index])
                    changed[10] = 0
                    cases.append(tuple(changed))
                # Unknown selectors cannot silently turn into pure verification.
                # Same-size SHAKE/SHA-2 identities must still bind distinct OIDs.
                other = {28: 4, 32: 1, 48: 2, 64: 3}[length]
                if other == algorithm:
                    other = {28: 7, 32: 11, 48: 9, 64: 12}[length]
                for index, value in ((0, 43), (5, 0), (5, 13), (5, 0xffffffff), (5, other),
                                     (6, len(key) - 1), (7, len(signature) - 1),
                                     (8, 256), (8, 0xffffffff), (9, len(digest) + 1 if hash_message else length - 1), (9, 0xffffffff)):
                    changed = list(row)
                    changed[index], changed[10] = value, 0
                    cases.append(tuple(changed))
                for other_set in (44, 65, 87):
                    if other_set != parameter:
                        changed = list(row)
                        changed[0], changed[10] = other_set, 0
                        cases.append(tuple(changed))
        self.assertEqual(corpus_count, 45)
        self.assertEqual(seen, {44, 65, 87})
        capacity = max(64, max(len(row[4]) for row in cases))
        stride = (28 + 3 + 2592 + 4627 + 255 + capacity + 3) & ~3
        records, expected = bytearray(stride.to_bytes(4, "little")), []
        for parameter, key, signature, context, digest, algorithm, pk_len, sig_len, ctx_len, dig_len, valid in cases:
            for offset in range(4):
                record = bytearray([0xA5] * stride)
                record[:28] = struct.pack("<7I", parameter, pk_len, sig_len, ctx_len, dig_len, offset, algorithm)
                base = 28 + offset
                for data, reserved_capacity in ((key, 2592), (signature, 4627), (context, 255), (digest, capacity)):
                    record[base:base + len(data)] = data
                    base += reserved_capacity
                records.extend(record)
                expected.append(valid)
        words = list(struct.unpack(f"<{len(records) // 4}I", records))
        entry = "hash_message_verification_components" if hash_message else "prehashed_verification_components"
        actual = self.execute(entry, words, len(expected), 1)
        self.assertEqual(actual, expected)
        # A HashML-DSA signature over a digest never admits pure-mode framing.
        self.assertEqual(self.execute("parameter_verification_components", words, len(expected), 1),
                         [0] * len(expected))


if __name__ == "__main__":
    unittest.main()
