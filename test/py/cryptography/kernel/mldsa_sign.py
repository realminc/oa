"""FIPS 204 signing against pinned independent NIST signatures.

Expanded keys and supplied randomness are public fixtures, not secret storage.
"""
import hashlib
import json
import struct
import unittest

if __package__ == "kernel":
    from _pqc_harness import ROOT, SlangComponentTest, MLDSA_PREHASH_HASHLIB
else:
    from .._pqc_harness import ROOT, SlangComponentTest, MLDSA_PREHASH_HASHLIB


class MldsaSignTests(SlangComponentTest):
    @staticmethod
    def message_transport(parameter, key, rnd, message, context, external):
        context_base = 60 + 4896
        message_base = context_base + len(context)
        raw = struct.pack("<7I", parameter, message_base, len(message), context_base,
                          len(context), int(external), 0)
        raw += rnd + key + bytes([0xA5]) * (4896 - len(key)) + context + message
        raw += bytes((-len(raw)) % 4)
        return list(struct.unpack(f"<{len(raw) // 4}I", raw))

    def test_nist_message_and_context_framing(self) -> None:
        directory = ROOT / "test/fixtures/cryptography/mldsa"
        prompt = json.loads((directory / "sigGen-prompt.json").read_text())
        result = json.loads((directory / "sigGen-expectedResults.json").read_text())
        results = {g["tgId"]: {c["tcId"]: c for c in g["tests"]} for g in result["testGroups"]}
        count = 0
        for group in prompt["testGroups"]:
            if group.get("preHash") == "preHash" or group.get("externalMu"):
                continue
            parameter = int(group["parameterSet"].rsplit("-", 1)[1])
            external = group["signatureInterface"] == "external"
            for case in group["tests"]:
                key = bytes.fromhex(case["sk"])
                rnd = bytes(32) if group["deterministic"] else bytes.fromhex(case["rnd"])
                message = bytes.fromhex(case["message"])
                context = bytes.fromhex(case["context"]) if external else b""
                values = self.message_transport(parameter, key, rnd, message, context, external)
                actual = self.execute("mldsa_sign_message_components", values, 1, 4629)
                signature = bytes.fromhex(results[group["tgId"]][case["tcId"]]["signature"])
                with self.subTest(group=group["tgId"], case=case["tcId"]):
                    self.assertEqual(actual[:4627], list(signature) + [0xA5] * (4627 - len(signature)))
                    self.assertEqual(actual[4627], 1)
                count += 1
        self.assertEqual(count, 180)

    def test_message_range_and_context_rejection(self) -> None:
        values = self.message_transport(44, bytes(2560), bytes(32), b"abc", b"ctx", True)
        malformed = []
        # Base/length overflow, beyond-storage ends, context length and internal framing.
        for field, value in ((1, 0xFFFFFFFF), (2, 0xFFFFFFFF),
                             (3, 0xFFFFFFFF), (4, 256), (5, 0)):
            row = values.copy()
            row[field] = value
            malformed.append(row)
        for row in malformed:
            self.assertEqual(self.execute("mldsa_sign_message_components", row, 1, 4629),
                             [0xA5] * 4627 + [0, 0])
        self.assertEqual(self.execute("mldsa_sign_message_components", [], 0, 4629), [])

    def test_nist_signatures_all_parameters_and_randomness_modes(self) -> None:
        directory = ROOT / "test/fixtures/cryptography/mldsa"
        manifest = json.loads((directory / "manifest.json").read_text())
        for filename in ("sigGen-prompt.json", "sigGen-expectedResults.json"):
            self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                             manifest["files"][filename]["sha256"])
        prompt = json.loads((directory / "sigGen-prompt.json").read_text())
        result = json.loads((directory / "sigGen-expectedResults.json").read_text())
        results = {g["tgId"]: {c["tcId"]: c for c in g["tests"]} for g in result["testGroups"]}
        coverage = {}
        for group in prompt["testGroups"]:
            if group.get("preHash") == "preHash":
                continue  # Separate HashML-DSA framing/OIDs are not claimed here.
            parameter = int(group["parameterSet"].rsplit("-", 1)[1])
            self.assertEqual(set(results[group["tgId"]]), {c["tcId"] for c in group["tests"]})
            values, references = [], []
            for case in group["tests"]:
                key = bytes.fromhex(case["sk"])
                self.assertEqual(len(key), {44: 2560, 65: 4032, 87: 4896}[parameter])
                rnd = bytes(32) if group["deterministic"] else bytes.fromhex(case["rnd"])
                if group.get("externalMu"):
                    mu = bytes.fromhex(case["mu"])
                else:
                    message = bytes.fromhex(case["message"])
                    if group["signatureInterface"] == "external":
                        context = bytes.fromhex(case["context"])
                        self.assertLessEqual(len(context), 255)
                        message = bytes([0, len(context)]) + context + message
                    mu = hashlib.shake_256(key[64:128] + message).digest(64)
                self.assertEqual((len(rnd), len(mu)), (32, 64))
                values += [parameter] + list(key) + [0xA5] * (4896 - len(key)) + list(rnd) + list(mu)
                signature = bytes.fromhex(results[group["tgId"]][case["tcId"]]["signature"])
                self.assertEqual(len(signature), {44: 2420, 65: 3309, 87: 4627}[parameter])
                references.append(list(signature) + [0xA5] * (4627 - len(signature)))
            actual = self.execute("mldsa_sign_components", values, len(references), 4629)
            for i, reference in enumerate(references):
                with self.subTest(group=group["tgId"], case=group["tests"][i]["tcId"]):
                    row = actual[i * 4629:(i + 1) * 4629]
                    self.assertEqual(row[:4627], reference)
                    self.assertEqual(row[4627], 1)
                    self.assertGreaterEqual(row[4628], 1)
            coverage[(parameter, group["deterministic"], group["signatureInterface"],
                      group.get("externalMu"))] = len(references)
        self.assertEqual(len(coverage), 18)
        self.assertEqual(sum(coverage.values()), 270)

    def test_mask_nonce_bitwidth_and_rate_boundaries(self) -> None:
        values, expected = [], []
        for parameter, bits, gamma1 in ((44, 18, 131072), (65, 20, 524288), (87, 20, 524288)):
            for nonce in (0, 255, 256, 65535):
                for seed in (bytes(64), bytes([255]) * 64, bytes(range(64))):
                    values += [nonce, parameter] + list(seed)
                    raw = hashlib.shake_256(seed + nonce.to_bytes(2, "little")).digest(bits * 32)
                    integer = int.from_bytes(raw, "little")
                    expected += [(gamma1 - ((integer >> (bits * i)) & ((1 << bits) - 1))) % 8380417
                                 for i in range(256)]
        self.assertEqual(self.execute("mldsa_mask_components", values, 36, 256), expected)

    def test_invalid_inputs_preserve_signature(self) -> None:
        invalid = []
        for parameter in (0, 43, 66, 88, 0xFFFFFFFF):
            invalid.append([parameter] + [0] * 4992)
        for parameter in (44, 65, 87):
            # Eta packing's maximum bit pattern is non-canonical for both eta values.
            value = [parameter] + [0] * 4992
            value[129] = 255
            invalid.append(value)
            for offset in (1, 4897, 4929):
                value = [parameter] + [0] * 4992
                value[offset] = 256  # Laboratory byte transport rejects high bits.
                invalid.append(value)
        actual = self.execute("mldsa_sign_components", sum(invalid, []), len(invalid), 4629)
        self.assertEqual(actual, ([0xA5] * 4627 + [0, 0]) * len(invalid))
        self.assertEqual(self.execute("mldsa_sign_components", [], 0, 4629), [])


class MldsaWorkspaceTests(SlangComponentTest):
    CACHE_WORDS = (7 + 8 + 8) * 256
    CACHE_BASE = 4630
    WIDTH = CACHE_BASE + CACHE_WORDS + 1
    POISON = 0xA5A5A5A5

    @staticmethod
    def cases():
        directory = ROOT / "test/fixtures/cryptography/mldsa"
        prompt = json.loads((directory / "sigGen-prompt.json").read_text())
        result = json.loads((directory / "sigGen-expectedResults.json").read_text())
        results = {g["tgId"]: {c["tcId"]: c for c in g["tests"]} for g in result["testGroups"]}
        for group in prompt["testGroups"]:
            if not group.get("externalMu"):
                continue
            parameter = int(group["parameterSet"].rsplit("-", 1)[1])
            for case in group["tests"]:
                key = bytes.fromhex(case["sk"])
                rnd = bytes(32) if group["deterministic"] else bytes.fromhex(case["rnd"])
                values = [parameter] + list(key) + [0xA5] * (4896 - len(key))
                values += list(rnd) + list(bytes.fromhex(case["mu"]))
                signature = bytes.fromhex(results[group["tgId"]][case["tcId"]]["signature"])
                yield parameter, group["deterministic"], values, signature

    def test_workspace_signatures_and_poisoned_unused_rows(self):
        seen, count = set(), 0
        for parameter, deterministic, values, signature in self.cases():
            actual = self.execute("mldsa_sign_workspace_components",
                                  [self.CACHE_BASE * 4] + values, 1, self.WIDTH)
            self.assertEqual(actual[:4627], list(signature) + [0xA5] * (4627 - len(signature)))
            self.assertEqual(actual[4627], 1)
            self.assertGreaterEqual(actual[4628], 1)
            self.assertEqual(actual[4629], self.POISON)
            self.assertEqual(actual[-1], self.POISON)
            cols, rows = {44: (4, 4), 65: (5, 6), 87: (7, 8)}[parameter]
            for base, active, capacity in ((0, cols, 7), (7, rows, 8), (15, rows, 8)):
                start = self.CACHE_BASE + base * 256
                self.assertTrue(all(word < 8380417 for word in actual[start:start + active * 256]))
                self.assertEqual(actual[start + active * 256:start + capacity * 256],
                                 [self.POISON] * ((capacity - active) * 256))
            seen.add((parameter, deterministic))
            count += 1
        self.assertEqual(count, 90)
        self.assertEqual(len(seen), 6)

    def test_buffered_message_and_prehash_framing(self):
        directory = ROOT / "test/fixtures/cryptography/mldsa"
        prompt = json.loads((directory / "sigGen-prompt.json").read_text())
        result = json.loads((directory / "sigGen-expectedResults.json").read_text())
        results = {g["tgId"]: {c["tcId"]: c for c in g["tests"]} for g in result["testGroups"]}
        count = {"pure": 0, "prehash": 0}
        for group in prompt["testGroups"]:
            if group.get("externalMu"):
                continue
            parameter = int(group["parameterSet"].rsplit("-", 1)[1])
            external = group["signatureInterface"] == "external"
            prehashed = group.get("preHash") == "preHash"
            mode = "prehash" if prehashed else "pure"
            entry = "mldsa_prehash_workspace_components" if prehashed else "mldsa_sign_message_workspace_components"
            for case in group["tests"]:
                key = bytes.fromhex(case["sk"])
                rnd = bytes(32) if group["deterministic"] else bytes.fromhex(case["rnd"])
                message = bytes.fromhex(case["message"])
                context = bytes.fromhex(case["context"]) if external else b""
                selector = int(external)
                if prehashed:
                    selector, name, length = MLDSA_PREHASH_HASHLIB[case["hashAlg"]]
                    h = hashlib.new(name, message)
                    message = h.digest(length) if name.startswith("shake_") else h.digest()
                values = MldsaSignTests.message_transport(parameter, key, rnd, message, context, selector)
                values[5] = selector
                values[6] = self.CACHE_BASE * 4
                actual = self.execute(entry, values, 1, self.WIDTH)
                signature = bytes.fromhex(results[group["tgId"]][case["tcId"]]["signature"])
                with self.subTest(mode=mode, group=group["tgId"], case=case["tcId"]):
                    self.assertEqual(actual[:4627], list(signature) + [0xA5] * (4627 - len(signature)))
                    self.assertEqual(actual[4627], 1)
                    self.assertGreaterEqual(actual[4628], 1)
                    self.assertEqual(actual[self.CACHE_BASE - 1], self.POISON)
                    self.assertEqual(actual[-1], self.POISON)
                count[mode] += 1
        self.assertEqual(count, {"pure": 180, "prehash": 90})

    def test_framed_workspace_preflight_preserves_poison(self):
        # The workspace check must happen before framing and secret cache writes.
        for entry, selector in (("mldsa_sign_message_workspace_components", 1),
                                ("mldsa_prehash_workspace_components", 3)):
            values = MldsaSignTests.message_transport(44, bytes(2560), bytes(32),
                                                       bytes(64), b"ctx", selector)
            values[5] = selector
            for base, width in ((self.CACHE_BASE * 4 + 1, self.WIDTH),
                                (0xFFFFFFFC, self.WIDTH),
                                (self.CACHE_BASE * 4, self.WIDTH - 2)):
                values[6] = base
                actual = self.execute(entry, values, 1, width)
                self.assertEqual(actual[:4629], [0xA5] * 4627 + [0, 0])
                self.assertEqual(actual[4629:], [self.POISON] * (width - 4629))
            values[6] = self.CACHE_BASE * 4
            # Context length >255 fails before any polynomial is written.
            values[4] = 256
            actual = self.execute(entry, values, 1, self.WIDTH)
            self.assertEqual(actual[:4629], [0xA5] * 4627 + [0, 0])
            self.assertEqual(actual[4629:], [self.POISON] * (self.WIDTH - 4629))

    def test_failed_decode_preserves_output_and_keeps_workspace_bounded(self):
        parameter, _, values, _ = next(self.cases())
        self.assertEqual(parameter, 44)
        for index, value in ((0, 0), (1, 256), (129, 255), (513, 255)):
            malformed = values.copy()
            malformed[index] = value
            actual = self.execute("mldsa_sign_workspace_components",
                                  [self.CACHE_BASE * 4] + malformed, 1, self.WIDTH)
            self.assertEqual(actual[:4629], [0xA5] * 4627 + [0, 0])
            self.assertEqual(actual[4629], self.POISON)
            self.assertEqual(actual[-1], self.POISON)
            if index == 513:  # s2 decode fails after s1 has already been cached.
                self.assertTrue(all(word < 8380417 for word in
                                    actual[self.CACHE_BASE:self.CACHE_BASE + 4 * 256]))
            else:
                self.assertEqual(actual[self.CACHE_BASE:-1], [self.POISON] * self.CACHE_WORDS)

    def test_workspace_exact_end_is_admitted(self):
        _, _, values, signature = next(self.cases())
        actual = self.execute("mldsa_sign_workspace_components",
                              [self.CACHE_BASE * 4] + values, 1, self.WIDTH - 1)
        self.assertEqual(actual[:4627], list(signature) + [0xA5] * (4627 - len(signature)))
        self.assertEqual(actual[4627], 1)
        self.assertEqual(actual[4629], self.POISON)

    def test_workspace_bounds_precede_secret_access(self):
        _, _, values, _ = next(self.cases())
        for base, width in ((self.CACHE_BASE * 4 + 1, self.WIDTH),
                            (0xFFFFFFFC, self.WIDTH),
                            (self.CACHE_BASE * 4, self.WIDTH - 2)):
            with self.subTest(base=base, width=width):
                actual = self.execute("mldsa_sign_workspace_components", [base] + values, 1, width)
                self.assertEqual(actual[:4629], [0xA5] * 4627 + [0, 0])
                self.assertEqual(actual[4629:], [self.POISON] * (width - 4629))


class MldsaPrehashTests(SlangComponentTest):
    @staticmethod
    def prehash_transport(parameter, key, rnd, digest, context, algorithm):
        context_base = 60 + 4896
        digest_base = context_base + len(context)
        raw = struct.pack("<7I", parameter, digest_base, len(digest), context_base,
                          len(context), algorithm, 0)
        raw += rnd + key + bytes([0xA5]) * (4896 - len(key)) + context + digest
        raw += bytes((-len(raw)) % 4)
        return list(struct.unpack(f"<{len(raw) // 4}I", raw))

    def test_nist_prehashed_signatures(self):
        directory = ROOT / "test/fixtures/cryptography/mldsa"
        manifest = json.loads((directory / "manifest.json").read_text())
        for filename in ("sigGen-prompt.json", "sigGen-expectedResults.json"):
            self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                             manifest["files"][filename]["sha256"])
        prompt = json.loads((directory / "sigGen-prompt.json").read_text())
        result = json.loads((directory / "sigGen-expectedResults.json").read_text())
        results = {g["tgId"]: {c["tcId"]: c for c in g["tests"]} for g in result["testGroups"]}
        # Independent hashlib / NIST identifiers, not the generated shader table.
        seen, count = set(), 0
        for group in prompt["testGroups"]:
            if group.get("preHash") != "preHash":
                continue
            self.assertEqual(set(results[group["tgId"]]), {c["tcId"] for c in group["tests"]})
            parameter = int(group["parameterSet"].rsplit("-", 1)[1])
            for case in group["tests"]:
                algorithm, name, length = MLDSA_PREHASH_HASHLIB[case["hashAlg"]]
                message = bytes.fromhex(case["message"])
                h = hashlib.new(name, message)
                digest = h.digest(length) if name.startswith("shake_") else h.digest()
                self.assertEqual(len(digest), length)
                context = bytes.fromhex(case["context"])
                rnd = bytes(32) if group["deterministic"] else bytes.fromhex(case["rnd"])
                values = self.prehash_transport(parameter, bytes.fromhex(case["sk"]), rnd,
                                               digest, context, algorithm)
                actual = self.execute("mldsa_prehash_components", values, 1, 4629)
                expected = bytes.fromhex(results[group["tgId"]][case["tcId"]]["signature"])
                with self.subTest(group=group["tgId"], case=case["tcId"]):
                    self.assertEqual(actual[:4627], list(expected) + [0xA5] * (4627 - len(expected)))
                    self.assertEqual(actual[4627], 1)
                seen.add((parameter, group["deterministic"]))
                count += 1
        self.assertEqual(count, 90)
        self.assertEqual(len(seen), 6)

    def test_prehashed_selector_and_range_rejection(self):
        values = self.prehash_transport(44, bytes(2560), bytes(32), bytes(64), b"ctx", 3)
        for field, value in ((0, 43), (1, 0xffffffff), (2, 63), (3, 0xffffffff),
                             (4, 256), (5, 0), (5, 13), (5, 1)):
            row = values.copy()
            row[field] = value
            with self.subTest(field=field, value=value):
                self.assertEqual(self.execute("mldsa_prehash_components", row, 1, 4629),
                                 [0xA5] * 4627 + [0, 0])


class MldsaHashMessageTests(SlangComponentTest):
    ENTRY = "mldsa_hash_message_workspace_components"
    CACHE_BASE = MldsaWorkspaceTests.CACHE_BASE
    WIDTH = MldsaWorkspaceTests.WIDTH
    POISON = MldsaWorkspaceTests.POISON

    def test_nist_message_taking_signatures(self):
        directory = ROOT / "test/fixtures/cryptography/mldsa"
        manifest = json.loads((directory / "manifest.json").read_text())
        for filename in ("sigGen-prompt.json", "sigGen-expectedResults.json"):
            self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                             manifest["files"][filename]["sha256"])
        prompt = json.loads((directory / "sigGen-prompt.json").read_text())
        result = json.loads((directory / "sigGen-expectedResults.json").read_text())
        results = {g["tgId"]: {c["tcId"]: c for c in g["tests"]} for g in result["testGroups"]}
        seen, algorithms, count = set(), set(), 0
        for group in prompt["testGroups"]:
            if group.get("preHash") != "preHash":
                continue
            parameter = int(group["parameterSet"].rsplit("-", 1)[1])
            for case in group["tests"]:
                algorithm, _, _ = MLDSA_PREHASH_HASHLIB[case["hashAlg"]]
                rnd = bytes(32) if group["deterministic"] else bytes.fromhex(case["rnd"])
                values = MldsaPrehashTests.prehash_transport(
                    parameter, bytes.fromhex(case["sk"]), rnd,
                    bytes.fromhex(case["message"]), bytes.fromhex(case["context"]), algorithm)
                values[6] = self.CACHE_BASE * 4
                actual = self.execute(self.ENTRY, values, 1, self.WIDTH)
                signature = bytes.fromhex(results[group["tgId"]][case["tcId"]]["signature"])
                with self.subTest(group=group["tgId"], case=case["tcId"]):
                    self.assertEqual(actual[:4627], list(signature) + [0xA5] * (4627 - len(signature)))
                    self.assertEqual(actual[4627], 1)
                    self.assertGreaterEqual(actual[4628], 1)
                    self.assertEqual(actual[self.CACHE_BASE - 1], self.POISON)
                    self.assertEqual(actual[-1], self.POISON)
                    cols, rows = {44: (4, 4), 65: (5, 6), 87: (7, 8)}[parameter]
                    for base, active, capacity in ((0, cols, 7), (7, rows, 8), (15, rows, 8)):
                        start = self.CACHE_BASE + base * 256
                        self.assertTrue(all(word < 8380417 for word in actual[start:start + active * 256]))
                        self.assertEqual(actual[start + active * 256:start + capacity * 256],
                                         [self.POISON] * ((capacity - active) * 256))
                seen.add((parameter, group["deterministic"]))
                algorithms.add(algorithm)
                count += 1
        self.assertEqual(count, 90)
        self.assertEqual(len(seen), 6)
        self.assertEqual(algorithms, set(range(1, 13)))

    def test_invalid_inputs_preserve_signature_and_cache(self):
        values = MldsaPrehashTests.prehash_transport(
            44, bytes(2560), bytes(32), b"abc", b"ctx", 1)
        values[6] = self.CACHE_BASE * 4
        malformed = []
        for field, value in ((0, 43), (1, 0xFFFFFFFF), (2, 0xFFFFFFFF),
                             (3, 0xFFFFFFFF), (4, 256), (5, 0), (5, 13),
                             (6, self.CACHE_BASE * 4 + 1), (6, 0xFFFFFFFC)):
            row = values.copy()
            row[field] = value
            malformed.append((row, self.WIDTH))
        malformed.append((values, self.WIDTH - 2))
        for row, width in malformed:
            with self.subTest(header=row[:7], width=width):
                actual = self.execute(self.ENTRY, row, 1, width)
                self.assertEqual(actual[:4629], [0xA5] * 4627 + [0, 0])
                self.assertEqual(actual[4629:], [self.POISON] * (width - 4629))


if __name__ == "__main__":
    unittest.main()
