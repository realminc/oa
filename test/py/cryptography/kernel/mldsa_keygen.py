"""FIPS 204 key generation against the complete pinned NIST sample corpus.

Expanded secret keys here are public test fixtures, never production material.
"""

import hashlib
import json
import unittest

if __package__ == "kernel":
    from _pqc_harness import ROOT, SlangComponentTest
else:
    from .._pqc_harness import ROOT, SlangComponentTest


class MldsaKeygenTests(SlangComponentTest):
    def test_complete_nist_keygen_corpus(self) -> None:
        directory = ROOT / "test/fixtures/cryptography/mldsa"
        manifest = json.loads((directory / "manifest.json").read_text())
        for filename, metadata in manifest["files"].items():
            self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                             metadata["sha256"], filename)
        prompt = json.loads((directory / "keyGen-prompt.json").read_text())
        result = json.loads((directory / "keyGen-expectedResults.json").read_text())
        expected_groups = {group["tgId"]: group for group in result["testGroups"]}
        values, expected = [], []
        coverage = {}
        for group in prompt["testGroups"]:
            parameter = int(group["parameterSet"].rsplit("-", 1)[1])
            results = {case["tcId"]: case for case in expected_groups[group["tgId"]]["tests"]}
            self.assertEqual(set(results), {case["tcId"] for case in group["tests"]})
            coverage[parameter] = len(group["tests"])
            for case in group["tests"]:
                seed = bytes.fromhex(case["seed"])
                self.assertEqual(len(seed), 32)
                values += [parameter] + list(seed)
                reference = results[case["tcId"]]
                pk, sk = bytes.fromhex(reference["pk"]), bytes.fromhex(reference["sk"])
                self.assertEqual((len(pk), len(sk)),
                                 {44: (1312, 2560), 65: (1952, 4032), 87: (2592, 4896)}[parameter])
                expected += list(pk) + [0xA5] * (2592 - len(pk))
                expected += list(sk) + [0xA5] * (4896 - len(sk)) + [1]
        self.assertEqual(coverage, {44: 25, 65: 25, 87: 25})
        self.assertEqual(self.execute("mldsa_keygen_components", values, 75, 7489), expected)

    def test_secret_sampling_nonce_and_rate_boundaries(self) -> None:
        values, expected = [], []
        for seed in (bytes(64), bytes([255]) * 64, bytes(range(64))):
            for eta in (2, 4):
                for nonce in (0, 255, 256, 65535):
                    values += [nonce, eta] + list(seed)
                    xof = hashlib.shake_256(seed + nonce.to_bytes(2, "little"))
                    length = 136
                    while True:
                        coefficients = []
                        for byte in xof.digest(length):
                            for nibble in (byte & 15, byte >> 4):
                                if nibble < (15 if eta == 2 else 9):
                                    coefficients.append((eta - (nibble % 5 if eta == 2 else nibble)) % 8380417)
                        if len(coefficients) >= 256:
                            expected += coefficients[:256]
                            break
                        length += 136
        self.assertEqual(self.execute("mldsa_secret_sampling_components", values, 24, 256), expected)

    def test_unknown_parameter_preserves_output(self) -> None:
        for parameter in (0, 43, 66, 88, 0xFFFFFFFF):
            self.assertEqual(self.execute("mldsa_keygen_components", [parameter] + [0] * 32, 1, 7489),
                             [0xA5] * 7488 + [0])
        self.assertEqual(self.execute("mldsa_keygen_components", [], 0, 7489), [])


if __name__ == "__main__":
    unittest.main()
