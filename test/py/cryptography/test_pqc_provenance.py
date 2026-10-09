"""Pinned official PQC vector provenance; no compiler or device required."""

import hashlib
import json
import unittest

try:
    from ._pqc_harness import ROOT
except ImportError:
    from _pqc_harness import ROOT

class PqcAcvpProvenance(unittest.TestCase):
    def test_pinned_all_mldsa_pure_groups(self):
        fixture = json.loads((ROOT / "test/fixtures/cryptography/acvp_mldsa_sigver.json").read_text())
        self.assertEqual(fixture["source"]["commit"], "975de31eb83d87039ec88934fdc47d8c312b892d")
        self.assertEqual(fixture["source"]["prompt_sha256"], "e2cba4589389756fa0bea1a7e6837138bf0a81f9d14234c9ee8f6d33caa1654e")
        self.assertEqual(fixture["source"]["expected_results_sha256"], "e1d84ef1b2f35196278ab0b0ed6a46ec62cc03d2dfa92c564199e1999bfb8ea6")
        groups = fixture["testGroups"]
        self.assertEqual([g["tgId"] for g in groups], [1, 3, 5])
        self.assertEqual([g["parameterSet"] for g in groups], ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"])
        payload = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), "c0ddb8a2d63899e74aefdbb17717fa959e194fb3f85f5509bee538c8b96deee6")
        old = json.loads((ROOT / "test/fixtures/cryptography/acvp_mldsa65_sigver.json").read_text())
        self.assertEqual(groups[1]["tests"], old["tests"])
        for group, sizes in zip(groups, ((1312, 2420), (1952, 3309), (2592, 4627)), strict=True):
            self.assertEqual(group["signatureInterface"], "external")
            self.assertEqual(group["preHash"], "pure")
            self.assertEqual(len(group["tests"]), 15)
            self.assertEqual(sum(case["testPassed"] for case in group["tests"]), 3)
            for case in group["tests"]:
                self.assertEqual(len(bytes.fromhex(case["pk"])), sizes[0])
                self.assertEqual(len(bytes.fromhex(case["signature"])), sizes[1])

    def test_pinned_mlkem_corpus(self):
        folder = ROOT / "test/fixtures/cryptography/mlkem"
        manifest = json.loads((folder / "manifest.json").read_text())
        self.assertEqual(manifest["commit"], "975de31eb83d87039ec88934fdc47d8c312b892d")
        hashes = {'keyGen-prompt.json': '3f9ce34f6c836c77958bad2729e837c3b213f44ac36c3065976e7acca6389523', 'keyGen-expectedResults.json': 'a253d0ad91c95ebea5b409673defef0aa49d65d4ed72286399e2e798ddf073a4', 'encapDecap-prompt.json': '998e22dfb12efb14ce9fdff911ca634b13612819a1806f25da69adba7e16db91', 'encapDecap-expectedResults.json': '9089ec6ff2424da9f2782b89b2f831a329a3e28d6e5e24b802b78ff36ac61cdf'}
        for name, digest in hashes.items():
            self.assertEqual(manifest["files"][name]["sha256"], digest)
            raw = (folder / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            corpus = json.loads(raw)
            self.assertEqual(corpus["algorithm"], "ML-KEM")
            self.assertEqual(corpus["revision"], "FIPS203")

    def test_pinned_external_pure_verification_group(self) -> None:
        fixture = json.loads((ROOT / "test/fixtures/cryptography/acvp_mldsa65_sigver.json").read_text())
        self.assertEqual(fixture["source"]["commit"], "975de31eb83d87039ec88934fdc47d8c312b892d")
        self.assertEqual(fixture["source"]["prompt_sha256"],
                         "e2cba4589389756fa0bea1a7e6837138bf0a81f9d14234c9ee8f6d33caa1654e")
        self.assertEqual(fixture["source"]["expected_results_sha256"],
                         "e1d84ef1b2f35196278ab0b0ed6a46ec62cc03d2dfa92c564199e1999bfb8ea6")
        self.assertEqual(fixture["revision"], "FIPS204")
        self.assertEqual(fixture["parameterSet"], "ML-DSA-65")
        self.assertEqual(fixture["signatureInterface"], "external")
        self.assertEqual(fixture["preHash"], "pure")
        self.assertEqual([case["tcId"] for case in fixture["tests"]], list(range(31, 46)))
        self.assertEqual([case["tcId"] for case in fixture["tests"] if case["testPassed"]], [33, 43, 44])
        payload = json.dumps(fixture["tests"], sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
                         "336f82e36742afb1584b22b40330380f1f1cf1a2c25a4eee6373f438d461a627")
        for case in fixture["tests"]:
            self.assertEqual(len(bytes.fromhex(case["pk"])), 1952)
            self.assertEqual(len(bytes.fromhex(case["signature"])), 3309)
            self.assertLessEqual(len(bytes.fromhex(case["context"])), 255)


    def test_pinned_mldsa_prehashed_verification_corpus(self):
        folder = ROOT / "test/fixtures/cryptography/mldsa"
        manifest = json.loads((folder / "manifest.json").read_text())
        hashes = {"sigVer-prompt.json": "e2cba4589389756fa0bea1a7e6837138bf0a81f9d14234c9ee8f6d33caa1654e",
                  "sigVer-expectedResults.json": "e1d84ef1b2f35196278ab0b0ed6a46ec62cc03d2dfa92c564199e1999bfb8ea6"}
        for name, digest in hashes.items():
            self.assertEqual(manifest["files"][name]["sha256"], digest)
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), digest)
        prompt = json.loads((folder / "sigVer-prompt.json").read_text())
        result = json.loads((folder / "sigVer-expectedResults.json").read_text())
        outcomes = {g["tgId"]: {c["tcId"]: c["testPassed"] for c in g["tests"]} for g in result["testGroups"]}
        groups = [g for g in prompt["testGroups"] if g.get("preHash") == "preHash"]
        self.assertEqual([g["tgId"] for g in groups], [2, 4, 6])
        for group in groups:
            self.assertEqual(len(group["tests"]), 15)
            self.assertEqual(set(outcomes[group["tgId"]]), {c["tcId"] for c in group["tests"]})
            self.assertEqual(sum(outcomes[group["tgId"]].values()), 3)
