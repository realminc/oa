#!/usr/bin/env python3
"""Contract tests for the public example inventory generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import generate_examples as examples


class GenerateExamplesTest(unittest.TestCase):
    def test_live_inventory_contains_registered_pairs(self) -> None:
        generated = examples.generate()["examples"]
        self.assertEqual(
            {entry["id"] for entry in generated},
            {
                "core-matrix-add",
                "audio-clip",
                "vision-resize",
                "crypto-shake256",
                "plot-line",
            },
        )
        for entry in generated:
            self.assertTrue(entry["cpp"]["code"])
            self.assertTrue(entry["python"]["code"])
            self.assertEqual(len(entry["cpp"]["sha256"]), 64)
            self.assertEqual(len(entry["python"]["sha256"]), 64)

    def test_extract_accepts_one_ordered_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Example.cpp"
            path.write_text(
                "// OA_DOC_BEGIN: sample\nint main() { return 0; }\n"
                "// OA_DOC_END: sample\n",
                encoding="utf-8",
            )
            self.assertEqual(
                examples._extract(path, "sample", "//"),
                "int main() { return 0; }",
            )

    def test_extract_rejects_duplicate_or_reversed_markers(self) -> None:
        fixtures = (
            "// OA_DOC_BEGIN: sample\n// OA_DOC_BEGIN: sample\n"
            "// OA_DOC_END: sample\n",
            "// OA_DOC_END: sample\n// OA_DOC_BEGIN: sample\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Example.cpp"
            for source in fixtures:
                path.write_text(source, encoding="utf-8")
                with self.assertRaises(examples.ExampleError):
                    examples._extract(path, "sample", "//")


if __name__ == "__main__":
    unittest.main()
