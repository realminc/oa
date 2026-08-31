#!/usr/bin/env python3
"""Contract tests for the public example inventory generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import generateExamples as examples


class GenerateExamplesTest(unittest.TestCase):
    def testLiveInventoryContainsRegisteredPairs(self) -> None:
        inventory = examples.generate()
        self.assertEqual(inventory["schema"], 2)
        generated = inventory["examples"]
        self.assertEqual(
            {entry["id"] for entry in generated},
            {
                "core-matrix-add",
                "audio-process",
                "ml-transformer",
                "vision-image",
                "crypto-shake256",
                "plot-line",
            },
        )
        for entry in generated:
            self.assertIn(entry["module"], examples.PUBLIC_MODULES)
            self.assertTrue(entry["cpp"]["symbols"])
            self.assertTrue(entry["python"]["symbols"])
            self.assertTrue(entry["cpp"]["code"])
            self.assertTrue(entry["python"]["code"])
            self.assertEqual(len(entry["cpp"]["sha256"]), 64)
            self.assertEqual(len(entry["python"]["sha256"]), 64)
        matrix = next(entry for entry in generated if entry["id"] == "core-matrix-add")
        terminalOutput = matrix["presentation"][0]
        self.assertEqual(terminalOutput["kind"], "terminalOutput")
        self.assertEqual(terminalOutput["language"], "bash")
        self.assertIn("Matrix addition verified", terminalOutput["code"])
        audio = next(entry for entry in generated if entry["id"] == "audio-process")
        presentation = audio["presentation"][0]
        self.assertEqual(presentation["kind"], "audioComparison")
        self.assertEqual(
            [item["role"] for item in presentation["items"]],
            ["source", "processed"],
        )
        for item in presentation["items"]:
            self.assertTrue(item["src"].startswith("/media/oa/"))
            self.assertEqual(len(item["sha256"]), 64)
            self.assertGreater(item["bytes"], 0)
        viewerCaptures = audio["presentation"][1:]
        self.assertEqual(len(viewerCaptures), 3)
        for viewerCapture in viewerCaptures:
            self.assertEqual(viewerCapture["kind"], "viewerCapture")
            self.assertEqual(viewerCapture["mimeType"], "image/jpeg")
            self.assertEqual(viewerCapture["width"], 1920)
            self.assertEqual(viewerCapture["height"], 720)
            self.assertTrue(viewerCapture["src"].startswith("/media/oa/"))
            self.assertEqual(len(viewerCapture["sha256"]), 64)
            self.assertGreater(viewerCapture["bytes"], 0)
        ml = next(entry for entry in generated if entry["id"] == "ml-transformer")
        self.assertEqual(
            [presentation["title"] for presentation in ml["presentation"]],
            ["C++", "Python"],
        )
        self.assertTrue(all(
            presentation["kind"] == "terminalOutput"
            for presentation in ml["presentation"]
        ))
        self.assertTrue(all(
            "Loss: 5.8423 -> 0.0415" in presentation["code"]
            for presentation in ml["presentation"]
        ))
        self.assertTrue(all(
            "Summary:" in presentation["code"]
            and "GPU: mean" in presentation["code"]
            and "training phases: steps=300" in presentation["code"]
            and "300/300" in presentation["code"]
            for presentation in ml["presentation"]
        ))

    def testExtractAcceptsOneOrderedPair(self) -> None:
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

    def testExtractRejectsDuplicateOrReversedMarkers(self) -> None:
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
