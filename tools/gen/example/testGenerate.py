#!/usr/bin/env python3
"""Contract tests for paired SDK example generation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR = REPO_ROOT / "tools" / "gen" / "example" / "generate.py"


class ExampleGeneratorTests(unittest.TestCase):
	def runGenerator(self, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
		return subprocess.run(
			[sys.executable, str(GENERATOR), "--out", str(output), *extra],
			cwd=REPO_ROOT,
			capture_output=True,
			text=True,
			check=False,
		)

	def test_generates_paired_sources_and_inventory(self):
		with tempfile.TemporaryDirectory() as temporary:
			output = Path(temporary)
			result = self.runGenerator(output)
			self.assertEqual(result.returncode, 0, result.stderr)
			matrixPython = (output / "sdk/py/examples/core/matrix.py").read_text()
			matrixCpp = (output / "sdk/cpp/examples/core/matrix.cpp").read_text()
			audioCpp = (output / "sdk/cpp/examples/audio/audio.cpp").read_text()
			cryptoCpp = (output / "sdk/cpp/examples/crypto/shake256.cpp").read_text()
			cryptoPython = (output / "sdk/py/examples/crypto/shake256.py").read_text()
			plotCpp = (output / "sdk/cpp/examples/ui/plotFigure.cpp").read_text()
			plotPython = (output / "sdk/py/examples/ui/plotFigure.py").read_text()
			visionCpp = (output / "sdk/cpp/examples/vision/image.cpp").read_text()
			visionPython = (output / "sdk/py/examples/vision/image.py").read_text()
			self.assertIn("import oa", matrixPython)
			self.assertNotIn("initComputeEngine", matrixPython)
			self.assertNotIn("shutdownComputeEngine", matrixPython)
			self.assertIn('OA_MAIN("ExampleCoreMatrix")', matrixCpp)
			self.assertIn("#include <oa/oa.h>", matrixCpp)
			self.assertNotIn("#include <oa/core/", matrixCpp)
			self.assertNotIn("oa::withEngine", matrixCpp)
			self.assertIn('OA_MAIN_PREVIEW("ExampleAudio")', audioCpp)
			self.assertIn("#include <oa/oa.h>", audioCpp)
			self.assertNotIn("#include <oa/audio/", audioCpp)
			self.assertNotIn("oa::withEngine", audioCpp)
			self.assertNotIn("oa::Engine::create", audioCpp)
			self.assertNotIn("engine->submit", audioCpp)
			self.assertIn("oa::FnAudio::saveWavF32", audioCpp)
			self.assertIn("oa::FnAudio::reverb", audioCpp)
			self.assertIn("oa::Viewer::preview", audioCpp)
			self.assertIn("oa::Viewer::preview(engine,", audioCpp)
			self.assertIn('"--preview"', audioCpp)
			self.assertNotIn("reflectionMilliseconds", audioCpp)
			self.assertNotIn("FnAudio::saturate", audioCpp)
			for source in (cryptoPython, plotPython, visionPython):
				self.assertIn("import oa", source)
				self.assertNotIn("initComputeEngine", source)
				self.assertNotIn("shutdownComputeEngine", source)
			for source in (cryptoCpp, plotCpp, visionCpp):
				self.assertIn("#include <oa/oa.h>", source)
				self.assertNotIn("oa::withEngine", source)
			self.assertIn('OA_MAIN("ExampleCryptoShake256")', cryptoCpp)
			self.assertIn("oa::FnHash::shake256", cryptoCpp)
			self.assertIn("oa.FnHash.shake256", cryptoPython)
			self.assertIn('OA_MAIN_MODE("ExamplePlotLine"', plotCpp)
			self.assertIn("oa::plot::Figure", plotCpp)
			self.assertIn("oa::Viewer::preview", plotCpp)
			self.assertIn("oa.Viewer.preview", plotPython)
			self.assertIn('OA_MAIN_PREVIEW("ExampleVisionImage")', visionCpp)
			self.assertIn("oa::FnImage::grayscale", visionCpp)
			self.assertIn("oa::Viewer::preview", visionCpp)
			self.assertIn("oa.FnImage.grayscale", visionPython)
			self.assertIn("oa.Viewer.preview", visionPython)
			manifest = json.loads((output / "sdk/examples.json").read_text())
			self.assertEqual(len(manifest["examples"]), 5)
			self.assertEqual(
				manifest["examples"][1]["python"]["path"],
				"sdk/py/examples/audio/audio.py",
			)
			presentation = manifest["examples"][1]["presentation"][0]
			self.assertEqual(presentation["kind"], "audioComparison")
			self.assertEqual(
				[item["role"] for item in presentation["items"]],
				["source", "processed"],
			)
			viewerCapture = manifest["examples"][1]["presentation"][1]
			self.assertEqual(viewerCapture["kind"], "viewerCapture")
			self.assertEqual(viewerCapture["width"], 960)
			self.assertEqual(viewerCapture["height"], 434)
			matrixOutput = manifest["examples"][0]["presentation"][0]
			self.assertEqual(matrixOutput["kind"], "terminalOutput")
			self.assertIn("Matrix addition verified", matrixOutput["code"])
			visionPresentation = manifest["examples"][2]["presentation"]
			self.assertEqual(
				[presentation["kind"] for presentation in visionPresentation],
				["imageGallery", "viewerCapture"],
			)
			self.assertEqual(visionPresentation[1]["width"], 1280)
			self.assertEqual(visionPresentation[1]["height"], 720)
			plotPresentation = manifest["examples"][4]["presentation"]
			self.assertEqual(
				[presentation["kind"] for presentation in plotPresentation],
				["imageGallery", "viewerCapture"],
			)
			self.assertEqual(plotPresentation[0]["items"][0]["width"], 960)

	def test_unchanged_generation_preserves_timestamps(self):
		with tempfile.TemporaryDirectory() as temporary:
			output = Path(temporary)
			first = self.runGenerator(output)
			self.assertEqual(first.returncode, 0, first.stderr)
			path = output / "sdk/examples.json"
			mtime = path.stat().st_mtime_ns
			time.sleep(0.02)
			second = self.runGenerator(output)
			self.assertEqual(second.returncode, 0, second.stderr)
			self.assertEqual(path.stat().st_mtime_ns, mtime)

	def test_rejects_unknown_operations(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			schema = root / "invalid.toml"
			schema.write_text("""
schemaVersion = 1
[[examples]]
id = "bad-operation"
module = "Core"
directory = "core"
stem = "bad"
title = "Bad"
summary = "Bad operation."
capability = "Vulkan compute"
cppTarget = "ExampleBad"
pythonProfile = "gpu"
generated = true
expectedOutput = "bad"
cppSymbols = ["oa::FnMatrix::add"]
pythonSymbols = ["oa.FnMatrix.add"]
[[examples.steps]]
operation = "matrix.unknown"
result = "bad"
[[examples.checks]]
kind = "matrixAllClose"
input = "bad"
count = 1
value = 0.0
tolerance = 0.1
""", encoding="utf-8")
			result = self.runGenerator(root / "out", "--schema", str(schema))
			self.assertNotEqual(result.returncode, 0)
			self.assertIn("unsupported operation", result.stderr)


if __name__ == "__main__":
	unittest.main()
