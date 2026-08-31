#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOL_PATH = Path(__file__).with_name("oaCheck.py")
SPEC = importlib.util.spec_from_file_location("oaCheck", TOOL_PATH)
assert SPEC and SPEC.loader
oaCheck = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oaCheck
SPEC.loader.exec_module(oaCheck)


class OaCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.root = self.repo / "source/cpp/include/oa"
        (self.root / "animation").mkdir(parents=True)
        (self.root / "core").mkdir(parents=True)
        (self.root / "runtime").mkdir(parents=True)
        (self.root / "ml").mkdir(parents=True)
        self.config = self.repo / "modules.json"
        self.baseline = self.repo / "baseline.json"
        self.config.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source_roots": ["source/cpp/include/oa"],
                    "modules": {
                        "animation": {"allows": ["core"]},
                        "core": {"allows": []},
                        "runtime": {"allows": ["core"]},
                        "ml": {"allows": ["core", "runtime"]},
                    },
                }
            ),
            encoding="utf-8",
        )
        self._writeBaseline({})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _writeBaseline(self, exceptions: dict[str, int]) -> None:
        self.baseline.write_text(
            json.dumps({"version": 1, "exceptions": exceptions}), encoding="utf-8"
        )

    def _write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _check(self):
        return oaCheck.check(self.repo, self.config, self.baseline)

    def testAllowedDependencyPasses(self) -> None:
        self._write("ml/model.cpp", "#include <oa/core/matrix.h>\n")
        self.assertTrue(self._check().ok)

    def testAnimationIsAnAdmittedCoreConsumer(self) -> None:
        self._write("animation/animClip.cpp", "#include <oa/core/status.h>\n")
        self.assertTrue(self._check().ok)

    def testNewForbiddenEdgeFails(self) -> None:
        self._write("core/matrix.cpp", "#include <oa/ml/itTraining.h>\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("new forbidden dependency: core->ml (1 includes)", result.errors)

    def testLegacyIncludeRootFails(self) -> None:
        self._write("ml/model.cpp", "#include <Oa/Core/matrix.h>\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn(
            "non-canonical include root Oa: "
            "source/cpp/include/oa/ml/model.cpp:1",
            result.errors,
        )

    def testBaselineIsAGrowthCap(self) -> None:
        self._writeBaseline({"core->runtime": 1})
        self._write("core/one.cpp", "#include <oa/runtime/engine.h>\n")
        self.assertTrue(self._check().ok)
        self._write("core/two.cpp", "#include <oa/runtime/stream.h>\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("dependency baseline exceeded: core->runtime 2 > 1", result.errors)

    def testReducedBaselineIsReported(self) -> None:
        self._writeBaseline({"core->runtime": 2})
        self._write("core/one.cpp", "#include <oa/runtime/engine.h>\n")
        result = self._check()
        self.assertTrue(result.ok)
        self.assertIn("baseline can shrink: core->runtime 2 -> 1", result.notices)

    def testForbiddenPublicPathFails(self) -> None:
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["forbidden_public_paths"] = ["runtime/dnn.h"]
        self.config.write_text(json.dumps(config), encoding="utf-8")
        self._write("runtime/dnn.h", "#pragma once\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("forbidden public path: runtime/dnn.h", result.errors)

    def testForbiddenPublicSymbolFails(self) -> None:
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["forbidden_public_symbols"] = ["oa::GemmKernel"]
        self.config.write_text(json.dumps(config), encoding="utf-8")
        self._write("runtime/leak.h", "enum class oa::GemmKernel {};\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("forbidden public symbol: oa::GemmKernel", result.errors)


if __name__ == "__main__":
    unittest.main()
