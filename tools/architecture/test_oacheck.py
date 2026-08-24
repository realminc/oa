#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOL_PATH = Path(__file__).with_name("oacheck.py")
SPEC = importlib.util.spec_from_file_location("oacheck", TOOL_PATH)
assert SPEC and SPEC.loader
oacheck = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oacheck
SPEC.loader.exec_module(oacheck)


class OaCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.root = self.repo / "source/cpp/include/oa"
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
                        "core": {"allows": []},
                        "runtime": {"allows": ["core"]},
                        "ml": {"allows": ["core", "runtime"]},
                    },
                }
            ),
            encoding="utf-8",
        )
        self._write_baseline({})

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_baseline(self, exceptions: dict[str, int]) -> None:
        self.baseline.write_text(
            json.dumps({"version": 1, "exceptions": exceptions}), encoding="utf-8"
        )

    def _write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _check(self):
        return oacheck.check(self.repo, self.config, self.baseline)

    def test_allowed_dependency_passes(self) -> None:
        self._write("ml/model.cpp", "#include <oa/core/matrix.h>\n")
        self.assertTrue(self._check().ok)

    def test_new_forbidden_edge_fails(self) -> None:
        self._write("core/matrix.cpp", "#include <oa/ml/itTraining.h>\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("new forbidden dependency: core->ml (1 includes)", result.errors)

    def test_legacy_include_root_fails(self) -> None:
        self._write("ml/model.cpp", "#include <Oa/Core/matrix.h>\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn(
            "non-canonical include root Oa: "
            "source/cpp/include/oa/ml/model.cpp:1",
            result.errors,
        )

    def test_baseline_is_a_growth_cap(self) -> None:
        self._write_baseline({"core->runtime": 1})
        self._write("core/one.cpp", "#include <oa/runtime/engine.h>\n")
        self.assertTrue(self._check().ok)
        self._write("core/two.cpp", "#include <oa/runtime/stream.h>\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("dependency baseline exceeded: core->runtime 2 > 1", result.errors)

    def test_reduced_baseline_is_reported(self) -> None:
        self._write_baseline({"core->runtime": 2})
        self._write("core/one.cpp", "#include <oa/runtime/engine.h>\n")
        result = self._check()
        self.assertTrue(result.ok)
        self.assertIn("baseline can shrink: core->runtime 2 -> 1", result.notices)

    def test_forbidden_public_path_fails(self) -> None:
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["forbidden_public_paths"] = ["runtime/dnn.h"]
        self.config.write_text(json.dumps(config), encoding="utf-8")
        self._write("runtime/dnn.h", "#pragma once\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("forbidden public path: runtime/dnn.h", result.errors)

    def test_forbidden_public_symbol_fails(self) -> None:
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["forbidden_public_symbols"] = ["oa::GemmKernel"]
        self.config.write_text(json.dumps(config), encoding="utf-8")
        self._write("runtime/leak.h", "enum class oa::GemmKernel {};\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("forbidden public symbol: oa::GemmKernel", result.errors)


if __name__ == "__main__":
    unittest.main()
