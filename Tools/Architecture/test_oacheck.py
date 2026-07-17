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
        self.root = self.repo / "Source/Public/Oa"
        (self.root / "Core").mkdir(parents=True)
        (self.root / "Runtime").mkdir(parents=True)
        (self.root / "Ml").mkdir(parents=True)
        self.config = self.repo / "modules.json"
        self.baseline = self.repo / "baseline.json"
        self.config.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source_roots": ["Source/Public/Oa"],
                    "modules": {
                        "Core": {"allows": []},
                        "Runtime": {"allows": ["Core"]},
                        "Ml": {"allows": ["Core", "Runtime"]},
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
        self._write("Ml/Model.cpp", "#include <Oa/Core/Matrix.h>\n")
        self.assertTrue(self._check().ok)

    def test_new_forbidden_edge_fails(self) -> None:
        self._write("Core/Matrix.cpp", "#include <Oa/Ml/Training.h>\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("new forbidden dependency: Core->Ml (1 includes)", result.errors)

    def test_baseline_is_a_growth_cap(self) -> None:
        self._write_baseline({"Core->Runtime": 1})
        self._write("Core/One.cpp", "#include <Oa/Runtime/Engine.h>\n")
        self.assertTrue(self._check().ok)
        self._write("Core/Two.cpp", "#include <Oa/Runtime/Stream.h>\n")
        result = self._check()
        self.assertFalse(result.ok)
        self.assertIn("dependency baseline exceeded: Core->Runtime 2 > 1", result.errors)

    def test_reduced_baseline_is_reported(self) -> None:
        self._write_baseline({"Core->Runtime": 2})
        self._write("Core/One.cpp", "#include <Oa/Runtime/Engine.h>\n")
        result = self._check()
        self.assertTrue(result.ok)
        self.assertIn("baseline can shrink: Core->Runtime 2 -> 1", result.notices)


if __name__ == "__main__":
    unittest.main()
