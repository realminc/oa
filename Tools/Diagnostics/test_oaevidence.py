#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import oaevidence


class OaEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="oa-evidence-test-")
        self.root = pathlib.Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(("git", "init", "-q"), cwd=self.repo, check=True)
        subprocess.run(
            ("git", "config", "user.email", "test@example.invalid"),
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ("git", "config", "user.name", "OA Test"), cwd=self.repo, check=True
        )
        (self.repo / "VERSION").write_text("0.7.5\n", encoding="utf-8")
        (self.repo / "CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=Release\n"
            "OA_VULKAN_VALIDATION:BOOL=OFF\n",
            encoding="utf-8",
        )
        subprocess.run(("git", "add", "VERSION"), cwd=self.repo, check=True)
        subprocess.run(("git", "commit", "-qm", "baseline"), cwd=self.repo, check=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @mock.patch.object(
        oaevidence,
        "_snapshot_vulkan",
        return_value={"available": False, "reason": "test"},
    )
    @mock.patch.object(oaevidence, "_find_registry", return_value=None)
    def test_collects_graph_and_build_provenance(self, _registry, _vulkan) -> None:
        graph = self.root / "graph.json"
        graph.write_text(
            json.dumps({"schema": "oa.execution_graph.v3", "nodes": []}),
            encoding="utf-8",
        )
        benchmark = self.root / "benchmark.json"
        benchmark.write_text(
            json.dumps({"schema": "oa.benchmark.v1", "result": "PASS"}),
            encoding="utf-8",
        )
        output = self.root / "bundle"
        args = oaevidence.parse_args(
            (
                "--repo",
                str(self.repo),
                "--output",
                str(output),
                "--cmake-cache",
                "CMakeCache.txt",
                "--graph",
                str(graph),
                "--benchmark",
                str(benchmark),
            )
        )
        actual, exit_code = oaevidence.collect(args)
        self.assertEqual(actual, output)
        self.assertEqual(exit_code, 0)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], oaevidence.SCHEMA)
        self.assertEqual(manifest["repository"]["version"], "0.7.5")
        self.assertEqual(manifest["build"]["CMAKE_BUILD_TYPE"], "Release")
        self.assertEqual(manifest["build"]["OA_VULKAN_VALIDATION"], "OFF")
        self.assertEqual(len(manifest["graphs"]), 1)
        self.assertEqual(len(manifest["benchmarks"]), 1)
        self.assertTrue((output / manifest["graphs"][0]["path"]).is_file())

    @mock.patch.object(
        oaevidence,
        "_snapshot_vulkan",
        return_value={"available": False, "reason": "test"},
    )
    @mock.patch.object(oaevidence, "_find_registry", return_value=None)
    def test_wraps_workload_and_observes_validation(self, _registry, _vulkan) -> None:
        output = self.root / "wrapped"
        program = (
            "import json, os; "
            "open(os.environ['OA_GRAPH_REPORT'], 'w').write("
            "json.dumps({'schema':'oa.execution_graph.v3','nodes':[]})); "
            "print('Validation layers: ON (VK_LAYER_KHRONOS_validation)'); "
            "print('Validation features: core,synchronization\\033[0m'); "
            "print('GemmRouter: M=64 N=64 K=64 requested=FP32 actual=FP32 "
            "kernel=GemmTiled path=Standard fallback=none grid=1,1,1')"
        )
        args = oaevidence.parse_args(
            (
                "--repo",
                str(self.repo),
                "--output",
                str(output),
                "--cmake-cache",
                "CMakeCache.txt",
                "--validation",
                "--selection-trace",
                "--",
                sys.executable,
                "-c",
                program,
                "--api-token=do-not-store",
            )
        )
        _, exit_code = oaevidence.collect(args)
        self.assertEqual(exit_code, 0)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        workload = manifest["workload"]
        self.assertTrue(workload["validation"]["requested"])
        self.assertTrue(workload["validation"]["observed_enabled"])
        self.assertEqual(
            workload["validation"]["observed_features"],
            ["core", "synchronization"],
        )
        self.assertEqual(workload["validation"]["reported_error_count"], 0)
        self.assertTrue(workload["selection"]["gate_passed"])
        self.assertEqual(workload["selection"]["record_count"], 1)
        self.assertEqual(workload["selection"]["kernel_counts"], {"GemmTiled": 1})
        self.assertEqual(workload["selection"]["unexpected_fallback_count"], 0)
        self.assertIsNotNone(workload["graph"])
        self.assertIn("--api-token=[REDACTED]", workload["command"])
        self.assertNotIn("do-not-store", json.dumps(manifest))

    def test_extracts_validation_layer_version(self) -> None:
        summary = """
VK_LAYER_KHRONOS_validation       Khronos Validation Layer       1.4.354  version 2
VK_LAYER_MESA_device_select       Linux device selection layer   1.4.303  version 1
"""
        self.assertEqual(
            oaevidence._validation_layer_record(summary),
            {
                "name": "VK_LAYER_KHRONOS_validation",
                "api_version": "1.4.354",
                "implementation_version": 2,
            },
        )

    @mock.patch.object(
        oaevidence,
        "_snapshot_vulkan",
        return_value={"available": False, "reason": "test"},
    )
    @mock.patch.object(oaevidence, "_find_registry", return_value=None)
    def test_rejects_non_oa_graph_schema(self, _registry, _vulkan) -> None:
        graph = self.root / "bad.json"
        graph.write_text('{"schema":"something.else"}', encoding="utf-8")
        args = oaevidence.parse_args(
            (
                "--repo",
                str(self.repo),
                "--output",
                str(self.root / "bad-bundle"),
                "--graph",
                str(graph),
            )
        )
        with self.assertRaises(ValueError):
            oaevidence.collect(args)


if __name__ == "__main__":
    unittest.main()
