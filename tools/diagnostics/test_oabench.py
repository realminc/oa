#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import oabench


class OaBenchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="oa-bench-test-")
        self.repo = pathlib.Path(self.temporary.name) / "repo"
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
        subprocess.run(("git", "add", "VERSION"), cwd=self.repo, check=True)
        subprocess.run(("git", "commit", "-qm", "baseline"), cwd=self.repo, check=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @mock.patch.object(oabench, "_vulkan_summary", return_value={"available": False})
    @mock.patch.object(oabench, "_power_metadata", return_value={})
    def test_emits_seven_run_metric_protocol(self, _power, _vulkan) -> None:
        output = self.repo / "result.json"
        args = oabench.parse_args(
            (
                "--repo",
                str(self.repo),
                "--output",
                str(output),
                "--name",
                "unit.metric",
                "--contract",
                "shape=1x1",
                "--warmup",
                "1",
                "--runs",
                "7",
                "--cooldown",
                "0",
                "--metric-regex",
                r"metric=(?P<value>[0-9.]+)",
                "--metric-name",
                "unit_time",
                "--metric-unit",
                "us",
                "--require-regex",
                "correct=yes",
                "--",
                sys.executable,
                "-c",
                "print('metric=1.25 correct=yes')",
            )
        )
        actual, status = oabench.run(args)
        self.assertEqual(actual, output)
        self.assertEqual(status, 0)
        document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], oabench.SCHEMA)
        self.assertEqual(document["result"], "PASS")
        self.assertEqual(document["metric"]["statistics"]["count"], 7)
        self.assertEqual(document["metric"]["statistics"]["median"], 1.25)
        self.assertEqual(len(document["samples"]), 8)
        self.assertEqual(document["workload"]["contract"], {"shape": "1x1"})
        self.assertTrue(document["workload"]["command_id"])

    def test_extracts_stable_selected_device_identity(self) -> None:
        summary = """GPU0:
\tvendorID = 0x8086
\tdeviceID = 0x9a49
\tdeviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName = Intel Iris Xe
\tdriverID = DRIVER_ID_INTEL_OPEN_SOURCE_MESA
\tdriverName = Intel Mesa
GPU1:
\tvendorID = 0x10de
\tdeviceID = 0x1234
\tdriverID = DRIVER_ID_NVIDIA_PROPRIETARY
"""
        identity = oabench._platform_identity(
            {"available": True, "summary": summary}, device_index=0
        )
        self.assertTrue(identity["available"])
        self.assertEqual(identity["vendor_id"], "0x8086")
        self.assertEqual(identity["device_id"], "0x9a49")
        self.assertEqual(identity["driver_id"], "DRIVER_ID_INTEL_OPEN_SOURCE_MESA")

    def test_thermal_summary_preserves_observed_range(self) -> None:
        summary = oabench._thermal_summary([
            {
                "thermal_before": [
                    {"sensor": "coretemp:Package id 0", "celsius": 61.0}
                ],
                "thermal_after": [
                    {"sensor": "coretemp:Package id 0", "celsius": 68.0}
                ],
            },
            {
                "thermal_before": [
                    {"sensor": "coretemp:Package id 0", "celsius": 59.0}
                ],
                "thermal_after": [],
            },
        ])
        self.assertEqual(summary, [{
            "sensor": "coretemp:Package id 0",
            "minimum_celsius": 59.0,
            "maximum_celsius": 68.0,
        }])

    @mock.patch.object(oabench, "_thermal_snapshot")
    @mock.patch.object(oabench.time, "sleep")
    def test_thermal_gate_waits_for_matching_sensor(self, sleep, snapshot) -> None:
        snapshot.side_effect = [
            [{"sensor": "coretemp:cpu:Package id 0", "celsius": 75.0}],
            [{"sensor": "coretemp:cpu:Package id 0", "celsius": 69.0}],
        ]
        observed, _, maximum = oabench._wait_for_thermal_limit(
            limit_celsius=70.0,
            sensor_pattern=oabench.re.compile("Package id 0"),
            timeout_seconds=30.0,
        )
        self.assertEqual(maximum, 69.0)
        self.assertEqual(observed[0]["celsius"], 69.0)
        sleep.assert_called_once()

    @mock.patch.object(oabench, "_vulkan_summary", return_value={"available": False})
    @mock.patch.object(oabench, "_power_metadata", return_value={})
    def test_preserves_non_utf8_workload_output(self, _power, _vulkan) -> None:
        output = self.repo / "binary-output.json"
        args = oabench.parse_args(
            (
                "--repo",
                str(self.repo),
                "--output",
                str(output),
                "--name",
                "unit.binary-output",
                "--warmup",
                "0",
                "--runs",
                "7",
                "--cooldown",
                "0",
                "--metric-regex",
                r"metric=([0-9.]+)",
                "--require-regex",
                "correct=yes",
                "--",
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'metric=1.25 correct=yes\\n\\xb0\\n')",
            )
        )

        _, status = oabench.run(args)

        self.assertEqual(status, 0)
        document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(document["result"], "PASS")
        self.assertEqual(document["metric"]["statistics"]["median"], 1.25)
        log = output.with_suffix(output.suffix + ".logs") / "measured-00.stdout.txt"
        self.assertIn(b"\xb0", log.read_bytes())

    def test_comparison_fails_actionable_regression(self) -> None:
        def document(median: float, mad_percent: float) -> dict:
            return {
                "repository": {"commit": "abc"},
                "platform": {
                    "available": True,
                    "system": "Linux",
                    "machine": "x86_64",
                    "device_index": 0,
                    "vendor_id": "0x8086",
                    "device_id": "0x9a49",
                    "device_type": "gpu",
                    "device_name": "gpu",
                    "driver_id": "driver",
                    "driver_name": "driver",
                },
                "build": {
                    "CMAKE_BUILD_TYPE": "Release",
                    "OA_BUILD_SHARED": "ON",
                    "OA_EMBED_SHADERS": "ON",
                },
                "workload": {
                    "name": "work",
                    "contract": {"shape": "1x1"},
                    "command_id": "unit-work-v1",
                },
                "metric": {
                    "name": "time",
                    "statistics": {
                        "unit": "ms",
                        "median": median,
                        "relative_mad_percent": mad_percent,
                    },
                },
            }

        result = oabench._comparison(
            document(104.0, 0.5),
            document(100.0, 0.4),
            threshold_percent=3.0,
            direction="lower",
        )
        self.assertTrue(result["actionable_regression"])
        self.assertEqual(result["result"], "FAIL")

    @mock.patch.object(oabench, "_vulkan_summary", return_value={"available": False})
    @mock.patch.object(oabench, "_power_metadata", return_value={})
    @mock.patch.object(oabench, "_thermal_snapshot", return_value=[])
    def test_rejects_result_above_spread_ceiling(
        self, _thermal, _power, _vulkan
    ) -> None:
        output = self.repo / "spread.json"
        counter = self.repo / "counter.txt"
        script = (
            "import pathlib; p=pathlib.Path(r'" + str(counter) + "'); "
            "n=int(p.read_text())+1 if p.exists() else 0; p.write_text(str(n)); "
            "print('metric=' + ('2.0' if n == 6 else '1.0'))"
        )
        args = oabench.parse_args((
            "--repo", str(self.repo),
            "--output", str(output),
            "--name", "unit.spread",
            "--runs", "7",
            "--cooldown", "0",
            "--max-spread-percent", "10",
            "--metric-regex", r"metric=(?P<value>[0-9.]+)",
            "--", sys.executable, "-c", script,
        ))
        _, status = oabench.run(args)
        self.assertEqual(status, 1)
        document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(document["result"], "FAIL")
        self.assertEqual(
            document["metric"]["statistics"]["spread_percent"], 100.0
        )

    def test_comparison_rejects_identity_mismatch(self) -> None:
        base = {
            "repository": {"commit": "abc"},
            "platform": {"available": True, "device_id": "a"},
            "build": {
                "CMAKE_BUILD_TYPE": "Release",
                "OA_BUILD_SHARED": "ON",
                "OA_EMBED_SHADERS": "ON",
            },
            "workload": {
                "name": "work",
                "contract": {"shape": "1x1"},
                "command_id": "work-v1",
            },
            "metric": {
                "name": "time",
                "statistics": {
                    "unit": "ms",
                    "median": 1.0,
                    "relative_mad_percent": 0.1,
                },
            },
        }
        mutations = (
            ("contract", lambda value: value["workload"].update(contract={})),
            ("command", lambda value: value["workload"].update(command_id="v2")),
            ("platform", lambda value: value.update(platform={"device_id": "b"})),
            ("build", lambda value: value["build"].update(OA_EMBED_SHADERS="OFF")),
        )
        for label, mutate in mutations:
            candidate = json.loads(json.dumps(base))
            mutate(candidate)
            with self.subTest(label=label), self.assertRaises(ValueError):
                oabench._comparison(
                    candidate,
                    base,
                    threshold_percent=3.0,
                    direction="lower",
                )


if __name__ == "__main__":
    unittest.main()
