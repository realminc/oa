#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import oaBaseline
import oaBench


def _result(*, dirty: bool = False, count: int = 7) -> dict:
    return {
        "schema": oaBench.SCHEMA,
        "result": "PASS",
        "repository": {"commit": "abc123", "dirty": dirty},
        "platform": {
            "available": True,
            "system": "Linux",
            "machine": "x86_64",
            "device_index": 0,
            "vendor_id": "0x8086",
            "device_id": "0x9a49",
            "device_type": "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
            "device_name": "Test GPU",
            "driver_id": "DRIVER_ID_TEST",
            "driver_name": "Test Driver",
        },
        "host": {
            "thermal": [{
                "sensor": "coretemp:Package id 0",
                "minimum_celsius": 50.0,
                "maximum_celsius": 65.0,
            }]
        },
        "build": {
            "CMAKE_BUILD_TYPE": "Release",
            "VCPKG_INSTALLED_DIR": str(pathlib.Path.home() / ".vcpkg/oa/release"),
        },
        "workload": {
            "name": "unit.work",
            "command_id": "unit-work-v1",
            "contract": {"shape": "1x1"},
        },
        "metric": {
            "name": "time",
            "statistics": {"count": count, "unit": "ms", "median": 1.0},
        },
    }


class OaBaselineTest(unittest.TestCase):
    def testAcceptsCleanPassingResult(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oa-baseline-test-") as directory:
            root = pathlib.Path(directory)
            result = root / "result.json"
            output = root / "baseline.json"
            result.write_text(json.dumps(_result()), encoding="utf-8")
            actual = oaBaseline.accept(
                result, output, reason="initial measured baseline", acceptedBy="test"
            )
            self.assertEqual(actual, output)
            baseline = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(baseline["schema"], oaBench.BASELINE_SCHEMA)
            self.assertEqual(baseline["source"]["repository_commit"], "abc123")
            self.assertEqual(baseline["workload"]["command_id"], "unit-work-v1")
            self.assertEqual(
                baseline["build"]["VCPKG_INSTALLED_DIR"],
                "~/.vcpkg/oa/release",
            )
            self.assertEqual(
                baseline["environment"]["thermal"][0]["maximum_celsius"],
                65.0,
            )

    def testRejectsDirtyOrUnderSampledResult(self) -> None:
        with self.assertRaisesRegex(ValueError, "dirty"):
            oaBaseline._validate(_result(dirty=True))
        with self.assertRaisesRegex(ValueError, "seven"):
            oaBaseline._validate(_result(count=6))


if __name__ == "__main__":
    unittest.main()
