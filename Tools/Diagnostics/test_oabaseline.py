#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import oabaseline
import oabench


def _result(*, dirty: bool = False, count: int = 7) -> dict:
    return {
        "schema": oabench.SCHEMA,
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
        "build": {"CMAKE_BUILD_TYPE": "Release"},
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
    def test_accepts_clean_passing_result(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oa-baseline-test-") as directory:
            root = pathlib.Path(directory)
            result = root / "result.json"
            output = root / "baseline.json"
            result.write_text(json.dumps(_result()), encoding="utf-8")
            actual = oabaseline.accept(
                result, output, reason="initial measured baseline", accepted_by="test"
            )
            self.assertEqual(actual, output)
            baseline = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(baseline["schema"], oabench.BASELINE_SCHEMA)
            self.assertEqual(baseline["source"]["repository_commit"], "abc123")
            self.assertEqual(baseline["workload"]["command_id"], "unit-work-v1")

    def test_rejects_dirty_or_under_sampled_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "dirty"):
            oabaseline._validate(_result(dirty=True))
        with self.assertRaisesRegex(ValueError, "seven"):
            oabaseline._validate(_result(count=6))


if __name__ == "__main__":
    unittest.main()
