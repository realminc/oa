#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import oaBenchSuite


class OaBenchSuiteTest(unittest.TestCase):
    def testCheckedInSuiteIsValidAndUnique(self) -> None:
        workloads = oaBenchSuite._loadSuite(oaBenchSuite.DEFAULT_CONFIG)
        self.assertEqual(len(workloads), 13)
        self.assertEqual(len({item["name"] for item in workloads}), 13)

    def testRejectsDuplicateWorkload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oa-bench-suite-test-") as directory:
            path = pathlib.Path(directory) / "suite.json"
            item = {"name": "same", "command_id": "v1", "command": ["true"]}
            path.write_text(
                json.dumps({"schema": oaBenchSuite.SCHEMA, "workloads": [item, item]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                oaBenchSuite._loadSuite(path)

    def testPlatformKeyIsHardwareAndDriverScoped(self) -> None:
        key = oaBenchSuite._platformKey(
            {
                "available": True,
                "system": "Linux",
                "machine": "x86_64",
                "vendor_id": "0x8086",
                "device_id": "0x9a49",
                "driver_id": "DRIVER_ID_INTEL_OPEN_SOURCE_MESA",
            }
        )
        self.assertEqual(
            key, "linux-x86-64-0x8086-0x9a49-driver-id-intel-open-source-mesa"
        )

    def testResultSummaryHandlesCorrectnessRejectionBeforeMeasuredSamples(self) -> None:
        summary = oaBenchSuite._resultSummary(
            "video.paced",
            {
                "metric": {"statistics": None},
                "result": "FAIL",
            },
        )
        self.assertEqual(
            summary,
            "video.paced: metric statistics unavailable result=FAIL",
        )

    def testRooflineSummaryCombinesFreshProcessMedians(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oa-roofline-summary-") as directory:
            root = pathlib.Path(directory)
            completed = []
            for name, median in (
                (oaBenchSuite.ROOFLINE_COPY, 40.0),
                (oaBenchSuite.ROOFLINE_COMPUTE, 480.0),
            ):
                result = root / f"{name}.json"
                result.write_text(
                    json.dumps(
                        {"metric": {"statistics": {"median": median}}}
                    ),
                    encoding="utf-8",
                )
                completed.append(({"name": name}, result, root / "unused"))
            output = oaBenchSuite._writeRooflineSummary(
                completed,
                platformIdentity={"device_name": "test"},
                outputDir=root,
            )
            self.assertIsNotNone(output)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["schema"], oaBenchSuite.ROOFLINE_SCHEMA)
            self.assertEqual(
                document["ceilings"]["ridge_point_flop_per_algorithmic_byte"],
                12.0,
            )
            self.assertEqual(
                document["interpretation"]["hardware_counter_bandwidth"],
                "unmeasured",
            )


if __name__ == "__main__":
    unittest.main()
