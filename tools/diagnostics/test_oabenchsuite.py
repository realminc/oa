#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import oabenchsuite


class OaBenchSuiteTest(unittest.TestCase):
    def test_checked_in_suite_is_valid_and_unique(self) -> None:
        workloads = oabenchsuite._load_suite(oabenchsuite.DEFAULT_CONFIG)
        self.assertEqual(len(workloads), 8)
        self.assertEqual(len({item["name"] for item in workloads}), 8)

    def test_rejects_duplicate_workload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oa-bench-suite-test-") as directory:
            path = pathlib.Path(directory) / "suite.json"
            item = {"name": "same", "command_id": "v1", "command": ["true"]}
            path.write_text(
                json.dumps({"schema": oabenchsuite.SCHEMA, "workloads": [item, item]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                oabenchsuite._load_suite(path)

    def test_platform_key_is_hardware_and_driver_scoped(self) -> None:
        key = oabenchsuite._platform_key(
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

    def test_roofline_summary_combines_fresh_process_medians(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oa-roofline-summary-") as directory:
            root = pathlib.Path(directory)
            completed = []
            for name, median in (
                (oabenchsuite.ROOFLINE_COPY, 40.0),
                (oabenchsuite.ROOFLINE_COMPUTE, 480.0),
            ):
                result = root / f"{name}.json"
                result.write_text(
                    json.dumps(
                        {"metric": {"statistics": {"median": median}}}
                    ),
                    encoding="utf-8",
                )
                completed.append(({"name": name}, result, root / "unused"))
            output = oabenchsuite._write_roofline_summary(
                completed,
                platform_identity={"device_name": "test"},
                output_dir=root,
            )
            self.assertIsNotNone(output)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["schema"], oabenchsuite.ROOFLINE_SCHEMA)
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
