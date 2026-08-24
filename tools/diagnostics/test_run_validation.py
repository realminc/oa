#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import tempfile
import unittest

import run_validation


class RunValidationTest(unittest.TestCase):
    def test_identifies_known_same_invocation_vvl_false_positive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oa-validation-test-") as name:
            bundle = pathlib.Path(name)
            logs = bundle / "logs"
            logs.mkdir()
            (logs / "workload.stdout.txt").write_text(
                """Validation Error: [ SharedMemoryDataRace-RaceOnStore ]
vkCmdDispatch(): A data race was detected on the shared memory variable
\"g_buf\" in local invocation index 64 while performing a store operation.
(Likely against local invocation index 64)
The other access in this race was at:
SPIR-V Instruction: %158 = OpLoad %8 %140
SPIR-V Instruction: OpStore %140 %159
""",
                encoding="utf-8",
            )
            affected = {
                "name": "VK_LAYER_KHRONOS_validation",
                "api_version": "1.4.350",
                "implementation_version": 1,
            }
            fixed = dict(affected, api_version="1.4.354")
            self.assertTrue(
                run_validation._known_vvl_shared_rmw_false_positive(
                    bundle, affected
                )
            )
            self.assertFalse(
                run_validation._known_vvl_shared_rmw_false_positive(bundle, fixed)
            )

    def test_does_not_hide_a_cross_invocation_race(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oa-validation-test-") as name:
            bundle = pathlib.Path(name)
            logs = bundle / "logs"
            logs.mkdir()
            (logs / "workload.stdout.txt").write_text(
                """Validation Error: [ SharedMemoryDataRace-RaceOnStore ]
local invocation index 64 while performing a store operation.
(Likely against local invocation index 65)
SPIR-V Instruction: %158 = OpLoad %8 %140
SPIR-V Instruction: OpStore %140 %159
""",
                encoding="utf-8",
            )
            layer = {
                "api_version": "1.4.350",
                "implementation_version": 1,
            }
            self.assertFalse(
                run_validation._known_vvl_shared_rmw_false_positive(bundle, layer)
            )


if __name__ == "__main__":
    unittest.main()
