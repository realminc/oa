import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools/profiling"))
import memory_rust


HEADER = b"operation,size_bytes,src_offset,dst_offset,implementation,iterations,median_ns,p10_ns,p90_ns,GB_per_s\n"
ROWS = b"copy,512,0,0,rust_std,10,8,7,9,64\ncopy,512,0,0,oa_rust,10,4,3,5,128\n"


class MemoryRustTests(unittest.TestCase):
	def test_pairs_ratios_inside_each_process(self):
		rows = memory_rust.summarize([HEADER + ROWS] * 7, 7)
		oa = next(row for row in rows if row["implementation"] == "oa_rust")
		self.assertEqual(oa["raw_paired_speedup"], [2.0] * 7)

	def test_rejects_missing_and_duplicate_policy_rows(self):
		with self.assertRaisesRegex(ValueError, "changed"):
			memory_rust.summarize([HEADER + ROWS, HEADER + ROWS.splitlines(keepends=True)[0]], 2)
		with self.assertRaisesRegex(ValueError, "duplicate"):
			memory_rust.summarize([HEADER + ROWS + ROWS], 1)

	def test_rejects_nonfinite_measurement_and_incomplete_runs(self):
		with self.assertRaisesRegex(ValueError, "non-finite"):
			memory_rust.summarize([HEADER + ROWS.replace(b",4,", b",nan,")], 1)
		with self.assertRaisesRegex(ValueError, "incomplete"):
			memory_rust.summarize([HEADER + ROWS], 7)
