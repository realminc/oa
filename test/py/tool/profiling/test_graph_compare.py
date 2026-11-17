from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tool/profiling"))
import graph_compare


def report():
	"""Small add -> activation report with a real retained intermediate edge."""
	nodes = []
	for index, (kernel, effects) in enumerate([
		("matrix.add.f32", [(0, "read"), (1, "read"), (2, "write")]),
		("ml.matrix.relu.f32", [(2, "read"), (3, "write")]),
	]):
		nodes.append({
			"index": index, "operation": kernel, "semantic_operations": [index],
			"implementation_id": "0x0000000000000001", "operation_contract_hash": None,
			"problem_contract_hash": None, "kernel_content_hash": "0x0000000000000001",
			"kernel_selection": "direct", "kernel": kernel, "dtype_class": None,
			"dtype": "float32", "physical_write": None, "groups": [1, 1, 1], "queue": "compute",
			"indirect": False,
			"effects": [{"resource": resource, "access": access} for resource, access in effects],
		})
	return {
		"schema": "oa.execution_graph.v3", "name": "forward+backward", "compiled": False,
		"completion": {"submitted": False, "timeline_value": 0},
		"stats": {"dispatches": 2, "barriers": 1, "war_barriers": 0, "buffer_bytes": 64},
		"resources": [
			{"id": i, "hazard_domain": i, "bytes": 16, "first_node": first, "last_node": last}
			for i, (first, last) in enumerate([(0, 0), (0, 0), (0, 1), (1, 1)])
		],
		"alias_groups": [],
		"barriers": [{
			"reason": "read_after_write", "scope": "buffer", "source_nodes": [0, 0],
			"destination_node": 1, "source_resource": 2, "destination_resource": 2,
			"hazard_domain": 2, "range": {"offset": 0, "bytes": 16},
			"source_queue": "compute", "destination_queue": "compute", "cross_queue": False,
			"ownership_transfer": False, "source_stage_mask": "0x0000000000000800",
			"source_stages": ["compute_shader"], "source_access_mask": "0x0000000400000000",
			"source_accesses": ["shader_storage_write"],
			"destination_stage_mask": "0x0000000000000800",
			"destination_stages": ["compute_shader"],
			"destination_access_mask": "0x0000000200000000",
			"destination_accesses": ["shader_storage_read"],
		}],
		"nodes": nodes,
	}


class GraphComparisonTests(unittest.TestCase):
	def test_only_root_runtime_envelope_is_ignored(self):
		before, after = report(), report()
		after.update(
			name="another label", compiled=True,
			completion={"submitted": True, "timeline_value": 42},
		)
		result = graph_compare.compare_reports(before, after)
		self.assertTrue(result["equal"])
		self.assertEqual(result["scope"], "reported_structure_only")
		self.assertEqual(result["differences"], [])
		self.assertEqual(result["kernel_count_delta"], {})
		before["nodes"][0]["compiled"] = False
		after["nodes"][0]["compiled"] = True
		self.assertFalse(graph_compare.compare_reports(before, after)["equal"])

	def test_detects_binding_order_saved_writes_semantics_and_launch_changes(self):
		mutations = [
			(lambda r: r["nodes"][0].update(kernel="ml.linear.f32"), "$.nodes[0].kernel"),
			(lambda r: r["nodes"][0]["effects"].reverse(), "$.nodes[0].effects[0].access"),
			(lambda r: r["nodes"][0]["effects"].pop(), "$.nodes[0].effects.length"),
			(
				lambda r: r["nodes"][0].update(semantic_operations=[5]),
				"$.nodes[0].semantic_operations[0]",
			),
			(lambda r: r["nodes"][0].update(groups=[2, 1, 1]), "$.nodes[0].groups[0]"),
			(
				lambda r: r["nodes"][0].update(implementation_id="0x0000000000000002"),
				"$.nodes[0].implementation_id",
			),
			(
				lambda r: r["nodes"][0].update(physical_write={"workspace": "none"}),
				"$.nodes[0].physical_write",
			),
			(
				lambda r: r["nodes"][0].update(operation_contract_hash="0x0000000000000002"),
				"$.nodes[0].operation_contract_hash",
			),
			(
				lambda r: r["nodes"][0]["effects"][0].update(resource=1),
				"$.nodes[0].effects[0].resource",
			),
		]
		for mutate, path in mutations:
			with self.subTest(path=path):
				after = report()
				mutate(after)
				result = graph_compare.compare_reports(report(), after)
				self.assertFalse(result["equal"])
				self.assertIn(path, [change["path"] for change in result["differences"]])

	def test_hazards_lifetimes_aliases_and_extra_fields_are_compared(self):
		after = report()
		after["resources"][2]["first_node"] = 1
		after["resources"][3]["hazard_domain"] = 2
		after["barriers"][0]["ownership_transfer"] = True
		after["barriers"][0]["source_accesses"] = ["shader_storage_read"]
		after["alias_groups"] = [{"members": [2, 3]}]
		after["new_evidence"] = {"version": 1}
		result = graph_compare.compare_reports(report(), after)
		paths = {change["path"] for change in result["differences"]}
		self.assertTrue({
			"$.resources[2].first_node", "$.resources[3].hazard_domain",
			"$.barriers[0].ownership_transfer", "$.barriers[0].source_accesses[0]",
			"$.alias_groups.length", "$.new_evidence",
		} <= paths)

	def test_histogram_does_not_hide_order_change_and_limit_is_bounded(self):
		after = report()
		for key in ("operation", "kernel"):
			left, right = after["nodes"]
			left[key], right[key] = right[key], left[key]
		result = graph_compare.compare_reports(report(), after, limit=1)
		self.assertFalse(result["equal"])
		self.assertEqual(result["kernel_count_delta"], {})
		self.assertEqual(len(result["differences"]), 1)
		self.assertTrue(result["truncated"])
		self.assertFalse(graph_compare.compare_reports(report(), report(), limit=1)["truncated"])
		after = report()
		after["nodes"][0]["groups"] = [2, 1, 1]
		self.assertFalse(graph_compare.compare_reports(report(), after, limit=1)["truncated"])
		after = report()
		after["nodes"][1]["kernel"] = "ml.linear.f32"
		delta = graph_compare.compare_reports(report(), after)["kernel_count_delta"]
		self.assertEqual(delta, {"ml.linear.f32": 1, "ml.matrix.relu.f32": -1})
		for limit in (0, -1, True):
			with self.assertRaisesRegex(ValueError, "limit"):
				graph_compare.compare_reports(report(), after, limit=limit)

	def test_missing_null_and_boolean_numeric_values_cannot_compare_equal(self):
		left, right = report(), report()
		left["new_evidence"] = None
		self.assertFalse(graph_compare.compare_reports(left, right)["equal"])
		right["new_evidence"] = None
		self.assertTrue(graph_compare.compare_reports(left, right)["equal"])
		left["new_evidence"], right["new_evidence"] = 1, True
		self.assertFalse(graph_compare.compare_reports(left, right)["equal"])

	def test_incomplete_and_invalid_reports_are_errors_even_when_identical(self):
		mutations = [
			lambda r: r.update(schema="oa.execution_graph.v2"),
			lambda r: r.update(nodes=[]),
			lambda r: r.pop("resources"),
			lambda r: r["nodes"][0].pop("effects"),
			lambda r: r["nodes"][0].update(index=2),
			lambda r: r["nodes"][0].update(semantic_operations=[1, 1]),
			lambda r: r["nodes"][0].update(groups=[True, 1, 1]),
			lambda r: r["nodes"][0].update(implementation_id="not a hash"),
			lambda r: r["nodes"][0].update(physical_write=[]),
			lambda r: r["nodes"][0]["effects"][0].update(resource=99),
			lambda r: r["nodes"][0]["effects"][0].update(access="unknown"),
			lambda r: r["resources"][0].update(id=5),
			lambda r: r["resources"][0].update(bytes=True),
			lambda r: r["resources"][0].update(last_node=99),
			lambda r: r["barriers"][0].update(source_nodes=[1, 1]),
			lambda r: r["barriers"][0].update(destination_resource=99),
			lambda r: r["barriers"][0].pop("source_accesses"),
			lambda r: r["stats"].update(dispatches=0),
			lambda r: r["stats"].update(barriers=0),
			lambda r: r["stats"].update(buffer_bytes=0),
			lambda r: r.update(compiled=1),
		]
		for mutate in mutations:
			invalid = report()
			mutate(invalid)
			with self.subTest(invalid=invalid):
				with self.assertRaises(ValueError):
					graph_compare.compare_reports(invalid, copy.deepcopy(invalid))

	def test_strict_json_reader_rejects_duplicates_and_nonfinite_numbers(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "report.json"
			for payload in [
				'{"schema": "wrong", "schema": "oa.execution_graph.v3"}',
				json.dumps(report())[:-1] + ', "extra": NaN}',
				json.dumps(report())[:-1] + ', "extra": Infinity}',
				json.dumps(report())[:-1] + ', "extra": 1e999}',
			]:
				path.write_text(payload)
				with self.assertRaises(ValueError):
					graph_compare.load_report(path)

	def test_cli_routes_and_exit_codes(self):
		with tempfile.TemporaryDirectory() as directory:
			left, right = Path(directory) / "before.json", Path(directory) / "after.json"
			left.write_text(json.dumps(report()))
			right.write_text(json.dumps(report()))
			command = [
				sys.executable, str(ROOT / "tool/oa"), "profile", "graph-compare",
				str(left), str(right), "--json",
			]
			result = subprocess.run(command, capture_output=True, text=True)
			self.assertEqual(result.returncode, 0, result.stderr)
			self.assertTrue(json.loads(result.stdout)["equal"])
			after = report()
			after["nodes"][0]["groups"] = [2, 1, 1]
			right.write_text(json.dumps(after))
			result = subprocess.run(command, capture_output=True, text=True)
			self.assertEqual(result.returncode, 1, result.stderr)
			self.assertFalse(json.loads(result.stdout)["equal"])
			right.write_text("{}")
			result = subprocess.run(command, capture_output=True, text=True)
			self.assertEqual(result.returncode, 2)
			self.assertEqual(result.stdout, "")
			self.assertIn("comparison failed", result.stderr)
			right.unlink()
			self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
