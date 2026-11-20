#!/usr/bin/env python3
"""Compare reported execution structure across compute-compiler checkpoints.

Consumes existing oa.execution_graph.v3 artifacts; never builds or runs OA.
Report equality is only a structural gate: push payloads, actual dynamic/indirect
dispatch details, semantic-value bindings and numerical results are not present
in this report. Those still require the region contract's independent proofs.

Donor audit: oacpp/tools/diagnostics/oaEvidence.py preserves and schema-checks
graph artifacts but has no structural comparator. This adapter consumes the
Rust ExecutionPlan::debug_report_json contract without changing its schema.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


GRAPH_SCHEMA = "oa.execution_graph.v3"
COMPARISON_SCHEMA = "oa.execution_graph_comparison.v1"
IGNORED_FIELDS = ("name", "compiled", "completion")
HASH = re.compile(r"0x[0-9a-f]{16}\Z")


def require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def integer(value: Any) -> bool:
	return type(value) is int and value >= 0


def record(value: Any, fields: dict[str, type], path: str) -> None:
	require(isinstance(value, dict), f"{path} must be an object")
	for key, expected in fields.items():
		require(
			key in value and type(value[key]) is expected,
			f"{path}.{key} must be {expected.__name__}",
		)


def validate_report(report: Any) -> None:
	record(report, {
		"schema": str, "name": str, "compiled": bool, "completion": dict,
		"stats": dict, "resources": list, "alias_groups": list, "barriers": list, "nodes": list,
	}, "$")
	require(report["schema"] == GRAPH_SCHEMA, f"unsupported graph schema: {report['schema']}")
	record(report["completion"], {"submitted": bool, "timeline_value": int}, "$.completion")
	require(integer(report["completion"]["timeline_value"]), "invalid completion timeline value")
	nodes, resources, barriers = report["nodes"], report["resources"], report["barriers"]
	require(bool(nodes), "graph report has no nodes; no comparison evidence")
	for key in ("dispatches", "barriers", "buffer_bytes"):
		require(integer(report["stats"].get(key)), f"$.stats.{key} must be a nonnegative integer")
	copies = report["stats"].get("copies", 0)
	require(integer(copies), "$.stats.copies must be a nonnegative integer")
	video_commands = report["stats"].get("video_commands", 0)
	require(integer(video_commands), "$.stats.video_commands must be a nonnegative integer")
	image_transfers = report["stats"].get("image_transfers", 0)
	require(integer(image_transfers), "$.stats.image_transfers must be a nonnegative integer")
	require(
		report["stats"]["dispatches"] + copies + video_commands == len(nodes),
		"command count does not match nodes",
	)
	require(report["stats"]["barriers"] == len(barriers), "barrier count does not match barriers")
	for index, resource in enumerate(resources):
		path = f"$.resources[{index}]"
		record(resource, {
			key: int for key in ("id", "hazard_domain", "bytes", "first_node", "last_node")
		}, path)
		require(resource["id"] == index, f"{path}.id must follow normalized resource order")
		require(
			integer(resource["hazard_domain"]) and integer(resource["bytes"]),
			f"{path} has invalid domain/size",
		)
		require(
			0 <= resource["first_node"] <= resource["last_node"] < len(nodes),
			f"{path} has an invalid lifetime",
		)
	require(
		report["stats"]["buffer_bytes"] == sum(r["bytes"] for r in resources),
		"buffer byte count does not match resources",
	)
	copy_nodes = 0
	video_nodes = 0
	handoff_nodes = 0
	for index, node in enumerate(nodes):
		path = f"$.nodes[{index}]"
		record(node, {
			"index": int, "operation": str, "semantic_operations": list,
			"implementation_id": str, "kernel_content_hash": str, "kernel_selection": str,
			"kernel": str, "dtype": str, "groups": list, "queue": str,
			"indirect": bool, "effects": list,
		}, path)
		for key in (
			"operation_contract_hash", "problem_contract_hash", "dtype_class", "physical_write"
		):
			require(key in node, f"{path}.{key} is missing")
		for key in (
			"implementation_id", "kernel_content_hash",
			"operation_contract_hash", "problem_contract_hash",
		):
			value = node[key]
			require(
				(key.endswith("contract_hash") and value is None)
				or isinstance(value, str) and HASH.fullmatch(value) is not None,
				f"{path}.{key} must be a 64-bit hex hash (or null for contracts)",
			)
		require(
			node["physical_write"] is None or isinstance(node["physical_write"], dict),
			f"{path}.physical_write must be an object or null",
		)
		require(node["index"] == index, f"{path}.index must follow execution order")
		kind = node.get("kind", "compute")
		require(kind in ("compute", "buffer_copy", "video_plane_command", "video_decode"), f"{path}.kind is unsupported")
		if kind == "buffer_copy":
			copy_nodes += 1
			require(
				integer(node.get("copy_bytes")) and node["copy_bytes"] > 0,
				f"{path}.copy_bytes must be positive",
			)
		if kind == "video_decode":
			video_nodes += 1
			picture = node.get("decode_picture")
			fixed_words, reference_words = {
				"video.decode_h264": (14, 4),
				"video.decode_h265": (13, 2),
				"video.decode_av1": (18, 2),
				"video.decode_vp9": (14, 1),
			}.get(node["kernel"], (0, 1))
			require(
				fixed_words > 0 and node["queue"] == "video_decode"
				and isinstance(picture, list) and len(picture) >= fixed_words
				and (len(picture) - fixed_words) % reference_words == 0
				and all(integer(v) and 0 <= v <= 0xFFFFFFFF for v in picture),
				f"{path}.decode_picture is invalid",
			)
		if kind == "video_plane_command":
			video_nodes += 1
			geometry = node.get("image_geometry")
			require(
				isinstance(geometry, list)
				and len(geometry) in (14, 19)
				and all(integer(v) for v in geometry),
				f"{path}.image_geometry is invalid",
			)
			phase = node["kernel"]
			transition = phase in ("video.plane_release", "video.plane_acquire")
			require(transition or phase == "video.copy_planes", f"{path} has an invalid video command")
			handoff_nodes += int(transition)
			require(
				integer(node.get("family")) and node["family"] == geometry[2 if transition else 3],
				f"{path}.family does not agree with plane ownership",
			)
		require(
			bool(node["kernel"]) and bool(node["operation"]), f"{path} has an empty kernel/operation"
		)
		require(
			len(node["groups"]) == 3 and all(integer(v) for v in node["groups"]),
			f"{path}.groups must have three nonnegative integers",
		)
		owners = node["semantic_operations"]
		require(
			all(integer(v) for v in owners) and len(owners) == len(set(owners)),
			f"{path} has invalid semantic owners",
		)
		for binding, effect in enumerate(node["effects"]):
			label = f"{path}.effects[{binding}]"
			record(effect, {"resource": int, "access": str}, label)
			require(
				0 <= effect["resource"] < len(resources), f"{label} references an unknown resource"
			)
			require(
				effect["access"] in {"read", "write", "read_write"}, f"{label} has invalid access"
			)
	require(copies == copy_nodes, "copy count does not match nodes")
	require(video_commands == video_nodes, "video command count does not match nodes")
	require(image_transfers == handoff_nodes, "image transfer count does not match nodes")
	for index, barrier in enumerate(barriers):
		path = f"$.barriers[{index}]"
		record(barrier, {
			"reason": str, "scope": str, "source_nodes": list, "destination_node": int,
			"source_resource": int, "destination_resource": int,
			"hazard_domain": int, "range": dict,
			"source_queue": str, "destination_queue": str,
			"cross_queue": bool, "ownership_transfer": bool,
			"source_stage_mask": str, "source_stages": list,
			"source_access_mask": str, "source_accesses": list,
			"destination_stage_mask": str, "destination_stages": list,
			"destination_access_mask": str, "destination_accesses": list,
		}, path)
		require(0 <= barrier["destination_node"] < len(nodes), f"{path} has invalid destination")
		sources = barrier["source_nodes"]
		require(
			len(sources) == 2
			and all(integer(v) and v < barrier["destination_node"] for v in sources)
			and sources[0] <= sources[1], f"{path} has invalid source range",
		)
		for key in ("source_resource", "destination_resource"):
			require(
				0 <= barrier[key] < len(resources), f"{path}.{key} references an unknown resource"
			)
		record(barrier["range"], {"offset": int, "bytes": int}, f"{path}.range")
		require(
			all(integer(v) for v in barrier["range"].values()), f"{path} has invalid barrier range"
		)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
	result: dict[str, Any] = {}
	for key, value in pairs:
		require(key not in result, f"duplicate JSON field: {key}")
		result[key] = value
	return result


def invalid_constant(token: str) -> None:
	raise ValueError(f"non-finite JSON number: {token}")


def finite_float(token: str) -> float:
	value = float(token)
	require(math.isfinite(value), f"non-finite JSON number: {token}")
	return value


def load_report(path: Path) -> dict[str, Any]:
	report = json.loads(
		path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
		parse_constant=invalid_constant, parse_float=finite_float,
	)
	validate_report(report)
	return report


def differences(before: Any, after: Any, path: str = "$"):
	"""Yield exact ordered-field differences; array order is load-bearing."""
	if type(before) is not type(after):
		yield {
			"path": path, "kind": "type",
			"before_type": type(before).__name__, "after_type": type(after).__name__,
		}
	elif isinstance(before, dict):
		for key in sorted(before.keys() | after.keys()):
			field = f"{path}.{key}"
			if key not in before or key not in after:
				yield {
					"path": field, "kind": "field",
					"before_present": key in before, "after_present": key in after,
				}
			else:
				yield from differences(before[key], after[key], field)
	elif isinstance(before, list):
		if len(before) != len(after):
			yield {
				"path": f"{path}.length", "kind": "length",
				"before": len(before), "after": len(after),
			}
		for index, (left, right) in enumerate(zip(before, after)):
			yield from differences(left, right, f"{path}[{index}]")
	elif before != after:
		yield {"path": path, "kind": "value", "before": before, "after": after}


def compare_reports(
	before: dict[str, Any],
	after: dict[str, Any],
	limit: int = 20,
) -> dict[str, Any]:
	validate_report(before)
	validate_report(after)
	require(type(limit) is int and limit > 0, "difference limit must be positive")
	left = {key: value for key, value in before.items() if key not in IGNORED_FIELDS}
	right = {key: value for key, value in after.items() if key not in IGNORED_FIELDS}
	changes = []
	truncated = False
	for change in differences(left, right):
		if len(changes) == limit:
			truncated = True
			break
		changes.append(change)
	old_kernels = Counter(node["kernel"] for node in before["nodes"])
	new_kernels = Counter(node["kernel"] for node in after["nodes"])
	return {
		"schema": COMPARISON_SCHEMA,
		"scope": "reported_structure_only",
		"equal": not changes,
		"ignored_root_fields": list(IGNORED_FIELDS),
		"nodes": {"before": len(before["nodes"]), "after": len(after["nodes"])},
		"resources": {"before": len(before["resources"]), "after": len(after["resources"])},
		"kernel_count_delta": {
			key: new_kernels[key] - old_kernels[key]
			for key in sorted(old_kernels.keys() | new_kernels.keys())
			if new_kernels[key] != old_kernels[key]
		},
		"differences": changes,
		"truncated": truncated,
	}


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("before", type=Path)
	parser.add_argument("after", type=Path)
	parser.add_argument("--json", action="store_true", help="emit a machine-readable comparison")
	parser.add_argument("--max-differences", type=int, default=20)
	args = parser.parse_args(argv)
	try:
		result = compare_reports(
			load_report(args.before), load_report(args.after), args.max_differences
		)
	except (OSError, ValueError, RecursionError) as error:
		print(f"graph comparison failed: {error}", file=sys.stderr)
		return 2
	if args.json:
		print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
	else:
		print("Reported structure matches." if result["equal"] else "Reported structure differs.")
		for change in result["differences"]:
			print(json.dumps(change, sort_keys=True, allow_nan=False))
		if result["truncated"]:
			print("Further differences omitted; increase --max-differences.")
	return 0 if result["equal"] else 1


if __name__ == "__main__":
	raise SystemExit(main())
