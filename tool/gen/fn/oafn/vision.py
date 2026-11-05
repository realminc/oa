"""Vision detection schema validation and emitters."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .model import validate_physical_write, validate_semantic_attributes, validate_unique
from .wire import generate_type, render_dispatch_body, validate_dispatch_generation
from .signature import operation_signature, signature_header
from .util import IDENTIFIER, banner, require

def validate_vision_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Vision schema_version must be 1")
	require(schema.get("family") == "vision_detection", "Vision family must be vision_detection")
	require(schema.get("domain") == "vision", "Vision domain must be vision")
	require(schema.get("dtype") == "f32", "Vision dtype must be f32")
	contracts = schema.get("contracts")
	require(isinstance(contracts, list) and len(contracts) == 6, "Vision contracts must contain the complete detection surface")
	expected_contracts = {
		"box_iou": (["matrix", "matrix"], ["matrix"], "f32"),
		"nms": (["matrix", "matrix", "matrix"], ["matrix", "matrix"], "explicit"),
		"confusion_matrix": (["matrix", "matrix"], ["matrix"], "explicit"),
		"binary_mask_counts": (["matrix", "matrix"], ["matrix"], "explicit"),
		"evaluate": (["matrix"] * 8, ["matrix"] * 4, "explicit"),
		"evaluate_segmentation": (["matrix", "matrix"], ["matrix"] * 4, "explicit"),
	}
	require(
		[contract.get("name") for contract in contracts] == list(expected_contracts),
		"Vision contracts are not in canonical detection order",
	)
	contract_names: set[str] = set()
	for index, contract in enumerate(contracts):
		where = f"Vision contracts[{index}]"
		require(isinstance(contract, dict), f"{where} must be an object")
		require(
			(set(contract) - {"python", "donor_op", "control_flow"})
			== {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes", "rust"},
			f"{where} fields are incomplete or unknown",
		)
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		expected_inputs, expected_outputs, expected_dtype = expected_contracts[name]
		require(contract["input_kinds"] == expected_inputs, f"{where}.input_kinds are invalid")
		require(contract["output_kinds"] == expected_outputs, f"{where}.output_kinds are invalid")
		require(contract["shape_rule"] == "explicit", f"{where}.shape_rule must be explicit")
		require(contract["dtype_rule"] == expected_dtype, f"{where}.dtype_rule is invalid")
		validate_semantic_attributes(contract["attributes"], where)
	kernels = schema.get("kernels")
	require(isinstance(kernels, list) and kernels, "Vision kernels must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernel_ids: set[str] = set()
	seen_sources: set[str] = set()
	for index, kernel in enumerate(kernels):
		where = f"Vision kernels[{index}]"
		require(isinstance(kernel, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernel_ids), ("source", seen_sources)):
			value = kernel.get(key)
			require(isinstance(value, str) and value, f"{where}.{key} must be non-empty")
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = kernel.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id is invalid")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		require(kernel.get("semantic_operation") in contract_names, f"{where}.semantic_operation is unknown")
		require(kernel.get("dtype", schema["dtype"]) in {"f32", "i32", "u8", "u32"}, f"{where}.dtype is unsupported")
		workgroup = kernel.get("workgroup_size")
		require(
			isinstance(workgroup, list)
			and len(workgroup) == 3
			and all(isinstance(value, int) and value > 0 for value in workgroup)
			and math.prod(workgroup) <= 1024,
			f"{where}.workgroup_size is invalid",
		)
		tile = kernel.get("dispatch_tile_size", workgroup)
		require(
			isinstance(tile, list)
			and len(tile) == 3
			and all(isinstance(value, int) and value > 0 for value in tile),
			f"{where}.dispatch_tile_size is invalid",
		)
		validate_physical_write(kernel.get("physical_write"), where, required=True)
		fields = kernel.get("push_fields")
		require(isinstance(fields, list) and fields, f"{where}.push_fields must be non-empty")
		seen_fields: set[str] = set()
		for field in fields:
			require(isinstance(field, list) and len(field) == 2, f"{where}.push_fields entry is invalid")
			name, scalar_type = field
			require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.push field name is invalid")
			validate_unique(name, seen_fields, f"{where}.push field name")
			require(scalar_type in {"uint32", "float32"}, f"{where}.push field type is unsupported")
		require(len(fields) * 4 <= 128, f"{where}.push constants exceed Vulkan minimum guarantee")

	validate_vision_generation(schema)


def vision_templates() -> dict[str, str]:
	path = FN_ROOT / "template/vision/detection.rs.in"
	sections = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", path.read_text(encoding="utf-8"), re.S)
	require(len(sections) == len({name for name, _ in sections}), "duplicate Vision template section")
	return dict(sections)


def validate_vision_generation(schema: dict[str, Any]) -> None:
	validate_dispatch_generation(
		schema, {"detection"}, {"&Matrix", "NmsConfig", "i32", "f32"},
		{"box_iou": "Matrix", "nms": "NmsResult", "confusion_matrix": "Matrix", "binary_mask_counts": "Matrix", "evaluate": "DetectionMetricsResult", "evaluate_segmentation": "SegmentationMetricsResult"},
		{"float": {"Float"}, "signed_integer": {"SignedInteger"}, "boolean": {"Boolean"}},
		read_write=True,
	)
	types = {record["name"]: record for record in schema["types"]}
	for op in schema["contracts"]:
		rust = op["rust"]
		for _, kind in rust["parameters"]:
			require(kind in {"&Matrix", "i32", "f32"} or kind in types, f"Vision {op['name']} parameter type has no owner")
		result = rust["returns"]
		if result == "Matrix":
			require(len(op["output_kinds"]) == 1, f"Vision {op['name']} output differs from return type")
		else:
			require(result in types and [field["type"] for field in types[result]["fields"]] == ["Matrix"] * len(op["output_kinds"]), f"Vision {op['name']} result fields differ from outputs")
	templates = vision_templates()
	require(set(templates) == {op["name"] for op in schema["contracts"]}, "Vision template coverage differs from schema")
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	for op in schema["contracts"]:
		render_dispatch_body(op, kernels, templates[op["name"]], "vision")


def generate_vision_api(schema: dict[str, Any], schema_hash: str) -> str:
	module = (FN_ROOT / "template/vision/detection_module.rs.in").read_text(encoding="utf-8")
	require(module.count("{{items}}") == 1, "Vision module requires exactly one item insertion")
	items = []
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	templates = vision_templates()
	for op in schema["contracts"]:
		rust = op["rust"]
		signature = operation_signature("vision", op, f"Vision {op['name']}")
		require(signature is not None, f"Vision {op['name']} has no public signature")
		items.extend(generate_type(record) for record in schema["types"] if record["before"] == op["name"])
		docs = "\n".join("///" + (" " + line if line else "") for line in rust["doc"])
		annotations = "".join(f"#[{annotation}]\n" for annotation in rust["annotations"])
		body = render_dispatch_body(op, kernels, templates[op["name"]], "vision")
		items.append(f"{docs}\n{annotations}{signature_header(signature)} {{\n{body}\n}}\n")
	return banner(schema_hash, "//", "vision_detection.json") + module.replace("{{items}}", "\n\n".join(items))
