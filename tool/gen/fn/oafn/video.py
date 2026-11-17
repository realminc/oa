"""Video schema validation for handwritten domain operations."""

from __future__ import annotations

import math
from typing import Any

from .model import validate_physical_write, validate_semantic_attributes, validate_unique
from .util import IDENTIFIER, require

_CONTRACT_FIELDS = {
	"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule",
	"effects", "differentiation", "lowering", "attributes",
	"variadic_input", "variadic_output", "optional_inputs", "control_flow",
	"mutated_inputs", "output_alias_inputs", "donor_op",
}
_KERNEL_FIELDS = {
	"stable_id", "registered", "name", "kernel_id", "source", "variant",
	"reflection_name", "dtype", "workgroup_size", "dispatch_tile_size",
	"semantic_operation", "physical_write", "push_fields", "training_replay_role",
}
_KINDS = {"matrix", "parameter", "audio", "image", "video_frame", "quant_matrix", "texture"}


def _validate_variadic(value: Any, where: str) -> None:
	require(
		isinstance(value, dict) and set(value) == {"kind", "minimum"},
		f"{where} must be {{kind, minimum}}",
	)
	require(value["kind"] in _KINDS, f"{where}.kind is unsupported")
	require(
		isinstance(value["minimum"], int) and value["minimum"] > 0,
		f"{where}.minimum must be a positive integer",
	)


def validate_video_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Video schema_version must be 1")
	require(schema.get("family") == "video", "Video family must be video")
	require(schema.get("domain") == "video", "Video domain must be video")
	contracts = schema.get("contracts", [])
	require(isinstance(contracts, list), "Video contracts must be an array")
	contract_names: set[str] = set()
	for index, contract in enumerate(contracts):
		where = f"Video contracts[{index}]"
		require(isinstance(contract, dict), f"{where} must be an object")
		require(
			set(contract) <= _CONTRACT_FIELDS,
			f"{where} contains unknown fields: {sorted(set(contract) - _CONTRACT_FIELDS)}",
		)
		for required_key in ("name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes"):
			require(required_key in contract, f"{where}.{required_key} is missing")
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		require(
			isinstance(contract["input_kinds"], list)
			and all(kind in _KINDS for kind in contract["input_kinds"]),
			f"{where}.input_kinds contains an unknown kind",
		)
		require(
			isinstance(contract["output_kinds"], list)
			and all(kind in _KINDS for kind in contract["output_kinds"]),
			f"{where}.output_kinds contains an unknown kind",
		)
		require(
			contract["input_kinds"] or "variadic_input" in contract,
			f"{where} needs input_kinds or a variadic input",
		)
		require(contract["output_kinds"], f"{where}.output_kinds must be non-empty")
		if "variadic_input" in contract:
			_validate_variadic(contract["variadic_input"], f"{where}.variadic_input")
		if "variadic_output" in contract:
			_validate_variadic(contract["variadic_output"], f"{where}.variadic_output")
		require(
			len(contract["input_kinds"]) + len(contract["output_kinds"]) <= 16,
			f"{where} fixed arity exceeds OperationContract capacity; use a typed variadic tail",
		)
		validate_semantic_attributes(contract["attributes"], where)
		require(
			len(contract["attributes"]) <= 8,
			f"{where}.attributes exceeds OperationContract::MAX_ATTRIBUTES",
		)
	kernels = schema.get("kernels")
	require(isinstance(kernels, list) and kernels, "Video kernels must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernel_ids: set[str] = set()
	seen_sources: set[str] = set()
	for index, kernel in enumerate(kernels):
		where = f"Video kernels[{index}]"
		require(isinstance(kernel, dict), f"{where} must be an object")
		require(
			set(kernel) <= _KERNEL_FIELDS,
			f"{where} contains unknown fields: {sorted(set(kernel) - _KERNEL_FIELDS)}",
		)
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernel_ids), ("source", seen_sources)):
			value = kernel.get(key)
			require(isinstance(value, str) and value, f"{where}.{key} must be non-empty")
			validate_unique(value, seen, f"{where}.{key}")
		require(
			kernel["source"].startswith("src/slang/video/") and kernel["source"].endswith(".slang"),
			f"{where}.source must be owned by video",
		)
		stable_id = kernel.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id is invalid")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		registered = kernel.get("registered", True)
		require(isinstance(registered, bool), f"{where}.registered must be a flag")
		if not registered:
			continue
		semantic = kernel.get("semantic_operation")
		require(
			semantic is None or semantic in contract_names,
			f"{where}.semantic_operation is unknown",
		)
		require(
			kernel.get("dtype", "u8") in {"u8", "u16", "u32", "i32", "f32", "f16"},
			f"{where}.dtype is unsupported",
		)
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
		validate_physical_write(kernel.get("physical_write"), where, required=semantic is not None)
		fields = kernel.get("push_fields")
		require(isinstance(fields, list) and fields, f"{where}.push_fields must be non-empty")
		seen_fields: set[str] = set()
		for field in fields:
			require(isinstance(field, list) and len(field) == 2, f"{where}.push_fields entry is invalid")
			field_name, scalar_type = field
			require(
				isinstance(field_name, str) and IDENTIFIER.fullmatch(field_name) is not None,
				f"{where}.push field name is invalid",
			)
			validate_unique(field_name, seen_fields, f"{where}.push field name")
			require(scalar_type in {"uint32", "float32"}, f"{where}.push field type is unsupported")
		require(len(fields) * 4 <= 128, f"{where}.push constants exceed Vulkan minimum guarantee")
