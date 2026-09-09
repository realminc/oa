#!/usr/bin/env python3
"""Generate schema-owned Rust operation surfaces and kernel metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


GENERATOR_VERSION = 18
DEFAULT_SCHEMA = Path("tools/gen/fn/schema/matrix_elemwise.json")
DEFAULT_BLAS_SCHEMA = Path("tools/gen/fn/schema/matrix_blas.json")
DEFAULT_ML_SCHEMA = Path("tools/gen/fn/schema/ml_training.json")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
KINDS = {"binary", "unary", "unary_scalar"}
DTYPES = {
	"f32": {
		"rust_type": "f32",
		"rust_dtype": "DType::F32",
		"slang_type": "float",
		"load": "load_f32",
		"store": "store_f32",
	},
	"i32": {
		"rust_type": "i32",
		"rust_dtype": "DType::I32",
		"slang_type": "int",
		"load": "load_i32",
		"store": "store_i32",
	},
}


class SchemaError(ValueError):
	"""The operation schema is structurally invalid.
	"""


def load_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_schema)


def load_blas_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_blas_schema)


def load_ml_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_ml_schema)


def load_validated_schema(path: Path, validator: Any) -> tuple[dict[str, Any], str]:
	raw = path.read_bytes()
	try:
		schema = json.loads(raw)
	except json.JSONDecodeError as error:
		raise SchemaError(f"invalid JSON: {error}") from error
	if not isinstance(schema, dict):
		raise SchemaError("schema root must be an object")
	validator(schema)
	return schema, hashlib.sha256(raw).hexdigest()


def validate_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "schema_version must be 1")
	require(schema.get("family") == "matrix_elemwise", "family must be matrix_elemwise")
	require(schema.get("domain") == "matrix", "domain must be matrix")
	require(schema.get("dtype") == "f32", "dtype must be f32")
	require(schema.get("workgroup_size") == [256, 1, 1], "workgroup_size must be [256, 1, 1]")
	contracts = schema.get("contracts")
	require(isinstance(contracts, dict) and set(contracts) == KINDS, f"contracts must define exactly {sorted(KINDS)}")
	for kind in sorted(KINDS):
		validate_contract(contracts[kind], kind)
	operations = schema.get("operations")
	require(isinstance(operations, list) and operations, "operations must be a non-empty array")

	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	seen_kernel_names: set[str] = set()
	seen_sources: set[str] = set()
	for index, operation in enumerate(operations):
		where = f"operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		name = operation.get("name")
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name must be an identifier")
		validate_unique(name, seen_names, f"{where}.name")
		kind = operation.get("kind")
		require(kind in KINDS, f"{where}.kind must be one of {sorted(KINDS)}")
		require(
			operation.get("differentiation", "none") in {"none", "reverse"},
			f"{where}.differentiation is unsupported",
		)
		validate_dnn_role(operation.get("dnn"), where)
		require(isinstance(operation.get("doc"), str) and operation["doc"].endswith("."), f"{where}.doc must end with a period")
		require(isinstance(operation.get("expression"), str) and operation["expression"].strip(), f"{where}.expression must be non-empty")
		if kind == "unary_scalar":
			scalar_name = operation.get("scalar_name")
			require(isinstance(scalar_name, str) and IDENTIFIER.fullmatch(scalar_name) is not None, f"{where}.scalar_name must be an identifier")
		else:
			require("scalar_name" not in operation, f"{where}.scalar_name is valid only for unary_scalar")
		seen_dtypes: set[str] = set()
		for variant_index, variant in enumerate(operation_variants(schema, operation)):
			variant_where = where if variant_index == 0 else f"{where}.additional_dtype_variants[{variant_index - 1}]"
			validate_variant(
				variant,
				kind,
				variant_where,
				seen_dtypes,
				seen_ids,
				seen_kernels,
				seen_kernel_names,
				seen_sources,
			)


def validate_blas_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "BLAS schema_version must be 1")
	require(schema.get("family") == "matrix_blas", "BLAS family must be matrix_blas")
	require(schema.get("domain") == "matrix", "BLAS domain must be matrix")
	require(schema.get("dtype") == "f32", "BLAS dtype must be f32")
	workgroup = schema.get("workgroup_size")
	require(
		isinstance(workgroup, list)
		and len(workgroup) == 3
		and all(isinstance(value, int) and value > 0 for value in workgroup),
		"BLAS workgroup_size must contain three positive integers",
	)
	require(workgroup == [256, 1, 1], "BLAS workgroup_size must be [256, 1, 1]")
	require(schema.get("output_tile_size") == [64, 64, 1], "BLAS output_tile_size must be [64, 64, 1]")
	require(schema.get("k_tile_size") == 16, "BLAS k_tile_size must be 16")
	operations = schema.get("operations")
	require(isinstance(operations, list) and operations, "BLAS operations must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	seen_sources: set[str] = set()
	for index, operation in enumerate(operations):
		where = f"BLAS operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		for key, seen in (
			("name", seen_names),
			("kernel_id", seen_kernels),
			("source_stem", seen_sources),
		):
			value = operation.get(key)
			require(
				isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None,
				f"{where}.{key} must be an identifier",
			)
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = operation.get("stable_id")
		require(
			isinstance(stable_id, int) and 0 < stable_id <= 65535,
			f"{where}.stable_id must be a non-zero u16",
		)
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		require(operation.get("kind") == "mat_mul_nt", f"{where}.kind must be mat_mul_nt")
		validate_dnn_role(operation.get("dnn"), where)
		require(operation.get("variant") == "tiled", f"{where}.variant must be tiled")
		require(
			isinstance(operation.get("doc"), str) and operation["doc"].endswith("."),
			f"{where}.doc must end with a period",
		)
		require(operation.get("kernel_name") == operation["name"], f"{where}.kernel_name must match name")
		validate_blas_contract(operation.get("contract"), where)
		validate_blas_test(operation.get("test"), where)


def validate_dnn_role(role: Any, where: str) -> None:
	if role is None:
		return
	require(isinstance(role, dict), f"{where}.dnn must be an object")
	require(
		set(role).issubset({"role", "providers", "epilogue", "epilogue_requires_input"})
		and {"role", "providers"}.issubset(role),
		f"{where}.dnn fields are incomplete or unknown",
	)
	require(
		role["role"]
		in {
			"matmul",
			"bias_add",
			"relu",
			"gelu",
			"silu",
			"multiply",
			"add",
			"rms_norm",
			"scaled_dot_product_attention",
			"grouped_gemm",
			"gated_multiply",
			"color_convert",
			"resize_normalize",
			"residual_rms_norm",
		},
		f"{where}.dnn.role is unsupported",
	)
	providers = role["providers"]
	allowed = {
		"blaslt_epilogue",
		"qkv_projection_group",
		"gated_ffn",
		"residual_norm",
		"attention",
		"grouped_moe",
		"vision_preprocess",
	}
	require(
		isinstance(providers, list)
		and providers
		and len(providers) == len(set(providers))
		and all(provider in allowed for provider in providers),
		f"{where}.dnn.providers are invalid",
	)
	epilogue = role.get("epilogue", "none")
	require(
		epilogue in {"none", "bias", "bias_relu", "bias_gelu", "bias_silu"},
		f"{where}.dnn.epilogue is unsupported",
	)
	required_input = role.get("epilogue_requires_input")
	require(
		required_input is None
		or isinstance(required_input, int)
		and 0 <= required_input < 8,
		f"{where}.dnn.epilogue_requires_input must be a fixed input index",
	)
	require(
		epilogue != "none" or required_input is None,
		f"{where}.dnn.epilogue_requires_input requires an epilogue",
	)


def validate_semantic_attributes(attributes: Any, where: str) -> None:
	require(isinstance(attributes, list), f"{where}.semantic_attributes must be an array")
	seen: set[str] = set()
	for index, attribute in enumerate(attributes):
		label = f"{where}.semantic_attributes[{index}]"
		require(
			isinstance(attribute, list) and len(attribute) == 2,
			f"{label} must contain name and kind",
		)
		name, kind = attribute
		require(
			isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None,
			f"{label} name must be an identifier",
		)
		validate_unique(name, seen, f"{label} name")
		require(
			kind
			in {
				"boolean",
				"signed_integer",
				"unsigned_integer",
				"float",
				"string",
				"shape",
				"enum",
			},
			f"{label} kind is unsupported",
		)


def validate_ml_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "ML schema_version must be 1")
	require(schema.get("family") == "ml_training", "ML family must be ml_training")
	require(schema.get("domain") == "ml", "ML domain must be ml")
	require(schema.get("dtype") == "f32", "ML dtype must be f32")
	require(schema.get("workgroup_size") == [256, 1, 1], "ML workgroup_size must be [256, 1, 1]")
	operations = schema.get("operations")
	require(isinstance(operations, list) and operations, "ML operations must be non-empty")
	validate_ml_port_provenance(schema.get("port_provenance"), operations)
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	seen_sources: set[str] = set()
	for index, operation in enumerate(operations):
		where = f"ML operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernels)):
			value = operation.get(key)
			require(
				isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None,
				f"{where}.{key} must be an identifier",
			)
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = operation.get("stable_id")
		require(
			isinstance(stable_id, int) and 0 < stable_id <= 65535,
			f"{where}.stable_id must be a non-zero u16",
		)
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		source = operation.get("source")
		require(isinstance(source, str) and source.endswith(".slang"), f"{where}.source must be a Slang path")
		validate_unique(source, seen_sources, f"{where}.source")
		require(operation.get("variant") == "generic", f"{where}.variant must be generic")
		operation_workgroup = operation.get("workgroup_size", schema["workgroup_size"])
		operation_dtype = operation.get("dtype", schema["dtype"])
		lowering_only = operation.get("lowering_only", False)
		require(isinstance(lowering_only, bool), f"{where}.lowering_only must be boolean")
		require(operation_dtype in {"f32", "u32"}, f"{where}.dtype is unsupported")
		require(
			isinstance(operation_workgroup, list)
			and len(operation_workgroup) == 3
			and all(isinstance(value, int) and value > 0 for value in operation_workgroup)
			and math.prod(operation_workgroup) <= 1024,
			f"{where}.workgroup_size must contain three positive dimensions with at most 1024 threads",
		)
		differentiation = operation.get("differentiation")
		require(
			differentiation in {"none", "backward"}
			or any(candidate.get("name") == differentiation for candidate in operations),
			f"{where}.differentiation must name an operation, backward, or none",
		)
		if lowering_only:
			require(differentiation == "none", f"{where} lowering-only kernel cannot differentiate")
			require("contract" not in operation, f"{where} lowering-only kernel cannot own a semantic contract")
			require("dnn" not in operation, f"{where} lowering-only kernel cannot own a DNN role")
		else:
			validate_ml_contract(operation.get("contract"), differentiation, where)
		replay_role = operation.get("training_replay_role", "safe")
		require(
			replay_role
			in {
				"safe",
				"host_stepped_optimizer",
				"optimizer_state_advance",
				"optimizer_state_update",
			},
			f"{where}.training_replay_role is unsupported",
		)
		validate_semantic_attributes(operation.get("semantic_attributes", []), where)
		validate_dnn_role(operation.get("dnn"), where)
		validate_ml_test(operation.get("test"), where)
		fields = operation.get("push_fields")
		require(isinstance(fields, list) and fields, f"{where}.push_fields must be non-empty")
		offset = 0
		seen_fields: set[str] = set()
		for field_index, field in enumerate(fields):
			field_where = f"{where}.push_fields[{field_index}]"
			require(
				isinstance(field, list) and len(field) == 2,
				f"{field_where} must contain name and scalar type",
			)
			name, scalar_type = field
			require(
				isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None,
				f"{field_where} name must be an identifier",
			)
			validate_unique(name, seen_fields, f"{field_where} name")
			require(scalar_type in {"uint32", "float32"}, f"{field_where} scalar type is unsupported")
			offset += 4
		require(offset <= 128, f"{where} push constants exceed the minimum Vulkan limit")


def validate_ml_port_provenance(provenance: Any, operations: list[Any]) -> None:
	require(isinstance(provenance, list) and provenance, "ML port_provenance must be non-empty")
	operation_names = {
		operation.get("name") for operation in operations if isinstance(operation, dict)
	}
	seen_families: set[str] = set()
	seen_operations: set[str] = set()
	for index, record in enumerate(provenance):
		where = f"ML port_provenance[{index}]"
		require(isinstance(record, dict), f"{where} must be an object")
		require(
			set(record) == {"family", "operations", "classification", "donors", "reason"},
			f"{where} fields are incomplete or unknown",
		)
		family = record["family"]
		require(
			isinstance(family, str) and IDENTIFIER.fullmatch(family) is not None,
			f"{where}.family must be an identifier",
		)
		validate_unique(family, seen_families, f"{where}.family")
		classification = record["classification"]
		require(
			classification
			in {"verbatim", "mechanical_adaptation", "rust_redesign", "replacement", "new"},
			f"{where}.classification is unsupported",
		)
		record_operations = record["operations"]
		require(
			isinstance(record_operations, list) and record_operations,
			f"{where}.operations must be non-empty",
		)
		for operation in record_operations:
			require(
				isinstance(operation, str) and operation in operation_names,
				f"{where}.operations must name ML operations",
			)
			validate_unique(operation, seen_operations, f"{where}.operations")
		donors = record["donors"]
		require(isinstance(donors, list), f"{where}.donors must be an array")
		if classification == "new":
			require(not donors, f"{where}.donors must be empty for new work")
		else:
			require(
				donors
				and all(
					isinstance(donor, str)
					and donor.startswith(("source/", "tools/"))
					and ".." not in donor
					for donor in donors
				),
				f"{where}.donors must contain OA repository paths",
			)
		reason = record["reason"]
		require(
			isinstance(reason, str) and reason.endswith("."),
			f"{where}.reason must end with a period",
		)
	require(
		seen_operations == operation_names,
		"ML port_provenance must cover every operation exactly once",
	)


def validate_ml_contract(contract: Any, differentiation: str, where: str) -> None:
	require(isinstance(contract, dict), f"{where}.contract must be an object")
	require(
		set(contract)
		== {
			"input_kinds",
			"output_kinds",
			"shape_rule",
			"dtype_rule",
			"effects",
			"mutated_inputs",
			"output_alias_inputs",
			"lowering",
		},
		f"{where}.contract fields are incomplete or unknown",
	)
	inputs = contract["input_kinds"]
	outputs = contract["output_kinds"]
	require(isinstance(inputs, list) and inputs, f"{where}.contract.input_kinds must be non-empty")
	require(isinstance(outputs, list) and outputs, f"{where}.contract.output_kinds must be non-empty")
	require(
		all(isinstance(value, str) and value for value in [*inputs, *outputs]),
		f"{where}.contract input/output kinds must be strings",
	)
	require(
		isinstance(contract["shape_rule"], str) and contract["shape_rule"],
		f"{where}.contract.shape_rule must be non-empty",
	)
	require(
		contract["dtype_rule"]
		in {
			"f32",
			"all_f32",
			"u32_state",
			"f32_parameter_u32_state",
			"f32_logits_u32_targets",
			"f32_weight_u32_indices",
			"u32_indices_f32_gradient_weight",
		},
		f"{where}.contract.dtype_rule is unsupported",
	)
	require(
		contract["effects"] in (["read_inputs", "write_output"], ["read_inputs", "write_outputs"]),
		f"{where}.contract.effects are unsupported",
	)
	mutated = contract["mutated_inputs"]
	require(
		isinstance(mutated, list)
		and all(isinstance(value, int) and 0 <= value < len(inputs) for value in mutated)
		and len(set(mutated)) == len(mutated),
		f"{where}.contract.mutated_inputs must contain unique input indices",
	)
	aliases = contract["output_alias_inputs"]
	require(
		isinstance(aliases, list)
		and len(aliases) == len(outputs)
		and all(value == -1 or value in mutated for value in aliases)
		and sorted(value for value in aliases if value >= 0) == sorted(mutated),
		f"{where}.contract.output_alias_inputs must map every mutated input exactly once",
	)
	require(contract["lowering"] == "compute_dispatch", f"{where}.contract lowering is unsupported")
	require(
		differentiation in {"none", "backward"} or isinstance(differentiation, str),
		f"{where} differentiation is invalid",
	)


def validate_ml_test(test: Any, where: str) -> None:
	require(isinstance(test, dict), f"{where}.test must be an object")
	require(
		isinstance(test.get("oracle"), str) and test["oracle"],
		f"{where}.test.oracle must be non-empty",
	)
	shapes = test.get("shapes")
	require(isinstance(shapes, list) and shapes, f"{where}.test.shapes must be non-empty")
	for index, shape in enumerate(shapes):
		require(
			isinstance(shape, list)
			and shape
			and all(isinstance(extent, int) and extent >= 0 for extent in shape),
			f"{where}.test.shapes[{index}] must contain non-negative extents",
		)
	tolerance = test.get("tolerance")
	require(
		isinstance(tolerance, (int, float))
		and not isinstance(tolerance, bool)
		and tolerance >= 0,
		f"{where}.test.tolerance must be non-negative",
	)


def validate_blas_contract(contract: Any, where: str) -> None:
	require(isinstance(contract, dict), f"{where}.contract must be an object")
	expected = {
		"input_kinds": ["matrix", "matrix"],
		"output_kinds": ["matrix"],
		"shape_rule": "left_mk_right_nk_to_mn",
		"dtype_rule": "matching_f32",
		"effects": ["read_inputs", "write_output"],
		"mutated_inputs": [],
		"output_alias_inputs": [-1],
		"differentiation": "none",
		"lowering": "compute_dispatch",
	}
	for key, value in expected.items():
		require(contract.get(key) == value, f"{where}.contract.{key} must be {value!r}")


def validate_blas_test(test: Any, where: str) -> None:
	require(isinstance(test, dict), f"{where}.test must be an object")
	shapes = test.get("shapes")
	require(isinstance(shapes, list) and shapes, f"{where}.test.shapes must be non-empty")
	for index, shape in enumerate(shapes):
		require(
			isinstance(shape, list)
			and len(shape) == 3
			and all(isinstance(value, int) and value >= 0 for value in shape),
			f"{where}.test.shapes[{index}] must contain non-negative M, N, K",
		)
	tolerance = test.get("tolerance")
	require(
		isinstance(tolerance, (int, float))
		and not isinstance(tolerance, bool)
		and tolerance >= 0,
		f"{where}.test.tolerance must be non-negative",
	)


def validate_contract(contract: Any, kind: str) -> None:
	where = f"contracts.{kind}"
	require(isinstance(contract, dict), f"{where} must be an object")
	expected_inputs = ["matrix", "matrix"] if kind == "binary" else ["matrix"]
	expected_shape = "equal_no_broadcast" if kind == "binary" else "preserve_input"
	require(contract.get("input_kinds") == expected_inputs, f"{where}.input_kinds do not match {kind}")
	require(contract.get("output_kinds") == ["matrix"], f"{where}.output_kinds must contain matrix")
	require(contract.get("shape_rule") == expected_shape, f"{where}.shape_rule must be {expected_shape}")
	expected_dtype = "same_admitted_dtype" if kind == "binary" else "all_f32"
	require(contract.get("dtype_rule") == expected_dtype, f"{where}.dtype_rule must be {expected_dtype}")
	require(contract.get("effects") == ["read_inputs", "write_output"], f"{where}.effects are invalid")
	require(contract.get("mutated_inputs") == [], f"{where}.mutated_inputs must be empty")
	require(contract.get("output_alias_inputs") == [-1], f"{where}.output_alias_inputs must be [-1]")
	require(contract.get("differentiation") == "none", f"{where}.differentiation must be none")
	require(contract.get("lowering") == "compute_dispatch", f"{where}.lowering must be compute_dispatch")


def operation_variants(schema: dict[str, Any], operation: dict[str, Any]) -> list[dict[str, Any]]:
	base = {
		"dtype": schema["dtype"],
		"stable_id": operation.get("stable_id"),
		"kernel_id": operation.get("kernel_id"),
		"kernel_name": operation.get("name"),
		"test": operation.get("test"),
		"source_stem": operation.get("name"),
	}
	additional = operation.get("additional_dtype_variants", [])
	require(isinstance(additional, list), "additional_dtype_variants must be an array")
	return [base, *additional]


def validate_variant(
	variant: dict[str, Any],
	kind: str,
	where: str,
	seen_dtypes: set[str],
	seen_ids: set[int],
	seen_kernels: set[str],
	seen_kernel_names: set[str],
	seen_sources: set[str],
) -> None:
	require(isinstance(variant, dict), f"{where} must be an object")
	dtype = variant.get("dtype")
	require(dtype in DTYPES, f"{where}.dtype must be one of {sorted(DTYPES)}")
	validate_unique(dtype, seen_dtypes, f"{where}.dtype")
	if dtype == "i32":
		require(variant.get("integer_overflow") == "wrap", f"{where}.integer_overflow must be wrap")
	else:
		require("integer_overflow" not in variant, f"{where}.integer_overflow is valid only for integer dtypes")
	stable_id = variant.get("stable_id")
	require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id must be a non-zero u16")
	validate_unique(stable_id, seen_ids, f"{where}.stable_id")
	for key, seen in (
		("kernel_id", seen_kernels),
		("kernel_name", seen_kernel_names),
	):
		value = variant.get(key)
		require(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None, f"{where}.{key} must be an identifier")
		validate_unique(value, seen, f"{where}.{key}")
	source_stem = variant.get("source_stem")
	require(isinstance(source_stem, str) and IDENTIFIER.fullmatch(source_stem) is not None, f"{where}.source_stem must be an identifier")
	validate_unique(source_stem, seen_sources, f"{where}.source_stem")
	validate_test(variant.get("test"), kind, dtype, where)


def validate_test(test: Any, kind: str, dtype: str, where: str) -> None:
	require(isinstance(test, dict), f"{where}.test must be an object")
	input_keys = ["left", "right"] if kind == "binary" else ["input"]
	length: int | None = None
	for key in [*input_keys, "expected"]:
		values = test.get(key)
		require(isinstance(values, list) and values, f"{where}.test.{key} must be a non-empty array")
		if dtype == "f32":
			require(all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values), f"{where}.test.{key} must contain numbers")
		else:
			require(all(isinstance(value, int) and not isinstance(value, bool) and -(2**31) <= value < 2**31 for value in values), f"{where}.test.{key} must contain i32 values")
		if length is None:
			length = len(values)
		else:
			require(len(values) == length, f"{where}.test arrays must have equal lengths")
	if kind == "unary_scalar":
		require(dtype == "f32", f"{where}: unary_scalar currently supports only f32")
		require(isinstance(test.get("scalar"), (int, float)) and not isinstance(test.get("scalar"), bool), f"{where}.test.scalar must be numeric")
	else:
		require("scalar" not in test, f"{where}.test.scalar is valid only for unary_scalar")
	tolerance = test.get("tolerance")
	if dtype == "f32":
		require(isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) and tolerance >= 0, f"{where}.test.tolerance must be non-negative")
	else:
		require("tolerance" not in test, f"{where}.test.tolerance is invalid for exact integer tests")


def validate_unique(value: Any, seen: set[Any], label: str) -> None:
	require(value not in seen, f"duplicate {label}: {value}")
	seen.add(value)


def require(condition: bool, message: str) -> None:
	if not condition:
		raise SchemaError(message)


def banner(schema_hash: str, prefix: str, schema_name: str = "matrix_elemwise.json") -> str:
	return (
		f"{prefix} @generated by tools/gen/fn/generate.py; DO NOT EDIT.\n"
		f"{prefix} schema={schema_name} schema_version=1 generator_version={GENERATOR_VERSION}\n"
		f"{prefix} schema_sha256={schema_hash}\n"
	)


def registry_banner(elementwise_hash: str, blas_hash: str, ml_hash: str) -> str:
	return (
		"// @generated by tools/gen/fn/generate.py; DO NOT EDIT.\n"
		f"// schemas=matrix_elemwise.json,matrix_blas.json,ml_training.json generator_version={GENERATOR_VERSION}\n"
		f"// matrix_elemwise_sha256={elementwise_hash}\n"
		f"// matrix_blas_sha256={blas_hash}\n"
		f"// ml_training_sha256={ml_hash}\n"
	)


def rust_float(value: int | float) -> str:
	bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
	return f"f32::from_bits(0x{bits:08x})"


def rust_value(dtype: str, value: int | float) -> str:
	if dtype == "f32":
		return rust_float(value)
	if dtype == "i32":
		return f"{value}_i32"
	raise SchemaError(f"unsupported Rust test dtype {dtype}")


def format_rust(source: str, root: Path) -> str:
	config_path = root / "rustfmt.toml"
	if not config_path.is_file():
		config_path = Path(__file__).resolve().parents[3] / "rustfmt.toml"
	result = subprocess.run(
		[
			"rustfmt",
			"--emit",
			"stdout",
			"--edition",
			"2024",
			"--config-path",
			str(config_path),
		],
		input=source,
		text=True,
		capture_output=True,
		check=False,
	)
	if result.returncode != 0:
		raise SchemaError(f"rustfmt failed while formatting generated Rust:\n{result.stderr}")
	return result.stdout


def static_name(domain: str, name: str, dtype: str) -> str:
	return f"{domain.upper()}_{name.upper()}_{dtype.upper()}"


def rust_routes(schema: dict[str, Any], operation: dict[str, Any]) -> str:
	return ", ".join(
		f"({DTYPES[variant['dtype']]['rust_dtype']}, KernelId::{variant['kernel_id']})"
		for variant in operation_variants(schema, operation)
	)


def generate_api(schema: dict[str, Any], schema_hash: str) -> str:
	lines = [banner(schema_hash, "//").rstrip(), ""]
	for operation in schema["operations"]:
		name = operation["name"]
		routes = rust_routes(schema, operation)
		lines.append(f"/// {operation['doc']}")
		if operation["kind"] == "binary":
			lines.extend(["///", "/// Shapes must match exactly; broadcasting is not yet supported."])
		if any(variant["dtype"] == "i32" for variant in operation_variants(schema, operation)):
			lines.extend(["///", "/// The I32 route uses two's-complement wrapping arithmetic."])
		lines.extend(
			[
				"///",
				"/// The call submits asynchronously and returns its output immediately.",
				"///",
				"/// # Errors",
				"///",
				"/// Returns an error when the shape, dtype, ownership, or runtime submission contract fails.",
			]
		)
		if operation["kind"] == "binary":
			lines.extend(
				[
					f"pub fn {name}(left: &Matrix, right: &Matrix) -> Result<Matrix> {{",
					f"\tbinary(left, right, &[{routes}], crate::core::operation::matrix::{rust_const_name(name)})",
					"}",
				]
			)
		elif operation["kind"] == "unary":
			lines.extend(
				[
					f"pub fn {name}(input: &Matrix) -> Result<Matrix> {{",
					f"\tunary(input, &[{routes}], crate::core::operation::matrix::{rust_const_name(name)})",
					"}",
				]
			)
		else:
			lines.extend(
				[
					f"pub fn {name}(input: &Matrix, {operation['scalar_name']}: f32) -> Result<Matrix> {{",
					f"\tunary_scalar(input, {operation['scalar_name']}, &[{routes}], crate::core::operation::matrix::{rust_const_name(name)})",
					"}",
				]
			)
		lines.append("")
	return "\n".join(lines)


def registry_entries(
	elementwise: dict[str, Any], blas: dict[str, Any], ml: dict[str, Any]
) -> list[dict[str, Any]]:
	entries = []
	for operation in elementwise["operations"]:
		for variant in operation_variants(elementwise, operation):
			entries.append(
				{
					"domain": "matrix",
					"name": operation["name"],
					"dtype": variant["dtype"],
					"kernel_id": variant["kernel_id"],
					"stable_id": variant["stable_id"],
					"workgroup_size": elementwise["workgroup_size"],
					"dispatch_tile_size": elementwise["workgroup_size"],
					"training_replay_role": "safe",
				}
			)
	for operation in blas["operations"]:
		entries.append(
			{
				"domain": "matrix",
				"name": operation["name"],
				"dtype": blas["dtype"],
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": blas["workgroup_size"],
				"dispatch_tile_size": blas["output_tile_size"],
				"training_replay_role": "safe",
			}
		)
	for operation in ml["operations"]:
		dtype = operation.get("dtype", ml["dtype"])
		entries.append(
			{
				"domain": "ml",
				"name": operation["name"],
				"dtype": dtype,
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": operation.get("workgroup_size", ml["workgroup_size"]),
				"dispatch_tile_size": operation.get("workgroup_size", ml["workgroup_size"]),
				"training_replay_role": operation.get("training_replay_role", "safe"),
			}
		)
	seen_ids: set[int] = set()
	seen_kernels: set[str] = set()
	for entry in entries:
		validate_unique(entry["stable_id"], seen_ids, "stable kernel ID across schemas")
		validate_unique(entry["kernel_id"], seen_kernels, "kernel ID across schemas")
	return entries


def generate_registry(
	elementwise: dict[str, Any],
	elementwise_hash: str,
	blas: dict[str, Any],
	blas_hash: str,
	ml: dict[str, Any],
	ml_hash: str,
) -> str:
	entries = registry_entries(elementwise, blas, ml)
	lines = [
		registry_banner(elementwise_hash, blas_hash, ml_hash).rstrip(),
		"",
		"use std::sync::OnceLock;",
		"",
		"use super::ShaderArtifact;",
		"",
	]
	for entry in entries:
		domain = entry["domain"]
		name = entry["name"]
		dtype = entry["dtype"]
		workgroup = entry["workgroup_size"]
		dispatch_tile = entry["dispatch_tile_size"]
		lines.extend(
			[
				f"static {static_name(domain, name, dtype)}: ShaderArtifact = ShaderArtifact {{",
				f'\tbytes: include_bytes!(concat!(env!("OUT_DIR"), "/{domain}_{name}_{dtype}.spv")),',
				f"\tworkgroup_size: [{workgroup[0]}, {workgroup[1]}, {workgroup[2]}],",
				f"\tdispatch_tile_size: [{dispatch_tile[0]}, {dispatch_tile[1]}, {dispatch_tile[2]}],",
				"\tcontent_id: OnceLock::new(),",
				"};",
				"",
			]
		)
	lines.extend(
		[
			"#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
			"pub(crate) enum TrainingReplayRole {",
			"\tSafe,",
			"\tHostSteppedOptimizer,",
			"\tOptimizerStateAdvance,",
			"\tOptimizerStateUpdate,",
			"}",
			"",
			"#[repr(u16)]",
			"#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
			"pub(crate) enum KernelId {",
		]
	)
	for entry in entries:
		lines.append(f"\t{entry['kernel_id']} = {entry['stable_id']},")
	lines.extend(["}", "", "impl KernelId {", f"\tpub(crate) const ALL: [Self; {len(entries)}] = ["])
	for entry in entries:
		lines.append(f"\t\tSelf::{entry['kernel_id']},")
	lines.extend(["\t];", "", "\tpub(crate) const fn artifact(self) -> &'static ShaderArtifact {", "\t\tmatch self {"])
	for entry in entries:
		lines.append(
			f"\t\t\tSelf::{entry['kernel_id']} => &{static_name(entry['domain'], entry['name'], entry['dtype'])},"
		)
	lines.extend(["\t\t}", "\t}", "", "\tpub(crate) const fn index(self) -> usize {", "\t\tmatch self {"])
	for index, entry in enumerate(entries):
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => {index},")
	replay_roles = {
		"safe": "Safe",
		"host_stepped_optimizer": "HostSteppedOptimizer",
		"optimizer_state_advance": "OptimizerStateAdvance",
		"optimizer_state_update": "OptimizerStateUpdate",
	}
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn report_name(self) -> &'static str {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		name = f"{entry['domain']}.{entry['name']}.{entry['dtype']}"
		lines.append(f'\t\t\tSelf::{entry["kernel_id"]} => "{name}",')
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn dtype_report_token(self) -> &'static str {",
			"\t\tmatch self {",
		]
	)
	dtype_tokens = {"f32": "float32", "i32": "int32", "u32": "uint32"}
	for entry in entries:
		lines.append(
			f'\t\t\tSelf::{entry["kernel_id"]} => "{dtype_tokens[entry["dtype"]]}",'
		)
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn training_replay_role(self) -> TrainingReplayRole {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		role = replay_roles[entry["training_replay_role"]]
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => TrainingReplayRole::{role},")
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) fn linear_workgroups(self, element_count: u32) -> [u32; 3] {",
			"\t\tlet width = self.artifact().dispatch_tile_size[0];",
			"\t\t[element_count.div_ceil(width), 1, 1]",
			"\t}",
			"",
			"\tpub(crate) fn output_workgroups(self, rows: u32, columns: u32) -> [u32; 3] {",
			"\t\tlet tile = self.artifact().dispatch_tile_size;",
			"\t\t[rows.div_ceil(tile[0]), columns.div_ceil(tile[1]), 1]",
			"\t}",
			"}",
			"",
		]
	)
	return "\n".join(lines)


def semantic_contract_hash(domain: str, name: str, contract: dict[str, Any], attributes: list[dict[str, str]]) -> int:
	payload = json.dumps(
		{
			"name": f"oa::{domain}::{name}",
			"contract": contract,
			"attributes": attributes,
		},
		sort_keys=True,
		separators=(",", ":"),
	).encode("utf-8")
	value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
	return value or 1


def rust_const_name(name: str) -> str:
	return name.upper()


def semantic_contract_lines(
	domain: str,
	name: str,
	contract: dict[str, Any],
	attributes: list[dict[str, str]],
) -> list[str]:
	inputs = ", ".join("OpValueKind::Matrix" for _ in contract["input_kinds"])
	outputs = ", ".join("OpValueKind::Matrix" for _ in contract["output_kinds"])
	shape_rule = (
		"OpShapeRule::MatMulNt"
		if contract["shape_rule"] == "left_mk_right_nk_to_mn"
		else "OpShapeRule::MatchInput"
		if contract["shape_rule"] in {"equal_no_broadcast", "preserve_input"}
		else "OpShapeRule::Explicit"
	)
	dtype_rule = (
		"OpDTypeRule::MatchInput"
		if contract["dtype_rule"] in {"same_admitted_dtype", "all_f32", "matching_f32"}
		else "OpDTypeRule::Explicit"
	)
	differentiation = (
		"OpDifferentiation::None"
		if contract.get("differentiation") == "none"
		else "OpDifferentiation::Reverse"
	)
	lowering = (
		"OpLowering::Gemm"
		if contract["lowering"] == "gemm"
		else "OpLowering::Dispatch"
	)
	value = semantic_contract_hash(domain, name, contract, attributes)
	lines = [
		f"pub const {rust_const_name(name)}: OperationContract = OperationContract::new(",
		f'\t"oa::{domain}::{name}",',
		f"\t0x{value:016x},",
		f"\t&[{inputs}],",
		f"\t&[{outputs}],",
		")",
		f".with_shape_rule({shape_rule})",
		f".with_dtype_rule({dtype_rule})",
		f".with_differentiation({differentiation})",
		f".with_lowering({lowering})",
		".effects(OpEffect::READ_INPUTS.union(OpEffect::WRITE_OUTPUTS))",
	]
	if attributes:
		attribute_kinds = {
			"boolean": "Boolean",
			"signed_integer": "SignedInteger",
			"unsigned_integer": "UnsignedInteger",
			"float": "Float",
			"string": "String",
			"shape": "Shape",
			"enum": "Enum",
		}
		specs = ", ".join(
			f'OpAttributeSpec::new("{attribute["name"]}", OpAttributeKind::{attribute_kinds[attribute["kind"]]})'
			for attribute in attributes
		)
		lines.append(f".attributes(&[{specs}])")
	mutation_mask = sum(1 << input_index for input_index in contract["mutated_inputs"])
	if mutation_mask:
		lines.append(f".mutated_inputs(0x{mutation_mask:02x})")
	for output_index, input_index in enumerate(contract["output_alias_inputs"]):
		if input_index >= 0:
			lines.append(f".alias({output_index}, {input_index})")
	lines[-1] += ";"
	return lines


def generate_operation_registry(
	elementwise: dict[str, Any],
	elementwise_hash: str,
	blas: dict[str, Any],
	blas_hash: str,
	ml: dict[str, Any],
	ml_hash: str,
) -> str:
	lines = [
		banner(elementwise_hash, "//").rstrip(),
		f"// matrix_blas_sha256={blas_hash}",
		f"// ml_training_sha256={ml_hash}",
		"// Current OARS compatibility contracts; donor OA identities replace each row when its full contract lands.",
		"",
		"use super::{",
		"\tOpAttributeKind, OpAttributeSpec, OpDTypeRule, OpDifferentiation, OpEffect, OpLowering,",
		"\tOpShapeRule, OpValueKind, OperationContract,",
		"};",
		"",
		"/// Generated semantic contracts for Matrix operations.",
		"pub mod matrix {",
		"\tuse super::*;",
		"",
	]
	for operation in elementwise["operations"]:
		contract = dict(elementwise["contracts"][operation["kind"]])
		contract["differentiation"] = operation.get(
			"differentiation", contract["differentiation"]
		)
		attributes = []
		if operation["kind"] == "unary_scalar":
			attributes = [{"name": operation["scalar_name"], "kind": "float"}]
		for line in semantic_contract_lines("matrix", operation["name"], contract, attributes):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	for operation in blas["operations"]:
		for line in semantic_contract_lines("matrix", operation["name"], operation["contract"], []):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend(["}", "", "/// Generated compatibility contracts for current ML kernels.", "pub mod ml {", "\tuse super::*;", ""])
	for operation in ml["operations"]:
		if operation.get("lowering_only", False):
			continue
		contract = dict(operation["contract"])
		contract["differentiation"] = (
			"none" if operation["differentiation"] in {"none", "backward"} else "reverse"
		)
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation.get("semantic_attributes", [])
		]
		for line in semantic_contract_lines("ml", operation["name"], contract, attributes):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend(["}", ""])
	return "\n".join(lines)


def dnn_rust_variant(value: str) -> str:
	overrides = {
		"blaslt_epilogue": "BlasLtEpilogue",
		"qkv_projection_group": "QkvProjectionGroup",
		"gated_ffn": "GatedFfn",
		"grouped_moe": "GroupedMoe",
		"rms_norm": "RmsNorm",
		"residual_rms_norm": "ResidualRmsNorm",
	}
	return overrides.get(value, "".join(part.capitalize() for part in value.split("_")))


def generate_dnn_roles(
	elementwise: dict[str, Any],
	elementwise_hash: str,
	blas: dict[str, Any],
	blas_hash: str,
	ml: dict[str, Any],
	ml_hash: str,
) -> str:
	lines = [
		banner(elementwise_hash, "//").rstrip(),
		f"// matrix_blas_sha256={blas_hash}",
		f"// ml_training_sha256={ml_hash}",
		"// Candidate roles mirror OA DNN vocabulary while retaining OARS compatibility identities.",
		"",
		"pub(super) static DNN_OP_ROLES: &[DnnOpRole] = &[",
	]
	for domain, schema in (("matrix", elementwise), ("matrix", blas), ("ml", ml)):
		for operation in schema["operations"]:
			role = operation.get("dnn")
			if role is None:
				continue
			providers = " | ".join(
				f"DnnProvider::{dnn_rust_variant(provider)}.bit()"
				for provider in role["providers"]
			)
			epilogue = dnn_rust_variant(role.get("epilogue", "none"))
			required_input = role.get("epilogue_requires_input")
			required_input = "None" if required_input is None else f"Some({required_input})"
			lines.extend(
				[
					"\tDnnOpRole {",
					f"\t\tcontract: crate::core::operation::{domain}::{rust_const_name(operation['name'])},",
					f"\t\top_type: DnnOpType::{dnn_rust_variant(role['role'])},",
					f"\t\tepilogue: DnnEpilogue::{epilogue},",
					f"\t\tepilogue_required_input: {required_input},",
					f"\t\tprovider_mask: {providers},",
					"\t},",
				]
			)
	lines.extend(["];", ""])
	return "\n".join(lines)


def generate_shader(
	operation: dict[str, Any],
	variant: dict[str, Any],
	schema: dict[str, Any],
	schema_hash: str,
) -> str:
	kind = operation["kind"]
	dtype = variant["dtype"]
	dtype_config = DTYPES[dtype]
	slang_type = dtype_config["slang_type"]
	load = dtype_config["load"]
	store = dtype_config["store"]
	workgroup = schema["workgroup_size"]
	if kind == "binary":
		fields = "\tuint left_index;\n\tuint right_index;\n\tuint output_index;\n\tuint element_count;"
		loads = f"\t{slang_type} left = {load}(storage_buffers[push.left_index], index);\n\t{slang_type} right = {load}(storage_buffers[push.right_index], index);"
	else:
		fields = "\tuint input_index;\n\tuint output_index;\n\tuint element_count;"
		loads = f"\t{slang_type} input = {load}(storage_buffers[push.input_index], index);"
		if kind == "unary_scalar":
			fields += "\n\tfloat scalar;"
	return f"""{banner(schema_hash, '//').rstrip()}
// Semantic operation: matrix.{operation['name']}; stable kernel ID: {variant['stable_id']}.

import storage;
import attributes;

struct PushConstants {{
{fields}
}};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("{variant['kernel_name']}")]
[domain("matrix")]
[variant("generic")]
[dtype("{dtype}")]
[status("experimental")]
[shader("compute")]
[numthreads({workgroup[0]}, {workgroup[1]}, {workgroup[2]})]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {{
\tuint index = dispatch_thread_id.x;
\tif (index >= push.element_count) {{
\t\treturn;
\t}}

{loads}
\t{store}(storage_buffers[push.output_index], index, {operation['expression']});
}}
"""


def generate_test(schema: dict[str, Any], schema_hash: str) -> str:
	lines = [
		banner(schema_hash, "//").rstrip(),
		"",
		"const ELEMENT_COUNT: usize = 257;",
		"",
		"fn repeated<T: Copy>(values: &[T]) -> Vec<T> {",
		"\t(0..ELEMENT_COUNT)",
		"\t\t.map(|index| values[index % values.len()])",
		"\t\t.collect()",
		"}",
		"",
		"fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {",
		"\tassert_eq!(actual.len(), expected.len());",
		"\tfor (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {",
		"\t\tlet error = (actual - expected).abs();",
		"\t\tassert!(error <= tolerance, \"element {index}: expected {expected}, found {actual}, error {error} exceeds {tolerance}\");",
		"\t}",
		"}",
		"",
		"test_vk!(",
		"\tgenerated_elementwise_dtype_variants_match_schema_oracles,",
		"\tengine,",
		"{",
	]
	samples: dict[tuple[str, str], str] = {}
	for operation in schema["operations"]:
		name = operation["name"]
		for variant in operation_variants(schema, operation):
			dtype = variant["dtype"]
			rust_type = DTYPES[dtype]["rust_type"]
			label = f"{name}_{dtype}"
			test = variant["test"]
			lines.append("")
			if operation["kind"] == "binary":
				samples[(name, dtype)] = f"{label}_left"
				left = ", ".join(rust_value(dtype, value) for value in test["left"])
				right = ", ".join(rust_value(dtype, value) for value in test["right"])
				lines.extend(
					[
						f"\tlet {label}_left = oa::Matrix::from_slice(&engine, [ELEMENT_COUNT], &repeated(&[{left}]))?;",
						f"\tlet {label}_right = oa::Matrix::from_slice(&engine, [ELEMENT_COUNT], &repeated(&[{right}]))?;",
						f"\tlet {label}_actual: Vec<{rust_type}> = oa::matrix::{name}(&{label}_left, &{label}_right)?.read()?;",
					]
				)
			elif operation["kind"] == "unary":
				samples[(name, dtype)] = f"{label}_input"
				values = ", ".join(rust_value(dtype, value) for value in test["input"])
				lines.extend(
					[
						f"\tlet {label}_input = oa::Matrix::from_slice(&engine, [ELEMENT_COUNT], &repeated(&[{values}]))?;",
						f"\tlet {label}_actual: Vec<{rust_type}> = oa::matrix::{name}(&{label}_input)?.read()?;",
					]
				)
			else:
				samples[(name, dtype)] = f"{label}_input"
				values = ", ".join(rust_value(dtype, value) for value in test["input"])
				lines.extend(
					[
						f"\tlet {label}_input = oa::Matrix::from_slice(&engine, [ELEMENT_COUNT], &repeated(&[{values}]))?;",
						f"\tlet {label}_actual: Vec<{rust_type}> = oa::matrix::{name}(&{label}_input, {rust_value(dtype, test['scalar'])})?.read()?;",
					]
				)
			expected = ", ".join(rust_value(dtype, value) for value in test["expected"])
			if dtype == "f32":
				lines.append(f"\tassert_close(&{label}_actual, &repeated(&[{expected}]), {rust_float(test['tolerance'])});")
			else:
				lines.append(f"\tassert_eq!({label}_actual, repeated(&[{expected}]));")

	multi_dtype = next(
		(operation for operation in schema["operations"] if len(operation_variants(schema, operation)) > 1),
		None,
	)
	if multi_dtype is not None:
		variants = operation_variants(schema, multi_dtype)
		first = variants[0]
		second = variants[1]
		first_sample = samples[(multi_dtype["name"], first["dtype"])]
		second_sample = samples[(multi_dtype["name"], second["dtype"])]
		lines.extend(
			[
				"",
				f"\tlet typed_read_error = {second_sample}",
				f"\t\t.read::<{DTYPES[first['dtype']]['rust_type']}>()",
				"\t\t.expect_err(\"mismatched typed readback was accepted\");",
				"\tassert_eq!(typed_read_error.kind(), oa::ErrorKind::InvalidArgument);",
				"",
				f"\tlet mixed_dtype_error = match oa::matrix::{multi_dtype['name']}(&{first_sample}, &{second_sample}) {{",
				"\t\tOk(_) => panic!(\"mixed dense dtypes were accepted\"),",
				"\t\tErr(error) => error,",
				"\t};",
				"\tassert_eq!(mixed_dtype_error.kind(), oa::ErrorKind::InvalidArgument);",
			]
		)
		unsupported = next(
			(
				operation
				for operation in schema["operations"]
				if operation["kind"] == "binary"
				and second["dtype"] not in {variant["dtype"] for variant in operation_variants(schema, operation)}
			),
			None,
		)
		if unsupported is not None:
			lines.extend(
				[
					"",
					f"\tlet unsupported_dtype_error = match oa::matrix::{unsupported['name']}(&{second_sample}, &{second_sample}) {{",
					"\t\tOk(_) => panic!(\"unsupported dense dtype was accepted\"),",
					"\t\tErr(error) => error,",
					"\t};",
					"\tassert_eq!(unsupported_dtype_error.kind(), oa::ErrorKind::InvalidArgument);",
				]
			)
	lines.extend(
		[
			"",
			"\tOk(())",
			"});",
			"",
			"test_vk!(",
			"\tgenerated_elementwise_dtype_variants_preserve_zero_extent,",
			"\tengine,",
			"{",
		]
	)
	dtypes = sorted(
		{variant["dtype"] for operation in schema["operations"] for variant in operation_variants(schema, operation)}
	)
	for dtype in dtypes:
		rust_type = DTYPES[dtype]["rust_type"]
		lines.append(f"\tlet empty_{dtype} = oa::Matrix::from_slice::<{rust_type}>(&engine, [0], &[])?;")
	for operation in schema["operations"]:
		name = operation["name"]
		for variant in operation_variants(schema, operation):
			dtype = variant["dtype"]
			rust_type = DTYPES[dtype]["rust_type"]
			label = f"{name}_{dtype}"
			if operation["kind"] == "binary":
				call = f"oa::matrix::{name}(&empty_{dtype}, &empty_{dtype})?"
			elif operation["kind"] == "unary":
				call = f"oa::matrix::{name}(&empty_{dtype})?"
			else:
				call = f"oa::matrix::{name}(&empty_{dtype}, {rust_value(dtype, 1)})?"
			lines.extend(
				[
					f"\tlet {label}_output = {call};",
					f"\tassert_eq!({label}_output.shape(), [0]);",
					f"\tassert_eq!({label}_output.dtype(), oa::{DTYPES[dtype]['rust_dtype']});",
					f"\tassert!({label}_output.read::<{rust_type}>()?.is_empty());",
				]
			)
	lines.extend(["\tOk(())", "});", ""])
	return "\n".join(lines)


def generate_blas_api(schema: dict[str, Any], schema_hash: str) -> str:
	lines = [banner(schema_hash, "//", "matrix_blas.json").rstrip(), ""]
	for operation in schema["operations"]:
		lines.extend(
			[
				f"/// {operation['doc']}",
				"///",
				"/// `left` has shape `[M, K]`, `right` has shape `[N, K]`, and the",
				"/// returned matrix has shape `[M, N]`. This matches OA's native",
				"/// weight-layout convention and does not transpose storage.",
				"///",
				"/// The call submits asynchronously and returns its output immediately.",
				"///",
				"/// # Errors",
				"///",
				"/// Returns an error when either input is not a rank-two FP32 matrix,",
				"/// their K extents differ, they belong to different engines, a dimension",
				"/// exceeds the admitted shader ABI, or runtime submission fails.",
				f"pub fn {operation['name']}(left: &Matrix, right: &Matrix) -> Result<Matrix> {{",
				f"\t{operation['name']}_impl(left, right, crate::core::operation::matrix::{rust_const_name(operation['name'])})",
				"}",
				"",
			]
		)
	return "\n".join(lines)


def generate_blas_shader(
	operation: dict[str, Any], schema: dict[str, Any], schema_hash: str
) -> str:
	workgroup = schema["workgroup_size"]
	output_tile = schema["output_tile_size"]
	k_tile = schema["k_tile_size"]
	return f"""{banner(schema_hash, '//', 'matrix_blas.json').rstrip()}
// Semantic operation: matrix.{operation['name']}; stable kernel ID: {operation['stable_id']}.
// FP32 specialization of OA's established 64x64x16 tiled GEMM arithmetic.

import storage;
import attributes;

struct PushConstants {{
	uint left_index;
	uint right_index;
	uint output_index;
	uint m;
	uint n;
	uint k;
}};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BM = {output_tile[0]};
static const uint BN = {output_tile[1]};
static const uint BK = {k_tile};
static const uint TM = 4;
static const uint TN = 4;
static const uint THREADS_COL = BN / TN;
static const uint LOAD_WIDTH = 4;
static const uint BK_PAD = BK + 1;

groupshared float shared_left[BM * BK_PAD];
groupshared float shared_right[BN * BK_PAD];

void load_slice(
	uint row,
	uint k_begin,
	uint buffer_index,
	uint row_limit,
	out float values[LOAD_WIDTH]
) {{
	if (row < row_limit && k_begin + LOAD_WIDTH <= push.k) {{
		float4 packed = storage_buffers[buffer_index].Load<float4>(
			(row * push.k + k_begin) * 4u
		);
		values[0] = packed.x;
		values[1] = packed.y;
		values[2] = packed.z;
		values[3] = packed.w;
	}} else {{
		[[unroll]] for (uint element = 0; element < LOAD_WIDTH; ++element) {{
			values[element] = row < row_limit && k_begin + element < push.k
				? load_f32(storage_buffers[buffer_index], row * push.k + k_begin + element)
				: 0.0f;
		}}
	}}
}}

[kernel_name("{operation['kernel_name']}")]
[domain("matrix")]
[variant("{operation['variant']}")]
[dtype("{schema['dtype']}")]
[status("experimental")]
[shader("compute")]
[numthreads({workgroup[0]}, {workgroup[1]}, {workgroup[2]})]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {{
	uint thread = group_thread_id.x;
	uint thread_row = thread / THREADS_COL;
	uint thread_column = thread % THREADS_COL;
	uint row_base = group_id.x * BM;
	uint column_base = group_id.y * BN;
	uint shared_row = (thread * LOAD_WIDTH) / BK;
	uint shared_column = (thread * LOAD_WIDTH) % BK;
	uint shared_offset = shared_row * BK_PAD + shared_column;

	float accumulators[TM * TN];
	[[unroll]] for (uint index = 0; index < TM * TN; ++index) {{
		accumulators[index] = 0.0f;
	}}

	for (uint k_base = 0; k_base < push.k; k_base += BK) {{
		float left_values[LOAD_WIDTH];
		float right_values[LOAD_WIDTH];
		load_slice(
			row_base + shared_row,
			k_base + shared_column,
			push.left_index,
			push.m,
			left_values
		);
		load_slice(
			column_base + shared_row,
			k_base + shared_column,
			push.right_index,
			push.n,
			right_values
		);
		[[unroll]] for (uint element = 0; element < LOAD_WIDTH; ++element) {{
			shared_left[shared_offset + element] = left_values[element];
			shared_right[shared_offset + element] = right_values[element];
		}}
		GroupMemoryBarrierWithGroupSync();

		[[unroll]] for (uint inner = 0; inner < BK; ++inner) {{
			float left_fragment[TM];
			float right_fragment[TN];
			[[unroll]] for (uint row = 0; row < TM; ++row) {{
				left_fragment[row] = shared_left[(thread_row * TM + row) * BK_PAD + inner];
			}}
			[[unroll]] for (uint column = 0; column < TN; ++column) {{
				right_fragment[column] = shared_right[(thread_column * TN + column) * BK_PAD + inner];
			}}
			[[unroll]] for (uint row = 0; row < TM; ++row) {{
				[[unroll]] for (uint column = 0; column < TN; ++column) {{
					accumulators[row * TN + column] += left_fragment[row] * right_fragment[column];
				}}
			}}
		}}
		GroupMemoryBarrierWithGroupSync();
	}}

	[[unroll]] for (uint row = 0; row < TM; ++row) {{
		[[unroll]] for (uint column = 0; column < TN; ++column) {{
			uint output_row = row_base + thread_row * TM + row;
			uint output_column = column_base + thread_column * TN + column;
			if (output_row < push.m && output_column < push.n) {{
				store_f32(
					storage_buffers[push.output_index],
					output_row * push.n + output_column,
					accumulators[row * TN + column]
				);
			}}
		}}
	}}
}}
"""


def generate_blas_test(schema: dict[str, Any], schema_hash: str) -> str:
	operation = schema["operations"][0]
	shapes = ", ".join(f"({m}, {n}, {k})" for m, n, k in operation["test"]["shapes"])
	tolerance = rust_float(operation["test"]["tolerance"])
	return f"""{banner(schema_hash, '//', 'matrix_blas.json').rstrip()}

fn values(length: usize, salt: usize) -> Vec<f32> {{
	(0..length)
		.map(|index| (((index * 17 + salt) % 29) as f32 - 14.0) / 7.0)
		.collect()
}}

fn cpu_mat_mul_nt(left: &[f32], right: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {{
	let mut output = vec![0.0; m * n];
	for row in 0..m {{
		for column in 0..n {{
			for inner in 0..k {{
				output[row * n + column] += left[row * k + inner] * right[column * k + inner];
			}}
		}}
	}}
	output
}}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {{
	assert_eq!(actual.len(), expected.len());
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {{
		let error = (actual - expected).abs();
		assert!(
			error <= tolerance,
			"element {{index}}: expected {{expected}}, found {{actual}}, error {{error}} exceeds {{tolerance}}"
		);
	}}
}}

test_vk!(generated_mat_mul_nt_matches_schema_oracle, engine, {{
	for (m, n, k) in [{shapes}] {{
		let left_values = values(m * k, 3);
		let right_values = values(n * k, 11);
		let expected = cpu_mat_mul_nt(&left_values, &right_values, m, n, k);
		let left = oa::Matrix::from_f32(&engine, [m, k], &left_values)?;
		let right = oa::Matrix::from_f32(&engine, [n, k], &right_values)?;
		let output = oa::matrix::{operation['name']}(&left, &right)?;
		assert_eq!(output.shape(), [m, n]);
		assert_eq!(output.dtype(), oa::DType::F32);
		assert_close(&output.read_f32()?, &expected, {tolerance});
	}}
	Ok(())
}});

test_vk!(generated_mat_mul_nt_rejects_invalid_contracts, engine, {{
	let other_engine = oa::Engine::new()?;
	let rank_one = oa::Matrix::from_f32(&engine, [6], &[0.0; 6])?;
	let left = oa::Matrix::from_f32(&engine, [2, 3], &[0.0; 6])?;
	let wrong_k = oa::Matrix::from_f32(&engine, [2, 4], &[0.0; 8])?;
	let integer = oa::Matrix::from_slice(&engine, [2, 3], &[0_i32; 6])?;
	let foreign = oa::Matrix::from_f32(&other_engine, [2, 3], &[0.0; 6])?;

	for error in [
		oa::matrix::{operation['name']}(&rank_one, &left).err().expect("rank-one input was accepted"),
		oa::matrix::{operation['name']}(&left, &wrong_k).err().expect("mismatched K was accepted"),
		oa::matrix::{operation['name']}(&integer, &integer).err().expect("I32 input was accepted"),
		oa::matrix::{operation['name']}(&left, &foreign).err().expect("cross-engine inputs were accepted"),
	] {{
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}}
	Ok(())
}});
"""


def expected_outputs(
	root: Path,
	elementwise: dict[str, Any],
	elementwise_hash: str,
	blas: dict[str, Any],
	blas_hash: str,
	ml: dict[str, Any],
	ml_hash: str,
) -> dict[Path, str]:
	outputs = {
		root / "src/rs/core/operation/generated.rs": format_rust(
			generate_operation_registry(
				elementwise, elementwise_hash, blas, blas_hash, ml, ml_hash
			),
			root,
		),
		root / "src/rs/matrix/elemwise.gen.rs": format_rust(generate_api(elementwise, elementwise_hash), root),
		root / "src/rs/matrix/blas.gen.rs": format_rust(generate_blas_api(blas, blas_hash), root),
		root / "src/rs/runtime/shader/generated.rs": format_rust(
			generate_registry(elementwise, elementwise_hash, blas, blas_hash, ml, ml_hash), root
		),
		root / "src/rs/runtime/dnn/generated.rs": format_rust(
			generate_dnn_roles(elementwise, elementwise_hash, blas, blas_hash, ml, ml_hash), root
		),
		root / "test/rs/matrix/test_elemwise.gen.rs": format_rust(generate_test(elementwise, elementwise_hash), root),
		root / "test/rs/matrix/test_blas.gen.rs": format_rust(generate_blas_test(blas, blas_hash), root),
	}
	for operation in elementwise["operations"]:
		for variant in operation_variants(elementwise, operation):
			outputs[root / f"src/slang/matrix/elemwise/{variant['source_stem']}.gen.slang"] = generate_shader(
				operation,
				variant,
				elementwise,
				elementwise_hash,
			)
	for operation in blas["operations"]:
		outputs[root / f"src/slang/matrix/blas/{operation['source_stem']}.gen.slang"] = generate_blas_shader(
			operation, blas, blas_hash
		)
	return outputs


def publish(outputs: dict[Path, str], shader_directories: list[Path]) -> None:
	for path, content in outputs.items():
		path.parent.mkdir(parents=True, exist_ok=True)
		with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
			temporary.write(content)
			temporary_path = Path(temporary.name)
		os.replace(temporary_path, path)
	for shader_directory in shader_directories:
		expected_shaders = {path.resolve() for path in outputs if path.parent == shader_directory}
		for stale in shader_directory.glob("*.gen.slang"):
			if stale.resolve() not in expected_shaders:
				stale.unlink()


def check(outputs: dict[Path, str], shader_directories: list[Path]) -> list[str]:
	errors = []
	for path, content in outputs.items():
		if not path.exists():
			errors.append(f"missing generated output: {path}")
		elif path.read_text(encoding="utf-8") != content:
			errors.append(f"stale generated output: {path}")
	for shader_directory in shader_directories:
		expected_shaders = {path.resolve() for path in outputs if path.parent == shader_directory}
		for stale in shader_directory.glob("*.gen.slang"):
			if stale.resolve() not in expected_shaders:
				errors.append(f"stale generated shader: {stale}")
	return errors


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
	parser.add_argument("--schema", type=Path)
	parser.add_argument("--blas-schema", type=Path)
	parser.add_argument("--ml-schema", type=Path)
	parser.add_argument("--check", action="store_true")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	root = args.root.resolve()
	schema_path = args.schema.resolve() if args.schema else root / DEFAULT_SCHEMA
	blas_schema_path = args.blas_schema.resolve() if args.blas_schema else root / DEFAULT_BLAS_SCHEMA
	ml_schema_path = args.ml_schema.resolve() if args.ml_schema else root / DEFAULT_ML_SCHEMA
	try:
		schema, schema_hash = load_schema(schema_path)
		blas_schema, blas_schema_hash = load_blas_schema(blas_schema_path)
		ml_schema, ml_schema_hash = load_ml_schema(ml_schema_path)
		registry_entries(schema, blas_schema, ml_schema)
	except (OSError, SchemaError) as error:
		print(f"operation generation failed: {error}", file=os.sys.stderr)
		return 1
	outputs = expected_outputs(
		root,
		schema,
		schema_hash,
		blas_schema,
		blas_schema_hash,
		ml_schema,
		ml_schema_hash,
	)
	shader_directories = [
		root / "src/slang/matrix/elemwise",
		root / "src/slang/matrix/blas",
	]
	if args.check:
		errors = check(outputs, shader_directories)
		if errors:
			print("\n".join(errors), file=os.sys.stderr)
			return 1
		return 0
	publish(outputs, shader_directories)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
