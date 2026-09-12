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


GENERATOR_VERSION = 52
DEFAULT_SCHEMA = Path("tools/gen/fn/schema/matrix_elemwise.json")
DEFAULT_BLAS_SCHEMA = Path("tools/gen/fn/schema/matrix_blas.json")
DEFAULT_REDUCE_SCHEMA = Path("tools/gen/fn/schema/matrix_reduce.json")
DEFAULT_RNG_SCHEMA = Path("tools/gen/fn/schema/matrix_rng.json")
DEFAULT_INDEX_SCHEMA = Path("tools/gen/fn/schema/matrix_index.json")
DEFAULT_ML_SCHEMA = Path("tools/gen/fn/schema/ml_training.json")
DEFAULT_AUDIO_SCHEMA = Path("tools/gen/fn/schema/audio.json")
DEFAULT_CRYPTOGRAPHY_HASH_SCHEMA = Path("tools/gen/fn/schema/cryptography_hash.json")
DEFAULT_IMAGE_SCHEMA = Path("tools/gen/fn/schema/image.json")
DEFAULT_VISION_SCHEMA = Path("tools/gen/fn/schema/vision_detection.json")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
KINDS = {"binary", "unary", "unary_scalar"}
DTYPES = {
	"u8": {
		"rust_type": "u8",
		"rust_dtype": "DType::U8",
		"slang_type": "uint",
		"load": "load_u8",
		"store": "store_u8",
	},
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
	"u32": {
		"rust_type": "u32",
		"rust_dtype": "DType::U32",
		"slang_type": "uint",
		"load": "load_u32",
		"store": "store_u32",
	},
}

PHYSICAL_WRITE_DOMAINS = {"axis_slices", "output_elements", "rows", "scalar"}
PHYSICAL_WRITE_PARTITIONS = {
	"exclusive_per_invocation",
	"exclusive_per_workgroup",
	"shared_atomic_contributors",
}
PHYSICAL_WRITE_EXTENTS = {
	"axis_slice",
	"one_element",
	"one_scalar",
	"row_width",
	"tile_16x16",
	"up_to_two_elements",
	"up_to_four_elements",
}
PHYSICAL_WRITE_COLLISIONS = {"exclusive", "atomic_u32"}
PHYSICAL_WRITE_TAILS = {"bounds_checked"}
WORKSPACE_PARTITIONS = {"exclusive_per_workgroup", "none"}


class SchemaError(ValueError):
	"""The operation schema is structurally invalid.
	"""


def validate_physical_write(value: Any, where: str, *, required: bool) -> None:
	if value is None:
		require(not required, f"{where}.physical_write is required")
		return
	require(isinstance(value, dict), f"{where}.physical_write must be an object")
	require(
		set(value) == {"writes", "workspace"},
		f"{where}.physical_write fields are incomplete or unknown",
	)
	writes = value.get("writes")
	require(
		isinstance(writes, list) and writes,
		f"{where}.physical_write.writes must be a non-empty array",
	)
	seen_bindings: set[int] = set()
	for index, write in enumerate(writes):
		label = f"{where}.physical_write.writes[{index}]"
		require(isinstance(write, dict), f"{label} must be an object")
		require(
			set(write) == {"binding", "domain", "partition", "extent", "collision", "tail"},
			f"{label} fields are incomplete or unknown",
		)
		binding = write.get("binding")
		require(
			isinstance(binding, int) and not isinstance(binding, bool) and 0 <= binding <= 255,
			f"{label}.binding must be a u8 binding ordinal",
		)
		validate_unique(binding, seen_bindings, f"{label}.binding")
		require(write.get("domain") in PHYSICAL_WRITE_DOMAINS, f"{label}.domain is unsupported")
		require(
			write.get("partition") in PHYSICAL_WRITE_PARTITIONS,
			f"{label}.partition is unsupported",
		)
		require(write.get("extent") in PHYSICAL_WRITE_EXTENTS, f"{label}.extent is unsupported")
		require(
			write.get("collision") in PHYSICAL_WRITE_COLLISIONS,
			f"{label}.collision is unsupported",
		)
		require(write.get("tail") in PHYSICAL_WRITE_TAILS, f"{label}.tail is unsupported")
	workspace = value.get("workspace")
	require(workspace in WORKSPACE_PARTITIONS, f"{where}.physical_write.workspace is unsupported")
	if workspace == "exclusive_per_workgroup":
		require(
			all(write["partition"] == "exclusive_per_workgroup" for write in writes),
			f"{where}.physical_write workspace and write partitions are contradictory",
		)
	for write in writes:
		if write["partition"] == "shared_atomic_contributors":
			require(
				write["collision"] == "atomic_u32",
				f"{where}.physical_write shared contributors require atomic_u32 collision policy",
			)
		if write["collision"] == "atomic_u32":
			require(
				write["partition"] == "shared_atomic_contributors",
				f"{where}.physical_write atomic_u32 requires shared contributors",
			)


def load_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_schema)


def load_blas_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_blas_schema)


def load_reduce_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_reduce_schema)


def load_ml_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_ml_schema)


def load_rng_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_rng_schema)


def load_index_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_index_schema)


def load_audio_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_audio_schema)


def load_cryptography_hash_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_cryptography_hash_schema)


def load_image_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_image_schema)


def load_vision_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_vision_schema)


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
		lowerings = operation.get("additional_lowering_variants", [])
		require(isinstance(lowerings, list), f"{where}.additional_lowering_variants must be an array")
		for lowering_index, lowering in enumerate(lowerings):
			lowering_where = f"{where}.additional_lowering_variants[{lowering_index}]"
			require(isinstance(lowering, dict), f"{lowering_where} must be an object")
			lowering_name = lowering.get("name")
			lowering_dtype = lowering.get("dtype")
			require(
				(operation["name"], lowering_name, lowering_dtype)
				in {
					("add", "add_broadcast", "f32"),
					("add", "add_broadcast_i32", "i32"),
					("sub", "sub_broadcast", "f32"),
					("mul", "mul_broadcast", "f32"),
					("div", "div_broadcast", "f32"),
				},
				f"{lowering_where} is not an admitted broadcast candidate",
			)
			require(lowering.get("variant") == "broadcast", f"{lowering_where}.variant must be broadcast")
			stable_id = lowering.get("stable_id")
			require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{lowering_where}.stable_id must be a non-zero u16")
			validate_unique(stable_id, seen_ids, f"{lowering_where}.stable_id")
			kernel_id = lowering.get("kernel_id")
			require(isinstance(kernel_id, str) and IDENTIFIER.fullmatch(kernel_id) is not None, f"{lowering_where}.kernel_id must be an identifier")
			validate_unique(kernel_id, seen_kernels, f"{lowering_where}.kernel_id")
			source = lowering.get("source")
			require(isinstance(source, str) and source.startswith("src/slang/matrix/elemwise/") and source.endswith(".slang"), f"{lowering_where}.source must be owned by matrix/elemwise")
			validate_unique(source, seen_sources, f"{lowering_where}.source")
			workgroup = lowering.get("workgroup_size", schema["workgroup_size"])
			require(workgroup == [256, 1, 1], f"{lowering_where}.workgroup_size must be [256, 1, 1]")
			validate_physical_write(lowering.get("physical_write"), lowering_where, required=True)
			fields = lowering.get("push_fields")
			require(isinstance(fields, list) and fields, f"{lowering_where}.push_fields must be non-empty")
			seen_fields: set[str] = set()
			for field in fields:
				require(isinstance(field, list) and len(field) == 2, f"{lowering_where}.push_fields entry is invalid")
				field_name, scalar_type = field
				require(isinstance(field_name, str) and IDENTIFIER.fullmatch(field_name) is not None, f"{lowering_where}.push field name is invalid")
				validate_unique(field_name, seen_fields, f"{lowering_where}.push field name")
				require(scalar_type == "uint32", f"{lowering_where}.push fields must be uint32")
			require(len(fields) * 4 <= 128, f"{lowering_where}.push constants exceed Vulkan minimum guarantee")


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


def validate_reduce_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Reduce schema_version must be 1")
	require(schema.get("family") == "matrix_reduce", "Reduce family must be matrix_reduce")
	require(schema.get("domain") == "matrix", "Reduce domain must be matrix")
	require(schema.get("dtype") == "f32", "Reduce dtype must be f32")
	require(
		schema.get("workgroup_size") == [256, 1, 1],
		"Reduce workgroup_size must be [256, 1, 1]",
	)
	provenance = schema.get("port_provenance")
	require(isinstance(provenance, dict), "Reduce port_provenance must be an object")
	require(
		set(provenance) == {"classification", "donors", "reason"},
		"Reduce port_provenance fields are incomplete or unknown",
	)
	require(
		provenance.get("classification") == "mechanical_adaptation",
		"Reduce port classification must be mechanical_adaptation",
	)
	require(
		isinstance(provenance.get("donors"), list)
		and provenance["donors"]
		and all(
			isinstance(donor, str)
			and donor.startswith(("source/", "tools/"))
			and ".." not in donor
			for donor in provenance["donors"]
		),
		"Reduce donor paths must be non-empty",
	)
	require(
		isinstance(provenance.get("reason"), str) and provenance["reason"].endswith("."),
		"Reduce port reason must end with a period",
	)
	operations = schema.get("operations")
	require(
		isinstance(operations, list) and len(operations) == 9,
		"Reduce operations must contain the admitted Softmax, LogSoftmax, Sum, and categorical-count families",
	)
	require(
		[operation.get("name") for operation in operations]
		== [
			"softmax",
			"softmax_backward",
			"log_softmax",
			"log_softmax_backward",
			"sum",
			"sum_axis",
			"sum_backward",
			"categorical_accuracy_count",
			"masked_categorical_accuracy_count",
		],
		"Reduce operations are not in canonical family order",
	)
	operation_names = {operation["name"] for operation in operations}
	seen_ids: set[int] = set()
	seen_kernels: set[str] = set()
	seen_sources: set[str] = set()
	for index, operation in enumerate(operations):
		where = f"Reduce operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		validate_physical_write(operation.get("physical_write"), where, required=True)
		stable_id = operation.get("stable_id")
		require(
			isinstance(stable_id, int) and 0 < stable_id <= 65535,
			f"{where}.stable_id must be a non-zero u16",
		)
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		kernel_id = operation.get("kernel_id")
		require(
			isinstance(kernel_id, str) and IDENTIFIER.fullmatch(kernel_id) is not None,
			f"{where}.kernel_id must be an identifier",
		)
		validate_unique(kernel_id, seen_kernels, f"{where}.kernel_id")
		source = operation.get("source")
		require(
			isinstance(source, str) and source.endswith(".slang"),
			f"{where}.source must be a Slang path",
		)
		validate_unique(source, seen_sources, f"{where}.source")
		require(operation.get("variant") == "generic", f"{where}.variant must be generic")
		lowering_only = operation.get("lowering_only", False)
		if lowering_only:
			require(
				operation["name"] == "sum_axis"
				and operation.get("semantic_operation") == "sum"
				and operation["semantic_operation"] in operation_names,
				f"{where} must be the private Sum axis lowering",
			)
			for forbidden in ("differentiation", "semantic_attributes", "contract", "test"):
				require(forbidden not in operation, f"{where}.{forbidden} is invalid for lowering-only work")
		else:
			require(
				"semantic_operation" not in operation,
				f"{where}.semantic_operation is valid only for lowering-only work",
			)
			differentiation = operation.get("differentiation")
			expected_differentiation = (
				"reverse"
				if operation["name"] in {"softmax", "log_softmax", "sum"}
				else "none"
			)
			require(
				differentiation == expected_differentiation,
				f"{where}.differentiation must be {expected_differentiation}",
			)
			validate_semantic_attributes(operation.get("semantic_attributes"), where)
			expected_attributes = (
				[]
				if operation["name"].endswith("categorical_accuracy_count")
				else [["dim", "signed_integer"]]
			)
			require(
				operation["semantic_attributes"] == expected_attributes,
				f"{where} semantic attributes are invalid; reductions require a signed dim attribute and categorical counts require none",
			)
			validate_ml_contract(operation.get("contract"), differentiation, where)
			expected_inputs = {
				"softmax": ["matrix"],
				"softmax_backward": ["matrix", "matrix"],
				"log_softmax": ["matrix"],
				"log_softmax_backward": ["matrix", "matrix"],
				"sum": ["matrix"],
				"sum_backward": ["matrix", "matrix"],
				"categorical_accuracy_count": ["matrix", "class_indices"],
				"masked_categorical_accuracy_count": [
					"matrix",
					"class_indices",
					"matrix",
				],
			}[operation["name"]]
			require(
				operation["contract"]["input_kinds"] == expected_inputs,
				f"{where} input kinds are invalid",
			)
			fields = operation.get("push_fields")
			expected_field_count = {
				"softmax": 5,
				"softmax_backward": 6,
				"log_softmax": 5,
				"log_softmax_backward": 6,
				"sum": 3,
				"sum_axis": 5,
				"sum_backward": 5,
				"categorical_accuracy_count": 6,
				"masked_categorical_accuracy_count": 7,
			}[operation["name"]]
		require(
			isinstance(fields, list) and len(fields) == expected_field_count,
			f"{where}.push_fields has the wrong width",
		)
		seen_fields: set[str] = set()
		for field in fields:
			require(
				isinstance(field, list) and len(field) == 2,
				f"{where}.push_fields entry is invalid",
			)
			name, scalar_type = field
			require(
				isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None,
				f"{where}.push field name is invalid",
			)
			validate_unique(name, seen_fields, f"{where}.push field name")
			require(scalar_type == "uint32", f"{where}.push fields must be uint32")
		if not lowering_only:
			test = operation.get("test")
			require(isinstance(test, dict), f"{where}.test must be an object")
			require(
				isinstance(test.get("shape"), list)
				and test["shape"]
				and all(isinstance(value, int) and value > 0 for value in test["shape"]),
				f"{where}.test.shape must contain positive dimensions",
			)
			if not operation["name"].endswith("categorical_accuracy_count"):
				require(isinstance(test.get("dim"), int), f"{where}.test.dim must be signed")
			require(
				isinstance(test.get("tolerance"), (int, float))
				and test["tolerance"] >= 0,
				f"{where}.test.tolerance must be non-negative",
			)


def validate_rng_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "RNG schema_version must be 1")
	require(schema.get("family") == "matrix_rng", "RNG family must be matrix_rng")
	require(schema.get("domain") == "matrix", "RNG domain must be matrix")
	require(schema.get("dtype") == "f32", "RNG dtype must be f32")
	require(schema.get("workgroup_size") == [256, 1, 1], "RNG workgroup_size must be [256, 1, 1]")
	operations = schema.get("operations")
	require(isinstance(operations, list) and operations, "RNG operations must be non-empty")
	names = {operation.get("name") for operation in operations if isinstance(operation, dict)}
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	seen_sources: set[str] = set()
	for index, operation in enumerate(operations):
		where = f"RNG operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernels)):
			value = operation.get(key)
			require(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None, f"{where}.{key} must be an identifier")
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = operation.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id must be a non-zero u16")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		source = operation.get("source")
		require(isinstance(source, str) and source.endswith(".slang"), f"{where}.source must be a Slang path")
		validate_unique(source, seen_sources, f"{where}.source")
		require(
			operation.get("variant") in {"generic", "greedy", "dense", "top_k_top_p"},
			f"{where}.variant is unsupported",
		)
		dtype = operation.get("dtype", schema["dtype"])
		require(dtype in {"f32", "u32"}, f"{where}.dtype is unsupported")
		workgroup = operation.get("workgroup_size", schema["workgroup_size"])
		require(isinstance(workgroup, list) and len(workgroup) == 3 and all(isinstance(value, int) and value > 0 for value in workgroup) and math.prod(workgroup) <= 1024, f"{where}.workgroup_size is invalid")
		tile = operation.get("dispatch_tile_size", workgroup)
		require(isinstance(tile, list) and len(tile) == 3 and all(isinstance(value, int) and value > 0 for value in tile), f"{where}.dispatch_tile_size is invalid")
		semantic_operation = operation.get("semantic_operation")
		if semantic_operation is None:
			differentiation = operation.get("differentiation", "none")
			require(
				differentiation in {"none", "reverse"},
				f"{where}.differentiation is unsupported",
			)
			validate_ml_contract(operation.get("contract"), differentiation, where)
		else:
			require(semantic_operation in names, f"{where}.semantic_operation must name an RNG operation")
			require("contract" not in operation, f"{where} implementation variant cannot own another contract")
		validate_semantic_attributes(operation.get("semantic_attributes", []), where)
		replay_role = operation.get("training_replay_role", "safe")
		require(replay_role in {"safe", "frozen_rng", "replay_rng", "rng_state_advance"}, f"{where}.training_replay_role is unsupported")
		fields = operation.get("push_fields")
		require(isinstance(fields, list) and fields, f"{where}.push_fields must be non-empty")
		seen_fields: set[str] = set()
		for field in fields:
			require(isinstance(field, list) and len(field) == 2, f"{where}.push_fields entry is invalid")
			name, scalar_type = field
			require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.push field name is invalid")
			validate_unique(name, seen_fields, f"{where}.push field name")
			require(scalar_type in {"uint32", "float32"}, f"{where}.push field type is unsupported")
		require(len(fields) * 4 <= 128, f"{where}.push constants exceed Vulkan minimum guarantee")


def validate_index_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Index schema_version must be 1")
	require(schema.get("family") == "matrix_index", "Index family must be matrix_index")
	require(schema.get("domain") == "matrix", "Index domain must be matrix")
	require(schema.get("dtype") == "f32", "Index default dtype must be f32")
	require(schema.get("workgroup_size") == [256, 1, 1], "Index workgroup_size must be [256, 1, 1]")
	operations = schema.get("operations")
	require(isinstance(operations, list) and operations, "Index operations must be non-empty")
	require(
		[operation.get("name") for operation in operations if isinstance(operation, dict)]
		== [
			"top_k",
			"top_k_mask",
			"moe_expert_plan",
			"moe_routing_bias_update",
			"slice",
			"slice_backward",
			"repeat_interleave",
			"repeat_interleave_backward",
			"gather_last_dim",
			"gather_last_dim_backward",
			"concat",
			"equal",
		],
		"Index operations must preserve the admitted donor family order",
	)
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	seen_sources: set[str] = set()
	for index, operation in enumerate(operations):
		where = f"Index operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernels)):
			value = operation.get(key)
			require(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None, f"{where}.{key} must be an identifier")
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = operation.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id must be a non-zero u16")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		source = operation.get("source")
		require(isinstance(source, str) and source.startswith("src/slang/matrix/index/") and source.endswith(".slang"), f"{where}.source must be owned by matrix/index")
		validate_unique(source, seen_sources, f"{where}.source")
		require(operation.get("variant") == "generic", f"{where}.variant must be generic")
		require(operation.get("dtype", schema["dtype"]) in {"f32", "u32"}, f"{where}.dtype is unsupported")
		validate_semantic_attributes(operation.get("semantic_attributes", []), where)
		differentiation = operation.get("differentiation", "none")
		require(differentiation in {"none", "reverse"}, f"{where}.differentiation is invalid")
		validate_ml_contract(operation.get("contract"), differentiation, where)
		validate_physical_write(operation.get("physical_write"), where, required=False)
		fields = operation.get("push_fields")
		require(isinstance(fields, list) and fields, f"{where}.push_fields must be non-empty")
		seen_fields: set[str] = set()
		for field in fields:
			require(isinstance(field, list) and len(field) == 2, f"{where}.push_fields entry is invalid")
			name, scalar_type = field
			require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.push field name is invalid")
			validate_unique(name, seen_fields, f"{where}.push field name")
			require(
				scalar_type in {"uint32", "float32"},
				f"{where}.push field type is unsupported",
			)
		require(len(fields) * 4 <= 128, f"{where}.push constants exceed Vulkan minimum guarantee")


def validate_audio_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Audio schema_version must be 1")
	require(schema.get("family") == "audio", "Audio family must be audio")
	require(schema.get("domain") == "audio", "Audio domain must be audio")
	require(schema.get("dtype") == "f32", "Audio dtype must be f32")
	contracts = schema.get("contracts")
	require(isinstance(contracts, list) and contracts, "Audio contracts must be non-empty")
	contract_names: set[str] = set()
	for index, contract in enumerate(contracts):
		where = f"Audio contracts[{index}]"
		require(isinstance(contract, dict), f"{where} must be an object")
		require(
			set(contract)
			== {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes"},
			f"{where} fields are incomplete or unknown",
		)
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		require(
			isinstance(contract["input_kinds"], list)
			and contract["input_kinds"]
			and all(kind == "audio" for kind in contract["input_kinds"]),
			f"{where}.input_kinds must contain Audio values",
		)
		require(
			contract["output_kinds"] in (["audio"], ["matrix"]),
			f"{where}.output_kinds must contain one Audio or Matrix value",
		)
		require(contract["shape_rule"] in {"match_input", "explicit"}, f"{where}.shape_rule is invalid")
		require(contract["dtype_rule"] == "f32", f"{where}.dtype_rule must be f32")
		validate_semantic_attributes(contract["attributes"], where)
	kernels = schema.get("kernels")
	require(isinstance(kernels, list) and kernels, "Audio kernels must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernel_ids: set[str] = set()
	seen_sources: set[str] = set()
	for index, kernel in enumerate(kernels):
		where = f"Audio kernels[{index}]"
		require(isinstance(kernel, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernel_ids), ("source", seen_sources)):
			value = kernel.get(key)
			require(isinstance(value, str) and value, f"{where}.{key} must be non-empty")
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = kernel.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id is invalid")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		semantic = kernel.get("semantic_operation")
		require(semantic is None or semantic in contract_names, f"{where}.semantic_operation is unknown")
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
			isinstance(tile, list) and len(tile) == 3 and all(isinstance(value, int) and value > 0 for value in tile),
			f"{where}.dispatch_tile_size is invalid",
		)
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


def validate_cryptography_hash_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Cryptography Hash schema_version must be 1")
	require(schema.get("family") == "cryptography_hash", "Cryptography Hash family is invalid")
	require(schema.get("domain") == "cryptography::hash", "Cryptography Hash domain is invalid")
	require(schema.get("dtype") == "u8", "Cryptography Hash dtype must be u8")
	contracts = schema.get("contracts")
	require(isinstance(contracts, list) and contracts, "Cryptography Hash contracts must be non-empty")
	contract_names: set[str] = set()
	for index, contract in enumerate(contracts):
		where = f"Cryptography Hash contracts[{index}]"
		require(isinstance(contract, dict), f"{where} must be an object")
		require(
			set(contract) == {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes"},
			f"{where} fields are incomplete or unknown",
		)
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		require(contract["input_kinds"] == ["matrix"], f"{where}.input_kinds must contain one Matrix")
		require(contract["output_kinds"] == ["matrix"], f"{where}.output_kinds must contain one Matrix")
		require(contract["shape_rule"] in {"match_input", "explicit"}, f"{where}.shape_rule is invalid")
		require(contract["dtype_rule"] == "u8", f"{where}.dtype_rule must be u8")
		validate_semantic_attributes(contract["attributes"], where)
	kernels = schema.get("kernels")
	require(isinstance(kernels, list) and kernels, "Cryptography Hash kernels must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernel_ids: set[str] = set()
	seen_sources: set[str] = set()
	for index, kernel in enumerate(kernels):
		where = f"Cryptography Hash kernels[{index}]"
		require(isinstance(kernel, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernel_ids), ("source", seen_sources)):
			value = kernel.get(key)
			require(isinstance(value, str) and value, f"{where}.{key} must be non-empty")
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = kernel.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id is invalid")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		require(kernel.get("semantic_operation") in contract_names, f"{where}.semantic_operation is unknown")
		workgroup = kernel.get("workgroup_size")
		require(
			isinstance(workgroup, list)
			and len(workgroup) == 3
			and all(isinstance(value, int) and value > 0 for value in workgroup)
			and math.prod(workgroup) <= 1024,
			f"{where}.workgroup_size is invalid",
		)
		fields = kernel.get("push_fields")
		require(isinstance(fields, list) and fields, f"{where}.push_fields must be non-empty")
		seen_fields: set[str] = set()
		for field in fields:
			require(isinstance(field, list) and len(field) == 2, f"{where}.push_fields entry is invalid")
			name, scalar_type = field
			require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.push field name is invalid")
			validate_unique(name, seen_fields, f"{where}.push field name")
			require(scalar_type == "uint32", f"{where}.push field type must be uint32")
		require(len(fields) * 4 <= 128, f"{where}.push constants exceed Vulkan minimum guarantee")


def validate_image_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Image schema_version must be 1")
	require(schema.get("family") == "image", "Image family must be image")
	require(schema.get("domain") == "image", "Image domain must be image")
	require(schema.get("dtype") == "f32", "Image dtype must be f32")
	contracts = schema.get("contracts")
	expected_contracts = {
		"resize": ["image"],
		"crop": ["image"],
		"flip": ["image"],
		"rotate": ["image"],
		"pad": ["image"],
		"center_crop": ["image"],
		"remap": ["image", "matrix"],
		"warp_affine": ["image", "matrix"],
		"warp_perspective": ["image", "matrix"],
		"threshold_binary": ["image"],
		"threshold_binary_inv": ["image"],
		"threshold_truncate": ["image"],
		"threshold_to_zero": ["image"],
		"threshold_to_zero_inv": ["image"],
		"in_range": ["image"],
		"clamp": ["image"],
		"invert": ["image"],
		"brightness_contrast": ["image"],
		"gamma_contrast": ["image"],
		"solarize": ["image"],
		"posterize": ["image"],
		"grayscale": ["image"],
		"alpha_blend": ["image", "image"],
		"composite": ["image", "image", "image"],
		"erase": ["image"],
		"color_twist": ["image", "matrix"],
		"channel_reorder": ["image"],
		"gaussian_noise": ["image"],
		"salt_pepper_noise": ["image"],
		"convolve_2d": ["image", "matrix"],
		"separable_convolve_2d": ["image", "matrix", "matrix"],
		"average_blur": ["image"],
		"sobel": ["image"],
		"scharr": ["image"],
		"laplacian": ["image"],
		"erode": ["image"],
		"dilate": ["image"],
		"morphology_open": ["image"],
		"morphology_close": ["image"],
		"morphology_gradient": ["image"],
		"gaussian_blur": ["image"],
		"sharpen": ["image"],
		"median_blur": ["image"],
		"bilateral_filter": ["image"],
		"unsharp_mask": ["image"],
		"morphology_top_hat": ["image"],
		"morphology_black_hat": ["image"],
		"adaptive_threshold_mean": ["image"],
		"adaptive_threshold_gaussian": ["image"],
		"normalize": ["image"],
		"convert_color": ["image"],
		"resize_normalize": ["image"],
		"segmentation_overlay": ["image", "matrix", "matrix"],
	}
	require(
		isinstance(contracts, list)
		and [contract.get("name") for contract in contracts] == list(expected_contracts),
		"Image contracts must contain the complete geometric surface in canonical order",
	)
	contract_names: set[str] = set()
	for index, contract in enumerate(contracts):
		where = f"Image contracts[{index}]"
		require(isinstance(contract, dict), f"{where} must be an object")
		require(
			set(contract)
			== {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes"},
			f"{where} fields are incomplete or unknown",
		)
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		require(
			contract["input_kinds"] == expected_contracts[name],
			f"{where}.input_kinds are invalid",
		)
		require(contract["output_kinds"] == ["image"], f"{where}.output_kinds must contain one Image")
		require(contract["shape_rule"] == "explicit", f"{where}.shape_rule must be explicit")
		require(contract["dtype_rule"] in {"f32", "explicit"}, f"{where}.dtype_rule is invalid")
		validate_semantic_attributes(contract["attributes"], where)
	kernels = schema.get("kernels")
	require(isinstance(kernels, list) and kernels, "Image kernels must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernel_ids: set[str] = set()
	for index, kernel in enumerate(kernels):
		where = f"Image kernels[{index}]"
		require(isinstance(kernel, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernel_ids)):
			value = kernel.get(key)
			require(isinstance(value, str) and value, f"{where}.{key} must be non-empty")
			validate_unique(value, seen, f"{where}.{key}")
		require(isinstance(kernel.get("source"), str) and kernel["source"], f"{where}.source must be non-empty")
		stable_id = kernel.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id is invalid")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		require(kernel.get("semantic_operation") in contract_names, f"{where}.semantic_operation is unknown")
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
			set(contract)
			== {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes"},
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
	composites = schema.get("composite_contracts", [])
	require(isinstance(composites, list), "ML composite_contracts must be an array")
	validate_ml_port_provenance(schema.get("port_provenance"), operations + composites)
	contract_names = {
		operation.get("name")
		for operation in operations + composites
		if isinstance(operation, dict) and not operation.get("lowering_only", False)
	}
	contract_domains = {
		operation.get("name"): operation.get("semantic_domain")
		for operation in operations + composites
		if isinstance(operation, dict) and not operation.get("lowering_only", False)
	}
	seen_ids = validate_retired_stable_ids(
		schema.get("retired_stable_ids", []), "ML retired_stable_ids"
	)
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	seen_sources: set[str] = set()
	for index, operation in enumerate(operations):
		where = f"ML operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		validate_physical_write(
			operation.get("physical_write"),
			where,
			required=operation.get("name")
			in {"layer_norm", "layer_norm_backward", "rms_norm", "rms_norm_backward"},
		)
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
		variant = operation.get("variant")
		require(
			isinstance(variant, str) and IDENTIFIER.fullmatch(variant) is not None,
			f"{where}.variant must be an identifier",
		)
		operation_workgroup = operation.get("workgroup_size", schema["workgroup_size"])
		operation_dtype = operation.get("dtype", schema["dtype"])
		lowering_only = operation.get("lowering_only", False)
		require(isinstance(lowering_only, bool), f"{where}.lowering_only must be boolean")
		semantic_domain = operation.get("semantic_domain")
		if lowering_only:
			require(
				semantic_domain is None,
				f"{where} lowering-only kernel cannot own a semantic domain",
			)
		else:
			require(
				semantic_domain
				in {"ml::matrix", "ml::loss", "ml::optim", "ml::flow", "ml::advantage", "ml::rollout", "ml::replay", "ml::environment"},
				f"{where}.semantic_domain must name its public Rust operation owner",
			)
		semantic_operation = operation.get("semantic_operation")
		require(
			semantic_operation is None or semantic_operation in contract_names,
			f"{where}.semantic_operation must name a semantic ML operation",
		)
		require(
			lowering_only or semantic_operation is None,
			f"{where}.semantic_operation is only valid for a lowering-only kernel",
		)
		owner_domain = (
			contract_domains.get(semantic_operation) if lowering_only else semantic_domain
		)
		if owner_domain == "ml::loss":
			require(
				Path(source).name not in {"forward.slang", "backward.slang"},
				f"{where}.source must retain the loss name before _forward.slang or _backward.slang",
			)
		if owner_domain == "ml::optim" or source.startswith("src/slang/ml/optim/"):
			optimizer_family = (
				"grad_clip"
				if operation["name"].startswith("clip_grad_norm")
				else "adamw"
				if operation["name"].startswith("adamw")
				else "adam"
				if operation["name"] == "adam"
				else "sgd"
				if operation["name"].startswith("sgd")
				else "muon"
				if operation["name"].startswith("muon")
				else None
			)
			require(
				optimizer_family is not None
				and Path(source).parent == Path("src/slang/ml/optim") / optimizer_family,
				f"{where}.source must live under its optimizer-family directory",
			)
		require(operation_dtype in {"f32", "u8", "u32"}, f"{where}.dtype is unsupported")
		require(
			isinstance(operation_workgroup, list)
			and len(operation_workgroup) == 3
			and all(isinstance(value, int) and value > 0 for value in operation_workgroup)
			and math.prod(operation_workgroup) <= 1024,
			f"{where}.workgroup_size must contain three positive dimensions with at most 1024 threads",
		)
		differentiation = operation.get("differentiation")
		require(
			differentiation in {"none", "backward", "reverse"}
			or any(candidate.get("name") == differentiation for candidate in operations),
			f"{where}.differentiation must name an operation, reverse, backward, or none",
		)
		if lowering_only:
			require(differentiation == "none", f"{where} lowering-only kernel cannot differentiate")
			require("contract" not in operation, f"{where} lowering-only kernel cannot own a semantic contract")
			require("dnn" not in operation, f"{where} lowering-only kernel cannot own a DNN role")
		else:
			validate_ml_contract(operation.get("contract"), differentiation, where)
		validate_ml_autograd(
			operation.get("autograd"), differentiation, semantic_domain, lowering_only, where
		)
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
	for index, operation in enumerate(composites):
		where = f"ML composite_contracts[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		require(
			set(operation)
			== {
				"name",
				"semantic_domain",
				"differentiation",
				"autograd",
				"semantic_attributes",
				"contract",
			},
			f"{where} fields are incomplete or unknown",
		)
		name = operation["name"]
		require(
			isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None,
			f"{where}.name must be an identifier",
		)
		validate_unique(name, seen_names, f"{where}.name")
		semantic_domain = operation["semantic_domain"]
		require(
			semantic_domain in {"ml::advantage", "ml::environment", "ml::loss", "ml::policy"},
			f"{where}.semantic_domain must name an admitted composite owner",
		)
		differentiation = operation["differentiation"]
		require(differentiation == "reverse", f"{where}.differentiation must be reverse")
		validate_ml_contract(operation["contract"], differentiation, where)
		validate_ml_autograd(
			operation["autograd"], differentiation, semantic_domain, False, where
		)
		validate_semantic_attributes(operation["semantic_attributes"], where)


def validate_ml_autograd(
	autograd: Any,
	differentiation: str,
	semantic_domain: Any,
	lowering_only: bool,
	where: str,
) -> None:
	differentiable = not lowering_only and differentiation not in {"none", "backward"}
	if not differentiable:
		require(autograd is None, f"{where}.autograd is valid only for a differentiable forward operation")
		return
	require(isinstance(autograd, dict), f"{where}.autograd must own attachment policy")
	mode = autograd.get("mode")
	family = autograd.get("family")
	require(mode in {"generated", "manual"}, f"{where}.autograd.mode is unsupported")
	require(
		isinstance(family, str)
		and family
		and all(IDENTIFIER.fullmatch(part) is not None for part in family.split("/")),
		f"{where}.autograd.family must be an identifier path",
	)
	expected_root = {
		"ml::matrix": "matrix/",
		"ml::loss": "loss/",
		"ml::flow": "flow/",
		"ml::advantage": "advantage/",
		"ml::environment": "environment/",
		"ml::policy": "policy/",
	}.get(semantic_domain)
	require(
		expected_root is not None and family.startswith(expected_root),
		f"{where}.autograd.family must match its semantic owner",
	)
	if mode == "manual":
		require(
			set(autograd) == {"mode", "family"},
			f"{where}.autograd manual policy has unknown fields",
		)
		return
	require(
		set(autograd) in (
			{"mode", "family", "node", "inputs", "saved_matrices"},
			{"mode", "family", "node", "inputs", "saved_matrices", "saved_scalars"},
		),
		f"{where}.autograd generated policy fields are incomplete or unknown",
	)
	node = autograd["node"]
	require(
		isinstance(node, str)
		and IDENTIFIER.fullmatch(node) is not None
		and node[0].isupper(),
		f"{where}.autograd.node must be a PascalCase identifier",
	)
	saved = autograd["saved_matrices"]
	inputs = autograd["inputs"]
	require(
		isinstance(inputs, list)
		and inputs
		and len(inputs) == len(set(inputs))
		and all(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None for name in inputs),
		f"{where}.autograd.inputs must contain unique identifiers",
	)
	require(
		isinstance(saved, list)
		and saved
		and len(saved) == len(set(saved))
		and all(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None for name in saved),
		f"{where}.autograd.saved_matrices must contain unique identifiers",
	)
	scalars = autograd.get("saved_scalars", [])
	require(isinstance(scalars, list), f"{where}.autograd.saved_scalars must be an array")
	seen_scalars: set[str] = set()
	for index, scalar in enumerate(scalars):
		label = f"{where}.autograd.saved_scalars[{index}]"
		require(
			isinstance(scalar, list)
			and len(scalar) == 2
			and isinstance(scalar[0], str)
			and IDENTIFIER.fullmatch(scalar[0]) is not None
			and scalar[1] in {"f32", "usize"},
			f"{label} must contain an identifier and supported Rust scalar type",
		)
		validate_unique(scalar[0], seen_scalars, f"{label} name")
		require(scalar[0] not in saved, f"{label} collides with a saved matrix")


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
					and donor.startswith(("source/", "tools/", "sdk/"))
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
	base_fields = {
		"input_kinds",
		"output_kinds",
		"shape_rule",
		"dtype_rule",
		"effects",
		"mutated_inputs",
		"output_alias_inputs",
		"lowering",
	}
	variadic_fields = {"variadic_input", "variadic_output", "aligned_variadic_aliases"}
	optional_fields = {"optional_inputs"}
	extra_fields = set(contract) - base_fields
	require(
		set(contract).issuperset(base_fields)
		and extra_fields.issubset(variadic_fields | optional_fields),
		f"{where}.contract fields are incomplete or unknown",
	)
	inputs = contract["input_kinds"]
	outputs = contract["output_kinds"]
	variadic_input = contract.get("variadic_input")
	variadic_output = contract.get("variadic_output")
	variadic = variadic_input is not None or variadic_output is not None
	require(
		isinstance(inputs, list) and (inputs or variadic_input is not None),
		f"{where}.contract.input_kinds must be non-empty without a variadic input",
	)
	require(
		isinstance(outputs, list) and (outputs or variadic_output is not None),
		f"{where}.contract.output_kinds must be non-empty without a variadic output",
	)
	require(
		all(isinstance(value, str) and value for value in [*inputs, *outputs]),
		f"{where}.contract input/output kinds must be strings",
	)
	for label, value in (("variadic_input", variadic_input), ("variadic_output", variadic_output)):
		if value is not None:
			require(
				isinstance(value, dict)
				and set(value) == {"kind", "minimum"}
				and isinstance(value["kind"], str)
				and value["kind"]
				and isinstance(value["minimum"], int)
				and not isinstance(value["minimum"], bool)
				and value["minimum"] > 0,
				f"{where}.contract.{label} must declare a kind and positive minimum",
			)
	if "aligned_variadic_aliases" in contract:
		require(
			isinstance(contract["aligned_variadic_aliases"], bool),
			f"{where}.contract.aligned_variadic_aliases must be boolean",
		)
	if contract.get("aligned_variadic_aliases", False):
		require(
			variadic_input is not None and variadic_input == variadic_output,
			f"{where}.contract aligned variadic aliases require identical input/output tails",
		)
	optional_inputs = contract.get("optional_inputs", [])
	require(
		isinstance(optional_inputs, list)
		and all(
			isinstance(index, int)
			and not isinstance(index, bool)
			and 0 <= index < len(inputs)
			for index in optional_inputs
		)
		and len(set(optional_inputs)) == len(optional_inputs),
		f"{where}.contract.optional_inputs must contain unique fixed input indices",
	)
	require(
		isinstance(contract["shape_rule"], str) and contract["shape_rule"],
		f"{where}.contract.shape_rule must be non-empty",
	)
	require(
		contract["dtype_rule"]
		in {
			"explicit",
			"f32",
			"all_f32",
			"f32_logits_u32_or_i32_targets_f32_mask",
			"f32_logits_u8_u32_or_i32_labels_u32_output",
			"f32_logits_u8_u32_or_i32_labels_f32_mask_u32_output",
			"u32_state",
			"f32_parameter_u32_state",
			"f32_logits_u32_targets",
			"f32_logits_u32_or_i32_targets",
			"f32_weight_u32_indices",
			"f32_with_u32_indices",
			"u32_indices_f32_gradient_weight",
			"f32_probabilities_i32_indices",
			"f32_gradients_probabilities_route_weights_i32_indices",
			"f32_values_u8_boundaries",
			"f32_values_i32_indices",
			"f32_logits_i32_indices",
			"categorical_rollout_sources_and_time_major_storage",
			"u8_validity_mask",
			"replay_transition_and_storage",
			"replay_storage_to_sample_batch",
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
		and all(value == -1 or 0 <= value < len(inputs) for value in aliases)
		and all(aliases.count(value) == 1 for value in mutated),
		f"{where}.contract.output_alias_inputs must reference inputs and map every mutated input exactly once",
	)
	if variadic:
		require(
			not mutated and all(alias == -1 for alias in aliases),
			f"{where}.contract variadic aliases cannot duplicate fixed mutation metadata",
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
		"differentiation": "reverse",
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


def operation_lowering_variants(operation: dict[str, Any]) -> list[dict[str, Any]]:
	return operation.get("additional_lowering_variants", [])


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
		elif dtype == "i32":
			require(all(isinstance(value, int) and not isinstance(value, bool) and -(2**31) <= value < 2**31 for value in values), f"{where}.test.{key} must contain i32 values")
		elif dtype == "u8":
			require(all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**8 for value in values), f"{where}.test.{key} must contain u8 values")
		else:
			require(all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**32 for value in values), f"{where}.test.{key} must contain u32 values")
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


def validate_retired_stable_ids(value: Any, label: str) -> set[int]:
	require(isinstance(value, list), f"{label} must be an array")
	retired: set[int] = set()
	for index, stable_id in enumerate(value):
		require(
			isinstance(stable_id, int) and 0 < stable_id <= 65535,
			f"{label}[{index}] must be a non-zero u16",
		)
		validate_unique(stable_id, retired, f"{label}[{index}]")
	return retired


def require(condition: bool, message: str) -> None:
	if not condition:
		raise SchemaError(message)


def banner(schema_hash: str, prefix: str, schema_name: str = "matrix_elemwise.json") -> str:
	return (
		f"{prefix} @generated by tools/gen/fn/generate.py; DO NOT EDIT.\n"
		f"{prefix} schema={schema_name} schema_version=1 generator_version={GENERATOR_VERSION}\n"
		f"{prefix} schema_sha256={schema_hash}\n"
	)


def registry_banner(
	elementwise_hash: str,
	blas_hash: str,
	reduce_hash: str,
	rng_hash: str,
	index_hash: str,
	ml_hash: str,
	audio_hash: str,
	cryptography_hash: str,
	image_hash: str,
	vision_hash: str,
) -> str:
	return (
		"// @generated by tools/gen/fn/generate.py; DO NOT EDIT.\n"
		f"// schemas=matrix_elemwise.json,matrix_blas.json,matrix_reduce.json,matrix_rng.json,matrix_index.json,ml_training.json,audio.json,cryptography_hash.json,image.json,vision_detection.json generator_version={GENERATOR_VERSION}\n"
		f"// matrix_elemwise_sha256={elementwise_hash}\n"
		f"// matrix_blas_sha256={blas_hash}\n"
		f"// matrix_reduce_sha256={reduce_hash}\n"
		f"// matrix_rng_sha256={rng_hash}\n"
		f"// matrix_index_sha256={index_hash}\n"
		f"// ml_training_sha256={ml_hash}\n"
		f"// audio_sha256={audio_hash}\n"
		f"// cryptography_hash_sha256={cryptography_hash}\n"
		f"// image_sha256={image_hash}\n"
		f"// vision_detection_sha256={vision_hash}\n"
	)


def rust_float(value: int | float) -> str:
	bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
	return f"f32::from_bits(0x{bits:08x})"


def rust_value(dtype: str, value: int | float) -> str:
	if dtype == "f32":
		return rust_float(value)
	if dtype == "i32":
		return f"{value}_i32"
	if dtype == "u8":
		return f"{value}_u8"
	if dtype == "u32":
		return f"{value}_u32"
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
			if operation.get("shape_rule") == "broadcast":
				lines.extend(["///", "/// Shapes use multidirectional broadcasting over at most eight axes."])
			else:
				lines.extend(["///", "/// Shapes must match exactly; broadcasting is not yet supported."])
		if any("integer_overflow" in variant for variant in operation_variants(schema, operation)) and operation["expression"] != "input":
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
	elementwise: dict[str, Any],
	blas: dict[str, Any],
	reduce: dict[str, Any],
	rng: dict[str, Any],
	index: dict[str, Any],
	ml: dict[str, Any],
	audio: dict[str, Any],
	cryptography: dict[str, Any],
	image: dict[str, Any],
	vision: dict[str, Any],
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
					"physical_write": operation.get("physical_write"),
					"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(operation['name'])}",
				}
			)
		for lowering in operation_lowering_variants(operation):
			entries.append(
				{
					"domain": "matrix",
					"name": lowering["name"],
					"dtype": lowering["dtype"],
					"kernel_id": lowering["kernel_id"],
					"stable_id": lowering["stable_id"],
					"workgroup_size": lowering.get("workgroup_size", elementwise["workgroup_size"]),
					"dispatch_tile_size": lowering.get("dispatch_tile_size", lowering.get("workgroup_size", elementwise["workgroup_size"])),
					"training_replay_role": "safe",
					"physical_write": lowering["physical_write"],
					"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(operation['name'])}",
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
				"physical_write": operation.get("physical_write"),
				"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(operation['name'])}",
			}
		)
	for operation in reduce["operations"]:
		semantic = operation.get("semantic_operation", operation["name"])
		entries.append(
			{
				"domain": "matrix",
				"name": operation["name"],
				"dtype": reduce["dtype"],
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": reduce["workgroup_size"],
				"dispatch_tile_size": [1, 1, 1],
				"training_replay_role": "safe",
				"physical_write": operation.get("physical_write"),
				"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(semantic)}",
			}
		)
	for operation in rng["operations"]:
		name = operation["name"]
		semantic = operation.get("semantic_operation", name)
		entries.append(
			{
				"domain": "matrix",
				"name": name,
				"dtype": operation.get("dtype", rng["dtype"]),
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": operation.get("workgroup_size", rng["workgroup_size"]),
				"dispatch_tile_size": operation.get(
					"dispatch_tile_size", operation.get("workgroup_size", rng["workgroup_size"])
				),
				"training_replay_role": operation.get("training_replay_role", "safe"),
				"physical_write": operation.get("physical_write"),
				"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(semantic)}",
			}
		)
	for operation in index["operations"]:
		entries.append(
			{
				"domain": "matrix",
				"name": operation["name"],
				"dtype": operation.get("dtype", index["dtype"]),
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": operation.get("workgroup_size", index["workgroup_size"]),
				"dispatch_tile_size": operation.get(
					"dispatch_tile_size", operation.get("workgroup_size", index["workgroup_size"])
				),
				"training_replay_role": "safe",
				"physical_write": operation.get("physical_write"),
				"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(operation['name'])}",
			}
		)
	for operation in ml["operations"]:
		dtype = operation.get("dtype", ml["dtype"])
		semantic = operation.get("semantic_operation", operation["name"])
		entries.append(
			{
				"domain": "ml",
				"report_domain": operation.get("semantic_domain", "ml").replace("::", "."),
				"name": operation["name"],
				"dtype": dtype,
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": operation.get("workgroup_size", ml["workgroup_size"]),
				"dispatch_tile_size": operation.get(
					"dispatch_tile_size", operation.get("workgroup_size", ml["workgroup_size"])
				),
				"training_replay_role": operation.get("training_replay_role", "safe"),
				"physical_write": operation.get("physical_write"),
				"semantic_contract": (
					None
					if operation.get("lowering_only", False) and "semantic_operation" not in operation
					else f"crate::core::operation::ml::{rust_const_name(semantic)}"
				),
			}
		)
	for kernel in audio["kernels"]:
		semantic = kernel.get("semantic_operation")
		entries.append(
			{
				"domain": "audio",
				"name": kernel["name"],
				"dtype": audio["dtype"],
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel.get("physical_write"),
				"semantic_contract": (
					None
					if semantic is None
					else f"crate::core::operation::audio::{rust_const_name(semantic)}"
				),
			}
		)
	for kernel in cryptography["kernels"]:
		semantic = kernel["semantic_operation"]
		entries.append(
			{
				"domain": "cryptography",
				"report_domain": "cryptography.hash",
				"name": kernel["name"],
				"dtype": cryptography["dtype"],
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel.get("physical_write"),
				"semantic_contract": f"crate::core::operation::cryptography::hash::{rust_const_name(semantic)}",
			}
		)
	for kernel in image["kernels"]:
		semantic = kernel["semantic_operation"]
		entries.append(
			{
				"domain": "image",
				"name": kernel["name"],
				"dtype": image["dtype"],
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel["physical_write"],
				"semantic_contract": f"crate::core::operation::image::{rust_const_name(semantic)}",
			}
		)
	for kernel in vision["kernels"]:
		semantic = kernel["semantic_operation"]
		entries.append(
			{
				"domain": "vision",
				"name": kernel["name"],
				"dtype": kernel.get("dtype", vision["dtype"]),
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel["physical_write"],
				"semantic_contract": f"crate::core::operation::vision::{rust_const_name(semantic)}",
			}
		)
	seen_ids = validate_retired_stable_ids(
		ml.get("retired_stable_ids", []), "ML retired_stable_ids"
	)
	seen_kernels: set[str] = set()
	for entry in entries:
		validate_physical_write(
			entry["physical_write"], f"kernel candidate {entry['kernel_id']}", required=False
		)
		validate_unique(entry["stable_id"], seen_ids, "stable kernel ID across schemas")
		validate_unique(entry["kernel_id"], seen_kernels, "kernel ID across schemas")
	return entries


def rust_physical_write(value: dict[str, Any] | None) -> str:
	if value is None:
		return "None"
	domains = {
		"axis_slices": "AxisSlices",
		"output_elements": "OutputElements",
		"rows": "Rows",
		"scalar": "Scalar",
	}
	partitions = {
		"exclusive_per_invocation": "ExclusivePerInvocation",
		"exclusive_per_workgroup": "ExclusivePerWorkgroup",
		"shared_atomic_contributors": "SharedAtomicContributors",
	}
	extents = {
		"axis_slice": "AxisSlice",
		"one_element": "OneElement",
		"one_scalar": "OneScalar",
		"row_width": "RowWidth",
		"tile_16x16": "Tile16x16",
		"up_to_two_elements": "UpToTwoElements",
		"up_to_four_elements": "UpToFourElements",
	}
	collisions = {
		"exclusive": "Exclusive",
		"atomic_u32": "AtomicU32",
	}
	tails = {"bounds_checked": "BoundsChecked", "exact": "Exact"}
	workspaces = {"exclusive_per_workgroup": "ExclusivePerWorkgroup", "none": "None"}
	writes = ", ".join(
		"PhysicalWrite { "
		f"binding: {write['binding']}, "
		f"domain: LogicalWriteDomain::{domains[write['domain']]}, "
		f"partition: WritePartition::{partitions[write['partition']]}, "
		f"extent: WriteExtent::{extents[write['extent']]}, "
		f"collision: CollisionPolicy::{collisions[write['collision']]}, "
		f"tail: TailPolicy::{tails[write['tail']]} "
		"}"
		for write in value["writes"]
	)
	return (
		"Some(PhysicalWriteContract { "
		f"writes: &[{writes}], "
		f"workspace: WorkspacePartition::{workspaces[value['workspace']]} "
		"})"
	)


def generate_registry(
	elementwise: dict[str, Any],
	elementwise_hash: str,
	blas: dict[str, Any],
	blas_hash: str,
	reduce: dict[str, Any],
	reduce_hash: str,
	rng: dict[str, Any],
	rng_hash: str,
	index: dict[str, Any],
	index_hash: str,
	ml: dict[str, Any],
	ml_hash: str,
	audio: dict[str, Any],
	audio_hash: str,
	cryptography: dict[str, Any],
	cryptography_hash: str,
	image: dict[str, Any],
	image_hash: str,
	vision: dict[str, Any],
	vision_hash: str,
) -> str:
	entries = registry_entries(elementwise, blas, reduce, rng, index, ml, audio, cryptography, image, vision)
	lines = [
		registry_banner(
			elementwise_hash,
			blas_hash,
			reduce_hash,
			rng_hash,
			index_hash,
			ml_hash,
			audio_hash,
			cryptography_hash,
			image_hash,
			vision_hash,
		).rstrip(),
		"",
		"use std::sync::OnceLock;",
		"",
		"use super::{CollisionPolicy, LogicalWriteDomain, PhysicalWrite, PhysicalWriteContract, ShaderArtifact, TailPolicy, WorkspacePartition, WriteExtent, WritePartition};",
		"",
	]
	for entry in entries:
		domain = entry["domain"]
		name = entry["name"]
		dtype = entry["dtype"]
		workgroup = entry["workgroup_size"]
		dispatch_tile = entry["dispatch_tile_size"]
		physical_write = rust_physical_write(entry["physical_write"])
		lines.extend(
			[
				f"static {static_name(domain, name, dtype)}: ShaderArtifact = ShaderArtifact {{",
				f'\tbytes: include_bytes!(concat!(env!("OUT_DIR"), "/{domain}_{name}_{dtype}.spv")),',
				f"\tworkgroup_size: [{workgroup[0]}, {workgroup[1]}, {workgroup[2]}],",
				f"\tdispatch_tile_size: [{dispatch_tile[0]}, {dispatch_tile[1]}, {dispatch_tile[2]}],",
				f"\tphysical_write: {physical_write},",
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
			"\tFrozenRng,",
			"\tReplayRng,",
			"\tRngStateAdvance,",
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
		"frozen_rng": "FrozenRng",
		"replay_rng": "ReplayRng",
		"rng_state_advance": "RngStateAdvance",
	}
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn semantic_contract(self) -> Option<crate::OperationContract> {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		contract = entry["semantic_contract"]
		value = "None" if contract is None else f"Some({contract})"
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => {value},")
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
		name = f"{entry.get('report_domain', entry['domain'])}.{entry['name']}.{entry['dtype']}"
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
	dtype_tokens = {"f32": "float32", "i32": "int32", "u32": "uint32", "u8": "uint8"}
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
	kind_names = {
		"matrix": "Matrix",
		"parameter": "Matrix",
		"audio": "Audio",
		"image": "Image",
	}
	inputs = ", ".join(f"OpValueKind::{kind_names.get(kind, 'Matrix')}" for kind in contract["input_kinds"])
	outputs = ", ".join(f"OpValueKind::{kind_names.get(kind, 'Matrix')}" for kind in contract["output_kinds"])
	shape_rule = (
		"OpShapeRule::MatMulNt"
		if contract["shape_rule"] == "left_mk_right_nk_to_mn"
		else "OpShapeRule::Broadcast"
		if contract["shape_rule"] == "broadcast"
		else "OpShapeRule::MatchInput"
		if contract["shape_rule"] in {"equal_no_broadcast", "preserve_input", "match_input"}
		else "OpShapeRule::Explicit"
	)
	dtype_rule = (
		"OpDTypeRule::MatchInput"
		if contract["dtype_rule"] in {"same_admitted_dtype", "all_f32", "matching_f32", "f32"}
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
	]
	if "variadic_input" in contract:
		variadic_input = contract["variadic_input"]
		lines.append(
			f".variadic_inputs(OpValueKind::{kind_names.get(variadic_input['kind'], 'Matrix')}, {variadic_input['minimum']})"
		)
	if "variadic_output" in contract:
		variadic_output = contract["variadic_output"]
		lines.append(
			f".variadic_outputs(OpValueKind::{kind_names.get(variadic_output['kind'], 'Matrix')}, {variadic_output['minimum']})"
		)
	if contract.get("aligned_variadic_aliases", False):
		lines.append(".aligned_variadic_aliases()")
	lines.extend(
		[
			f".with_shape_rule({shape_rule})",
			f".with_dtype_rule({dtype_rule})",
			f".with_differentiation({differentiation})",
			f".with_lowering({lowering})",
			".effects(OpEffect::READ_INPUTS.union(OpEffect::WRITE_OUTPUTS))",
		]
	)
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
	optional_mask = sum(1 << input_index for input_index in contract.get("optional_inputs", []))
	if optional_mask:
		lines.append(f".optional_inputs(0x{optional_mask:02x})")
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
	reduce: dict[str, Any],
	reduce_hash: str,
	rng: dict[str, Any],
	rng_hash: str,
	index: dict[str, Any],
	index_hash: str,
	ml: dict[str, Any],
	ml_hash: str,
	audio: dict[str, Any],
	audio_hash: str,
	cryptography: dict[str, Any],
	cryptography_hash: str,
	image: dict[str, Any],
	image_hash: str,
	vision: dict[str, Any],
	vision_hash: str,
) -> str:
	lines = [
		banner(elementwise_hash, "//").rstrip(),
		f"// matrix_blas_sha256={blas_hash}",
		f"// matrix_reduce_sha256={reduce_hash}",
		f"// matrix_rng_sha256={rng_hash}",
		f"// matrix_index_sha256={index_hash}",
		f"// ml_training_sha256={ml_hash}",
		f"// audio_sha256={audio_hash}",
		f"// cryptography_hash_sha256={cryptography_hash}",
		f"// image_sha256={image_hash}",
		f"// vision_detection_sha256={vision_hash}",
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
		contract["shape_rule"] = operation.get("shape_rule", contract["shape_rule"])
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
	for operation in reduce["operations"]:
		if operation.get("lowering_only", False):
			continue
		contract = dict(operation["contract"])
		contract["differentiation"] = operation["differentiation"]
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation["semantic_attributes"]
		]
		for line in semantic_contract_lines("matrix", operation["name"], contract, attributes):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	for operation in rng["operations"]:
		if "contract" not in operation:
			continue
		contract = dict(operation["contract"])
		contract["differentiation"] = operation.get("differentiation", "none")
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation.get("semantic_attributes", [])
		]
		for line in semantic_contract_lines("matrix", operation["name"], contract, attributes):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	for operation in index["operations"]:
		contract = dict(operation["contract"])
		contract["differentiation"] = operation.get("differentiation", "none")
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation.get("semantic_attributes", [])
		]
		for line in semantic_contract_lines("matrix", operation["name"], contract, attributes):
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
		for line in semantic_contract_lines(
			operation["semantic_domain"], operation["name"], contract, attributes
		):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	for operation in ml.get("composite_contracts", []):
		contract = dict(operation["contract"])
		contract["differentiation"] = operation["differentiation"]
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation["semantic_attributes"]
		]
		for line in semantic_contract_lines(
			operation["semantic_domain"], operation["name"], contract, attributes
		):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend(["}", "", "/// Generated semantic contracts for Audio operations.", "pub mod audio {", "\tuse super::*;", ""])
	for operation in audio["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1] * len(operation["output_kinds"]),
			"differentiation": "none",
			"lowering": "compute_dispatch",
		}
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("audio", operation["name"], contract, attributes):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend([
		"}",
		"",
		"/// Generated semantic contracts for cryptographic batch hashing.",
		"pub mod cryptography {",
		"\tuse super::*;",
		"",
		"\tpub mod hash {",
		"\t\tuse super::*;",
		"",
	])
	for operation in cryptography["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1],
			"differentiation": "none",
			"lowering": "compute_dispatch",
		}
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("cryptography::hash", operation["name"], contract, attributes):
			lines.append(f"\t\t{line}" if line else "")
		lines.append("")
	lines.extend(["\t}", "}", "", "/// Generated semantic contracts for Image operations.", "pub mod image {", "\tuse super::*;", ""])
	for operation in image["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1],
			"differentiation": "none",
			"lowering": "compute_dispatch",
		}
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("image", operation["name"], contract, attributes):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend(["}", ""])
	lines.extend(["/// Generated semantic contracts for Vision operations.", "pub mod vision {", "\tuse super::*;", ""])
	for operation in vision["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1] * len(operation["output_kinds"]),
			"differentiation": "none",
			"lowering": "compute_dispatch",
		}
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("vision", operation["name"], contract, attributes):
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
	reduce: dict[str, Any],
	reduce_hash: str,
	rng_hash: str,
	ml: dict[str, Any],
	ml_hash: str,
) -> str:
	lines = [
		banner(elementwise_hash, "//").rstrip(),
		f"// matrix_blas_sha256={blas_hash}",
		f"// matrix_reduce_sha256={reduce_hash}",
		f"// matrix_rng_sha256={rng_hash}",
		f"// ml_training_sha256={ml_hash}",
		"// Candidate roles mirror OA DNN vocabulary while retaining OARS compatibility identities.",
		"",
		"pub(super) static DNN_OP_ROLES: &[DnnOpRole] = &[",
	]
	for domain, schema in (
		("matrix", elementwise),
		("matrix", blas),
		("matrix", reduce),
		("ml", ml),
	):
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


def generate_ml_autograd_attachments(
	ml: dict[str, Any], ml_hash: str, family: str
) -> str:
	operations = [
		operation
		for operation in ml["operations"]
		if operation.get("autograd", {}).get("mode") == "generated"
		and operation["autograd"]["family"] == family
	]
	lines = [
		banner(ml_hash, "//", "ml_training.json").rstrip(),
		"// Mechanical tape attachments; concrete saved-state node behavior remains private.",
		"",
		"use crate::{Matrix, Result};",
		"",
		"use super::{node::Node, tape::record_node};",
		"",
	]
	for operation in operations:
		autograd = operation["autograd"]
		inputs = autograd["inputs"]
		saved = autograd["saved_matrices"]
		scalars = autograd.get("saved_scalars", [])
		matrices = [*inputs, *(name for name in saved if name not in inputs)]
		arguments = ", ".join(
			[
				*(f"{name}: &Matrix" for name in matrices),
				*(f"{name}: {scalar_type}" for name, scalar_type in scalars),
				"result: &Matrix",
			]
		)
		lines.extend(
			[
				f"pub(in crate::ml) fn record_{operation['name']}({arguments}) -> Result<()> {{",
				f"\trecord_node(Node::{autograd['node']} {{",
			]
		)
		for name in matrices:
			lines.append(f"\t\t{name}: {name}.clone(),")
		for name, _ in scalars:
			lines.append(f"\t\t{name},")
		lines.extend(
			[
				"\t\toutput_id: result.value_id(),",
				"\t})",
				"}",
				"",
			]
		)
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


def generate_reduce_api(schema: dict[str, Any], schema_hash: str) -> str:
	(
		softmax,
		softmax_backward,
		log_softmax,
		log_softmax_backward,
		sum_operation,
		_,
		sum_backward,
		accuracy_count,
		masked_accuracy_count,
	) = schema["operations"]
	return f"""{banner(schema_hash, '//', 'matrix_reduce.json').rstrip()}

/// Stable Softmax over `dim`, where `-1` selects the last dimension.
///
/// The operation records asynchronously and returns its output immediately.
///
/// # Errors
///
/// Returns an error when the input is empty or not FP32, `dim` is invalid, an
/// axis extent or dispatch count exceeds the shader ABI, allocation fails, or
/// runtime recording fails.
pub fn {softmax['name']}(input: &Matrix, dim: i32) -> Result<Matrix> {{
	softmax_impl(
		input,
		dim,
		crate::core::operation::matrix::{rust_const_name(softmax['name'])},
	)
}}

pub(crate) fn {softmax_backward['name']}(
	forward_output: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
) -> Result<Matrix> {{
	softmax_backward_impl(
		forward_output,
		output_gradient,
		dim,
		crate::core::operation::matrix::{rust_const_name(softmax_backward['name'])},
	)
}}

/// Stable LogSoftmax over `dim`, where `-1` selects the last dimension.
///
/// The operation records asynchronously and returns its output immediately.
///
/// # Errors
///
/// Returns an error when the input is empty or not FP32, `dim` is invalid, an
/// axis extent or dispatch count exceeds the shader ABI, allocation fails, or
/// runtime recording fails.
pub fn {log_softmax['name']}(input: &Matrix, dim: i32) -> Result<Matrix> {{
	log_softmax_impl(
		input,
		dim,
		crate::core::operation::matrix::{rust_const_name(log_softmax['name'])},
	)
}}

pub(crate) fn {log_softmax_backward['name']}(
	forward_output: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
) -> Result<Matrix> {{
	log_softmax_backward_impl(
		forward_output,
		output_gradient,
		dim,
		crate::core::operation::matrix::{rust_const_name(log_softmax_backward['name'])},
	)
}}

/// Sum all values when `dim` is `-1`, or keep and reduce one selected axis.
///
/// A full reduction returns shape `[1]`; an axis reduction preserves rank and
/// replaces the selected extent with one. The operation records asynchronously.
///
/// # Errors
///
/// Returns an error when the input is empty or not FP32, `dim` is invalid, a
/// size exceeds the admitted shader ABI, allocation fails, or recording fails.
pub fn {sum_operation['name']}(input: &Matrix, dim: i32) -> Result<Matrix> {{
	sum_impl(
		input,
		dim,
		crate::core::operation::matrix::{rust_const_name(sum_operation['name'])},
	)
}}

pub(crate) fn {sum_backward['name']}(
	input: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
) -> Result<Matrix> {{
	sum_backward_impl(
		input,
		output_gradient,
		dim,
		crate::core::operation::matrix::{rust_const_name(sum_backward['name'])},
	)
}}

/// Count rows whose last-axis FP32 argmax equals the integer class label.
///
/// The returned `[1]` U32 Matrix remains device-resident. Reading it is an
/// explicit synchronization boundary.
///
/// # Errors
///
/// Returns an error when logits are not a nonempty rank-two-or-greater FP32
/// Matrix, labels do not match the leading logits shape or use U8/U32/I32,
/// inputs belong to different engines, a size exceeds the shader ABI,
/// allocation fails, or recording fails.
pub fn {accuracy_count['name']}(logits: &Matrix, labels: &Matrix) -> Result<Matrix> {{
	categorical_accuracy_count_impl(
		logits,
		labels,
		None,
		crate::core::operation::matrix::{rust_const_name(accuracy_count['name'])},
	)
}}

/// Count correct categorical predictions only where the FP32 mask is nonzero.
///
/// The returned `[1]` U32 Matrix remains device-resident. Reading it is an
/// explicit synchronization boundary.
///
/// # Errors
///
/// Returns an error under the same conditions as
/// [`categorical_accuracy_count`], or when the mask is not FP32, does not match
/// the labels shape, or belongs to another engine.
pub fn {masked_accuracy_count['name']}(
	logits: &Matrix,
	labels: &Matrix,
	mask: &Matrix,
) -> Result<Matrix> {{
	categorical_accuracy_count_impl(
		logits,
		labels,
		Some(mask),
		crate::core::operation::matrix::{rust_const_name(masked_accuracy_count['name'])},
	)
}}
"""


def generate_reduce_test(schema: dict[str, Any], schema_hash: str) -> str:
	softmax = schema["operations"][0]
	log_softmax = schema["operations"][2]
	sum_operation = schema["operations"][4]
	accuracy_count = schema["operations"][7]
	masked_accuracy_count = schema["operations"][8]
	shape = softmax["test"]["shape"]
	dim = softmax["test"]["dim"]
	tolerance = rust_float(softmax["test"]["tolerance"])
	return f"""{banner(schema_hash, '//', 'matrix_reduce.json').rstrip()}

fn softmax_reference(input: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {{
	let outer: usize = shape[..dim].iter().product();
	let width = shape[dim];
	let inner: usize = shape[dim + 1..].iter().product();
	let mut output = vec![0.0; input.len()];
	for outer_index in 0..outer {{
		for inner_index in 0..inner {{
			let base = outer_index * width * inner + inner_index;
			let maximum = (0..width)
				.map(|axis_index| input[base + axis_index * inner])
				.fold(f32::NEG_INFINITY, f32::max);
			let sum: f32 = (0..width)
				.map(|axis_index| (input[base + axis_index * inner] - maximum).exp())
				.sum();
			for axis_index in 0..width {{
				let index = base + axis_index * inner;
				output[index] = (input[index] - maximum).exp() / sum;
			}}
		}}
	}}
	output
}}

fn log_softmax_reference(input: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {{
	let outer: usize = shape[..dim].iter().product();
	let width = shape[dim];
	let inner: usize = shape[dim + 1..].iter().product();
	let mut output = vec![0.0; input.len()];
	for outer_index in 0..outer {{
		for inner_index in 0..inner {{
			let base = outer_index * width * inner + inner_index;
			let maximum = (0..width)
				.map(|axis_index| input[base + axis_index * inner])
				.fold(f32::NEG_INFINITY, f32::max);
			let sum: f32 = (0..width)
				.map(|axis_index| (input[base + axis_index * inner] - maximum).exp())
				.sum();
			let log_normalizer = maximum + sum.ln();
			for axis_index in 0..width {{
				let index = base + axis_index * inner;
				output[index] = input[index] - log_normalizer;
			}}
		}}
	}}
	output
}}

fn rejected(result: oa::Result<oa::Matrix>, message: &str) -> oa::Error {{
	match result {{
		Err(error) => error,
		Ok(_) => panic!("{{message}}"),
	}}
}}

fn sum_reference(input: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {{
	let outer: usize = shape[..dim].iter().product();
	let width = shape[dim];
	let inner: usize = shape[dim + 1..].iter().product();
	let mut output = vec![0.0; outer * inner];
	for outer_index in 0..outer {{
		for inner_index in 0..inner {{
			let base = outer_index * width * inner + inner_index;
			for axis_index in 0..width {{
				output[outer_index * inner + inner_index] +=
					input[base + axis_index * inner];
			}}
		}}
	}}
	output
}}

test_vk!(generated_softmax_matches_axis_oracle, engine, {{
	let shape = {shape!r};
	let values = (0..shape.iter().product::<usize>())
		.map(|index| ((index * 7 % 17) as f32 - 8.0) * 0.17)
		.collect::<Vec<_>>();
	let input = oa::Matrix::from_f32(&engine, shape, &values)?;
	let expected = softmax_reference(&values, &shape, {dim});
	let output = oa::matrix::{softmax['name']}(&input, {dim})?;
	let actual = output.read_f32()?;
	for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {{
		assert!(
			(actual - expected).abs() <= {tolerance},
			"element {{index}}: expected {{expected}}, found {{actual}}"
		);
	}}
	let last = oa::matrix::{softmax['name']}(&input, -1)?.read_f32()?;
	let expected_last = softmax_reference(&values, &shape, shape.len() - 1);
	for (actual, expected) in last.iter().zip(expected_last) {{
		assert!((actual - expected).abs() <= {tolerance});
	}}
	Ok(())
}});

test_vk!(generated_log_softmax_matches_axis_oracle, engine, {{
	let shape = {shape!r};
	let values = (0..shape.iter().product::<usize>())
		.map(|index| ((index * 7 % 17) as f32 - 8.0) * 0.17)
		.collect::<Vec<_>>();
	let input = oa::Matrix::from_f32(&engine, shape, &values)?;
	let expected = log_softmax_reference(&values, &shape, {dim});
	let output = oa::matrix::{log_softmax['name']}(&input, {dim})?;
	for (index, (actual, expected)) in output.read_f32()?.iter().zip(expected).enumerate() {{
		assert!(
			(actual - expected).abs() <= {tolerance},
			"element {{index}}: expected {{expected}}, found {{actual}}"
		);
	}}
	let last = oa::matrix::{log_softmax['name']}(&input, -1)?.read_f32()?;
	let expected_last = log_softmax_reference(&values, &shape, shape.len() - 1);
	for (actual, expected) in last.iter().zip(expected_last) {{
		assert!((actual - expected).abs() <= {tolerance});
	}}
	Ok(())
}});

test_vk!(generated_softmax_rejects_invalid_contracts, engine, {{
	let empty = oa::Matrix::from_f32(&engine, [0], &[])?;
	let integer = oa::Matrix::from_slice(&engine, [2], &[1_i32, 2])?;
	let input = oa::Matrix::from_f32(&engine, [2, 2], &[1.0, 2.0, 3.0, 4.0])?;
	for error in [
		rejected(oa::matrix::{softmax['name']}(&empty, -1), "empty input was accepted"),
		rejected(oa::matrix::{softmax['name']}(&integer, -1), "I32 input was accepted"),
		rejected(oa::matrix::{softmax['name']}(&input, -2), "negative non-sentinel dim was accepted"),
		rejected(oa::matrix::{softmax['name']}(&input, 2), "out-of-range dim was accepted"),
		rejected(oa::matrix::{log_softmax['name']}(&empty, -1), "empty LogSoftmax input was accepted"),
		rejected(oa::matrix::{log_softmax['name']}(&integer, -1), "I32 LogSoftmax input was accepted"),
		rejected(oa::matrix::{log_softmax['name']}(&input, -2), "negative non-sentinel LogSoftmax dim was accepted"),
		rejected(oa::matrix::{log_softmax['name']}(&input, 2), "out-of-range LogSoftmax dim was accepted"),
	] {{
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}}
	Ok(())
}});

test_vk!(generated_sum_matches_full_and_axis_oracles, engine, {{
	let shape = {shape!r};
	let values = (0..shape.iter().product::<usize>())
		.map(|index| ((index * 11 % 23) as f32 - 11.0) * 0.13)
		.collect::<Vec<_>>();
	let input = oa::Matrix::from_f32(&engine, shape, &values)?;
	let axis = oa::matrix::{sum_operation['name']}(&input, {dim})?;
	assert_eq!(axis.shape(), [shape[0], 1, shape[2]]);
	for (actual, expected) in axis.read_f32()?.iter().zip(sum_reference(&values, &shape, {dim})) {{
		assert!((actual - expected).abs() <= {tolerance});
	}}
	let full = oa::matrix::{sum_operation['name']}(&input, -1)?;
	assert_eq!(full.shape(), [1]);
	assert!((full.read_f32()?[0] - values.iter().sum::<f32>()).abs() <= {tolerance});
	Ok(())
}});

test_vk!(generated_categorical_accuracy_counts_match_exact_oracles, engine, {{
	let logits = oa::Matrix::from_f32(
		&engine,
		[4, 3],
		&[
			4.0, 4.0, 1.0,
			0.0, 2.0, 1.0,
			0.0, 1.0, 3.0,
			3.0, 1.0, 0.0,
		],
	)?;
	for labels in [
		oa::Matrix::from_slice(&engine, [4], &[0_u8, 1, 0, 0])?,
		oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 0, 0])?,
		oa::Matrix::from_slice(&engine, [4], &[0_i32, 1, -1, 0])?,
	] {{
		let count = oa::matrix::{accuracy_count['name']}(&logits, &labels)?;
		assert_eq!(count.shape(), [1]);
		assert_eq!(count.dtype(), oa::DType::U32);
		assert_eq!(count.read::<u32>()?, vec![3]);
	}}
	let labels = oa::Matrix::from_slice(&engine, [4], &[0_u32, 1, 2, 0])?;
	let mask = oa::Matrix::from_f32(&engine, [4], &[1.0, 0.0, -2.0, 1.0])?;
	assert_eq!(
		oa::matrix::{masked_accuracy_count['name']}(&logits, &labels, &mask)?
			.read::<u32>()?,
		vec![3]
	);
	Ok(())
}});

test_vk!(generated_categorical_accuracy_count_rejects_invalid_contracts, engine, {{
	let other_engine = oa::Engine::new()?;
	let logits = oa::Matrix::from_f32(&engine, [2, 3], &[1.0, 2.0, 3.0, 3.0, 2.0, 1.0])?;
	let rank_one = oa::Matrix::from_f32(&engine, [6], &[0.0; 6])?;
	let labels = oa::Matrix::from_slice(&engine, [2], &[2_u32, 0])?;
	let wrong_shape = oa::Matrix::from_slice(&engine, [1], &[2_u32])?;
	let float_labels = oa::Matrix::from_f32(&engine, [2], &[2.0, 0.0])?;
	let foreign_labels = oa::Matrix::from_slice(&other_engine, [2], &[2_u32, 0])?;
	let mask = oa::Matrix::from_f32(&engine, [2], &[1.0, 1.0])?;
	let integer_mask = oa::Matrix::from_slice(&engine, [2], &[1_u32, 1])?;

	for error in [
		rejected(oa::matrix::{accuracy_count['name']}(&rank_one, &labels), "rank-one logits were accepted"),
		rejected(oa::matrix::{accuracy_count['name']}(&logits, &wrong_shape), "wrong label shape was accepted"),
		rejected(oa::matrix::{accuracy_count['name']}(&logits, &float_labels), "F32 labels were accepted"),
		rejected(oa::matrix::{accuracy_count['name']}(&logits, &foreign_labels), "foreign labels were accepted"),
		rejected(
			oa::matrix::{masked_accuracy_count['name']}(&logits, &labels, &integer_mask),
			"integer mask was accepted",
		),
	] {{
		assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	}}
	assert_eq!(
		oa::matrix::{masked_accuracy_count['name']}(&logits, &labels, &mask)?
			.read::<u32>()?,
		vec![2]
	);
	Ok(())
}});

test_vk!(generated_categorical_accuracy_capture_retains_semantic_and_physical_evidence, engine, {{
	let logits = oa::Matrix::from_f32(&engine, [2, 3], &[1.0, 2.0, 3.0, 3.0, 2.0, 1.0])?;
	let labels = oa::Matrix::from_slice(&engine, [2], &[2_u32, 0])?;
	let (plan, count) = engine.capture(|| oa::matrix::{accuracy_count['name']}(&logits, &labels))?;
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(
		plan.semantic_graph().operations()[0].name(),
		"oa::matrix::{accuracy_count['name']}"
	);
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("accuracy"))
		.expect("categorical-accuracy execution report must be valid JSON");
	let nodes = report["nodes"].as_array().expect("execution nodes must be an array");
	assert_eq!(nodes.len(), 1);
	assert_eq!(nodes[0]["kernel"], "matrix.{accuracy_count['name']}.f32");
	assert!(!nodes[0]["physical_write"].is_null());
	engine.submit(&plan)?.wait()?;
	assert_eq!(count.read::<u32>()?, vec![2]);
	Ok(())
}});
"""


def expected_outputs(
	root: Path,
	elementwise: dict[str, Any],
	elementwise_hash: str,
	blas: dict[str, Any],
	blas_hash: str,
	reduce: dict[str, Any],
	reduce_hash: str,
	rng: dict[str, Any],
	rng_hash: str,
	index: dict[str, Any],
	index_hash: str,
	ml: dict[str, Any],
	ml_hash: str,
	audio: dict[str, Any],
	audio_hash: str,
	cryptography: dict[str, Any],
	cryptography_hash: str,
	image: dict[str, Any],
	image_hash: str,
	vision: dict[str, Any],
	vision_hash: str,
) -> dict[Path, str]:
	outputs = {
		root / "src/rs/core/operation/generated.rs": format_rust(
			generate_operation_registry(
				elementwise, elementwise_hash, blas, blas_hash, reduce, reduce_hash, rng, rng_hash, index, index_hash, ml, ml_hash, audio, audio_hash, cryptography, cryptography_hash, image, image_hash, vision, vision_hash
			),
			root,
		),
		root / "src/rs/matrix/elemwise.gen.rs": format_rust(generate_api(elementwise, elementwise_hash), root),
		root / "src/rs/matrix/blas.gen.rs": format_rust(generate_blas_api(blas, blas_hash), root),
		root / "src/rs/matrix/reduce.gen.rs": format_rust(
			generate_reduce_api(reduce, reduce_hash), root
		),
		root / "src/rs/runtime/shader/registry.gen.rs": format_rust(
			generate_registry(elementwise, elementwise_hash, blas, blas_hash, reduce, reduce_hash, rng, rng_hash, index, index_hash, ml, ml_hash, audio, audio_hash, cryptography, cryptography_hash, image, image_hash, vision, vision_hash), root
		),
		root / "src/rs/runtime/dnn/generated.rs": format_rust(
			generate_dnn_roles(elementwise, elementwise_hash, blas, blas_hash, reduce, reduce_hash, rng_hash, ml, ml_hash), root
		),
		root / "test/rs/matrix/test_elemwise.gen.rs": format_rust(generate_test(elementwise, elementwise_hash), root),
		root / "test/rs/matrix/test_blas.gen.rs": format_rust(generate_blas_test(blas, blas_hash), root),
		root / "test/rs/matrix/test_reduce.gen.rs": format_rust(
			generate_reduce_test(reduce, reduce_hash), root
		),
	}
	autograd_families = sorted(
		{
			operation["autograd"]["family"]
			for operation in ml["operations"]
			if operation.get("autograd", {}).get("mode") == "generated"
		}
	)
	for family in autograd_families:
		outputs[root / f"src/rs/ml/autograd/{family}.gen.rs"] = format_rust(
			generate_ml_autograd_attachments(ml, ml_hash, family), root
		)
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
	parser.add_argument("--reduce-schema", type=Path)
	parser.add_argument("--rng-schema", type=Path)
	parser.add_argument("--index-schema", type=Path)
	parser.add_argument("--ml-schema", type=Path)
	parser.add_argument("--audio-schema", type=Path)
	parser.add_argument("--cryptography-hash-schema", type=Path)
	parser.add_argument("--image-schema", type=Path)
	parser.add_argument("--vision-schema", type=Path)
	parser.add_argument("--check", action="store_true")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	root = args.root.resolve()
	schema_path = args.schema.resolve() if args.schema else root / DEFAULT_SCHEMA
	blas_schema_path = args.blas_schema.resolve() if args.blas_schema else root / DEFAULT_BLAS_SCHEMA
	reduce_schema_path = (
		args.reduce_schema.resolve() if args.reduce_schema else root / DEFAULT_REDUCE_SCHEMA
	)
	rng_schema_path = args.rng_schema.resolve() if args.rng_schema else root / DEFAULT_RNG_SCHEMA
	index_schema_path = args.index_schema.resolve() if args.index_schema else root / DEFAULT_INDEX_SCHEMA
	ml_schema_path = args.ml_schema.resolve() if args.ml_schema else root / DEFAULT_ML_SCHEMA
	audio_schema_path = args.audio_schema.resolve() if args.audio_schema else root / DEFAULT_AUDIO_SCHEMA
	cryptography_schema_path = args.cryptography_hash_schema.resolve() if args.cryptography_hash_schema else root / DEFAULT_CRYPTOGRAPHY_HASH_SCHEMA
	image_schema_path = args.image_schema.resolve() if args.image_schema else root / DEFAULT_IMAGE_SCHEMA
	vision_schema_path = args.vision_schema.resolve() if args.vision_schema else root / DEFAULT_VISION_SCHEMA
	try:
		schema, schema_hash = load_schema(schema_path)
		blas_schema, blas_schema_hash = load_blas_schema(blas_schema_path)
		reduce_schema, reduce_schema_hash = load_reduce_schema(reduce_schema_path)
		rng_schema, rng_schema_hash = load_rng_schema(rng_schema_path)
		index_schema, index_schema_hash = load_index_schema(index_schema_path)
		ml_schema, ml_schema_hash = load_ml_schema(ml_schema_path)
		audio_schema, audio_schema_hash = load_audio_schema(audio_schema_path)
		cryptography_schema, cryptography_schema_hash = load_cryptography_hash_schema(cryptography_schema_path)
		image_schema, image_schema_hash = load_image_schema(image_schema_path)
		vision_schema, vision_schema_hash = load_vision_schema(vision_schema_path)
		registry_entries(
			schema,
			blas_schema,
			reduce_schema,
			rng_schema,
			index_schema,
			ml_schema,
			audio_schema,
			cryptography_schema,
			image_schema,
			vision_schema,
		)
	except (OSError, SchemaError) as error:
		print(f"operation generation failed: {error}", file=os.sys.stderr)
		return 1
	outputs = expected_outputs(
		root,
		schema,
		schema_hash,
		blas_schema,
		blas_schema_hash,
		reduce_schema,
		reduce_schema_hash,
		rng_schema,
		rng_schema_hash,
		index_schema,
		index_schema_hash,
		ml_schema,
		ml_schema_hash,
		audio_schema,
		audio_schema_hash,
		cryptography_schema,
		cryptography_schema_hash,
		image_schema,
		image_schema_hash,
		vision_schema,
		vision_schema_hash,
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
