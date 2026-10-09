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


GENERATOR_VERSION = 64
DEFAULT_SCHEMA = Path("tool/gen/fn/schema/matrix/matrix_elemwise.json")
DEFAULT_BLAS_SCHEMA = Path("tool/gen/fn/schema/matrix/matrix_blas.json")
DEFAULT_REDUCE_SCHEMA = Path("tool/gen/fn/schema/matrix/matrix_reduce.json")
DEFAULT_RNG_SCHEMA = Path("tool/gen/fn/schema/matrix/matrix_rng.json")
DEFAULT_VIEW_SCHEMA = Path("tool/gen/fn/schema/matrix/matrix_view.json")
DEFAULT_INDEX_SCHEMA = Path("tool/gen/fn/schema/matrix/matrix_index.json")
DEFAULT_ML_SCHEMA = Path("tool/gen/fn/schema/ml/ml_training.json")
DEFAULT_ML_ACTIVATION_SCHEMA = Path("tool/gen/fn/schema/ml/ml_activation.json")
DEFAULT_AUDIO_SCHEMA = Path("tool/gen/fn/schema/audio/audio.json")
DEFAULT_CRYPTOGRAPHY_HASH_SCHEMA = Path("tool/gen/fn/schema/cryptography/cryptography_hash.json")
DEFAULT_CRYPTOGRAPHY_PQC_SCHEMA = Path("tool/gen/fn/schema/cryptography/cryptography_pqc.json")
DEFAULT_IMAGE_SCHEMA = Path("tool/gen/fn/schema/image/image.json")
DEFAULT_VISION_SCHEMA = Path("tool/gen/fn/schema/vision/vision_detection.json")
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
	"tile_32x32",
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


def validate_activation_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Activation schema_version must be 1")
	require(schema.get("family") == "ml_activation", "Activation family must be ml_activation")
	require(schema.get("domain") == "ml", "Activation domain must be ml")
	require(schema.get("dtype") == "f32", "Activation dtype must be f32")
	operations = schema.get("operations")
	require(isinstance(operations, list) and operations, "Activation operations must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	VALID_KINDS = {"unary", "unary_backward", "unary_scalar", "unary_scalar_backward", "binary", "binary_backward", "silu_mul", "silu_mul_backward"}
	for index, operation in enumerate(operations):
		where = f"Activation operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernels)):
			value = operation.get(key)
			require(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None, f"{where}.{key} must be an identifier")
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = operation.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id must be a non-zero u16")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		kind = operation.get("kind")
		require(kind in VALID_KINDS, f"{where}.kind must be one of {VALID_KINDS}")
		require(operation.get("rust_family") in {"activation", "swiglu"},
			f"{where}.rust_family must name activation or swiglu")


def load_activation_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_activation_schema)


def load_rng_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_rng_schema)


def load_index_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_index_schema)


def load_audio_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_audio_schema)


def load_cryptography_hash_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_cryptography_hash_schema)


def load_cryptography_pqc_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_cryptography_pqc_schema)


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
	if schema.get("domain") == "matrix":
		validate_matrix_autograd_schema(schema)
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
		require(all(isinstance(lowering, dict) for lowering in lowerings), f"{where}: lowering variants must be objects")
		broadcast = operation.get("shape_rule") == "broadcast"
		require(not broadcast or kind == "binary", f"{where}: broadcasting requires a binary operation")
		require(not lowerings or broadcast, f"{where}: broadcast candidates require broadcast shape semantics")
		if broadcast:
			require(
				all(isinstance(lowering.get("dtype"), str) for lowering in lowerings)
				and len(lowerings) == len(seen_dtypes)
				and {lowering["dtype"] for lowering in lowerings} == seen_dtypes,
				f"{where}: broadcast candidates must cover exactly the admitted dtypes",
			)
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
			"gather",
			"gather_backward",
			"gather_last_dim",
			"gather_last_dim_backward",
			"concat",
			"transpose",
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
			== {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes", "rust"},
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

	validate_audio_generation(schema)
	validate_generated_types(schema)


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

	validate_cryptography_api(schema, "hash")

# FIPS 204 final Table 1 / Algorithms 22 and 26 encoded wire layouts.
# One mechanical authority for Rust metadata and dispatch wrapper strides.
MLDSA_WIRE_LAYOUTS = {44: (1312, 2420), 65: (1952, 3309), 87: (2592, 4627)}


# NIST CSOR hashAlgs 1..12; SHAKE prehash output is fixed to 256/512 bits.
MLDSA_PREHASH_LAYOUTS = [
	{"name": name, "oid_suffix": suffix, "digest_bytes": length}
	for suffix, (name, length) in enumerate(zip(
		("SHA2-256", "SHA2-384", "SHA2-512", "SHA2-224", "SHA2-512/224", "SHA2-512/256",
		 "SHA3-224", "SHA3-256", "SHA3-384", "SHA3-512", "SHAKE-128", "SHAKE-256"),
		(32, 48, 64, 28, 28, 32, 28, 32, 48, 64, 32, 64)), 1)
]


def validate_cryptography_pqc_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Cryptography PQC schema_version must be 1")
	require(schema.get("family") == "cryptography_pqc", "Cryptography PQC family is invalid")
	prehashes = schema.get("prehashes")
	require(isinstance(prehashes, list) and all(isinstance(row, dict)
		and type(row.get("oid_suffix")) is int and type(row.get("digest_bytes")) is int
		for row in prehashes), "ML-DSA prehash identifiers and extents must be integers")
	require(prehashes == MLDSA_PREHASH_LAYOUTS, "ML-DSA prehash layout mismatch")
	require(schema.get("domain") == "cryptography::pqc", "Cryptography PQC domain is invalid")
	require(schema.get("dtype") == "u8", "Cryptography PQC dtype must be u8")
	contracts = schema.get("contracts")
	require(isinstance(contracts, list) and contracts, "Cryptography PQC contracts must be non-empty")
	contract_names: set[str] = set()
	for index, contract in enumerate(contracts):
		where = f"Cryptography PQC contracts[{index}]"
		require(isinstance(contract, dict), f"{where} must be an object")
		require(
			set(contract) == {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes"},
			f"{where} fields are incomplete or unknown",
		)
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		require(isinstance(contract["input_kinds"], list), f"{where}.input_kinds must be a list")
		require(all(k == "matrix" for k in contract["input_kinds"]), f"{where}.input_kinds must be all matrix")
		require(contract["output_kinds"] == ["matrix"], f"{where}.output_kinds must contain one Matrix")
		require(contract["shape_rule"] in {"match_input", "explicit"}, f"{where}.shape_rule is invalid")
		require(contract["dtype_rule"] == "u8", f"{where}.dtype_rule must be u8")
		validate_semantic_attributes(contract["attributes"], where)
	kernels = schema.get("kernels")
	require(isinstance(kernels, list) and kernels, "Cryptography PQC kernels must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernel_ids: set[str] = set()
	seen_sources: set[str] = set()
	seen_parameters: set[tuple[int, str]] = set()
	for index, kernel in enumerate(kernels):
		where = f"Cryptography PQC kernels[{index}]"
		require(isinstance(kernel, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernel_ids), ("source", seen_sources)):
			value = kernel.get(key)
			require(isinstance(value, str) and value, f"{where}.{key} must be non-empty")
			validate_unique(value, seen, f"{where}.{key}")
		parameter = kernel.get("parameter_set")
		require(type(parameter) is int and parameter in MLDSA_WIRE_LAYOUTS, f"{where}.parameter_set is invalid")
		hash_message = kernel["name"].endswith("_verify_hash_message")
		prehashed = kernel["name"].endswith("_verify_prehashed")
		suffix = "_hash_message" if hash_message else ("_prehashed" if prehashed else "")
		validate_unique((parameter, suffix), seen_parameters, f"{where}.parameter_set")
		require(kernel["kernel_id"] == f"CryptographyMlDsa{parameter}Verify{suffix.title().replace('_', '')}U8", f"{where}.kernel_id mismatch")
		source_parameter = "" if parameter == 65 and not suffix else str(parameter)
		require(kernel["source"] == f"src/slang/cryptography/pqc/mldsa/mldsa{source_parameter}_verify{suffix}.slang", f"{where}.source mismatch")
		require(kernel["name"] == f"ml_dsa_{parameter}_verify{suffix}", f"{where}.name does not match parameter_set")
		require(kernel.get("semantic_operation") == f"ml_dsa_{parameter}_verify{suffix}_batch", f"{where}.semantic_operation does not match parameter_set")
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
			fname, scalar_type = field
			require(isinstance(fname, str) and IDENTIFIER.fullmatch(fname) is not None, f"{where}.push field name is invalid")
			validate_unique(fname, seen_fields, f"{where}.push field name")
			require(scalar_type == "uint32", f"{where}.push field type must be uint32")
		require(len(fields) * 4 <= 128, f"{where}.push constants exceed Vulkan minimum guarantee")
		expected_fields = ["input_index", "offsets_index", "lengths_index", "sigs_index", "pks_index", "results_index", "batch_size", "message_bytes", "context_offset", "context_length"] + (["algorithm"] if prehashed or hash_message else [])
		require(fields == [[name, "uint32"] for name in expected_fields], f"{where}.push_fields ABI mismatch")
		contract = next(row for row in contracts if row["name"] == kernel["semantic_operation"])
		expected_attributes = ["batch_size", "context_offset", "context_length"] + (["algorithm"] if prehashed or hash_message else [])
		require(contract["attributes"] == [[name, "unsigned_integer"] for name in expected_attributes], f"{where} semantic attributes mismatch")
		require(contract["input_kinds"] == ["matrix"] * 5, f"{where} semantic inputs mismatch")

	private = schema.get("private_kernels", [])
	require(isinstance(private, list), "PQC private_kernels must be a list")
	private_abis = {
		"ml_kem_keygen": (["seed_index", "private_index", "public_index", "k"], [0, 1], [1, 2]),
		"ml_kem_encaps": (["seed_index", "public_index", "shared_index", "ciphertext_index", "status_index", "k"], [0, 2], [2, 3, 4]),
		"ml_kem_decaps": (["private_index", "ciphertext_index", "shared_index", "k"], [0, 2], [2]),
	}
	private_abis["ml_dsa_keygen"] = (["seed_index", "private_index", "public_index", "parameter_set"], [0, 1], [1, 2])
	private_abis["ml_dsa_sign"] = (["seed_index", "private_index", "mu_index", "signature_index", "status_index", "workspace_index", "parameter_set"], [0, 1, 5], [3, 4, 5])
	private_abis["ml_dsa_sign_message"] = (["seed_index", "private_index", "message_index", "context_index", "signature_index", "status_index", "workspace_index", "parameter_set", "message_length", "context_length"], [0, 1, 6], [4, 5, 6])
	private_abis["ml_dsa_sign_prehashed"] = (["seed_index", "private_index", "digest_index", "context_index", "signature_index", "status_index", "workspace_index", "parameter_set", "algorithm", "context_length"], [0, 1, 6], [4, 5, 6])
	private_abis["ml_dsa_sign_hash_message"] = (["seed_index", "private_index", "message_index", "context_index", "signature_index", "status_index", "workspace_index", "parameter_set", "message_length", "context_length", "algorithm"], [0, 1, 6], [4, 5, 6])
	seen_private: set[str] = set()
	for kernel in private:
		require(set(kernel) == ({"stable_id", "name", "kernel_id", "source", "workgroup_size", "push_fields", "physical_write", "secret_bindings"} | ({"parameter_layouts"} if kernel.get("name", "").startswith("ml_dsa_") else set()) | ({"workspace_bytes"} if kernel.get("name", "").startswith("ml_dsa_sign") else set())), "private PQC fields are incomplete or unknown")
		name = kernel["name"]
		require(name in private_abis, "unknown private PQC kernel")
		validate_unique(name, seen_private, "private PQC name")
		mldsa = name.startswith("ml_dsa_")
		if mldsa:
			expected = [{"parameter_set": parameter, "seed_bytes": 32, "public_bytes": public, "private_bytes": private} for parameter, public, private in ((44, 1312, 2560), (65, 1952, 4032), (87, 2592, 4896))]
			if name in ("ml_dsa_sign", "ml_dsa_sign_message", "ml_dsa_sign_prehashed", "ml_dsa_sign_hash_message"):
				expected = [{"parameter_set": a, "seed_bytes": 32, "private_bytes": b, "mu_bytes": 64, "signature_bytes": c} for a, b, c in ((44, 2560, 2420), (65, 4032, 3309), (87, 4896, 4627))]
				if name != "ml_dsa_sign":
					for row in expected:
						del row["mu_bytes"]
			require(kernel["parameter_layouts"] == expected, "ML-DSA layout mismatch")
		suffix = name.removeprefix("ml_dsa_" if mldsa else "ml_kem_")
		fields, secrets, writes = private_abis[name]
		family = "MlDsa" if mldsa else "MlKem"
		source = f"src/slang/cryptography/pqc/mldsa/mldsa_{suffix}_dispatch.slang" if mldsa else f"src/slang/cryptography/pqc/mlkem/mlkem_{suffix}.slang"
		require(kernel["kernel_id"] == f"Cryptography{family}{suffix.title().replace('_', '')}U8", "private PQC identity mismatch")
		require(kernel["source"] == source, "private PQC source mismatch")
		require(type(kernel["stable_id"]) is int and 0 < kernel["stable_id"] <= 65535, "invalid private PQC stable ID")
		validate_unique(kernel["stable_id"], seen_ids, "private PQC stable ID")
		validate_unique(kernel["kernel_id"], seen_kernel_ids, "private PQC kernel ID")
		require(kernel["workgroup_size"] == [1, 1, 1], "PQC serial ABI requires one invocation")
		require(kernel["push_fields"] == [[field, "uint32"] for field in fields], "PQC push ABI mismatch")
		require(kernel["secret_bindings"] == secrets, "PQC secret storage classification mismatch")
		validate_physical_write(kernel["physical_write"], f"private PQC {suffix}", required=True)
		expected_writes = [{"binding": binding, "domain": "rows", "partition": "exclusive_per_workgroup", "extent": "row_width", "collision": "exclusive", "tail": "bounds_checked"} for binding in writes]
		workspace = "exclusive_per_workgroup" if name.startswith("ml_dsa_sign") else "none"
		require(kernel["physical_write"] == {"writes": expected_writes, "workspace": workspace}, "PQC write ownership mismatch")
		if name.startswith("ml_dsa_sign"):
			require(type(kernel["workspace_bytes"]) is int and kernel["workspace_bytes"] == (7 + 8 + 8) * 256 * 4, "ML-DSA signing workspace extent mismatch")

	validate_cryptography_api(schema, "verify")


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
			== {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes", "rust"},
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
	validate_image_generation(schema)


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
	seen_autograd_nodes: set[str] = set()
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
		if "rust_family" in operation:
			require(operation["rust_family"] in {"rope", "upsample"}
				and semantic_domain == "ml::matrix" and not lowering_only,
				f"{where}.rust_family must name an admitted complete ML Matrix family")
		if operation.get("autograd", {}).get("mode") == "generated":
			node = operation["autograd"]["node"]
			require(node not in seen_autograd_nodes, f"{where}.autograd.node must be unique")
			seen_autograd_nodes.add(node)
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
		"ml::matrix": "matrix",
		"ml::loss": "loss",
		"ml::flow": "flow",
		"ml::advantage": "advantage",
		"ml::environment": "environment",
		"ml::policy": "policy",
	}.get(semantic_domain)
	require(
		expected_root is not None and (
			family == expected_root or family.startswith(expected_root + "/")
		),
		f"{where}.autograd.family must match its semantic owner",
	)
	if mode == "manual":
		require(
			set(autograd) == {"mode", "family"},
			f"{where}.autograd manual policy has unknown fields",
		)
		return
	require(
		{"mode", "family", "node", "inputs", "saved_matrices"} <= set(autograd)
		and set(autograd) <= {
			"mode", "family", "node", "inputs", "saved_matrices",
			"output_param", "saved_scalars", "owned_matrices",
		},
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
	owned = autograd.get("owned_matrices", [])
	require(
		isinstance(owned, list)
		and all(isinstance(name, str) and name in saved and name not in inputs for name in owned)
		and len(owned) == len(set(owned)),
		f"{where}.autograd.owned_matrices must contain unique saved-only matrix names",
	)
	output_param = autograd.get("output_param", "result")
	require(
		isinstance(output_param, str) and IDENTIFIER.fullmatch(output_param) is not None,
		f"{where}.autograd.output_param must be an identifier",
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
			and scalar[1] in {"f32", "usize", "u32", "UpsampleMode"},
			f"{label} must contain an identifier and supported Rust scalar type",
		)
		validate_unique(scalar[0], seen_scalars, f"{label} name")
		require(scalar[0] not in saved and scalar[0] not in inputs,
			f"{label} collides with a saved matrix or input")
	require("output_id" not in [*inputs, *saved, *seen_scalars],
		f"{where}.autograd output_id is reserved for output identity")
	require(output_param not in [*inputs, *saved, *seen_scalars],
		f"{where}.autograd.output_param collides with saved state")


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
			"same_admitted_dtype",
			"f32_logits_u32_or_i32_targets_f32_mask",
			"f32_logits_u8_u32_or_i32_labels_u32_output",
			"f32_logits_u8_u32_or_i32_labels_f32_mask_u32_output",
			"u32_state",
			"f32_parameter_u32_state",
			"f32_logits_u32_targets",
			"f32_logits_u32_or_i32_targets",
			"f32_weight_u32_indices",
			"f32_weight_u8_or_u32_indices",
			"f32_weight_u8_u32_or_nonnegative_i32_indices",
			"f32_with_u32_indices",
			"u32_indices_f32_gradient_weight",
			"u8_or_u32_indices_f32_gradient_weight",
			"u8_u32_or_i32_indices_f32_gradient_weight",
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
	if contract.get("aligned_variadic_aliases", False):
		require(
			not mutated and all(alias == -1 for alias in aliases),
			f"{where}.contract aligned variadic aliases cannot duplicate fixed mutation metadata",
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
		f"{prefix} @generated by tool/gen/fn/generate.py; DO NOT EDIT.\n"
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
		"// @generated by tool/gen/fn/generate.py; DO NOT EDIT.\n"
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


def rust_broadcast_routes(operation: dict[str, Any]) -> str:
	"""Use only this operation's schema-owned broadcast candidates."""
	return "&[" + ", ".join(
		f"({DTYPES[variant['dtype']]['rust_dtype']}, KernelId::{variant['kernel_id']})"
		for variant in operation.get("additional_lowering_variants", [])
		if variant["variant"] == "broadcast"
	) + "]"


def matrix_body(template: str, name: str, *, autograd: dict[str, Any] | None = None, **bindings: str) -> str:
	"""Instantiate a complete operation body; templates own reusable Rust syntax."""
	path = Path(__file__).parent / "template" / "matrix" / f"{template}.rs.in"
	body = path.read_text()
	if "node" in bindings:
		body = body.replace("$node", bindings.pop("node"))
	prefix = f"\tlet contract = crate::core::operation::matrix::{rust_const_name(name)};\n"
	for token, value in bindings.items():
		if token == "kernel":
			prefix += f"\tlet {token} = {value};\n"
		else:
			body = re.sub(r"\b" + re.escape(token) + r"\b", lambda _: value, body)
	if template in {"binary", "unary", "unary_scalar"}:
		attachments = dict(re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", (path.parent / "autograd.rs.in").read_text(), re.S))
		attachment = attachments.get(name, "").rstrip()
		require(bool(attachment) == (autograd is not None), f"Matrix {name} attachment recipe differs from its schema policy")
		if autograd is not None:
			target = autograd.get("target", name)
			attachment = re.sub(r"autograd::record_\w+", "autograd::record_" + target, attachment)
		if attachment and template == "binary":
			attachment = "\tif left.dtype() == DType::F32 {\n" + "\n".join("\t" + line for line in attachment.splitlines()) + "\n\t}"
		if "scalar" in bindings:
			attachment = re.sub(r"\bscalar\b(?!:)", bindings["scalar"], attachment)
		attachment = re.sub(r"\b(\w+): \1,", r"\1,", attachment)
		body = body.replace("$autograd", attachment)
	if template == "accuracy":
		prefix += "\tlet mask = " + ("Some(mask)" if name.startswith("masked_") else "None::<&Matrix>") + ";\n"
	if "$node" in body or "$autograd" in body:
		raise ValueError(f"unresolved Matrix body template binding: {template}/{name}")
	return prefix + body.rstrip()


IMAGE_CATEGORIES = ("pixel", "filter", "geometric", "color")


def image_templates(category: str) -> dict[str, str]:
	path = Path(__file__).parent / "template/image" / f"{category}.rs.in"
	sections = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", path.read_text(encoding="utf-8"), re.S)
	require(len(sections) == len({name for name, _ in sections}), f"duplicate Image template section: {category}")
	return dict(sections)


def wire_parts(kernel: dict[str, Any]) -> tuple[list, list]:
	"""Split schema ABI descriptor indices from the scalar push payload."""
	wire = kernel["push_fields"]
	count = sum(name.endswith("_index") for name, _ in wire)
	require(all(name.endswith("_index") and kind == "uint32" for name, kind in wire[:count]), "descriptor indices must precede scalar pushes")
	return wire[:count], wire[count:]


def render_wire_values(values: dict[str, list[str]], fields: list, *, buffers: bool) -> str:
	"""Emit named expressions in schema ABI order, shared by Audio and Image."""
	return ",\n".join(
		f"BufferBinding::{values[name][0]}({values[name][1]}.storage())" if buffers
		else f"PushConstant::{values[name][0]}({values[name][1]})"
		for name, _ in fields
	) + ("," if fields else "")


def validate_dispatch_generation(
	schema: dict[str, Any], categories: set[str], allowed_types: set[str],
	returns: dict[str, str], encodings: dict[str, set[str]], *, read_write: bool = False,
) -> None:
	"""Validate complete operation signatures and named per-stage ABI expressions."""
	domain = schema["domain"].title()
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	for op in schema["contracts"]:
		rust = op.get("rust")
		where = f"{domain} {op['name']}.rust"
		require(isinstance(rust, dict) and set(rust) == {"category", "parameters", "returns", "doc", "attribute_sources", "bindings", "push_constants", "dispatches", "annotations"}, f"{where} fields are incomplete or unknown")
		require(isinstance(rust["category"], str) and rust["category"] in categories, f"{where} category is invalid")
		require(isinstance(rust["annotations"], list) and rust["annotations"] in ([], ["allow(clippy::too_many_arguments)"]), f"{where} annotations are invalid")
		require(rust["returns"] == returns[op["name"]], f"{where} return differs from contract")
		require(isinstance(rust["doc"], list) and rust["doc"] and all(isinstance(line, str) and "\n" not in line for line in rust["doc"]), f"{where} docs are invalid")
		require(isinstance(rust["parameters"], list) and rust["parameters"], f"{where} parameters are invalid")
		names = set()
		inputs = []
		for parameter in rust["parameters"]:
			require(isinstance(parameter, list) and len(parameter) == 2, f"{where} parameter is invalid")
			name, kind = parameter
			require(isinstance(name, str) and IDENTIFIER.fullmatch(name), f"{where} parameter name is invalid")
			validate_unique(name, names, where)
			require(isinstance(kind, str) and kind in allowed_types, f"{where} parameter type is unsupported")
			if kind in {"&Image", "&Matrix"}:
				inputs.append(kind[1:].lower())
		require(inputs == op["input_kinds"], f"{where} semantic inputs differ from signature")
		sources = rust["attribute_sources"]
		require(isinstance(sources, dict) and set(sources) == {name for name, _ in op["attributes"]}, f"{where} attribute sources differ from contract")
		for name, kind in op["attributes"]:
			source = sources[name]
			require(isinstance(source, dict) and set(source) == {"encoding", "value"}, f"{where} attribute source is invalid")
			require(isinstance(source["encoding"], str) and source["encoding"] in encodings[kind], f"{where} attribute encoding differs from contract")
			require(isinstance(source["value"], str) and source["value"].strip() and ";" not in source["value"], f"{where} attribute expression is invalid")
		stages = rust["dispatches"]
		require(isinstance(stages, list) and stages, f"{where} dispatches must be nonempty")
		require(all(isinstance(rust[field], list) and len(rust[field]) == len(stages) for field in ("bindings", "push_constants")), f"{where} dispatch array coverage is invalid")
		used = set()
		for index, stage in enumerate(stages):
			require(isinstance(stage, dict) and set(stage) == {"kernels", "kernel", "workgroups"}, f"{where} dispatch fields are invalid")
			candidates = stage["kernels"]
			require(isinstance(candidates, list) and candidates and all(isinstance(name, str) and name in kernels for name in candidates) and len(set(candidates)) == len(candidates), f"{where} kernel candidates are invalid")
			require(all(kernels[name]["semantic_operation"] == op["name"] for name in candidates), f"{where} kernel belongs to another operation")
			require(all(kernels[name]["push_fields"] == kernels[candidates[0]]["push_fields"] for name in candidates), f"{where} kernel variants have different ABI")
			used.update(candidates)
			for field in ("kernel", "workgroups"):
				require(isinstance(stage[field], str) and stage[field].strip() and ";" not in stage[field], f"{where} dispatch expression is invalid")
			if len(candidates) == 1:
				require(stage["kernel"] == f"KernelId::{candidates[0]}", f"{where} kernel expression differs from candidates")
			else:
				require(op["name"] == "resize" and stage["kernel"] == "kernel", f"{where} dynamic kernel route is unsupported")
			wire = wire_parts(kernels[candidates[0]])
			for field, fields, allowed in (("bindings", wire[0], ({"read", "write", "read_write"} if read_write else {"read", "write"})), ("push_constants", wire[1], {"U32", "F32"})):
				declaration = rust[field][index]
				require(isinstance(declaration, dict) and set(declaration) == {"name", "values"}, f"{where} declaration is invalid")
				require(declaration["name"] is None or isinstance(declaration["name"], str) and IDENTIFIER.fullmatch(declaration["name"]), f"{where} declaration name is invalid")
				values = declaration["values"]
				require(isinstance(values, dict) and set(values) == {name for name, _ in fields}, f"{where} named fields differ from kernel ABI")
				for name, kind in fields:
					value = values[name]
					require(isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in allowed and isinstance(value[1], str) and value[1].strip() and ";" not in value[1], f"{where} wire expression is invalid")
					if field == "push_constants":
						require(value[0] == ("U32" if kind == "uint32" else "F32"), f"{where} push scalar type differs from ABI")
		require(used == {k["kernel_id"] for k in schema["kernels"] if k["semantic_operation"] == op["name"]}, f"{where} kernel variant coverage differs from schema")
	validate_generated_types(schema)


def validate_image_generation(schema: dict[str, Any]) -> None:
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	validate_dispatch_generation(
		schema, set(IMAGE_CATEGORIES),
		{"&Image", "&Matrix", "f32", "u32", "u64", "bool", "[u32; 4]", "BorderMode", "InterpolationMode", "ImageFormat", "crate::ImageFormat", "NormalizationParams"},
		{op["name"]: "Image" for op in schema["contracts"]},
		{"float": {"float"}, "unsigned_integer": {"unsigned", "unsigned64", "inline_unsigned"}, "boolean": {"boolean"}, "enum": {"enumeration", "inline_enum"}},
	)
	selectors = [record for record in schema["types"] if record["name"] == "InterpolationMode"]
	require(len(selectors) == 1 and selectors[0]["kind"] == "enum" and all("kernel" in variant for variant in selectors[0]["variants"]), "Image resize requires its interpolation selector")
	selector = selectors[0]
	resize = next(op for op in schema["contracts"] if op["name"] == "resize")
	require({variant["kernel"] for variant in selector["variants"]} == {"KernelId::" + name for stage in resize["rust"]["dispatches"] for name in stage["kernels"]}, "Image resize selector differs from dispatch candidates")
	for category in IMAGE_CATEGORIES:
		expected = {op["name"] for op in schema["contracts"] if op["rust"]["category"] == category}
		require(set(image_templates(category)) == expected, f"Image {category} template coverage differs from schema")
	for op in schema["contracts"]:
		render_image_body(op, kernels)


def render_dispatch_body(op: dict[str, Any], kernels: dict[str, Any], body: str, domain: str) -> str:
	rust = op["rust"]
	bindings = {"contract": f"crate::core::operation::{domain}::{op['name'].upper()}"}
	attributes = []
	for name, _ in op["attributes"]:
		source = rust["attribute_sources"][name]
		encoding, value = source["encoding"], source["value"]
		if domain == "vision":
			attributes.append(f'OpAttribute::{encoding} {{ name: "{name}".into(), value: {value}, }}')
		elif encoding == "inline_unsigned":
			attributes.append(f'OpAttribute::UnsignedInteger {{ name: "{name}".into(), value: u64::from({value}), }}')
		elif encoding == "inline_enum":
			attributes.append(f'OpAttribute::Enum {{ name: "{name}".into(), value: {value}.into(), }}')
		else:
			attributes.append(f'{encoding}("{name}", {value})')
	bindings["attributes"] = ",\n".join(attributes) + ("," if attributes else "")
	for index, stage in enumerate(rust["dispatches"]):
		wire = wire_parts(kernels[stage["kernels"][0]])
		for field, fields, token in (("bindings", wire[0], "bindings"), ("push_constants", wire[1], "push")):
			bindings[f"{token}:{index}"] = render_wire_values(rust[field][index]["values"], fields, buffers=field == "bindings")
			if rust[field][index]["name"] is not None:
				prefix = "binding_name" if field == "bindings" else "push_name"
				bindings[f"{prefix}:{index}"] = rust[field][index]["name"]
		for field in ("kernel", "workgroups"):
			# Existing Rust shorthand is preserved when it names the local variable.
			if domain == "vision" or stage[field] != field:
				bindings[f"{field}:{index}"] = stage[field]
	tokens = set(re.findall(r"\{\{([^{}]+)\}\}", body))
	require(tokens == set(bindings), f"{domain.title()} {op['name']} template bindings are unresolved, missing or unused")
	return re.sub(r"\{\{([^{}]+)\}\}", lambda match: bindings[match[1]], body).rstrip()


def render_image_body(op: dict[str, Any], kernels: dict[str, Any]) -> str:
	return render_dispatch_body(op, kernels, image_templates(op["rust"]["category"])[op["name"]], "image")


def vision_templates() -> dict[str, str]:
	path = Path(__file__).parent / "template/vision/detection.rs.in"
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
	module = (Path(__file__).parent / "template/vision/detection_module.rs.in").read_text(encoding="utf-8")
	require(module.count("{{items}}") == 1, "Vision module requires exactly one item insertion")
	items = []
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	templates = vision_templates()
	for op in schema["contracts"]:
		rust = op["rust"]
		items.extend(generate_type(record) for record in schema["types"] if record["before"] == op["name"])
		docs = "\n".join("///" + (" " + line if line else "") for line in rust["doc"])
		parameters = ", ".join(f"{name}: {kind}" for name, kind in rust["parameters"])
		annotations = "".join(f"#[{annotation}]\n" for annotation in rust["annotations"])
		body = render_dispatch_body(op, kernels, templates[op["name"]], "vision")
		items.append(f"{docs}\n{annotations}pub fn {op['name']}({parameters}) -> Result<{rust['returns']}> {{\n{body}\n}}\n")
	return banner(schema_hash, "//", "vision_detection.json") + module.replace("{{items}}", "\n\n".join(items))



def cryptography_templates(category: str) -> dict[str, str]:
	path = Path(__file__).parent / "template/cryptography" / f"{category}.rs.in"
	sections = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", path.read_text(encoding="utf-8"), re.S)
	require(len(sections) == len({name for name, _ in sections}), "duplicate Cryptography template section")
	return dict(sections)


def cryptography_body(schema: dict[str, Any], api: dict[str, Any], category: str) -> str:
	kernels = {k["kernel_id"]: k for k in schema["kernels"]}
	contracts = {op["name"]: op for op in schema["contracts"]}
	tokens = dict(api["values"])
	if category == "verify":
		tokens["prehash_validation"] = 'if KernelId::mldsa_prehash_bytes(algorithm).is_none() { return Err(Error::invalid_argument("unknown ML-DSA prehash identifier")); }' if any(name == "algorithm" for name, _ in api["parameters"]) else ""
	if category == "hash":
		tokens["contract"] = f"crate::core::operation::{schema['domain']}::{api['contracts'][0].upper()}"
	else:
		tokens["routes"] = "\n".join(
			f"MlDsaParameters::MlDsa{kernels[stage['kernel']]['parameter_set']} => (crate::core::operation::{schema['domain']}::{contract.upper()}, KernelId::{stage['kernel']}),"
			for contract, stage in zip(api["contracts"], api["stages"])
		)
	# API expressions are keyed by ABI field, never by JSON map iteration order.
	for index, stage in enumerate(api["stages"] if category == "hash" else api["stages"][:1]):
		buffers, pushes = wire_parts(kernels[stage["kernel"]])
		tokens[f"bindings:{index}"] = render_wire_values(stage["bindings"], buffers, buffers=True)
		tokens[f"push:{index}"] = render_wire_values({name: ["U32", value] for name, value in stage["push_constants"].items()}, pushes, buffers=False)
		if category == "hash":
			tokens[f"kernel:{index}"] = f"KernelId::{stage['kernel']}"
	if api["attributes"]:
		tokens["attributes"] = ",\n".join(
			f'OpAttribute::UnsignedInteger {{ name: "{name}".into(), value: {api["attributes"][name]}, }}'
			for name, _ in contracts[api["contracts"][0]]["attributes"]
		) + ","
	body = cryptography_templates(category)[api["template"]]
	require(set(re.findall(r"\{\{([^{}]+)\}\}", body)) == set(tokens), f"Cryptography {api['name']} template bindings are unresolved, missing or unused")
	return re.sub(r"\{\{([^{}]+)\}\}", lambda match: tokens[match[1]], body).rstrip()


def validate_cryptography_api(schema: dict[str, Any], category: str) -> None:
	apis = schema.get("rust_api")
	require(isinstance(apis, list) and apis, "Cryptography rust_api must be nonempty")
	kernels = {k["kernel_id"]: k for k in schema["kernels"]}
	contracts = {op["name"]: op for op in schema["contracts"]}
	names, covered = set(), set()
	for api in apis:
		require(isinstance(api, dict) and set(api) == {"name", "template", "visibility", "parameters", "doc", "annotations", "contracts", "values", "stages", "attributes"}, "Cryptography API fields are incomplete or unknown")
		name = api["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name), "Cryptography API name is invalid")
		validate_unique(name, names, "Cryptography API name")
		require(isinstance(api["visibility"], str) and api["visibility"] in {"pub", "pub(crate)"}, "Cryptography API visibility is invalid")
		require(isinstance(api["doc"], list) and api["doc"] and all(isinstance(line, str) and "\n" not in line for line in api["doc"]), "Cryptography API docs are invalid")
		require(isinstance(api["annotations"], list) and all(isinstance(line, str) and "\n" not in line for line in api["annotations"]), "Cryptography API annotations are invalid")
		require(isinstance(api["parameters"], list) and api["parameters"], "Cryptography API parameters are invalid")
		parameter_names, inputs = set(), []
		for parameter in api["parameters"]:
			require(isinstance(parameter, list) and len(parameter) == 2, "Cryptography API parameter is invalid")
			arg, kind = parameter
			require(isinstance(arg, str) and IDENTIFIER.fullmatch(arg), "Cryptography API parameter name is invalid")
			validate_unique(arg, parameter_names, "Cryptography API parameter")
			require(isinstance(kind, str) and kind in {"&Matrix", "usize", "u32", "MlDsaParameters", "std::ops::Range<usize>"}, "Cryptography API parameter type is invalid")
			if kind == "&Matrix": inputs.append("matrix")
		require(isinstance(api["values"], dict) and all(isinstance(key, str) and IDENTIFIER.fullmatch(key) and isinstance(value, str) and value.strip() and ";" not in value for key, value in api["values"].items()), "Cryptography API values are invalid")
		require(isinstance(api["attributes"], dict), "Cryptography API attributes must be an object")
		if category == "hash":
			expected = [["input", "&Matrix"]] + ([["output_length", "usize"]] if name in {"shake128", "shake256"} else [])
		else:
			expected = [["parameters", "MlDsaParameters"]] + ([["algorithm", "u32"]] if "algorithm" in api["attributes"] else []) + [[arg, "&Matrix"] for arg in ("messages", "msg_offsets", "msg_lengths", "signatures", "public_keys")] + [["context", "std::ops::Range<usize>"]]
		require(api["parameters"] == expected, "Cryptography API signature differs from template contract")
		owned = api["contracts"]
		require(isinstance(owned, list) and owned and all(isinstance(c, str) and c in contracts for c in owned) and len(set(owned)) == len(owned), "Cryptography API contracts are invalid")
		require(not covered.intersection(owned), "Cryptography API contract has duplicate owner")
		covered.update(owned)
		require(all(contracts[c]["input_kinds"] == inputs for c in owned), "Cryptography API semantic inputs differ from signature")
		attributes = contracts[owned[0]]["attributes"]
		require(all(contracts[c]["attributes"] == attributes for c in owned), "Cryptography API route attributes differ")
		require(isinstance(api["attributes"], dict) and set(api["attributes"]) == {n for n, _ in attributes} and all(isinstance(v, str) and v.strip() and ";" not in v for v in api["attributes"].values()), "Cryptography API attribute sources differ from contract")
		stages = api["stages"]
		require(isinstance(stages, list) and stages, "Cryptography API stages must be nonempty")
		used = set()
		for stage in stages:
			require(isinstance(stage, dict) and set(stage) == {"kernel", "bindings", "push_constants"} and isinstance(stage["kernel"], str) and stage["kernel"] in kernels, "Cryptography API stage is invalid")
			kernel = kernels[stage["kernel"]]
			used.add(stage["kernel"])
			require(kernel["semantic_operation"] in owned, "Cryptography API kernel belongs to another operation")
			buffers, pushes = wire_parts(kernel)
			require(isinstance(stage["bindings"], dict) and set(stage["bindings"]) == {n for n, _ in buffers}, "Cryptography API bindings differ from ABI")
			for access in stage["bindings"].values():
				require(isinstance(access, list) and len(access) == 2 and access[0] in {"read", "write"} and isinstance(access[1], str) and IDENTIFIER.fullmatch(access[1]), "Cryptography API binding expression is invalid")
			require(isinstance(stage["push_constants"], dict) and set(stage["push_constants"]) == {n for n, _ in pushes} and all(kind == "uint32" for _, kind in pushes), "Cryptography API pushes differ from ABI")
			require(all(isinstance(v, str) and v.strip() and ";" not in v for v in stage["push_constants"].values()), "Cryptography API push expression is invalid")
		require(used == {k["kernel_id"] for k in schema["kernels"] if k["semantic_operation"] in owned}, "Cryptography API kernel coverage differs from schema")
		if category == "verify":
			require(api["visibility"] == ("pub(crate)" if "algorithm" in api["attributes"] else "pub"), "Cryptography verifier visibility differs from admission")
			require(len(stages) == 3 and [kernels[s["kernel"]]["semantic_operation"] for s in stages] == owned, "Cryptography verifier route coverage differs")
			require({kernels[s["kernel"]]["parameter_set"] for s in stages} == {44, 65, 87}, "Cryptography verifier parameter coverage differs")
			require(all(kernels[s["kernel"]]["push_fields"] == kernels[stages[0]["kernel"]]["push_fields"] and s["bindings"] == stages[0]["bindings"] and s["push_constants"] == stages[0]["push_constants"] for s in stages), "Cryptography verifier route ABI differs")
		require(isinstance(api["template"], str) and api["template"] in cryptography_templates(category), "Cryptography API template is unknown")
		cryptography_body(schema, api, category)
	require(covered == set(contracts), "Cryptography API contract coverage differs from schema")
	require({api["template"] for api in apis} == set(cryptography_templates(category)), "Cryptography template coverage differs from schema")


def generate_cryptography_api(schema: dict[str, Any], schema_hash: str, category: str) -> str:
	module = (Path(__file__).parent / "template/cryptography" / f"{category}_module.rs.in").read_text(encoding="utf-8")
	require(module.count("{{items}}") == 1, "Cryptography module requires exactly one item insertion")
	items = []
	for api in schema["rust_api"]:
		docs = "\n".join("///" + (" " + line if line else "") for line in api["doc"])
		annotations = "".join(f"#[{line}]\n" for line in api["annotations"])
		parameters = ", ".join(f"{name}: {kind}" for name, kind in api["parameters"])
		items.append(f"{docs}\n{annotations}{api['visibility']} fn {api['name']}({parameters}) -> Result<Matrix> {{\n" + cryptography_body(schema, api, category) + "\n}\n")
	return banner(schema_hash, "//", "cryptography_hash.json" if category == "hash" else "cryptography_pqc.json") + module.replace("{{items}}", "\n\n".join(items))


def generate_image_api(schema: dict[str, Any], schema_hash: str, category: str) -> str:
	module = (Path(__file__).parent / "template/image" / f"{category}_module.rs.in").read_text(encoding="utf-8")
	require(module.count("{{items}}") == 1, f"Image {category} module requires exactly one item insertion")
	items = []
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	for op in schema["contracts"]:
		rust = op["rust"]
		if rust["category"] != category:
			continue
		items.extend(generate_type(record) for record in schema["types"] if record["before"] == op["name"])
		docs = "\n".join("///" + (" " + line if line else "") for line in rust["doc"])
		parameters = ", ".join(f"{name}: {kind}" for name, kind in rust["parameters"])
		annotations = "".join(f"#[{annotation}]\n" for annotation in rust["annotations"])
		items.append(f"{docs}\n{annotations}pub fn {op['name']}({parameters}) -> Result<{rust['returns']}> {{\n" + render_image_body(op, kernels) + "\n}\n")
	return banner(schema_hash, "//", "image.json") + module.replace("{{items}}", "\n\n".join(items))


def audio_templates(category: str) -> dict[str, str]:
	path = Path(__file__).parent / "template/audio" / f"{category}.rs.in"
	text = path.read_text(encoding="utf-8")
	sections = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", text, re.S)
	require(len(sections) == len({name for name, _ in sections}), f"duplicate Audio template section: {category}")
	return dict(sections)


def validate_audio_generation(schema: dict[str, Any]) -> None:
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	for op in schema["contracts"]:
		rust = op["rust"]
		where = f"Audio {op['name']}.rust"
		require(isinstance(rust, dict), f"{where} must be an object")
		require(set(rust) == {"category", "parameters", "returns", "doc", "attribute_sources", "bindings", "push_constants", "dispatches"}, f"{where} fields are incomplete or unknown")
		require(isinstance(rust["category"], str) and rust["category"] in {"signal", "transform"}, f"{where}.category is invalid")
		require(isinstance(rust["doc"], str) and rust["doc"] and "\n" not in rust["doc"], f"{where}.doc must be one summary line")
		require(rust["returns"] == ("Audio" if op["output_kinds"] == ["audio"] else "Matrix"), f"{where}.returns contradicts semantic output")
		parameters = rust["parameters"]
		require(isinstance(parameters, list) and parameters, f"{where}.parameters must be nonempty")
		names = set()
		for parameter in parameters:
			require(isinstance(parameter, list) and len(parameter) == 2, f"{where}.parameter is invalid")
			name, kind = parameter
			require(isinstance(name, str) and IDENTIFIER.fullmatch(name), f"{where}.parameter name is invalid")
			validate_unique(name, names, where)
			require(isinstance(kind, str) and kind in {"&Audio", "f32", "u32", "u64", "NormalizeAudioConfig", "ResampleConfig", "BiquadCoefficients", "&[BiquadCoefficients]", "StftConfig", "MelConfig", "MfccConfig"}, f"{where}.parameter type is unsupported")
		require(sum(kind == "&Audio" for _, kind in parameters) == len(op["input_kinds"]), f"{where} Audio input count differs from contract")
		require(all(kind in {"float", "unsigned_integer", "boolean"} for _, kind in op["attributes"]), f"{where} attribute kind has no generated encoding")
		sources = rust["attribute_sources"]
		require(isinstance(sources, dict) and set(sources) == {name for name, _ in op["attributes"]}, f"{where}.attribute_sources differs from contract")
		for expression in sources.values():
			require(isinstance(expression, str) and expression.strip() and ";" not in expression, f"{where} requires an attribute expression")
		for field in ("bindings", "push_constants", "dispatches"):
			require(isinstance(rust[field], list) and rust[field], f"{where}.{field} must be nonempty")
		for field, allowed in (("bindings", {"read", "write"}), ("push_constants", {"U32", "F32"})):
			for declaration in rust[field]:
				require(isinstance(declaration, dict) and set(declaration) == {"name", "values"}, f"{where}.{field} declaration is invalid")
				require(isinstance(declaration["name"], str) and IDENTIFIER.fullmatch(declaration["name"]), f"{where}.{field} name is invalid")
				require(isinstance(declaration["values"], dict) and declaration["values"], f"{where}.{field} values are empty")
				for field_name, value in declaration["values"].items():
					require(isinstance(field_name, str) and IDENTIFIER.fullmatch(field_name), f"{where}.{field} wire name is invalid")
					require(isinstance(value, list) and len(value) == 2 and value[0] in allowed and isinstance(value[1], str) and value[1].strip() and ";" not in value[1], f"{where}.{field} value is invalid")
		for stage in rust["dispatches"]:
			require(isinstance(stage, dict) and set(stage) == {"kernel", "buffers", "push_constants", "workgroups"}, f"{where} dispatch is invalid")
			require(isinstance(stage["kernel"], str) and stage["kernel"] in kernels, f"{where} references unknown kernel")
			kernel = kernels[stage["kernel"]]
			wire = kernel["push_fields"]
			indices = [name for name, _ in wire if name.endswith("_index")]
			require(all(name.endswith("_index") for name, _ in wire[:len(indices)]), f"{where} descriptor indices must precede scalar pushes")
			for field, reference, expected in (("bindings", "buffers", wire[:len(indices)]), ("push_constants", "push_constants", wire[len(indices):])):
				ref = stage[reference]
				require(isinstance(ref, str) and ref.startswith("&"), f"{where}.{reference} must borrow a declared array")
				declarations = [d for d in rust[field] if d["name"] == ref[1:]]
				require(declarations and all(set(d["values"]) == {name for name, _ in expected} for d in declarations), f"{where}.{reference} length differs from kernel ABI")
				if field == "push_constants":
					types = ["U32" if kind == "uint32" else "F32" for _, kind in wire[len(indices):]]
					require(all([d["values"][name][0] for name, _ in expected] == types for d in declarations), f"{where} push types differ from kernel ABI")
			require(isinstance(stage["workgroups"], str) and stage["workgroups"].strip() and ";" not in stage["workgroups"], f"{where} workgroups expression is invalid")
	for category in ("signal", "transform"):
		expected = {op["name"] for op in schema["contracts"] if op["rust"]["category"] == category}
		require(set(audio_templates(category)) == expected, f"Audio {category} template coverage differs from schema")
	for op in schema["contracts"]:
		# Rendering also rejects unresolved, missing and unused section bindings.
		render_audio_body(op, kernels)


def render_audio_body(op: dict[str, Any], kernels: dict[str, Any]) -> str:
	rust = op["rust"]
	body = audio_templates(rust["category"])[op["name"]]
	bindings = {"contract": f"crate::core::operation::audio::{op['name'].upper()}"}
	attributes = []
	for name, kind in op["attributes"]:
		expression = rust["attribute_sources"][name]
		if kind == "float":
			attributes.append(f'float_attribute("{name}", {expression})')
		elif kind == "unsigned_integer":
			attributes.append(f'unsigned_attribute("{name}", {expression})')
		else:
			attributes.append(f'OpAttribute::Boolean {{ name: "{name}".into(), value: {expression} }}')
	bindings["attributes"] = ",\n".join(attributes) + ("," if attributes else "")
	for field, token in (("bindings", "bindings"), ("push_constants", "push")):
		for index, declaration in enumerate(rust[field]):
			reference = "buffers" if field == "bindings" else "push_constants"
			stage = next(stage for stage in rust["dispatches"] if stage[reference] == "&" + declaration["name"])
			wire = kernels[stage["kernel"]]["push_fields"]
			fields = [field for field in wire if field[0] in declaration["values"]]
			values = render_wire_values(declaration["values"], fields, buffers=field == "bindings")
			bindings[f"{token}:{index}"] = f"let {declaration['name']} = [\n{values}\n];"
	for index, stage in enumerate(rust["dispatches"]):
		bindings[f"dispatch:{index}"] = "ComputeDispatch {\n" + f"kernel: KernelId::{stage['kernel']},\nbuffers: {stage['buffers']},\npush_constants: {stage['push_constants']},\nworkgroups: {stage['workgroups']},\n" + "}"
	tokens = set(re.findall(r"\{\{([^{}]+)\}\}", body))
	require(tokens == set(bindings), f"Audio {op['name']} template bindings are unresolved, missing or unused")
	return re.sub(r"\{\{([^{}]+)\}\}", lambda match: bindings[match[1]], body).rstrip()



def validate_generated_types(schema: dict[str, Any]) -> None:
	domain = schema["domain"].title()
	types = schema.get("types")
	require(isinstance(types, list) and types, f"{domain} types must be a nonempty array")
	names = set()
	operations = {op["name"]: op["rust"]["category"] for op in schema["contracts"]}
	for record in types:
		require(isinstance(record, dict) and set(record) in ({"category", "name", "before", "kind", "visibility", "derive", "doc", "fields", "variants", "defaults", "default_variant"}, {"category", "name", "before", "kind", "visibility", "derive", "doc", "fields", "variants", "defaults", "default_variant", "mappings"}), f"{domain} type record is incomplete or unknown")
		name = record["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name), f"{domain} type name is invalid")
		validate_unique(name, names, f"{domain} types")
		require(isinstance(record["kind"], str) and record["kind"] in {"struct", "enum"} and isinstance(record["visibility"], str) and record["visibility"] in {"pub", ""}, f"{domain} {name} kind/visibility is invalid")
		require(isinstance(record["before"], str) and record["before"] in operations and operations[record["before"]] == record["category"], f"{domain} {name} type anchor is invalid")
		require(isinstance(record["doc"], str) and "\n" not in record["doc"], f"{domain} {name} docs are invalid")
		derives = record["derive"]
		require(isinstance(derives, list) and (derives or schema["domain"] == "vision") and all(isinstance(trait, str) and trait in {"Clone", "Copy", "Debug", "Default", "PartialEq", "Eq"} for trait in derives) and len(set(derives)) == len(derives), f"{domain} {name} derives are invalid")
	for record in types:
		name = record["name"]
		mappings = record.get("mappings", [])
		require(isinstance(mappings, list) and (not mappings or record["kind"] == "enum"), f"{domain} {name} mappings require an enum")
		methods, mapping_fields = set(), set()
		for mapping in mappings:
			require(isinstance(mapping, dict) and set(mapping) == {"name", "visibility", "returns", "field"}, f"{domain} {name} mapping record is invalid")
			require(isinstance(mapping["name"], str) and IDENTIFIER.fullmatch(mapping["name"]), f"{domain} {name} mapping name is invalid")
			validate_unique(mapping["name"], methods, f"{domain} {name} methods")
			require(isinstance(mapping["field"], str) and IDENTIFIER.fullmatch(mapping["field"]) and mapping["field"] not in {"name", "doc"}, f"{domain} {name} mapping field is invalid")
			validate_unique(mapping["field"], mapping_fields, f"{domain} {name} mapping fields")
			require(isinstance(mapping["visibility"], str) and mapping["visibility"] in {"", "pub(super)"} and isinstance(mapping["returns"], str) and mapping["returns"] in {"u32", "&'static str", "KernelId"}, f"{domain} {name} mapping metadata is invalid")
		if record["kind"] == "struct":
			require(isinstance(record["fields"], list) and record["fields"] and record["variants"] == [] and record["default_variant"] is None, f"{domain} {name} struct members are invalid")
			fields = set()
			for field in record["fields"]:
				require(isinstance(field, dict) and set(field) == {"name", "type", "visibility", "doc"}, f"{domain} {name} field record is invalid")
				require(isinstance(field["name"], str) and IDENTIFIER.fullmatch(field["name"]), f"{domain} {name} field name is invalid")
				validate_unique(field["name"], fields, f"{domain} {name} fields")
				require(isinstance(field["type"], str) and (field["type"] in names | {"u32", "f32", "bool"} | ({"i32", "Matrix"} if schema["domain"] == "vision" else set()) or re.fullmatch(r"\[(?:u32|f32|bool); [1-9][0-9]*\]", field["type"])) and isinstance(field["visibility"], str) and field["visibility"] in {"pub", ""} and isinstance(field["doc"], str) and "\n" not in field["doc"], f"{domain} {name} field metadata is invalid")
			defaults = record["defaults"]
			require(defaults is None or isinstance(defaults, dict) and set(defaults) == fields and all(isinstance(value, str) and value.strip() and (";" not in value or re.fullmatch(r"\[[^;\n]+; [1-9][0-9]*\]", value)) for value in defaults.values()), f"{domain} {name} defaults do not cover fields")
			require(defaults is None or "Default" not in record["derive"], f"{domain} {name} has two Default implementations")
		else:
			require(record["fields"] == [] and record["defaults"] is None and isinstance(record["variants"], list) and record["variants"], f"{domain} {name} enum members are invalid")
			variants = set()
			for variant in record["variants"]:
				require(isinstance(variant, dict) and set(variant) == {"name", "doc"} | mapping_fields and isinstance(variant["name"], str) and IDENTIFIER.fullmatch(variant["name"]) and isinstance(variant["doc"], str) and "\n" not in variant["doc"], f"{domain} {name} variant is invalid")
				validate_unique(variant["name"], variants, f"{domain} {name} variants")
				for mapping in mappings:
					value = variant[mapping["field"]]
					require(isinstance(value, str) and value, f"{domain} {name} mapping value is invalid")
					if mapping["returns"] == "u32":
						require(value.isascii() and value.isdecimal() and int(value) <= 0xffffffff, f"{domain} {name} mapping code exceeds u32")
					elif mapping["returns"] == "KernelId":
						require(value in {"KernelId::" + kernel["kernel_id"] for kernel in schema["kernels"]}, f"{domain} {name} mapping kernel is unknown")
					else:
						require(re.fullmatch(r"[a-z][a-z0-9_]*", value), f"{domain} {name} mapping token is invalid")
			for mapping in mappings:
				values = [variant[mapping["field"]] for variant in record["variants"]]
				require(len(set(values)) == len(values), f"{domain} {name} mapping identities collide")
			require(record["default_variant"] is None or isinstance(record["default_variant"], str), f"{domain} {name} default variant is invalid")
			require((record["default_variant"] in variants) == ("Default" in record["derive"]), f"{domain} {name} enum default is inconsistent")


def generate_type(record: dict[str, Any]) -> str:
	doc = f"/// {record['doc']}\n" if record["doc"] else ""
	text = doc
	if record["derive"]:
		text += "#[derive(" + ", ".join(record["derive"]) + ")]\n"
	visibility = record["visibility"] + " " if record["visibility"] else ""
	text += f"{visibility}{record['kind']} {record['name']} {{\n"
	if record["kind"] == "struct":
		for field in record["fields"]:
			if field["doc"]:
				text += f"/// {field['doc']}\n"
			visibility = field["visibility"] + " " if field["visibility"] else ""
			text += f"{visibility}{field['name']}: {field['type']},\n"
	else:
		for variant in record["variants"]:
			if variant["doc"]:
				text += f"/// {variant['doc']}\n"
			if variant["name"] == record["default_variant"]:
				text += "#[default]\n"
			text += variant["name"] + ",\n"
	text += "}\n"
	if record["defaults"] is not None:
		text += f"\nimpl Default for {record['name']} {{\nfn default() -> Self {{\nSelf {{\n"
		text += "".join(f"{field['name']}: {record['defaults'][field['name']]},\n" for field in record["fields"])
		text += "}\n}\n}\n"
	if record.get("mappings"):
		text += f"\nimpl {record['name']} {{\n"
		for mapping in record["mappings"]:
			visibility = mapping["visibility"] + " " if mapping["visibility"] else ""
			text += f"{visibility}const fn {mapping['name']}(self) -> {mapping['returns']} {{\nmatch self {{\n"
			for variant in record["variants"]:
				value = variant[mapping["field"]]
				if mapping["returns"] == "&'static str":
					value = json.dumps(value)
				text += f"Self::{variant['name']} => {value},\n"
			text += "}\n}\n\n"
		text += "}\n"
	return text


def generate_audio_api(schema: dict[str, Any], schema_hash: str, category: str) -> str:
	text = ""
	module = (Path(__file__).parent / "template/audio" / f"{category}_module.rs.in").read_text(encoding="utf-8")
	require(module.count("{{items}}") == 1, f"Audio {category} module requires exactly one item insertion")
	for op in schema["contracts"]:
		if op["rust"]["category"] != category:
			continue
		rust = op["rust"]
		for record in schema["types"]:
			if record["before"] == op["name"]:
				text += "\n" + generate_type(record)
		parameters = ", ".join(f"{name}: {kind}" for name, kind in rust["parameters"])
		text += f"\n/// {rust['doc']}\npub fn {op['name']}({parameters}) -> Result<{rust['returns']}> {{\n"
		text += render_audio_body(op, {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}) + "\n}\n"
	return banner(schema_hash, "//", "audio.json") + module.replace("{{items}}", text)


def generate_api(schema: dict[str, Any], schema_hash: str) -> str:
	lines = []
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
				"/// The call records device work and returns without a host wait.",
				"///",
				"/// # Errors",
				"///",
				"/// Returns an error when the shape, dtype, ownership, allocation, or recording contract fails.",
			]
		)
		if operation["kind"] == "binary":
			lines.extend(
				[
					f"pub fn {name}(left: &Matrix, right: &Matrix) -> Result<Matrix> {{",
					matrix_body("binary", name, autograd=operation.get("autograd"), routes=f"&[{routes}]", broadcast_routes=rust_broadcast_routes(operation)),
					"}",
				]
			)
		elif operation["kind"] == "unary":
			lines.extend(
				[
					f"pub fn {name}(input: &Matrix) -> Result<Matrix> {{",
					matrix_body("unary", name, autograd=operation.get("autograd"), routes=f"&[{routes}]"),
					"}",
				]
			)
		else:
			lines.extend(
				[
					f"pub fn {name}(input: &Matrix, {operation['scalar_name']}: f32) -> Result<Matrix> {{",
					matrix_body("unary_scalar", name, autograd=operation.get("autograd"), routes=f"&[{routes}]", scalar=operation["scalar_name"]),
					"}",
				]
			)
		lines.append("")
	module = (Path(__file__).parent / "template/matrix/elemwise_module.rs.in").read_text(encoding="utf-8")
	require(
		module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		"Matrix elementwise module must contain only one items placeholder",
	)
	return banner(schema_hash, "//") + module.replace("{{items}}", "\n".join(lines) + "\n" + generate_matrix_backward_providers(schema, "elemwise"))


def registry_entries(
	elementwise: dict[str, Any],
	blas: dict[str, Any],
	reduce: dict[str, Any],
	rng: dict[str, Any],
	index: dict[str, Any],
	ml: dict[str, Any],
	audio: dict[str, Any],
	cryptography: dict[str, Any],
	cryptography_pqc: dict[str, Any],
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
	for kernel in cryptography_pqc["kernels"]:
		semantic = kernel["semantic_operation"]
		entries.append(
			{
				"domain": "cryptography",
				"report_domain": "cryptography.pqc",
				"name": kernel["name"],
				"dtype": cryptography_pqc["dtype"],
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel.get("physical_write"),
				"semantic_contract": f"crate::core::operation::cryptography::pqc::{rust_const_name(semantic)}",
			}
		)
	for kernel in cryptography_pqc.get("private_kernels", []):
		entries.append({
			"domain": "cryptography", "report_domain": "cryptography.pqc",
			"name": kernel["name"], "dtype": cryptography_pqc["dtype"],
			"kernel_id": kernel["kernel_id"], "stable_id": kernel["stable_id"],
			"workgroup_size": kernel["workgroup_size"], "dispatch_tile_size": kernel["workgroup_size"],
			"training_replay_role": "safe", "physical_write": kernel["physical_write"],
			"semantic_contract": None, "requires_secret_storage": True,
		})
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
	buffer_counts: dict[str, int] = {}
	for operation in elementwise["operations"]:
		kind = operation["kind"]
		count = {"binary": 3, "unary": 2, "unary_scalar": 2}[kind]
		for variant in operation_variants(elementwise, operation):
			buffer_counts[variant["kernel_id"]] = count
		for lowering in operation_lowering_variants(operation):
			buffer_counts[lowering["kernel_id"]] = count_buffer_indices(lowering)
	for operation in blas["operations"]:
		buffer_counts[operation["kernel_id"]] = 3
	for schema in [reduce, rng, index, ml, audio, cryptography, cryptography_pqc, image, vision]:
		for operation in schema.get("operations", schema.get("kernels", [])) + schema.get("private_kernels", []):
			buffer_counts[operation["kernel_id"]] = count_buffer_indices(operation)
	for entry in entries:
		entry["bounded_buffer_count"] = buffer_counts[entry["kernel_id"]]
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


def count_buffer_indices(operation: dict[str, Any]) -> int:
	fields = operation["push_fields"]
	count = sum(field[0].endswith(("_index", "_idx")) for field in fields)
	require(count > 0, f"{operation['name']} has no storage-buffer push indices")
	return count


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
		"tile_32x32": "Tile32x32",
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
	cryptography_pqc: dict[str, Any],
	cryptography_pqc_hash: str,
	image: dict[str, Any],
	image_hash: str,
	vision: dict[str, Any],
	vision_hash: str,
) -> str:
	entries = registry_entries(elementwise, blas, reduce, rng, index, ml, audio, cryptography, cryptography_pqc, image, vision)
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
		f"pub(crate) const MAX_BOUNDED_STORAGE_BUFFERS: u32 = {max(entry['bounded_buffer_count'] for entry in entries)};",
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
				f"static BOUNDED_{static_name(domain, name, dtype)}: ShaderArtifact = ShaderArtifact {{",
				f'\tbytes: include_bytes!(concat!(env!("OUT_DIR"), "/{domain}_{name}_{dtype}_bounded.spv")),',
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
	lines.extend(["}", "", "impl KernelId {", "\tpub(crate) const fn mldsa_keygen_layout(parameter: u32) -> Option<(u64, u64, u64)> {", "\t\tmatch parameter {"])
	for kernel in cryptography_pqc.get("private_kernels", []):
		if kernel["name"] == "ml_dsa_keygen":
			for row in kernel["parameter_layouts"]:
				lines.append(f"\t\t\t{row['parameter_set']} => Some(({row['seed_bytes']}, {row['public_bytes']}, {row['private_bytes']})),")
	lines.extend(["\t\t\t_ => None,", "\t\t}", "\t}", "\tpub(crate) const fn mldsa_sign_layout(parameter: u32) -> Option<(u64, u64, u64, u64)> {", "\t\tmatch parameter {"])
	for kernel in cryptography_pqc.get("private_kernels", []):
		if kernel["name"] == "ml_dsa_sign":
			for row in kernel["parameter_layouts"]:
				lines.append(f"\t\t\t{row['parameter_set']} => Some(({row['seed_bytes']}, {row['private_bytes']}, {row['mu_bytes']}, {row['signature_bytes']})),")
	workspace = next(kernel["workspace_bytes"] for kernel in cryptography_pqc["private_kernels"] if kernel["name"] == "ml_dsa_sign")
	lines.extend(["\t\t\t_ => None,", "\t\t}", "\t}", "\tpub(crate) const fn mldsa_sign_workspace_bytes() -> usize {", f"\t\t{workspace}", "\t}", "\tpub(crate) const fn mldsa_prehash_bytes(algorithm: u32) -> Option<u64> {", "\t\tmatch algorithm {"])
	for row in cryptography_pqc["prehashes"]:
		lines.append(f"\t\t\t{row['oid_suffix']} => Some({row['digest_bytes']}),")
	lines.extend(["\t\t\t_ => None,", "\t\t}", "\t}", "}", "", "impl KernelId {", f"\tpub(crate) const ALL: [Self; {len(entries)}] = ["])
	for entry in entries:
		lines.append(f"\t\tSelf::{entry['kernel_id']},")
	lines.extend(["\t];", "", "\tpub(crate) const fn artifact(self) -> &'static ShaderArtifact {", "\t\tmatch self {"])
	for entry in entries:
		lines.append(
			f"\t\t\tSelf::{entry['kernel_id']} => &{static_name(entry['domain'], entry['name'], entry['dtype'])},"
		)
	lines.extend(["\t\t}", "\t}", "", "\tpub(crate) const fn bounded_artifact(self) -> &'static ShaderArtifact {", "\t\tmatch self {"])
	for entry in entries:
		lines.append(
			f"\t\t\tSelf::{entry['kernel_id']} => &BOUNDED_{static_name(entry['domain'], entry['name'], entry['dtype'])},"
		)
	secret_candidates = [f"Self::{entry['kernel_id']}" for entry in entries if entry.get("requires_secret_storage", False)]
	secret_match = " | ".join(secret_candidates)
	lines.extend(["\t\t}", "\t}", "", "\tpub(crate) const fn requires_secret_storage(self) -> bool {", f"\t\tmatches!(self, {secret_match})" if secret_match else "\t\tfalse", "\t}", "", "\tpub(crate) const fn bounded_buffer_count(self) -> u32 {", "\t\tmatch self {"])
	for entry in entries:
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => {entry['bounded_buffer_count']},")
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
	cryptography_pqc: dict[str, Any],
	cryptography_pqc_hash: str,
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
		f"// cryptography_pqc_sha256={cryptography_pqc_hash}",
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
	lines.extend([
		"\t}",
		"",
		"\tpub mod pqc {",
		"\t\tuse super::*;",
		"",
	])
	for parameter in dict.fromkeys(kernel["parameter_set"] for kernel in cryptography_pqc["kernels"]):
		pk_bytes, sig_bytes = MLDSA_WIRE_LAYOUTS[parameter]
		lines.append(f"\t\tpub(crate) const ML_DSA_{parameter}_PUBLIC_KEY_SIZE: usize = {pk_bytes};")
		lines.append(f"\t\tpub(crate) const ML_DSA_{parameter}_SIGNATURE_SIZE: usize = {sig_bytes};")
	lines.append("")
	for operation in cryptography_pqc["contracts"]:
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
		for line in semantic_contract_lines("cryptography::pqc", operation["name"], contract, attributes):
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
		"use crate::ml::autograd::{node::{GradNode, operation::GradNodeOperation}, tape::record_node};",
		"",
	]
	lines.append("use crate::ml::autograd::node::operation::{" + ", ".join(
		"GradNode" + operation["autograd"]["node"] for operation in operations
	) + "};")
	lines.append("")
	if any(
		scalar_type == "UpsampleMode"
		for operation in operations
		for _, scalar_type in operation["autograd"].get("saved_scalars", [])
	):
		lines.extend(["use crate::ml::matrix::UpsampleMode;", ""])
	for operation in operations:
		autograd = operation["autograd"]
		inputs = autograd["inputs"]
		saved = autograd["saved_matrices"]
		owned = autograd.get("owned_matrices", [])
		scalars = autograd.get("saved_scalars", [])
		output_param = autograd.get("output_param", "result")
		matrices = [*inputs, *(name for name in saved if name not in inputs)]
		arguments = ", ".join(
			[
				*(f"{name}: Matrix" if name in owned else f"{name}: &Matrix" for name in matrices),
				*(f"{name}: {scalar_type}" for name, scalar_type in scalars),
				f"{output_param}: &Matrix",
			]
		)
		lines.extend(
			[
				f"pub(in crate::ml) fn record_{operation['name']}({arguments}) -> Result<()> {{",
				f"\trecord_node(GradNode::Operation(GradNodeOperation::{autograd['node']}(GradNode{autograd['node']} {{",
			]
		)
		for name in matrices:
			lines.append(f"\t\t{name}," if name in owned else f"\t\t{name}: {name}.clone(),")
		for name, _ in scalars:
			lines.append(f"\t\t{name},")
		lines.extend(
			[
				f"\t\toutput_id: {output_param}.value_id(),",
				"\t})))",
				"}",
				"",
			]
		)
	return "\n".join(lines)


def ml_autograd_operations(ml: dict[str, Any]) -> list[dict[str, Any]]:
	return [operation for operation in ml["operations"]
		if operation.get("autograd", {}).get("mode") == "generated"]


def generate_ml_autograd_records(ml: dict[str, Any], ml_hash: str, family: str) -> str:
	"""Emit saved values and audited reverse recipes from the same forward rows."""
	operations = [op for op in ml_autograd_operations(ml) if op["autograd"]["family"] == family]
	path = Path(__file__).resolve().parent / "template/ml/autograd" / f"{family}.rs.in"
	require(path.is_file(), f"missing autograd recipe template for {family}")
	entries = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", path.read_text(), re.S)
	recipes = dict(entries)
	require(len(entries) == len(recipes), f"duplicate autograd recipe in {family}")
	require(set(recipes) == {op["name"] for op in operations},
		f"autograd recipe coverage must exactly match {family} operation rows")
	lines = [banner(ml_hash, "//", "ml_training.json").rstrip(),
		"// Saved-state and backward recipes; lifecycle remains tape-owned.",
		"use crate::{Matrix, Result};",
		"use std::collections::HashMap;",
		"use crate::ml::autograd::node::operation::BackwardStatus;",
		"use crate::ml::autograd::tape::backward::accumulate_value_gradient;",
	]
	# Recipe dependencies are family-local; unused imports remain lint errors.
	if family == "loss":
		lines.append("use crate::{matrix, ml::loss};")
	elif family in {"matrix/activation", "matrix/swiglu"}:
		lines.append("use crate::ml::matrix as ml_matrix;")
	elif family == "matrix/rope":
		lines.append("use crate::ml::matrix as ml_matrix;")
	if any(typ == "UpsampleMode" for op in operations for _, typ in op["autograd"].get("saved_scalars", [])):
		lines.append("use crate::ml::matrix::UpsampleMode;")
	for operation in operations:
		autograd = operation["autograd"]
		node = "GradNode" + autograd["node"]
		matrices = list(dict.fromkeys([*autograd["inputs"], *autograd["saved_matrices"]]))
		fields = [(name, "Matrix") for name in matrices] + autograd.get("saved_scalars", []) + [("output_id", "u64")]
		recipe = recipes[operation["name"]].strip()
		binding = re.search(r"let Self \{([^}]+)\} = self;", recipe)
		bound = [name.strip() for name in binding[1].split(",") if name.strip()] if binding else []
		require(len(bound) == len(set(bound)) and set(bound) == {name for name, _ in fields},
			f"autograd recipe saved fields must match {operation['name']} operation row")
		lines.extend(["", f"pub(in crate::ml::autograd) struct {node} {{"])
		lines.extend(f"\tpub(in crate::ml::autograd) {name}: {typ}," for name, typ in fields)
		lines.extend(["}", "", f"impl {node} {{",
			"\tpub(in crate::ml::autograd) fn backward(&self, gradients: &mut HashMap<u64, Matrix>) -> Result<BackwardStatus> {"])
		lines.extend("\t\t" + line for line in recipe.splitlines())
		lines.extend(["\t}", "}"])
	return "\n".join(lines) + "\n"


def generate_ml_autograd_routing(ml: dict[str, Any], ml_hash: str) -> str:
	"""Generate exhaustive output identity and reverse routing without a second catalog."""
	operations = ml_autograd_operations(ml)
	families = sorted({op["autograd"]["family"] for op in operations})
	lines = [banner(ml_hash, "//", "ml_training.json").rstrip(),
		"// Exhaustive routing for schema-owned saved operation records.",
		"use crate::{Matrix, Result};", "use std::collections::HashMap;"]
	for family in families:
		module = family.replace("/", "_")
		lines.extend([f'#[path = "{family}.gen.rs"]', f"mod {module};"])
		lines.append(f"pub(in crate::ml::autograd) use {module}::{{" + ", ".join(
			"GradNode" + op["autograd"]["node"] for op in operations if op["autograd"]["family"] == family
		) + "};")
	lines.extend(["", "pub(in crate::ml::autograd) enum BackwardStatus {", "\tSkipped,", "\tRecorded,", "}",
		"", "pub(in crate::ml::autograd) enum GradNodeOperation {"])
	lines.extend(f"\t{op['autograd']['node']}(GradNode{op['autograd']['node']})," for op in operations)
	lines.extend(["}", "", "impl GradNodeOperation {", "\tpub(in crate::ml::autograd) fn output_id(&self) -> u64 {", "\t\tmatch self {"])
	lines.extend(f"\t\t\tSelf::{op['autograd']['node']}(node) => node.output_id," for op in operations)
	lines.extend(["\t\t}", "\t}", "", "\tpub(in crate::ml::autograd) fn backward(&self, gradients: &mut HashMap<u64, Matrix>) -> Result<BackwardStatus> {", "\t\tmatch self {"])
	lines.extend(f"\t\t\tSelf::{op['autograd']['node']}(node) => node.backward(gradients)," for op in operations)
	lines.extend(["\t\t}", "\t}", "}", ""])
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
	items = f"""
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
{matrix_body('axis_normalization', softmax['name'], kernel='KernelId::' + softmax['kernel_id'], node='softmax')}
}}

pub(crate) fn {softmax_backward['name']}(
	forward_output: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
) -> Result<Matrix> {{
{matrix_body('axis_normalization_backward', softmax_backward['name'], kernel='KernelId::' + softmax_backward['kernel_id'])}
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
{matrix_body('axis_normalization', log_softmax['name'], kernel='KernelId::' + log_softmax['kernel_id'], node='log_softmax')}
}}

pub(crate) fn {log_softmax_backward['name']}(
	forward_output: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
) -> Result<Matrix> {{
{matrix_body('axis_normalization_backward', log_softmax_backward['name'], kernel='KernelId::' + log_softmax_backward['kernel_id'])}
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
{matrix_body('sum', sum_operation['name'])}
}}

pub(crate) fn {sum_backward['name']}(
	input: &Matrix,
	output_gradient: &Matrix,
	dim: i32,
) -> Result<Matrix> {{
{matrix_body('sum_backward', sum_backward['name'])}
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
{matrix_body('accuracy', accuracy_count['name'])}
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
{matrix_body('accuracy', masked_accuracy_count['name'])}
}}
"""
	module = (Path(__file__).parent / "template/matrix/reduce_module.rs.in").read_text(encoding="utf-8")
	require(
		module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		"Matrix reduce module must contain only one items placeholder",
	)
	return banner(schema_hash, "//", "matrix_reduce.json") + module.replace("{{items}}", items)


def generate_blas_api(schema: dict[str, Any], schema_hash: str) -> str:
	operation = schema["operations"][0]
	kernel_id = operation["kernel_id"]
	contract_const = rust_const_name(operation["name"])
	body_template = (Path(__file__).parent / "template/matrix/mat_mul_nt.rs.in").read_text(encoding="utf-8")
	items = f"""
/// {operation['doc']}
///
/// `left` has shape `[M, K]`, `right` has shape `[N, K]`, and the
/// returned matrix has shape `[M, N]`. This matches OA's native
/// weight-layout convention and does not transpose storage.
///
/// The call records device work and returns without a host wait.
///
/// # Errors
///
/// Returns an error when either input is not a rank-two FP32 matrix,
/// their K extents differ, they belong to different engines, a dimension
/// exceeds the admitted shader ABI, or allocation or recording fails.
pub fn {operation['name']}(left: &Matrix, right: &Matrix) -> Result<Matrix> {{
\tlet contract = crate::core::operation::matrix::{contract_const};
\tlet kernel = KernelId::{kernel_id};
{body_template.rstrip()}
}}
"""
	module = (Path(__file__).parent / "template/matrix/blas_module.rs.in").read_text(encoding="utf-8")
	require(
		module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		"Matrix BLAS module must contain only one items placeholder",
	)
	return banner(schema_hash, "//", "matrix_blas.json") + module.replace("{{items}}", items + "\n" + generate_matrix_backward_providers(schema, "blas"))


def _render_ml_activation_template(kind: str, values: dict[str, str]) -> str:
	template = (Path(__file__).parent / f"template/ml/activation_{kind}.rs.in").read_text(encoding="utf-8")
	placeholders = set(re.findall(r"{{(\w+)}}", template))
	require(placeholders == set(values), f"ML activation {kind} template placeholders disagree with renderer")
	return re.sub(r"{{(\w+)}}", lambda match: values[match.group(1)], template).rstrip()


def _ml_activation_fn(op: dict[str, Any]) -> str:
	"""Emit one complete Rust function body for an ML activation operation."""
	name = op["name"]
	kind = op["kind"]
	kernel = op["kernel_id"]
	contract = f"crate::core::operation::ml::{rust_const_name(name)}"

	if kind == "unary":
		doc = op.get("doc", f"Apply {name} elementwise.")
		saved = op["autograd"]["saved_matrices"][0]  # "input" or "output"
		# saved == "input": record(input, output) / saved == "output": record(output, output)
		if saved == "input":
			autograd_call = f"\tautograd::record_{name}(input, &output)?;"
		else:
			autograd_call = f"\tautograd::record_{name}(input, &output, &output)?;"
		return _render_ml_activation_template(
			"unary",
			{
				"doc": doc,
				"name": name,
				"contract": contract,
				"kernel": kernel,
				"autograd_call": autograd_call,
			},
		)

	if kind == "unary_backward":
		saved = op["saved"]  # "input" or "output"
		saved_param = f"saved_{saved}"
		return _render_ml_activation_template(
			"unary_backward",
			{
				"name": name,
				"saved_param": saved_param,
				"contract": contract,
				"kernel": kernel,
			},
		)

	if kind == "unary_scalar":
		scalar = op["scalar"]
		doc = op.get("doc", f"Apply {name} elementwise.")
		autograd = op["autograd"]
		saved = autograd["saved_matrices"][0]
		if saved == "input":
			autograd_call = f"\tautograd::record_{name}(input, {scalar}, &output)?;"
		else:
			autograd_call = f"\tautograd::record_{name}(input, &output, {scalar}, &output)?;"
		return _render_ml_activation_template(
			"unary_scalar",
			{
				"doc": doc,
				"name": name,
				"scalar": scalar,
				"contract": contract,
				"kernel": kernel,
				"autograd_call": autograd_call,
			},
		)

	if kind == "unary_scalar_backward":
		scalar = op["scalar"]
		saved = op["saved"]
		saved_param = f"saved_{saved}"
		return _render_ml_activation_template(
			"unary_scalar_backward",
			{
				"name": name,
				"saved_param": saved_param,
				"scalar": scalar,
				"contract": contract,
				"kernel": kernel,
			},
		)

	if kind == "binary":
		inputs = op.get("inputs", ["left", "right"])
		left, right = inputs[0], inputs[1]
		doc = op.get("doc", f"Apply {name} elementwise.")
		autograd_call = f"\tautograd::record_{name}({left}, {right}, &output)?;"
		return _render_ml_activation_template(
			"binary",
			{
				"doc": doc,
				"name": name,
				"left": left,
				"right": right,
				"contract": contract,
				"kernel": kernel,
				"autograd_call": autograd_call,
			},
		)

	if kind == "binary_backward":
		inputs = op.get("inputs", ["left", "right"])
		outputs = op.get("outputs", ["left_gradient", "right_gradient"])
		left, right = inputs[0], inputs[1]
		out0, out1 = outputs[0], outputs[1]
		return _render_ml_activation_template(
			"binary_backward",
			{
				"name": name,
				"left": left,
				"right": right,
				"contract": contract,
				"out0": out0,
				"out1": out1,
				"kernel": kernel,
			},
		)

	if kind == "silu_mul":
		doc = op.get("doc", "Apply SiLU(gate) * up to concatenated halves of the final input axis.")
		return _render_ml_activation_template(
			"silu_mul",
			{
				"doc": doc,
				"name": name,
				"contract": contract,
				"kernel": kernel,
			},
		)

	require(kind == "silu_mul_backward", f"Unsupported ML activation body kind: {kind}")
	return _render_ml_activation_template("silu_mul_backward", {"name": name, "contract": contract, "kernel": kernel})



def generate_ml_activation_api(schema: dict[str, Any], schema_hash: str, family: str = "activation") -> str:
	module = (Path(__file__).parent / "template/ml/activation_module.rs.in").read_text(encoding="utf-8")
	require(
		module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		"ML activation module must contain only one items placeholder",
	)
	items = "\n\n".join(_ml_activation_fn(op) for op in schema["operations"] if op["rust_family"] == family)
	return banner(schema_hash, "//", "ml_activation.json") + module.replace("{{items}}", items)


def generate_ml_matrix_family(ml: dict[str, Any], ml_hash: str, family: str) -> str:
	"""Publish complete provider bodies selected by the canonical operation rows."""
	operations = [op for op in ml["operations"] if op.get("rust_family") == family]
	path = Path(__file__).resolve().parent / "template/ml/matrix"
	module = (path / f"{family}_module.rs.in").read_text()
	entries = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", (path / f"{family}.rs.in").read_text(), re.S)
	bodies = dict(entries)
	require(len(entries) == len(bodies) and set(bodies) == {op["name"] for op in operations},
		f"ML Matrix {family} body coverage must exactly match operation rows")
	require(module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		f"ML Matrix {family} module must contain exactly one items placeholder")
	for operation in operations:
		require(re.search(r"fn " + operation["name"] + r"\(", bodies[operation["name"]]) is not None,
			f"ML Matrix {family} body must implement its operation name")
	items = "\n\n".join(bodies[op["name"]].strip() for op in operations)
	return banner(ml_hash, "//", "ml_training.json") + module.replace("{{items}}", items)


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


def generate_mldsa_prehash_layout_shader(schema: dict[str, Any], digest: str) -> str:
	cases = "\n".join(f"\t\tcase {row['oid_suffix']}u: return {row['digest_bytes']}u;" for row in schema["prehashes"])
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// FIPS 204 Algorithms 4/5 framing and NIST CSOR hashAlgs OIDs.
// No live C++ GPU prehash donor exists. Existing signing core is unchanged.
// Digest computation, secret execution and security qualification are separate.
import shake;

uint mldsa_prehash_bytes(uint algorithm) {{
	switch (algorithm) {{
{cases}
		default: return 0u;
	}}
}}

// Absorb the shared HashML-DSA M' framing after tr. Ranges, algorithm and
// exact digest length must already be checked by the signing/verifying caller.
void mldsa_absorb_prehash_header(inout PqcShake256 sponge,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm) {{
	pqc_shake256_absorb_byte(sponge, 1u);
	pqc_shake256_absorb_byte(sponge, context_length);
	pqc_shake256_absorb_buffer(sponge, contexts, context_base, context_length);
	uint prefix[10] = {{6u, 9u, 96u, 134u, 72u, 1u, 101u, 3u, 4u, 2u}};
	for (uint i = 0u; i < 10u; ++i) pqc_shake256_absorb_byte(sponge, prefix[i]);
	pqc_shake256_absorb_byte(sponge, algorithm);
}}

void mldsa_absorb_prehash_framing(inout PqcShake256 sponge,
	RWByteAddressBuffer digests, uint digest_base, uint digest_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm) {{
	mldsa_absorb_prehash_header(sponge, contexts, context_base, context_length, algorithm);
	pqc_shake256_absorb_buffer(sponge, digests, digest_base, digest_length);
}}
"""


def generate_mldsa_prehash_digest_shader(schema: dict[str, Any], digest: str) -> str:
	cases = []
	for row in schema["prehashes"]:
		name = row["name"]
		if name in ("SHA2-224", "SHA2-256"):
			bits = int(name.split("-")[1])
			body = (f"uint words[8];\n"
				f"\t\t\tif (!sha2_u32_digest(messages, base, length, {bits}u, words)) return false;\n"
				f"\t\t\tfor (uint i = 0u; i < {row['digest_bytes'] // 4}u; ++i) digest[i] = words[i];")
		elif name.startswith("SHA2-"):
			bits = int(name.split("/")[-1] if "/" in name else name.split("-")[1])
			body = f"return sha2_u32_pair_digest(messages, base, length, {bits}u, digest);"
		elif name.startswith("SHA3-"):
			bits = int(name.split("-")[1])
			body = ("uint2 state[25];\n"
				"\t\t\tfor (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);\n"
				f"\t\t\tif (!sha3_absorb(state, messages, base, length, SHA3_{bits}_RATE_BYTES)) return false;\n"
				f"\t\t\tfor (uint i = 0u; i < {row['digest_bytes'] // 4}u; ++i) digest[i] = (i & 1u) == 0u ? state[i / 2u].x : state[i / 2u].y;")
		else:
			body = f"return pqc_shake_prehash(messages, base, length, {int(name.split('-')[1])}u, digest);"
		ending = "" if body.startswith("return ") else "\n\t\t\tbreak;"
		cases.append(f"\t\tcase {row['oid_suffix']}u: {{\n\t\t\t{body}{ending}\n\t\t}}")
	dispatch = "\n".join(cases)
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// FIPS 204 hash selector/OID/extent authority: cryptography_pqc.json.
// No OA C++ HashML-DSA composition donor exists. Reuses checked digest formulas.
// Private public-data composition; not Engine or secret-operation admission.
import mldsa_prehash_layout;
import sha2_u32;
import sha2_u32_pair;
import sha3;
import shake;
import shake_prehash;

// Packed little-endian digest words; leave the unused tail and rejected output
// untouched. Selector 11/12 fixes SHAKE output to 32/64 bytes per FIPS 204.
bool mldsa_prehash_digest(
	RWByteAddressBuffer messages,
	uint base,
	uint length,
	uint algorithm,
	inout uint digest[16]
) {{
	if (mldsa_prehash_bytes(algorithm) == 0u || !pqc_buffer_range(messages, base, length)) return false;
	switch (algorithm) {{
{dispatch}
		default: return false;
	}}
	return true;
}}

// Compute mu = SHAKE256(tr || 1 || ctx_len || ctx || OID || PH(M), 64).
// Digest stays local: no intermediate storage allocation or host observation.
// All ranges are checked before hashing. Rejected output remains untouched.
bool mldsa_prehash_mu(
	RWByteAddressBuffer messages,
	uint message_base,
	uint message_length,
	RWByteAddressBuffer tr,
	uint tr_base,
	RWByteAddressBuffer contexts,
	uint context_base,
	uint context_length,
	uint algorithm,
	inout uint mu[16]
) {{
	uint digest_length = mldsa_prehash_bytes(algorithm);
	if (digest_length == 0u || context_length > 255u
		|| !pqc_buffer_range(tr, tr_base, 64u)
		|| !pqc_buffer_range(contexts, context_base, context_length)) return false;
	uint digest[16];
	if (!mldsa_prehash_digest(messages, message_base, message_length, algorithm, digest)) return false;
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	pqc_shake256_absorb_buffer(sponge, tr, tr_base, 64u);
	mldsa_absorb_prehash_header(sponge, contexts, context_base, context_length, algorithm);
	for (uint i = 0u; i < digest_length; ++i)
		pqc_shake256_absorb_byte(sponge, digest[i / 4u] >> ((i & 3u) * 8u));
	pqc_shake256_finalize(sponge);
	for (uint word = 0u; word < 16u; ++word) {{
		uint value = 0u;
		for (uint byte = 0u; byte < 4u; ++byte)
			value |= pqc_shake256_squeeze_byte(sponge) << (byte * 8u);
		mu[word] = value;
	}}
	return true;
}}
"""


def generate_mldsa_prehash_shader(schema: dict[str, Any], digest: str) -> str:
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// FIPS 204 Algorithms 4/5; no live C++ GPU prehash donor exists.
import mldsa_sign;
import mldsa_prehash_layout;
import mldsa_prehash_digest;
import mldsa_sign_workspace_layout;
import shake;

// Accept a precomputed public digest, not an arbitrary message. The fixed
// selector binds both digest length and DER OID; callers cannot supply an OID.
// SHAKE-128/256 lengths are fixed to 32/64 bytes, respectively.
bool mldsa_sign_prehashed_cached<T : IMldsaSignCache>(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer digests, uint digest_base, uint digest_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, inout T cache, inout uint signature[4627], out uint attempts) {{
	attempts = 0u;
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	uint length = mldsa_prehash_bytes(algorithm);
	if (length == 0u || digest_length != length || context_length > 255u) return false;
	if (!pqc_buffer_range(digests, digest_base, length)
		|| !pqc_buffer_range(contexts, context_base, context_length)) return false;
	for (uint i = 64u; i < 128u; ++i) if (key[i] > 255u) return false;
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	for (uint i = 64u; i < 128u; ++i) pqc_shake256_absorb_byte(sponge, key[i]);
	mldsa_absorb_prehash_framing(sponge, digests, digest_base, length,
		contexts, context_base, context_length, algorithm);
	pqc_shake256_finalize(sponge);
	uint mu[64];
	for (uint i = 0u; i < 64u; ++i) mu[i] = pqc_shake256_squeeze_byte(sponge);
	return mldsa_sign_mu_cached(parameter_set, key, mu, rnd, cache, signature, attempts);
}}
// Existing fixture callers retain their local cache.
bool mldsa_sign_prehashed(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer digests, uint digest_base, uint digest_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, inout uint signature[4627], out uint attempts) {{
	MldsaLocalSignCache cache;
	return mldsa_sign_prehashed_cached(parameter_set, key, rnd,
		digests, digest_base, digest_length, contexts, context_base, context_length,
		algorithm, cache, signature, attempts);
}}

// Workspace extent is checked before message hashing or cache access.
bool mldsa_sign_prehashed_buffer(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer digests, uint digest_base, uint digest_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, RWByteAddressBuffer workspace, uint workspace_base,
	inout uint signature[4627], out uint attempts) {{
	attempts = 0u;
	if ((workspace_base & 3u) != 0u
		|| !pqc_buffer_range(workspace, workspace_base, MLDSA_SIGN_CACHE_BYTES)) return false;
	MldsaBufferSignCache cache;
	cache.buffer = workspace;
	cache.base = workspace_base;
	return mldsa_sign_prehashed_cached(parameter_set, key, rnd,
		digests, digest_base, digest_length, contexts, context_base, context_length,
		algorithm, cache, signature, attempts);
}}

// HashML-DSA accepts the original public message; digest words stay local.
// No secret-key bytes are transported through an ordinary digest buffer.
bool mldsa_sign_hash_message_cached<T : IMldsaSignCache>(
	uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, inout T cache, inout uint signature[4627], out uint attempts) {{
	attempts = 0u;
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	uint length = mldsa_prehash_bytes(algorithm);
	if (length == 0u || context_length > 255u
		|| !pqc_buffer_range(contexts, context_base, context_length)) return false;
	for (uint i = 64u; i < 128u; ++i) if (key[i] > 255u) return false;
	uint digest[16];
	if (!mldsa_prehash_digest(messages, message_base, message_length, algorithm, digest)) return false;
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	for (uint i = 64u; i < 128u; ++i) pqc_shake256_absorb_byte(sponge, key[i]);
	mldsa_absorb_prehash_header(sponge, contexts, context_base, context_length, algorithm);
	for (uint i = 0u; i < length; ++i)
		pqc_shake256_absorb_byte(sponge, (digest[i >> 2u] >> ((i & 3u) * 8u)) & 255u);
	pqc_shake256_finalize(sponge);
	uint mu[64];
	for (uint i = 0u; i < 64u; ++i) mu[i] = pqc_shake256_squeeze_byte(sponge);
	return mldsa_sign_mu_cached(parameter_set, key, mu, rnd, cache, signature, attempts);
}}

// Reject workspace extents before hashing or touching the signing cache.
bool mldsa_sign_hash_message_buffer(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, RWByteAddressBuffer workspace, uint workspace_base,
	inout uint signature[4627], out uint attempts) {{
	attempts = 0u;
	if ((workspace_base & 3u) != 0u
		|| !pqc_buffer_range(workspace, workspace_base, MLDSA_SIGN_CACHE_BYTES)) return false;
	MldsaBufferSignCache cache;
	cache.buffer = workspace;
	cache.base = workspace_base;
	return mldsa_sign_hash_message_cached(parameter_set, key, rnd,
		messages, message_base, message_length, contexts, context_base, context_length,
		algorithm, cache, signature, attempts);
}}
"""


def generate_mldsa_sign_workspace_layout_shader(schema: dict[str, Any], digest: str) -> str:
	kernel = next(row for row in schema["private_kernels"] if row["name"] == "ml_dsa_sign")
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Maximum NTT(s1/s2/t0) cache for all FIPS 204 parameter sets.
static const uint MLDSA_SIGN_CACHE_WORDS = {kernel['workspace_bytes'] // 4}u;
static const uint MLDSA_SIGN_CACHE_BYTES = {kernel['workspace_bytes']}u;
"""


def generate_mldsa_sign_shader(kernel: dict[str, Any], digest: str) -> str:
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	layouts = kernel["parameter_layouts"]
	prehashed = kernel["name"] == "ml_dsa_sign_prehashed"
	hash_message = kernel["name"] == "ml_dsa_sign_hash_message"
	message = kernel["name"] != "ml_dsa_sign"
	prehash_import = "\nimport mldsa_prehash_layout;" if prehashed or hash_message else ""
	input_dimensions = """
	storage_buffers[push.message_index].GetDimensions(message_size);
	storage_buffers[push.context_index].GetDimensions(context_size);""" if message else "\n\tstorage_buffers[push.mu_index].GetDimensions(mu_size);"
	input_check = """push.message_length > 0xfffffffcu || push.context_length > 255u
		|| message_size != max(4u, (push.message_length + 3u) & ~3u)
		|| context_size != max(4u, (push.context_length + 3u) & ~3u)""" if message else "mu_size != 64u"
	input_load = "" if message else """
	for (uint i = 0u; i < 64u; ++i)
		mu[i] = (storage_buffers[push.mu_index].Load(i & ~3u) >> (8u * (i & 3u))) & 255u;"""
	call = """mldsa_sign_message_buffer(parameter, key, rnd,
		storage_buffers[push.message_index], 0u, push.message_length,
		storage_buffers[push.context_index], 0u, push.context_length,
		true, storage_buffers[push.workspace_index], 0u, signature, attempts)""" if message else "mldsa_sign_mu_buffer(parameter, key, mu, rnd, storage_buffers[push.workspace_index], 0u, signature, attempts)"
	if prehashed:
		input_dimensions = """\n\tstorage_buffers[push.digest_index].GetDimensions(message_size);
	storage_buffers[push.context_index].GetDimensions(context_size);"""
		input_check = """mldsa_prehash_bytes(push.algorithm) == 0u || push.context_length > 255u
		|| message_size != mldsa_prehash_bytes(push.algorithm)
		|| context_size != max(4u, (push.context_length + 3u) & ~3u)"""
		call = """mldsa_sign_prehashed_buffer(parameter, key, rnd,
		storage_buffers[push.digest_index], 0u, message_size,
		storage_buffers[push.context_index], 0u, push.context_length,
		push.algorithm, storage_buffers[push.workspace_index], 0u, signature, attempts)"""
	if hash_message:
		input_check = "mldsa_prehash_bytes(push.algorithm) == 0u || " + input_check
		call = """mldsa_sign_hash_message_buffer(parameter, key, rnd,
		storage_buffers[push.message_index], 0u, push.message_length,
		storage_buffers[push.context_index], 0u, push.context_length,
		push.algorithm, storage_buffers[push.workspace_index], 0u, signature, attempts)"""
	workspace_import = "\nimport mldsa_sign_workspace_layout;"
	workspace_dimension = "\n\tuint workspace_size;\n\tstorage_buffers[push.workspace_index].GetDimensions(workspace_size);"
	workspace_check = " || workspace_size != MLDSA_SIGN_CACHE_BYTES"
	def sizes(field: str) -> str:
		return " : ".join(f"parameter == {row['parameter_set']}u ? {row[field]}u" for row in layouts) + " : 0u"
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Private FIPS 204 {"hashed message/context" if hash_message else "prehashed digest/context" if prehashed else "pure message/context" if message else "internal-mu"} adapter; no live C++ GPU signing donor exists.
// Formula unchanged. Function-local secret spills remain unqualified.
import attributes;
import {"mldsa_prehash" if prehashed or hash_message else "mldsa_sign"};{prehash_import}{workspace_import}
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	if (any(tid != uint3(0u, 0u, 0u))) return;
	uint parameter = push.parameter_set;
	uint key_len = {sizes('private_bytes')};
	uint sig_len = {sizes('signature_bytes')};
	uint seed_size, key_size, {"message_size, context_size" if message else "mu_size"}, sig_size, status_size;
	storage_buffers[push.seed_index].GetDimensions(seed_size);
	storage_buffers[push.private_index].GetDimensions(key_size);
{input_dimensions}
	storage_buffers[push.signature_index].GetDimensions(sig_size);
	storage_buffers[push.status_index].GetDimensions(status_size);{workspace_dimension}
	if (key_len == 0u || seed_size != 32u || key_size != key_len || {input_check}
		|| sig_size != ((sig_len + 3u) & ~3u) || status_size != 4u{workspace_check}) return;
	uint key[4896], rnd[32], {"" if message else "mu[64], "}signature[4627];
	for (uint i = 0u; i < 4896u; ++i) key[i] = 0u;
	for (uint i = 0u; i < 4627u; ++i) signature[i] = 0u;
	for (uint i = 0u; i < key_len; ++i)
		key[i] = (storage_buffers[push.private_index].Load(i & ~3u) >> (8u * (i & 3u))) & 255u;
	for (uint i = 0u; i < 32u; ++i)
		rnd[i] = (storage_buffers[push.seed_index].Load(i & ~3u) >> (8u * (i & 3u))) & 255u;
{input_load}
	uint attempts;
	bool success = {call};
	// Always overwrite the complete public output, including aligned padding.
	// Failure exposes no partial signature and never exposes rejection counts.
	for (uint i = 0u; i < sig_size; i += 4u) {{
		uint word = 0u;
		for (uint j = 0u; j < 4u; ++j)
			if (success && i + j < sig_len) word |= signature[i + j] << (8u * j);
		storage_buffers[push.signature_index].Store(i, word);
	}}
	storage_buffers[push.status_index].Store(0u, success ? 1u : 0u);
}}
"""


def generate_mldsa_keygen_shader(kernel: dict[str, Any], digest: str) -> str:
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	layouts = kernel["parameter_layouts"]
	public_sizes = " : ".join(f"parameter == {row['parameter_set']}u ? {row['public_bytes']}u" for row in layouts) + " : 0u"
	private_sizes = " : ".join(f"parameter == {row['parameter_set']}u ? {row['private_bytes']}u" for row in layouts) + " : 0u"
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Private FIPS 204 adapter; no live C++ GPU keygen donor exists.
// Existing vector-proven formulas are unchanged; secret spills remain unqualified.
import attributes;
import mldsa_keygen;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	if (any(tid != uint3(0u, 0u, 0u))) return;
	uint parameter = push.parameter_set;
	if (parameter != 44u && parameter != 65u && parameter != 87u) return;
	uint public_len = {public_sizes};
	uint private_len = {private_sizes};
	uint seed_size, private_size, public_size;
	storage_buffers[push.seed_index].GetDimensions(seed_size);
	storage_buffers[push.private_index].GetDimensions(private_size);
	storage_buffers[push.public_index].GetDimensions(public_size);
	if (seed_size != 32u || private_size != private_len || public_size != public_len) return;
	uint seed[32], public_key[2592], private_key[4896];
	for (uint i = 0u; i < 2592u; ++i) public_key[i] = 0u;
	for (uint i = 0u; i < 4896u; ++i) private_key[i] = 0u;
	for (uint i = 0u; i < 32u; ++i) {{
		uint word = storage_buffers[push.seed_index].Load(i & ~3u);
		seed[i] = (word >> (8u * (i & 3u))) & 255u;
	}}
	mldsa_keygen_internal(parameter, seed, public_key, private_key);
	for (uint i = 0u; i < public_len; i += 4u)
		storage_buffers[push.public_index].Store(i, public_key[i] | (public_key[i+1u] << 8u) | (public_key[i+2u] << 16u) | (public_key[i+3u] << 24u));
	for (uint i = 0u; i < private_len; i += 4u)
		storage_buffers[push.private_index].Store(i, private_key[i] | (private_key[i+1u] << 8u) | (private_key[i+2u] << 16u) | (private_key[i+3u] << 24u));
}}
"""


def generate_mlkem_keygen_shader(kernel: dict[str, Any], digest: str) -> str:
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Private FIPS 203 Algorithm 16 adapter. C++ oaPqc.md has no live donor.
// One workgroup owns all output words; host preflight admits k and exact extents.
import attributes;
import mlkem_kem;
struct PushConstants {{
	uint seed_index;
	uint private_index;
	uint public_index;
	uint k;
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("ml_kem_keygen")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	if (any(tid != uint3(0u, 0u, 0u)) || push.k < 2u || push.k > 4u) return;
	uint seed_size, private_size, public_size;
	storage_buffers[push.seed_index].GetDimensions(seed_size);
	storage_buffers[push.private_index].GetDimensions(private_size);
	storage_buffers[push.public_index].GetDimensions(public_size);
	uint el = 384u * push.k + 32u, dl = 768u * push.k + 96u;
	if (seed_size != 64u || private_size != dl || public_size != el) return;
	uint d[8], z[32], ek[1568], dk[3168];
	for (uint i = 0u; i < 1568u; ++i) ek[i] = 0u;
	for (uint i = 0u; i < 3168u; ++i) dk[i] = 0u;
	for (uint i = 0u; i < 8u; ++i) d[i] = storage_buffers[push.seed_index].Load(4u * i);
	for (uint i = 0u; i < 32u; ++i) {{
		uint word = storage_buffers[push.seed_index].Load(32u + (i & ~3u));
		z[i] = (word >> (8u * (i & 3u))) & 255u;
	}}
	mlkem_keygen_internal(d, z, push.k, ek, dk);
	for (uint i = 0u; i < el; i += 4u)
		storage_buffers[push.public_index].Store(i, ek[i] | (ek[i+1u] << 8u) | (ek[i+2u] << 16u) | (ek[i+3u] << 24u));
	for (uint i = 0u; i < dl; i += 4u)
		storage_buffers[push.private_index].Store(i, dk[i] | (dk[i+1u] << 8u) | (dk[i+2u] << 16u) | (dk[i+3u] << 24u));
}}
"""


def generate_mlkem_kem_shader(kernel: dict[str, Any], digest: str) -> str:
	encaps = kernel["name"] == "ml_kem_encaps"
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	# All arithmetic remains in the independently vector-proven mlkem_kem module.
	# Public status reports encapsulation-key validity only. Decapsulation has
	# no rejection status; the FIPS implicit-rejection result stays private.
	body = """
	uint seed_size, public_size, shared_size, ciphertext_size, status_size;
	storage_buffers[push.seed_index].GetDimensions(seed_size);
	storage_buffers[push.public_index].GetDimensions(public_size);
	storage_buffers[push.shared_index].GetDimensions(shared_size);
	storage_buffers[push.ciphertext_index].GetDimensions(ciphertext_size);
	storage_buffers[push.status_index].GetDimensions(status_size);
	if (seed_size != 32u || public_size != el || shared_size != 32u || ciphertext_size != cl || status_size != 4u) return;
	uint ek[1568], message[32], key[32], ciphertext[1568];
	for (uint i = 0u; i < 1568u; ++i) { ek[i] = 0u; ciphertext[i] = 0u; }
	for (uint i = 0u; i < el; ++i) {
		uint word = storage_buffers[push.public_index].Load(i & ~3u);
		ek[i] = (word >> (8u * (i & 3u))) & 255u;
	}
	bool valid = mlkem_encapsulation_key_valid(ek, push.k, el);
	storage_buffers[push.status_index].Store(0u, uint(valid));
	if (!valid) {
		for (uint i = 0u; i < 32u; i += 4u) storage_buffers[push.shared_index].Store(i, 0u);
		for (uint i = 0u; i < cl; i += 4u) storage_buffers[push.ciphertext_index].Store(i, 0u);
		return;
	}
	for (uint i = 0u; i < 32u; ++i) {
		uint word = storage_buffers[push.seed_index].Load(i & ~3u);
		message[i] = (word >> (8u * (i & 3u))) & 255u;
	}
	mlkem_encaps_internal(ek, message, push.k, key, ciphertext);
	for (uint i = 0u; i < cl; i += 4u)
		storage_buffers[push.ciphertext_index].Store(i, ciphertext[i] | (ciphertext[i+1u] << 8u) | (ciphertext[i+2u] << 16u) | (ciphertext[i+3u] << 24u));
""" if encaps else """
	uint private_size, ciphertext_size, shared_size;
	storage_buffers[push.private_index].GetDimensions(private_size);
	storage_buffers[push.ciphertext_index].GetDimensions(ciphertext_size);
	storage_buffers[push.shared_index].GetDimensions(shared_size);
	if (private_size != 768u * push.k + 96u || ciphertext_size != cl || shared_size != 32u) return;
	// Only internally generated retained keys are admitted by the private owner.
	uint dk[3168], ciphertext[1568], key[32];
	for (uint i = 0u; i < 3168u; ++i) dk[i] = 0u;
	for (uint i = 0u; i < 1568u; ++i) ciphertext[i] = 0u;
	for (uint i = 0u; i < private_size; ++i) {
		uint word = storage_buffers[push.private_index].Load(i & ~3u);
		dk[i] = (word >> (8u * (i & 3u))) & 255u;
	}
	for (uint i = 0u; i < cl; ++i) {
		uint word = storage_buffers[push.ciphertext_index].Load(i & ~3u);
		ciphertext[i] = (word >> (8u * (i & 3u))) & 255u;
	}
	mlkem_decaps_internal(dk, ciphertext, push.k, key);
"""
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Private FIPS 203 adapter; C++ oaPqc.md has no live donor.
// Serial correctness route; secret spills and physical timing remain unqualified.
import attributes;
import mlkem_kem;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	if (any(tid != uint3(0u, 0u, 0u)) || push.k < 2u || push.k > 4u) return;
	uint el = 384u * push.k + 32u;
	uint cl = 32u * (push.k == 4u ? 11u * push.k + 5u : 10u * push.k + 4u);
{body}	for (uint i = 0u; i < 32u; i += 4u)
		storage_buffers[push.shared_index].Store(i, key[i] | (key[i+1u] << 8u) | (key[i+2u] << 16u) | (key[i+3u] << 24u));
}}
"""


def generate_mldsa_verify_shader(kernel: dict[str, Any], schema_hash: str) -> str:
	parameter = kernel["parameter_set"]
	pk_bytes, sig_bytes = MLDSA_WIRE_LAYOUTS[parameter]
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	workgroup = ", ".join(str(size) for size in kernel["workgroup_size"])
	prehashed = kernel["name"].endswith("_verify_prehashed")
	hash_message = kernel["name"].endswith("_verify_hash_message")
	verification = "mldsa_verify_hash_message" if hash_message else ("mldsa_verify_prehashed" if prehashed else "mldsa_verify_external")
	algorithm_argument = ", push.algorithm" if prehashed or hash_message else ""
	mode = "original-message prehash" if hash_message else ("precomputed-digest" if prehashed else "pure")
	return f"// Generated by tool/gen/fn/generate.py; do not edit.\n// schema_sha256={schema_hash}\n" + f"""// ML-DSA-{parameter} batch external {mode} verification with a shared context.
// Experimental public-only route; physical-device qualification remains separate.
// FIPS 204 Algorithms {"5/8" if prehashed or hash_message else "3/8"}. New implementation; OA historical shader was a stub.
import attributes;
import mldsa_verification;
import shake;

struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads({workgroup})]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {{
	uint idx = dispatch_thread_id.x;
	if (idx >= push.batch_size || idx > 0xffffffffu / 4u) return;
	uint output_base = idx * 4u;
	if (!pqc_buffer_range(storage_buffers[push.results_index], output_base, 4u)) return;
	storage_buffers[push.results_index].Store(output_base, 0u);
	if (!pqc_buffer_range(storage_buffers[push.offsets_index], output_base, 4u)
		|| !pqc_buffer_range(storage_buffers[push.lengths_index], output_base, 4u)
		|| idx > 0xffffffffu / {pk_bytes}u
		|| idx > 0xffffffffu / {sig_bytes}u) return;
	uint offset = storage_buffers[push.offsets_index].Load(output_base);
	uint length = storage_buffers[push.lengths_index].Load(output_base);
	// Allocation padding is not part of the semantic message buffer.
	if (offset > push.message_bytes || length > push.message_bytes - offset
		|| push.context_length > 255u
		|| push.context_offset > push.message_bytes
		|| push.context_length > push.message_bytes - push.context_offset) return;
	bool valid = {verification}(
		{parameter}u,
		storage_buffers[push.pks_index], idx * {pk_bytes}u, {pk_bytes}u,
		storage_buffers[push.sigs_index], idx * {sig_bytes}u, {sig_bytes}u,
		storage_buffers[push.input_index], offset, length,
		storage_buffers[push.input_index], push.context_offset, push.context_length{algorithm_argument});
	storage_buffers[push.results_index].Store(output_base, valid ? 1u : 0u);
}}
"""


MATRIX_SAVED_TYPES = {
	"matrix": "Matrix", "matrix_list": "Vec<Matrix>", "usize_list": "Vec<usize>",
	"output_identity": "u64", "u64": "u64", "usize": "usize", "i32": "i32", "f32": "f32",
}


def validate_matrix_autograd_schema(schema: dict[str, Any]) -> None:
	require(isinstance(schema.get("family"), str), "Matrix autograd family must be a string")
	family = schema["family"].removeprefix("matrix_")
	require(family in {"elemwise", "blas", "reduce", "index", "rng", "view"}, "unknown Matrix autograd family")
	operations = {op["name"]: op for op in schema["operations"]}
	seen_nodes: set[str] = set()
	for operation in schema["operations"]:
		where = f"{schema['family']}.{operation['name']}.autograd"
		differentiation = operation.get("differentiation", operation.get("contract", {}).get("differentiation", "none"))
		policy = operation.get("autograd")
		if differentiation != "reverse":
			require(policy is None, f"{where} requires a differentiable operation")
			continue
		require(isinstance(policy, dict), f"{where} requires an explicit policy")
		require(policy.get("family") == family, f"{where}.family must match its numerical owner")
		if policy.get("mode") == "alias":
			require(set(policy) == {"mode", "family", "target"}, f"{where} alias has unknown fields")
			require(isinstance(policy["target"], str) and IDENTIFIER.fullmatch(policy["target"]) is not None, f"{where} alias target must be an identifier")
			target = operations.get(policy["target"])
			require(target is not None and target.get("autograd", {}).get("mode") == "generated", f"{where} alias must reference a canonical generated operation in the same family")
			continue
		require(policy.get("mode") == "generated" and set(policy) == {"mode", "family", "node", "saved_fields"}, f"{where} generated policy fields are incomplete or unknown")
		node = policy["node"]
		require(isinstance(node, str) and re.fullmatch(r"[A-Z][A-Za-z0-9]*", node) is not None and node != "Self", f"{where}.node must be PascalCase")
		validate_unique(node, seen_nodes, f"{where}.node")
		fields = policy["saved_fields"]
		require(isinstance(fields, list) and fields, f"{where}.saved_fields must be non-empty")
		names: set[str] = set()
		reserved = set("self Self super crate gradients gen _ as break const continue else enum extern false fn for if impl in let loop match mod move mut pub ref return static struct trait true type unsafe use where while async await dyn abstract become box do final macro override priv typeof unsized virtual yield try".split())
		for field in fields:
			require(isinstance(field, list) and len(field) == 2, f"{where} fields must be name/type pairs")
			name, kind = field
			require(isinstance(name, str) and re.fullmatch(r"[a-z_][a-z0-9_]*", name) is not None and name not in reserved, f"{where} field must be a non-reserved identifier")
			validate_unique(name, names, f"{where} saved field")
			require(isinstance(kind, str) and kind in MATRIX_SAVED_TYPES, f"{where} saved type is not admitted")
			require((name == "output_id") == (kind == "output_identity"), f"{where} output identity must be output_id")
			require(name != "output" or kind == "matrix", f"{where} output argument must be a Matrix")
		require("output_id" in names and any(kind in {"matrix", "matrix_list"} for _, kind in fields), f"{where} requires an output identity and saved Matrix destination")


def validate_view_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1 and schema.get("family") == "matrix_view" and schema.get("domain") == "matrix", "expected Matrix view schema")
	operations = schema.get("operations")
	require(isinstance(operations, list) and len(operations) == 1 and isinstance(operations[0], dict) and operations[0].get("name") == "reshape" and operations[0].get("metadata_only") is True, "view schema must own the existing metadata-only reshape")
	require(not ({"kernel_id", "stable_id", "source", "contract"} & set(operations[0])), "reshape metadata must not invent executable or registry identity")


def load_view_schema(path: Path) -> tuple[dict[str, Any], str]:
	return load_validated_schema(path, validate_view_schema)


def matrix_autograd_rows(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
	rows = []
	seen: set[str] = set()
	for schema in schemas:
		validate_matrix_autograd_schema(schema)
		for operation in schema["operations"]:
			if operation.get("autograd", {}).get("mode") == "generated":
				validate_unique(operation["autograd"]["node"], seen, "Matrix autograd node across schemas")
				rows.append(operation)
	return rows


def matrix_autograd_recipes(family: str, rows: list[dict[str, Any]]) -> dict[str, str]:
	path = Path(__file__).parent / "template/matrix/autograd" / f"{family}.rs.in"
	sections = re.findall(r"// @operation (\w+)\n// @fields ([^\n]+)\n(.*?)(?=// @operation|\Z)", path.read_text(), re.S)
	require(len(sections) == len({name for name, _, _ in sections}), f"duplicate Matrix autograd recipe: {family}")
	require({name for name, _, _ in sections} == {row["name"] for row in rows}, f"Matrix autograd recipe coverage differs from schema: {family}")
	by_name = {row["name"]: row for row in rows}
	for name, fields, body in sections:
		require(fields.split(",") == [field for field, _ in by_name[name]["autograd"]["saved_fields"]], f"Matrix autograd saved fields differ from recipe: {name}")
		calls = re.findall(r"matrix::(\w+)\(", body)
		require(calls == [f"{name}_backward"], f"Matrix reverse adapter must call only its numerical backward provider: {name}")
		require(not any(token in body for token in ("ComputeDispatch", "BufferBinding", "PushConstant", "sum_to_shape(", ".reshape(", "gradients.bmm")), f"Matrix reverse adapter contains numerical machinery: {name}")
	return {name: body.strip() for name, _, body in sections}


def generate_matrix_backward_providers(schema: dict[str, Any], family: str) -> str:
	"""Complete numerical adjoints, selected by their canonical operation rows."""
	rows = matrix_autograd_rows([schema])
	path = Path(__file__).parent / "template/matrix/backward" / f"{family}.rs.in"
	sections = re.findall(r"// @operation (\w+)\n// @fields ([^\n]+)\n(.*?)(?=// @operation|\Z)", path.read_text(), re.S)
	require(len(sections) == len({name for name, _, _ in sections}), f"duplicate Matrix numerical backward recipe: {family}")
	require({name for name, _, _ in sections} == {row["name"] for row in rows}, f"Matrix numerical backward coverage differs from schema: {family}")
	by_name = {row["name"]: row for row in rows}
	for name, fields, body in sections:
		require(fields.split(",") == [field for field, _ in by_name[name]["autograd"]["saved_fields"]], f"Matrix numerical backward saved fields differ from recipe: {name}")
		require(f"pub(crate) fn {name}_backward(" in body, f"Matrix numerical backward provider name differs from schema: {name}")
		require("gradients." not in body and "GradientContext" not in body, f"Matrix numerical backward provider must not access tape services: {name}")
	return "\n\n".join(body.strip() for _, _, body in sections)


def generate_matrix_attachments(rows: list[dict[str, Any]], digest: str, family: str) -> str:
	lines = [banner(digest, "//", f"matrix_{family}.json").rstrip(), "//! Complete saved-state attachments; observer lifecycle is handwritten.", "", "use super::super::{GradNodeMatrix, record};", "use crate::{Matrix, Result};", ""]
	for row in rows:
		policy = row["autograd"]
		fields = policy["saved_fields"]
		output_saved = ["output", "matrix"] in fields
		arguments = []
		values = []
		for name, kind in fields:
			if kind == "output_identity":
				if not output_saved:
					arguments.append("output: &Matrix")
				values.append("output_id: output.value_id()")
			elif kind == "matrix":
				arguments.append(f"{name}: &Matrix")
				values.append(f"{name}: {name}.clone()")
			elif kind == "matrix_list":
				arguments.append(f"{name}: &[Matrix]")
				values.append(f"{name}: {name}.to_vec()")
			else:
				arguments.append(f"{name}: {MATRIX_SAVED_TYPES[kind]}")
				values.append(name)
		lines.extend([f"pub(in crate::matrix) fn record_{row['name']}({', '.join(arguments)}) -> Result<()> {{", f"record(GradNodeMatrix::{policy['node']} {{ {', '.join(values)} }})", "}", ""])
	return "\n".join(lines)


def generate_matrix_reverse(rows: list[dict[str, Any]], digest: str, family: str) -> str:
	recipes = matrix_autograd_recipes(family, rows)
	text = "\n".join(recipes.values())
	lines = [banner(digest, "//", f"matrix_{family}.json").rstrip(), "//! Tape traversal adapters; numerical adjoints live in Matrix operation providers.", "", "use super::GradientContext;", "use crate::{Matrix, Result};"]
	if "Error::" in text:
		lines.append("use crate::Error;")
	if "matrix::" in text:
		lines.append("use crate::matrix;")
	for row in rows:
		args = []
		for name, kind in row["autograd"]["saved_fields"]:
			typ = {"matrix_list": "[Matrix]", "usize_list": "[usize]"}.get(kind, MATRIX_SAVED_TYPES[kind])
			args.append(f"{name}: &{typ}")
		args.append("gradients: &mut impl GradientContext")
		lines.extend(["", f"pub(super) fn {row['name']}({', '.join(args)}) -> Result<bool> {{", recipes[row["name"]], "}"])
	return "\n".join(lines) + "\n"


def generate_matrix_autograd_routing(rows: list[dict[str, Any]], digest: str, sources: str | None = None) -> str:
	families = sorted({row["autograd"]["family"] for row in rows})
	label = sources or ",".join(f"matrix_{family}.json" for family in families)
	lines = [banner(digest, "//", label).replace("// schema=", "// schemas=").rstrip(), "//! Schema-owned saved state and exhaustive Matrix reverse routing.", "", "use super::GradientContext;", "use crate::{Matrix, Result};", ""]
	for family in families:
		lines.extend([f'#[path = "node/matrix/{family}.gen.rs"]', f"mod {family};"])
	lines.extend(["", "pub(crate) enum GradNodeMatrix {"])
	for row in rows:
		lines.append(row["autograd"]["node"] + " {")
		lines.extend(f"{name}: {MATRIX_SAVED_TYPES[kind]}," for name, kind in row["autograd"]["saved_fields"])
		lines.append("},")
	lines.extend(["}", "", "impl GradNodeMatrix {", "pub(crate) fn output_id(&self) -> u64 {", "match self {"])
	lines.extend(("" if index == 0 else "| ") + f"Self::{row['autograd']['node']} {{ output_id, .. }}" for index, row in enumerate(rows))
	lines.extend(["=> *output_id,", "}", "}", "", "pub(crate) fn backward(&self, gradients: &mut impl GradientContext) -> Result<bool> {", "match self {"])
	for row in rows:
		policy = row["autograd"]
		names = ", ".join(name for name, _ in policy["saved_fields"])
		lines.append(f"Self::{policy['node']} {{ {names} }} => {policy['family']}::{row['name']}({names}, gradients),")
	lines.extend(["}", "}", "}"])
	return "\n".join(lines) + "\n"


def matrix_autograd_outputs(root: Path, schemas: list[tuple[dict[str, Any], str]]) -> dict[Path, str]:
	rows = matrix_autograd_rows([schema for schema, _ in schemas])
	digest = hashlib.sha256("".join(value for _, value in schemas).encode()).hexdigest()
	outputs = {}
	sources = ",".join(schema["family"] + ".json" for schema, _ in schemas)
	combined_banner = banner(digest, "//", sources).replace("// schema=", "// schemas=")
	facade = [combined_banner.rstrip(), "//! Schema-owned foundational attachment facade.", ""]
	for schema, value in schemas:
		family = schema["family"].removeprefix("matrix_")
		selected = [row for row in rows if row["autograd"]["family"] == family]
		outputs[root / f"src/rs/matrix/autograd/matrix/{family}.gen.rs"] = format_rust(generate_matrix_attachments(selected, value, family), root)
		outputs[root / f"src/rs/matrix/autograd/node/matrix/{family}.gen.rs"] = format_rust(generate_matrix_reverse(selected, value, family), root)
		facade.extend([f'#[path = "matrix/{family}.gen.rs"]', f"mod {family};", f"pub(in crate::matrix) use {family}::{{" + ", ".join("record_" + row["name"] for row in selected) + "};"])
	outputs[root / "src/rs/matrix/autograd/matrix.gen.rs"] = format_rust("\n".join(facade), root)
	outputs[root / "src/rs/matrix/autograd/node.gen.rs"] = format_rust(generate_matrix_autograd_routing(rows, digest, sources), root)
	exports = combined_banner + "pub(in crate::matrix) use matrix::{" + ", ".join("record_" + row["name"] for row in rows) + "};\n"
	outputs[root / "src/rs/matrix/autograd/exports.gen.rs"] = format_rust(exports, root)
	return outputs


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
	activation: dict[str, Any],
	activation_hash: str,
	audio: dict[str, Any],
	audio_hash: str,
	cryptography: dict[str, Any],
	cryptography_hash: str,
	cryptography_pqc: dict[str, Any],
	cryptography_pqc_hash: str,
	image: dict[str, Any],
	image_hash: str,
	vision: dict[str, Any],
	vision_hash: str,
	view: dict[str, Any] | None = None,
	view_hash: str | None = None,
) -> dict[Path, str]:
	if view is None:
		view, view_hash = load_view_schema(Path(__file__).resolve().parents[3] / DEFAULT_VIEW_SCHEMA)
	assert view_hash is not None
	outputs = {
		root / "src/rs/core/operation/operation.gen.rs": format_rust(
			generate_operation_registry(
				elementwise, elementwise_hash, blas, blas_hash, reduce, reduce_hash, rng, rng_hash, index, index_hash, ml, ml_hash, audio, audio_hash, cryptography, cryptography_hash, cryptography_pqc, cryptography_pqc_hash, image, image_hash, vision, vision_hash
			),
			root,
		),
		root / "src/rs/matrix/elemwise.gen.rs": format_rust(generate_api(elementwise, elementwise_hash), root),
		root / "src/rs/matrix/blas.gen.rs": format_rust(generate_blas_api(blas, blas_hash), root),
		root / "src/rs/matrix/reduce.gen.rs": format_rust(
			generate_reduce_api(reduce, reduce_hash), root
		),
		root / "src/rs/runtime/shader/registry.gen.rs": format_rust(
			generate_registry(elementwise, elementwise_hash, blas, blas_hash, reduce, reduce_hash, rng, rng_hash, index, index_hash, ml, ml_hash, audio, audio_hash, cryptography, cryptography_hash, cryptography_pqc, cryptography_pqc_hash, image, image_hash, vision, vision_hash), root
		),
		root / "src/rs/runtime/dnn/generated.rs": format_rust(
			generate_dnn_roles(elementwise, elementwise_hash, blas, blas_hash, reduce, reduce_hash, rng_hash, ml, ml_hash), root
		),
		root / "test/rs/matrix/test_elemwise.gen.rs": format_rust(generate_test(elementwise, elementwise_hash), root),
		root / "test/rs/matrix/test_blas.gen.rs": format_rust(generate_blas_test(blas, blas_hash), root),
		root / "test/rs/matrix/test_reduce.gen.rs": format_rust(
			generate_reduce_test(reduce, reduce_hash), root
		),
		root / "src/rs/ml/matrix/activation.gen.rs": format_rust(
			generate_ml_activation_api(activation, activation_hash), root
		),
	}
	outputs.update(matrix_autograd_outputs(root, [(elementwise, elementwise_hash), (blas, blas_hash), (reduce, reduce_hash), (rng, rng_hash), (index, index_hash), (view, view_hash)]))
	outputs[root / "src/rs/ml/matrix/swiglu.gen.rs"] = format_rust(
		generate_ml_activation_api(activation, activation_hash, "swiglu"), root
	)
	for family in sorted({op["rust_family"] for op in ml["operations"] if "rust_family" in op}):
		outputs[root / f"src/rs/ml/matrix/{family}.gen.rs"] = format_rust(
			generate_ml_matrix_family(ml, ml_hash, family), root
		)
	outputs[root / "src/rs/cryptography/hash/batch.gen.rs"] = format_rust(generate_cryptography_api(cryptography, cryptography_hash, "hash"), root)
	outputs[root / "src/rs/cryptography/pqc/verify.gen.rs"] = format_rust(generate_cryptography_api(cryptography_pqc, cryptography_pqc_hash, "verify"), root)
	outputs[root / "src/rs/vision/detection.gen.rs"] = format_rust(generate_vision_api(vision, vision_hash), root)
	for category in ("signal", "transform"):
		outputs[root / f"src/rs/audio/{category}.gen.rs"] = format_rust(generate_audio_api(audio, audio_hash, category), root)
	for category in IMAGE_CATEGORIES:
		outputs[root / f"src/rs/image/{category}.gen.rs"] = format_rust(generate_image_api(image, image_hash, category), root)
	outputs[root / "src/slang/cryptography/pqc/mldsa/mldsa_prehash_layout.slang"] = generate_mldsa_prehash_layout_shader(cryptography_pqc, cryptography_pqc_hash)
	outputs[root / "src/slang/cryptography/pqc/mldsa/mldsa_prehash_digest.slang"] = generate_mldsa_prehash_digest_shader(cryptography_pqc, cryptography_pqc_hash)
	outputs[root / "src/slang/cryptography/pqc/mldsa/mldsa_sign_workspace_layout.slang"] = generate_mldsa_sign_workspace_layout_shader(cryptography_pqc, cryptography_pqc_hash)
	outputs[root / "src/slang/cryptography/pqc/mldsa/mldsa_prehash.slang"] = generate_mldsa_prehash_shader(cryptography_pqc, cryptography_pqc_hash)
	for kernel in cryptography_pqc["kernels"]:
		outputs[root / kernel["source"]] = generate_mldsa_verify_shader(kernel, cryptography_pqc_hash)
	for kernel in cryptography_pqc.get("private_kernels", []):
		outputs[root / kernel["source"]] = (generate_mlkem_keygen_shader(kernel, cryptography_pqc_hash)
			if kernel["name"] == "ml_kem_keygen" else generate_mldsa_keygen_shader(kernel, cryptography_pqc_hash)
			if kernel["name"] == "ml_dsa_keygen" else generate_mldsa_sign_shader(kernel, cryptography_pqc_hash)
			if kernel["name"] in ("ml_dsa_sign", "ml_dsa_sign_message", "ml_dsa_sign_prehashed", "ml_dsa_sign_hash_message") else generate_mlkem_kem_shader(kernel, cryptography_pqc_hash))
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
		outputs[root / f"src/rs/ml/autograd/node/{family}.gen.rs"] = format_rust(
			generate_ml_autograd_records(ml, ml_hash, family), root
		)
	outputs[root / "src/rs/ml/autograd/node/operation.gen.rs"] = format_rust(
		generate_ml_autograd_routing(ml, ml_hash), root
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


OWNERSHIP_BANNERS = {
	"// @generated by tools/gen/fn/generate.py; DO NOT EDIT.",
	"// Generated by tools/gen/fn/generate.py; do not edit.",
	"// @generated by tool/gen/fn/generate.py; DO NOT EDIT.",
	"// Generated by tool/gen/fn/generate.py; do not edit.",
}


def owned_outputs(root: Path) -> set[Path]:
	"""Find this authority's outputs only; never infer ownership from a suffix."""
	owned = set()
	for relative in ("src/rs", "src/slang", "test/rs"):
		directory = root / relative
		if directory.is_symlink() or directory.parent.is_symlink():
			continue
		for parent, directories, files in os.walk(directory, followlinks=False):
			directories[:] = sorted(
				name for name in directories if not (Path(parent) / name).is_symlink()
			)
			for name in sorted(files):
				path = Path(parent) / name
				if path.is_symlink() or path.suffix not in {".rs", ".slang"}:
					continue
				with path.open(encoding="utf-8") as source:
					if source.readline().rstrip("\r\n") in OWNERSHIP_BANNERS:
						owned.add(path)
	return owned


def validate_output_paths(outputs: dict[Path, str], root: Path) -> None:
	"""Reject escape and symlink destinations before any publication."""
	root = root.resolve()
	for path in outputs:
		relative = path.absolute().relative_to(root)
		require(".." not in relative.parts, f"generated output has parent traversal: {path}")
		require(relative.parts[:2] in {("src", "rs"), ("src", "slang"), ("test", "rs")},
			f"generated output outside owned source trees: {path}")
		cursor = root
		for part in relative.parts:
			cursor /= part
			require(not cursor.is_symlink(), f"generated output traverses symlink: {path}")
		require(path.resolve().is_relative_to(root), f"generated output escapes root: {path}")


def publish(outputs: dict[Path, str], root: Path) -> None:
	validate_output_paths(outputs, root)
	stale = owned_outputs(root) - set(outputs)
	for path, content in sorted(outputs.items()):
		data = content.encode("utf-8")
		if path.is_file() and path.read_bytes() == data:
			continue
		path.parent.mkdir(parents=True, exist_ok=True)
		temporary_path = None
		try:
			with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as temporary:
				temporary_path = Path(temporary.name)
				temporary.write(data)
			os.replace(temporary_path, path)
		finally:
			if temporary_path is not None:
				temporary_path.unlink(missing_ok=True)
	for path in sorted(stale):
		path.unlink()


def check(outputs: dict[Path, str], root: Path) -> list[str]:
	validate_output_paths(outputs, root)
	errors = []
	for path, content in sorted(outputs.items()):
		if not path.exists():
			errors.append(f"missing generated output: {path}")
		elif path.read_bytes() != content.encode("utf-8"):
			errors.append(f"stale generated output: {path}")
	for path in sorted(owned_outputs(root) - set(outputs)):
		errors.append(f"orphaned generated output: {path}")
	return errors


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
	parser.add_argument("--schema", type=Path)
	parser.add_argument("--blas-schema", type=Path)
	parser.add_argument("--reduce-schema", type=Path)
	parser.add_argument("--rng-schema", type=Path)
	parser.add_argument("--index-schema", type=Path)
	parser.add_argument("--view-schema", type=Path)
	parser.add_argument("--ml-schema", type=Path)
	parser.add_argument("--ml-activation-schema", type=Path)
	parser.add_argument("--audio-schema", type=Path)
	parser.add_argument("--cryptography-hash-schema", type=Path)
	parser.add_argument("--cryptography-pqc-schema", type=Path)
	parser.add_argument("--image-schema", type=Path)
	parser.add_argument("--vision-schema", type=Path)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument("--check", action="store_true")
	mode.add_argument("--live", action="store_true", help="update checked-in outputs (default)")
	mode.add_argument("--output-dir", type=Path, help="write a disposable source-tree preview")
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
	view_schema_path = args.view_schema.resolve() if args.view_schema else root / DEFAULT_VIEW_SCHEMA
	ml_schema_path = args.ml_schema.resolve() if args.ml_schema else root / DEFAULT_ML_SCHEMA
	activation_schema_path = args.ml_activation_schema.resolve() if args.ml_activation_schema else root / DEFAULT_ML_ACTIVATION_SCHEMA
	audio_schema_path = args.audio_schema.resolve() if args.audio_schema else root / DEFAULT_AUDIO_SCHEMA
	cryptography_schema_path = args.cryptography_hash_schema.resolve() if args.cryptography_hash_schema else root / DEFAULT_CRYPTOGRAPHY_HASH_SCHEMA
	cryptography_pqc_schema_path = args.cryptography_pqc_schema.resolve() if args.cryptography_pqc_schema else root / DEFAULT_CRYPTOGRAPHY_PQC_SCHEMA
	image_schema_path = args.image_schema.resolve() if args.image_schema else root / DEFAULT_IMAGE_SCHEMA
	vision_schema_path = args.vision_schema.resolve() if args.vision_schema else root / DEFAULT_VISION_SCHEMA
	try:
		schema, schema_hash = load_schema(schema_path)
		blas_schema, blas_schema_hash = load_blas_schema(blas_schema_path)
		reduce_schema, reduce_schema_hash = load_reduce_schema(reduce_schema_path)
		rng_schema, rng_schema_hash = load_rng_schema(rng_schema_path)
		index_schema, index_schema_hash = load_index_schema(index_schema_path)
		view_schema, view_schema_hash = load_view_schema(view_schema_path)
		ml_schema, ml_schema_hash = load_ml_schema(ml_schema_path)
		activation_schema, activation_schema_hash = load_activation_schema(activation_schema_path)
		audio_schema, audio_schema_hash = load_audio_schema(audio_schema_path)
		cryptography_schema, cryptography_schema_hash = load_cryptography_hash_schema(cryptography_schema_path)
		cryptography_pqc_schema, cryptography_pqc_schema_hash = load_cryptography_pqc_schema(cryptography_pqc_schema_path)
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
			cryptography_pqc_schema,
			image_schema,
			vision_schema,
		)
	except (OSError, SchemaError) as error:
		print(f"operation generation failed: {error}", file=os.sys.stderr)
		return 1
	try:
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
			activation_schema,
			activation_schema_hash,
			audio_schema,
			audio_schema_hash,
			cryptography_schema,
			cryptography_schema_hash,
			cryptography_pqc_schema,
			cryptography_pqc_schema_hash,
			image_schema,
			image_schema_hash,
			vision_schema,
			vision_schema_hash,
			view_schema,
			view_schema_hash,
		)
	except (OSError, SchemaError) as error:
		print(f"operation assembly failed: {error}", file=os.sys.stderr)
		return 1
	output_root = root
	if args.output_dir:
		output_root = args.output_dir.resolve()
		if output_root == root or root.is_relative_to(output_root) or output_root.is_relative_to(root / "src") or output_root.is_relative_to(root / "test"):
			print("preview must not overlap the repository source trees", file=os.sys.stderr)
			return 1
		outputs = {output_root / path.relative_to(root): content for path, content in outputs.items()}
	try:
		if args.check:
			errors = check(outputs, output_root)
			if errors:
				print("\n".join(errors), file=os.sys.stderr)
				return 1
		else:
			publish(outputs, output_root)
	except (OSError, ValueError) as error:
		print(f"operation publication failed: {error}", file=os.sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
