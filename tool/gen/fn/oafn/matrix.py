"""Matrix operation schemas: validation and complete emitters."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .ml import validate_ml_contract
from .model import DTYPES, KINDS, operation_variants, validate_blas_contract, validate_blas_test, validate_contract, validate_dnn_role, validate_physical_write, validate_semantic_attributes, validate_unique, validate_variant
from .signature import operation_signature, signature_header
from .util import IDENTIFIER, banner, format_rust, require, rust_const_name, rust_float, rust_value
from .wire import validate_type_class

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
		lowering_only = operation.get("lowering_only", False)
		require(isinstance(lowering_only, bool), f"{where}.lowering_only must be boolean")
		if lowering_only:
			require(
				operation["name"] == "fused_chain",
				f"{where} must be the admitted fused elementwise chain lowering",
			)
			for forbidden in (
				"kind", "doc", "expression", "signature", "test", "shape_rule",
				"scalar_name", "differentiation", "dnn", "donor_op", "autograd",
				"python", "additional_dtype_variants", "additional_lowering_variants",
			):
				require(forbidden not in operation, f"{where}.{forbidden} is invalid for lowering-only work")
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
				isinstance(source, str)
				and source.startswith("src/slang/matrix/elemwise/")
				and source.endswith(".slang")
				and not source.endswith(".gen.slang"),
				f"{where}.source must be a handwritten matrix/elemwise Slang path",
			)
			validate_unique(source, seen_sources, f"{where}.source")
			require(operation.get("variant") == "generic", f"{where}.variant must be generic")
			validate_physical_write(operation.get("physical_write"), where, required=True)
			fields = operation.get("push_fields")
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
				require(scalar_type == "uint32", f"{where}.push fields must be uint32")
			require(len(fields) * 4 <= 128, f"{where}.push constants exceed Vulkan minimum guarantee")
			continue
		kind = operation.get("kind")
		require(kind in KINDS, f"{where}.kind must be one of {sorted(KINDS)}")
		require("signature" not in operation, f"{where}.signature is derived from kind metadata")
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
					("bias_add", "bias_add_broadcast", "f32"),
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
		require("signature" not in operation, f"{where}.signature is derived from kind metadata")
		validate_dnn_role(operation.get("dnn"), where)
		require(operation.get("variant") == "tiled", f"{where}.variant must be tiled")
		require(
			isinstance(operation.get("doc"), str) and operation["doc"].endswith("."),
			f"{where}.doc must end with a period",
		)
		require(operation.get("kernel_name") == operation["name"], f"{where}.kernel_name must match name")
		validate_blas_contract(operation.get("contract"), where)
		validate_blas_test(operation.get("test"), where)
	candidates = schema.get("candidate_kernels", [])
	require(isinstance(candidates, list), "BLAS candidate_kernels must be an array")
	operation_names = {operation["name"] for operation in operations}
	for index, candidate in enumerate(candidates):
		where = f"BLAS candidate_kernels[{index}]"
		require(isinstance(candidate, dict), f"{where} must be an object")
		require(
			set(candidate)
			== {
				"name",
				"kind",
				"variant",
				"stable_id",
				"kernel_id",
				"kernel_name",
				"source_stem",
				"dispatch_tile_size",
				"semantic_operation",
			},
			f"{where} must declare exactly the candidate kernel fields",
		)
		for key, seen in (
			("name", seen_names),
			("kernel_id", seen_kernels),
			("source_stem", seen_sources),
		):
			value = candidate.get(key)
			require(
				isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None,
				f"{where}.{key} must be an identifier",
			)
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = candidate.get("stable_id")
		require(
			isinstance(stable_id, int) and 0 < stable_id <= 65535,
			f"{where}.stable_id must be a non-zero u16",
		)
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		require(candidate["kind"] == "mat_mul_nt", f"{where}.kind must be mat_mul_nt")
		require(
			candidate["variant"] in {"naive", "gemv", "bias", "bias_relu", "bias_gelu", "bias_silu"},
			f"{where}.variant is not an admitted BLAS candidate",
		)
		require(
			candidate["kernel_name"] == candidate["name"],
			f"{where}.kernel_name must match name",
		)
		require(
			candidate["dispatch_tile_size"]
			== (
				[1, 1, 1]
				if candidate["variant"] in {"naive", "gemv"}
				else schema["output_tile_size"]
			),
			f"{where}.dispatch_tile_size must match its kernel family tile",
		)
		require(
			candidate["semantic_operation"] in operation_names,
			f"{where}.semantic_operation must name a BLAS operation",
		)


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
		isinstance(operations, list) and len(operations) == 13,
		"Reduce operations must contain the admitted Softmax, LogSoftmax, Sum, Mean, Max, Argmax, and categorical-count families",
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
			"mean",
			"max",
			"max_backward",
			"argmax",
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
		composite = operation.get("composite", False)
		require(isinstance(composite, bool), f"{where}.composite must be a flag")
		if composite:
			for forbidden in ("kernel_id", "source", "variant", "physical_write", "push_fields"):
				require(forbidden not in operation, f"{where}.{forbidden} is invalid for a composite lowering")
		else:
			validate_physical_write(operation.get("physical_write"), where, required=True)
		stable_id = operation.get("stable_id")
		require(
			isinstance(stable_id, int) and 0 < stable_id <= 65535,
			f"{where}.stable_id must be a non-zero u16",
		)
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		if not composite:
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
		require(not (composite and lowering_only), f"{where} cannot be both composite and lowering-only")
		if lowering_only:
			require(
				operation["name"] == "sum_axis"
				and operation.get("semantic_operation") == "sum"
				and operation["semantic_operation"] in operation_names,
				f"{where} must be the private Sum axis lowering",
			)
			for forbidden in ("differentiation", "semantic_attributes", "contract", "test", "signature"):
				require(forbidden not in operation, f"{where}.{forbidden} is invalid for lowering-only work")
		else:
			require(
				"semantic_operation" not in operation,
				f"{where}.semantic_operation is valid only for lowering-only work",
			)
			differentiation = operation.get("differentiation")
			expected_differentiation = (
				"reverse"
				if operation["name"] in {"softmax", "log_softmax", "sum", "mean", "max"}
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
				or operation["name"] in {"max", "max_backward", "argmax"}
				else [["dim", "signed_integer"]]
			)
			require(
				operation["semantic_attributes"] == expected_attributes,
				f"{where} semantic attributes are invalid; reductions require a signed dim attribute and scalar-free backward/count operations require none",
			)
			validate_ml_contract(operation.get("contract"), differentiation, where)
			expected_inputs = {
				"softmax": ["matrix"],
				"softmax_backward": ["matrix", "matrix"],
				"log_softmax": ["matrix"],
				"log_softmax_backward": ["matrix", "matrix"],
				"sum": ["matrix"],
				"sum_backward": ["matrix", "matrix"],
				"mean": ["matrix"],
				"max": ["matrix"],
				"max_backward": ["matrix", "matrix", "matrix"],
				"argmax": ["matrix"],
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
			signature = operation_signature(
				"matrix", operation, where, allowed_types={"&Matrix", "i32"}
			)
			require(signature is not None, f"{where}.signature is required for an emitted operation")
			require(
				len([kind for _, kind in signature["parameters"] if kind == "&Matrix"])
				== len(expected_inputs),
				f"{where}.signature matrix parameters differ from contract inputs",
			)
			require(
				{name for name, _ in operation["semantic_attributes"]}
				<= {name for name, kind in signature["parameters"] if kind != "&Matrix"},
				f"{where} semantic attributes are not declared as signature parameters",
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
				"max": 3,
				"max_backward": 5,
				"argmax": 3,
				"categorical_accuracy_count": 6,
				"masked_categorical_accuracy_count": 7,
			}.get(operation["name"])
		require(
			lowering_only or composite or expected_field_count is not None,
			f"{where}.push_fields has no admitted width",
		)
		if not composite and fields is not None:
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
			"causal_mask",
			"causal_mask_backward",
			"greater_equal",
			"split",
			"compact_rows",
			"compact_rows_backward",
			"scatter_rows",
			"scatter_rows_backward_source",
			"scatter_rows_indirect",
			"compact_rows_backward_indirect",
			"scatter_rows_backward_source_indirect",
		],
		"Index operations must preserve the admitted donor family order",
	)
	names = {operation.get("name") for operation in operations if isinstance(operation, dict)}
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	seen_sources: set[str] = set()
	for index, operation in enumerate(operations):
		where = f"Index operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		composite = operation.get("composite", False)
		require(isinstance(composite, bool), f"{where}.composite must be a flag")
		if composite:
			for forbidden in ("kernel_id", "source", "variant", "physical_write", "push_fields"):
				require(forbidden not in operation, f"{where}.{forbidden} is invalid for a composite lowering")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernels)):
			value = operation.get(key)
			if composite and key == "kernel_id":
				continue
			require(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None, f"{where}.{key} must be an identifier")
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = operation.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id must be a non-zero u16")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		if not composite:
			source = operation.get("source")
			require(isinstance(source, str) and source.startswith("src/slang/matrix/index/") and source.endswith(".slang"), f"{where}.source must be owned by matrix/index")
			validate_unique(source, seen_sources, f"{where}.source")
			require(operation.get("variant") in {"generic", "indirect"}, f"{where}.variant is unsupported")
		require(operation.get("dtype", schema["dtype"]) in {"f32", "u32"}, f"{where}.dtype is unsupported")
		validate_semantic_attributes(operation.get("semantic_attributes", []), where)
		differentiation = operation.get("differentiation", "none")
		require(differentiation in {"none", "reverse"}, f"{where}.differentiation is invalid")
		semantic_operation = operation.get("semantic_operation")
		if semantic_operation is None:
			validate_ml_contract(operation.get("contract"), differentiation, where)
		else:
			require(semantic_operation in names, f"{where}.semantic_operation must name an Index operation")
			require("contract" not in operation, f"{where} implementation variant cannot own another contract")
		validate_physical_write(operation.get("physical_write"), where, required=False)
		fields = operation.get("push_fields", [])
		require(composite or fields, f"{where}.push_fields must be non-empty")
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
	types = schema.get("types")
	require(isinstance(types, list) and types, "Index types must be a nonempty array")
	type_names: set[str] = set()
	for record in types:
		where = "Index types"
		require(isinstance(record, dict) and set(record) == {"class", "fields", "kind", "name"}, "Index type record is incomplete or unknown")
		name = record["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where} name must be an identifier")
		validate_unique(name, type_names, where)
		require(record["kind"] == "struct", f"{where} {name} kind must be struct")
		fields = record["fields"]
		require(isinstance(fields, list) and fields and all(isinstance(field, dict) and set(field) == {"name", "type"} for field in fields), f"{where} {name} fields are invalid")
		validate_type_class(name, record["class"], record["kind"], public=True, where=f"{where} {name}")


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
	path = FN_ROOT / "template" / "matrix" / f"{template}.rs.in"
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


def generate_api(schema: dict[str, Any], schema_hash: str) -> str:
	lines = []
	for operation in schema["operations"]:
		if operation.get("lowering_only", False):
			continue
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
		signature = operation_signature("matrix", operation, f"Matrix {name}")
		require(signature is not None, f"Matrix {name} has no public signature")
		if operation["kind"] == "binary":
			body = matrix_body("binary", name, autograd=operation.get("autograd"), routes=f"&[{routes}]", broadcast_routes=rust_broadcast_routes(operation))
		elif operation["kind"] == "unary":
			body = matrix_body("unary", name, autograd=operation.get("autograd"), routes=f"&[{routes}]")
		else:
			body = matrix_body("unary_scalar", name, autograd=operation.get("autograd"), routes=f"&[{routes}]", scalar=operation["scalar_name"])
		lines.extend([f"{signature_header(signature)} {{", body, "}"])
		lines.append("")
	module = (FN_ROOT / "template/matrix/elemwise_module.rs.in").read_text(encoding="utf-8")
	require(
		module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		"Matrix elementwise module must contain only one items placeholder",
	)
	return banner(schema_hash, "//") + module.replace("{{items}}", "\n".join(lines) + "\n" + generate_matrix_backward_providers(schema, "elemwise"))


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
		if operation.get("lowering_only", False):
			continue
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
						f"\tlet {label}_left = oa::Matrix::new_from_slice(&engine, [ELEMENT_COUNT], &repeated(&[{left}]))?;",
						f"\tlet {label}_right = oa::Matrix::new_from_slice(&engine, [ELEMENT_COUNT], &repeated(&[{right}]))?;",
						f"\tlet {label}_actual: Vec<{rust_type}> = oa::matrix::{name}(&{label}_left, &{label}_right)?.read()?;",
					]
				)
			elif operation["kind"] == "unary":
				samples[(name, dtype)] = f"{label}_input"
				values = ", ".join(rust_value(dtype, value) for value in test["input"])
				lines.extend(
					[
						f"\tlet {label}_input = oa::Matrix::new_from_slice(&engine, [ELEMENT_COUNT], &repeated(&[{values}]))?;",
						f"\tlet {label}_actual: Vec<{rust_type}> = oa::matrix::{name}(&{label}_input)?.read()?;",
					]
				)
			else:
				samples[(name, dtype)] = f"{label}_input"
				values = ", ".join(rust_value(dtype, value) for value in test["input"])
				lines.extend(
					[
						f"\tlet {label}_input = oa::Matrix::new_from_slice(&engine, [ELEMENT_COUNT], &repeated(&[{values}]))?;",
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
				if not operation.get("lowering_only", False)
				and operation["kind"] == "binary"
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
		lines.append(f"\tlet empty_{dtype} = oa::Matrix::new_from_slice::<{rust_type}>(&engine, [0], &[])?;")
	for operation in schema["operations"]:
		if operation.get("lowering_only", False):
			continue
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


BLAS_EPILOGUE_EXPRESSIONS = {
	"bias": "value",
	"bias_relu": "max(0.0f, value)",
	"bias_gelu": (
		"0.5f * value * (1.0f + tanh(0.7978845608f * (value + 0.044715f * value * value * value)))"
	),
	"bias_silu": "value * (1.0f / (1.0f + exp(-value)))",
}


def generate_blas_shader(
	operation: dict[str, Any], schema: dict[str, Any], schema_hash: str, *, epilogue: str = "none"
) -> str:
	workgroup = schema["workgroup_size"]
	output_tile = schema["output_tile_size"]
	k_tile = schema["k_tile_size"]
	bias_index = "\tuint bias_index;\n" if epilogue != "none" else ""
	epilogue_expression = BLAS_EPILOGUE_EXPRESSIONS.get(epilogue)
	require(epilogue == "none" or epilogue_expression is not None, f"BLAS epilogue {epilogue} is unsupported")
	if epilogue_expression is None:
		store_value = "accumulators[row * TN + column]"
		store_prefix = ""
	else:
		store_value = epilogue_expression
		store_prefix = (
			"\t\t\t\tfloat value = accumulators[row * TN + column]\n"
			"\t\t\t\t\t+ load_f32(storage_buffers[push.bias_index], output_column);\n"
		)
	semantic_name = operation.get("semantic_operation", operation["name"])
	return f"""{banner(schema_hash, '//', 'matrix_blas.json').rstrip()}
// Semantic operation: matrix.{semantic_name}; stable kernel ID: {operation['stable_id']}.
// FP32 specialization of OA's established 64x64x16 tiled GEMM arithmetic.

import storage;
import attributes;

struct PushConstants {{
	uint left_index;
	uint right_index;
{bias_index}	uint output_index;
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
{store_prefix}				store_f32(
					storage_buffers[push.output_index],
					output_row * push.n + output_column,
					{store_value}
				);
			}}
		}}
	}}
}}
"""


def generate_blas_candidate_shader(
	candidate: dict[str, Any], schema: dict[str, Any], schema_hash: str
) -> str:
	if candidate["variant"] == "gemv":
		return generate_blas_gemv_shader(candidate, schema, schema_hash)
	if candidate["variant"] != "naive":
		return generate_blas_shader(
			candidate, schema, schema_hash, epilogue=candidate["variant"]
		)
	workgroup = schema["workgroup_size"]
	return f"""{banner(schema_hash, '//', 'matrix_blas.json').rstrip()}
// Semantic operation: matrix.{candidate['semantic_operation']}; stable kernel ID: {candidate['stable_id']}.
// FP32 scalar fallback: one invocation per output element.

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

[kernel_name("{candidate['kernel_name']}")]
[domain("matrix")]
[variant("{candidate['variant']}")]
[dtype("{schema['dtype']}")]
[status("experimental")]
[shader("compute")]
[numthreads({workgroup[0]}, {workgroup[1]}, {workgroup[2]})]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {{
	uint element = dispatch_thread_id.x;
	if (element >= push.m * push.n) {{
		return;
	}}
	uint row = element / push.n;
	uint column = element % push.n;
	float accumulator = 0.0f;
	for (uint inner = 0; inner < push.k; ++inner) {{
		accumulator += load_f32(storage_buffers[push.left_index], row * push.k + inner)
			* load_f32(storage_buffers[push.right_index], column * push.k + inner);
	}}
	store_f32(storage_buffers[push.output_index], row * push.n + column, accumulator);
}}
"""


def generate_blas_gemv_shader(
	candidate: dict[str, Any], schema: dict[str, Any], schema_hash: str
) -> str:
	workgroup = schema["workgroup_size"]
	lanes = workgroup[0]
	return f"""{banner(schema_hash, '//', 'matrix_blas.json').rstrip()}
// Semantic operation: matrix.{candidate['semantic_operation']}; stable kernel ID: {candidate['stable_id']}.
// Donor: oacpp/source/cpp/lib/oa/core/shader/compute/blas/gemm/gemmSmallM.slang.
// Adaptation: compile-time specialization OUTPUTS=1, REDUCTION=WORKGROUP=256,
// EPILOGUE=0; mechanical storage/ABI naming and schema metadata adaptation.
// One workgroup exclusively writes one output; all lanes join every barrier.

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

groupshared float partials[{lanes}];

[kernel_name("{candidate['kernel_name']}")]
[domain("matrix")]
[variant("{candidate['variant']}")]
[dtype("{schema['dtype']}")]
[status("experimental")]
[shader("compute")]
[numthreads({workgroup[0]}, {workgroup[1]}, {workgroup[2]})]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {{
	uint row = group_id.x;
	uint column = group_id.y;
	uint lane = group_thread_id.x;
	float accumulator = 0.0f;
	if (row < push.m && column < push.n) {{
		for (uint inner = lane; inner < push.k; inner += {lanes}) {{
			accumulator += load_f32(storage_buffers[push.left_index], row * push.k + inner)
				* load_f32(storage_buffers[push.right_index], column * push.k + inner);
		}}
	}}
	partials[lane] = accumulator;
	GroupMemoryBarrierWithGroupSync();
	[[unroll]] for (uint stride = {lanes // 2}; stride > 0; stride >>= 1) {{
		if (lane < stride) {{
			partials[lane] += partials[lane + stride];
		}}
		GroupMemoryBarrierWithGroupSync();
	}}
	if (lane == 0 && row < push.m && column < push.n) {{
		store_f32(storage_buffers[push.output_index], row * push.n + column, partials[0]);
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
		let left = oa::Matrix::new_from_f32(&engine, [m, k], &left_values)?;
		let right = oa::Matrix::new_from_f32(&engine, [n, k], &right_values)?;
		let output = oa::matrix::{operation['name']}(&left, &right)?;
		assert_eq!(output.shape(), [m, n]);
		assert_eq!(output.dtype(), oa::DType::F32);
		assert_close(&output.read_f32()?, &expected, {tolerance});
	}}
	Ok(())
}});

test_vk!(generated_mat_mul_nt_rejects_invalid_contracts, engine, {{
	let other_engine = oa::Engine::new()?;
	let rank_one = oa::Matrix::new_from_f32(&engine, [6], &[0.0; 6])?;
	let left = oa::Matrix::new_from_f32(&engine, [2, 3], &[0.0; 6])?;
	let wrong_k = oa::Matrix::new_from_f32(&engine, [2, 4], &[0.0; 8])?;
	let integer = oa::Matrix::new_from_slice(&engine, [2, 3], &[0_i32; 6])?;
	let foreign = oa::Matrix::new_from_f32(&other_engine, [2, 3], &[0.0; 6])?;

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
		mean,
		maximum,
		max_backward,
		argmax,
		accuracy_count,
		masked_accuracy_count,
	) = schema["operations"]
	signatures = {}
	for operation in schema["operations"]:
		if operation.get("lowering_only"):
			continue
		signature = operation_signature("matrix", operation, f"Reduce {operation['name']}")
		require(signature is not None, f"Reduce {operation['name']} has no public signature")
		signatures[operation["name"]] = signature_header(signature)
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
{signatures['softmax']} {{
{matrix_body('axis_normalization', softmax['name'], kernel='KernelId::' + softmax['kernel_id'], node='softmax')}
}}

{signatures['softmax_backward']} {{
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
{signatures['log_softmax']} {{
{matrix_body('axis_normalization', log_softmax['name'], kernel='KernelId::' + log_softmax['kernel_id'], node='log_softmax')}
}}

{signatures['log_softmax_backward']} {{
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
{signatures['sum']} {{
{matrix_body('sum', sum_operation['name'])}
}}

{signatures['sum_backward']} {{
{matrix_body('sum_backward', sum_backward['name'])}
}}

/// Arithmetic mean of all values when `dim` is `-1`, or one selected axis.
///
/// The semantic operation lowers to one Sum stage followed by one Scale stage,
/// matching the donor composite. A full reduction returns shape `[1]`; an axis
/// reduction preserves rank and replaces the selected extent with one.
///
/// # Errors
///
/// Returns an error when the input is empty or not FP32, `dim` is invalid, a
/// size exceeds the admitted shader ABI, allocation fails, or recording fails.
{signatures['mean']} {{
{matrix_body('mean', mean['name'])}
}}

/// Full FP32 maximum reduction into a `[1]` result.
///
/// `dim` is retained for donor signature parity and must be `-1`; axis maxima
/// are not admitted. The operation records asynchronously.
///
/// # Errors
///
/// Returns an error when the input is empty or not FP32, `dim` is not `-1`,
/// the element count exceeds the shader ABI, allocation fails, or recording
/// fails.
{signatures['max']} {{
{matrix_body('max', maximum['name'])}
}}

{signatures['max_backward']} {{
{matrix_body('max_backward', max_backward['name'])}
}}

/// Index of the maximum value in an FP32 matrix, returned as `i64`.
///
/// `dim` is retained for donor signature parity and must be `-1`. The
/// reduction result is read back synchronously, which is an explicit
/// host-observation boundary: it submits pending work and waits.
///
/// # Errors
///
/// Returns an error when the input is empty or not FP32, `dim` is not `-1`,
/// the element count exceeds the shader ABI, allocation fails, or recording,
/// submission, completion, or readback fails.
{signatures['argmax']} {{
{matrix_body('argmax', argmax['name'])}
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
{signatures['categorical_accuracy_count']} {{
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
{signatures['masked_categorical_accuracy_count']} {{
{matrix_body('accuracy', masked_accuracy_count['name'])}
}}
"""
	module = (FN_ROOT / "template/matrix/reduce_module.rs.in").read_text(encoding="utf-8")
	require(
		module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		"Matrix reduce module must contain only one items placeholder",
	)
	items += "\n" + generate_matrix_backward_providers(schema, "reduce")
	return banner(schema_hash, "//", "matrix_reduce.json") + module.replace("{{items}}", items)


def generate_blas_api(schema: dict[str, Any], schema_hash: str) -> str:
	operation = schema["operations"][0]
	require(
		operation.get("kernel_id") == "MatrixMatMulNtTiledF32",
		"Matrix mat_mul_nt must declare the tiled FP32 kernel as its shader contract",
	)
	contract_const = rust_const_name(operation["name"])
	signature = operation_signature("matrix", operation, "Matrix mat_mul_nt")
	require(signature is not None, "Matrix mat_mul_nt has no public signature")
	header = signature_header(signature)
	body_template = (FN_ROOT / "template/matrix/mat_mul_nt.rs.in").read_text(encoding="utf-8")
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
{header} {{
\tlet contract = crate::core::operation::matrix::{contract_const};
{body_template.rstrip()}
}}
"""
	module = (FN_ROOT / "template/matrix/blas_module.rs.in").read_text(encoding="utf-8")
	require(
		module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		"Matrix BLAS module must contain only one items placeholder",
	)
	return banner(schema_hash, "//", "matrix_blas.json") + module.replace("{{items}}", items + "\n" + generate_matrix_backward_providers(schema, "blas"))


def generate_reduce_test(schema: dict[str, Any], schema_hash: str) -> str:
	softmax = schema["operations"][0]
	log_softmax = schema["operations"][2]
	sum_operation = schema["operations"][4]
	mean = schema["operations"][7]
	maximum = schema["operations"][8]
	argmax = schema["operations"][10]
	accuracy_count = schema["operations"][11]
	masked_accuracy_count = schema["operations"][12]
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

fn rejected<T>(result: oa::Result<T>, message: &str) -> oa::Error {{
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
	let input = oa::Matrix::new_from_f32(&engine, shape, &values)?;
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
	let input = oa::Matrix::new_from_f32(&engine, shape, &values)?;
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
	let empty = oa::Matrix::new_from_f32(&engine, [0], &[])?;
	let integer = oa::Matrix::new_from_slice(&engine, [2], &[1_i32, 2])?;
	let input = oa::Matrix::new_from_f32(&engine, [2, 2], &[1.0, 2.0, 3.0, 4.0])?;
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
	let input = oa::Matrix::new_from_f32(&engine, shape, &values)?;
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

test_vk!(generated_mean_matches_full_and_axis_oracles, engine, {{
	let shape = {shape!r};
	let values = (0..shape.iter().product::<usize>())
		.map(|index| ((index * 11 % 23) as f32 - 11.0) * 0.13)
		.collect::<Vec<_>>();
	let input = oa::Matrix::new_from_f32(&engine, shape, &values)?;
	let axis = oa::matrix::{mean['name']}(&input, {dim})?;
	assert_eq!(axis.shape(), [shape[0], 1, shape[2]]);
	let width = shape[{dim}] as f32;
	for (actual, expected) in axis.read_f32()?.iter().zip(sum_reference(&values, &shape, {dim})) {{
		assert!((actual - expected / width).abs() <= {tolerance});
	}}
	let full = oa::matrix::{mean['name']}(&input, -1)?;
	assert_eq!(full.shape(), [1]);
	assert!(
		(full.read_f32()?[0] - values.iter().sum::<f32>() / values.len() as f32).abs()
			<= {tolerance}
	);
	Ok(())
}});

test_vk!(generated_max_matches_full_oracle_and_rejects_axis, engine, {{
	let shape = {shape!r};
	let values = (0..shape.iter().product::<usize>())
		.map(|index| ((index * 13 % 29) as f32 - 14.0) * 0.11)
		.collect::<Vec<_>>();
	let input = oa::Matrix::new_from_f32(&engine, shape, &values)?;
	let output = oa::matrix::{maximum['name']}(&input, -1)?;
	assert_eq!(output.shape(), [1]);
	let expected = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
	assert_eq!(output.read_f32()?, vec![expected]);
	assert_eq!(
		rejected(oa::matrix::{maximum['name']}(&input, 0), "axis maxima were accepted").kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
}});

test_vk!(generated_argmax_returns_first_maximum_index, engine, {{
	let input = oa::Matrix::new_from_f32(
		&engine,
		[2, 4],
		&[1.0, -3.0, 7.5, 2.0, -8.0, 7.5, 0.25, 6.0],
	)?;
	assert_eq!(oa::matrix::{argmax['name']}(&input, -1)?, 2);
	assert_eq!(
		rejected(oa::matrix::{argmax['name']}(&input, 1), "axis argmax was accepted").kind(),
		oa::ErrorKind::InvalidArgument
	);
	Ok(())
}});

test_vk!(generated_mean_capture_owns_one_semantic_operation_over_two_dispatches, engine, {{
	let input = oa::Matrix::new_from_f32(
		&engine,
		[2, 4],
		&[1.0, -3.0, 7.5, 2.0, -8.0, 7.5, 0.25, 6.0],
	)?;
	let (plan, output) = engine.capture(|| oa::matrix::{mean['name']}(&input, -1))?;
	assert_eq!(plan.semantic_graph().operations().len(), 1);
	assert_eq!(plan.semantic_graph().operations()[0].name(), "oa::FnMatrix::mean");
	let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("mean"))
		.expect("mean execution report must be valid JSON");
	let nodes = report["nodes"]
		.as_array()
		.expect("execution nodes must be an array");
	assert_eq!(nodes.len(), 2);
	assert_eq!(nodes[0]["kernel"], "matrix.sum.f32");
	assert_eq!(nodes[1]["kernel"], "matrix.scale.f32");
	engine.submit(&plan)?.wait()?;
	assert_eq!(output.shape(), [1]);
	Ok(())
}});

test_vk!(generated_categorical_accuracy_counts_match_exact_oracles, engine, {{
	let logits = oa::Matrix::new_from_f32(
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
		oa::Matrix::new_from_slice(&engine, [4], &[0_u8, 1, 0, 0])?,
		oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 1, 0, 0])?,
		oa::Matrix::new_from_slice(&engine, [4], &[0_i32, 1, -1, 0])?,
	] {{
		let count = oa::matrix::{accuracy_count['name']}(&logits, &labels)?;
		assert_eq!(count.shape(), [1]);
		assert_eq!(count.dtype(), oa::DType::U32);
		assert_eq!(count.read::<u32>()?, vec![3]);
	}}
	let labels = oa::Matrix::new_from_slice(&engine, [4], &[0_u32, 1, 2, 0])?;
	let mask = oa::Matrix::new_from_f32(&engine, [4], &[1.0, 0.0, -2.0, 1.0])?;
	assert_eq!(
		oa::matrix::{masked_accuracy_count['name']}(&logits, &labels, &mask)?
			.read::<u32>()?,
		vec![3]
	);
	Ok(())
}});

test_vk!(generated_categorical_accuracy_count_rejects_invalid_contracts, engine, {{
	let other_engine = oa::Engine::new()?;
	let logits = oa::Matrix::new_from_f32(&engine, [2, 3], &[1.0, 2.0, 3.0, 3.0, 2.0, 1.0])?;
	let rank_one = oa::Matrix::new_from_f32(&engine, [6], &[0.0; 6])?;
	let labels = oa::Matrix::new_from_slice(&engine, [2], &[2_u32, 0])?;
	let wrong_shape = oa::Matrix::new_from_slice(&engine, [1], &[2_u32])?;
	let float_labels = oa::Matrix::new_from_f32(&engine, [2], &[2.0, 0.0])?;
	let foreign_labels = oa::Matrix::new_from_slice(&other_engine, [2], &[2_u32, 0])?;
	let mask = oa::Matrix::new_from_f32(&engine, [2], &[1.0, 1.0])?;
	let integer_mask = oa::Matrix::new_from_slice(&engine, [2], &[1_u32, 1])?;

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
	let logits = oa::Matrix::new_from_f32(&engine, [2, 3], &[1.0, 2.0, 3.0, 3.0, 2.0, 1.0])?;
	let labels = oa::Matrix::new_from_slice(&engine, [2], &[2_u32, 0])?;
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


MATRIX_SAVED_TYPES = {
	"matrix": "Matrix", "optional_matrix": "Option<Matrix>", "matrix_list": "Vec<Matrix>", "usize_list": "Vec<usize>",
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
	path = FN_ROOT / "template/matrix/autograd" / f"{family}.rs.in"
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
	path = FN_ROOT / "template/matrix/backward" / f"{family}.rs.in"
	sections = re.findall(r"// @operation (\w+)\n// @fields ([^\n]+)\n(.*?)(?=// @operation|\Z)", path.read_text(), re.S)
	require(len(sections) == len({name for name, _, _ in sections}), f"duplicate Matrix numerical backward recipe: {family}")
	schema_backwards = {
		operation["name"][:-len("_backward")]
		for operation in schema["operations"]
		if isinstance(operation.get("name"), str) and operation["name"].endswith("_backward")
	}
	require({name for name, _, _ in sections} == {row["name"] for row in rows} - schema_backwards, f"Matrix numerical backward coverage differs from schema: {family}")
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
			elif kind == "optional_matrix":
				arguments.append(f"{name}: Option<&Matrix>")
				values.append(f"{name}: {name}.cloned()")
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
