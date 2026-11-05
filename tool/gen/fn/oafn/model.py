"""Shared schema-model validation, expansion and record helpers."""

from __future__ import annotations

from typing import Any

from .util import IDENTIFIER, require

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


PHYSICAL_WRITE_DOMAINS = {
	"axis_slices",
	"blocks",
	"output_elements",
	"payload_words",
	"rows",
	"scalar",
}


PHYSICAL_WRITE_PARTITIONS = {
	"exclusive_per_invocation",
	"exclusive_per_workgroup",
	"shared_atomic_contributors",
}


PHYSICAL_WRITE_EXTENTS = {
	"axis_slice",
	"one_element",
	"one_scalar",
	"one_word",
	"row_width",
	"tile_16x16",
	"tile_32x32",
	"up_to_two_elements",
	"up_to_four_elements",
}


PHYSICAL_WRITE_COLLISIONS = {"exclusive", "atomic_u32"}


PHYSICAL_WRITE_TAILS = {"bounds_checked"}


WORKSPACE_PARTITIONS = {"exclusive_per_workgroup", "none"}


def validate_physical_write(value: Any, where: str, *, required: bool) -> None:
	if value is None:
		require(not required, f"{where}.physical_write is required")
		return
	require(isinstance(value, dict), f"{where}.physical_write must be an object")
	require(
		set(value) <= {"writes", "uniform_stores", "workspace"} and {"workspace"} <= set(value),
		f"{where}.physical_write fields are incomplete or unknown",
	)
	writes = value.get("writes") or []
	uniform_stores = value.get("uniform_stores")
	require(
		writes or uniform_stores is not None,
		f"{where}.physical_write must name writes or a uniform store proof",
	)
	if uniform_stores is not None:
		label = f"{where}.physical_write.uniform_stores"
		require(isinstance(uniform_stores, dict), f"{label} must be an object")
		require(
			set(uniform_stores) == {"domain", "partition", "extent", "collision", "tail"},
			f"{label} fields are incomplete or unknown",
		)
		require(
			uniform_stores.get("domain") in PHYSICAL_WRITE_DOMAINS,
			f"{label}.domain is unsupported",
		)
		require(
			uniform_stores.get("partition") in PHYSICAL_WRITE_PARTITIONS,
			f"{label}.partition is unsupported",
		)
		require(
			uniform_stores.get("extent") in PHYSICAL_WRITE_EXTENTS,
			f"{label}.extent is unsupported",
		)
		require(
			uniform_stores.get("collision") in PHYSICAL_WRITE_COLLISIONS,
			f"{label}.collision is unsupported",
		)
		require(uniform_stores.get("tail") in PHYSICAL_WRITE_TAILS, f"{label}.tail is unsupported")
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
		"lowering": "gemm",
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
		"blocks": "Blocks",
		"output_elements": "OutputElements",
		"payload_words": "PayloadWords",
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
		"one_word": "OneWord",
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
		for write in value.get("writes") or []
	)
	uniform = value.get("uniform_stores")
	if uniform is None:
		uniform_store = "None"
	else:
		uniform_store = (
			"Some(UniformStoreProof { "
			f"domain: LogicalWriteDomain::{domains[uniform['domain']]}, "
			f"partition: WritePartition::{partitions[uniform['partition']]}, "
			f"extent: WriteExtent::{extents[uniform['extent']]}, "
			f"collision: CollisionPolicy::{collisions[uniform['collision']]}, "
			f"tail: TailPolicy::{tails[uniform['tail']]} "
			"})"
		)
	return (
		"Some(PhysicalWriteContract { "
		f"writes: &[{writes}], "
		f"uniform_store: {uniform_store}, "
		f"workspace: WorkspacePartition::{workspaces[value['workspace']]} "
		"})"
	)
