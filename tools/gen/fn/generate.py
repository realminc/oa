#!/usr/bin/env python3
"""Generate the schema-owned Rust matrix elementwise surface and kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


GENERATOR_VERSION = 2
DEFAULT_SCHEMA = Path("tools/gen/fn/schema/matrix_elemwise.json")
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
	"""The operation schema is structurally invalid."""


def load_schema(path: Path) -> tuple[dict[str, Any], str]:
	raw = path.read_bytes()
	try:
		schema = json.loads(raw)
	except json.JSONDecodeError as error:
		raise SchemaError(f"invalid JSON: {error}") from error
	if not isinstance(schema, dict):
		raise SchemaError("schema root must be an object")
	validate_schema(schema)
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
	seen_entries: set[str] = set()
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
				seen_entries,
				seen_kernel_names,
				seen_sources,
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
		"entry_point": operation.get("entry_point"),
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
	seen_entries: set[str],
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
		("entry_point", seen_entries),
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


def banner(schema_hash: str, prefix: str) -> str:
	return (
		f"{prefix} @generated by tools/gen/fn/generate.py; DO NOT EDIT.\n"
		f"{prefix} schema=matrix_elemwise.json schema_version=1 generator_version={GENERATOR_VERSION}\n"
		f"{prefix} schema_sha256={schema_hash}\n"
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


def static_name(name: str, dtype: str) -> str:
	return f"MATRIX_{name.upper()}_{dtype.upper()}"


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
					f'\tbinary(left, right, &[{routes}], "matrix.{name}")',
					"}",
				]
			)
		elif operation["kind"] == "unary":
			lines.extend(
				[
					f"pub fn {name}(input: &Matrix) -> Result<Matrix> {{",
					f'\tunary(input, &[{routes}], "matrix.{name}")',
					"}",
				]
			)
		else:
			lines.extend(
				[
					f"pub fn {name}(input: &Matrix, {operation['scalar_name']}: f32) -> Result<Matrix> {{",
					f'\tunary_scalar(input, {operation["scalar_name"]}, &[{routes}], "matrix.{name}")',
					"}",
				]
			)
		lines.append("")
	return "\n".join(lines)


def generate_registry(schema: dict[str, Any], schema_hash: str) -> str:
	variants = [
		(operation, variant)
		for operation in schema["operations"]
		for variant in operation_variants(schema, operation)
	]
	workgroup = schema["workgroup_size"]
	lines = [banner(schema_hash, "//").rstrip(), "", "use super::ShaderArtifact;", ""]
	for operation, variant in variants:
		name = operation["name"]
		dtype = variant["dtype"]
		lines.extend(
			[
				f"static {static_name(name, dtype)}: ShaderArtifact = ShaderArtifact {{",
				f'\tbytes: include_bytes!(concat!(env!("OUT_DIR"), "/matrix_{name}_{dtype}.spv")),',
				f'\tentry_point: c"{variant["entry_point"]}",',
				f"\tworkgroup_size: [{workgroup[0]}, {workgroup[1]}, {workgroup[2]}],",
				f"\tpush_constant_size: {16 if operation['kind'] != 'unary' else 12},",
				"};",
				"",
			]
		)
	lines.extend(["#[repr(u16)]", "#[derive(Clone, Copy, Debug, PartialEq, Eq)]", "pub(crate) enum KernelId {"])
	for _, variant in variants:
		lines.append(f"\t{variant['kernel_id']} = {variant['stable_id']},")
	lines.extend(["}", "", "impl KernelId {", f"\tpub(crate) const ALL: [Self; {len(variants)}] = ["])
	for _, variant in variants:
		lines.append(f"\t\tSelf::{variant['kernel_id']},")
	lines.extend(["\t];", "", "\tpub(crate) const fn artifact(self) -> &'static ShaderArtifact {", "\t\tmatch self {"])
	for operation, variant in variants:
		lines.append(f"\t\t\tSelf::{variant['kernel_id']} => &{static_name(operation['name'], variant['dtype'])},")
	lines.extend(["\t\t}", "\t}", "", "\tpub(crate) const fn index(self) -> usize {", "\t\tmatch self {"])
	for index, (_, variant) in enumerate(variants):
		lines.append(f"\t\t\tSelf::{variant['kernel_id']} => {index},")
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) fn linear_workgroups(self, element_count: u32) -> [u32; 3] {",
			"\t\tlet width = self.artifact().workgroup_size[0];",
			"\t\t[element_count.div_ceil(width), 1, 1]",
			"\t}",
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
void {variant['entry_point']}(uint3 dispatch_thread_id : SV_DispatchThreadID) {{
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
		"#[test]",
		"#[ignore = \"requires a hardware Vulkan 1.3 compute device\"]",
		"fn generated_elementwise_dtype_variants_match_schema_oracles() -> oa::Result<()> {",
		"\tlet engine = oa::Engine::new()?;",
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
			"}",
			"",
			"#[test]",
			"#[ignore = \"requires a hardware Vulkan 1.3 compute device\"]",
			"fn generated_elementwise_dtype_variants_preserve_zero_extent() -> oa::Result<()> {",
			"\tlet engine = oa::Engine::new()?;",
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
	lines.extend(["\tOk(())", "}", ""])
	return "\n".join(lines)


def expected_outputs(root: Path, schema: dict[str, Any], schema_hash: str) -> dict[Path, str]:
	outputs = {
		root / "src/rs/matrix/elemwise.gen.rs": format_rust(generate_api(schema, schema_hash), root),
		root / "src/rs/runtime/shader/generated.rs": format_rust(generate_registry(schema, schema_hash), root),
		root / "tests/matrix_elemwise_generated.rs": format_rust(generate_test(schema, schema_hash), root),
	}
	for operation in schema["operations"]:
		for variant in operation_variants(schema, operation):
			outputs[root / f"src/slang/matrix/elemwise/{variant['source_stem']}.gen.slang"] = generate_shader(
				operation,
				variant,
				schema,
				schema_hash,
			)
	return outputs


def publish(outputs: dict[Path, str], shader_directory: Path) -> None:
	for path, content in outputs.items():
		path.parent.mkdir(parents=True, exist_ok=True)
		with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
			temporary.write(content)
			temporary_path = Path(temporary.name)
		os.replace(temporary_path, path)
	expected_shaders = {path.resolve() for path in outputs if path.parent == shader_directory}
	for stale in shader_directory.glob("*.gen.slang"):
		if stale.resolve() not in expected_shaders:
			stale.unlink()


def check(outputs: dict[Path, str], shader_directory: Path) -> list[str]:
	errors = []
	for path, content in outputs.items():
		if not path.exists():
			errors.append(f"missing generated output: {path}")
		elif path.read_text(encoding="utf-8") != content:
			errors.append(f"stale generated output: {path}")
	expected_shaders = {path.resolve() for path in outputs if path.parent == shader_directory}
	for stale in shader_directory.glob("*.gen.slang"):
		if stale.resolve() not in expected_shaders:
			errors.append(f"stale generated shader: {stale}")
	return errors


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
	parser.add_argument("--schema", type=Path)
	parser.add_argument("--check", action="store_true")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	root = args.root.resolve()
	schema_path = args.schema.resolve() if args.schema else root / DEFAULT_SCHEMA
	try:
		schema, schema_hash = load_schema(schema_path)
	except (OSError, SchemaError) as error:
		print(f"matrix elementwise generation failed: {error}", file=os.sys.stderr)
		return 1
	outputs = expected_outputs(root, schema, schema_hash)
	shader_directory = root / "src/slang/matrix/elemwise"
	if args.check:
		errors = check(outputs, shader_directory)
		if errors:
			print("\n".join(errors), file=os.sys.stderr)
			return 1
		return 0
	publish(outputs, shader_directory)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
