"""Python binding emitters: native adapters, stubs and fragments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .model import validate_unique
from .util import IDENTIFIER, _strip_generated_banner, banner, require

def merge_python_fragments(outputs: dict[Path, str], root: Path, schema_hashes: list[str], schema_names: str) -> None:
	"""Fold every generated `oa/_gen/` fragment into `native.pyi` and `crypto.py`."""
	import re
	binding = root / "src/py/oa/_gen"
	digest = hashlib.sha256("".join(schema_hashes).encode()).hexdigest()
	stub_fragments = {path: outputs.pop(path) for path in list(outputs) if path.parent == binding and path.suffix == ".pyi"}
	facade_fragments = {path: outputs.pop(path) for path in list(outputs) if path.parent == binding and path.name.startswith("crypto_") and path.suffix == ".py"}

	def merge(prefix: str, fragments: dict[Path, str]) -> str:
		bodies = [_strip_generated_banner(fragments[path]) for path in fragments]
		import_re = re.compile(r"from \.\._native import(?: \(([^)]*)\)| ([^\n]+))\n?")
		names: set[str] = set()
		rest = []
		for body in bodies:
			for group in import_re.findall(body):
				names.update(part.strip() for chunk in group for part in chunk.split(",") if part.strip())
			rest.append(import_re.sub("", body).strip("\n"))
		imports = f"from .._native import (\n{''.join(f'\t{name},\n' for name in sorted(names))})\n" if names else ""
		all_re = re.compile(r"__all__ = \[([^\]]*)\]\n?")
		exported: set[str] = set()
		merged_rest = []
		for body in rest:
			for group in all_re.findall(body):
				exported.update(name.strip().strip('"').strip("'") for name in group.split(",") if name.strip())
			merged_rest.append(all_re.sub("", body).strip("\n"))
		out = banner(digest, prefix, schema_names).replace(f"{prefix} schema=", f"{prefix} schemas=") + imports
		out += "\n".join(part for part in merged_rest if part) + "\n"
		if exported:
			out += "\n__all__ = [\n" + "".join(f'\t"{name}",\n' for name in sorted(exported)) + "]\n"
		return out

	if stub_fragments:
		outputs[binding / "native.pyi"] = merge("#", stub_fragments)
	if facade_fragments:
		outputs[binding / "crypto.py"] = merge("#", facade_fragments)


PYTHON_VALUE_TYPES = {"Matrix": "PythonMatrix", "Audio": "PythonAudio", "Image": "PythonImage"}


PYTHON_RESULT_TYPES = PYTHON_VALUE_TYPES | {
	"NmsResult": "PythonNmsResult",
	"DetectionMetricsResult": "PythonDetectionMetricsResult",
	"SegmentationMetricsResult": "PythonSegmentationMetricsResult",
	"TopKResult": "(PythonMatrix, PythonMatrix)",
	"MoeExpertPlan": "PythonMoeExpertPlan",
	"CompactRowsResult": "PythonCompactRowsResult",
	"Vec<Matrix>": "Vec<PythonMatrix>",
	"i64": "i64",
	"()": "()",
}


# Owning py-binding module for each Python result class.
PYTHON_RESULT_MODULES = {
	"Matrix": "matrix",
	"Audio": "audio",
	"Image": "image",
	"NmsResult": "vision",
	"DetectionMetricsResult": "vision",
	"SegmentationMetricsResult": "vision",
	"MoeExpertPlan": "matrix",
	"CompactRowsResult": "matrix",
}


# Native struct parameters rebuilt field-by-field from their Python result
# wrappers; those wrappers own Matrix fields rather than an `.inner` owner.
PYTHON_STRUCT_PARAMETERS = {
	"CompactRowsResult": "&oa::matrix::CompactRowsResult {{ values: {name}.values.inner.clone(), row_map: {name}.row_map.inner.clone(), count: {name}.count.inner.clone(), dispatch_args: {name}.dispatch_args.inner.clone() }}",
}


PYTHON_PRIMITIVE_TYPES = {
	"f32", "f64", "u8", "u16", "u32", "u64", "u128", "usize",
	"i8", "i16", "i32", "i64", "i128", "isize", "bool", "[u32; 4]",
}


# Token enums adapt through one handwritten str converter per domain enum.
PYTHON_ENUM_TYPES = {
	"InterpolationMode": "super::interpolation",
	"BorderMode": "super::border",
	"ImageFormat": "super::format",
	"crate::ImageFormat": "super::format",
	"NormalizeAudioMode": "super::normalize_mode",
	"StftWindow": "super::stft_window",
}


PYTHON_STUB_TYPES = {
	"f32": "float", "f64": "float", "bool": "bool", "[u32; 4]": "Sequence[int]",
	"[f32; 3]": "Sequence[float]", "Vec<[f32; 5]>": "Sequence[Sequence[float]]",
	"Vec<usize>": "Sequence[int]", "Vec<i64>": "Sequence[int]", "Vec<Matrix>": "Sequence[Matrix]",
	"&[Matrix]": "Sequence[Matrix]", "&[i64]": "Sequence[int]",
	"u8": "int", "u16": "int", "u32": "int", "u64": "int", "u128": "int",
	"usize": "int", "i8": "int", "i16": "int", "i32": "int", "i64": "int", "i128": "int", "isize": "int",
}


# Native return kinds that spell differently in stubs.
PYTHON_STUB_RETURNS = {
	"()": "None",
	"TopKResult": "tuple[Matrix, Matrix]",
	"CompactRowsResult": "CompactRowsResult",
	"Vec<Matrix>": "list[Matrix]",
	"i64": "int",
}


def python_operation(
	signature: dict[str, Any],
	struct_results: dict[str, list[str]] | None = None,
) -> tuple[list[tuple[str, str, str]], str, list[str] | None] | None:
	"""Map a signature to Python adapter parameters, result class, and struct
	result fields (None for single-value results).

	Returns one `(name, pyo3_type, call_argument)` triple per parameter.
	Unsupported types (configuration records, engine handles, borrowed slices)
	keep their handwritten adapters until the schema owns a Python-adaptation
	record for them.
	"""
	struct_results = struct_results or {}
	adapter = signature.get("python", {})
	require(isinstance(adapter, dict) and set(adapter) <= {"parameters", "arguments", "defaults"}, f"Python {signature['name']} adaptation fields are invalid")
	parameters = adapter.get("parameters", signature["parameters"])
	require(isinstance(parameters, list) and parameters, f"Python {signature['name']} parameters are invalid")
	names: set[str] = set()
	mapped: list[tuple[str, str, str]] = []
	for parameter in parameters:
		require(isinstance(parameter, list) and len(parameter) == 2, f"Python {signature['name']} parameter is invalid")
		name, kind = parameter
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"Python {signature['name']} parameter name is invalid")
		validate_unique(name, names, f"Python {signature['name']} parameter")
		require(isinstance(kind, str), f"Python {signature['name']} parameter type is invalid")
		if kind in PYTHON_PRIMITIVE_TYPES or kind in {"[f32; 3]", "Vec<[f32; 5]>", "Vec<usize>", "Vec<i64>", "Vec<f32>"}:
			mapped.append((name, kind, name))
		elif kind == "&[i64]":
			mapped.append((name, "Vec<i64>", f"&{name}"))
		elif kind in {"&[Matrix]", "Vec<Matrix>"}:
			collect = f"{name}.iter().map(|m| m.inner.clone()).collect::<Vec<_>>()"
			mapped.append((name, "Vec<PyRef<'_, PythonMatrix>>", f"&{collect}" if kind.startswith("&") else collect))
		elif kind in PYTHON_ENUM_TYPES:
			mapped.append((name, "&str", f"{PYTHON_ENUM_TYPES[kind]}({name})?"))
		elif kind.startswith("&") and kind[1:] in PYTHON_STRUCT_PARAMETERS:
			mapped.append((
				name,
				f"&{PYTHON_RESULT_TYPES[kind[1:]]}",
				PYTHON_STRUCT_PARAMETERS[kind[1:]].format(name=name),
			))
		elif kind.startswith("&") and kind[1:] in PYTHON_VALUE_TYPES:
			mapped.append((name, f"&{PYTHON_VALUE_TYPES[kind[1:]]}", f"&{name}.inner"))
		else:
			require(not adapter, f"Python {signature['name']} parameter type is unsupported: {kind}")
			return None
	defaults = adapter.get("defaults", {})
	require(isinstance(defaults, dict) and set(defaults) <= names, f"Python {signature['name']} defaults reference unknown parameters")
	optional = False
	for name, kind in parameters:
		if name not in defaults:
			require(not optional, f"Python {signature['name']} required parameter follows a default")
			continue
		optional = True
		value = defaults[name]
		if kind in PYTHON_ENUM_TYPES:
			valid = isinstance(value, str)
		elif kind == "bool":
			valid = type(value) is bool
		elif kind in {"f32", "f64"}:
			valid = type(value) in {int, float} and math.isfinite(value)
		elif PYTHON_STUB_TYPES.get(kind) == "int":
			bits = int(kind[1:]) if kind[1:].isdigit() else 32
			lower = 0 if kind.startswith("u") else -(1 << (bits - 1))
			upper = 1 << (bits if kind.startswith("u") else bits - 1)
			valid = type(value) is int and lower <= value < upper
		else:
			valid = False
		require(valid, f"Python {signature['name']} default type differs from {kind}")
	receiver = signature["receiver"]
	if receiver is not None:
		require(parameters[0] == signature["parameters"][0] and receiver not in defaults, f"Python {signature['name']} receiver differs from the native receiver")
	# A reviewed conversion replaces a native argument, using $name tokens for
	# the checked language conversions above. Never maintain a second call order.
	arguments = adapter.get("arguments", {})
	native_names = {name for name, _ in signature["parameters"]}
	require(isinstance(arguments, dict) and set(arguments) <= native_names and receiver not in arguments, f"Python {signature['name']} argument conversion is invalid")
	used: set[str] = set()
	for name, kind in signature["parameters"]:
		if name in arguments:
			expression = arguments[name]
			require(isinstance(expression, str) and expression.strip() and ";" not in expression and "\n" not in expression, f"Python {signature['name']} conversion expression is invalid")
			tokens = set(re.findall(r"\$(\w+)", expression))
			require(tokens and tokens <= names, f"Python {signature['name']} conversion references unknown parameters")
			used |= tokens
		else:
			require([name, kind] in parameters, f"Python {signature['name']} native argument has no conversion: {name}")
			used.add(name)
	require(used == names, f"Python {signature['name']} has unused adaptation parameters")
	returns = signature["returns"]
	if returns in struct_results:
		return mapped, PYTHON_RESULT_TYPES[returns], struct_results[returns]
	if returns in PYTHON_RESULT_TYPES:
		return mapped, PYTHON_RESULT_TYPES[returns], None
	return None


def python_default(signature: dict[str, Any], name: str, *, stub: bool = False) -> str | None:
	defaults = signature.get("python", {}).get("defaults", {})
	if name not in defaults:
		return None
	value = defaults[name]
	return repr(value) if stub and type(value) is bool else json.dumps(value)


def python_stub_parameter(signature: dict[str, Any], name: str, kind: str) -> str:
	stub_kind = (
		"str" if kind in PYTHON_ENUM_TYPES
		else kind.lstrip("&")
		if kind.lstrip("&") in PYTHON_VALUE_TYPES or kind.lstrip("&") in PYTHON_STRUCT_PARAMETERS
		else PYTHON_STUB_TYPES[kind]
	)
	default = python_default(signature, name, stub=True)
	return f"{name}: {stub_kind}" + (f" = {default}" if default is not None else "")


def python_struct_results(schemas: list[dict[str, Any]]) -> dict[str, list[str]]:
	"""Field names of schema struct types whose fields are all Matrix-typed."""
	results = {}
	for schema in schemas:
		for record in schema.get("types", []):
			if record.get("kind") == "struct" and record.get("fields") and all(
				field["type"] == "Matrix" for field in record["fields"]
			):
				results[record["name"]] = [field["name"] for field in record["fields"]]
	return results


def generate_python_domain(
	domain: str,
	signatures: list[dict[str, Any]],
	schema_hash: str,
	schema_name: str,
	struct_results: dict[str, list[str]] | None = None,
) -> tuple[str, str, str]:
	"""Emit the pyfunction adapters, value methods, and stub for one domain.

	Returns (rust_source, native_stub, class_method_stubs). Operations that
	python_operation cannot map keep their handwritten bindings.
	"""
	emitted = []
	for signature in signatures:
		if signature["visibility"] != "pub":
			continue
		mapped = python_operation(signature, struct_results)
		require(mapped is not None, f"Python {domain}::{signature['name']} requires a schema-owned adaptation")
		emitted.append((signature, *mapped))
	names: set[str] = set()
	types: set[str] = set()
	functions = []
	methods = []
	stub_defs = []
	method_stubs = []
	for signature, parameters, returns, result_fields in emitted:
		validate_unique(signature["name"], names, f"{domain.title()} python operation")
		returns_import = "Matrix" if signature["returns"] == "Vec<Matrix>" else signature["returns"]
		if returns_import in PYTHON_RESULT_MODULES:
			types.add(returns_import)
		types.update(
			kind.lstrip("&") for _, kind in signature["parameters"]
			if kind.lstrip("&") in PYTHON_VALUE_TYPES or kind.lstrip("&") in PYTHON_STRUCT_PARAMETERS
		)
		target = f"oa::{domain}::{signature['name']}"
		public_parameters = signature.get("python", {}).get("parameters", signature["parameters"])
		defaults = {param: python_default(signature, param) for param, _ in public_parameters}
		needs_signature = any(value is not None for value in defaults.values())
		python_defaults = signature.get("python", {}).get("defaults", {})
		text_signature_of = lambda names: "(" + ", ".join(
			name + (f"={repr(python_defaults[name])}" if name in python_defaults else "")
			for name in names
		) + ")"
		attr = ""
		if needs_signature:
			attr_args = ", ".join(
				name + (f"={defaults[name]}" if defaults[name] is not None else "")
				for name, _, _ in parameters
			)
			text_signature = text_signature_of(name for name, _, _ in parameters)
			attr = f"#[pyo3(signature = ({attr_args}), text_signature = \"{text_signature}\")]\n"
		stub_name = f"{domain}_{signature['name']}"
		stub_returns = PYTHON_STUB_RETURNS.get(
			signature["returns"],
			PYTHON_STUB_TYPES.get(signature["returns"], signature["returns"]),
		)
		stub_defs.append(
			f"def {stub_name}({', '.join(python_stub_parameter(signature, name, kind) for name, kind in public_parameters)}) -> {stub_returns}: ..."
		)
		conversions = {name: argument for name, _, argument in parameters}
		call_arguments = [
			re.sub(r"\$(\w+)", lambda match: conversions[match[1]], expression)
			if (expression := signature.get("python", {}).get("arguments", {}).get(name)) is not None
			else conversions[name]
			for name, _ in signature["parameters"]
		]
		call_arguments = [re.sub(r"\b(\w+): \1\b", r"\1", argument) for argument in call_arguments]
		allow = "#[allow(clippy::too_many_arguments)]\n" if len(parameters) > 7 else ""
		docs = "".join("///" + (" " + line if line else "") + "\n" for line in signature["doc"] or [])
		if returns == "()":
			result_map = ".map(|_| ())"
		elif returns in PYTHON_PRIMITIVE_TYPES:
			result_map = ""
		elif signature["returns"] == "Vec<Matrix>":
			result_map = ".map(|r| r.into_iter().map(PythonMatrix::wrap).collect::<Vec<_>>())"
		elif result_fields is None:
			result_map = f".map({returns}::wrap)"
		elif returns.startswith("("):
			result_map = ".map(|r| (" + ", ".join(
				f"PythonMatrix::wrap(r.{field})" for field in result_fields
			) + "))"
		else:
			result_map = f".map(|r| {returns} {{ " + ", ".join(
				f"{field}: PythonMatrix::wrap(r.{field})" for field in result_fields
			) + " })"
		functions.append(
			f"{docs}#[pyfunction]\n{attr}{allow}"
			f"fn {stub_name}({', '.join(f'{name}: {kind}' for name, kind, _ in parameters)}) -> PyResult<{returns}> {{\n"
			f"\t{target}({', '.join(call_arguments)})\n"
			f"\t\t{result_map}\n\t\t.map_err(python_error)\n}}"
		)
		receiver = signature["receiver"]
		if receiver is None or signature["visibility"] != "pub":
			continue
		rest = [(name, kind, argument) for name, kind, argument in parameters if name != receiver]
		require(rest != parameters, f"{domain.title()} {signature['name']} receiver is not a parameter")
		method_attr = ""
		if needs_signature:
			method_attr_args = ", ".join(
				name + (f"={defaults[name]}" if defaults[name] is not None else "")
				for name, _, _ in rest
			)
			method_text_signature = text_signature_of(["self"] + [name for name, _, _ in rest])
			method_attr = f"#[pyo3(signature = ({method_attr_args}), text_signature = \"{method_text_signature}\")]\n"
		method_returns = "Self" if signature["returns"] == domain.title() else returns
		method_arguments = ["&self.inner", *call_arguments[1:]]
		methods.append(
			f"{docs}{method_attr}{allow}"
			f"pub(super) fn {signature['name']}(&self{', ' + ', '.join(f'{name}: {kind}' for name, kind, _ in rest) if rest else ''}) -> PyResult<{method_returns}> {{\n"
			f"\t{target}({', '.join(method_arguments)})\n"
			f"\t\t{result_map}\n\t\t.map_err(python_error)\n}}"
		)
		stub_rest = ", ".join(
			python_stub_parameter(signature, name, kind)
			for name, kind in public_parameters
			if name != receiver
		)
		method_stubs.append(
			f"\tdef {signature['name']}(self{', ' + stub_rest if stub_rest else ''}) -> {stub_returns}: ..."
		)
	imports = sorted(types)
	use = "use pyo3::prelude::*;\nuse crate::{" + ", ".join(
		["error::python_error"] + [f"{PYTHON_RESULT_MODULES[kind]}::{PYTHON_RESULT_TYPES[kind]}" for kind in imports]
	) + "};"
	rust = banner(schema_hash, "//", schema_name) + use + "\n\n" + "\n\n".join(functions)
	if methods:
		rust += f"\n\n#[pymethods]\nimpl {PYTHON_VALUE_TYPES[domain.title()]} {{\n" + "\n\n".join(methods) + "\n}\n"
	rust += "\n\npub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {\n"
	rust += "".join(
		f"\tmodule.add_function(wrap_pyfunction!({domain}_{signature['name']}, module)?)?;\n"
		for signature, _, _, _ in emitted
	)
	rust += "\tOk(())\n}\n"
	stub = banner(schema_hash, "#", schema_name) + f"from .._native import {', '.join(imports)}\n\n" + "\n".join(stub_defs) + "\n"
	return rust, stub, "\n".join(method_stubs) + "\n"


def generate_native_stub(template: str, methods: dict[str, str], generated: str) -> str:
	"""Publish independent lifecycle declarations with schema-derived methods.

	The template owns only signatures outside fn generation. Class methods and
	imports are substituted on every generation, including stale-name removal.
	"""
	for domain, declarations in methods.items():
		token = "{{" + domain + "_methods}}"
		require(template.count(token) == 1, f"Python native stub requires one {domain} methods insertion")
		template = template.replace(token, declarations.rstrip())
	names = sorted(re.findall(r"^(?:def|class) (\w+)[(:]", generated, re.M))
	imports = "from ._gen.native import (\n" + "".join(f"\t{name} as {name},\n" for name in names) + ")"
	require(template.count("{{operation_imports}}") == 1, "Python native stub requires one operation import insertion")
	template = template.replace("{{operation_imports}}", imports)
	require("{{" not in template, "Python native stub has unresolved insertions")
	return banner(hashlib.sha256((template + generated).encode()).hexdigest(), "#", "native.pyi.in and operation schemas") + template
