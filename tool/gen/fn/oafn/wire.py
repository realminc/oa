"""Wire-ABI rendering and dispatch-generation helpers shared by emitters."""

from __future__ import annotations

import json
import re
from typing import Any

from .model import validate_dnn_role, validate_unique
from .signature import operation_signature
from .util import IDENTIFIER, require


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
		validate_dnn_role(op.get("dnn"), f"{domain} {op['name']}")
		rust = op.get("rust")
		where = f"{domain} {op['name']}.rust"
		require(isinstance(rust, dict) and set(rust) == {"category", "parameters", "returns", "doc", "attribute_sources", "bindings", "push_constants", "dispatches", "annotations"}, f"{where} fields are incomplete or unknown")
		require(isinstance(rust["category"], str) and rust["category"] in categories, f"{where} category is invalid")
		require(isinstance(rust["annotations"], list) and rust["annotations"] in ([], ["allow(clippy::too_many_arguments)"]), f"{where} annotations are invalid")
		require(rust["returns"] == returns[op["name"]], f"{where} return differs from contract")
		require(isinstance(rust["doc"], list) and rust["doc"] and all(isinstance(line, str) and "\n" not in line for line in rust["doc"]), f"{where} docs are invalid")
		signature = operation_signature(schema["domain"], op, where, allowed_types=allowed_types)
		require(signature is not None, f"{where} has no public signature")
		inputs = [
			kind[1:].lower()
			for _, kind in signature["parameters"]
			if kind in {"&Image", "&Matrix"}
		]
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


TYPE_CLASSES = {"config", "result", "info", "capabilities", "record", "mode"}
CLASS_SUFFIXES = {"config": "Config", "result": "Result", "info": "Info", "capabilities": "Capabilities"}
DISALLOWED_PUBLIC_SUFFIXES = ("Settings", "Params", "Options", "CreateInfo", "Features", "Desc")


def validate_type_class(name: str, type_class: Any, kind: str, public: bool, where: str) -> None:
	"""Enforce the admitted public type taxonomy from oaTypeRegistry.md."""
	require(isinstance(type_class, str) and type_class in TYPE_CLASSES, f"{where} class is invalid")
	required_kind = "enum" if type_class == "mode" else "struct"
	require(kind == required_kind, f"{where} class {type_class} requires kind {required_kind}")
	if not public:
		return
	require(not name.endswith(DISALLOWED_PUBLIC_SUFFIXES), f"{where} uses a non-admitted public suffix")
	suffix = CLASS_SUFFIXES.get(type_class)
	if suffix is not None:
		require(name.endswith(suffix), f"{where} class {type_class} requires the {suffix} suffix")
	elif type_class == "record":
		require(not name.endswith(tuple(CLASS_SUFFIXES.values())), f"{where} record uses a class-bound suffix")


def validate_generated_types(schema: dict[str, Any]) -> None:
	domain = schema["domain"].title()
	types = schema.get("types")
	require(isinstance(types, list) and types, f"{domain} types must be a nonempty array")
	names = set()
	operations = {op["name"]: op["rust"]["category"] for op in schema["contracts"]}
	for record in types:
		require(isinstance(record, dict) and set(record) in ({"category", "class", "name", "before", "kind", "visibility", "derive", "doc", "fields", "variants", "defaults", "default_variant"}, {"category", "class", "name", "before", "kind", "visibility", "derive", "doc", "fields", "variants", "defaults", "default_variant", "mappings"}), f"{domain} type record is incomplete or unknown")
		name = record["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name), f"{domain} type name is invalid")
		validate_unique(name, names, f"{domain} types")
		require(isinstance(record["kind"], str) and record["kind"] in {"struct", "enum"} and isinstance(record["visibility"], str) and record["visibility"] in {"pub", ""}, f"{domain} {name} kind/visibility is invalid")
		validate_type_class(name, record["class"], record["kind"], record["visibility"] == "pub", f"{domain} {name}")
		require(isinstance(record["before"], str) and record["before"] in operations and operations[record["before"]] == record["category"], f"{domain} {name} type anchor is invalid")
		require(isinstance(record["doc"], str) and "\n" not in record["doc"], f"{domain} {name} docs are invalid")
		derives = record["derive"]
		require(isinstance(derives, list) and (derives or schema["domain"] == "vision") and all(isinstance(trait, str) and trait in {"Clone", "Copy", "Debug", "Default", "PartialEq", "Eq"} for trait in derives) and len(set(derives)) == len(derives), f"{domain} {name} derives are invalid")
	validate_type_record_bodies(types, names, domain, kernels=schema["kernels"])


RUST_FIELD_TYPE = re.compile(r"[A-Za-z_(\[][A-Za-z0-9_:<>',&\[\] ();]*")


def validate_type_record_bodies(
	types: list[dict[str, Any]], names: set[str], domain: str, *,
	kernels: list[dict[str, Any]] | None = None,
	permissive_field_types: bool = False,
) -> None:
	"""Validate struct fields, enum variants, defaults, and enum mappings.

	Shared by contract-anchored and standalone type-record schemas. Permissive
	mode admits any plausible Rust type expression; anchored schemas keep their
	closed field-type sets.
	"""
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
				if permissive_field_types:
					valid_type = isinstance(field["type"], str) and RUST_FIELD_TYPE.fullmatch(field["type"]) is not None
				else:
					valid_type = isinstance(field["type"], str) and (field["type"] in names | {"u32", "f32", "bool"} | ({"i32", "Matrix"} if domain == "Vision" else set()) or re.fullmatch(r"\[(?:u32|f32|bool); [1-9][0-9]*\]", field["type"]))
				require(valid_type and isinstance(field["visibility"], str) and field["visibility"] in {"pub", "pub(crate)", "pub(super)", ""} and isinstance(field["doc"], str) and "\n" not in field["doc"], f"{domain} {name} field metadata is invalid")
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
						require(value in {"KernelId::" + kernel["kernel_id"] for kernel in kernels or []}, f"{domain} {name} mapping kernel is unknown")
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

