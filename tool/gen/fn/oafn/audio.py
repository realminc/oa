"""Audio DSP schema validation and complete-body emitters."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .model import validate_semantic_attributes, validate_unique
from .wire import generate_type, render_wire_values, validate_generated_types
from .signature import operation_signature, signature_header
from .util import IDENTIFIER, banner, require

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
			(set(contract) - {"python", "donor_op", "control_flow"})
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


def audio_templates(category: str) -> dict[str, str]:
	path = FN_ROOT / "template/audio" / f"{category}.rs.in"
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
		signature = operation_signature(
			"audio",
			op,
			where,
			allowed_types={"&Audio", "f32", "u32", "u64", "NormalizeAudioConfig", "ResampleConfig", "BiquadCoefficients", "&[BiquadCoefficients]", "StftConfig", "MelConfig", "MfccConfig"},
		)
		require(signature is not None, f"{where} has no public signature")
		require(
			sum(kind == "&Audio" for _, kind in signature["parameters"]) == len(op["input_kinds"]),
			f"{where} Audio input count differs from contract",
		)
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
		bindings[f"dispatch:{index}"] = "ComputeDispatch {\n" + f"kernel: KernelId::{stage['kernel']},\nbuffers: {stage['buffers']},\npush_constants: {stage['push_constants']},\nworkgroups: {stage['workgroups']},\nindirect: None,\n}}"
	tokens = set(re.findall(r"\{\{([^{}]+)\}\}", body))
	require(tokens == set(bindings), f"Audio {op['name']} template bindings are unresolved, missing or unused")
	return re.sub(r"\{\{([^{}]+)\}\}", lambda match: bindings[match[1]], body).rstrip()


def generate_audio_api(schema: dict[str, Any], schema_hash: str, category: str) -> str:
	text = ""
	module = (FN_ROOT / "template/audio" / f"{category}_module.rs.in").read_text(encoding="utf-8")
	require(module.count("{{items}}") == 1, f"Audio {category} module requires exactly one item insertion")
	for op in schema["contracts"]:
		if op["rust"]["category"] != category:
			continue
		rust = op["rust"]
		for record in schema["types"]:
			if record["before"] == op["name"]:
				text += "\n" + generate_type(record)
		signature = operation_signature("audio", op, f"Audio {op['name']}")
		require(signature is not None, f"Audio {op['name']} has no public signature")
		text += f"\n/// {rust['doc']}\n{signature_header(signature)} {{\n"
		text += render_audio_body(op, {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}) + "\n}\n"
	return banner(schema_hash, "//", "audio.json") + module.replace("{{items}}", text)
