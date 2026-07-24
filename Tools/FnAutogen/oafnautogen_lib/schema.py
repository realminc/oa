from __future__ import annotations

import re
from pathlib import Path

from .config import (
	AUTO_BODY_KINDS,
	SLANG_EMIT_KINDS,
	VALID_BODIES,
	VALID_CONTRACT_DIFFERENTIATION,
	VALID_CONTRACT_ATTRIBUTE_KINDS,
	VALID_CONTRACT_DTYPE_RULES,
	VALID_CONTRACT_EFFECTS,
	VALID_CONTRACT_LOWERING,
	VALID_CONTRACT_CONTROL_FLOW,
	VALID_CONTRACT_SHAPE_RULES,
	VALID_CONTRACT_VALUE_KINDS,
	VALID_DISPATCH_WORKGROUPS,
	VALID_FORMULAS_PREFIX,
	VALID_KINDS,
	VALID_KERNEL_CATEGORIES,
	VALID_KERNEL_PREFIXES,
	VALID_OUTPUT_DTYPES,
	VALID_OUTPUT_SHAPES,
)
from .errors import fail


_SCALAR_ATTRIBUTE_KINDS = {
	"bool": "boolean",
	"OaBool": "boolean",
	"OaI8": "signed_integer",
	"OaI16": "signed_integer",
	"OaI32": "signed_integer",
	"OaI64": "signed_integer",
	"OaU8": "unsigned_integer",
	"OaU16": "unsigned_integer",
	"OaU32": "unsigned_integer",
	"OaU64": "unsigned_integer",
	"OaF32": "float",
	"OaF64": "float",
}


def semantic_attribute_specs(op: dict) -> list[dict[str, str]]:
	"""Return the normalized ordered non-value semantic inputs."""
	attributes: list[dict[str, str]] = []
	scalar = op.get("scalar_param")
	if scalar is not None:
		if not isinstance(scalar, dict):
			fail(f"{op.get('name', '<unnamed>')}: scalar_param must be a table")
		if not isinstance(scalar.get("name"), str) or not scalar["name"]:
			fail(f"{op.get('name', '<unnamed>')}: scalar_param.name is missing")
		if not isinstance(scalar.get("type"), str) or not scalar["type"]:
			fail(f"{op.get('name', '<unnamed>')}: scalar_param.type is missing")
		kind = _SCALAR_ATTRIBUTE_KINDS.get(scalar.get("type"))
		if kind is None:
			fail(
				f"{op.get('name', '<unnamed>')}: scalar_param.type "
				f"{scalar.get('type')!r} has no semantic attribute mapping"
			)
		attributes.append({"name": scalar["name"], "kind": kind})
	contract = op.get("contract", {})
	attributes.extend(contract.get("attributes", []))
	return attributes


def validate_operation_contract(ctx: str, op: dict) -> None:
	contract = op.get("contract")
	if contract is None:
		return
	if not isinstance(contract, dict):
		fail(f"{ctx}: contract must be a table")

	for field in ("input_kinds", "output_kinds"):
		kinds = contract.get(field)
		if not isinstance(kinds, list) or not kinds:
			fail(f"{ctx}: contract.{field} must be a non-empty array")
		invalid = [kind for kind in kinds if kind not in VALID_CONTRACT_VALUE_KINDS]
		if invalid:
			fail(f"{ctx}: contract.{field} contains invalid kinds {invalid!r}")
		if len(kinds) > 8:
			fail(f"{ctx}: contract.{field} supports at most 8 values")

	checks = (
		("shape_rule", VALID_CONTRACT_SHAPE_RULES),
		("dtype_rule", VALID_CONTRACT_DTYPE_RULES),
		("differentiation", VALID_CONTRACT_DIFFERENTIATION),
		("lowering", VALID_CONTRACT_LOWERING),
	)
	for field, valid in checks:
		value = contract.get(field)
		if value not in valid:
			fail(f"{ctx}: contract.{field} {value!r} not in {sorted(valid)}")

	value_validation = contract.get("value_validation")
	if op.get("kind") == "session_command":
		if value_validation != "session_command":
			fail(
				f"{ctx}: session_command requires "
				"contract.value_validation = 'session_command'"
			)
		if contract.get("shape_rule") != "explicit":
			fail(f"{ctx}: session_command requires contract.shape_rule = 'explicit'")
		if contract.get("dtype_rule") != "match_input":
			fail(
				f"{ctx}: session_command uses frozen match_input as its dtype "
				"sentinel and validates exact value dtypes manually"
			)
	elif value_validation is not None:
		fail(
			f"{ctx}: contract.value_validation is reserved for session_command"
		)

	effects = contract.get("effects")
	if not isinstance(effects, list) or not effects:
		fail(f"{ctx}: contract.effects must be a non-empty array")
	if len(set(effects)) != len(effects):
		fail(f"{ctx}: contract.effects contains duplicates")
	invalid_effects = [effect for effect in effects if effect not in VALID_CONTRACT_EFFECTS]
	if invalid_effects:
		fail(f"{ctx}: contract.effects contains invalid effects {invalid_effects!r}")

	input_count = len(contract["input_kinds"])
	output_count = len(contract["output_kinds"])
	mutated_inputs = contract.get("mutated_inputs")
	if not isinstance(mutated_inputs, list):
		fail(f"{ctx}: contract.mutated_inputs must be an array")
	if len(set(mutated_inputs)) != len(mutated_inputs):
		fail(f"{ctx}: contract.mutated_inputs contains duplicates")
	if any(not isinstance(index, int) or isinstance(index, bool)
		or index < 0 or index >= input_count for index in mutated_inputs):
		fail(f"{ctx}: contract.mutated_inputs contains an invalid input index")

	aliases = contract.get("output_alias_inputs")
	if not isinstance(aliases, list) or len(aliases) != output_count:
		fail(f"{ctx}: contract.output_alias_inputs must contain one entry per output")
	if any(not isinstance(index, int) or isinstance(index, bool)
		or index < -1 or index >= input_count for index in aliases):
		fail(f"{ctx}: contract.output_alias_inputs contains an invalid input index")

	control_flow = contract.get("control_flow")
	if control_flow not in VALID_CONTRACT_CONTROL_FLOW:
		fail(
			f"{ctx}: contract.control_flow {control_flow!r} not in "
			f"{sorted(VALID_CONTRACT_CONTROL_FLOW)}"
		)

	attributes = contract.get("attributes", [])
	if not isinstance(attributes, list):
		fail(f"{ctx}: contract.attributes must be an array")
	for index, attribute in enumerate(attributes):
		if not isinstance(attribute, dict):
			fail(f"{ctx}: contract.attributes[{index}] must be a table")
		if not isinstance(attribute.get("name"), str) or not attribute["name"]:
			fail(f"{ctx}: contract.attributes[{index}].name must be a non-empty string")
		if attribute.get("kind") not in VALID_CONTRACT_ATTRIBUTE_KINDS:
			fail(
				f"{ctx}: contract.attributes[{index}].kind "
				f"{attribute.get('kind')!r} not in "
				f"{sorted(VALID_CONTRACT_ATTRIBUTE_KINDS)}"
			)
	normalized_attributes = semantic_attribute_specs(op)
	if len(normalized_attributes) > 8:
		fail(f"{ctx}: semantic operation supports at most 8 attributes")
	attribute_names = [attribute["name"] for attribute in normalized_attributes]
	if len(set(attribute_names)) != len(attribute_names):
		fail(f"{ctx}: semantic attribute names contain duplicates")


def validate_python_binding(ctx: str, op: dict) -> None:
	python = op.get("python")
	if python is None:
		return
	if not isinstance(python, dict):
		fail(f"{ctx}: python must be a table")
	api_params = op.get("api_params", [])
	if not api_params:
		fail(f"{ctx}: python binding requires schema-owned api_params")
	if op.get("api_return", "OaMatrix") != "OaMatrix":
		fail(f"{ctx}: generated python binding currently requires api_return = 'OaMatrix'")
	args = python.get("args")
	if not isinstance(args, list) or len(args) != len(api_params):
		fail(f"{ctx}: python.args must contain one name per api parameter")
	if any(not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name) for name in args):
		fail(f"{ctx}: python.args must use PascalCase identifiers")
	if len(set(args)) != len(args):
		fail(f"{ctx}: python.args contains duplicates")
	if "name" in python and (
		not isinstance(python["name"], str)
		or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", python["name"])
	):
		fail(f"{ctx}: python.name must be a valid identifier")
	if "doc" in python and not isinstance(python["doc"], str):
		fail(f"{ctx}: python.doc must be a string")


def validate_kernel_metadata(ctx: str, op: dict) -> None:
	kernel = op.get("kernel")
	if kernel is None:
		return
	if not isinstance(kernel, dict):
		fail(f"{ctx}: kernel must be a table")
	if not op.get("kernel_forward"):
		fail(f"{ctx}: schema-owned kernel requires kernel_forward")
	for field in ("id_prefix", "id_local", "category", "origin", "source"):
		if field not in kernel:
			fail(f"{ctx}: kernel.{field} is missing")
	if kernel.get("id_prefix") not in VALID_KERNEL_PREFIXES:
		fail(
			f"{ctx}: kernel.id_prefix {kernel.get('id_prefix')!r} not in "
			f"{sorted(VALID_KERNEL_PREFIXES)}"
		)
	if kernel.get("category") not in VALID_KERNEL_CATEGORIES:
		fail(
			f"{ctx}: kernel.category {kernel.get('category')!r} not in "
			f"{sorted(VALID_KERNEL_CATEGORIES)}"
		)
	local = kernel.get("id_local")
	if not isinstance(local, int) or isinstance(local, bool) or local <= 0:
		fail(f"{ctx}: kernel.id_local must be a positive integer")
	for field in ("origin", "source"):
		value = kernel.get(field)
		if not isinstance(value, str) or not value:
			fail(f"{ctx}: kernel.{field} must be a non-empty string")
	if not re.fullmatch(r"[A-Za-z0-9_./-]+", kernel["source"]):
		fail(f"{ctx}: kernel.source contains unsupported path characters")


def load_kernel_registry(path: Path) -> set[str]:
	"""Parse KernelRegistry.h for all kernel name strings."""
	if not path.exists():
		fail(f"kernel registry not found: {path}")
	text = path.read_text()
	pat = re.compile(r'\{\s*"([^"]+)"\s*,\s*OA_COMPUTE_KERNEL_ID')
	names = set(pat.findall(text))
	if not names:
		fail(f"no kernel names parsed from {path} — registry format changed?")
	return names


def validate_schema_metadata(schema_path: Path, data: dict) -> None:
	"""Reject schema metadata that would emit into the wrong tree."""
	namespace = data.get("namespace", "")
	cpp_subdir = data.get("cpp_subdir", "")
	if namespace == "OaFnVideo" and cpp_subdir == "Video":
		fail(
			f"{schema_path.name}: cpp_subdir must be 'FnVideo' for OaFnVideo schemas "
			f"(got 'Video' — that lands autogen under Source/Private/Oa/Vision/Video/ "
			f"instead of FnVideo/<Category>/)"
		)
	if namespace == "OaFnImage" and cpp_subdir == "Video":
		fail(
			f"{schema_path.name}: cpp_subdir must be 'FnImage' for OaFnImage schemas "
			f"(got 'Video')"
		)


def validate_schema(schema_path: Path, ops: list[dict], registry: set[str]) -> None:
	seen = set()
	for op in ops:
		name = op.get("name")
		ctx = f"{schema_path.name}:{name or '<unnamed>'}"
		if not name or not name[0].isupper():
			fail(f"{ctx}: op name missing or not PascalCase")
		if name in seen:
			fail(f"{ctx}: duplicate op name")
		seen.add(name)
		validate_operation_contract(ctx, op)
		validate_python_binding(ctx, op)
		validate_kernel_metadata(ctx, op)

		kind = op.get("kind")
		if kind not in VALID_KINDS:
			fail(f"{ctx}: invalid kind {kind!r} (want one of {sorted(VALID_KINDS)})")

		body = op.get("body", "auto")
		if body not in VALID_BODIES:
			fail(f"{ctx}: body {body!r} not in {sorted(VALID_BODIES)}")
		if body in ("auto", "bias_add_broadcast") and kind not in AUTO_BODY_KINDS:
			fail(f"{ctx}: body {body!r} does not support kind {kind!r}")
		if body == "cpp_expr" and not (op.get("cpp_expr") or op.get("cpp_body")):
			fail(f"{ctx}: body 'cpp_expr' requires cpp_expr or cpp_body")
		if kind == "session_command" and body != "manual_context":
			fail(f"{ctx}: kind 'session_command' requires body = 'manual_context'")

		kf = op.get("kernel_forward")
		if body == "cpp_expr":
			kf = kf or name
			op["kernel_forward"] = kf
		# kernel_forward is optional for body types that don't dispatch a GPU
		# kernel: `manual_context` ops (session recorders like DecodeFrame /
		# EncodeFrame) and `cpu_util` ops (host-only utilities like NAL
		# Annex-B parsing). For everything else it's required.
		if not kf and body not in ("manual_context", "cpu_util"):
			fail(f"{ctx}: kernel_forward missing")
		if kf and body not in ("manual_context", "cpu_util", "cpp_expr") and kf not in registry:
			fail(f"{ctx}: kernel '{kf}' not in KernelRegistry.h")

		if op.get("forward_op") and kind not in SLANG_EMIT_KINDS:
			fail(f"{ctx}: forward_op/slang emission not supported for kind {kind!r}")
		if body == "bias_add_broadcast" and kind != "binary":
			fail(f"{ctx}: bias_add_broadcast requires kind = 'binary'")

		if kind in ("unary_scalar", "nullary_scalar"):
			sp = op.get("scalar_param")
			if not sp:
				fail(f"{ctx}: {kind} requires scalar_param table")
			for key in ("name", "type", "push_field"):
				if key not in sp:
					fail(f"{ctx}: scalar_param.{key} missing")

		for i, ep in enumerate(op.get("extra_params", [])):
			for key in ("name", "type"):
				if key not in ep:
					fail(f"{ctx}: extra_params[{i}].{key} missing")

		api_params = op.get("api_params", [])
		seen_api_params = set()
		api_param_types = {}
		seen_default = False
		for i, param in enumerate(api_params):
			if isinstance(param, str):
				decl, separator, _ = param.partition("=")
				parts = decl.strip().rsplit(None, 1)
				if len(parts) != 2 or not parts[0] or not parts[1]:
					fail(f"{ctx}: api_params[{i}] must be '<type> <name> [= default]'")
				param_type = parts[0]
				name = parts[1]
				has_default = bool(separator)
			elif isinstance(param, dict):
				for key in ("name", "type"):
					if key not in param:
						fail(f"{ctx}: api_params[{i}].{key} missing")
				param_type = param.get("type", "")
				name = f"In{param.get('name', '')}"
				has_default = "default" in param
			else:
				fail(f"{ctx}: api_params[{i}] must be a string or table")
				continue
			if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
				fail(f"{ctx}: api_params[{i}] has invalid C++ parameter name {name!r}")
			if name in seen_api_params:
				fail(f"{ctx}: duplicate api parameter {name}")
			seen_api_params.add(name)
			api_param_types[name] = param_type
			if has_default:
				seen_default = True
			elif seen_default:
				fail(f"{ctx}: non-default api parameter {name} follows a default")
		if op.get("generate_forwarder") and not api_params:
			fail(f"{ctx}: generate_forwarder requires api_params")
		if op.get("generate_engine_overload") and not api_params:
			fail(f"{ctx}: generate_engine_overload requires api_params")

		wg = op.get("dispatch_workgroups", "elemwise")
		if not isinstance(wg, int) and wg not in VALID_DISPATCH_WORKGROUPS:
			fail(f"{ctx}: unsupported dispatch_workgroups {wg!r}")

		out = op.get("output", {})
		shape = out.get("shape", "match_input")
		dtype = out.get("dtype", "match_input")
		if shape not in VALID_OUTPUT_SHAPES:
			fail(f"{ctx}: output.shape {shape!r} not in {sorted(VALID_OUTPUT_SHAPES)}")
		if dtype not in VALID_OUTPUT_DTYPES:
			fail(f"{ctx}: output.dtype {dtype!r} not in {sorted(VALID_OUTPUT_DTYPES)}")

		ag = op.get("autograd", {})
		formula = ag.get("formula", "none")
		if not any(formula.startswith(p) or formula == p for p in VALID_FORMULAS_PREFIX):
			fail(f"{ctx}: autograd.formula {formula!r} invalid")
		attach = ag.get("attach")
		if attach is not None:
			if attach not in ("standard", "broadcast_binary"):
				fail(f"{ctx}: autograd.attach {attach!r} must be 'standard' or 'broadcast_binary'")
			inputs = ag.get("inputs", [])
			if not ag.get("grad_class") or not inputs:
				fail(f"{ctx}: autograd.attach requires grad_class and non-empty inputs")
			input_ranks = ag.get("input_ranks")
			if input_ranks is not None and (
				not isinstance(input_ranks, list)
				or len(input_ranks) != len(inputs)
				or any(not isinstance(rank, int) or rank < 0 for rank in input_ranks)
			):
				fail(f"{ctx}: autograd.input_ranks must contain one non-negative integer per input")
			if attach == "broadcast_binary":
				if len(inputs) != 2:
					fail(f"{ctx}: broadcast_binary attachment requires exactly two inputs")
				if ag.get("broadcast_op") not in ("Add", "Sub", "Mul", "Div"):
					fail(f"{ctx}: broadcast_binary attachment requires broadcast_op Add/Sub/Mul/Div")
			state = ag.get("state", [])
			if not isinstance(state, list):
				fail(f"{ctx}: autograd.state must be an array")
			seen_state_members = set()
			seen_state_sources = set()
			for i, item in enumerate(state):
				if not isinstance(item, dict):
					fail(f"{ctx}: autograd.state[{i}] must be a table")
				for key in ("member", "source", "type"):
					if not isinstance(item.get(key), str) or not item[key]:
						fail(f"{ctx}: autograd.state[{i}].{key} must be a non-empty string")
				member = item.get("member", "")
				source = item.get("source", "")
				state_type = item.get("type", "")
				if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", member):
					fail(f"{ctx}: autograd.state[{i}].member is not a C++ identifier")
				if source not in api_param_types:
					fail(f"{ctx}: autograd.state[{i}].source must name an api_params entry")
				if source in inputs:
					fail(f"{ctx}: autograd.state[{i}].source duplicates a matrix input")
				if api_param_types.get(source) != state_type:
					fail(f"{ctx}: autograd.state[{i}].type must match api_params source type")
				if member in seen_state_members or source in seen_state_sources:
					fail(f"{ctx}: autograd.state members and sources must be unique")
				seen_state_members.add(member)
				seen_state_sources.add(source)


def category_from_schema(schema_path: Path, data: dict) -> tuple[str, str]:
	"""Returns (category, file_category) for the schema's output files."""
	if cat := data.get("category"):
		category = cat
	else:
		stem = schema_path.stem
		for domain in ["Core", "Ml", "Vision", "Audio", "Ui", "Crypto"]:
			if stem.startswith(domain):
				stem = stem[len(domain):]
				break
		for prefix in ["FnMatrix", "FnImage", "FnAudio"]:
			if stem.startswith(prefix):
				category = stem[len(prefix):]
				break
		else:
			category = stem[2:] if stem.startswith("Fn") else stem

	file_category = data.get("file_category", category)
	return category, file_category
