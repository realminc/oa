from __future__ import annotations

import re
from pathlib import Path

from tools.gen.fn.config import (
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
 VALID_DNN_EPILOGUES,
 VALID_DNN_ROLES,
 VALID_FORMULAS_PREFIX,
 VALID_KINDS,
 VALID_KERNEL_CATEGORIES,
 VALID_KERNEL_PREFIXES,
 VALID_OPERATION_SURFACES,
 VALID_OUTPUT_DTYPES,
 VALID_OUTPUT_SHAPES,
 VALID_SESSION_COMPLETIONS,
 VALID_SESSION_EFFECTS,
)
from tools.gen.fn.errors import fail


_SCALAR_ATTRIBUTE_KINDS = {
 "bool": "boolean",
 "oa::I8": "signed_integer",
 "oa::I16": "signed_integer",
 "oa::I32": "signed_integer",
 "oa::I64": "signed_integer",
 "oa::U8": "unsigned_integer",
 "oa::U16": "unsigned_integer",
 "oa::U32": "unsigned_integer",
 "oa::U64": "unsigned_integer",
 "oa::F32": "float",
 "oa::F64": "float",
}


def semanticAttributeSpecs(op: dict) -> list[dict[str, str]]:
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


def validateOperationContract(ctx: str, op: dict) -> None:
	contract = op.get("contract")
	if contract is None:
		return
	if not isinstance(contract, dict):
		fail(f"{ctx}: contract must be a table")

	variadic: dict[str, dict | None] = {}
	for field in ("variadic_inputs", "variadic_outputs"):
		descriptor = contract.get(field)
		variadic[field] = descriptor
		if descriptor is None:
			continue
		if not isinstance(descriptor, dict):
			fail(f"{ctx}: contract.{field} must be a table")
		if set(descriptor) != {"kind", "minimum"}:
			fail(
			 f"{ctx}: contract.{field} must contain exactly kind and minimum"
			)
		if descriptor["kind"] not in VALID_CONTRACT_VALUE_KINDS:
			fail(
			 f"{ctx}: contract.{field}.kind "
			 f"{descriptor['kind']!r} is invalid"
			)
		minimum = descriptor["minimum"]
		if (
		 not isinstance(minimum, int)
		 or isinstance(minimum, bool)
		 or minimum < 1
		 or minimum > 255
		):
			fail(
			 f"{ctx}: contract.{field}.minimum must be an integer in [1, 255]"
			)

	for field in ("input_kinds", "output_kinds"):
		kinds = contract.get(field)
		if not isinstance(kinds, list):
			fail(f"{ctx}: contract.{field} must be an array")
		if (
		 field == "output_kinds"
		 and not kinds
		 and variadic["variadic_outputs"] is None
		 and not (
		  op.get("api_return") == "void"
		  and contract.get("mutated_inputs")
		 )
		):
			fail(
			 f"{ctx}: contract.output_kinds may be empty only for a "
			 "void operation with a mutated input"
			)
		if (
		 field == "input_kinds"
		 and not kinds
		 and variadic["variadic_inputs"] is None
		 and op.get("kind") != "nullary_scalar"
		):
			fail(
			 f"{ctx}: only nullary_scalar operations may use an empty "
			 "fixed contract.input_kinds array unless variadic_inputs is present"
			)
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

	valueValidation = contract.get("value_validation")
	if op.get("kind") == "session_command":
		if valueValidation != "session_command":
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
	elif valueValidation is not None:
		fail(
		 f"{ctx}: contract.value_validation is reserved for session_command"
		)

	effects = contract.get("effects")
	if not isinstance(effects, list) or not effects:
		fail(f"{ctx}: contract.effects must be a non-empty array")
	if len(set(effects)) != len(effects):
		fail(f"{ctx}: contract.effects contains duplicates")
	invalidEffects = [effect for effect in effects if effect not in VALID_CONTRACT_EFFECTS]
	if invalidEffects:
		fail(f"{ctx}: contract.effects contains invalid effects {invalidEffects!r}")

	inputCount = len(contract["input_kinds"])
	outputCount = len(contract["output_kinds"])
	mutatedInputs = contract.get("mutated_inputs")
	if not isinstance(mutatedInputs, list):
		fail(f"{ctx}: contract.mutated_inputs must be an array")
	if len(set(mutatedInputs)) != len(mutatedInputs):
		fail(f"{ctx}: contract.mutated_inputs contains duplicates")
	if any(not isinstance(index, int) or isinstance(index, bool)
	 or index < 0 or index >= inputCount for index in mutatedInputs):
		fail(f"{ctx}: contract.mutated_inputs contains an invalid input index")

	optionalInputs = contract.get("optional_inputs", [])
	if not isinstance(optionalInputs, list):
		fail(f"{ctx}: contract.optional_inputs must be an array")
	if len(set(optionalInputs)) != len(optionalInputs):
		fail(f"{ctx}: contract.optional_inputs contains duplicates")
	if any(not isinstance(index, int) or isinstance(index, bool)
	 or index < 0 or index >= inputCount for index in optionalInputs):
		fail(f"{ctx}: contract.optional_inputs contains an invalid input index")
	if set(optionalInputs) & set(mutatedInputs):
		fail(f"{ctx}: optional semantic inputs cannot be mutated")

	aliases = contract.get("output_alias_inputs")
	if not isinstance(aliases, list) or len(aliases) != outputCount:
		fail(f"{ctx}: contract.output_alias_inputs must contain one entry per output")
	if any(not isinstance(index, int) or isinstance(index, bool)
	 or index < -1 or index >= inputCount for index in aliases):
		fail(f"{ctx}: contract.output_alias_inputs contains an invalid input index")
	if any(index in optionalInputs for index in aliases):
		fail(f"{ctx}: outputs cannot alias optional semantic inputs")

	controlFlow = contract.get("control_flow")
	if controlFlow not in VALID_CONTRACT_CONTROL_FLOW:
		fail(
		 f"{ctx}: contract.control_flow {controlFlow!r} not in "
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
	normalizedAttributes = semanticAttributeSpecs(op)
	if len(normalizedAttributes) > 8:
		fail(f"{ctx}: semantic operation supports at most 8 attributes")
	attributeNames = [attribute["name"] for attribute in normalizedAttributes]
	if len(set(attributeNames)) != len(attributeNames):
		fail(f"{ctx}: semantic attribute names contain duplicates")


def validateDnnMetadata(ctx: str, op: dict) -> None:
	dnn = op.get("dnn")
	if dnn is None:
		return
	if "contract" not in op:
		fail(f"{ctx}: dnn metadata requires a semantic operation contract")
	if not isinstance(dnn, dict):
		fail(f"{ctx}: dnn must be a table")
	allowed = {"role", "epilogue", "epilogue_requires_input"}
	unknown = set(dnn) - allowed
	if unknown:
		fail(f"{ctx}: dnn contains unknown fields {sorted(unknown)!r}")
	role = dnn.get("role")
	if role not in VALID_DNN_ROLES:
		fail(f"{ctx}: dnn.role {role!r} not in {sorted(VALID_DNN_ROLES)}")
	epilogue = dnn.get("epilogue", "none")
	if epilogue not in VALID_DNN_EPILOGUES:
		fail(
		 f"{ctx}: dnn.epilogue {epilogue!r} not in "
		 f"{sorted(VALID_DNN_EPILOGUES)}"
		)
	if role != "matmul" and epilogue != "none":
		fail(f"{ctx}: only the matmul dnn role may declare an epilogue")
	if role == "matmul" and op["contract"].get("lowering") != "gemm":
		fail(f"{ctx}: the matmul dnn role requires contract.lowering = 'gemm'")
	requiredInput = dnn.get("epilogue_requires_input")
	if requiredInput is None:
		return
	if epilogue == "none":
		fail(f"{ctx}: epilogue_requires_input requires a non-none epilogue")
	if (
	 not isinstance(requiredInput, int)
	 or isinstance(requiredInput, bool)
	 or requiredInput < 0
	 or requiredInput >= len(op["contract"]["input_kinds"])
	):
		fail(f"{ctx}: dnn.epilogue_requires_input is outside the input tuple")
	if requiredInput not in op["contract"].get("optional_inputs", []):
		fail(f"{ctx}: dnn.epilogue_requires_input must name an optional input")


def validateSessionDescriptor(ctx: str, op: dict) -> None:
	session = op.get("session")
	if session is None:
		fail(f"{ctx}: session_command requires a session descriptor")
	if not isinstance(session, dict):
		fail(f"{ctx}: session must be a table")
	owners = session.get("owners")
	if not isinstance(owners, list) or not owners:
		fail(f"{ctx}: session.owners must be a non-empty array")
	if any(not isinstance(owner, str) or not owner for owner in owners):
		fail(f"{ctx}: session.owners must contain non-empty strings")
	if len(set(owners)) != len(owners):
		fail(f"{ctx}: session.owners contains duplicates")
	transition = session.get("transition")
	if not isinstance(transition, str) or not transition:
		fail(f"{ctx}: session.transition must be a non-empty string")
	effects = session.get("effects")
	if not isinstance(effects, list) or not effects:
		fail(f"{ctx}: session.effects must be a non-empty array")
	if len(set(effects)) != len(effects):
		fail(f"{ctx}: session.effects contains duplicates")
	invalidEffects = [
	 effect for effect in effects if effect not in VALID_SESSION_EFFECTS
	]
	if invalidEffects:
		fail(
		 f"{ctx}: session.effects contains invalid effects "
		 f"{invalidEffects!r}"
		)
	completion = session.get("completion")
	if completion not in VALID_SESSION_COMPLETIONS:
		fail(
		 f"{ctx}: session.completion {completion!r} not in "
		 f"{sorted(VALID_SESSION_COMPLETIONS)}"
		)


def validatePythonBinding(ctx: str, op: dict) -> None:
	python = op.get("python")
	if python is None:
		return
	if not isinstance(python, dict):
		fail(f"{ctx}: python must be a table")
	apiParams = op.get("api_params", [])
	if not apiParams:
		fail(f"{ctx}: python binding requires schema-owned api_params")
	binding = python.get("binding", "generated")
	if binding not in ("generated", "manual"):
		fail(f"{ctx}: python.binding must be 'generated' or 'manual'")
	apiReturn = op.get("api_return", "oa::Matrix")
	result = op.get("python_result")
	if (
	 binding == "generated"
	 and
	 apiReturn not in (
	  "void", "oa::Matrix", "oa::Audio", "oa::Image",
	  "oa::Vec<oa::Matrix>"
	 )
	 and result is None
	):
		fail(
		 f"{ctx}: generated python binding requires api_return "
		 "= 'void'/'oa::Matrix'/'oa::Audio'/'oa::Image'/"
		 "'oa::Vec<oa::Matrix>' "
		 "or a python_result descriptor"
		)
	if result is not None:
		if not isinstance(result, dict):
			fail(f"{ctx}: python_result must be a table")
		cppType = result.get("cpp_type")
		if (
		 not isinstance(cppType, str)
		 or not re.fullmatch(
		  r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*",
		  cppType,
		 )
		):
			fail(f"{ctx}: python_result.cpp_type must be a qualified C++ type")
		members = result.get("members")
		if (
		 not isinstance(members, list)
		 or not members
		 or any(
		  not isinstance(member, str)
		  or not re.fullmatch(r"[a-z][A-Za-z0-9]*", member)
		  for member in members
		 )
		):
			fail(
			 f"{ctx}: python_result.members must be a non-empty array "
			 "of camelCase identifiers"
			)
		if len(set(members)) != len(members):
			fail(f"{ctx}: python_result.members contains duplicates")
		name = result.get("name")
		if name is not None and (
		 not isinstance(name, str)
		 or not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name)
		):
			fail(f"{ctx}: python_result.name must use PascalCase")
		if "bind_type" in result and not isinstance(result["bind_type"], bool):
			fail(f"{ctx}: python_result.bind_type must be a bool")
	args = python.get("args")
	if not isinstance(args, list) or len(args) != len(apiParams):
		fail(f"{ctx}: python.args must contain one name per api parameter")
	if any(not isinstance(name, str) or not re.fullmatch(r"[a-z][A-Za-z0-9]*", name) for name in args):
		fail(f"{ctx}: python.args must use camelCase identifiers")
	if len(set(args)) != len(args):
		fail(f"{ctx}: python.args contains duplicates")
	if "name" in python and (
	 not isinstance(python["name"], str)
	 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", python["name"])
	):
		fail(f"{ctx}: python.name must be a valid identifier")
	if "compatibility_name" in python:
		fail(
		 f"{ctx}: python.compatibility_name is removed; Python namespace "
		 "members must match the public C++ spelling"
		)
	if python.get("result_validation") not in (None, "not_empty", "is_valid"):
		fail(
		 f"{ctx}: python.result_validation must be 'not_empty' or "
		 "'is_valid'"
		)
	if "error" in python and (
	 not isinstance(python["error"], str) or not python["error"]
	):
		fail(f"{ctx}: python.error must be a non-empty string")
	if "doc" in python and not isinstance(python["doc"], str):
		fail(f"{ctx}: python.doc must be a string")


def validateKernelMetadata(ctx: str, op: dict) -> None:
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


def loadKernelRegistry(path: Path) -> set[str]:
	"""Parse the fixed registry and its generated oa/runtime includes."""
	if not path.exists():
		fail(f"kernel registry not found: {path}")
	texts = [path.read_text()]
	# Match includes under oa/runtime/ (case-insensitive prefix for migration compat).
	for include in re.findall(r'#include\s+<[Oo][Aa]/[Rr]untime/([^>]+)>', texts[0]):
		included = path.parent / include
		if included.exists():
			texts.append(included.read_text())
	pat = re.compile(r'\{\s*"([^"]+)"\s*,\s*OA_COMPUTE_KERNEL_ID')
	names = {name for text in texts for name in pat.findall(text)}
	if not names:
		fail(f"no kernel names parsed from {path} or its generated includes")
	return names


def validateSchemaMetadata(schemaPath: Path, data: dict) -> None:
	"""Reject schema metadata that would emit into the wrong tree."""
	if "surface" in data and data["surface"] not in VALID_OPERATION_SURFACES:
		fail(
		 f"{schemaPath.name}: surface {data['surface']!r} not in "
		 f"{sorted(VALID_OPERATION_SURFACES)}"
		)
	namespace = data.get("namespace", "")
	if not namespace:
		return
	if data.get("surface") == "session_command":
		if not re.fullmatch(r"oa::[A-Z][A-Za-z0-9]*", namespace):
			fail(
			 f"{schemaPath.name}: session-command namespace must name an "
			 "oa::PascalCase session type"
			)
		return
	if not re.fullmatch(r"oa::Fn[A-Z][A-Za-z0-9]*", namespace):
		fail(
		 f"{schemaPath.name}: operation namespace must be an "
		 "oa::FnPascalCase namespace"
		)


def validateSchema(
 schemaPath: Path,
 ops: list[dict],
 registry: set[str],
 defaultSurface: str | None = None,
) -> None:
	seen = set()
	for op in ops:
		name = op.get("name")
		ctx = f"{schemaPath.name}:{name or '<unnamed>'}"
		if not isinstance(name, str) or not re.fullmatch(r"[a-z][A-Za-z0-9]*", name):
			fail(f"{ctx}: op name missing or not camelCase")
		if name in seen:
			fail(f"{ctx}: duplicate op name")
		seen.add(name)
		surface = op.get("surface", defaultSurface)
		if surface not in VALID_OPERATION_SURFACES:
			fail(
			 f"{ctx}: surface {surface!r} not in "
			 f"{sorted(VALID_OPERATION_SURFACES)}"
			)
		if op.get("kind") == "session_command" and surface != "session_command":
			fail(f"{ctx}: session_command kind requires surface = 'session_command'")
		if surface == "session_command" and op.get("kind") != "session_command":
			fail(f"{ctx}: session_command surface requires kind = 'session_command'")
		if (
		 surface in ("public_operation", "public_cpp_operation", "stable_composite")
		 and "contract" not in op
		):
			fail(f"{ctx}: {surface} requires a semantic operation contract")
		if surface in ("value_view", "kernel", "alias", "host_utility") and "contract" in op:
			fail(f"{ctx}: {surface} rows cannot own semantic operation contracts")
		if surface == "session_command":
			validateSessionDescriptor(ctx, op)
		elif "session" in op:
			fail(f"{ctx}: only session_command rows may own a session descriptor")
		if surface == "host_utility":
			if op.get("body", "auto") != "cpu_util":
				fail(f"{ctx}: host_utility requires body = 'cpu_util'")
			if op.get("kernel_forward"):
				fail(f"{ctx}: host_utility cannot name an executable kernel")
		if surface == "value_view":
			if op.get("body", "auto") != "manual_session" or not op.get("api_params"):
				fail(
				 f"{ctx}: value_view requires a manual body and exact api_params"
				)
			if op.get("kernel_forward"):
				fail(f"{ctx}: value_view cannot name an executable kernel")
		if surface == "kernel":
			for field in ("api_params", "python", "autograd"):
				if op.get(field):
					fail(f"{ctx}: kernel row cannot own {field}")
		if surface == "alias" and op.get("kernel_forward"):
			fail(f"{ctx}: alias row cannot name an executable kernel")
		validateOperationContract(ctx, op)
		validateDnnMetadata(ctx, op)
		validatePythonBinding(ctx, op)
		validateKernelMetadata(ctx, op)

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
		if kind == "session_command" and body != "manual_session":
			fail(f"{ctx}: kind 'session_command' requires body = 'manual_session'")

		kf = op.get("kernel_forward")
		if body == "cpp_expr":
			kf = kf or name
			op["kernel_forward"] = kf
  # kernel_forward is optional for body types that don't dispatch a GPU
  # kernel: `manual_session` ops (session recorders like DecodeFrame /
  # EncodeFrame) and `cpu_util` ops (host-only utilities like NAL
  # Annex-B parsing). For everything else it's required.
		if not kf and body not in ("manual_session", "cpu_util"):
			fail(f"{ctx}: kernel_forward missing")
		if kf and body not in ("manual_session", "cpu_util", "cpp_expr") and kf not in registry:
			fail(f"{ctx}: kernel '{kf}' not in the composed kernel registry")

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

		apiParams = op.get("api_params", [])
		seenApiParams = set()
		apiParamTypes = {}
		seenDefault = False
		for i, param in enumerate(apiParams):
			if isinstance(param, str):
				decl, separator, _ = param.partition("=")
				parts = decl.strip().rsplit(None, 1)
				if len(parts) != 2 or not parts[0] or not parts[1]:
					fail(f"{ctx}: api_params[{i}] must be '<type> <name> [= default]'")
				paramType = parts[0]
				name = parts[1]
				hasDefault = bool(separator)
			elif isinstance(param, dict):
				for key in ("name", "type"):
					if key not in param:
						fail(f"{ctx}: api_params[{i}].{key} missing")
				paramType = param.get("type", "")
				baseName = param.get("name", "")
				name = f"in{baseName[:1].upper()}{baseName[1:]}"
				hasDefault = "default" in param
			else:
				fail(f"{ctx}: api_params[{i}] must be a string or table")
				continue
			if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
				fail(f"{ctx}: api_params[{i}] has invalid C++ parameter name {name!r}")
			if name in seenApiParams:
				fail(f"{ctx}: duplicate api parameter {name}")
			seenApiParams.add(name)
			apiParamTypes[name] = paramType
			if hasDefault:
				seenDefault = True
			elif seenDefault:
				fail(f"{ctx}: non-default api parameter {name} follows a default")
		if op.get("generate_forwarder") and not apiParams:
			fail(f"{ctx}: generate_forwarder requires api_params")
		if op.get("generate_engine_overload") and not apiParams:
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
		nodeHeader = ag.get("node_header")
		if nodeHeader is not None and (
		 not isinstance(nodeHeader, str)
		 or not re.fullmatch(r"oa/[A-Za-z0-9_/.-]+\.h", nodeHeader)
		):
			fail(
			 f"{ctx}: autograd.node_header must be an oa/... header path"
			)
		attach = ag.get("attach")
		if attach is not None:
			if attach not in ("standard", "broadcast_binary", "manual"):
				fail(
				 f"{ctx}: autograd.attach {attach!r} must be 'standard', "
				 "'broadcast_binary', or 'manual'"
				)
			inputs = ag.get("inputs", [])
			if attach != "manual" and (not ag.get("grad_class") or not inputs):
				fail(f"{ctx}: autograd.attach requires grad_class and non-empty inputs")
			if attach == "manual":
				if formula != "manual" or not inputs:
					fail(
					 f"{ctx}: manual autograd attachment requires "
					 "formula = 'manual' and non-empty inputs"
					)
				outputs = ag.get("outputs")
				if (
				 not isinstance(outputs, list)
				 or not outputs
				 or any(not isinstance(output, str) or not output for output in outputs)
				):
					fail(
					 f"{ctx}: manual autograd attachment requires "
					 "non-empty outputs"
					)
				implementation = ag.get("implementation")
				if not isinstance(implementation, str) or not implementation:
					fail(
					 f"{ctx}: manual autograd attachment requires "
					 "implementation provenance"
					)
			inputRanks = ag.get("input_ranks")
			if inputRanks is not None and (
			 not isinstance(inputRanks, list)
			 or len(inputRanks) != len(inputs)
			 or any(not isinstance(rank, int) or rank < 0 for rank in inputRanks)
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
			seenStateMembers = set()
			seenStateSources = set()
			for i, item in enumerate(state):
				if not isinstance(item, dict):
					fail(f"{ctx}: autograd.state[{i}] must be a table")
				for key in ("member", "source", "type"):
					if not isinstance(item.get(key), str) or not item[key]:
						fail(f"{ctx}: autograd.state[{i}].{key} must be a non-empty string")
				member = item.get("member", "")
				source = item.get("source", "")
				stateType = item.get("type", "")
				if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", member):
					fail(f"{ctx}: autograd.state[{i}].member is not a C++ identifier")
				if source not in apiParamTypes:
					fail(f"{ctx}: autograd.state[{i}].source must name an api_params entry")
				if source in inputs:
					fail(f"{ctx}: autograd.state[{i}].source duplicates a matrix input")
				if apiParamTypes.get(source) != stateType:
					fail(f"{ctx}: autograd.state[{i}].type must match api_params source type")
				if member in seenStateMembers or source in seenStateSources:
					fail(f"{ctx}: autograd.state members and sources must be unique")
				seenStateMembers.add(member)
				seenStateSources.add(source)


def categoryFromSchema(schemaPath: Path, data: dict) -> tuple[str, str]:
	"""Returns (category, file_category) for the schema's output files."""
	if cat := data.get("category"):
		category = cat
	else:
		stem = schemaPath.stem
		for domain in ["core", "ml", "vision", "audio", "ui", "crypto"]:
			if stem.startswith(domain):
				stem = stem[len(domain):]
				break
		for prefix in ["FnMatrix", "FnImage", "FnAudio"]:
			if stem.startswith(prefix):
				category = stem[len(prefix):]
				break
		else:
			category = stem[2:] if stem.startswith("Fn") else stem

	fileCategory = data.get("file_category", category)
	return category, fileCategory
