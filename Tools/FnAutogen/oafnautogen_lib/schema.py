from __future__ import annotations

import re
from pathlib import Path

from .config import (
	AUTO_BODY_KINDS,
	SLANG_EMIT_KINDS,
	VALID_BODIES,
	VALID_DISPATCH_WORKGROUPS,
	VALID_FORMULAS_PREFIX,
	VALID_KINDS,
	VALID_OUTPUT_DTYPES,
	VALID_OUTPUT_SHAPES,
)
from .errors import fail


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
		seen_default = False
		for i, param in enumerate(api_params):
			if isinstance(param, str):
				decl, separator, _ = param.partition("=")
				parts = decl.strip().rsplit(None, 1)
				if len(parts) != 2 or not parts[0] or not parts[1]:
					fail(f"{ctx}: api_params[{i}] must be '<type> <name> [= default]'")
				name = parts[1]
				has_default = bool(separator)
			elif isinstance(param, dict):
				for key in ("name", "type"):
					if key not in param:
						fail(f"{ctx}: api_params[{i}].{key} missing")
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
