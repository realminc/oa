"""Normalized signature machinery: methods, parity and adapters."""

from __future__ import annotations

import re
from typing import Any

from .model import validate_unique
from .python import python_operation
from .util import IDENTIFIER, banner, require

SIGNATURE_KEYS = {"parameters", "returns", "visibility", "receiver", "doc", "target", "raw_return", "parity"}


def normalize_signature(
	name: Any,
	parameters: Any,
	returns: Any,
	*,
	visibility: Any = "pub",
	receiver: Any = None,
	doc: Any = None,
	target: Any = None,
	raw_return: Any = None,
	parity: Any = None,
	domain: str,
	where: str,
	allowed_types: set[str] | None = None,
	python: Any = None,
) -> dict[str, Any]:
	"""Validate and normalize one operation's public signature record.

	The record is the single source for the emitted free-function declaration;
	methods and language adapters derive their own surfaces from the same
	ordered parameters. A leading parameter typed as a reference to the domain
	value is the method receiver; `"none"` keeps the operation module-only.
	"""
	require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where} name is invalid")
	require(visibility == "pub" or (isinstance(visibility, str) and visibility.startswith("pub(") and visibility.endswith(")")), f"{where} visibility is invalid")
	require(isinstance(returns, str) and returns.strip() and ";" not in re.sub(r"\[[^\]]*\]", "", returns), f"{where} returns is invalid")
	require(isinstance(parameters, list) and (parameters or isinstance(target, str)), f"{where} parameters must be a non-empty array")
	normalized = []
	names: set[str] = set()
	for parameter in parameters:
		require(isinstance(parameter, list) and len(parameter) == 2, f"{where} parameter is invalid")
		parameter_name, parameter_type = parameter
		require(isinstance(parameter_name, str) and IDENTIFIER.fullmatch(parameter_name) is not None, f"{where} parameter name is invalid")
		validate_unique(parameter_name, names, f"{where} parameter")
		require(
			isinstance(parameter_type, str) and parameter_type.strip(),
			f"{where} parameter type is invalid",
		)
		if allowed_types is not None:
			require(parameter_type in allowed_types, f"{where} parameter type is unsupported")
		normalized.append([parameter_name, parameter_type])
	receiver_name = normalized[0][0] if normalized and normalized[0][1] == f"&{domain.title()}" else None
	if receiver is not None:
		require(
			receiver == "none" or receiver == receiver_name,
			f"{where} receiver must be none or the leading domain-value parameter",
		)
		receiver_name = None if receiver == "none" else receiver_name
	doc_lines = None
	if doc is not None:
		if isinstance(doc, str):
			doc_lines = [doc]
		else:
			require(
				isinstance(doc, list) and doc and all(isinstance(line, str) and "\n" not in line for line in doc),
				f"{where} doc is invalid",
			)
			doc_lines = list(doc)
	result = {
		"name": name,
		"visibility": visibility,
		"parameters": normalized,
		"returns": returns,
		"receiver": receiver_name,
		"doc": doc_lines,
	}
	if target is not None:
		result["target"] = target
	if raw_return is not None:
		result["raw_return"] = raw_return
	if parity is not None:
		result["parity"] = parity
	if python is not None:
		result["python"] = python
		python_operation(result)
	return result


def operation_signature(
	domain: str,
	operation: dict[str, Any],
	where: str,
	*,
	allowed_types: set[str] | None = None,
) -> dict[str, Any] | None:
	"""Normalize one operation's public signature from its schema record.

	Precedence: an explicit `signature` record, a complete-generation `rust`
	record, then mechanical derivation from Matrix kind/scalar metadata.
	Returns None for records without a public signature, such as
	lowering-only or metadata-only operations.
	"""
	signature = operation.get("signature")
	if signature is not None:
		require(
			isinstance(signature, dict)
			and set(signature) <= SIGNATURE_KEYS
			and "parameters" in signature
			and "returns" in signature,
			f"{where}.signature fields are incomplete or unknown",
		)
		return normalize_signature(
			operation["name"],
			signature["parameters"],
			signature["returns"],
			visibility=signature.get("visibility", "pub"),
			receiver=signature.get("receiver"),
			doc=signature.get("doc"),
			target=signature.get("target"),
			raw_return=signature.get("raw_return"),
			parity=signature.get("parity"),
			domain=domain,
			where=f"{where}.signature",
			allowed_types=allowed_types,
			python=operation.get("python"),
		)
	rust = operation.get("rust")
	if isinstance(rust, dict) and "parameters" in rust:
		return normalize_signature(
			operation["name"],
			rust["parameters"],
			rust.get("returns"),
			receiver=rust.get("receiver"),
			doc=rust.get("doc"),
			domain=domain,
			where=where,
			allowed_types=allowed_types,
			python=operation.get("python"),
		)
	kind = operation.get("kind")
	if kind in {"binary", "mat_mul_nt"}:
		parameters = [["left", "&Matrix"], ["right", "&Matrix"]]
	elif kind == "unary":
		parameters = [["input", "&Matrix"]]
	elif kind == "unary_scalar":
		parameters = [["input", "&Matrix"], [operation.get("scalar_name"), "f32"]]
	else:
		return None
	return normalize_signature(
		operation["name"],
		parameters,
		"Matrix",
		doc=operation.get("doc"),
		domain=domain,
		where=where,
	)


def api_signature(
	domain: str,
	api: dict[str, Any],
	where: str,
	*,
	allowed_types: set[str] | None = None,
) -> dict[str, Any]:
	"""Normalize a `rust_api` operation record into the shared signature model."""
	return normalize_signature(
		api["name"],
		api["parameters"],
		api.get("returns", "Matrix"),
		visibility=api.get("visibility", "pub"),
		receiver=api.get("receiver"),
		doc=api.get("doc"),
		domain=domain,
		where=where,
		allowed_types=allowed_types,
	)


def signature_parameters(signature: dict[str, Any]) -> str:
	return ", ".join(f"{name}: {kind}" for name, kind in signature["parameters"])


def signature_header(signature: dict[str, Any]) -> str:
	return f"{signature['visibility']} fn {signature['name']}({signature_parameters(signature)}) -> Result<{signature['returns']}>"


def signature_method(domain: str, signature: dict[str, Any]) -> str | None:
	"""Emit a value method delegating to the free operation, or None.

	The method removes the receiver parameter, keeps the remaining ordered
	parameters, and forwards once to `crate::{domain}::{name}`. Internal
	operations and records without a domain-value receiver stay module-only.
	"""
	receiver = signature["receiver"]
	if receiver is None or signature["visibility"] != "pub":
		return None
	rest = [parameter for parameter in signature["parameters"] if parameter[0] != receiver]
	require(rest != signature["parameters"], f"{domain.title()} {signature['name']} receiver is not a parameter")
	parameters = "&self" + (", " + ", ".join(f"{name}: {kind}" for name, kind in rest) if rest else "")
	returns = "Self" if signature["returns"] == domain.title() else signature["returns"]
	arguments = ", ".join(["self", *(name for name, _ in rest)])
	target = f"crate::{domain}::{signature['name']}"
	doc = f"/// {signature['doc'][0]}\n///\n" if signature["doc"] else ""
	return f"{doc}/// Delegates to [`{target}`].\npub fn {signature['name']}({parameters}) -> Result<{returns}> {{\n\t{target}({arguments})\n}}"


SIGNATURE_PRIMITIVE_TYPES = {
	"bool", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize",
	"u8", "u16", "u32", "u64", "u128", "usize", "str", "Self", "Vec", "Option", "String", "Box",
}


def signature_named_types(signature: dict[str, Any]) -> set[str]:
	"""Named Rust types referenced by a signature that need `use` imports."""
	named = set()
	kinds = [kind for _, kind in signature["parameters"]]
	kinds += [kind for _, kind in signature.get("parity") or []]
	kinds.append(signature["returns"])
	for kind in kinds:
		core = re.sub(r"^&\s*(mut\s+)?", "", kind)
		if core.startswith(("crate::", "std::", "impl ")) or core == "()":
			continue
		for part in re.findall(r"[A-Z][A-Za-z0-9_]*", core):
			if part not in SIGNATURE_PRIMITIVE_TYPES:
				named.add(part)
	return named


def domain_signatures(domain: str, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Normalized public signatures for one value domain, in schema order."""
	signatures = []
	for schema in schemas:
		contracts = schema.get("contracts")
		records = list(contracts.values()) if isinstance(contracts, dict) else list(contracts or [])
		records += schema.get("operations", []) + schema.get("signature_records", [])
		for record in records:
			if not isinstance(record, dict) or "name" not in record:
				continue
			signature = operation_signature(domain, record, f"{domain.title()} {record['name']}")
			if signature is not None:
				signatures.append(signature)
	return signatures


def generate_domain_methods(domain: str, signatures: list[dict[str, Any]], schema_hash: str, schema_name: str) -> str:
	"""Emit the generated value impl delegating each eligible operation."""
	emitted = [(signature, signature_method(domain, signature)) for signature in signatures]
	emitted = [(signature, method) for signature, method in emitted if method is not None]
	require(bool(emitted), f"{domain.title()} has no method-eligible operations")
	names: set[str] = set()
	named: set[str] = set()
	for signature, _ in emitted:
		validate_unique(signature["name"], names, f"{domain.title()} method")
		named |= signature_named_types(signature)
	value = domain.title()
	named.discard(value)
	imports = {value, "Result"}
	if "Matrix" in named:
		imports.add("Matrix")
		named.discard("Matrix")
	use = "use crate::{" + ", ".join(sorted(imports)) + "};"
	if named:
		use += f"\nuse crate::{domain}::{{{', '.join(sorted(named))}}};"
	preamble = banner(schema_hash, "//", schema_name)
	if "," in schema_name:
		preamble = preamble.replace("// schema=", "// schemas=")
	parity = []
	for signature, _ in emitted:
		kinds = [kind for _, kind in signature["parameters"]]
		if any(kind.startswith("impl ") for kind in kinds):
			continue
		parameters = ", ".join(kinds)
		function_type = f"fn({parameters}) -> Result<{signature['returns']}>"
		parity.extend((
			f"const _: {function_type} = crate::{domain}::{signature['name']};",
			f"const _: {function_type} = {value}::{signature['name']};",
		))
	return preamble + use + f"\n\nimpl {value} {{\n" + "\n\n".join(method for _, method in emitted) + "\n}\n\n" + "\n".join(parity) + "\n"


def generate_signature_parity(
	module_path: str,
	signatures: list[dict[str, Any]],
	schema_hash: str,
	schema_name: str,
	import_paths: dict[str, str] | None = None,
) -> str:
	"""Compile-time parity checks proving signature records match real fns."""
	named: set[str] = set()
	for signature in signatures:
		named |= signature_named_types(signature)
	use = "use crate::{" + ", ".join(sorted({"Result"} | (named & {"Matrix", "Engine"}))) + "};"
	by_path: dict[str, list[str]] = {}
	for name in sorted(named - {"Matrix", "Engine"}):
		by_path.setdefault((import_paths or {}).get(name, module_path), []).append(name)
	for path in sorted(by_path):
		use += f"\nuse {path}::{{{', '.join(by_path[path])}}};"
	parity = []
	for signature in signatures:
		kinds = [kind for _, kind in signature["parameters"]]
		target = signature.get("target", f"{module_path}::{signature['name']}")
		if any(kind.startswith("impl ") for kind in kinds):
			concrete = signature.get("parity")
			if not concrete:
				continue
			require(
				isinstance(concrete, list) and len(concrete) == len(kinds) and all(
					isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)
					for pair in concrete),
				f"{signature['name']} parity must mirror its parameters",
			)
			concrete_kinds = [re.sub(r"'[a-zA-Z_]+", "'static", kind) for _, kind in concrete]
			args = ", ".join(name for name, _ in concrete)
			returns = re.sub(r"'[a-zA-Z_]+", "'static", signature["returns"])
			function_type = (
				f"fn({', '.join(concrete_kinds)}) -> {returns}"
				if signature.get("raw_return")
				else f"fn({', '.join(concrete_kinds)}) -> Result<{returns}>"
			)
			parity.append(f"#[allow(clippy::type_complexity)]\nconst _: {function_type} = |{args}| {target}({args});")
			continue
		kinds = [re.sub(r"'[a-zA-Z_]+", "'static", kind) for kind in kinds]
		returns = re.sub(r"'[a-zA-Z_]+", "'static", signature["returns"])
		function_type = (
			f"fn({', '.join(kinds)}) -> {returns}"
			if signature.get("raw_return")
			else f"fn({', '.join(kinds)}) -> Result<{returns}>"
		)
		parity.append(f"#[allow(clippy::type_complexity)]\nconst _: {function_type} = {target};")
	return banner(schema_hash, "//", schema_name) + use + "\n\n" + "\n".join(parity) + "\n"
