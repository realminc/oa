"""Shared cryptography emit and validation helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .model import validate_unique
from .wire import render_wire_values, wire_parts
from .signature import api_signature, signature_header
from .util import IDENTIFIER, banner, require

# Checked helper ABIs for FIPS 180-4 / FIPS 202 fixed digests. Formula bodies
# remain in their existing Slang modules; schemas own adapters and public APIs.
FIXED_HASH_HELPERS = {
	"sha224": ("sha2_u32", "sha2_224", 28, "uint"),
	"sha256": ("sha2_u32", "sha2_256", 32, "uint"),
	"sha384": ("sha2_u32_pair", "sha2_384", 48, "uint"),
	"sha512": ("sha2_u32_pair", "sha2_512", 64, "uint"),
	"sha512_224": ("sha2_u32_pair", "sha2_512_224", 28, "uint"),
	"sha512_256": ("sha2_u32_pair", "sha2_512_256", 32, "uint"),
	"sha3_224": ("sha3", "sha3_224", 28, "uint"),
	"sha3_256": ("sha3", "sha3_256", 32, "uint2"),
	"sha3_384": ("sha3", "sha3_384", 48, "uint2"),
	"sha3_512": ("sha3", "sha3_512", 64, "uint2"),
}


def cryptography_templates(category: str) -> dict[str, str]:
	path = FN_ROOT / "template/crypto" / f"{category}.rs.in"
	sections = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", path.read_text(encoding="utf-8"), re.S)
	require(len(sections) == len({name for name, _ in sections}), "duplicate Cryptography template section")
	return dict(sections)


def cryptography_body(schema: dict[str, Any], api: dict[str, Any], category: str) -> str:
	kernels = {k["kernel_id"]: k for k in schema["kernels"]}
	contracts = {op["name"]: op for op in schema["contracts"]}
	tokens = dict(api["values"])
	if category == "verify":
		tokens["prehash_validation"] = 'if KernelId::mldsa_prehash_bytes(algorithm).is_none() { return Err(Error::invalid_argument("unknown ML-DSA prehash identifier")); }' if any(name == "algorithm" for name, _ in api["parameters"]) else ""
	if category == "hash":
		tokens["contract"] = f"crate::core::operation::{schema['domain']}::{api['contracts'][0].upper()}"
	else:
		tokens["routes"] = "\n".join(
			f"MlDsaParameters::MlDsa{kernels[stage['kernel']]['parameter_set']} => (crate::core::operation::{schema['domain']}::{contract.upper()}, KernelId::{stage['kernel']}),"
			for contract, stage in zip(api["contracts"], api["stages"])
		)
	# API expressions are keyed by ABI field, never by JSON map iteration order.
	for index, stage in enumerate(api["stages"] if category == "hash" else api["stages"][:1]):
		buffers, pushes = wire_parts(kernels[stage["kernel"]])
		tokens[f"bindings:{index}"] = render_wire_values(stage["bindings"], buffers, buffers=True)
		tokens[f"push:{index}"] = render_wire_values({name: ["U32", value] for name, value in stage["push_constants"].items()}, pushes, buffers=False)
		if category == "hash":
			tokens[f"kernel:{index}"] = f"KernelId::{stage['kernel']}"
	if api["attributes"]:
		tokens["attributes"] = ",\n".join(
			f'OpAttribute::UnsignedInteger {{ name: "{name}".into(), value: {api["attributes"][name]}, }}'
			for name, _ in contracts[api["contracts"][0]]["attributes"]
		) + ","
	body = cryptography_templates(category)[api["template"]]
	require(set(re.findall(r"\{\{([^{}]+)\}\}", body)) == set(tokens), f"Cryptography {api['name']} template bindings are unresolved, missing or unused")
	return re.sub(r"\{\{([^{}]+)\}\}", lambda match: tokens[match[1]], body).rstrip()


def validate_cryptography_api(schema: dict[str, Any], category: str) -> None:
	apis = schema.get("rust_api")
	require(isinstance(apis, list) and apis, "Cryptography rust_api must be nonempty")
	kernels = {k["kernel_id"]: k for k in schema["kernels"]}
	contracts = {op["name"]: op for op in schema["contracts"]}
	names, covered = set(), set()
	for api in apis:
		require(isinstance(api, dict) and set(api) == {"name", "template", "visibility", "parameters", "doc", "annotations", "contracts", "values", "stages", "attributes"}, "Cryptography API fields are incomplete or unknown")
		name = api["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name), "Cryptography API name is invalid")
		validate_unique(name, names, "Cryptography API name")
		require(isinstance(api["visibility"], str) and api["visibility"] in {"pub", "pub(crate)"}, "Cryptography API visibility is invalid")
		require(isinstance(api["doc"], list) and api["doc"] and all(isinstance(line, str) and "\n" not in line for line in api["doc"]), "Cryptography API docs are invalid")
		require(isinstance(api["annotations"], list) and all(isinstance(line, str) and "\n" not in line for line in api["annotations"]), "Cryptography API annotations are invalid")
		require(isinstance(api["parameters"], list) and api["parameters"], "Cryptography API parameters are invalid")
		parameter_names, inputs = set(), []
		for parameter in api["parameters"]:
			require(isinstance(parameter, list) and len(parameter) == 2, "Cryptography API parameter is invalid")
			arg, kind = parameter
			require(isinstance(arg, str) and IDENTIFIER.fullmatch(arg), "Cryptography API parameter name is invalid")
			validate_unique(arg, parameter_names, "Cryptography API parameter")
			require(isinstance(kind, str) and kind in {"&Matrix", "usize", "u32", "MlDsaParameters", "std::ops::Range<usize>"}, "Cryptography API parameter type is invalid")
			if kind == "&Matrix": inputs.append("matrix")
		require(isinstance(api["values"], dict) and all(isinstance(key, str) and IDENTIFIER.fullmatch(key) and isinstance(value, str) and value.strip() and ";" not in value for key, value in api["values"].items()), "Cryptography API values are invalid")
		require(isinstance(api["attributes"], dict), "Cryptography API attributes must be an object")
		if category == "hash":
			expected = [["input", "&Matrix"]] + ([["output_length", "usize"]] if name in {"shake128", "shake256"} or (name.startswith("poseidon2_") and not name.endswith("_permute")) else [])
			if name == "merkle_proofs": expected = [["leaves", "&Matrix"], ["indices", "&Matrix"]]
			if name == "verify_merkle_proofs": expected = [[arg, "&Matrix"] for arg in ("leaves", "proofs", "roots")] + [["leaf_count", "u32"]]
			if name in {"merkle_proofs", "verify_merkle_proofs"}:
				keys = ("header_bytes", "hash_bytes", "max_depth") if name == "verify_merkle_proofs" else ("header_bytes", "hash_bytes")
				require(api["values"] == {key: str(schema["proof_layout"][key]) for key in keys}, "Merkle API wire layout differs")
			if name in FIXED_HASH_HELPERS:
				require(api["values"] == {"digest_bytes": str(FIXED_HASH_HELPERS[name][2])}
					and api["template"] == "fixed_digest" and api["contracts"] == [name],
					"Cryptography fixed digest API extent or contract differs")
		else:
			expected = [["parameters", "MlDsaParameters"]] + ([["algorithm", "u32"]] if "algorithm" in api["attributes"] else []) + [[arg, "&Matrix"] for arg in ("messages", "msg_offsets", "msg_lengths", "signatures", "public_keys")] + [["context", "std::ops::Range<usize>"]]
		require(api["parameters"] == expected, "Cryptography API signature differs from template contract")
		owned = api["contracts"]
		require(isinstance(owned, list) and owned and all(isinstance(c, str) and c in contracts for c in owned) and len(set(owned)) == len(owned), "Cryptography API contracts are invalid")
		require(not covered.intersection(owned), "Cryptography API contract has duplicate owner")
		covered.update(owned)
		require(all(contracts[c]["input_kinds"] == inputs for c in owned), "Cryptography API semantic inputs differ from signature")
		attributes = contracts[owned[0]]["attributes"]
		require(all(contracts[c]["attributes"] == attributes for c in owned), "Cryptography API route attributes differ")
		require(isinstance(api["attributes"], dict) and set(api["attributes"]) == {n for n, _ in attributes} and all(isinstance(v, str) and v.strip() and ";" not in v for v in api["attributes"].values()), "Cryptography API attribute sources differ from contract")
		stages = api["stages"]
		require(isinstance(stages, list) and stages, "Cryptography API stages must be nonempty")
		used = set()
		for stage in stages:
			require(isinstance(stage, dict) and set(stage) == {"kernel", "bindings", "push_constants"} and isinstance(stage["kernel"], str) and stage["kernel"] in kernels, "Cryptography API stage is invalid")
			kernel = kernels[stage["kernel"]]
			used.add(stage["kernel"])
			require(set([kernel["semantic_operation"]] + kernel.get("auxiliary_operations", [])).intersection(owned), "Cryptography API kernel belongs to another operation")
			buffers, pushes = wire_parts(kernel)
			require(isinstance(stage["bindings"], dict) and set(stage["bindings"]) == {n for n, _ in buffers}, "Cryptography API bindings differ from ABI")
			for access in stage["bindings"].values():
				require(isinstance(access, list) and len(access) == 2 and access[0] in {"read", "write"} and isinstance(access[1], str) and IDENTIFIER.fullmatch(access[1]), "Cryptography API binding expression is invalid")
			require(isinstance(stage["push_constants"], dict) and set(stage["push_constants"]) == {n for n, _ in pushes} and all(kind == "uint32" for _, kind in pushes), "Cryptography API pushes differ from ABI")
			require(all(isinstance(v, str) and v.strip() and ";" not in v for v in stage["push_constants"].values()), "Cryptography API push expression is invalid")
		require(used == {k["kernel_id"] for k in schema["kernels"] if set([k["semantic_operation"]] + k.get("auxiliary_operations", [])).intersection(owned)}, "Cryptography API kernel coverage differs from schema")
		if category == "verify":
			require(api["visibility"] == ("pub(crate)" if "algorithm" in api["attributes"] else "pub"), "Cryptography verifier visibility differs from admission")
			require(len(stages) == 3 and [kernels[s["kernel"]]["semantic_operation"] for s in stages] == owned, "Cryptography verifier route coverage differs")
			require({kernels[s["kernel"]]["parameter_set"] for s in stages} == {44, 65, 87}, "Cryptography verifier parameter coverage differs")
			require(all(kernels[s["kernel"]]["push_fields"] == kernels[stages[0]["kernel"]]["push_fields"] and s["bindings"] == stages[0]["bindings"] and s["push_constants"] == stages[0]["push_constants"] for s in stages), "Cryptography verifier route ABI differs")
		require(isinstance(api["template"], str) and api["template"] in cryptography_templates(category), "Cryptography API template is unknown")
		cryptography_body(schema, api, category)
	require(covered == set(contracts), "Cryptography API contract coverage differs from schema")
	require({api["template"] for api in apis} == set(cryptography_templates(category)), "Cryptography template coverage differs from schema")


def generate_cryptography_api(schema: dict[str, Any], schema_hash: str, category: str) -> str:
	module = (FN_ROOT / "template/crypto" / f"{category}_module.rs.in").read_text(encoding="utf-8")
	require(module.count("{{items}}") == 1, "Cryptography module requires exactly one item insertion")
	items = []
	for api in schema["rust_api"]:
		signature = api_signature("crypto", api, f"Cryptography {api['name']}")
		docs = "\n".join("///" + (" " + line if line else "") for line in api["doc"])
		annotations = "".join(f"#[{line}]\n" for line in api["annotations"])
		items.append(f"{docs}\n{annotations}{signature_header(signature)} {{\n" + cryptography_body(schema, api, category) + "\n}\n")
	return banner(schema_hash, "//", "crypto_hash.json" if category == "hash" else "crypto_pqc.json") + module.replace("{{items}}", "\n\n".join(items))
