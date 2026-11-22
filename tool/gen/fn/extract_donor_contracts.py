#!/usr/bin/env python3
"""Extract donor OA operation-contract identities into a checked-in snapshot.

Reads the donor C++ repository's ``tools/gen/fn/schema/**.toml`` operation
schemas and emits ``donor_contracts.json``: for every donor op carrying a
``[ops.contract]`` record, the qualified ``Fn*`` name, the canonical
``oa.operation_contract.v1`` payload, the FNV-1a64 contract hash, and the
attribute-signature hash exactly as the donor generator computes them
(``oacpp/tools/gen/fn/generate.py``).

The snapshot is the identity oracle for ``donor_op`` links in the Rust
schemas. Re-run this script when the donor operation surface changes:

    python3 tool/gen/fn/extract_donor_contracts.py --donor ../oacpp --check

The Rust generator validates the snapshot against its own recompute before
use; a stale or hand-edited snapshot fails generation rather than silently
labeling a contract as donor-exact.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

DONOR_DOMAIN_NAMESPACE = {
	"core": "oa::FnMatrix",
	"ml": "oa::FnMatrix",
	"audio": "oa::FnAudio",
	"vision": "oa::FnImage",
	"ui": "oa::FnMatrix",
	"crypto": "oa::FnHash",
	"data": "oa::FnDataset",
	"render": "oa::FnCamera",
	"runtime": "oa::FnTexture",
}

# oacpp/tools/gen/fn/schema.py _SCALAR_ATTRIBUTE_KINDS.
SCALAR_ATTRIBUTE_KINDS = {
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

# oacpp/tools/gen/fn/generate.py _ATTRIBUTE_KIND_BYTE.
ATTRIBUTE_KIND_BYTE = {
	"boolean": 1,
	"signed_integer": 2,
	"unsigned_integer": 3,
	"float": 4,
	"string": 5,
	"shape": 6,
	"enum": 7,
}

SNAPSHOT_SCHEMA = "oa.donor_operation_contracts.v1"


def fail(message: str) -> None:
	print(f"extract_donor_contracts: {message}", file=sys.stderr)
	raise SystemExit(1)


def fnv1a64(data: bytes) -> int:
	"""Donor ``_fnv1a64`` — offset basis/prime and 64-bit wrap must not change."""
	value = 14695981039346656037
	for byte in data:
		value ^= byte
		value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
	return value


def infer_domain(path: Path) -> str:
	"""Donor ``inferDomain`` — directory name if known, else ``core``."""
	name = path.parent.name.lower()
	return name if name in DONOR_DOMAIN_NAMESPACE else "core"


def semantic_attribute_specs(op: dict) -> list[dict]:
	"""Donor ``semanticAttributeSpecs`` — scalar param then contract attributes."""
	attributes: list[dict] = []
	scalar = op.get("scalar_param")
	if scalar is not None:
		if not isinstance(scalar, dict):
			fail(f"{op.get('name', '<unnamed>')}: scalar_param must be a table")
		kind = SCALAR_ATTRIBUTE_KINDS.get(scalar.get("type"))
		if kind is None:
			fail(f"{op.get('name', '<unnamed>')}: scalar_param.type has no semantic kind")
		attributes.append({"name": scalar["name"], "kind": kind})
	contract = op.get("contract", {})
	attributes.extend(
		{"name": attribute["name"], "kind": attribute["kind"]}
		for attribute in contract.get("attributes", [])
	)
	return attributes


def attribute_signature_hash(attributes: list[dict]) -> int:
	"""Donor ``operationAttributeSignatureHash``."""
	if not attributes:
		return 0
	payload = bytearray()
	for attribute in attributes:
		kind = attribute["kind"]
		if kind not in ATTRIBUTE_KIND_BYTE:
			fail(f"attribute {attribute['name']!r}: unknown kind {kind!r}")
		payload.append(ATTRIBUTE_KIND_BYTE[kind])
		payload.extend(attribute["name"].encode("utf-8"))
		payload.append(0)
	return fnv1a64(bytes(payload))


def contract_payload(namespace: str, op: dict, attributes: list[dict]) -> dict:
	"""Donor ``operationContractHash`` payload — field set must not change."""
	contract = dict(op["contract"])
	contract.pop("attributes", None)
	payload = {
		"schema": "oa.operation_contract.v1",
		"namespace": namespace,
		"name": op["name"],
		"api_params": op.get("api_params", []),
		"api_return": op.get("api_return", "oa::Matrix"),
		"contract": contract,
	}
	if attributes:
		payload["attributes"] = attributes
	return payload


def contract_hash(payload: dict) -> int:
	canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
	return fnv1a64(canonical.encode("utf-8"))


def extract(schema_root: Path) -> list[dict]:
	if not schema_root.is_dir():
		fail(f"donor schema root {schema_root} does not exist")
	records: list[dict] = []
	seen: set[str] = set()
	for path in sorted(schema_root.rglob("*.toml")):
		data = tomllib.loads(path.read_text(encoding="utf-8"))
		namespace = data.get(
			"namespace", DONOR_DOMAIN_NAMESPACE.get(infer_domain(path), "oa::FnMatrix")
		)
		for op in data.get("ops", []):
			if "contract" not in op:
				continue
			name = op.get("name")
			if not isinstance(name, str) or not name:
				fail(f"{path}: contract op without a name")
			qualified = f"{namespace}::{name}"
			if qualified in seen:
				fail(f"duplicate donor contract identity {qualified}")
			seen.add(qualified)
			attributes = semantic_attribute_specs(op)
			payload = contract_payload(namespace, op, attributes)
			records.append(
				{
					"name": qualified,
					"namespace": namespace,
					"operation": name,
					"schema": path.relative_to(schema_root).as_posix(),
					"payload": payload,
					"hash": contract_hash(payload),
					"attribute_signature_hash": attribute_signature_hash(attributes),
				}
			)
	records.sort(key=lambda record: record["name"])
	return records


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--donor",
		type=Path,
		default=Path(__file__).resolve().parents[4] / "oacpp",
		help="donor C++ repository root (default: sibling oacpp checkout)",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=Path(__file__).resolve().parent / "donor_contracts.json",
	)
	parser.add_argument(
		"--check",
		action="store_true",
		help="verify the checked-in snapshot against the donor tree",
	)
	args = parser.parse_args()

	schema_root = args.donor / "tools" / "gen" / "fn" / "schema"
	records = extract(schema_root)
	document = {
		"schema": SNAPSHOT_SCHEMA,
		"generator": "tool/gen/fn/extract_donor_contracts.py",
		"source": "oacpp tools/gen/fn/schema",
		"records": records,
	}
	text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
	if args.check:
		if args.out.read_text(encoding="utf-8") != text:
			fail(f"{args.out} is stale; re-run extract_donor_contracts.py")
		print(f"donor_contracts.json is current ({len(records)} records)")
		return
	args.out.write_text(text, encoding="utf-8")
	print(f"wrote {args.out}: {len(records)} donor contract records")


if __name__ == "__main__":
	main()
