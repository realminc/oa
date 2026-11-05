"""Standalone type-record schemas emitted as ``<domain>/type.gen.rs``.

``<domain>_types`` schemas carry unanchored public records — records that
parameterize handwritten values and sessions rather than belonging to one
generated operation. The emitter writes ``src/rs/<domain>/type.gen.rs``;
facades declare ``mod types`` with ``#[path]`` and re-export the names so
public paths stay ``crate::<domain>::<Name>``, matching the oacpp/vk
``type.gen`` convention.

Taxonomy is enforced by :func:`oafn.wire.validate_type_class`; the owning
contract is the ``oaTypeRegistry`` architecture document.
"""

from __future__ import annotations

from typing import Any

from .model import validate_unique
from .util import IDENTIFIER, banner, require
from .wire import (
	generate_type,
	validate_type_class,
	validate_type_record_bodies,
)

_TYPE_RECORD_KEYS = {
	"class", "name", "kind", "visibility", "derive", "doc",
	"fields", "variants", "defaults", "default_variant",
}


def validate_types_schema(schema: dict[str, Any]) -> None:
	"""Validate a standalone ``<domain>_types`` record schema."""
	require(schema.get("schema_version") == 1, "Types schema_version must be 1")
	domain = schema.get("domain")
	require(isinstance(domain, str) and IDENTIFIER.fullmatch(domain), "Types domain must be an identifier")
	require(schema.get("family") == f"{domain}_types", f"Types family must be {domain}_types")
	imports = schema.get("imports")
	require(isinstance(imports, list) and all(
		isinstance(line, str) and line.startswith("use ") and line.endswith(";") and "\n" not in line
		for line in imports), "Types imports must be complete use statements")
	types = schema.get("types")
	require(isinstance(types, list) and types, "Types must be a nonempty array")
	names: set[str] = set()
	for record in types:
		where = f"{domain.title()} types"
		require(isinstance(record, dict) and set(record) in (_TYPE_RECORD_KEYS, _TYPE_RECORD_KEYS | {"mappings"}), f"{where} record is incomplete or unknown")
		name = record["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name), f"{where} name is invalid")
		validate_unique(name, names, where)
		require(isinstance(record["kind"], str) and record["kind"] in {"struct", "enum"} and isinstance(record["visibility"], str) and record["visibility"] in {"pub", ""}, f"{where} {name} kind/visibility is invalid")
		validate_type_class(name, record["class"], record["kind"], record["visibility"] == "pub", f"{where} {name}")
		require(isinstance(record["doc"], str) and "\n" not in record["doc"], f"{where} {name} docs are invalid")
		derives = record["derive"]
		require(isinstance(derives, list) and all(
			isinstance(trait, str) and trait in {"Clone", "Copy", "Debug", "Default", "PartialEq", "Eq"}
			for trait in derives) and len(set(derives)) == len(derives), f"{where} {name} derives are invalid")
	validate_type_record_bodies(types, names, domain.title(), permissive_field_types=True)


def generate_types_file(schema: dict[str, Any], schema_hash: str, schema_name: str) -> str:
	"""Emit one ``type.gen.rs`` module from a ``<domain>_types`` schema."""
	text = banner(schema_hash, "//", schema_name)
	imports = schema.get("imports") or []
	if imports:
		text += "\n".join(imports) + "\n"
	text += "\n".join(generate_type(record) + "\n" for record in schema["types"])
	return text
