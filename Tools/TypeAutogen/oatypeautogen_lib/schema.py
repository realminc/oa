"""
Schema validation and loading for OaTypeAutogen.
"""
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnumValue:
	name: str
	value: int | None = None
	comment: str = ""


@dataclass
class EnumDef:
	name: str
	underlying_type: str
	values: list[EnumValue]
	generate_tostring: bool = False
	generate_fromstring: bool = False


@dataclass
class StructField:
	name: str
	type: str
	default: str = ""
	comment: str = ""


@dataclass
class StructDef:
	name: str
	fields: list[StructField]
	generate_serialize: bool = False
	generate_validate: bool = False


@dataclass
class TypeSchema:
	domain: str
	namespace: str
	enums: list[EnumDef]
	structs: list[StructDef]


def load_schema(schema_path: Path) -> TypeSchema:
	"""Load and parse a TOML type schema."""
	with schema_path.open("rb") as f:
		data = tomllib.load(f)
	
	domain = data.get("domain", "Core")
	namespace = data.get("namespace", "Oa")
	
	enums = []
	for enum_data in data.get("enums", []):
		values = [
			EnumValue(
				name=v.get("name"),
				value=v.get("value"),
				comment=v.get("comment", "")
			)
			for v in enum_data.get("values", [])
		]
		enums.append(EnumDef(
			name=enum_data.get("name"),
			underlying_type=enum_data.get("underlying_type", "OaU8"),
			values=values,
			generate_tostring=enum_data.get("generate_tostring", False),
			generate_fromstring=enum_data.get("generate_fromstring", False),
		))
	
	structs = []
	for struct_data in data.get("structs", []):
		fields = [
			StructField(
				name=f.get("name"),
				type=f.get("type"),
				default=f.get("default", ""),
				comment=f.get("comment", "")
			)
			for f in struct_data.get("fields", [])
		]
		structs.append(StructDef(
			name=struct_data.get("name"),
			fields=fields,
			generate_serialize=struct_data.get("generate_serialize", False),
			generate_validate=struct_data.get("generate_validate", False),
		))
	
	return TypeSchema(
		domain=domain,
		namespace=namespace,
		enums=enums,
		structs=structs,
	)


def validate_schema(schema: TypeSchema) -> list[str]:
	"""Validate a schema and return list of errors (empty if valid)."""
	errors = []
	
	for enum in schema.enums:
		if not enum.name:
			errors.append("Enum missing name")
		if not enum.underlying_type:
			errors.append(f"Enum {enum.name} missing underlying_type")
		for val in enum.values:
			if not val.name:
				errors.append(f"Enum {enum.name} has value without name")
	
	for struct in schema.structs:
		if not struct.name:
			errors.append("Struct missing name")
		for field in struct.fields:
			if not field.name:
				errors.append(f"Struct {struct.name} has field without name")
			if not field.type:
				errors.append(f"Struct {struct.name} field {field.name} missing type")
	
	return errors
