"""
Schema validation and loading for OA type generation.
"""
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnumValue:
	name: str
	value: int | None = None
	comment: str = ""
	serializedName: str = ""


@dataclass
class EnumDef:
	name: str
	underlyingType: str
	values: list[EnumValue]
	generateTostring: bool = False
	generateFromstring: bool = False


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
	generateSerialize: bool = False
	generateValidate: bool = False


@dataclass
class TypeSchema:
	domain: str
	namespace: str
	enums: list[EnumDef]
	structs: list[StructDef]


def loadSchema(schemaPath: Path) -> TypeSchema:
	"""Load and parse a TOML type schema."""
	with schemaPath.open("rb") as f:
		data = tomllib.load(f)

	domain = data.get("domain", "core")
	namespace = data.get("namespace", "oa")

	enums = []
	for enumData in data.get("enums", []):
		values = [
		 EnumValue(
		  name=v.get("name"),
		  value=v.get("value"),
		  comment=v.get("comment", ""),
		  serializedName=v.get("serialized_name", ""),
		 )
		 for v in enumData.get("values", [])
		]
		enums.append(EnumDef(
		 name=enumData.get("name"),
		 underlyingType=enumData.get("underlying_type", "U8"),
		 values=values,
		 generateTostring=enumData.get("generate_tostring", False),
		 generateFromstring=enumData.get("generate_fromstring", False),
		))

	structs = []
	for structData in data.get("structs", []):
		fields = [
		 StructField(
		  name=f.get("name"),
		  type=f.get("type"),
		  default=f.get("default", ""),
		  comment=f.get("comment", "")
		 )
		 for f in structData.get("fields", [])
		]
		structs.append(StructDef(
		 name=structData.get("name"),
		 fields=fields,
		 generateSerialize=structData.get("generate_serialize", False),
		 generateValidate=structData.get("generate_validate", False),
		))

	return TypeSchema(
	 domain=domain,
	 namespace=namespace,
	 enums=enums,
	 structs=structs,
	)


def validateSchema(schema: TypeSchema) -> list[str]:
	"""Validate a schema and return list of errors (empty if valid)."""
	errors = []

	for enum in schema.enums:
		if not enum.name:
			errors.append("Enum missing name")
		if not enum.underlyingType:
			errors.append(f"Enum {enum.name} missing underlying_type")
		for val in enum.values:
			if not val.name:
				errors.append(f"Enum {enum.name} has value without name")

	for struct in schema.structs:
		if not struct.name:
			errors.append("Struct missing name")
		if struct.generateSerialize:
			errors.append(
			 f"Struct {struct.name} requests unsupported generated serialization"
			)
		if struct.generateValidate:
			errors.append(
			 f"Struct {struct.name} requests unsupported generated validation"
			)
		for field in struct.fields:
			if not field.name:
				errors.append(f"Struct {struct.name} has field without name")
			if not field.type:
				errors.append(f"Struct {struct.name} field {field.name} missing type")

	return errors
