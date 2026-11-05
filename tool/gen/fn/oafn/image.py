"""Image schema validation and complete-body emitters."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .model import validate_physical_write, validate_semantic_attributes, validate_unique
from .wire import generate_type, render_dispatch_body, validate_dispatch_generation
from .signature import operation_signature, signature_header
from .util import IDENTIFIER, banner, require

def validate_image_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Image schema_version must be 1")
	require(schema.get("family") == "image", "Image family must be image")
	require(schema.get("domain") == "image", "Image domain must be image")
	require(schema.get("dtype") == "f32", "Image dtype must be f32")
	contracts = schema.get("contracts")
	expected_contracts = {
		"resize": ["image"],
		"crop": ["image"],
		"flip": ["image"],
		"rotate": ["image"],
		"pad": ["image"],
		"center_crop": ["image"],
		"remap": ["image", "matrix"],
		"warp_affine": ["image", "matrix"],
		"warp_perspective": ["image", "matrix"],
		"threshold_binary": ["image"],
		"threshold_binary_inv": ["image"],
		"threshold_truncate": ["image"],
		"threshold_to_zero": ["image"],
		"threshold_to_zero_inv": ["image"],
		"in_range": ["image"],
		"clamp": ["image"],
		"invert": ["image"],
		"brightness_contrast": ["image"],
		"gamma_contrast": ["image"],
		"solarize": ["image"],
		"posterize": ["image"],
		"grayscale": ["image"],
		"alpha_blend": ["image", "image"],
		"composite": ["image", "image", "image"],
		"erase": ["image"],
		"color_twist": ["image", "matrix"],
		"channel_reorder": ["image"],
		"gaussian_noise": ["image"],
		"salt_pepper_noise": ["image"],
		"convolve_2d": ["image", "matrix"],
		"separable_convolve_2d": ["image", "matrix", "matrix"],
		"average_blur": ["image"],
		"sobel": ["image"],
		"scharr": ["image"],
		"laplacian": ["image"],
		"erode": ["image"],
		"dilate": ["image"],
		"morphology_open": ["image"],
		"morphology_close": ["image"],
		"morphology_gradient": ["image"],
		"gaussian_blur": ["image"],
		"sharpen": ["image"],
		"median_blur": ["image"],
		"bilateral_filter": ["image"],
		"unsharp_mask": ["image"],
		"morphology_top_hat": ["image"],
		"morphology_black_hat": ["image"],
		"adaptive_threshold_mean": ["image"],
		"adaptive_threshold_gaussian": ["image"],
		"normalize": ["image"],
		"convert_color": ["image"],
		"resize_normalize": ["image"],
		"segmentation_overlay": ["image", "matrix", "matrix"],
	}
	require(
		isinstance(contracts, list)
		and [contract.get("name") for contract in contracts] == list(expected_contracts),
		"Image contracts must contain the complete geometric surface in canonical order",
	)
	contract_names: set[str] = set()
	for index, contract in enumerate(contracts):
		where = f"Image contracts[{index}]"
		require(isinstance(contract, dict), f"{where} must be an object")
		require(
			(set(contract) - {"python", "donor_op", "control_flow", "dnn"})
			== {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes", "rust"},
			f"{where} fields are incomplete or unknown",
		)
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		require(
			contract["input_kinds"] == expected_contracts[name],
			f"{where}.input_kinds are invalid",
		)
		require(contract["output_kinds"] == ["image"], f"{where}.output_kinds must contain one Image")
		require(contract["shape_rule"] == "explicit", f"{where}.shape_rule must be explicit")
		require(contract["dtype_rule"] in {"f32", "explicit"}, f"{where}.dtype_rule is invalid")
		validate_semantic_attributes(contract["attributes"], where)
	kernels = schema.get("kernels")
	require(isinstance(kernels, list) and kernels, "Image kernels must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernel_ids: set[str] = set()
	for index, kernel in enumerate(kernels):
		where = f"Image kernels[{index}]"
		require(isinstance(kernel, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernel_ids)):
			value = kernel.get(key)
			require(isinstance(value, str) and value, f"{where}.{key} must be non-empty")
			validate_unique(value, seen, f"{where}.{key}")
		require(isinstance(kernel.get("source"), str) and kernel["source"], f"{where}.source must be non-empty")
		stable_id = kernel.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id is invalid")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		require(kernel.get("semantic_operation") in contract_names, f"{where}.semantic_operation is unknown")
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
			isinstance(tile, list)
			and len(tile) == 3
			and all(isinstance(value, int) and value > 0 for value in tile),
			f"{where}.dispatch_tile_size is invalid",
		)
		validate_physical_write(kernel.get("physical_write"), where, required=True)
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
	validate_image_generation(schema)


IMAGE_CATEGORIES = ("pixel", "filter", "geometric", "color")


def image_templates(category: str) -> dict[str, str]:
	path = FN_ROOT / "template/image" / f"{category}.rs.in"
	sections = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", path.read_text(encoding="utf-8"), re.S)
	require(len(sections) == len({name for name, _ in sections}), f"duplicate Image template section: {category}")
	return dict(sections)


def validate_image_generation(schema: dict[str, Any]) -> None:
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	validate_dispatch_generation(
		schema, set(IMAGE_CATEGORIES),
		{"&Image", "&Matrix", "f32", "u32", "u64", "bool", "[u32; 4]", "BorderMode", "InterpolationMode", "ImageFormat", "crate::ImageFormat", "NormalizationConfig"},
		{op["name"]: "Image" for op in schema["contracts"]},
		{"float": {"float"}, "unsigned_integer": {"unsigned", "unsigned64", "inline_unsigned"}, "boolean": {"boolean"}, "enum": {"enumeration", "inline_enum"}},
	)
	selectors = [record for record in schema["types"] if record["name"] == "InterpolationMode"]
	require(len(selectors) == 1 and selectors[0]["kind"] == "enum" and all("kernel" in variant for variant in selectors[0]["variants"]), "Image resize requires its interpolation selector")
	selector = selectors[0]
	resize = next(op for op in schema["contracts"] if op["name"] == "resize")
	require({variant["kernel"] for variant in selector["variants"]} == {"KernelId::" + name for stage in resize["rust"]["dispatches"] for name in stage["kernels"]}, "Image resize selector differs from dispatch candidates")
	for category in IMAGE_CATEGORIES:
		expected = {op["name"] for op in schema["contracts"] if op["rust"]["category"] == category}
		require(set(image_templates(category)) == expected, f"Image {category} template coverage differs from schema")
	for op in schema["contracts"]:
		render_image_body(op, kernels)


def render_image_body(op: dict[str, Any], kernels: dict[str, Any]) -> str:
	return render_dispatch_body(op, kernels, image_templates(op["rust"]["category"])[op["name"]], "image")


def generate_image_api(schema: dict[str, Any], schema_hash: str, category: str) -> str:
	module = (FN_ROOT / "template/image" / f"{category}_module.rs.in").read_text(encoding="utf-8")
	require(module.count("{{items}}") == 1, f"Image {category} module requires exactly one item insertion")
	items = []
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	for op in schema["contracts"]:
		rust = op["rust"]
		if rust["category"] != category:
			continue
		signature = operation_signature("image", op, f"Image {op['name']}")
		require(signature is not None, f"Image {op['name']} has no public signature")
		items.extend(generate_type(record) for record in schema["types"] if record["before"] == op["name"])
		docs = "\n".join("///" + (" " + line if line else "") for line in rust["doc"])
		annotations = "".join(f"#[{annotation}]\n" for annotation in rust["annotations"])
		items.append(f"{docs}\n{annotations}{signature_header(signature)} {{\n" + render_image_body(op, kernels) + "\n}\n")
	return banner(schema_hash, "//", "image.json") + module.replace("{{items}}", "\n\n".join(items))
