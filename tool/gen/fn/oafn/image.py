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

TILE_PROGRAM = re.compile(r"[a-z][a-z0-9_]*(\((true|false|0x[0-9a-fA-F]+|[0-9]+)(,\s*(true|false|0x[0-9a-fA-F]+|[0-9]+))*\))?")

def validate_image_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Image schema_version must be 1")
	require(schema.get("family") == "image", "Image family must be image")
	require(schema.get("domain") == "image", "Image domain must be image")
	require(schema.get("dtype") == "f32", "Image dtype must be f32")
	contracts = schema.get("contracts")
	expected_contracts = {
		"resize": (["image"], ["image"]),
		"crop": (["image"], ["image"]),
		"flip": (["image"], ["image"]),
		"rotate": (["image"], ["image"]),
		"pad": (["image"], ["image"]),
		"center_crop": (["image"], ["image"]),
		"remap": (["image", "matrix"], ["image"]),
		"warp_affine": (["image", "matrix"], ["image"]),
		"warp_perspective": (["image", "matrix"], ["image"]),
		"threshold_binary": (["image"], ["image"]),
		"threshold_binary_inv": (["image"], ["image"]),
		"threshold_truncate": (["image"], ["image"]),
		"threshold_to_zero": (["image"], ["image"]),
		"threshold_to_zero_inv": (["image"], ["image"]),
		"in_range": (["image"], ["image"]),
		"clamp": (["image"], ["image"]),
		"invert": (["image"], ["image"]),
		"brightness_contrast": (["image"], ["image"]),
		"gamma_contrast": (["image"], ["image"]),
		"solarize": (["image"], ["image"]),
		"posterize": (["image"], ["image"]),
		"grayscale": (["image"], ["image"]),
		"alpha_blend": (["image", "image"], ["image"]),
		"composite": (["image", "image", "image"], ["image"]),
		"erase": (["image"], ["image"]),
		"color_twist": (["image", "matrix"], ["image"]),
		"channel_reorder": (["image"], ["image"]),
		"gaussian_noise": (["image"], ["image"]),
		"salt_pepper_noise": (["image"], ["image"]),
		"convolve_2d": (["image", "matrix"], ["image"]),
		"separable_convolve_2d": (["image", "matrix", "matrix"], ["image"]),
		"average_blur": (["image"], ["image"]),
		"sobel": (["image"], ["image"]),
		"scharr": (["image"], ["image"]),
		"laplacian": (["image"], ["image"]),
		"erode": (["image"], ["image"]),
		"dilate": (["image"], ["image"]),
		"morphology_open": (["image"], ["image"]),
		"morphology_close": (["image"], ["image"]),
		"morphology_gradient": (["image"], ["image"]),
		"gaussian_blur": (["image"], ["image"]),
		"sharpen": (["image"], ["image"]),
		"median_blur": (["image"], ["image"]),
		"bilateral_filter": (["image"], ["image"]),
		"unsharp_mask": (["image"], ["image"]),
		"morphology_top_hat": (["image"], ["image"]),
		"morphology_black_hat": (["image"], ["image"]),
		"adaptive_threshold_mean": (["image"], ["image"]),
		"adaptive_threshold_gaussian": (["image"], ["image"]),
		"normalize": (["image"], ["image"]),
		"convert_color": (["image"], ["image"]),
		"resize_normalize": (["image"], ["image"]),
		"segmentation_overlay": (["image", "matrix", "matrix"], ["image"]),
		"calc_hist": (["image"], ["matrix"]),
		"equalize_hist": (["image"], ["image"]),
		"compare_hist": (["matrix", "matrix"], ["matrix"]),
		"pyr_down": (["image"], ["image"]),
		"pyr_up": (["image"], ["image"]),
		"threshold_auto": (["image"], ["image", "matrix"]),
		"integral": (["image"], ["matrix"]),
		"cvt_color": (["image"], ["image"]),
		"apply_color_map": (["image"], ["image"]),
		"calc_back_project": (["image", "matrix"], ["image"]),
		"clahe": (["image"], ["image"]),
		"accumulate": (["image", "image"], ["image"]),
		"accumulate_weighted": (["image", "image"], ["image"]),
		"moments": (["image"], ["matrix"]),
		"psnr": (["image", "image"], ["matrix"]),
		"ssim": (["image", "image"], ["matrix"]),
		"match_template": (["image", "image"], ["matrix"]),
		"warp_polar": (["image"], ["image"]),
		"canny": (["image"], ["image"]),
		"connected_components": (["image"], ["matrix"]),
		"distance_transform": (["image"], ["matrix"]),
		"hough_lines": (["image"], ["matrix"]),
		"hough_lines_p": (["image"], ["matrix"]),
		"find_contours": (["image"], ["matrix", "matrix"]),
		"flood_fill": (["image"], ["image"]),
		"draw_line": (["image"], ["image"]),
		"draw_rectangle": (["image"], ["image"]),
		"draw_circle": (["image"], ["image"]),
		"fill_polygon": (["image", "matrix"], ["image"]),
		"corner_score": (["image"], ["image"]),
		"corner_fast": (["image"], ["image"]),
		"bf_match": (["matrix", "matrix"], ["matrix"]),
		"good_features_to_track": (["image"], ["matrix", "matrix"]),
		"orb": (["image"], ["matrix", "matrix", "matrix"]),
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
			(set(contract) - {"python", "donor_op", "control_flow", "dnn", "fusion_class", "region_read", "tile_program"})
			== {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes", "rust"},
			f"{where} fields are incomplete or unknown",
		)
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		expected_inputs, expected_outputs = expected_contracts[name]
		require(
			contract["input_kinds"] == expected_inputs,
			f"{where}.input_kinds are invalid",
		)
		require(contract["output_kinds"] == expected_outputs, f"{where}.output_kinds are invalid")
		require(contract["shape_rule"] == "explicit", f"{where}.shape_rule must be explicit")
		require(contract["dtype_rule"] in {"f32", "explicit"}, f"{where}.dtype_rule is invalid")
		validate_semantic_attributes(contract["attributes"], where)
		require(
			contract.get("region_read") in {"elementwise", "gather", "bordered"},
			f"{where}.region_read must declare the input read pattern",
		)
		require(
			contract.get("tile_program") is None or TILE_PROGRAM.fullmatch(contract["tile_program"]) is not None,
			f"{where}.tile_program must name a BlockProgram factory call",
		)
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


IMAGE_CATEGORIES = ("pixel", "filter", "geometric", "color", "statistics")


def image_templates(category: str) -> dict[str, str]:
	path = FN_ROOT / "template/image" / f"{category}.rs.in"
	sections = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", path.read_text(encoding="utf-8"), re.S)
	require(len(sections) == len({name for name, _ in sections}), f"duplicate Image template section: {category}")
	return dict(sections)


def validate_image_generation(schema: dict[str, Any]) -> None:
	kernels = {kernel["kernel_id"]: kernel for kernel in schema["kernels"]}
	return_types = {
		"calc_hist": "Matrix",
		"compare_hist": "Matrix",
		"integral": "Matrix",
		"moments": "Matrix",
		"psnr": "Matrix",
		"ssim": "Matrix",
		"match_template": "Matrix",
		"connected_components": "Matrix",
		"distance_transform": "Matrix",
		"hough_lines": "Matrix",
		"hough_lines_p": "Matrix",
		"find_contours": "(Matrix, Matrix)",
		"flood_fill": "Image",
		"draw_line": "Image",
		"draw_rectangle": "Image",
		"draw_circle": "Image",
		"fill_polygon": "Image",
		"corner_score": "Image",
		"corner_fast": "Image",
		"bf_match": "Matrix",
		"good_features_to_track": "(Matrix, Matrix)",
		"orb": "(Matrix, Matrix, Matrix)",
		"threshold_auto": "(Image, Matrix)",
	}
	validate_dispatch_generation(
		schema, set(IMAGE_CATEGORIES),
		{"&Image", "&Matrix", "f32", "u32", "u64", "bool", "[u32; 4]", "BorderMode", "InterpolationMode", "ImageFormat", "crate::ImageFormat", "NormalizationConfig", "HistogramComparison", "AutoThreshold", "ColorConversion", "ColorMap", "MatchTemplate", "PolarMode", "Connectivity", "DistanceType", "CornerScore"},
		{op["name"]: return_types.get(op["name"], "Image") for op in schema["contracts"]},
		{"float": {"float"}, "unsigned_integer": {"unsigned", "unsigned64", "inline_unsigned"}, "boolean": {"boolean"}, "enum": {"enumeration", "inline_enum"}},
		read_write=True,
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
