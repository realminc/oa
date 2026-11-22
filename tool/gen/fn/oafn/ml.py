"""ML training and activation schemas: validation and autograd emitters."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .model import validate_dnn_role, validate_physical_write, validate_retired_stable_ids, validate_semantic_attributes, validate_unique
from .util import IDENTIFIER, banner, require, rust_const_name

def validate_activation_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Activation schema_version must be 1")
	require(schema.get("family") == "ml_activation", "Activation family must be ml_activation")
	require(schema.get("domain") == "ml", "Activation domain must be ml")
	require(schema.get("dtype") == "f32", "Activation dtype must be f32")
	operations = schema.get("operations")
	require(isinstance(operations, list) and operations, "Activation operations must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	VALID_KINDS = {"unary", "unary_backward", "unary_scalar", "unary_scalar_backward", "binary", "binary_backward", "silu_mul", "silu_mul_backward"}
	for index, operation in enumerate(operations):
		where = f"Activation operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernels)):
			value = operation.get(key)
			require(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None, f"{where}.{key} must be an identifier")
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = operation.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id must be a non-zero u16")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		kind = operation.get("kind")
		require(kind in VALID_KINDS, f"{where}.kind must be one of {VALID_KINDS}")
		require(operation.get("rust_family") in {"activation", "swiglu"},
			f"{where}.rust_family must name activation or swiglu")


def validate_ml_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "ML schema_version must be 1")
	require(schema.get("family") == "ml_training", "ML family must be ml_training")
	require(schema.get("domain") == "ml", "ML domain must be ml")
	require(schema.get("dtype") == "f32", "ML dtype must be f32")
	require(schema.get("workgroup_size") == [256, 1, 1], "ML workgroup_size must be [256, 1, 1]")
	operations = schema.get("operations")
	require(isinstance(operations, list) and operations, "ML operations must be non-empty")
	composites = schema.get("composite_contracts", [])
	require(isinstance(composites, list), "ML composite_contracts must be an array")
	validate_ml_port_provenance(schema.get("port_provenance"), operations + composites)
	contract_names = {
		operation.get("name")
		for operation in operations + composites
		if isinstance(operation, dict) and not operation.get("lowering_only", False)
	}
	contract_domains = {
		operation.get("name"): operation.get("semantic_domain")
		for operation in operations + composites
		if isinstance(operation, dict) and not operation.get("lowering_only", False)
	}
	seen_ids = validate_retired_stable_ids(
		schema.get("retired_stable_ids", []), "ML retired_stable_ids"
	)
	seen_autograd_nodes: set[str] = set()
	seen_names: set[str] = set()
	seen_kernels: set[str] = set()
	seen_sources: set[str] = set()
	for index, operation in enumerate(operations):
		where = f"ML operations[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		validate_physical_write(
			operation.get("physical_write"),
			where,
			required=operation.get("name")
			in {"layer_norm", "layer_norm_backward", "rms_norm", "rms_norm_backward"},
		)
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernels)):
			value = operation.get(key)
			require(
				isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None,
				f"{where}.{key} must be an identifier",
			)
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = operation.get("stable_id")
		require(
			isinstance(stable_id, int) and 0 < stable_id <= 65535,
			f"{where}.stable_id must be a non-zero u16",
		)
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		source = operation.get("source")
		require(isinstance(source, str) and source.endswith(".slang"), f"{where}.source must be a Slang path")
		validate_unique(source, seen_sources, f"{where}.source")
		variant = operation.get("variant")
		require(
			isinstance(variant, str) and IDENTIFIER.fullmatch(variant) is not None,
			f"{where}.variant must be an identifier",
		)
		operation_workgroup = operation.get("workgroup_size", schema["workgroup_size"])
		operation_dtype = operation.get("dtype", schema["dtype"])
		lowering_only = operation.get("lowering_only", False)
		require(isinstance(lowering_only, bool), f"{where}.lowering_only must be boolean")
		semantic_domain = operation.get("semantic_domain")
		if lowering_only:
			require(
				semantic_domain is None,
				f"{where} lowering-only kernel cannot own a semantic domain",
			)
		else:
			require(
				semantic_domain
				in {"ml::matrix", "ml::loss", "ml::optim", "ml::flow", "ml::advantage", "ml::rollout", "ml::replay", "ml::environment"},
				f"{where}.semantic_domain must name its public Rust operation owner",
			)
		semantic_operation = operation.get("semantic_operation")
		require(
			semantic_operation is None or semantic_operation in contract_names,
			f"{where}.semantic_operation must name a semantic ML operation",
		)
		require(
			lowering_only or semantic_operation is None,
			f"{where}.semantic_operation is only valid for a lowering-only kernel",
		)
		owner_domain = (
			contract_domains.get(semantic_operation) if lowering_only else semantic_domain
		)
		if owner_domain == "ml::loss":
			require(
				Path(source).name not in {"forward.slang", "backward.slang"},
				f"{where}.source must retain the loss name before _forward.slang or _backward.slang",
			)
		if owner_domain == "ml::optim" or source.startswith("src/slang/ml/optim/"):
			optimizer_family = (
				"grad_clip"
				if operation["name"].startswith("clip_grad_norm")
				else "adamw"
				if operation["name"].startswith("adamw")
				else "adam"
				if operation["name"] == "adam"
				else "sgd"
				if operation["name"].startswith("sgd")
				else "muon"
				if operation["name"].startswith("muon")
				else None
			)
			require(
				optimizer_family is not None
				and Path(source).parent == Path("src/slang/ml/optim") / optimizer_family,
				f"{where}.source must live under its optimizer-family directory",
			)
		require(operation_dtype in {"f32", "u8", "u32"}, f"{where}.dtype is unsupported")
		require(
			isinstance(operation_workgroup, list)
			and len(operation_workgroup) == 3
			and all(isinstance(value, int) and value > 0 for value in operation_workgroup)
			and math.prod(operation_workgroup) <= 1024,
			f"{where}.workgroup_size must contain three positive dimensions with at most 1024 threads",
		)
		differentiation = operation.get("differentiation")
		require(
			differentiation in {"none", "backward", "reverse"}
			or any(candidate.get("name") == differentiation for candidate in operations),
			f"{where}.differentiation must name an operation, reverse, backward, or none",
		)
		if lowering_only:
			require(differentiation == "none", f"{where} lowering-only kernel cannot differentiate")
			require("contract" not in operation, f"{where} lowering-only kernel cannot own a semantic contract")
			require("dnn" not in operation, f"{where} lowering-only kernel cannot own a DNN role")
		else:
			validate_ml_contract(operation.get("contract"), differentiation, where)
		validate_ml_autograd(
			operation.get("autograd"), differentiation, semantic_domain, lowering_only, where
		)
		if "rust_family" in operation:
			require(operation["rust_family"] in {"rope", "upsample"}
				and semantic_domain == "ml::matrix" and not lowering_only,
				f"{where}.rust_family must name an admitted complete ML Matrix family")
		if operation.get("autograd", {}).get("mode") == "generated":
			node = operation["autograd"]["node"]
			require(node not in seen_autograd_nodes, f"{where}.autograd.node must be unique")
			seen_autograd_nodes.add(node)
		replay_role = operation.get("training_replay_role", "safe")
		require(
			replay_role
			in {
				"safe",
				"host_stepped_optimizer",
				"optimizer_state_advance",
				"optimizer_state_update",
			},
			f"{where}.training_replay_role is unsupported",
		)
		validate_semantic_attributes(operation.get("semantic_attributes", []), where)
		validate_dnn_role(operation.get("dnn"), where)
		validate_ml_test(operation.get("test"), where)
		fields = operation.get("push_fields")
		require(isinstance(fields, list) and fields, f"{where}.push_fields must be non-empty")
		offset = 0
		seen_fields: set[str] = set()
		for field_index, field in enumerate(fields):
			field_where = f"{where}.push_fields[{field_index}]"
			require(
				isinstance(field, list) and len(field) == 2,
				f"{field_where} must contain name and scalar type",
			)
			name, scalar_type = field
			require(
				isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None,
				f"{field_where} name must be an identifier",
			)
			validate_unique(name, seen_fields, f"{field_where} name")
			require(scalar_type in {"uint32", "float32"}, f"{field_where} scalar type is unsupported")
			offset += 4
			require(offset <= 128, f"{where} push constants exceed the minimum Vulkan limit")
	for index, operation in enumerate(composites):
		where = f"ML composite_contracts[{index}]"
		require(isinstance(operation, dict), f"{where} must be an object")
		require(
			(set(operation) - {"donor_op"})
			== {
				"name",
				"semantic_domain",
				"differentiation",
				"autograd",
				"semantic_attributes",
				"contract",
			},
			f"{where} fields are incomplete or unknown",
		)
		name = operation["name"]
		require(
			isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None,
			f"{where}.name must be an identifier",
		)
		validate_unique(name, seen_names, f"{where}.name")
		semantic_domain = operation["semantic_domain"]
		require(
			semantic_domain in {"ml::advantage", "ml::environment", "ml::loss", "ml::matrix", "ml::policy"},
			f"{where}.semantic_domain must name an admitted composite owner",
		)
		differentiation = operation["differentiation"]
		require(
			differentiation in {"reverse", "none"},
			f"{where}.differentiation must be reverse or none",
		)
		validate_ml_contract(operation["contract"], differentiation, where)
		validate_ml_autograd(
			operation["autograd"], differentiation, semantic_domain, False, where
		)
		validate_semantic_attributes(operation["semantic_attributes"], where)


def validate_ml_autograd(
	autograd: Any,
	differentiation: str,
	semantic_domain: Any,
	lowering_only: bool,
	where: str,
) -> None:
	differentiable = not lowering_only and differentiation not in {"none", "backward"}
	if not differentiable:
		require(autograd is None, f"{where}.autograd is valid only for a differentiable forward operation")
		return
	require(isinstance(autograd, dict), f"{where}.autograd must own attachment policy")
	mode = autograd.get("mode")
	family = autograd.get("family")
	require(mode in {"generated", "manual"}, f"{where}.autograd.mode is unsupported")
	require(
		isinstance(family, str)
		and family
		and all(IDENTIFIER.fullmatch(part) is not None for part in family.split("/")),
		f"{where}.autograd.family must be an identifier path",
	)
	expected_root = {
		"ml::matrix": "matrix",
		"ml::loss": "loss",
		"ml::flow": "flow",
		"ml::advantage": "advantage",
		"ml::environment": "environment",
		"ml::policy": "policy",
	}.get(semantic_domain)
	require(
		expected_root is not None and (
			family == expected_root or family.startswith(expected_root + "/")
		),
		f"{where}.autograd.family must match its semantic owner",
	)
	if mode == "manual":
		require(
			set(autograd) == {"mode", "family"},
			f"{where}.autograd manual policy has unknown fields",
		)
		return
	require(
		{"mode", "family", "node", "inputs", "saved_matrices"} <= set(autograd)
		and set(autograd) <= {
			"mode", "family", "node", "inputs", "saved_matrices",
			"output_param", "saved_scalars", "owned_matrices",
		},
		f"{where}.autograd generated policy fields are incomplete or unknown",
	)
	node = autograd["node"]
	require(
		isinstance(node, str)
		and IDENTIFIER.fullmatch(node) is not None
		and node[0].isupper(),
		f"{where}.autograd.node must be a PascalCase identifier",
	)
	saved = autograd["saved_matrices"]
	inputs = autograd["inputs"]
	require(
		isinstance(inputs, list)
		and inputs
		and len(inputs) == len(set(inputs))
		and all(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None for name in inputs),
		f"{where}.autograd.inputs must contain unique identifiers",
	)
	require(
		isinstance(saved, list)
		and saved
		and len(saved) == len(set(saved))
		and all(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None for name in saved),
		f"{where}.autograd.saved_matrices must contain unique identifiers",
	)
	owned = autograd.get("owned_matrices", [])
	require(
		isinstance(owned, list)
		and all(isinstance(name, str) and name in saved and name not in inputs for name in owned)
		and len(owned) == len(set(owned)),
		f"{where}.autograd.owned_matrices must contain unique saved-only matrix names",
	)
	output_param = autograd.get("output_param", "result")
	require(
		isinstance(output_param, str) and IDENTIFIER.fullmatch(output_param) is not None,
		f"{where}.autograd.output_param must be an identifier",
	)
	scalars = autograd.get("saved_scalars", [])
	require(isinstance(scalars, list), f"{where}.autograd.saved_scalars must be an array")
	seen_scalars: set[str] = set()
	for index, scalar in enumerate(scalars):
		label = f"{where}.autograd.saved_scalars[{index}]"
		require(
			isinstance(scalar, list)
			and len(scalar) == 2
			and isinstance(scalar[0], str)
			and IDENTIFIER.fullmatch(scalar[0]) is not None
			and scalar[1] in {"f32", "usize", "u32", "UpsampleMode"},
			f"{label} must contain an identifier and supported Rust scalar type",
		)
		validate_unique(scalar[0], seen_scalars, f"{label} name")
		require(scalar[0] not in saved and scalar[0] not in inputs,
			f"{label} collides with a saved matrix or input")
	require("output_id" not in [*inputs, *saved, *seen_scalars],
		f"{where}.autograd output_id is reserved for output identity")
	require(output_param not in [*inputs, *saved, *seen_scalars],
		f"{where}.autograd.output_param collides with saved state")


def validate_ml_port_provenance(provenance: Any, operations: list[Any]) -> None:
	require(isinstance(provenance, list) and provenance, "ML port_provenance must be non-empty")
	operation_names = {
		operation.get("name") for operation in operations if isinstance(operation, dict)
	}
	seen_families: set[str] = set()
	seen_operations: set[str] = set()
	for index, record in enumerate(provenance):
		where = f"ML port_provenance[{index}]"
		require(isinstance(record, dict), f"{where} must be an object")
		require(
			set(record) == {"family", "operations", "classification", "donors", "reason"},
			f"{where} fields are incomplete or unknown",
		)
		family = record["family"]
		require(
			isinstance(family, str) and IDENTIFIER.fullmatch(family) is not None,
			f"{where}.family must be an identifier",
		)
		validate_unique(family, seen_families, f"{where}.family")
		classification = record["classification"]
		require(
			classification
			in {"verbatim", "mechanical_adaptation", "rust_redesign", "replacement", "new"},
			f"{where}.classification is unsupported",
		)
		record_operations = record["operations"]
		require(
			isinstance(record_operations, list) and record_operations,
			f"{where}.operations must be non-empty",
		)
		for operation in record_operations:
			require(
				isinstance(operation, str) and operation in operation_names,
				f"{where}.operations must name ML operations",
			)
			validate_unique(operation, seen_operations, f"{where}.operations")
		donors = record["donors"]
		require(isinstance(donors, list), f"{where}.donors must be an array")
		if classification == "new":
			require(not donors, f"{where}.donors must be empty for new work")
		else:
			require(
				donors
				and all(
					isinstance(donor, str)
					and donor.startswith(("source/", "tools/", "sdk/"))
					and ".." not in donor
					for donor in donors
				),
				f"{where}.donors must contain OA repository paths",
			)
		reason = record["reason"]
		require(
			isinstance(reason, str) and reason.endswith("."),
			f"{where}.reason must end with a period",
		)
	require(
		seen_operations == operation_names,
		"ML port_provenance must cover every operation exactly once",
	)


def validate_ml_contract(contract: Any, differentiation: str, where: str) -> None:
	require(isinstance(contract, dict), f"{where}.contract must be an object")
	base_fields = {
		"input_kinds",
		"output_kinds",
		"shape_rule",
		"dtype_rule",
		"effects",
		"mutated_inputs",
		"output_alias_inputs",
		"lowering",
	}
	variadic_fields = {"variadic_input", "variadic_output", "aligned_variadic_aliases"}
	optional_fields = {"optional_inputs", "control_flow"}
	extra_fields = set(contract) - base_fields
	require(
		set(contract).issuperset(base_fields)
		and extra_fields.issubset(variadic_fields | optional_fields),
		f"{where}.contract fields are incomplete or unknown",
	)
	inputs = contract["input_kinds"]
	outputs = contract["output_kinds"]
	variadic_input = contract.get("variadic_input")
	variadic_output = contract.get("variadic_output")
	require(
		isinstance(inputs, list) and (inputs or variadic_input is not None),
		f"{where}.contract.input_kinds must be non-empty without a variadic input",
	)
	require(
		isinstance(outputs, list) and (outputs or variadic_output is not None),
		f"{where}.contract.output_kinds must be non-empty without a variadic output",
	)
	require(
		len(inputs) <= 8 and len(outputs) <= 8,
		f"{where}.contract fixed arity exceeds OperationContract capacity; use a typed variadic tail",
	)
	require(
		all(isinstance(value, str) and value for value in [*inputs, *outputs]),
		f"{where}.contract input/output kinds must be strings",
	)
	for label, value in (("variadic_input", variadic_input), ("variadic_output", variadic_output)):
		if value is not None:
			require(
				isinstance(value, dict)
				and set(value) == {"kind", "minimum"}
				and isinstance(value["kind"], str)
				and value["kind"]
				and isinstance(value["minimum"], int)
				and not isinstance(value["minimum"], bool)
				and value["minimum"] > 0,
				f"{where}.contract.{label} must declare a kind and positive minimum",
			)
	if "aligned_variadic_aliases" in contract:
		require(
			isinstance(contract["aligned_variadic_aliases"], bool),
			f"{where}.contract.aligned_variadic_aliases must be boolean",
		)
	if contract.get("aligned_variadic_aliases", False):
		require(
			variadic_input is not None and variadic_input == variadic_output,
			f"{where}.contract aligned variadic aliases require identical input/output tails",
		)
	optional_inputs = contract.get("optional_inputs", [])
	require(
		isinstance(optional_inputs, list)
		and all(
			isinstance(index, int)
			and not isinstance(index, bool)
			and 0 <= index < len(inputs)
			for index in optional_inputs
		)
		and len(set(optional_inputs)) == len(optional_inputs),
		f"{where}.contract.optional_inputs must contain unique fixed input indices",
	)
	require(
		isinstance(contract["shape_rule"], str) and contract["shape_rule"],
		f"{where}.contract.shape_rule must be non-empty",
	)
	require(
		contract["dtype_rule"]
		in {
			"explicit",
			"f32",
			"all_f32",
			"same_admitted_dtype",
			"match_input",
			"promote_float",
			"f32_logits_u32_or_i32_targets_f32_mask",
			"f32_logits_u8_u32_or_i32_labels_u32_output",
			"f32_logits_u8_u32_or_i32_labels_f32_mask_u32_output",
			"u32_state",
			"f32_parameter_u32_state",
			"f32_logits_u32_targets",
			"f32_logits_u32_or_i32_targets",
			"f32_weight_u32_indices",
			"f32_weight_u8_or_u32_indices",
			"f32_weight_u8_u32_or_nonnegative_i32_indices",
			"f32_with_u32_indices",
			"u32_indices_f32_gradient_weight",
			"u8_or_u32_indices_f32_gradient_weight",
			"u8_u32_or_i32_indices_f32_gradient_weight",
			"f32_probabilities_i32_indices",
			"f32_gradients_probabilities_route_weights_i32_indices",
			"f32_values_u8_boundaries",
			"f32_values_i32_indices",
			"f32_logits_i32_indices",
			"f32_input_u32_output",
			"categorical_rollout_sources_and_time_major_storage",
            "f32_observation_i32_action_f32_statistics_preserve_dtype",
            "f32_accumulator_f32_value_u32_cursor",
            "f32_logits_u32_cursor_i32_actions",
			"u8_validity_mask",
			"replay_transition_and_storage",
			"replay_storage_to_sample_batch",
		},
		f"{where}.contract.dtype_rule is unsupported",
	)
	require(
		contract["effects"] in (["read_inputs", "write_output"], ["read_inputs", "write_outputs"]),
		f"{where}.contract.effects are unsupported",
	)
	mutated = contract["mutated_inputs"]
	require(
		isinstance(mutated, list)
		and all(isinstance(value, int) and 0 <= value < len(inputs) for value in mutated)
		and len(set(mutated)) == len(mutated),
		f"{where}.contract.mutated_inputs must contain unique input indices",
	)
	aliases = contract["output_alias_inputs"]
	require(
		isinstance(aliases, list)
		and len(aliases) == len(outputs)
		and all(value == -1 or 0 <= value < len(inputs) for value in aliases)
		and all(aliases.count(value) == 1 for value in mutated),
		f"{where}.contract.output_alias_inputs must reference inputs and map every mutated input exactly once",
	)
	if contract.get("aligned_variadic_aliases", False):
		require(
			not mutated and all(alias == -1 for alias in aliases),
			f"{where}.contract aligned variadic aliases cannot duplicate fixed mutation metadata",
		)
	require(
		contract["lowering"] in {"compute_dispatch", "gemm"},
		f"{where}.contract lowering is unsupported",
	)
	require(
		contract.get("control_flow", "straight_line") in {"straight_line", "conditional", "loop"},
		f"{where}.contract.control_flow is unsupported",
	)
	require(
		differentiation in {"none", "backward"} or isinstance(differentiation, str),
		f"{where} differentiation is invalid",
	)


def validate_ml_test(test: Any, where: str) -> None:
	require(isinstance(test, dict), f"{where}.test must be an object")
	require(
		isinstance(test.get("oracle"), str) and test["oracle"],
		f"{where}.test.oracle must be non-empty",
	)
	shapes = test.get("shapes")
	require(isinstance(shapes, list) and shapes, f"{where}.test.shapes must be non-empty")
	for index, shape in enumerate(shapes):
		require(
			isinstance(shape, list)
			and shape
			and all(isinstance(extent, int) and extent >= 0 for extent in shape),
			f"{where}.test.shapes[{index}] must contain non-negative extents",
		)
	tolerance = test.get("tolerance")
	require(
		isinstance(tolerance, (int, float))
		and not isinstance(tolerance, bool)
		and tolerance >= 0,
		f"{where}.test.tolerance must be non-negative",
	)


def generate_ml_autograd_attachments(
	ml: dict[str, Any], ml_hash: str, family: str
) -> str:
	operations = [
		operation
		for operation in ml["operations"]
		if operation.get("autograd", {}).get("mode") == "generated"
		and operation["autograd"]["family"] == family
	]
	lines = [
		banner(ml_hash, "//", "ml_training.json").rstrip(),
		"// Mechanical tape attachments; concrete saved-state node behavior remains private.",
		"",
		"use crate::{Matrix, Result};",
		"",
		"use crate::ml::autograd::{node::{GradNode, operation::GradNodeOperation}, tape::record_node};",
		"",
	]
	lines.append("use crate::ml::autograd::node::operation::{" + ", ".join(
		"GradNode" + operation["autograd"]["node"] for operation in operations
	) + "};")
	lines.append("")
	if any(
		scalar_type == "UpsampleMode"
		for operation in operations
		for _, scalar_type in operation["autograd"].get("saved_scalars", [])
	):
		lines.extend(["use crate::ml::matrix::UpsampleMode;", ""])
	for operation in operations:
		autograd = operation["autograd"]
		inputs = autograd["inputs"]
		saved = autograd["saved_matrices"]
		owned = autograd.get("owned_matrices", [])
		scalars = autograd.get("saved_scalars", [])
		output_param = autograd.get("output_param", "result")
		matrices = [*inputs, *(name for name in saved if name not in inputs)]
		arguments = ", ".join(
			[
				*(f"{name}: Matrix" if name in owned else f"{name}: &Matrix" for name in matrices),
				*(f"{name}: {scalar_type}" for name, scalar_type in scalars),
				f"{output_param}: &Matrix",
			]
		)
		lines.extend(
			[
				f"pub(in crate::ml) fn record_{operation['name']}({arguments}) -> Result<()> {{",
				f"\trecord_node(GradNode::Operation(GradNodeOperation::{autograd['node']}(GradNode{autograd['node']} {{",
			]
		)
		for name in matrices:
			lines.append(f"\t\t{name}," if name in owned else f"\t\t{name}: {name}.clone(),")
		for name, _ in scalars:
			lines.append(f"\t\t{name},")
		lines.extend(
			[
				f"\t\toutput_id: {output_param}.value_id(),",
				"\t})))",
				"}",
				"",
			]
		)
	return "\n".join(lines)


def ml_autograd_operations(ml: dict[str, Any]) -> list[dict[str, Any]]:
	return [operation for operation in ml["operations"]
		if operation.get("autograd", {}).get("mode") == "generated"]


def generate_ml_autograd_records(ml: dict[str, Any], ml_hash: str, family: str) -> str:
	"""Emit saved values and audited reverse recipes from the same forward rows."""
	operations = [op for op in ml_autograd_operations(ml) if op["autograd"]["family"] == family]
	path = FN_ROOT / "template/ml/autograd" / f"{family}.rs.in"
	require(path.is_file(), f"missing autograd recipe template for {family}")
	entries = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", path.read_text(), re.S)
	recipes = dict(entries)
	require(len(entries) == len(recipes), f"duplicate autograd recipe in {family}")
	require(set(recipes) == {op["name"] for op in operations},
		f"autograd recipe coverage must exactly match {family} operation rows")
	lines = [banner(ml_hash, "//", "ml_training.json").rstrip(),
		"// Saved-state and backward recipes; lifecycle remains tape-owned.",
		"use crate::{Matrix, Result};",
		"use std::collections::HashMap;",
		"use crate::ml::autograd::node::operation::BackwardStatus;",
		"use crate::ml::autograd::tape::backward::accumulate_value_gradient;",
	]
	# Recipe dependencies are family-local; unused imports remain lint errors.
	if family == "loss":
		lines.append("use crate::{matrix, ml::loss};")
	elif family in {"matrix/activation", "matrix/swiglu"}:
		lines.append("use crate::ml::matrix as ml_matrix;")
	elif family == "matrix/rope":
		lines.append("use crate::ml::matrix as ml_matrix;")
	if any(typ == "UpsampleMode" for op in operations for _, typ in op["autograd"].get("saved_scalars", [])):
		lines.append("use crate::ml::matrix::UpsampleMode;")
	for operation in operations:
		autograd = operation["autograd"]
		node = "GradNode" + autograd["node"]
		matrices = list(dict.fromkeys([*autograd["inputs"], *autograd["saved_matrices"]]))
		fields = [(name, "Matrix") for name in matrices] + autograd.get("saved_scalars", []) + [("output_id", "u64")]
		recipe = recipes[operation["name"]].strip()
		binding = re.search(r"let Self \{([^}]+)\} = self;", recipe)
		bound = [name.strip() for name in binding[1].split(",") if name.strip()] if binding else []
		require(len(bound) == len(set(bound)) and set(bound) == {name for name, _ in fields},
			f"autograd recipe saved fields must match {operation['name']} operation row")
		lines.extend(["", f"pub(in crate::ml::autograd) struct {node} {{"])
		lines.extend(f"\tpub(in crate::ml::autograd) {name}: {typ}," for name, typ in fields)
		lines.extend(["}", "", f"impl {node} {{",
			"\tpub(in crate::ml::autograd) fn backward(&self, gradients: &mut HashMap<u64, Matrix>) -> Result<BackwardStatus> {"])
		lines.extend("\t\t" + line for line in recipe.splitlines())
		lines.extend(["\t}", "}"])
	return "\n".join(lines) + "\n"


def generate_ml_autograd_routing(ml: dict[str, Any], ml_hash: str) -> str:
	"""Generate exhaustive output identity and reverse routing without a second catalog."""
	operations = ml_autograd_operations(ml)
	families = sorted({op["autograd"]["family"] for op in operations})
	lines = [banner(ml_hash, "//", "ml_training.json").rstrip(),
		"// Exhaustive routing for schema-owned saved operation records.",
		"use crate::{Matrix, Result};", "use std::collections::HashMap;"]
	for family in families:
		module = family.replace("/", "_")
		lines.extend([f'#[path = "{family}.gen.rs"]', f"mod {module};"])
		lines.append(f"pub(in crate::ml::autograd) use {module}::{{" + ", ".join(
			"GradNode" + op["autograd"]["node"] for op in operations if op["autograd"]["family"] == family
		) + "};")
	lines.extend(["", "pub(in crate::ml::autograd) enum BackwardStatus {", "\tSkipped,", "\tRecorded,", "}",
		"", "pub(in crate::ml::autograd) enum GradNodeOperation {"])
	lines.extend(f"\t{op['autograd']['node']}(GradNode{op['autograd']['node']})," for op in operations)
	lines.extend(["}", "", "impl GradNodeOperation {", "\tpub(in crate::ml::autograd) fn output_id(&self) -> u64 {", "\t\tmatch self {"])
	lines.extend(f"\t\t\tSelf::{op['autograd']['node']}(node) => node.output_id," for op in operations)
	lines.extend(["\t\t}", "\t}", "", "\tpub(in crate::ml::autograd) fn backward(&self, gradients: &mut HashMap<u64, Matrix>) -> Result<BackwardStatus> {", "\t\tmatch self {"])
	lines.extend(f"\t\t\tSelf::{op['autograd']['node']}(node) => node.backward(gradients)," for op in operations)
	lines.extend(["\t\t}", "\t}", "}", ""])
	return "\n".join(lines)


def _render_ml_activation_template(kind: str, values: dict[str, str]) -> str:
	template = (FN_ROOT / f"template/ml/activation_{kind}.rs.in").read_text(encoding="utf-8")
	placeholders = set(re.findall(r"{{(\w+)}}", template))
	require(placeholders == set(values), f"ML activation {kind} template placeholders disagree with renderer")
	return re.sub(r"{{(\w+)}}", lambda match: values[match.group(1)], template).rstrip()


def _ml_activation_fn(op: dict[str, Any]) -> str:
	"""Emit one complete Rust function body for an ML activation operation."""
	name = op["name"]
	kind = op["kind"]
	kernel = op["kernel_id"]
	contract = f"crate::core::operation::ml::{rust_const_name(name)}"

	if kind == "unary":
		doc = op.get("doc", f"Apply {name} elementwise.")
		saved = op["autograd"]["saved_matrices"][0]  # "input" or "output"
		# saved == "input": record(input, output) / saved == "output": record(output, output)
		if saved == "input":
			autograd_call = f"\tautograd::record_{name}(input, &output)?;"
		else:
			autograd_call = f"\tautograd::record_{name}(input, &output, &output)?;"
		return _render_ml_activation_template(
			"unary",
			{
				"doc": doc,
				"name": name,
				"contract": contract,
				"kernel": kernel,
				"autograd_call": autograd_call,
			},
		)

	if kind == "unary_backward":
		saved = op["saved"]  # "input" or "output"
		saved_param = f"saved_{saved}"
		return _render_ml_activation_template(
			"unary_backward",
			{
				"name": name,
				"saved_param": saved_param,
				"contract": contract,
				"kernel": kernel,
			},
		)

	if kind == "unary_scalar":
		scalar = op["scalar"]
		doc = op.get("doc", f"Apply {name} elementwise.")
		autograd = op["autograd"]
		saved = autograd["saved_matrices"][0]
		if saved == "input":
			autograd_call = f"\tautograd::record_{name}(input, {scalar}, &output)?;"
		else:
			autograd_call = f"\tautograd::record_{name}(input, &output, {scalar}, &output)?;"
		return _render_ml_activation_template(
			"unary_scalar",
			{
				"doc": doc,
				"name": name,
				"scalar": scalar,
				"contract": contract,
				"kernel": kernel,
				"autograd_call": autograd_call,
			},
		)

	if kind == "unary_scalar_backward":
		scalar = op["scalar"]
		saved = op["saved"]
		saved_param = f"saved_{saved}"
		return _render_ml_activation_template(
			"unary_scalar_backward",
			{
				"name": name,
				"saved_param": saved_param,
				"scalar": scalar,
				"contract": contract,
				"kernel": kernel,
			},
		)

	if kind == "binary":
		inputs = op.get("inputs", ["left", "right"])
		left, right = inputs[0], inputs[1]
		doc = op.get("doc", f"Apply {name} elementwise.")
		autograd_call = f"\tautograd::record_{name}({left}, {right}, &output)?;"
		return _render_ml_activation_template(
			"binary",
			{
				"doc": doc,
				"name": name,
				"left": left,
				"right": right,
				"contract": contract,
				"kernel": kernel,
				"autograd_call": autograd_call,
			},
		)

	if kind == "binary_backward":
		inputs = op.get("inputs", ["left", "right"])
		outputs = op.get("outputs", ["left_gradient", "right_gradient"])
		left, right = inputs[0], inputs[1]
		out0, out1 = outputs[0], outputs[1]
		return _render_ml_activation_template(
			"binary_backward",
			{
				"name": name,
				"left": left,
				"right": right,
				"contract": contract,
				"out0": out0,
				"out1": out1,
				"kernel": kernel,
			},
		)

	if kind == "silu_mul":
		doc = op.get("doc", "Apply SiLU(gate) * up to concatenated halves of the final input axis.")
		return _render_ml_activation_template(
			"silu_mul",
			{
				"doc": doc,
				"name": name,
				"contract": contract,
				"kernel": kernel,
			},
		)

	require(kind == "silu_mul_backward", f"Unsupported ML activation body kind: {kind}")
	return _render_ml_activation_template("silu_mul_backward", {"name": name, "contract": contract, "kernel": kernel})


def generate_ml_activation_api(schema: dict[str, Any], schema_hash: str, family: str = "activation") -> str:
	module = (FN_ROOT / "template/ml/activation_module.rs.in").read_text(encoding="utf-8")
	require(
		module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		"ML activation module must contain only one items placeholder",
	)
	items = "\n\n".join(_ml_activation_fn(op) for op in schema["operations"] if op["rust_family"] == family)
	return banner(schema_hash, "//", "ml_activation.json") + module.replace("{{items}}", items)


def generate_ml_matrix_family(ml: dict[str, Any], ml_hash: str, family: str) -> str:
	"""Publish complete provider bodies selected by the canonical operation rows."""
	operations = [op for op in ml["operations"] if op.get("rust_family") == family]
	path = FN_ROOT / "template/ml/matrix"
	module = (path / f"{family}_module.rs.in").read_text()
	entries = re.findall(r"// @operation (\w+)\n(.*?)(?=// @operation|\Z)", (path / f"{family}.rs.in").read_text(), re.S)
	bodies = dict(entries)
	require(len(entries) == len(bodies) and set(bodies) == {op["name"] for op in operations},
		f"ML Matrix {family} body coverage must exactly match operation rows")
	require(module.count("{{items}}") == 1 and set(re.findall(r"{{(\w+)}}", module)) == {"items"},
		f"ML Matrix {family} module must contain exactly one items placeholder")
	for operation in operations:
		require(re.search(r"fn " + operation["name"] + r"\(", bodies[operation["name"]]) is not None,
			f"ML Matrix {family} body must implement its operation name")
	items = "\n\n".join(bodies[op["name"]].strip() for op in operations)
	return banner(ml_hash, "//", "ml_training.json") + module.replace("{{items}}", items)
