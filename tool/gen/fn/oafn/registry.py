"""Kernel manifest, semantic contracts, donor identities and DNN roles."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .effects import kernel_fusion_effects, rust_fusion_effects
from .model import FUSION_CLASS_VARIANTS, count_buffer_indices, operation_lowering_variants, operation_variants, rust_physical_write, validate_physical_write, validate_retired_stable_ids, validate_unique
from .paths import FN_ROOT
from .pqc import MLDSA_WIRE_LAYOUTS
from .schema import Schemas, ensure_schemas
from .util import SchemaError, banner, fnv1a64, registry_banner, rust_const_name, static_name

def registry_entries(schemas: Schemas) -> list[dict[str, Any]]:
	schemas = ensure_schemas(schemas)
	elementwise = schemas["elemwise"][0]
	blas = schemas["blas"][0]
	reduce = schemas["reduce"][0]
	rng = schemas["rng"][0]
	index = schemas["index"][0]
	ml = schemas["ml"][0]
	audio = schemas["audio"][0]
	cryptography = schemas["hash"][0]
	cryptography_pqc = schemas["pqc"][0]
	image = schemas["image"][0]
	vision = schemas["vision"][0]
	video = schemas["video"][0]
	bigint = schemas["bigint"][0]
	merkle = schemas["merkle"][0]
	ntt = schemas["ntt"][0]
	entries = []
	for operation in elementwise["operations"]:
		if operation.get("lowering_only", False):
			entries.append(
				{
					"domain": "matrix",
					"name": operation["name"],
					"dtype": elementwise["dtype"],
					"kernel_id": operation["kernel_id"],
					"stable_id": operation["stable_id"],
					"workgroup_size": elementwise["workgroup_size"],
					"dispatch_tile_size": elementwise["workgroup_size"],
					"training_replay_role": "safe",
					"physical_write": operation.get("physical_write"),
					"fusion_class": operation.get("fusion_class"),
					"scalar_tail": operation.get("scalar_tail"),
					"prologue_kind": operation.get("prologue_kind"),
					"tape_op": operation.get("tape_op"),
					"tape_shape": operation.get("tape_shape"),
					"fusion_effects": kernel_fusion_effects(operation),
					"semantic_contract": None,
				}
			)
			continue
		for variant in operation_variants(elementwise, operation):
			entries.append(
				{
					"domain": "matrix",
					"name": operation["name"],
					"dtype": variant["dtype"],
					"kernel_id": variant["kernel_id"],
					"stable_id": variant["stable_id"],
					"workgroup_size": elementwise["workgroup_size"],
					"dispatch_tile_size": elementwise["workgroup_size"],
					"training_replay_role": "safe",
					"physical_write": operation.get("physical_write"),
					"fusion_class": operation.get("fusion_class"),
					"scalar_tail": operation.get("scalar_tail") if variant["kernel_id"] == operation["kernel_id"] else None,
					"prologue_kind": operation.get("prologue_kind") if variant["kernel_id"] == operation["kernel_id"] else None,
					"tape_op": operation.get("tape_op") if variant["kernel_id"] == operation["kernel_id"] else None,
					"tape_shape": operation.get("tape_shape") if variant["kernel_id"] == operation["kernel_id"] else None,
					"fusion_effects": kernel_fusion_effects(operation if variant["kernel_id"] == operation["kernel_id"] else variant, inherited=operation),
					"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(operation['name'])}",
				}
			)
		for lowering in operation_lowering_variants(operation):
			entries.append(
				{
					"domain": "matrix",
					"name": lowering["name"],
					"dtype": lowering["dtype"],
					"kernel_id": lowering["kernel_id"],
					"stable_id": lowering["stable_id"],
					"workgroup_size": lowering.get("workgroup_size", elementwise["workgroup_size"]),
					"dispatch_tile_size": lowering.get("dispatch_tile_size", lowering.get("workgroup_size", elementwise["workgroup_size"])),
					"training_replay_role": "safe",
					"physical_write": lowering["physical_write"],
					"fusion_class": operation.get("fusion_class"),
					"scalar_tail": None,
					"prologue_kind": lowering.get("prologue_kind"),
					"tape_op": lowering.get("tape_op"),
					"tape_shape": lowering.get("tape_shape"),
					"fusion_effects": kernel_fusion_effects(lowering, inherited=operation),
					"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(operation['name'])}",
				}
			)
	for operation in blas["operations"]:
		entries.append(
			{
				"domain": "matrix",
				"name": operation["name"],
				"dtype": blas["dtype"],
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": blas["workgroup_size"],
				"dispatch_tile_size": blas["output_tile_size"],
				"training_replay_role": "safe",
				"physical_write": operation.get("physical_write"),
				"fusion_class": operation.get("fusion_class"),
					"fusion_effects": kernel_fusion_effects(operation),
				"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(operation['name'])}",
			}
		)
	def candidate_contract_ref(semantic):
		parts = semantic.split("::")
		domain = parts[0] if len(parts) == 2 else "matrix"
		operation = parts[-1]
		return f"crate::core::operation::{domain}::{rust_const_name(operation)}"

	for candidate in blas.get("candidate_kernels", []):
		entries.append(
			{
				"domain": "matrix",
				"name": candidate["name"],
				"dtype": candidate.get("dtype", blas["dtype"]),
				"kernel_id": candidate["kernel_id"],
				"stable_id": candidate["stable_id"],
				"workgroup_size": blas["workgroup_size"],
				"dispatch_tile_size": candidate["dispatch_tile_size"],
				"training_replay_role": "safe",
				"physical_write": candidate.get("physical_write"),
				"fusion_class": operation.get("fusion_class"),
					"fusion_effects": kernel_fusion_effects(candidate),
				"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(candidate['semantic_operation'])}",
				"semantic_contracts": [
					candidate_contract_ref(semantic)
					for semantic in candidate.get("semantic_operations", [])
				],
			}
		)
	for operation in reduce["operations"]:
		if "kernel_id" not in operation:
			continue
		semantic = operation.get("semantic_operation", operation["name"])
		entries.append(
			{
				"domain": "matrix",
				"name": operation["name"],
				"dtype": reduce["dtype"],
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": reduce["workgroup_size"],
				"dispatch_tile_size": [1, 1, 1],
				"training_replay_role": "safe",
				"physical_write": operation.get("physical_write"),
				"fusion_class": operation.get("fusion_class"),
					"fusion_effects": kernel_fusion_effects(operation),
				"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(semantic)}",
			}
		)
	for operation in rng["operations"]:
		name = operation["name"]
		semantic = operation.get("semantic_operation", name)
		entries.append(
			{
				"domain": "matrix",
				"name": name,
				"dtype": operation.get("dtype", rng["dtype"]),
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": operation.get("workgroup_size", rng["workgroup_size"]),
				"dispatch_tile_size": operation.get(
					"dispatch_tile_size", operation.get("workgroup_size", rng["workgroup_size"])
				),
				"training_replay_role": operation.get("training_replay_role", "safe"),
				"physical_write": operation.get("physical_write"),
				"fusion_class": operation.get("fusion_class"),
					"fusion_effects": kernel_fusion_effects(operation),
				"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(semantic)}",
			}
		)
	for operation in index["operations"]:
		if "kernel_id" not in operation:
			continue
		semantic = operation.get("semantic_operation", operation["name"])
		entries.append(
			{
				"domain": "matrix",
				"name": operation["name"],
				"dtype": operation.get("dtype", index["dtype"]),
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": operation.get("workgroup_size", index["workgroup_size"]),
				"dispatch_tile_size": operation.get(
					"dispatch_tile_size", operation.get("workgroup_size", index["workgroup_size"])
				),
				"training_replay_role": "safe",
				"physical_write": operation.get("physical_write"),
				"fusion_class": operation.get("fusion_class"),
				"tape_op": operation.get("tape_op"),
				"tape_shape": operation.get("tape_shape"),
					"fusion_effects": kernel_fusion_effects(operation),
				"semantic_contract": f"crate::core::operation::matrix::{rust_const_name(semantic)}",
			}
		)
	for operation in ml["operations"]:
		dtype = operation.get("dtype", ml["dtype"])
		semantic = operation.get("semantic_operation", operation["name"])
		entries.append(
			{
				"domain": "ml",
				"report_domain": operation.get("semantic_domain", "ml").replace("::", "."),
				"name": operation["name"],
				"dtype": dtype,
				"kernel_id": operation["kernel_id"],
				"stable_id": operation["stable_id"],
				"workgroup_size": operation.get("workgroup_size", ml["workgroup_size"]),
				"dispatch_tile_size": operation.get(
					"dispatch_tile_size", operation.get("workgroup_size", ml["workgroup_size"])
				),
				"training_replay_role": operation.get("training_replay_role", "safe"),
				"physical_write": operation.get("physical_write"),
				"fusion_class": operation.get("fusion_class"),
				"scalar_tail": operation.get("scalar_tail"),
				"prologue_kind": operation.get("prologue_kind"),
				"tape_op": operation.get("tape_op"),
				"tape_shape": operation.get("tape_shape"),
				"act_kind": operation.get("act_kind"),
					"fusion_effects": kernel_fusion_effects(operation),
				"semantic_contract": (
					None
					if operation.get("lowering_only", False) and "semantic_operation" not in operation
					else f"crate::core::operation::ml::{rust_const_name(semantic)}"
				),
			}
		)
	cryptography_operations = {operation["name"]: operation for operation in cryptography["contracts"]}
	pqc_operations = {operation["name"]: operation for operation in cryptography_pqc["contracts"]}
	image_operations = {operation["name"]: operation for operation in image["contracts"]}
	vision_operations = {operation["name"]: operation for operation in vision["contracts"]}
	video_operations = {operation["name"]: operation for operation in video["contracts"]}
	bigint_operations = {operation["name"]: operation for operation in bigint["contracts"]} if bigint is not None else {}
	audio_operations = {operation["name"]: operation for operation in audio["contracts"]}
	for kernel in audio["kernels"]:
		semantic = kernel.get("semantic_operation")
		entries.append(
			{
				"domain": "audio",
				"name": kernel["name"],
				"dtype": audio["dtype"],
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel.get("physical_write"),
				"fusion_class": kernel_fusion_class(kernel, audio_operations),
				"region_read": kernel_region_read(kernel, audio_operations),
				"tile_program": kernel.get("tile_program"),
				"tape_op": kernel.get("tape_op"),
				"tape_shape": kernel.get("tape_shape"),
					"fusion_effects": kernel_fusion_effects(kernel),
				"semantic_contract": (
					None
					if semantic is None
					else f"crate::core::operation::audio::{rust_const_name(semantic)}"
				),
			}
		)
	for kernel in cryptography["kernels"]:
		semantic = kernel["semantic_operation"]
		entries.append(
			{
				"domain": "cryptography",
				"report_domain": "cryptography.hash",
				"name": kernel["name"],
				"dtype": kernel.get("dtype", cryptography["dtype"]),
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel.get("physical_write"),
				"fusion_class": kernel_fusion_class(kernel, cryptography_operations),
					"fusion_effects": kernel_fusion_effects(kernel),
				"semantic_contract": f"crate::core::operation::crypto::hash::{rust_const_name(semantic)}",
			}
		)
	for kernel in cryptography_pqc["kernels"]:
		semantic = kernel["semantic_operation"]
		entries.append(
			{
				"domain": "cryptography",
				"report_domain": "cryptography.pqc",
				"name": kernel["name"],
				"dtype": cryptography_pqc["dtype"],
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel.get("physical_write"),
				"fusion_class": kernel_fusion_class(kernel, pqc_operations),
					"fusion_effects": kernel_fusion_effects(kernel),
				"semantic_contract": f"crate::core::operation::crypto::pqc::{rust_const_name(semantic)}",
			}
		)
	for kernel in cryptography_pqc.get("private_kernels", []):
		entries.append({
			"domain": "cryptography", "report_domain": "cryptography.pqc",
			"name": kernel["name"], "dtype": cryptography_pqc["dtype"],
			"kernel_id": kernel["kernel_id"], "stable_id": kernel["stable_id"],
			"workgroup_size": kernel["workgroup_size"], "dispatch_tile_size": kernel["workgroup_size"],
			"training_replay_role": "safe", "physical_write": kernel["physical_write"],
			"fusion_class": kernel_fusion_class(kernel, pqc_operations),
					"fusion_effects": kernel_fusion_effects(kernel),
			"semantic_contract": None,
			"semantic_contracts": [
				f"crate::core::operation::crypto::pqc::{rust_const_name(semantic)}"
				for semantic in kernel.get("semantic_operations", [])
			],
			"requires_secret_storage": bool(kernel["secret_bindings"]),
		})
	for kernel in image["kernels"]:
		semantic = kernel["semantic_operation"]
		entries.append(
			{
				"domain": "image",
				"name": kernel["name"],
				"dtype": kernel.get("dtype", image["dtype"]),
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel["physical_write"],
				"fusion_class": kernel_fusion_class(kernel, image_operations),
				"region_read": kernel_region_read(kernel, image_operations),
				"tape_op": kernel.get("tape_op"),
				"tape_shape": kernel.get("tape_shape"),
					"fusion_effects": kernel_fusion_effects(kernel),
				"semantic_contract": f"crate::core::operation::image::{rust_const_name(semantic)}",
				"tile_program": kernel_tile_program(kernel, image_operations),
			}
		)
	for kernel in vision["kernels"]:
		semantic = kernel["semantic_operation"]
		entries.append(
			{
				"domain": "vision",
				"name": kernel["name"],
				"dtype": kernel.get("dtype", vision["dtype"]),
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": "safe",
				"physical_write": kernel["physical_write"],
				"fusion_class": kernel_fusion_class(kernel, vision_operations),
					"fusion_effects": kernel_fusion_effects(kernel),
				"semantic_contract": f"crate::core::operation::vision::{rust_const_name(semantic)}",
			}
		)
	for kernel in video["kernels"]:
		if not kernel.get("registered", True):
			continue
		semantic = kernel.get("semantic_operation")
		entries.append(
			{
				"domain": "video",
				"name": kernel["name"],
				"dtype": kernel.get("dtype", "u8"),
				"kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"],
				"workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel.get("dispatch_tile_size", kernel["workgroup_size"]),
				"training_replay_role": kernel.get("training_replay_role", "safe"),
				"physical_write": kernel.get("physical_write"),
				"fusion_class": kernel_fusion_class(kernel, video_operations),
					"fusion_effects": kernel_fusion_effects(kernel),
				"semantic_contract": (
					None
					if semantic is None
					else f"crate::core::operation::video::{rust_const_name(semantic)}"
				),
			}
		)
	if bigint is not None:
		for kernel in bigint["kernels"]:
			entries.append({
				"domain": "cryptography", "report_domain": "cryptography.bigint",
				"name": kernel["name"], "dtype": "u32", "kernel_id": kernel["kernel_id"],
				"stable_id": kernel["stable_id"], "workgroup_size": kernel["workgroup_size"],
				"dispatch_tile_size": kernel["workgroup_size"], "training_replay_role": "safe",
				"physical_write": kernel["physical_write"],
				"fusion_class": kernel_fusion_class(kernel, bigint_operations),
					"fusion_effects": kernel_fusion_effects(kernel),
				"semantic_contract": f"crate::core::operation::crypto::bigint::{rust_const_name(kernel['semantic_operation'])}",
			})
	for kernel in merkle['kernels']:
		entries.append({'domain': 'cryptography', 'report_domain': 'cryptography.merkle', 'name': kernel['name'], 'dtype': 'u32',
			'kernel_id': kernel['kernel_id'], 'stable_id': kernel['stable_id'], 'workgroup_size': kernel['workgroup_size'], 'dispatch_tile_size': kernel['workgroup_size'],
			'training_replay_role': 'safe', 'physical_write': kernel['physical_write'], 'fusion_class': 'opaque', 'fusion_effects': kernel_fusion_effects(kernel), 'semantic_contract': f"crate::core::operation::crypto::merkle::{rust_const_name(kernel['semantic_operation'])}"})
	for kernel in ntt['kernels']:
		entries.append({'domain': 'cryptography', 'report_domain': 'cryptography.ntt', 'name': kernel['name'], 'dtype': 'u32',
			'kernel_id': kernel['kernel_id'], 'stable_id': kernel['stable_id'], 'workgroup_size': kernel['workgroup_size'], 'dispatch_tile_size': kernel['workgroup_size'],
			'training_replay_role': 'safe', 'physical_write': kernel['physical_write'], 'fusion_class': 'opaque', 'fusion_effects': kernel_fusion_effects(kernel), 'semantic_contract': f"crate::core::operation::crypto::ntt::{rust_const_name(kernel['semantic_operation'])}"})
	buffer_counts: dict[str, int] = {}
	for kernel in ntt['kernels'] + merkle['kernels']:
		buffer_counts[kernel['kernel_id']] = count_buffer_indices(kernel)
	for operation in elementwise["operations"]:
		if operation.get("lowering_only", False):
			buffer_counts[operation["kernel_id"]] = count_buffer_indices(operation)
			continue
		kind = operation["kind"]
		count = {"binary": 3, "unary": 2, "unary_scalar": 2}[kind]
		for variant in operation_variants(elementwise, operation):
			buffer_counts[variant["kernel_id"]] = count
		for lowering in operation_lowering_variants(operation):
			buffer_counts[lowering["kernel_id"]] = count_buffer_indices(lowering)
	for operation in blas["operations"]:
		buffer_counts[operation["kernel_id"]] = 3
	def candidate_contract_ref(semantic):
		parts = semantic.split("::")
		domain = parts[0] if len(parts) == 2 else "matrix"
		operation = parts[-1]
		return f"crate::core::operation::{domain}::{rust_const_name(operation)}"

	for candidate in blas.get("candidate_kernels", []):
		buffer_counts[candidate["kernel_id"]] = (
			3
			if candidate["variant"] in {"naive", "gemv", "strided", "strided_tiled", "split_k", "f16"}
			else 2
			if candidate["variant"] in {"split_k_reduce", "cast_f32_f16", "cast_f16_f32", "sum_f16"}
			else 4
		)
	for schema in [reduce, rng, index, ml, audio, cryptography, cryptography_pqc, image, vision]:
		for operation in schema.get("operations", schema.get("kernels", [])) + schema.get("private_kernels", []):
			if "kernel_id" in operation:
				buffer_counts[operation["kernel_id"]] = count_buffer_indices(operation)
	for kernel in video["kernels"]:
		if kernel.get("registered", True):
			buffer_counts[kernel["kernel_id"]] = count_buffer_indices(kernel)
	if bigint is not None:
		for kernel in bigint["kernels"]:
			buffer_counts[kernel["kernel_id"]] = count_buffer_indices(kernel)
	for entry in entries:
		entry["bounded_buffer_count"] = buffer_counts[entry["kernel_id"]]
	seen_ids = validate_retired_stable_ids(
		ml.get("retired_stable_ids", []), "ML retired_stable_ids"
	)
	seen_kernels: set[str] = set()
	for entry in entries:
		validate_physical_write(
			entry["physical_write"], f"kernel candidate {entry['kernel_id']}", required=False
		)
		validate_unique(entry["stable_id"], seen_ids, "stable kernel ID across schemas")
		validate_unique(entry["kernel_id"], seen_kernels, "kernel ID across schemas")
	return entries


def generate_kernel_manifest(schemas: Schemas) -> str:
	schemas = ensure_schemas(schemas)
	elementwise, elementwise_hash = schemas["elemwise"]
	blas, blas_hash = schemas["blas"]
	reduce, reduce_hash = schemas["reduce"]
	rng, rng_hash = schemas["rng"]
	index, index_hash = schemas["index"]
	ml, ml_hash = schemas["ml"]
	audio, audio_hash = schemas["audio"]
	cryptography, cryptography_hash = schemas["hash"]
	cryptography_pqc, cryptography_pqc_hash = schemas["pqc"]
	image, image_hash = schemas["image"]
	vision, vision_hash = schemas["vision"]
	video, video_hash = schemas["video"]
	bigint, bigint_hash = schemas["bigint"]
	entries = registry_entries(schemas)
	lines = [
		registry_banner(
			elementwise_hash,
			blas_hash,
			reduce_hash,
			rng_hash,
			index_hash,
			ml_hash,
			audio_hash,
			cryptography_hash,
			image_hash,
			vision_hash,
		).rstrip(),
		"",
		"use super::{CollisionPolicy, KernelEffects, KernelRegionRead, KernelScalarTail, KernelTapeShape, LogicalWriteDomain, PhysicalWrite, PhysicalWriteContract, ShaderArtifact, TailPolicy, UniformStoreProof, WorkspacePartition, WriteExtent, WritePartition, content_ids};",
		"",
		f"pub(crate) const MAX_BOUNDED_STORAGE_BUFFERS: u32 = {max(entry['bounded_buffer_count'] for entry in entries)};",
		"",
	]
	if bigint_hash is not None:
		lines.insert(1, f"// cryptography_bigint_sha256={bigint_hash}")
	lines.insert(1, f"// video_sha256={video_hash}")
	for entry in entries:
		domain = entry["domain"]
		name = entry["name"]
		dtype = entry["dtype"]
		workgroup = entry["workgroup_size"]
		dispatch_tile = entry["dispatch_tile_size"]
		physical_write = rust_physical_write(entry["physical_write"])
		lines.extend(
			[
				f"static {static_name(domain, name, dtype)}: ShaderArtifact = ShaderArtifact {{",
				f'\tbytes: include_bytes!(concat!(env!("OA_SHADER_DIR"), "/{domain}_{name}_{dtype}.spv")),',
				f"\tcontent_id: content_ids::{static_name(domain, name, dtype)},",
				f"\tworkgroup_size: [{workgroup[0]}, {workgroup[1]}, {workgroup[2]}],",
				f"\tdispatch_tile_size: [{dispatch_tile[0]}, {dispatch_tile[1]}, {dispatch_tile[2]}],",
				f"\tphysical_write: {physical_write},",
				"};",
				"",
			]
		)

	lines.extend(
		[
			"#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
			"pub(crate) enum TrainingReplayRole {",
			"\tSafe,",
			"\tHostSteppedOptimizer,",
			"\tOptimizerStateAdvance,",
			"\tOptimizerStateUpdate,",
			"\tFrozenRng,",
			"\tReplayRng,",
			"\tRngStateAdvance,",
			"}",
			"",
			"#[repr(u16)]",
			"#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
			"pub(crate) enum KernelId {",
		]
	)
	for entry in entries:
		lines.append(f"\t{entry['kernel_id']} = {entry['stable_id']},")
	lines.extend(["}", "", "impl KernelId {", "\tpub(crate) const fn mldsa_keygen_layout(parameter: u32) -> Option<(u64, u64, u64)> {", "\t\tmatch parameter {"])
	for kernel in cryptography_pqc.get("private_kernels", []):
		if kernel["name"] == "ml_dsa_keygen":
			for row in kernel["parameter_layouts"]:
				lines.append(f"\t\t\t{row['parameter_set']} => Some(({row['seed_bytes']}, {row['public_bytes']}, {row['private_bytes']})),")
	lines.extend(["\t\t\t_ => None,", "\t\t}", "\t}", "\tpub(crate) const fn mldsa_sign_layout(parameter: u32) -> Option<(u64, u64, u64, u64)> {", "\t\tmatch parameter {"])
	for kernel in cryptography_pqc.get("private_kernels", []):
		if kernel["name"] == "ml_dsa_sign":
			for row in kernel["parameter_layouts"]:
				lines.append(f"\t\t\t{row['parameter_set']} => Some(({row['seed_bytes']}, {row['private_bytes']}, {row['mu_bytes']}, {row['signature_bytes']})),")
	lines.extend(["\t\t\t_ => None,", "\t\t}", "\t}"])
	workspace = next(kernel["workspace_bytes"] for kernel in cryptography_pqc["private_kernels"] if kernel["name"] == "ml_dsa_sign")
	lines.extend(["\tpub(crate) const fn mldsa_sign_workspace_bytes() -> usize {", f"\t\t{workspace}", "\t}", "\tpub(crate) const fn mldsa_prehash_bytes(algorithm: u32) -> Option<u64> {", "\t\tmatch algorithm {"])
	for row in cryptography_pqc["prehashes"]:
		lines.append(f"\t\t\t{row['oid_suffix']} => Some({row['digest_bytes']}),")
	lines.extend(["\t\t\t_ => None,", "\t\t}", "\t}", "",
		"\t/// FIPS 204 M' head words for the public mu kernel: the 0x00 or 0x01",
		"\t/// domain tag, the context length byte and, for HashML-DSA, the",
		"\t/// eleven-byte OID trailer, packed little-endian. Returns the four",
		"\t/// push words and the live byte count (2 or 13).",
		"\tpub(crate) const fn mldsa_mu_head(algorithm: u32, context_length: u32) -> Option<([u32; 4], u32)> {",
		"\t\tif context_length > 255 {",
		"\t\t\treturn None;",
		"\t\t}",
		"\t\tif algorithm == 0 {",
		"\t\t\treturn Some(([context_length << 8, 0, 0, 0], 2));",
		"\t\t}",
		"\t\tif Self::mldsa_prehash_bytes(algorithm).is_none() {",
		"\t\t\treturn None;",
		"\t\t}",
		"\t\tSome((",
		"\t\t\t[0x0906_0001 | context_length << 8, 0x0148_8660, 0x0204_0365, algorithm],",
		"\t\t\t13,",
		"\t\t))",
		"\t}",
		"}", "", "impl KernelId {", f"\tpub(crate) const ALL: [Self; {len(entries)}] = ["])
	for entry in entries:
		lines.append(f"\t\tSelf::{entry['kernel_id']},")
	lines.extend(["\t];", "", "\tpub(crate) const fn artifact(self) -> &'static ShaderArtifact {", "\t\tmatch self {"])
	for entry in entries:
		lines.append(
			f"\t\t\tSelf::{entry['kernel_id']} => &{static_name(entry['domain'], entry['name'], entry['dtype'])},"
		)

	secret_candidates = [f"Self::{entry['kernel_id']}" for entry in entries if entry.get("requires_secret_storage", False)]
	secret_match = " | ".join(secret_candidates)
	lines.extend(["\t\t}", "\t}", "", "\tpub(crate) const fn requires_secret_storage(self) -> bool {", f"\t\tmatches!(self, {secret_match})" if secret_match else "\t\tfalse", "\t}", "", "\tpub(crate) const fn bounded_buffer_count(self) -> u32 {", "\t\tmatch self {"])
	for entry in entries:
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => {entry['bounded_buffer_count']},")
	lines.extend(["\t\t}", "\t}", "", "\tpub(crate) const fn index(self) -> usize {", "\t\tmatch self {"])
	for index, entry in enumerate(entries):
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => {index},")
	replay_roles = {
		"safe": "Safe",
		"host_stepped_optimizer": "HostSteppedOptimizer",
		"optimizer_state_advance": "OptimizerStateAdvance",
		"optimizer_state_update": "OptimizerStateUpdate",
		"frozen_rng": "FrozenRng",
		"replay_rng": "ReplayRng",
		"rng_state_advance": "RngStateAdvance",
	}
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn semantic_contract(self) -> Option<crate::OperationContract> {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		contract = entry["semantic_contract"]
		value = "None" if contract is None else f"Some({contract})"
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => {value},")
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\t/// Additional semantic contracts this kernel serves as a stage of a",
			"\t/// split lowering, when no single physical kernel owns the contract.",
			"\tpub(crate) fn semantic_contracts(self) -> &'static [crate::OperationContract] {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		served = entry.get("semantic_contracts") or []
		if served:
			lines.append(f"\t\t\tSelf::{entry['kernel_id']} => &[")
			lines.extend(f"\t\t\t\t{contract}," for contract in served)
			lines.append("\t\t\t],")
	lines.extend(
		[
			"\t\t\t_ => &[],",
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn report_name(self) -> &'static str {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		name = f"{entry.get('report_domain', entry['domain'])}.{entry['name']}.{entry['dtype']}"
		lines.append(f'\t\t\tSelf::{entry["kernel_id"]} => "{name}",')
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn dtype_report_token(self) -> &'static str {",
			"\t\tmatch self {",
		]
	)
	dtype_tokens = {"f32": "float32", "f16": "float16", "i32": "int32", "u32": "uint32", "u8": "uint8"}
	for entry in entries:
		lines.append(
			f'\t\t\tSelf::{entry["kernel_id"]} => "{dtype_tokens[entry["dtype"]]}",'
		)
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn training_replay_role(self) -> TrainingReplayRole {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		role = replay_roles[entry["training_replay_role"]]
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => TrainingReplayRole::{role},")
	lines.extend(["\t\t}", "\t}", "", "\tpub(crate) const fn fusion_effects(self) -> KernelEffects {", "\t\tmatch self {"])
	for entry in entries:
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => {rust_fusion_effects(entry['fusion_effects'])},")
	lines.extend(["\t\t}", "\t}", "", "\tpub(crate) const fn fusion_class(self) -> crate::core::OpFusionClass {", "\t\tmatch self {"])
	for entry in entries:
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => crate::core::OpFusionClass::{fusion_class_variant(entry['fusion_class'] or 'opaque')},")
	lines.extend(
		[
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn region_read(self) -> KernelRegionRead {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		lines.append(f"\t\t\tSelf::{entry['kernel_id']} => KernelRegionRead::{region_read_variant(entry.get('region_read'))},")
	lines.extend(["\t\t}", "\t}", "", "\tpub(crate) const fn scalar_tail(self) -> Option<KernelScalarTail> {", "\t\tmatch self {"])
	for entry in entries:
		tail = scalar_tail_variant(entry.get("scalar_tail"))
		if tail is not None:
			lines.append(f"\t\t\tSelf::{entry['kernel_id']} => Some(KernelScalarTail::{tail}),")
	lines.extend(
		[
			"\t\t\t_ => None,",
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn tape_op(self) -> Option<(u32, KernelTapeShape)> {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		code = tape_op_code(entry.get("tape_op"))
		shape = tape_shape_variant(entry.get("tape_shape"))
		if code is not None and shape is not None:
			lines.append(f"\t\t\tSelf::{entry['kernel_id']} => Some(({code}, KernelTapeShape::{shape})),")
		elif code is not None or shape is not None:
			raise SchemaError(f"{entry['kernel_id']} declares tape_op without tape_shape or vice versa")
	lines.extend(
		[
			"\t\t\t_ => None,",
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn act_kind(self) -> Option<u32> {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		kind = act_kind_code(entry.get("act_kind"))
		if kind is not None:
			lines.append(f"\t\t\tSelf::{entry['kernel_id']} => Some({kind}),")
	lines.extend(
		[
			"\t\t\t_ => None,",
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) const fn prologue_kind(self) -> Option<u32> {",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		kind = prologue_kind_code(entry.get("prologue_kind"))
		if kind is not None:
			lines.append(f"\t\t\tSelf::{entry['kernel_id']} => Some({kind}),")
	lines.extend(
		[
			"\t\t\t_ => None,",
			"\t\t}",
			"\t}",
			"",
			"\tpub(in crate::runtime) fn tile_program(self) -> Option<crate::runtime::dna::ir::tile::BlockProgram> {",
			"\t\tuse crate::runtime::dna::ir::tile::BlockProgram;",
			"\t\tmatch self {",
		]
	)
	for entry in entries:
		program = entry.get("tile_program")
		if program is not None:
			name = program.split("(")[0]
			if name not in TILE_PROGRAM_FACTORIES:
				raise SchemaError(f"{entry['kernel_id']} declares unknown tile_program '{program}'")
			call = program if "(" in program else f"{program}()"
			expr = f"BlockProgram::{call}.ok()" if name in TILE_PROGRAM_RESULT_FACTORIES else f"Some(BlockProgram::{call})"
			lines.append(f"\t\t\tSelf::{entry['kernel_id']} => {expr},")
	lines.extend(
		[
			"\t\t\t_ => None,",
			"\t\t}",
			"\t}",
			"",
			"\tpub(crate) fn linear_workgroups(self, element_count: u32) -> [u32; 3] {",
			"\t\tlet width = self.artifact().dispatch_tile_size[0];",
			"\t\t[element_count.div_ceil(width), 1, 1]",
			"\t}",
			"",
			"\tpub(crate) fn output_workgroups(self, rows: u32, columns: u32) -> [u32; 3] {",
			"\t\tlet tile = self.artifact().dispatch_tile_size;",
			"\t\t[rows.div_ceil(tile[0]), columns.div_ceil(tile[1]), 1]",
			"\t}",
			"}",
			"",
		]
	)
	return "\n".join(lines)


def semantic_contract_hash(domain: str, name: str, contract: dict[str, Any], attributes: list[dict[str, str]]) -> int:
	payload = json.dumps(
		{
			"name": f"oa::{domain}::{name}",
			"contract": contract,
			"attributes": attributes,
		},
		sort_keys=True,
		separators=(",", ":"),
	).encode("utf-8")
	value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
	return value or 1


DONOR_CONTRACTS_PATH = FN_ROOT / "donor_contracts.json"
_DONOR_CONTRACTS: dict[str, dict] | None = None


def donor_contract_hash(payload: dict) -> int:
	canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
	return fnv1a64(canonical.encode("utf-8"))


def donor_contract_records() -> dict[str, dict]:
	"""Load the checked-in donor snapshot, re-verifying every stored hash."""
	global _DONOR_CONTRACTS
	if _DONOR_CONTRACTS is not None:
		return _DONOR_CONTRACTS
	if not DONOR_CONTRACTS_PATH.exists():
		raise SchemaError(f"donor contract snapshot {DONOR_CONTRACTS_PATH} is missing")
	document = json.loads(DONOR_CONTRACTS_PATH.read_text(encoding="utf-8"))
	if document.get("schema") != "oa.donor_operation_contracts.v1":
		raise SchemaError(f"{DONOR_CONTRACTS_PATH}: unsupported snapshot schema")
	records: dict[str, dict] = {}
	for record in document.get("records", []):
		name = record.get("name")
		if not isinstance(name, str) or name in records:
			raise SchemaError(f"{DONOR_CONTRACTS_PATH}: invalid or duplicate record name {name!r}")
		if record.get("hash") != donor_contract_hash(record["payload"]):
			raise SchemaError(f"{DONOR_CONTRACTS_PATH}: {name} hash does not match its payload")
		records[name] = record
	_DONOR_CONTRACTS = records
	return records


# Emitted-contract normalization. These maps translate a Rust schema contract
# to the donor's ``oa.operation_contract.v1`` field spellings by following the
# exact emission used below — a linked ``donor_op`` therefore certifies that
# the emitted Rust contract declares precisely the donor contract.
_CONTRACT_VALUE_KIND_VARIANT = {
	"matrix": "Matrix",
	"parameter": "Matrix",
	"audio": "Audio",
	"image": "Image",
	"video_frame": "VideoFrame",
	"quant_matrix": "QuantMatrix",
	"texture": "Texture",
}


_CONTRACT_VALUE_KIND_DONOR = {
	"Matrix": "matrix",
	"Image": "image",
	"Audio": "audio",
	"VideoFrame": "video_frame",
	"QuantMatrix": "quant_matrix",
	"Texture": "texture",
}


def contract_value_kind_variant(token: str) -> str:
	return _CONTRACT_VALUE_KIND_VARIANT.get(token, "Matrix")


def fusion_class_variant(token: str) -> str:
	"""Map a schema ``fusion_class`` token to its ``OpFusionClass`` variant."""
	if token not in FUSION_CLASS_VARIANTS:
		raise SchemaError(f"unknown fusion_class {token}")
	return FUSION_CLASS_VARIANTS[token]


def region_read_variant(token: str | None) -> str:
	"""Map the schema `region_read` token to the generated enum variant.

	Absent metadata stays `Gather` — the non-admitting default for the
	elementwise-prologue admission this read pattern feeds.
	"""
	return {
		"elementwise": "Elementwise",
		"gather": "Gather",
		"bordered": "Bordered",
	}.get(token, "Gather")


def scalar_tail_variant(token: str | None) -> str | None:
	"""Map the schema `scalar_tail` token to the generated enum variant.

	Absent metadata means the kernel does not admit a scalar store tail.
	"""
	if token is None:
		return None
	variants = {
		"add": "Add",
		"sub": "Sub",
		"mul": "Mul",
		"div": "Div",
		"max": "Max",
		"min": "Min",
		"neg": "Neg",
		"abs": "Abs",
		"log": "Log",
		"exp": "Exp",
		"sqrt": "Sqrt",
		"recip": "Recip",
		"relu": "Relu",
	}
	if token not in variants:
		raise SchemaError(f"unknown scalar_tail {token}")
	return variants[token]


def tape_op_code(token: str | None) -> int | None:
	"""Map the schema `tape_op` token to the fused-chain tape opcode.

	The numeric values are the `matrix_fused_chain` Slang ABI in
	`elementwise.slang` `eval_element` — the schema token names the op,
	the generator owns the ABI number."""
	if token is None:
		return None
	codes = {
		"gather": 4,
		"add": 10, "sub": 11, "mul": 12, "div": 13, "pow": 14,
		"min": 15, "max": 16,
		"neg": 20, "recip": 21, "abs": 22, "exp": 23, "log": 24,
		"sqrt": 25, "sin": 26, "cos": 27, "tanh": 28,
		"sigmoid": 30, "relu": 31, "gelu": 32, "silu": 33,
		"relu_backward": 40, "gelu_backward": 41, "silu_backward": 42,
		"tanh_backward": 43, "sigmoid_backward": 44,
		"softplus_backward": 45, "mish_backward": 46,
		"elu_backward": 47, "leaky_relu_backward": 48,
		"gain_db": 49, "clamp": 50,
	}
	if token not in codes:
		raise SchemaError(f"unknown tape_op {token}")
	return codes[token]


def tape_shape_variant(token: str | None) -> str | None:
	"""Map the schema `tape_shape` token to the generated enum variant."""
	if token is None:
		return None
	variants = {
		"binary": "Binary",
		"unary": "Unary",
		"scalar": "Scalar",
		"scalar_first": "ScalarFirst",
		"scalar_pair": "ScalarPair",
		"broadcast": "Broadcast",
		"gather": "Gather",
		"binary_scalar": "BinaryScalar",
	}
	if token not in variants:
		raise SchemaError(f"unknown tape_shape {token}")
	return variants[token]


def act_kind_code(token: str | None) -> int | None:
	"""Map the schema `act_kind` token to the fused `linear_act`
	family's activation-kind push value (Slang ABI: `apply_act`
	switch). Forward and backward ops declare the same family token."""
	if token is None:
		return None
	codes = {
		"relu": 0, "gelu": 1, "silu": 2, "tanh": 3, "sigmoid": 4,
		"mish": 5, "elu": 6, "leaky_relu": 7, "softplus": 8,
	}
	if token not in codes:
		raise SchemaError(f"unknown act_kind {token}")
	return codes[token]


def prologue_kind_code(token: str | None) -> int | None:
	"""Map the schema `prologue_kind` token to the fused
	`linear_backward_prologue` family's prologue-kind push value."""
	if token is None:
		return None
	codes = {
		"silu_mul": 9, "add": 10, "sub": 11, "mul": 12, "scalar_mul": 13,
	}
	if token not in codes:
		raise SchemaError(f"unknown prologue_kind {token}")
	return codes[token]


TILE_PROGRAM_FACTORIES = {
	"image_resize", "image_flip", "image_pad", "image_warp", "image_remap",
	"image_crop", "image_rotate", "image_convolve", "image_separable_convolve",
	"image_average_blur", "image_morphology_once", "image_morphology_twice",
	"image_morphology_gradient", "image_morphology_hat", "image_directional_linear",
	"image_laplacian", "image_sharpen", "image_gaussian", "image_bilateral",
	"image_unsharp", "image_adaptive_threshold", "image_pointwise",
	"image_segmentation_overlay", "image_normalize", "image_convert_color",
	"image_resize_normalize", "image_grayscale", "image_channel_reorder",
	"image_color_twist", "image_composite", "audio_mel_filterbank",
	"channel_normalize", "video_to_matrix",
}

TILE_PROGRAM_RESULT_FACTORIES = {"image_resize", "channel_normalize", "video_to_matrix"}


def kernel_tile_program(kernel: dict, contracts: dict) -> str | None:
	"""Resolve the `tile_program` BlockProgram factory token: kernel record
	first, owning contract via `semantic_operation` second, absent -> None."""
	if kernel.get("tile_program"):
		return kernel["tile_program"]
	semantic = kernel.get("semantic_operation")
	if semantic in contracts:
		return contracts[semantic].get("tile_program")
	return None


def kernel_region_read(kernel: dict, contracts: dict) -> str | None:
	"""Resolve a kernel entry's input read pattern from its own record or
	owning contract. Absent metadata means the consuming admission stays
	conservative (`region_read_variant` defaults to `gather`)."""
	if kernel.get("region_read"):
		return kernel["region_read"]
	semantic = kernel.get("semantic_operation")
	if semantic is not None and semantic in contracts:
		return contracts[semantic].get("region_read")
	return None


def kernel_fusion_class(kernel: dict, operations: dict) -> str | None:
	"""Resolve a kernel entry's fusion class from its own record or owning operation."""
	if kernel.get("fusion_class"):
		return kernel["fusion_class"]
	semantic = kernel.get("semantic_operation")
	if semantic is not None and semantic in operations:
		return operations[semantic].get("fusion_class")
	return None


def contract_shape_rule_variant(token: str) -> str:
	if token in {"left_mk_right_nk_to_mn", "matmul_nt"}:
		return "MatMulNt"
	if token == "broadcast":
		return "Broadcast"
	if token in {"equal_no_broadcast", "preserve_input", "match_input"}:
		return "MatchInput"
	return "Explicit"


def contract_dtype_rule_variant(token: str) -> str:
	if token in {"same_admitted_dtype", "all_f32", "matching_f32", "f32", "match_input"}:
		return "MatchInput"
	if token == "promote_float":
		return "PromoteFloat"
	return "Explicit"


def normalized_semantic_contract(contract: dict[str, Any]) -> dict[str, Any]:
	"""Return the contract fields as emitted, spelled in donor vocabulary."""
	out: dict[str, Any] = {}
	out["input_kinds"] = [
		_CONTRACT_VALUE_KIND_DONOR[contract_value_kind_variant(kind)]
		for kind in contract["input_kinds"]
	]
	out["output_kinds"] = [
		_CONTRACT_VALUE_KIND_DONOR[contract_value_kind_variant(kind)]
		for kind in contract["output_kinds"]
	]
	if "variadic_input" in contract:
		variadic = contract["variadic_input"]
		out["variadic_inputs"] = {
			"kind": _CONTRACT_VALUE_KIND_DONOR[contract_value_kind_variant(variadic["kind"])],
			"minimum": variadic["minimum"],
		}
	if "variadic_output" in contract:
		variadic = contract["variadic_output"]
		out["variadic_outputs"] = {
			"kind": _CONTRACT_VALUE_KIND_DONOR[contract_value_kind_variant(variadic["kind"])],
			"minimum": variadic["minimum"],
		}
	if contract.get("aligned_variadic_aliases", False):
		out["aligned_variadic_aliases"] = True
	shape_donor = {
		"MatchInput": "match_input",
		"Broadcast": "broadcast",
		"MatMulNt": "matmul_nt",
		"Explicit": "explicit",
	}
	dtype_donor = {
		"MatchInput": "match_input",
		"PromoteFloat": "promote_float",
		"Explicit": "explicit",
	}
	out["shape_rule"] = shape_donor[contract_shape_rule_variant(contract["shape_rule"])]
	out["dtype_rule"] = dtype_donor[contract_dtype_rule_variant(contract["dtype_rule"])]
	effect_donor = {
		"read_inputs": "read_inputs",
		"write_output": "write_outputs",
		"write_outputs": "write_outputs",
	}
	out["effects"] = [effect_donor[effect] for effect in contract["effects"]]
	out["mutated_inputs"] = list(contract["mutated_inputs"])
	if contract.get("optional_inputs"):
		out["optional_inputs"] = list(contract["optional_inputs"])
	out["output_alias_inputs"] = list(contract["output_alias_inputs"])
	out["control_flow"] = contract.get("control_flow") or "straight_line"
	out["differentiation"] = (
		"none" if contract.get("differentiation") == "none" else "reverse"
	)
	out["lowering"] = "gemm" if contract["lowering"] == "gemm" else "dispatch"
	return out


def resolve_donor_contract(
	domain: str,
	name: str,
	contract: dict[str, Any],
	attributes: list[dict[str, str]],
	donor_op: str,
) -> tuple[str, int]:
	"""Validate a ``donor_op`` link and return ``(qualified_name, hash)``."""
	records = donor_contract_records()
	record = records.get(donor_op)
	label = f"oa::{domain}::{name} donor link {donor_op}"
	if record is None:
		raise SchemaError(f"{label}: no donor contract record with that name")
	payload = record["payload"]
	problems: list[str] = []
	expected = payload["contract"]
	actual = normalized_semantic_contract(contract)
	for field in sorted(set(actual) | set(expected)):
		if actual.get(field, "<absent>") != expected.get(field, "<absent>"):
			problems.append(
				f"{field}: rust {actual.get(field, '<absent>')} != donor {expected.get(field, '<absent>')}"
			)
	expected_attributes = payload.get("attributes", [])
	actual_attributes = [{"name": a["name"], "kind": a["kind"]} for a in attributes]
	if actual_attributes != expected_attributes:
		problems.append(
			f"attributes: rust {actual_attributes} != donor {expected_attributes}"
		)
	if problems:
		raise SchemaError(f"{label}: semantic contract does not match the donor record: " + "; ".join(problems))
	return payload["namespace"] + "::" + payload["name"], record["hash"]


def semantic_contract_lines(
	domain: str,
	name: str,
	contract: dict[str, Any],
	attributes: list[dict[str, str]],
	donor_op: str | None = None,
	fusion_class: str | None = None,
) -> list[str]:
	inputs = ", ".join(f"OpValueKind::{contract_value_kind_variant(kind)}" for kind in contract["input_kinds"])
	outputs = ", ".join(f"OpValueKind::{contract_value_kind_variant(kind)}" for kind in contract["output_kinds"])
	shape_rule = f"OpShapeRule::{contract_shape_rule_variant(contract['shape_rule'])}"
	dtype_rule = f"OpDTypeRule::{contract_dtype_rule_variant(contract['dtype_rule'])}"
	differentiation = (
		"OpDifferentiation::None"
		if contract.get("differentiation") == "none"
		else "OpDifferentiation::Reverse"
	)
	lowering = (
		"OpLowering::Gemm"
		if contract["lowering"] == "gemm"
		else "OpLowering::Dispatch"
	)
	if donor_op is not None:
		contract_name, value = resolve_donor_contract(
			domain, name, contract, attributes, donor_op
		)
	else:
		contract_name = f"oa::{domain}::{name}"
		value = semantic_contract_hash(domain, name, contract, attributes)
	lines = [
		f"pub const {rust_const_name(name)}: OperationContract = OperationContract::new(",
		f'\t"{contract_name}",',
		f"\t0x{value:016x},",
		f"\t&[{inputs}],",
		f"\t&[{outputs}],",
		")",
	]
	if "variadic_input" in contract:
		variadic_input = contract["variadic_input"]
		lines.append(
			f".variadic_inputs(OpValueKind::{contract_value_kind_variant(variadic_input['kind'])}, {variadic_input['minimum']})"
		)
	if "variadic_output" in contract:
		variadic_output = contract["variadic_output"]
		lines.append(
			f".variadic_outputs(OpValueKind::{contract_value_kind_variant(variadic_output['kind'])}, {variadic_output['minimum']})"
		)
	if contract.get("aligned_variadic_aliases", False):
		lines.append(".aligned_variadic_aliases()")
	lines.extend(
		[
			f".with_shape_rule({shape_rule})",
			f".with_dtype_rule({dtype_rule})",
			f".with_differentiation({differentiation})",
			f".with_lowering({lowering})",
			f".with_fusion_class(OpFusionClass::{fusion_class_variant(fusion_class or 'opaque')})",
			".effects(OpEffect::READ_INPUTS.union(OpEffect::WRITE_OUTPUTS))",
		]
	)
	control_flow = contract.get("control_flow")
	if control_flow is not None and control_flow != "straight_line":
		flow_names = {"conditional": "Conditional", "loop": "Loop"}
		lines.append(f".with_control_flow(OpControlFlow::{flow_names[control_flow]})")
	if attributes:
		attribute_kinds = {
			"boolean": "Boolean",
			"signed_integer": "SignedInteger",
			"unsigned_integer": "UnsignedInteger",
			"float": "Float",
			"string": "String",
			"shape": "Shape",
			"enum": "Enum",
		}
		specs = ", ".join(
			f'OpAttributeSpec::new("{attribute["name"]}", OpAttributeKind::{attribute_kinds[attribute["kind"]]})'
			for attribute in attributes
		)
		lines.append(f".attributes(&[{specs}])")
	optional_mask = sum(1 << input_index for input_index in contract.get("optional_inputs", []))
	if optional_mask:
		lines.append(f".optional_inputs(0x{optional_mask:02x})")
	mutation_mask = sum(1 << input_index for input_index in contract["mutated_inputs"])
	if mutation_mask:
		lines.append(f".mutated_inputs(0x{mutation_mask:02x})")
	for output_index, input_index in enumerate(contract["output_alias_inputs"]):
		if input_index >= 0:
			lines.append(f".alias({output_index}, {input_index})")
	lines[-1] += ";"
	return lines


def generate_operation_registry(schemas: Schemas) -> str:
	schemas = ensure_schemas(schemas)
	elementwise, elementwise_hash = schemas["elemwise"]
	blas, blas_hash = schemas["blas"]
	reduce, reduce_hash = schemas["reduce"]
	rng, rng_hash = schemas["rng"]
	index, index_hash = schemas["index"]
	ml, ml_hash = schemas["ml"]
	audio, audio_hash = schemas["audio"]
	cryptography, cryptography_hash = schemas["hash"]
	cryptography_pqc, cryptography_pqc_hash = schemas["pqc"]
	image, image_hash = schemas["image"]
	vision, vision_hash = schemas["vision"]
	video, video_hash = schemas["video"]
	bigint, bigint_hash = schemas["bigint"]
	merkle, merkle_hash = schemas["merkle"]
	ntt, ntt_hash = schemas["ntt"]
	lines = [
		banner(elementwise_hash, "//").rstrip(),
		f"// matrix_blas_sha256={blas_hash}",
		f"// matrix_reduce_sha256={reduce_hash}",
		f"// matrix_rng_sha256={rng_hash}",
		f"// matrix_index_sha256={index_hash}",
		f"// ml_training_sha256={ml_hash}",
		f"// audio_sha256={audio_hash}",
		f"// cryptography_hash_sha256={cryptography_hash}",
		f"// cryptography_pqc_sha256={cryptography_pqc_hash}",
		f"// image_sha256={image_hash}",
		f"// vision_detection_sha256={vision_hash}",
		"// Current OARS compatibility contracts; donor OA identities replace each row when its full contract lands.",
		"",
		"use super::{",
		"\tOpAttributeKind, OpAttributeSpec, OpControlFlow, OpDTypeRule, OpDifferentiation,",
		"\tOpEffect, OpFusionClass, OpLowering, OpShapeRule, OpValueKind,",
		"\tOperationContract,",
		"};",
		"",
		"/// Generated semantic contracts for Matrix operations.",
		"pub mod matrix {",
		"\tuse super::*;",
		"",
	]
	for operation in elementwise["operations"]:
		if operation.get("lowering_only", False):
			continue
		contract = dict(elementwise["contracts"][operation["kind"]])
		contract["shape_rule"] = operation.get("shape_rule", contract["shape_rule"])
		contract["differentiation"] = operation.get(
			"differentiation", contract["differentiation"]
		)
		attributes = []
		if operation["kind"] == "unary_scalar":
			attributes = [{"name": operation["scalar_name"], "kind": "float"}]
		for line in semantic_contract_lines("matrix", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	for operation in blas["operations"]:
		for line in semantic_contract_lines("matrix", operation["name"], operation["contract"], [], operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	for operation in reduce["operations"]:
		if operation.get("lowering_only", False):
			continue
		contract = dict(operation["contract"])
		contract["differentiation"] = operation["differentiation"]
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation["semantic_attributes"]
		]
		for line in semantic_contract_lines("matrix", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	for operation in rng["operations"]:
		if "contract" not in operation:
			continue
		contract = dict(operation["contract"])
		contract["differentiation"] = operation.get("differentiation", "none")
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation.get("semantic_attributes", [])
		]
		for line in semantic_contract_lines("matrix", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	for operation in index["operations"]:
		if "contract" not in operation:
			continue
		contract = dict(operation["contract"])
		contract["differentiation"] = operation.get("differentiation", "none")
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation.get("semantic_attributes", [])
		]
		for line in semantic_contract_lines("matrix", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend(["}", "", "/// Generated compatibility contracts for current ML kernels.", "pub mod ml {", "\tuse super::*;", ""])
	for operation in ml["operations"]:
		if operation.get("lowering_only", False):
			continue
		contract = dict(operation["contract"])
		contract["differentiation"] = (
			"none" if operation["differentiation"] in {"none", "backward"} else "reverse"
		)
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation.get("semantic_attributes", [])
		]
		for line in semantic_contract_lines(
			operation["semantic_domain"], operation["name"], contract, attributes,
			operation.get("donor_op"), operation.get("fusion_class"),
		):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	for operation in ml.get("composite_contracts", []):
		contract = dict(operation["contract"])
		contract["differentiation"] = operation["differentiation"]
		attributes = [
			{"name": name, "kind": kind}
			for name, kind in operation["semantic_attributes"]
		]
		for line in semantic_contract_lines(
			operation["semantic_domain"], operation["name"], contract, attributes,
			operation.get("donor_op"), operation.get("fusion_class"),
		):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend(["}", "", "/// Generated semantic contracts for Audio operations.", "pub mod audio {", "\tuse super::*;", ""])
	for operation in audio["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1] * len(operation["output_kinds"]),
			"differentiation": operation.get("differentiation", "none"),
			"lowering": "compute_dispatch",
		}
		if "control_flow" in operation:
			contract["control_flow"] = operation["control_flow"]
		if "optional_inputs" in operation:
			contract["optional_inputs"] = operation["optional_inputs"]
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("audio", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend([
		"}",
		"",
		"/// Generated semantic contracts for cryptographic batch hashing.",
		"pub mod crypto {",
		"\tuse super::*;",
		"",
		"\tpub mod hash {",
		"\t\tuse super::*;",
		"",
	])
	for operation in cryptography["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1],
			"differentiation": "none",
			"lowering": "compute_dispatch",
		}
		if "control_flow" in operation:
			contract["control_flow"] = operation["control_flow"]
		if "optional_inputs" in operation:
			contract["optional_inputs"] = operation["optional_inputs"]
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("crypto::hash", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t\t{line}" if line else "")
		lines.append("")
	if bigint is not None:
		lines.extend(["\t}", "", "\tpub mod bigint {", "\t\tuse super::*;", ""])
		for operation in bigint["contracts"]:
			contract = {key: operation[key] for key in ("input_kinds", "output_kinds", "shape_rule", "dtype_rule")}
			contract.update(effects=["read_inputs", "write_outputs"], mutated_inputs=[], output_alias_inputs=[-1] * len(operation["output_kinds"]), differentiation="none", lowering="compute_dispatch")
			attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
			for line in semantic_contract_lines("crypto::bigint", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
				lines.append(f"\t\t{line}" if line else "")
			lines.append("")
		lines.insert(1, f"// cryptography_bigint_sha256={bigint_hash}")
	lines.extend(["\t}", "", "\tpub mod ntt {", "\t\tuse super::*;", ""])
	for operation in ntt['contracts']:
		contract = {key: operation[key] for key in ('input_kinds', 'output_kinds', 'shape_rule', 'dtype_rule')}
		contract.update(effects=['read_inputs', 'write_outputs'], mutated_inputs=[], output_alias_inputs=[-1] * len(operation['output_kinds']), differentiation='none', lowering='compute_dispatch')
		attributes = [{'name': name, 'kind': kind} for name, kind in operation['attributes']]
		for line in semantic_contract_lines('crypto::ntt', operation['name'], contract, attributes, operation.get('donor_op'), operation.get('fusion_class')):
			lines.append(f"\t\t{line}" if line else "")
		lines.append('')
	lines.insert(1, f'// cryptography_ntt_sha256={ntt_hash}')
	lines.extend(["\t}", "", "\tpub mod merkle {", "\t\tuse super::*;", ""])
	for operation in merkle['contracts']:
		contract = {key: operation[key] for key in ('input_kinds', 'output_kinds', 'shape_rule', 'dtype_rule', 'mutated_inputs', 'output_alias_inputs')}
		contract.update(effects=['read_inputs', 'write_outputs'], differentiation='none', lowering='compute_dispatch')
		if 'control_flow' in operation:
			contract['control_flow'] = operation['control_flow']
		if 'optional_inputs' in operation:
			contract['optional_inputs'] = operation['optional_inputs']
		attributes = [{'name': name, 'kind': kind} for name, kind in operation['attributes']]
		for line in semantic_contract_lines('crypto::merkle', operation['name'], contract, attributes, operation.get('donor_op'), operation.get('fusion_class')):
			lines.append(f"\t\t{line}" if line else "")
		lines.append('')
	lines.insert(1, f'// cryptography_merkle_sha256={merkle_hash}')
	lines.extend([
		"\t}",
		"",
		"\tpub mod pqc {",
		"\t\tuse super::*;",
		"",
	])
	for parameter in MLDSA_WIRE_LAYOUTS:
		pk_bytes, sig_bytes = MLDSA_WIRE_LAYOUTS[parameter]
		lines.append(f"\t\tpub(crate) const ML_DSA_{parameter}_PUBLIC_KEY_SIZE: usize = {pk_bytes};")
		lines.append(f"\t\tpub(crate) const ML_DSA_{parameter}_SIGNATURE_SIZE: usize = {sig_bytes};")
	lines.append("")
	for operation in cryptography_pqc["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1],
			"differentiation": "none",
			"lowering": "compute_dispatch",
		}
		if "control_flow" in operation:
			contract["control_flow"] = operation["control_flow"]
		if "optional_inputs" in operation:
			contract["optional_inputs"] = operation["optional_inputs"]
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("crypto::pqc", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t\t{line}" if line else "")
		lines.append("")
	lines.extend(["\t}", "}", "", "/// Generated semantic contracts for Image operations.", "pub mod image {", "\tuse super::*;", ""])
	for operation in image["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1],
			"differentiation": "none",
			"lowering": "compute_dispatch",
		}
		if "control_flow" in operation:
			contract["control_flow"] = operation["control_flow"]
		if "optional_inputs" in operation:
			contract["optional_inputs"] = operation["optional_inputs"]
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("image", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend(["}", ""])
	lines.extend(["/// Generated semantic contracts for Vision operations.", "pub mod vision {", "\tuse super::*;", ""])
	for operation in vision["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": ["read_inputs", "write_outputs"],
			"mutated_inputs": [],
			"output_alias_inputs": [-1] * len(operation["output_kinds"]),
			"differentiation": operation.get("differentiation", "none"),
			"lowering": "compute_dispatch",
		}
		if "control_flow" in operation:
			contract["control_flow"] = operation["control_flow"]
		if "optional_inputs" in operation:
			contract["optional_inputs"] = operation["optional_inputs"]
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("vision", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend(["}", ""])
	lines.extend(["/// Generated semantic contracts for Video operations.", "pub mod video {", "\tuse super::*;", ""])
	for operation in video["contracts"]:
		contract = {
			"input_kinds": operation["input_kinds"],
			"output_kinds": operation["output_kinds"],
			"shape_rule": operation["shape_rule"],
			"dtype_rule": operation["dtype_rule"],
			"effects": operation.get("effects", ["read_inputs", "write_outputs"]),
			"mutated_inputs": operation.get("mutated_inputs", []),
			"output_alias_inputs": operation.get("output_alias_inputs", [-1] * len(operation["output_kinds"])),
			"differentiation": operation.get("differentiation", "none"),
			"lowering": operation.get("lowering", "compute_dispatch"),
		}
		for key in ("control_flow", "optional_inputs", "variadic_input", "variadic_output", "aligned_variadic_aliases"):
			if key in operation:
				contract[key] = operation[key]
		attributes = [{"name": name, "kind": kind} for name, kind in operation["attributes"]]
		for line in semantic_contract_lines("video", operation["name"], contract, attributes, operation.get("donor_op"), operation.get("fusion_class")):
			lines.append(f"\t{line}" if line else "")
		lines.append("")
	lines.extend(["}", ""])
	return "\n".join(lines)


def dnn_rust_variant(value: str) -> str:
	overrides = {
		"blaslt_epilogue": "BlasLtEpilogue",
		"qkv_projection_group": "QkvProjectionGroup",
		"gated_ffn": "GatedFfn",
		"grouped_moe": "GroupedMoe",
		"rms_norm": "RmsNorm",
		"residual_rms_norm": "ResidualRmsNorm",
	}
	return overrides.get(value, "".join(part.capitalize() for part in value.split("_")))


def generate_dnn_roles(schemas: Schemas) -> str:
	schemas = ensure_schemas(schemas)
	elementwise, elementwise_hash = schemas["elemwise"]
	blas, blas_hash = schemas["blas"]
	reduce, reduce_hash = schemas["reduce"]
	rng_hash = schemas["rng"][1]
	ml, ml_hash = schemas["ml"]
	image, image_hash = schemas["image"]
	lines = [
		banner(elementwise_hash, "//").rstrip(),
		f"// matrix_blas_sha256={blas_hash}",
		f"// matrix_reduce_sha256={reduce_hash}",
		f"// matrix_rng_sha256={rng_hash}",
		f"// ml_training_sha256={ml_hash}",
		f"// image_sha256={image_hash}",
		"// Candidate roles mirror OA DNN vocabulary while retaining OARS compatibility identities.",
		"",
		"pub(super) static DNN_OP_ROLES: &[DnnOpRole] = &[",
	]
	for domain, schema in (
		("matrix", elementwise),
		("matrix", blas),
		("matrix", reduce),
		("ml", ml),
		("image", image),
	):
		for operation in schema.get("operations", schema.get("contracts", [])):
			role = operation.get("dnn")
			if role is None:
				continue
			providers = " | ".join(
				f"DnnProvider::{dnn_rust_variant(provider)}.bit()"
				for provider in role["providers"]
			)
			epilogue = dnn_rust_variant(role.get("epilogue", "none"))
			required_input = role.get("epilogue_requires_input")
			required_input = "None" if required_input is None else f"Some({required_input})"
			lines.extend(
				[
					"\tDnnOpRole {",
					f"\t\tcontract: crate::core::operation::{domain}::{rust_const_name(operation['name'])},",
					f"\t\top_type: DnnOpType::{dnn_rust_variant(role['role'])},",
					f"\t\tepilogue: DnnEpilogue::{epilogue},",
					f"\t\tepilogue_required_input: {required_input},",
					f"\t\tprovider_mask: {providers},",
					"\t},",
				]
			)
	lines.extend(["];", ""])
	return "\n".join(lines)
