"""PQC schema validation and ML-DSA/ML-KEM shader emitters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .crypto import validate_cryptography_api
from .model import validate_physical_write, validate_semantic_attributes, validate_unique
from .util import IDENTIFIER, banner, format_rust, require

# FIPS 204 final Table 1 / Algorithms 22 and 26 encoded wire layouts.
# One mechanical authority for Rust metadata and dispatch wrapper strides.
MLDSA_WIRE_LAYOUTS = {44: (1312, 2420), 65: (1952, 3309), 87: (2592, 4627)}


# NIST CSOR hashAlgs 1..12; SHAKE prehash output is fixed to 256/512 bits.
MLDSA_PREHASH_LAYOUTS = [
	{"name": name, "oid_suffix": suffix, "digest_bytes": length}
	for suffix, (name, length) in enumerate(zip(
		("SHA2-256", "SHA2-384", "SHA2-512", "SHA2-224", "SHA2-512/224", "SHA2-512/256",
		 "SHA3-224", "SHA3-256", "SHA3-384", "SHA3-512", "SHAKE-128", "SHAKE-256"),
		(32, 48, 64, 28, 28, 32, 28, 32, 48, 64, 32, 64)), 1)
]


def validate_pqc_device_api(schema: dict[str, Any]) -> None:
	api = schema.get("device_api")
	require(isinstance(api, dict), "PQC device_api must be an object")
	require(set(api) == {"status", "secret_policy", "capture_policy", "key_import", "mlkem_parameters", "prehash_rust_names", "operations"}, "PQC device API fields mismatch")
	require(api["status"] == "experimental_candidate", "PQC device status must not imply qualification")
	require(api["secret_policy"] == "unique_nonmappable_consuming_erasure", "PQC secret ownership mismatch")
	require(api["capture_policy"] == "reject_ordinary_capture", "PQC capture policy mismatch")
	require(api["key_import"] in ("not_admitted", "checked_staging_upload"), "PQC key import policy mismatch")
	expected = [
		{"parameter": n, "k": k, "public_key_bytes": 384 * k + 32, "private_key_bytes": 768 * k + 96,
		 "ciphertext_bytes": 32 * (11 * k + 5 if k == 4 else 10 * k + 4), "shared_secret_bytes": 32}
		for n, k in ((512, 2), (768, 3), (1024, 4))
	]
	require(api["mlkem_parameters"] == expected, "FIPS 203 public device layouts mismatch")
	require(all(type(value) is int for row in api["mlkem_parameters"] for value in row.values()), "ML-KEM layout fields must be integers")
	require(api["prehash_rust_names"] == ["Sha256", "Sha384", "Sha512", "Sha224", "Sha512Truncated224", "Sha512Truncated256", "Sha3_224", "Sha3_256", "Sha3_384", "Sha3_512", "Shake128", "Shake256"], "PQC typed hash names mismatch")
	expected_operations = [
		{"name": name, "executor": executor, "private_kernel": kernel} for name, executor, kernel in (
			("mldsa_keygen", "pqc_mldsa_keygen", "ml_dsa_keygen"),
			("mlkem_keygen", "pqc_mlkem_keygen", "ml_kem_keygen"),
			("mldsa_sign", "pqc_mldsa_sign", "ml_dsa_sign_message"),
			("mldsa_sign_hash_message", "pqc_mldsa_sign", "ml_dsa_sign_hash_message"),
			("mldsa_sign_prehashed", "pqc_mldsa_sign_prehashed", "ml_dsa_sign_prehashed"),
			("mlkem_encapsulate", "pqc_mlkem_encapsulate", "ml_kem_encaps"),
			("mlkem_decapsulate", "pqc_mlkem_decapsulate", "ml_kem_decaps"),
			("mldsa_import_key", "pqc_mldsa_import", None),
			("mlkem_import_key", "pqc_mlkem_import", None),
			("mlkem_shared_derive", "pqc_mlkem_derive", "ml_kem_shared_derive"),
		)
	]
	require(api["operations"] == expected_operations, "PQC device operation/executor mismatch")
	private = {k["name"] for k in schema.get("private_kernels", [])}
	require(all(op["private_kernel"] is None or op["private_kernel"] in private for op in api["operations"]), "PQC device API references a missing private kernel")


def validate_cryptography_pqc_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Cryptography PQC schema_version must be 1")
	require(schema.get("family") == "cryptography_pqc", "Cryptography PQC family is invalid")
	validate_pqc_device_api(schema)
	prehashes = schema.get("prehashes")
	require(isinstance(prehashes, list) and all(isinstance(row, dict)
		and type(row.get("oid_suffix")) is int and type(row.get("digest_bytes")) is int
		for row in prehashes), "ML-DSA prehash identifiers and extents must be integers")
	require(prehashes == MLDSA_PREHASH_LAYOUTS, "ML-DSA prehash layout mismatch")
	require(schema.get("domain") == "crypto::pqc", "Cryptography PQC domain is invalid")
	require(schema.get("dtype") == "u8", "Cryptography PQC dtype must be u8")
	contracts = schema.get("contracts")
	require(isinstance(contracts, list) and contracts, "Cryptography PQC contracts must be non-empty")
	contract_names: set[str] = set()
	for index, contract in enumerate(contracts):
		where = f"Cryptography PQC contracts[{index}]"
		require(isinstance(contract, dict), f"{where} must be an object")
		require(
			set(contract) == {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes"},
			f"{where} fields are incomplete or unknown",
		)
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		require(isinstance(contract["input_kinds"], list), f"{where}.input_kinds must be a list")
		require(all(k == "matrix" for k in contract["input_kinds"]), f"{where}.input_kinds must be all matrix")
		require(contract["output_kinds"] == ["matrix"], f"{where}.output_kinds must contain one Matrix")
		require(contract["shape_rule"] in {"match_input", "explicit"}, f"{where}.shape_rule is invalid")
		require(contract["dtype_rule"] == "u8", f"{where}.dtype_rule must be u8")
		validate_semantic_attributes(contract["attributes"], where)
	kernels = schema.get("kernels")
	require(isinstance(kernels, list) and kernels, "Cryptography PQC kernels must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernel_ids: set[str] = set()
	seen_sources: set[str] = set()
	seen_parameters: set[tuple[int, str]] = set()
	for index, kernel in enumerate(kernels):
		where = f"Cryptography PQC kernels[{index}]"
		require(isinstance(kernel, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernel_ids), ("source", seen_sources)):
			value = kernel.get(key)
			require(isinstance(value, str) and value, f"{where}.{key} must be non-empty")
			validate_unique(value, seen, f"{where}.{key}")
		parameter = kernel.get("parameter_set")
		require(type(parameter) is int and parameter in MLDSA_WIRE_LAYOUTS, f"{where}.parameter_set is invalid")
		hash_message = kernel["name"].endswith("_verify_hash_message")
		prehashed = kernel["name"].endswith("_verify_prehashed")
		suffix = "_hash_message" if hash_message else ("_prehashed" if prehashed else "")
		validate_unique((parameter, suffix), seen_parameters, f"{where}.parameter_set")
		require(kernel["kernel_id"] == f"CryptographyMlDsa{parameter}Verify{suffix.title().replace('_', '')}U8", f"{where}.kernel_id mismatch")
		source_parameter = "" if parameter == 65 and not suffix else str(parameter)
		require(kernel["source"] == f"src/slang/crypto/pqc/mldsa/mldsa{source_parameter}_verify{suffix}.slang", f"{where}.source mismatch")
		require(kernel["name"] == f"ml_dsa_{parameter}_verify{suffix}", f"{where}.name does not match parameter_set")
		require(kernel.get("semantic_operation") == f"ml_dsa_{parameter}_verify{suffix}_batch", f"{where}.semantic_operation does not match parameter_set")
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
		fields = kernel.get("push_fields")
		require(isinstance(fields, list) and fields, f"{where}.push_fields must be non-empty")
		seen_fields: set[str] = set()
		for field in fields:
			require(isinstance(field, list) and len(field) == 2, f"{where}.push_fields entry is invalid")
			fname, scalar_type = field
			require(isinstance(fname, str) and IDENTIFIER.fullmatch(fname) is not None, f"{where}.push field name is invalid")
			validate_unique(fname, seen_fields, f"{where}.push field name")
			require(scalar_type == "uint32", f"{where}.push field type must be uint32")
		require(len(fields) * 4 <= 128, f"{where}.push constants exceed Vulkan minimum guarantee")
		expected_fields = ["input_index", "offsets_index", "lengths_index", "sigs_index", "pks_index", "results_index", "batch_size", "message_bytes", "context_offset", "context_length"] + (["algorithm"] if prehashed or hash_message else [])
		require(fields == [[name, "uint32"] for name in expected_fields], f"{where}.push_fields ABI mismatch")
		contract = next(row for row in contracts if row["name"] == kernel["semantic_operation"])
		expected_attributes = ["batch_size", "context_offset", "context_length"] + (["algorithm"] if prehashed or hash_message else [])
		require(contract["attributes"] == [[name, "unsigned_integer"] for name in expected_attributes], f"{where} semantic attributes mismatch")
		require(contract["input_kinds"] == ["matrix"] * 5, f"{where} semantic inputs mismatch")

	private = schema.get("private_kernels", [])
	require(isinstance(private, list), "PQC private_kernels must be a list")
	private_abis = {
		"ml_kem_keygen": (["seed_index", "private_index", "public_index", "workspace_index", "k"], [0, 1, 3], [1, 2, 3]),
		"ml_kem_encaps": (["seed_index", "public_index", "shared_index", "ciphertext_index", "status_index", "workspace_index", "k"], [0, 2, 5], [2, 3, 4, 5]),
		"ml_kem_decaps": (["private_index", "ciphertext_index", "shared_index", "workspace_index", "k"], [0, 2, 3], [2, 3]),
		"ml_kem_shared_derive": (["shared_index", "info_index", "output_index", "info_len", "output_len"], [0], [2]),
	}
	private_abis["ml_dsa_keygen"] = (["seed_index", "private_index", "public_index", "parameter_set"], [0, 1], [1, 2])
	private_abis["ml_dsa_sign"] = (["seed_index", "private_index", "mu_index", "signature_index", "status_index", "workspace_index", "parameter_set"], [0, 1, 5], [3, 4, 5])
	private_abis["ml_dsa_sign_message"] = (["seed_index", "private_index", "message_index", "context_index", "signature_index", "status_index", "workspace_index", "parameter_set", "message_length", "context_length"], [0, 1, 6], [4, 5, 6])
	private_abis["ml_dsa_sign_prehashed"] = (["seed_index", "private_index", "digest_index", "context_index", "signature_index", "status_index", "workspace_index", "parameter_set", "algorithm", "context_length"], [0, 1, 6], [4, 5, 6])
	private_abis["ml_dsa_sign_hash_message"] = (["seed_index", "private_index", "message_index", "context_index", "signature_index", "status_index", "workspace_index", "parameter_set", "message_length", "context_length", "algorithm"], [0, 1, 6], [4, 5, 6])
	seen_private: set[str] = set()
	for kernel in private:
		require(set(kernel) == ({"stable_id", "name", "kernel_id", "source", "workgroup_size", "push_fields", "physical_write", "secret_bindings"} | ({"parameter_layouts"} if kernel.get("name", "").startswith("ml_dsa_") else set()) | ({"workspace_bytes"} if (kernel.get("name", "").startswith("ml_dsa_sign") or (kernel.get("name", "").startswith("ml_kem_") and kernel.get("name") != "ml_kem_shared_derive")) else set())), "private PQC fields are incomplete or unknown")
		name = kernel["name"]
		require(name in private_abis, "unknown private PQC kernel")
		validate_unique(name, seen_private, "private PQC name")
		mldsa = name.startswith("ml_dsa_")
		if mldsa:
			expected = [{"parameter_set": parameter, "seed_bytes": 32, "public_bytes": public, "private_bytes": private} for parameter, public, private in ((44, 1312, 2560), (65, 1952, 4032), (87, 2592, 4896))]
			if name in ("ml_dsa_sign", "ml_dsa_sign_message", "ml_dsa_sign_prehashed", "ml_dsa_sign_hash_message"):
				expected = [{"parameter_set": a, "seed_bytes": 32, "private_bytes": b, "mu_bytes": 64, "signature_bytes": c} for a, b, c in ((44, 2560, 2420), (65, 4032, 3309), (87, 4896, 4627))]
				if name != "ml_dsa_sign":
					for row in expected:
						del row["mu_bytes"]
			require(kernel["parameter_layouts"] == expected, "ML-DSA layout mismatch")
		suffix = name.removeprefix("ml_dsa_" if mldsa else "ml_kem_")
		fields, secrets, writes = private_abis[name]
		family = "MlDsa" if mldsa else "MlKem"
		source = f"src/slang/crypto/pqc/mldsa/mldsa_{suffix}_dispatch.slang" if mldsa else f"src/slang/crypto/pqc/mlkem/mlkem_{suffix}.slang"
		require(kernel["kernel_id"] == f"Cryptography{family}{suffix.title().replace('_', '')}U8", "private PQC identity mismatch")
		require(kernel["source"] == source, "private PQC source mismatch")
		require(type(kernel["stable_id"]) is int and 0 < kernel["stable_id"] <= 65535, "invalid private PQC stable ID")
		validate_unique(kernel["stable_id"], seen_ids, "private PQC stable ID")
		validate_unique(kernel["kernel_id"], seen_kernel_ids, "private PQC kernel ID")
		require(kernel["workgroup_size"] == [1, 1, 1], "PQC serial ABI requires one invocation")
		require(kernel["push_fields"] == [[field, "uint32"] for field in fields], "PQC push ABI mismatch")
		require(kernel["secret_bindings"] == secrets, "PQC secret storage classification mismatch")
		validate_physical_write(kernel["physical_write"], f"private PQC {suffix}", required=True)
		expected_writes = [{"binding": binding, "domain": "rows", "partition": "exclusive_per_workgroup", "extent": "row_width", "collision": "exclusive", "tail": "bounds_checked"} for binding in writes]
		workspace = "exclusive_per_workgroup" if name.startswith("ml_dsa_sign") or (name.startswith("ml_kem_") and name != "ml_kem_shared_derive") else "none"
		require(kernel["physical_write"] == {"writes": expected_writes, "workspace": workspace}, "PQC write ownership mismatch")
		if name.startswith("ml_dsa_sign"):
			require(type(kernel["workspace_bytes"]) is int and kernel["workspace_bytes"] == (7 + 8 + 8) * 256 * 4, "ML-DSA signing workspace extent mismatch")

		if name.startswith("ml_kem_") and name != "ml_kem_shared_derive":
			require(type(kernel["workspace_bytes"]) is int and kernel["workspace_bytes"] == 4 * 256 * 4, "ML-KEM workspace extent mismatch")

	validate_cryptography_api(schema, "verify")


def generate_mldsa_prehash_layout_shader(schema: dict[str, Any], digest: str) -> str:
	cases = "\n".join(f"\t\tcase {row['oid_suffix']}u: return {row['digest_bytes']}u;" for row in schema["prehashes"])
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// FIPS 204 Algorithms 4/5 framing and NIST CSOR hashAlgs OIDs.
// No live C++ GPU prehash donor exists. Existing signing core is unchanged.
// Digest computation, secret execution and security qualification are separate.
import shake;

uint mldsa_prehash_bytes(uint algorithm) {{
	switch (algorithm) {{
{cases}
		default: return 0u;
	}}
}}

// Absorb the shared HashML-DSA M' framing after tr. Ranges, algorithm and
// exact digest length must already be checked by the signing/verifying caller.
void mldsa_absorb_prehash_header(inout PqcShake256 sponge,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm) {{
	pqc_shake256_absorb_byte(sponge, 1u);
	pqc_shake256_absorb_byte(sponge, context_length);
	pqc_shake256_absorb_buffer(sponge, contexts, context_base, context_length);
	uint prefix[10] = {{6u, 9u, 96u, 134u, 72u, 1u, 101u, 3u, 4u, 2u}};
	for (uint i = 0u; i < 10u; ++i) pqc_shake256_absorb_byte(sponge, prefix[i]);
	pqc_shake256_absorb_byte(sponge, algorithm);
}}

void mldsa_absorb_prehash_framing(inout PqcShake256 sponge,
	RWByteAddressBuffer digests, uint digest_base, uint digest_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm) {{
	mldsa_absorb_prehash_header(sponge, contexts, context_base, context_length, algorithm);
	pqc_shake256_absorb_buffer(sponge, digests, digest_base, digest_length);
}}
"""


def generate_mldsa_prehash_digest_shader(schema: dict[str, Any], digest: str) -> str:
	cases = []
	for row in schema["prehashes"]:
		name = row["name"]
		if name in ("SHA2-224", "SHA2-256"):
			bits = int(name.split("-")[1])
			body = (f"uint words[8];\n"
				f"\t\t\tif (!sha2_u32_digest(messages, base, length, {bits}u, words)) return false;\n"
				f"\t\t\tfor (uint i = 0u; i < {row['digest_bytes'] // 4}u; ++i) digest[i] = words[i];")
		elif name.startswith("SHA2-"):
			bits = int(name.split("/")[-1] if "/" in name else name.split("-")[1])
			body = f"return sha2_u32_pair_digest(messages, base, length, {bits}u, digest);"
		elif name.startswith("SHA3-"):
			bits = int(name.split("-")[1])
			body = ("uint2 state[25];\n"
				"\t\t\tfor (uint i = 0u; i < 25u; ++i) state[i] = uint2(0u, 0u);\n"
				f"\t\t\tif (!sha3_absorb(state, messages, base, length, SHA3_{bits}_RATE_BYTES)) return false;\n"
				f"\t\t\tfor (uint i = 0u; i < {row['digest_bytes'] // 4}u; ++i) digest[i] = (i & 1u) == 0u ? state[i / 2u].x : state[i / 2u].y;")
		else:
			body = f"return pqc_shake_prehash(messages, base, length, {int(name.split('-')[1])}u, digest);"
		ending = "" if body.startswith("return ") else "\n\t\t\tbreak;"
		cases.append(f"\t\tcase {row['oid_suffix']}u: {{\n\t\t\t{body}{ending}\n\t\t}}")
	dispatch = "\n".join(cases)
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// FIPS 204 hash selector/OID/extent authority: crypto_pqc.json.
// No OA C++ HashML-DSA composition donor exists. Reuses checked digest formulas.
// Private public-data composition; not Engine or secret-operation admission.
import mldsa_prehash_layout;
import sha2_u32;
import sha2_u32_pair;
import sha3;
import shake;
import shake_prehash;

// Packed little-endian digest words; leave the unused tail and rejected output
// untouched. Selector 11/12 fixes SHAKE output to 32/64 bytes per FIPS 204.
bool mldsa_prehash_digest(
	RWByteAddressBuffer messages,
	uint base,
	uint length,
	uint algorithm,
	inout uint digest[16]
) {{
	if (mldsa_prehash_bytes(algorithm) == 0u || !pqc_buffer_range(messages, base, length)) return false;
	switch (algorithm) {{
{dispatch}
		default: return false;
	}}
	return true;
}}

// Compute mu = SHAKE256(tr || 1 || ctx_len || ctx || OID || PH(M), 64).
// Digest stays local: no intermediate storage allocation or host observation.
// All ranges are checked before hashing. Rejected output remains untouched.
bool mldsa_prehash_mu(
	RWByteAddressBuffer messages,
	uint message_base,
	uint message_length,
	RWByteAddressBuffer tr,
	uint tr_base,
	RWByteAddressBuffer contexts,
	uint context_base,
	uint context_length,
	uint algorithm,
	inout uint mu[16]
) {{
	uint digest_length = mldsa_prehash_bytes(algorithm);
	if (digest_length == 0u || context_length > 255u
		|| !pqc_buffer_range(tr, tr_base, 64u)
		|| !pqc_buffer_range(contexts, context_base, context_length)) return false;
	uint digest[16];
	if (!mldsa_prehash_digest(messages, message_base, message_length, algorithm, digest)) return false;
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	pqc_shake256_absorb_buffer(sponge, tr, tr_base, 64u);
	mldsa_absorb_prehash_header(sponge, contexts, context_base, context_length, algorithm);
	for (uint i = 0u; i < digest_length; ++i)
		pqc_shake256_absorb_byte(sponge, digest[i / 4u] >> ((i & 3u) * 8u));
	pqc_shake256_finalize(sponge);
	for (uint word = 0u; word < 16u; ++word) {{
		uint value = 0u;
		for (uint byte = 0u; byte < 4u; ++byte)
			value |= pqc_shake256_squeeze_byte(sponge) << (byte * 8u);
		mu[word] = value;
	}}
	return true;
}}
"""


def generate_mldsa_prehash_shader(schema: dict[str, Any], digest: str) -> str:
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// FIPS 204 Algorithms 4/5; no live C++ GPU prehash donor exists.
import mldsa_sign;
import mldsa_prehash_layout;
import mldsa_prehash_digest;
import mldsa_sign_workspace_layout;
import shake;

// Accept a precomputed public digest, not an arbitrary message. The fixed
// selector binds both digest length and DER OID; callers cannot supply an OID.
// SHAKE-128/256 lengths are fixed to 32/64 bytes, respectively.
bool mldsa_sign_prehashed_cached<T : IMldsaSignCache>(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer digests, uint digest_base, uint digest_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, inout T cache, inout uint signature[4627], out uint attempts) {{
	attempts = 0u;
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	uint length = mldsa_prehash_bytes(algorithm);
	if (length == 0u || digest_length != length || context_length > 255u) return false;
	if (!pqc_buffer_range(digests, digest_base, length)
		|| !pqc_buffer_range(contexts, context_base, context_length)) return false;
	for (uint i = 64u; i < 128u; ++i) if (key[i] > 255u) return false;
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	for (uint i = 64u; i < 128u; ++i) pqc_shake256_absorb_byte(sponge, key[i]);
	mldsa_absorb_prehash_framing(sponge, digests, digest_base, length,
		contexts, context_base, context_length, algorithm);
	pqc_shake256_finalize(sponge);
	uint mu[64];
	for (uint i = 0u; i < 64u; ++i) mu[i] = pqc_shake256_squeeze_byte(sponge);
	return mldsa_sign_mu_cached(parameter_set, key, mu, rnd, cache, signature, attempts);
}}
// Existing fixture callers retain their local cache.
bool mldsa_sign_prehashed(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer digests, uint digest_base, uint digest_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, inout uint signature[4627], out uint attempts) {{
	MldsaLocalSignCache cache;
	return mldsa_sign_prehashed_cached(parameter_set, key, rnd,
		digests, digest_base, digest_length, contexts, context_base, context_length,
		algorithm, cache, signature, attempts);
}}

// Workspace extent is checked before message hashing or cache access.
bool mldsa_sign_prehashed_buffer(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer digests, uint digest_base, uint digest_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, RWByteAddressBuffer workspace, uint workspace_base,
	inout uint signature[4627], out uint attempts) {{
	attempts = 0u;
	if ((workspace_base & 3u) != 0u
		|| !pqc_buffer_range(workspace, workspace_base, MLDSA_SIGN_CACHE_BYTES)) return false;
	MldsaBufferSignCache cache;
	cache.buffer = workspace;
	cache.base = workspace_base;
	return mldsa_sign_prehashed_cached(parameter_set, key, rnd,
		digests, digest_base, digest_length, contexts, context_base, context_length,
		algorithm, cache, signature, attempts);
}}

// HashML-DSA accepts the original public message; digest words stay local.
// No secret-key bytes are transported through an ordinary digest buffer.
bool mldsa_sign_hash_message_cached<T : IMldsaSignCache>(
	uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, inout T cache, inout uint signature[4627], out uint attempts) {{
	attempts = 0u;
	if (parameter_set != 44u && parameter_set != 65u && parameter_set != 87u) return false;
	uint length = mldsa_prehash_bytes(algorithm);
	if (length == 0u || context_length > 255u
		|| !pqc_buffer_range(contexts, context_base, context_length)) return false;
	for (uint i = 64u; i < 128u; ++i) if (key[i] > 255u) return false;
	uint digest[16];
	if (!mldsa_prehash_digest(messages, message_base, message_length, algorithm, digest)) return false;
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	for (uint i = 64u; i < 128u; ++i) pqc_shake256_absorb_byte(sponge, key[i]);
	mldsa_absorb_prehash_header(sponge, contexts, context_base, context_length, algorithm);
	for (uint i = 0u; i < length; ++i)
		pqc_shake256_absorb_byte(sponge, (digest[i >> 2u] >> ((i & 3u) * 8u)) & 255u);
	pqc_shake256_finalize(sponge);
	uint mu[64];
	for (uint i = 0u; i < 64u; ++i) mu[i] = pqc_shake256_squeeze_byte(sponge);
	return mldsa_sign_mu_cached(parameter_set, key, mu, rnd, cache, signature, attempts);
}}

// Reject workspace extents before hashing or touching the signing cache.
bool mldsa_sign_hash_message_buffer(uint parameter_set, uint key[4896], uint rnd[32],
	RWByteAddressBuffer messages, uint message_base, uint message_length,
	RWByteAddressBuffer contexts, uint context_base, uint context_length,
	uint algorithm, RWByteAddressBuffer workspace, uint workspace_base,
	inout uint signature[4627], out uint attempts) {{
	attempts = 0u;
	if ((workspace_base & 3u) != 0u
		|| !pqc_buffer_range(workspace, workspace_base, MLDSA_SIGN_CACHE_BYTES)) return false;
	MldsaBufferSignCache cache;
	cache.buffer = workspace;
	cache.base = workspace_base;
	return mldsa_sign_hash_message_cached(parameter_set, key, rnd,
		messages, message_base, message_length, contexts, context_base, context_length,
		algorithm, cache, signature, attempts);
}}
"""


def generate_mldsa_sign_workspace_layout_shader(schema: dict[str, Any], digest: str) -> str:
	kernel = next(row for row in schema["private_kernels"] if row["name"] == "ml_dsa_sign")
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Maximum NTT(s1/s2/t0) cache for all FIPS 204 parameter sets.
static const uint MLDSA_SIGN_CACHE_WORDS = {kernel['workspace_bytes'] // 4}u;
static const uint MLDSA_SIGN_CACHE_BYTES = {kernel['workspace_bytes']}u;
"""


def generate_mldsa_sign_shader(kernel: dict[str, Any], digest: str) -> str:
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	layouts = kernel["parameter_layouts"]
	prehashed = kernel["name"] == "ml_dsa_sign_prehashed"
	hash_message = kernel["name"] == "ml_dsa_sign_hash_message"
	message = kernel["name"] != "ml_dsa_sign"
	prehash_import = "\nimport mldsa_prehash_layout;" if prehashed or hash_message else ""
	input_dimensions = """
	storage_buffers[push.message_index].GetDimensions(message_size);
	storage_buffers[push.context_index].GetDimensions(context_size);""" if message else "\n\tstorage_buffers[push.mu_index].GetDimensions(mu_size);"
	input_check = """push.message_length > 0xfffffffcu || push.context_length > 255u
		|| message_size != max(4u, (push.message_length + 3u) & ~3u)
		|| context_size != max(4u, (push.context_length + 3u) & ~3u)""" if message else "mu_size != 64u"
	input_load = "" if message else """
	for (uint i = 0u; i < 64u; ++i)
		mu[i] = (storage_buffers[push.mu_index].Load(i & ~3u) >> (8u * (i & 3u))) & 255u;"""
	call = """mldsa_sign_message_buffer(parameter, key, rnd,
		storage_buffers[push.message_index], 0u, push.message_length,
		storage_buffers[push.context_index], 0u, push.context_length,
		true, storage_buffers[push.workspace_index], 0u, signature, attempts)""" if message else "mldsa_sign_mu_buffer(parameter, key, mu, rnd, storage_buffers[push.workspace_index], 0u, signature, attempts)"
	if prehashed:
		input_dimensions = """\n\tstorage_buffers[push.digest_index].GetDimensions(message_size);
	storage_buffers[push.context_index].GetDimensions(context_size);"""
		input_check = """mldsa_prehash_bytes(push.algorithm) == 0u || push.context_length > 255u
		|| message_size != mldsa_prehash_bytes(push.algorithm)
		|| context_size != max(4u, (push.context_length + 3u) & ~3u)"""
		call = """mldsa_sign_prehashed_buffer(parameter, key, rnd,
		storage_buffers[push.digest_index], 0u, message_size,
		storage_buffers[push.context_index], 0u, push.context_length,
		push.algorithm, storage_buffers[push.workspace_index], 0u, signature, attempts)"""
	if hash_message:
		input_check = "mldsa_prehash_bytes(push.algorithm) == 0u || " + input_check
		call = """mldsa_sign_hash_message_buffer(parameter, key, rnd,
		storage_buffers[push.message_index], 0u, push.message_length,
		storage_buffers[push.context_index], 0u, push.context_length,
		push.algorithm, storage_buffers[push.workspace_index], 0u, signature, attempts)"""
	workspace_import = "\nimport mldsa_sign_workspace_layout;"
	workspace_dimension = "\n\tuint workspace_size;\n\tstorage_buffers[push.workspace_index].GetDimensions(workspace_size);"
	workspace_check = " || workspace_size != MLDSA_SIGN_CACHE_BYTES"
	def sizes(field: str) -> str:
		return " : ".join(f"parameter == {row['parameter_set']}u ? {row[field]}u" for row in layouts) + " : 0u"
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Private FIPS 204 {"hashed message/context" if hash_message else "prehashed digest/context" if prehashed else "pure message/context" if message else "internal-mu"} adapter; no live C++ GPU signing donor exists.
// Formula unchanged. Function-local secret spills remain unqualified.
import attributes;
import {"mldsa_prehash" if prehashed or hash_message else "mldsa_sign"};{prehash_import}{workspace_import}
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	if (any(tid != uint3(0u, 0u, 0u))) return;
	uint parameter = push.parameter_set;
	uint key_len = {sizes('private_bytes')};
	uint sig_len = {sizes('signature_bytes')};
	uint seed_size, key_size, {"message_size, context_size" if message else "mu_size"}, sig_size, status_size;
	storage_buffers[push.seed_index].GetDimensions(seed_size);
	storage_buffers[push.private_index].GetDimensions(key_size);
{input_dimensions}
	storage_buffers[push.signature_index].GetDimensions(sig_size);
	storage_buffers[push.status_index].GetDimensions(status_size);{workspace_dimension}
	if (key_len == 0u || seed_size != 32u || key_size != key_len || {input_check}
		|| sig_size != ((sig_len + 3u) & ~3u) || status_size != 4u{workspace_check}) return;
	uint key[4896], rnd[32], {"" if message else "mu[64], "}signature[4627];
	for (uint i = 0u; i < 4896u; ++i) key[i] = 0u;
	for (uint i = 0u; i < 4627u; ++i) signature[i] = 0u;
	for (uint i = 0u; i < key_len; ++i)
		key[i] = (storage_buffers[push.private_index].Load(i & ~3u) >> (8u * (i & 3u))) & 255u;
	for (uint i = 0u; i < 32u; ++i)
		rnd[i] = (storage_buffers[push.seed_index].Load(i & ~3u) >> (8u * (i & 3u))) & 255u;
{input_load}
	uint attempts;
	bool success = {call};
	// Always overwrite the complete public output, including aligned padding.
	// Failure exposes no partial signature and never exposes rejection counts.
	for (uint i = 0u; i < sig_size; i += 4u) {{
		uint word = 0u;
		for (uint j = 0u; j < 4u; ++j)
			if (success && i + j < sig_len) word |= signature[i + j] << (8u * j);
		storage_buffers[push.signature_index].Store(i, word);
	}}
	storage_buffers[push.status_index].Store(0u, success ? 1u : 0u);
}}
"""


def generate_mldsa_keygen_shader(kernel: dict[str, Any], digest: str) -> str:
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	layouts = kernel["parameter_layouts"]
	public_sizes = " : ".join(f"parameter == {row['parameter_set']}u ? {row['public_bytes']}u" for row in layouts) + " : 0u"
	private_sizes = " : ".join(f"parameter == {row['parameter_set']}u ? {row['private_bytes']}u" for row in layouts) + " : 0u"
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Private FIPS 204 adapter; no live C++ GPU keygen donor exists.
// Existing vector-proven formulas are unchanged; secret spills remain unqualified.
import attributes;
import mldsa_keygen;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	if (any(tid != uint3(0u, 0u, 0u))) return;
	uint parameter = push.parameter_set;
	if (parameter != 44u && parameter != 65u && parameter != 87u) return;
	uint public_len = {public_sizes};
	uint private_len = {private_sizes};
	uint seed_size, private_size, public_size;
	storage_buffers[push.seed_index].GetDimensions(seed_size);
	storage_buffers[push.private_index].GetDimensions(private_size);
	storage_buffers[push.public_index].GetDimensions(public_size);
	if (seed_size != 32u || private_size != private_len || public_size != public_len) return;
	uint seed[32], public_key[2592], private_key[4896];
	for (uint i = 0u; i < 2592u; ++i) public_key[i] = 0u;
	for (uint i = 0u; i < 4896u; ++i) private_key[i] = 0u;
	for (uint i = 0u; i < 32u; ++i) {{
		uint word = storage_buffers[push.seed_index].Load(i & ~3u);
		seed[i] = (word >> (8u * (i & 3u))) & 255u;
	}}
	mldsa_keygen_internal(parameter, seed, public_key, private_key);
	for (uint i = 0u; i < public_len; i += 4u)
		storage_buffers[push.public_index].Store(i, public_key[i] | (public_key[i+1u] << 8u) | (public_key[i+2u] << 16u) | (public_key[i+3u] << 24u));
	for (uint i = 0u; i < private_len; i += 4u)
		storage_buffers[push.private_index].Store(i, private_key[i] | (private_key[i+1u] << 8u) | (private_key[i+2u] << 16u) | (private_key[i+3u] << 24u));
}}
"""


def generate_mlkem_keygen_shader(kernel: dict[str, Any], digest: str) -> str:
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Private FIPS 203 Algorithm 16 adapter. C++ oaPqc.md has no live donor.
// One workgroup owns all output words; host preflight admits k and exact extents.
import attributes;
import mlkem_kem;
import mlkem_kpke;
struct PushConstants {{
	uint seed_index;
	uint private_index;
	uint public_index;
	uint workspace_index;
	uint k;
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("ml_kem_keygen")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	if (any(tid != uint3(0u, 0u, 0u)) || push.k < 2u || push.k > 4u) return;
	uint seed_size, private_size, public_size, workspace_size;
	storage_buffers[push.seed_index].GetDimensions(seed_size);
	storage_buffers[push.private_index].GetDimensions(private_size);
	storage_buffers[push.public_index].GetDimensions(public_size);
	storage_buffers[push.workspace_index].GetDimensions(workspace_size);
	uint el = 384u * push.k + 32u, dl = 768u * push.k + 96u;
	if (seed_size != 64u || private_size != dl || public_size != el || workspace_size != {kernel['workspace_bytes']}u) return;
	uint d[8], z[32], ek[1568], dk[3168];
	for (uint i = 0u; i < 1568u; ++i) ek[i] = 0u;
	for (uint i = 0u; i < 3168u; ++i) dk[i] = 0u;
	for (uint i = 0u; i < 8u; ++i) d[i] = storage_buffers[push.seed_index].Load(4u * i);
	for (uint i = 0u; i < 32u; ++i) {{
		uint word = storage_buffers[push.seed_index].Load(32u + (i & ~3u));
		z[i] = (word >> (8u * (i & 3u))) & 255u;
	}}
	if (!mlkem_keygen_buffer(d, z, push.k, storage_buffers[push.workspace_index], 0u, ek, dk)) return;
	for (uint i = 0u; i < el; i += 4u)
		storage_buffers[push.public_index].Store(i, ek[i] | (ek[i+1u] << 8u) | (ek[i+2u] << 16u) | (ek[i+3u] << 24u));
	for (uint i = 0u; i < dl; i += 4u)
		storage_buffers[push.private_index].Store(i, dk[i] | (dk[i+1u] << 8u) | (dk[i+2u] << 16u) | (dk[i+3u] << 24u));
}}
"""


def generate_mlkem_kem_shader(kernel: dict[str, Any], digest: str) -> str:
	encaps = kernel["name"] == "ml_kem_encaps"
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	# All arithmetic remains in the independently vector-proven mlkem_kem module.
	# Public status reports encapsulation-key validity only. Decapsulation has
	# no rejection status; the FIPS implicit-rejection result stays private.
	body = """
	uint seed_size, public_size, shared_size, ciphertext_size, status_size, workspace_size;
	storage_buffers[push.workspace_index].GetDimensions(workspace_size);
	storage_buffers[push.seed_index].GetDimensions(seed_size);
	storage_buffers[push.public_index].GetDimensions(public_size);
	storage_buffers[push.shared_index].GetDimensions(shared_size);
	storage_buffers[push.ciphertext_index].GetDimensions(ciphertext_size);
	storage_buffers[push.status_index].GetDimensions(status_size);
	if (seed_size != 32u || public_size != el || shared_size != 32u || ciphertext_size != cl || status_size != 4u || workspace_size != 4096u) return;
	uint ek[1568], message[32], key[32], ciphertext[1568];
	for (uint i = 0u; i < 1568u; ++i) { ek[i] = 0u; ciphertext[i] = 0u; }
	for (uint i = 0u; i < el; ++i) {
		uint word = storage_buffers[push.public_index].Load(i & ~3u);
		ek[i] = (word >> (8u * (i & 3u))) & 255u;
	}
	bool valid = mlkem_encapsulation_key_valid(ek, push.k, el);
	storage_buffers[push.status_index].Store(0u, uint(valid));
	if (!valid) {
		for (uint i = 0u; i < 32u; i += 4u) storage_buffers[push.shared_index].Store(i, 0u);
		for (uint i = 0u; i < cl; i += 4u) storage_buffers[push.ciphertext_index].Store(i, 0u);
		return;
	}
	for (uint i = 0u; i < 32u; ++i) {
		uint word = storage_buffers[push.seed_index].Load(i & ~3u);
		message[i] = (word >> (8u * (i & 3u))) & 255u;
	}
	if (!mlkem_encaps_buffer(ek, message, push.k, storage_buffers[push.workspace_index], 0u, key, ciphertext)) return;
	for (uint i = 0u; i < cl; i += 4u)
		storage_buffers[push.ciphertext_index].Store(i, ciphertext[i] | (ciphertext[i+1u] << 8u) | (ciphertext[i+2u] << 16u) | (ciphertext[i+3u] << 24u));
""" if encaps else """
	uint private_size, ciphertext_size, shared_size, workspace_size;
	storage_buffers[push.workspace_index].GetDimensions(workspace_size);
	storage_buffers[push.private_index].GetDimensions(private_size);
	storage_buffers[push.ciphertext_index].GetDimensions(ciphertext_size);
	storage_buffers[push.shared_index].GetDimensions(shared_size);
	if (private_size != 768u * push.k + 96u || ciphertext_size != cl || shared_size != 32u || workspace_size != 4096u) return;
	// Only internally generated retained keys are admitted by the private owner.
	uint dk[3168], ciphertext[1568], key[32];
	for (uint i = 0u; i < 3168u; ++i) dk[i] = 0u;
	for (uint i = 0u; i < 1568u; ++i) ciphertext[i] = 0u;
	for (uint i = 0u; i < private_size; ++i) {
		uint word = storage_buffers[push.private_index].Load(i & ~3u);
		dk[i] = (word >> (8u * (i & 3u))) & 255u;
	}
	for (uint i = 0u; i < cl; ++i) {
		uint word = storage_buffers[push.ciphertext_index].Load(i & ~3u);
		ciphertext[i] = (word >> (8u * (i & 3u))) & 255u;
	}
	if (!mlkem_decaps_buffer(dk, ciphertext, push.k, storage_buffers[push.workspace_index], 0u, key)) return;
"""
	body = body.replace("workspace_size != 4096u", f"workspace_size != {kernel['workspace_bytes']}u")
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Private FIPS 203 adapter; C++ oaPqc.md has no live donor.
// Serial correctness route; secret spills and physical timing remain unqualified.
import attributes;
import mlkem_kem;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	if (any(tid != uint3(0u, 0u, 0u)) || push.k < 2u || push.k > 4u) return;
	uint el = 384u * push.k + 32u;
	uint cl = 32u * (push.k == 4u ? 11u * push.k + 5u : 10u * push.k + 4u);
{body}	for (uint i = 0u; i < 32u; i += 4u)
		storage_buffers[push.shared_index].Store(i, key[i] | (key[i+1u] << 8u) | (key[i+2u] << 16u) | (key[i+3u] << 24u));
}}
"""


def generate_mlkem_shared_derive_shader(kernel: dict[str, Any], digest: str) -> str:
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	# Declassification boundary: the 32-byte shared secret is absorbed from
	# private storage; only keyed-hash output becomes public.
	return f"""// Generated by tool/gen/fn/generate.py; do not edit.
// schema_sha256={digest}
// Keyed SHAKE-256 declassification consumer for ML-KEM shared secrets.
// Serial correctness route; secret spills and physical timing remain unqualified.
import attributes;
import shake;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	if (any(tid != uint3(0u, 0u, 0u))) return;
	uint shared_size, info_size, output_size;
	storage_buffers[push.shared_index].GetDimensions(shared_size);
	storage_buffers[push.info_index].GetDimensions(info_size);
	storage_buffers[push.output_index].GetDimensions(output_size);
	if (shared_size != 32u || output_size != push.output_len || push.output_len == 0u
		|| push.output_len > 4096u || (push.output_len & 3u) != 0u
		|| info_size < (push.info_len + 3u) / 4u * 4u) return;
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	pqc_shake256_absorb_buffer(sponge, storage_buffers[push.shared_index], 0u, 32u);
	pqc_shake256_absorb_buffer(sponge, storage_buffers[push.info_index], 0u, push.info_len);
	pqc_shake256_finalize(sponge);
	for (uint i = 0u; i < push.output_len; i += 4u) {{
		uint word = pqc_shake256_squeeze_byte(sponge)
			| (pqc_shake256_squeeze_byte(sponge) << 8u)
			| (pqc_shake256_squeeze_byte(sponge) << 16u)
			| (pqc_shake256_squeeze_byte(sponge) << 24u);
		storage_buffers[push.output_index].Store(i, word);
	}}
}}
"""


def generate_mldsa_verify_shader(kernel: dict[str, Any], schema_hash: str) -> str:
	parameter = kernel["parameter_set"]
	pk_bytes, sig_bytes = MLDSA_WIRE_LAYOUTS[parameter]
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	workgroup = ", ".join(str(size) for size in kernel["workgroup_size"])
	prehashed = kernel["name"].endswith("_verify_prehashed")
	hash_message = kernel["name"].endswith("_verify_hash_message")
	verification = "mldsa_verify_hash_message" if hash_message else ("mldsa_verify_prehashed" if prehashed else "mldsa_verify_external")
	algorithm_argument = ", push.algorithm" if prehashed or hash_message else ""
	mode = "original-message prehash" if hash_message else ("precomputed-digest" if prehashed else "pure")
	return f"// Generated by tool/gen/fn/generate.py; do not edit.\n// schema_sha256={schema_hash}\n" + f"""// ML-DSA-{parameter} batch external {mode} verification with a shared context.
// Experimental public-only route; physical-device qualification remains separate.
// FIPS 204 Algorithms {"5/8" if prehashed or hash_message else "3/8"}. New implementation; OA historical shader was a stub.
import attributes;
import mldsa_verification;
import shake;

struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u8")]
[status("experimental")]
[shader("compute")]
[numthreads({workgroup})]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {{
	uint idx = dispatch_thread_id.x;
	if (idx >= push.batch_size || idx > 0xffffffffu / 4u) return;
	uint output_base = idx * 4u;
	if (!pqc_buffer_range(storage_buffers[push.results_index], output_base, 4u)) return;
	storage_buffers[push.results_index].Store(output_base, 0u);
	if (!pqc_buffer_range(storage_buffers[push.offsets_index], output_base, 4u)
		|| !pqc_buffer_range(storage_buffers[push.lengths_index], output_base, 4u)
		|| idx > 0xffffffffu / {pk_bytes}u
		|| idx > 0xffffffffu / {sig_bytes}u) return;
	uint offset = storage_buffers[push.offsets_index].Load(output_base);
	uint length = storage_buffers[push.lengths_index].Load(output_base);
	// Allocation padding is not part of the semantic message buffer.
	if (offset > push.message_bytes || length > push.message_bytes - offset
		|| push.context_length > 255u
		|| push.context_offset > push.message_bytes
		|| push.context_length > push.message_bytes - push.context_offset) return;
	bool valid = {verification}(
		{parameter}u,
		storage_buffers[push.pks_index], idx * {pk_bytes}u, {pk_bytes}u,
		storage_buffers[push.sigs_index], idx * {sig_bytes}u, {sig_bytes}u,
		storage_buffers[push.input_index], offset, length,
		storage_buffers[push.input_index], push.context_offset, push.context_length{algorithm_argument});
	storage_buffers[push.results_index].Store(output_base, valid ? 1u : 0u);
}}
"""


def pqc_device_outputs(root: Path, schema: dict[str, Any], schema_hash: str) -> dict[Path, str]:
	api = schema["device_api"]
	lines = ["/// FIPS 203 device-key parameter set.", "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]", "pub enum MlKemParameters {"]
	for row in api["mlkem_parameters"]:
		lines += [f"\t/// ML-KEM-{row['parameter']}.", f"\tMlKem{row['parameter']},"]
	lines += ["}", "", "impl MlKemParameters {"]
	for method, field, typ, vis, doc in (
		("k", "k", "u32", "pub(crate)", "Private polynomial-vector width."),
		("public_key_size", "public_key_bytes", "usize", "pub", "Encoded encapsulation-key bytes."),
		("ciphertext_size", "ciphertext_bytes", "usize", "pub", "Encoded ciphertext bytes."),
	):
		lines += [f"\t/// {doc}", "\t#[must_use]", f"\t{vis} const fn {method}(self) -> {typ} {{", "\t\tmatch self {"]
		for row in api["mlkem_parameters"]: lines.append(f"\t\t\tSelf::MlKem{row['parameter']} => {row[field]},")
		lines += ["\t\t}", "\t}"]
	lines += ["}", "", "/// FIPS 204 HashML-DSA identity and exact digest extent.", "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]", "pub enum MlDsaPrehash {"]
	for name, row in zip(api["prehash_rust_names"], schema["prehashes"]):
		lines += [f"\t/// {row['name']}, {row['digest_bytes']} digest bytes.", f"\t{name},"]
	lines += ["}", "", "impl MlDsaPrehash {"]
	for method, field, typ, vis, doc in (("id", "oid_suffix", "u32", "pub(crate)", "Private NIST OID suffix."), ("digest_bytes", "digest_bytes", "usize", "pub", "Exact encoded digest bytes.")):
		lines += [f"\t/// {doc}", "\t#[must_use]", f"\t{vis} const fn {method}(self) -> {typ} {{", "\t\tmatch self {"]
		for name, row in zip(api["prehash_rust_names"], schema["prehashes"]): lines.append(f"\t\t\tSelf::{name} => {row[field]},")
		lines += ["\t\t}", "\t}"]
	lines += ["}"]
	template_root = FN_ROOT / "template/crypto/pqc"
	body = (template_root / "device.rs.in").read_text().replace("{{metadata}}", "\n".join(lines))
	for op in api["operations"]:
		require(f"pub fn {op['name']}(" in body and f".{op['executor']}(" in body, "PQC full-body template does not implement schema operation")
	binding = (template_root / "binding.rs.in").read_text()
	kem_cases = "\n".join(f"\t\t{r['parameter']} => Ok(device::MlKemParameters::MlKem{r['parameter']})," for r in api["mlkem_parameters"])
	dsa_cases = "\n".join(f"\t\t{p} => Ok(oa::crypto::pqc::MlDsaParameters::MlDsa{p})," for p in (44,65,87))
	hash_cases = "\n".join(f"\t\t\"{row['name']}\" => Ok(device::MlDsaPrehash::{name})," for name,row in zip(api["prehash_rust_names"],schema["prehashes"]))
	binding = binding.replace("{{kem_cases}}", kem_cases).replace("{{dsa_cases}}", dsa_cases).replace("{{hash_cases}}", hash_cases)
	stub = (template_root / "binding.pyi.in").read_text()
	aliases = ["DeviceMlDsaKey", "DeviceMlKemKey", "DeviceSharedSecret", "DeviceErasure", "DeviceSignature", "DeviceEncapsulation", "DeviceDecapsulation"]
	py = "from .._native import " + ", ".join(aliases) + "\n"
	for op in api["operations"]: py += f"from .._native import crypto_device_{op['name']} as device_{op['name']}\n"
	for name in ("mldsa_verify_message", "mldsa_verify_hash_message", "mldsa_verify_prehashed"): py += f"from .._native import crypto_device_{name} as device_{name}\n"
	return {
		root / "src/rs/crypto/pqc/device/operations.gen.rs": format_rust(banner(schema_hash, "//", "crypto_pqc.json") + body, root),
		root / "src/py/pyo3/crypto/pqc_device.gen.rs": format_rust(banner(schema_hash, "//", "crypto_pqc.json") + binding, root),
		root / "src/py/oa/_gen/crypto_pqc_device.py": banner(schema_hash, "#", "crypto_pqc.json") + py,
		root / "src/py/oa/_gen/crypto_pqc_device_native.pyi": banner(schema_hash, "#", "crypto_pqc.json") + stub,
	}
