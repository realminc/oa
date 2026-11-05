"""Hash schema validation, Poseidon2 and digest shader emitters."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .crypto import FIXED_HASH_HELPERS, validate_cryptography_api
from .model import validate_physical_write, validate_semantic_attributes, validate_unique
from .util import IDENTIFIER, banner, format_rust, require

def validate_poseidon2_profiles(schema: dict[str, Any]) -> None:
	profiles = schema.get('poseidon2_profiles')
	require(isinstance(profiles, list) and len(profiles) == 4, 'Poseidon2 needs four pinned profiles')
	expected = {'babybear16': (2013265921, 16, 7, 13), 'babybear24': (2013265921, 24, 7, 21),
		'koalabear16': (2130706433, 16, 3, 20), 'koalabear24': (2130706433, 24, 3, 23)}
	seen = set()
	for p in profiles:
		require(isinstance(p, dict) and set(p) == {'name', 'prime', 'width', 'sbox_degree', 'capacity', 'half_full_rounds', 'initial_constants', 'final_constants', 'internal_constants', 'diagonal'}, 'Poseidon2 profile fields differ')
		require(p['name'] in expected, 'Poseidon2 profile is unknown'); validate_unique(p['name'], seen, 'Poseidon2 profile')
		prime, width, degree, partial = expected[p['name']]
		require((p['prime'], p['width'], p['sbox_degree'], p['capacity'], p['half_full_rounds']) == (prime, width, degree, 8, 4), 'Poseidon2 parameters differ')
		for k in ('initial_constants', 'final_constants'):
			require(isinstance(p[k], list) and len(p[k]) == 4 and all(isinstance(r, list) and len(r) == width for r in p[k]), 'Poseidon2 full round dimensions differ')
		require(isinstance(p['internal_constants'], list) and len(p['internal_constants']) == partial and isinstance(p['diagonal'], list) and len(p['diagonal']) == width, 'Poseidon2 partial round dimensions differ')
		values = p['internal_constants'] + p['diagonal'] + sum(p['initial_constants'] + p['final_constants'], [])
		require(all(type(v) is int and 0 <= v < prime for v in values), 'Poseidon2 constants must be canonical field elements')
	provenance = schema.get('poseidon2_provenance')
	require(isinstance(provenance, dict) and provenance.get('commit') == 'f98331687e47106a64ab82a77e2e1d0d66c9161b'
		and provenance.get('license') == 'MIT' and provenance.get('sponge') == 'oa_poseidon2_field_xof_v1', 'Poseidon2 reference identity differs')


def generate_poseidon2_constants(schema: dict[str, Any], schema_hash: str) -> str:
	body = banner(schema_hash, '//', 'crypto_hash.json') + '// Pinned Plonky3 constants and round schedule (MIT; see NOTICE.md).\nimport prime31;\nimport poseidon2_field;\n\n'
	for p in schema['poseidon2_profiles']:
		prime, width, name = p['prime'], p['width'], p['name']
		factor = (-pow(prime, -1, 1 << 32)) % (1 << 32)
		def array(label: str, values: list[int]) -> str:
			words = [(v << 32) % prime for v in values]
			return f'static const uint POSEIDON2_{name.upper()}_{label}[{len(words)}] = {{\n' + ''.join('\t' + ', '.join(str(v)+'u' for v in words[i:i+8]) + ',\n' for i in range(0, len(words), 8)) + '};\n'
		body += array('INITIAL', sum(p['initial_constants'], [])) + array('FINAL', sum(p['final_constants'], [])) + array('PARTIAL', p['internal_constants']) + array('DIAGONAL', p['diagonal'] + [0]*(24-width))
		body += f'''\nvoid poseidon2_{name}_permutation(inout uint state[24]) {{
	const uint prime = {prime}u;
	const uint factor = {factor}u;
	poseidon2_external(state, {width}u, prime);
	for (uint round = 0u; round < 4u; ++round) {{
		for (uint i = 0u; i < {width}u; ++i)
			state[i] = poseidon2_sbox(prime31_add(state[i], POSEIDON2_{name.upper()}_INITIAL[round * {width}u + i], prime), {p['sbox_degree']}u, prime, factor);
		poseidon2_external(state, {width}u, prime);
	}}
	uint diagonal[24];
	for (uint i = 0u; i < 24u; ++i) diagonal[i] = POSEIDON2_{name.upper()}_DIAGONAL[i];
	for (uint round = 0u; round < {len(p['internal_constants'])}u; ++round) {{
		state[0] = poseidon2_sbox(prime31_add(state[0], POSEIDON2_{name.upper()}_PARTIAL[round], prime), {p['sbox_degree']}u, prime, factor);
		poseidon2_internal(state, diagonal, {width}u, prime, factor);
	}}
	for (uint round = 0u; round < 4u; ++round) {{
		for (uint i = 0u; i < {width}u; ++i)
			state[i] = poseidon2_sbox(prime31_add(state[i], POSEIDON2_{name.upper()}_FINAL[round * {width}u + i], prime), {p['sbox_degree']}u, prime, factor);
		poseidon2_external(state, {width}u, prime);
	}}
}}\n\n'''
	return body.rstrip() + '\n'


def generate_poseidon2_shader(kernel: dict[str, Any], schema: dict[str, Any], schema_hash: str) -> str:
	profile = next(p for p in schema['poseidon2_profiles'] if p['name'] == kernel['poseidon2']['profile'])
	prime, width, name = profile['prime'], profile['width'], profile['name']
	permutation = kernel['poseidon2']['permutation']
	factor = (-pow(prime, -1, 1 << 32)) % (1 << 32)
	r2 = (1 << 64) % prime; one = (1 << 32) % prime; rate = width - profile['capacity']
	fields = '\n'.join(f'\tuint {n};' for n, _ in kernel['push_fields'])
	output = f'{width}u' if permutation else 'push.output_elements'
	guard = f'push.input_elements != {width}u' if permutation else 'push.input_elements > 0x3fffffffu || push.output_elements == 0u || push.output_elements > 0x3fffffffu'
	body = banner(schema_hash, '//', 'crypto_hash.json') + f'''// Public field data only. Pinned permutation; OA's explicitly framed field XOF.
import attributes;
import prime31;
import poseidon2;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	uint row = tid.x;
	if (row >= push.count || {guard}) return;
	uint input_extent, output_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	uint input_stride = push.input_elements * 4u;
	uint output_stride = {output} * 4u;
	if ((input_stride != 0u && push.count > input_extent / input_stride) || push.count > output_extent / output_stride) return;
	const uint prime = {prime}u;
	const uint factor = {factor}u;
	const uint r2 = {r2}u;
	uint state[24];
	for (uint i = 0u; i < 24u; ++i) state[i] = 0u;
'''
	if permutation:
		body += f'''\tfor (uint i = 0u; i < {width}u; ++i)
		state[i] = prime31_multiply(storage_buffers[push.input_index].Load(row * input_stride + i * 4u) % prime, r2, prime, factor);
	poseidon2_{name}_permutation(state);
	for (uint i = 0u; i < {width}u; ++i)
		storage_buffers[push.output_index].Store(row * output_stride + i * 4u, prime31_multiply(state[i], 1u, prime, factor));
}}\n'''
	else:
		body += f'''\t// Capacity: explicit OA-P2 domain and original field-element length.
	state[{rate}u] = prime31_multiply(0x4f415032u, r2, prime, factor);
	state[{rate+1}u] = prime31_multiply(push.input_elements, r2, prime, factor);
	uint cursor = 0u;
	for (uint i = 0u; i < push.input_elements; ++i) {{
		uint value = storage_buffers[push.input_index].Load(row * input_stride + i * 4u) % prime;
		state[cursor] = prime31_add(state[cursor], prime31_multiply(value, r2, prime, factor), prime);
		if (++cursor == {rate}u) {{ poseidon2_{name}_permutation(state); cursor = 0u; }}
	}}
	// A mandatory delimiter block distinguishes every input length, including
	// empty and exact-rate messages. Longer outputs preserve shorter prefixes.
	state[cursor] = prime31_add(state[cursor], {one}u, prime);
	poseidon2_{name}_permutation(state);
	cursor = 0u;
	for (uint i = 0u; i < push.output_elements; ++i) {{
		if (cursor == {rate}u) {{ poseidon2_{name}_permutation(state); cursor = 0u; }}
		storage_buffers[push.output_index].Store(row * output_stride + i * 4u, prime31_multiply(state[cursor++], 1u, prime, factor));
	}}
}}\n'''
	return body


def validate_cryptography_hash_schema(schema: dict[str, Any]) -> None:
	require(schema.get("schema_version") == 1, "Cryptography Hash schema_version must be 1")
	require(schema.get("family") == "cryptography_hash", "Cryptography Hash family is invalid")
	require(schema.get("domain") == "crypto::hash", "Cryptography Hash domain is invalid")
	require(schema.get("dtype") == "u8", "Cryptography Hash dtype must be u8")
	validate_poseidon2_profiles(schema)
	require(schema.get("proof_layout") == {"version": 1, "header_bytes": 16, "hash_bytes": 32, "max_depth": 32}
		and all(type(v) is int for v in schema["proof_layout"].values()), "Merkle proof wire layout mismatch")
	contracts = schema.get("contracts")
	require(isinstance(contracts, list) and contracts, "Cryptography Hash contracts must be non-empty")
	contract_names: set[str] = set()
	for index, contract in enumerate(contracts):
		where = f"Cryptography Hash contracts[{index}]"
		require(isinstance(contract, dict), f"{where} must be an object")
		require(
			(set(contract) - {"donor_op", "control_flow"})
			== {"name", "input_kinds", "output_kinds", "shape_rule", "dtype_rule", "attributes"},
			f"{where} fields are incomplete or unknown",
		)
		name = contract["name"]
		require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.name is invalid")
		validate_unique(name, contract_names, f"{where}.name")
		expected_inputs = 2 if name == "merkle_proofs" else 3 if name == "verify_merkle_proofs" else 1
		require(contract["input_kinds"] == ["matrix"] * expected_inputs, f"{where}.input_kinds differs from operation")
		require(contract["output_kinds"] == ["matrix"], f"{where}.output_kinds must contain one Matrix")
		require(contract["shape_rule"] in {"match_input", "explicit"}, f"{where}.shape_rule is invalid")
		require(
			contract["dtype_rule"]
			== (
				"u32"
				if name == "verify_merkle_proofs" or name.startswith("poseidon2_")
				else "match_input"
				if "donor_op" in contract
				else "u8"
			),
			f"{where}.dtype_rule is invalid",
		)
		validate_semantic_attributes(contract["attributes"], where)
	kernels = schema.get("kernels")
	require(isinstance(kernels, list) and kernels, "Cryptography Hash kernels must be non-empty")
	seen_ids: set[int] = set()
	seen_names: set[str] = set()
	seen_kernel_ids: set[str] = set()
	seen_sources: set[str] = set()
	for index, kernel in enumerate(kernels):
		where = f"Cryptography Hash kernels[{index}]"
		require(isinstance(kernel, dict), f"{where} must be an object")
		for key, seen in (("name", seen_names), ("kernel_id", seen_kernel_ids), ("source", seen_sources)):
			value = kernel.get(key)
			require(isinstance(value, str) and value, f"{where}.{key} must be non-empty")
			validate_unique(value, seen, f"{where}.{key}")
		stable_id = kernel.get("stable_id")
		require(isinstance(stable_id, int) and 0 < stable_id <= 65535, f"{where}.stable_id is invalid")
		validate_unique(stable_id, seen_ids, f"{where}.stable_id")
		require(kernel.get("semantic_operation") in contract_names, f"{where}.semantic_operation is unknown")
		require(kernel.get("auxiliary_operations", []) == (["merkle_proofs"] if kernel["name"] == "merkle_reduce" else []), f"{where} auxiliary ownership differs")
		require(kernel.get("dtype", "u8") == ("u32" if kernel["name"] == "merkle_proof_verify" or kernel["name"].startswith("poseidon2_") else "u8"), f"{where} kernel dtype differs")
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
			name, scalar_type = field
			require(isinstance(name, str) and IDENTIFIER.fullmatch(name) is not None, f"{where}.push field name is invalid")
			validate_unique(name, seen_fields, f"{where}.push field name")
			require(scalar_type == "uint32", f"{where}.push field type must be uint32")
		require(len(fields) * 4 <= 128, f"{where}.push constants exceed Vulkan minimum guarantee")
		if kernel["name"] in {"merkle_proof_gather", "merkle_proof_verify"}:
			gather = kernel["name"] == "merkle_proof_gather"
			names = ("level_index", "indices_index", "output_index", "count", "leaf_count", "source_count", "depth", "level", "frame_bytes") if gather else ("leaves_index", "proofs_index", "roots_index", "output_index", "count", "root_count", "frame_bytes", "leaf_count")
			require(fields == [[name, "uint32"] for name in names], f"{where} Merkle proof ABI differs")
			validate_physical_write(kernel.get("physical_write"), where, required=True)
			require(kernel["physical_write"] == {"writes": [{"binding": 2 if gather else 3,
				"domain": "rows", "partition": "exclusive_per_invocation",
				"extent": "row_width" if gather else "one_scalar", "collision": "exclusive",
				"tail": "bounds_checked"}], "workspace": "none"}, f"{where} Merkle proof ownership differs")
		if kernel['name'].startswith('poseidon2_'):
			profile = kernel.get('poseidon2')
			require(isinstance(profile, dict) and set(profile) == {'profile', 'permutation'} and type(profile['permutation']) is bool, 'Poseidon2 kernel profile differs')
			require(profile['profile'] in {p['name'] for p in schema['poseidon2_profiles']}, 'Poseidon2 kernel profile unknown')
			require(kernel['name'] == 'poseidon2_' + profile['profile'] + ('_permute' if profile['permutation'] else '')
				and kernel['semantic_operation'] == kernel['name'] and kernel['source'] == f"src/slang/crypto/hash/{kernel['name']}.gen.slang"
				and fields == [[n, 'uint32'] for n in ['input_index', 'output_index', 'count', 'input_elements'] + ([] if profile['permutation'] else ['output_elements'])], 'Poseidon2 ABI differs')
			validate_physical_write(kernel.get('physical_write'), where, required=True)
			require(kernel['physical_write'] == {'writes': [{'binding': 1, 'domain': 'rows', 'partition': 'exclusive_per_invocation', 'extent': 'row_width', 'collision': 'exclusive', 'tail': 'bounds_checked'}], 'workspace': 'none'}, 'Poseidon2 ownership differs')
		if kernel["name"] in FIXED_HASH_HELPERS:
			module, helper, size, word_type = FIXED_HASH_HELPERS[kernel["name"]]
			digest = kernel.get("digest")
			require(isinstance(digest, dict) and type(digest.get("bytes")) is int
				and digest == {"module": module, "helper": helper, "bytes": size, "word_type": word_type},
				f"{where}.digest differs from checked helper ABI")
			require(kernel["semantic_operation"] == kernel["name"]
				and kernel["source"] == f"src/slang/crypto/hash/{kernel['name']}.gen.slang"
				and fields == [[name, "uint32"] for name in ("input_index", "output_index", "count", "message_len")],
				f"{where} fixed digest adapter ABI differs")
			validate_physical_write(kernel.get("physical_write"), where, required=True)
			require(kernel["physical_write"] == {"writes": [{"binding": 1, "domain": "rows",
				"partition": "exclusive_per_invocation", "extent": "row_width",
				"collision": "exclusive", "tail": "bounds_checked"}], "workspace": "none"},
				f"{where} fixed digest write ownership differs")
		else:
			require("digest" not in kernel, f"{where} digest helper is unknown")

	validate_cryptography_api(schema, "hash")


def generate_merkle_proof_shader(kernel: dict[str, Any], schema_hash: str, layout: dict[str, Any]) -> str:
	path = FN_ROOT / "template/crypto" / f"{kernel['name']}.slang.in"
	values = {"fields": "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"]),
		"workgroup": ", ".join(str(v) for v in kernel["workgroup_size"]),
		"name": kernel["name"], "dtype": kernel.get("dtype", "u8"), **{key: str(value) for key, value in layout.items()}}
	body = path.read_text()
	require(set(re.findall(r"\{\{([^{}]+)\}\}", body)) == set(values), "Merkle shader template fields differ")
	return banner(schema_hash, "//", "crypto_hash.json") + re.sub(r"\{\{([^{}]+)\}\}", lambda m: values[m[1]], body)


def generate_fixed_hash_shader(kernel: dict[str, Any], schema_hash: str) -> str:
	digest = kernel["digest"]
	word_bytes = 8 if digest["word_type"] == "uint2" else 4
	words = digest["bytes"] // word_bytes
	if word_bytes == 8:
		store = "storage_buffers[push.output_index].Store<uint>(base + word * 8u, digest[word].x);\n\t\tstorage_buffers[push.output_index].Store<uint>(base + word * 8u + 4u, digest[word].y);"
	else:
		store = "storage_buffers[push.output_index].Store<uint>(base + word * 4u, digest[word]);"
	fields = "\n".join(f"\tuint {name};" for name, _ in kernel["push_fields"])
	workgroup = ", ".join(str(value) for value in kernel["workgroup_size"])
	return banner(schema_hash, "//", "crypto_hash.json") + f'''// No OA C++ fixed SHA-2/SHA-3 batch donor exists (cryptoFnHash.toml).
// Adaptation: schema-owned ABI adapter over existing checked Slang formulas.
// Each invocation owns one whole digest row; all rows contain complete words.
// Input row starts may be unaligned; the imported helpers own packed reads.
import attributes;
import {digest['module']};

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
	uint index = dispatch_thread_id.x;
	if (index >= push.count) return;
	uint input_extent, output_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	// Divide before multiplying, including zero-length messages. Host admission
	// also checks logical byte extents; descriptor extents include word padding.
	if ((input_extent & 3u) != 0u || (output_extent & 3u) != 0u
		|| (push.message_len != 0u && push.count > input_extent / push.message_len)
		|| push.count > output_extent / {digest['bytes']}u) return;
	{digest['word_type']} digest[{words}];
	if (!{digest['helper']}(storage_buffers[push.input_index],
		index * push.message_len, push.message_len, digest)) return;
	uint base = index * {digest['bytes']}u;
	[loop]
	for (uint word = 0u; word < {words}u; ++word) {{
		{store}
	}}
}}
'''


def cryptography_hash_binding_outputs(root: Path, schema: dict[str, Any], schema_hash: str) -> dict[Path, str]:
	apis = schema["rust_api"]
	rust = banner(schema_hash, "//", "crypto_hash.json") + 'use pyo3::prelude::*;\nuse crate::{error::python_error, matrix::PythonMatrix};\n\n'
	stub = banner(schema_hash, "#", "crypto_hash.json") + 'from .._native import Matrix\n\n'
	facade = banner(schema_hash, "#", "crypto_hash.json") + '"""Schema-owned public-data Vulkan batch hashing adapters."""\n\nfrom .._native import (\n'
	for api in apis:
		name = api["name"]
		variable = name in {"shake128", "shake256"} or (name.startswith("poseidon2_") and not name.endswith("_permute"))
		rust_parameters = ", ".join(f"{arg}: {'&PythonMatrix' if kind == '&Matrix' else kind}" for arg, kind in api["parameters"])
		arguments = ", ".join(f"&{arg}.inner" if kind == "&Matrix" else arg for arg, kind in api["parameters"])
		stub_parameters = ", ".join(f"{arg}: {'Matrix' if kind == '&Matrix' else 'int'}" + (" = 0" if arg == "output_length" else "") for arg, kind in api["parameters"])
		rust += "\n".join("///" + (" " + line if line else "") for line in api["doc"]) + "\n"
		rust += '#[pyfunction]\n' + ('#[pyo3(signature = (input, output_length=0))]\n' if variable else '')
		rust += f'pub(crate) fn crypto_hash_{name}({rust_parameters}) -> PyResult<PythonMatrix> {{\n'
		rust += f'\toa::crypto::hash::{name}({arguments})\n\t\t.map(PythonMatrix::wrap)\n\t\t.map_err(python_error)\n}}\n\n'
		stub += f'def crypto_hash_{name}({stub_parameters}) -> Matrix: ...\n'
		facade += f'\tcrypto_hash_{name} as hash_{name},\n'
	rust += 'pub(super) fn register(module: &Bound<\'_, PyModule>) -> PyResult<()> {\n'
	for api in apis:
		rust += f'\tmodule.add_function(wrap_pyfunction!(crypto_hash_{api["name"]}, module)?)?;\n'
	rust += '\tOk(())\n}\n'
	facade += ')\n\n__all__ = [\n' + ''.join(f'\t"hash_{api["name"]}",\n' for api in apis) + ']\n'
	reexports = 'pub use batch::{\n' + ''.join(f'\t{api["name"]},\n' for api in apis) + '};\n'
	return {
		root / 'src/py/pyo3/crypto/hash.gen.rs': format_rust(rust, root),
		root / 'src/py/oa/_gen/crypto_hash.py': facade,
		root / 'src/py/oa/_gen/crypto_hash_native.pyi': stub,
		root / 'src/rs/crypto/hash/exports.gen.rs': format_rust(banner(schema_hash, "//", "crypto_hash.json") + reexports, root),
	}
