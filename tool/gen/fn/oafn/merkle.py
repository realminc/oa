"""Merkle schema validation, cooperative shaders and output inventory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .model import validate_physical_write, validate_unique
from .util import IDENTIFIER, banner, format_rust, require

def generate_merkle_cooperative(kernel: dict[str, Any], mark: str, layout: dict[str, Any]) -> str:
	capacity = kernel['shared_nodes']
	fields = '\n'.join(f'\tuint {n};' for n, _ in kernel['push_fields'])
	return mark + f'''// Shared-memory subtree reduction; one workgroup owns each subtree root.
// Selected-hash public Merkle operation, no host digest copies.
import attributes;
import merkle_general;
import merkle_general_hash;
static const uint MERKLE_FRAME_VERSION = {layout['version']}u;
static const uint MERKLE_HEADER_BYTES = {layout['header_bytes']}u;
static const uint MERKLE_MAX_DEPTH = {layout['max_depth']}u;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
groupshared uint nodes[{capacity} * 16];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {{
	uint row = group.x + group.y * push.grid_width;
	// Every early exit is uniform across the workgroup, before any barrier.
	if (row >= push.count || push.algorithm >= 10u || push.leaf_count < 2u
		|| push.leaf_count > {capacity}u || (push.leaf_count & (push.leaf_count - 1u)) != 0u) return;
	uint bytes = merkle_digest_bytes(push.algorithm);
	uint input_extent, output_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	if (push.count * push.leaf_count > input_extent / bytes
		|| push.count > output_extent / bytes) return;
	uint words = bytes / 4u;
	uint current[16], sibling[16];
	for (uint i = lane; i < push.leaf_count; i += 128u)
		for (uint w = 0u; w < words; ++w)
			nodes[i * 16u + w] = storage_buffers[push.input_index].Load((row * push.leaf_count + i) * bytes + w * 4u);
	GroupMemoryBarrierWithGroupSync();
	for (uint count = push.leaf_count; count > 1u; count /= 2u) {{
		uint half = count / 2u;
		for (uint i = lane; i < half; i += 128u) {{
			for (uint w = 0u; w < 16u; ++w) {{
				current[w] = nodes[(2u * i) * 16u + w];
				sibling[w] = nodes[(2u * i + 1u) * 16u + w];
			}}
			merkle_general_pair(current, sibling, false, push.algorithm);
			for (uint w = 0u; w < words; ++w) nodes[i * 16u + w] = current[w];
		}}
		GroupMemoryBarrierWithGroupSync();
	}}
	if (lane == 0u)
		for (uint w = 0u; w < words; ++w)
			storage_buffers[push.output_index].Store(row * bytes + w * 4u, nodes[w]);
}}\n'''


MERKLE_SHADER_BODIES = {
'merkle_tree_reduce': '''	uint input_extent, output_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	if (push.leaf_count == 0u || push.count != (push.leaf_count + 1u) / 2u
		|| push.leaf_count > input_extent / bytes || push.count > output_extent / bytes) return;
	uint current[16], sibling[16];
	uint left = index * 2u, right = min(left + 1u, push.leaf_count - 1u);
	for (uint i = 0u; i < 16u; ++i) { current[i] = 0u; sibling[i] = 0u; }
	for (uint i = 0u; i < bytes / 4u; ++i) {
		current[i] = storage_buffers[push.input_index].Load(left * bytes + i * 4u);
		sibling[i] = storage_buffers[push.input_index].Load(right * bytes + i * 4u);
	}
	merkle_general_pair(current, sibling, false, push.algorithm);
	for (uint i = 0u; i < bytes / 4u; ++i) storage_buffers[push.output_index].Store(index * bytes + i * 4u, current[i]);''',
'merkle_frame_init': '''	uint indices_extent, output_extent;
	storage_buffers[push.indices_index].GetDimensions(indices_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	if (push.frame_bytes != MERKLE_HEADER_BYTES + merkle_depth(push.leaf_count) * bytes
		|| push.count > indices_extent / 4u || push.count > output_extent / push.frame_bytes) return;
	uint leaf_index = storage_buffers[push.indices_index].Load(index * 4u);
	uint base = index * push.frame_bytes;
	for (uint i = 0u; i < push.frame_bytes; i += 4u) storage_buffers[push.output_index].Store(base + i, 0u);
	storage_buffers[push.output_index].Store(base, MERKLE_FRAME_VERSION);
	storage_buffers[push.output_index].Store(base + 4u, push.leaf_count);
	storage_buffers[push.output_index].Store(base + 8u, leaf_index);
	storage_buffers[push.output_index].Store(base + 12u, push.algorithm);
	storage_buffers[push.output_index].Store(base + 16u, uint(leaf_index < push.leaf_count));''',
'merkle_frame_append': '''	uint level_extent, indices_extent, previous_extent, output_extent;
	storage_buffers[push.level_index].GetDimensions(level_extent);
	storage_buffers[push.indices_index].GetDimensions(indices_extent);
	storage_buffers[push.previous_index].GetDimensions(previous_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	uint depth = merkle_depth(push.leaf_count);
	uint expected_count = push.leaf_count;
	for (uint i = 0u; i < push.level && i < MERKLE_MAX_DEPTH; ++i) expected_count = (expected_count + 1u) / 2u;
	if (push.level >= depth || push.source_count != expected_count
		|| push.frame_bytes != MERKLE_HEADER_BYTES + depth * bytes
		|| push.count > indices_extent / 4u || push.source_count > level_extent / bytes
		|| push.count > previous_extent / push.frame_bytes || push.count > output_extent / push.frame_bytes) return;
	uint leaf_index = storage_buffers[push.indices_index].Load(index * 4u);
	// Rust's schema declares previous/output as the same private physical frame
	// with a fresh semantic version; only this invocation mutates its own row.
	if (leaf_index >= push.leaf_count) return;
	uint source_index = leaf_index >> push.level;
	uint sibling = source_index ^ 1u;
	if (sibling >= push.source_count) sibling = source_index;
	uint target = index * push.frame_bytes + MERKLE_HEADER_BYTES + push.level * bytes;
	for (uint i = 0u; i < bytes; i += 4u)
		storage_buffers[push.output_index].Store(target + i, storage_buffers[push.level_index].Load(sibling * bytes + i));''',
'merkle_frame_verify': '''	uint leaves_extent, frames_extent, roots_extent, output_extent;
	storage_buffers[push.leaves_index].GetDimensions(leaves_extent);
	storage_buffers[push.frames_index].GetDimensions(frames_extent);
	storage_buffers[push.roots_index].GetDimensions(roots_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	uint depth = merkle_depth(push.leaf_count);
	if (push.frame_bytes != MERKLE_HEADER_BYTES + depth * bytes
		|| push.count > leaves_extent / bytes || push.count > frames_extent / push.frame_bytes
		|| (push.root_count != 1u && push.root_count != push.count)
		|| push.root_count > roots_extent / bytes || push.count > output_extent / 4u) return;
	uint base = index * push.frame_bytes;
	uint leaf_index = storage_buffers[push.frames_index].Load(base + 8u);
	bool valid = storage_buffers[push.frames_index].Load(base) == MERKLE_FRAME_VERSION
		&& storage_buffers[push.frames_index].Load(base + 4u) == push.leaf_count
		&& storage_buffers[push.frames_index].Load(base + 12u) == push.algorithm
		&& storage_buffers[push.frames_index].Load(base + 16u) == 1u && leaf_index < push.leaf_count;
	uint current[16], sibling[16];
	for (uint i = 0u; i < 16u; ++i) { current[i] = 0u; sibling[i] = 0u; }
	for (uint i = 0u; i < bytes / 4u; ++i) current[i] = storage_buffers[push.leaves_index].Load(index * bytes + i * 4u);
	uint source_count = push.leaf_count;
	for (uint level = 0u; level < depth && valid; ++level) {
		for (uint i = 0u; i < bytes / 4u; ++i)
			sibling[i] = storage_buffers[push.frames_index].Load(base + MERKLE_HEADER_BYTES + level * bytes + i * 4u);
		if ((leaf_index & 1u) == 0u && leaf_index + 1u >= source_count)
			for (uint i = 0u; i < bytes / 4u; ++i) valid = valid && sibling[i] == current[i];
		merkle_general_pair(current, sibling, (leaf_index & 1u) != 0u, push.algorithm);
		leaf_index >>= 1u; source_count = (source_count + 1u) / 2u;
	}
	uint root_base = (push.root_count == 1u ? 0u : index) * bytes;
	for (uint i = 0u; i < bytes / 4u; ++i) valid = valid && current[i] == storage_buffers[push.roots_index].Load(root_base + i * 4u);
	storage_buffers[push.output_index].Store(index * 4u, uint(valid));''',
}


def validate_merkle_schema(schema: dict[str, Any]) -> None:
	require(set(schema) <= {'schema_version', 'family', 'domain', 'dtype', 'algorithms', 'proof_layout', 'contracts', 'kernels', 'signature_records'} and {'schema_version', 'family', 'domain', 'dtype', 'algorithms', 'proof_layout', 'contracts', 'kernels'} <= set(schema)
		and schema['schema_version'] == 1 and schema['family'] == 'cryptography_merkle' and schema['domain'] == 'crypto::merkle' and schema['dtype'] == 'u32', 'General Merkle schema identity differs')
	require(schema['proof_layout'] == {'version': 2, 'header_bytes': 20, 'max_depth': 24}, 'General Merkle frame layout differs')
	names = ['sha256', 'sha512', 'sha3_256', 'sha3_512', 'shake128', 'shake256', 'poseidon2_babybear16', 'poseidon2_babybear24', 'poseidon2_koalabear16', 'poseidon2_koalabear24']
	require(isinstance(schema['algorithms'], list) and len(schema['algorithms']) == len(names), 'General Merkle algorithm inventory differs')
	for index, algorithm in enumerate(schema['algorithms']):
		require(set(algorithm) == {'name', 'variant', 'identifier', 'digest_bytes', 'dtype'} and algorithm['name'] == names[index]
			and type(algorithm['identifier']) is int and algorithm['identifier'] == index
			and algorithm['digest_bytes'] == (64 if index in {1, 3} else 32) and algorithm['dtype'] == ('u32' if index >= 6 else 'u8')
			and isinstance(algorithm['variant'], str) and IDENTIFIER.fullmatch(algorithm['variant']), 'General Merkle algorithm framing differs')
	expected_fields = {
		'merkle_tree_reduce': ['input_index', 'output_index', 'count', 'grid_width', 'leaf_count', 'algorithm'],
		'merkle_frame_init': ['indices_index', 'output_index', 'count', 'grid_width', 'leaf_count', 'algorithm', 'frame_bytes'],
		'merkle_frame_append': ['level_index', 'indices_index', 'previous_index', 'output_index', 'count', 'grid_width', 'leaf_count', 'source_count', 'algorithm', 'frame_bytes', 'level'],
		'merkle_frame_verify': ['leaves_index', 'frames_index', 'roots_index', 'output_index', 'count', 'grid_width', 'leaf_count', 'algorithm', 'frame_bytes', 'root_count'],
		'merkle_subtree_cooperative_256': ['input_index', 'output_index', 'count', 'grid_width', 'leaf_count', 'algorithm'],
		'merkle_subtree_cooperative_512': ['input_index', 'output_index', 'count', 'grid_width', 'leaf_count', 'algorithm'],
	}
	require(len(schema['kernels']) == 8 and len(schema['contracts']) == 5, 'General Merkle operation inventory differs')
	seen = set()
	for kernel in schema['kernels']:
		name = kernel['name']
		base = name[:-10] if name.endswith('_poseidon2') else name
		require(base in expected_fields, 'General Merkle kernel is unknown'); validate_unique(name, seen, 'General Merkle kernel')
		cooperative = name.startswith('merkle_subtree_cooperative_')
		expected_keys = {'name', 'stable_id', 'kernel_id', 'source', 'shared_nodes', 'workgroup_size', 'push_fields', 'semantic_operation', 'physical_write'} if cooperative else {'name', 'stable_id', 'kernel_id', 'source', 'workgroup_size', 'push_fields', 'semantic_operation', 'physical_write'}
		require(set(kernel) == expected_keys, 'General Merkle kernel fields differ')
		require(kernel['push_fields'] == [[n, 'uint32'] for n in expected_fields[base]] and kernel['workgroup_size'] == [128, 1, 1]
			and kernel['source'] == f'src/slang/crypto/merkle/{name}.gen.slang'
			and kernel['kernel_id'] == 'Cryptography' + ''.join(w.title() for w in name.split('_')) + 'U32', 'General Merkle ABI differs')
		if cooperative:
			capacity = int(name.rsplit('_', 1)[1])
			require(kernel['shared_nodes'] == capacity and kernel['semantic_operation'] == 'merkle_subtree_reduce', 'General Merkle cooperative ABI differs')
		else:
			require(kernel['semantic_operation'] == base, 'General Merkle ABI differs')
		validate_physical_write(kernel['physical_write'], 'General Merkle', required=True)
		inputs = 1 if base in {'merkle_tree_reduce', 'merkle_frame_init'} else 3
		contract = next((c for c in schema['contracts'] if c['name'] == kernel['semantic_operation']), None)
		attrs = ['leaf_count', 'algorithm'] + (['level'] if base == 'merkle_frame_append' else [])
		require(contract == {'name': kernel['semantic_operation'], 'input_kinds': ['matrix'] * (1 if cooperative else inputs), 'output_kinds': ['matrix'], 'shape_rule': 'explicit', 'dtype_rule': 'explicit', 'attributes': [[n, 'unsigned_integer'] for n in attrs], 'mutated_inputs': [2] if base == 'merkle_frame_append' else [], 'output_alias_inputs': [2] if base == 'merkle_frame_append' else [-1]}, 'General Merkle semantic mutation contract differs')
		partition = 'exclusive_per_workgroup' if cooperative else 'exclusive_per_invocation'
		require(kernel['physical_write'] == {'writes': [{'binding': 1 if cooperative else inputs, 'domain': 'rows', 'partition': partition, 'extent': 'one_scalar' if base == 'merkle_frame_verify' else 'row_width', 'collision': 'exclusive', 'tail': 'bounds_checked'}], 'workspace': 'none'}, 'General Merkle output ownership differs')


def merkle_outputs(root: Path, schema: dict[str, Any], schema_hash: str, hash_schema: dict[str, Any]) -> dict[Path, str]:
	mark = banner(schema_hash, '//', 'crypto_merkle.json')
	algorithms = '/// Selected parent-hash identity; digest sizes are part of the contract.\n#[derive(Clone, Copy, Debug, PartialEq, Eq)]\n#[repr(u32)]\npub enum MerkleAlgorithm {\n'
	for a in schema['algorithms']:
		algorithms += f"\t{a['variant']} = {a['identifier']},\n"
	algorithms += '}\n\nimpl MerkleAlgorithm {\n\t/// Exact digest byte extent.\n\tpub const fn digest_bytes(self) -> usize {\n\t\tmatch self {\n'
	for a in schema['algorithms']: algorithms += f"\t\t\tSelf::{a['variant']} => {a['digest_bytes']},\n"
	algorithms += '\t\t}\n\t}\n\t/// Storage type of supplied leaf and root rows.\n\tpub const fn dtype(self) -> DType {\n\t\tmatch self {\n'
	for a in schema['algorithms']: algorithms += f"\t\t\tSelf::{a['variant']} => DType::{a['dtype'].upper()},\n"
	algorithms += '\t\t}\n\t}\n\t/// Stable API hash spelling.\n\tpub const fn name(self) -> &\'static str {\n\t\tmatch self {\n'
	for a in schema['algorithms']: algorithms += f"\t\t\tSelf::{a['variant']} => \"{a['name']}\",\n"
	algorithms += '\t\t}\n\t}\n\t/// Field-hash algorithms lowered to the poseidon2 kernels.\n\tpub(crate) const fn is_poseidon2(self) -> bool {\n\t\tmatches!(self, '
	algorithms += ' | '.join(f"Self::{a['variant']}" for a in schema['algorithms'] if a['dtype'] == 'u32')
	algorithms += ')\n\t}\n}\n'
	template = (FN_ROOT / 'template/crypto/merkle/operations.rs.in').read_text()
	values = dict(algorithms=algorithms, header_bytes=str(schema['proof_layout']['header_bytes']))
	require(set(re.findall(r'\{\{([^{}]+)\}\}', template)) == set(values), 'General Merkle operation template differs')
	outputs = {root / 'src/rs/crypto/merkle/operations.gen.rs': format_rust(mark + re.sub(r'\{\{([^{}]+)\}\}', lambda m: values[m[1]], template), root)}
	poseidon = mark + '// Pinned Plonky3 permutation; OA Merkle domain is distinct from field XOF.\nimport prime31;\nimport poseidon2;\n\nvoid merkle_poseidon2_pair(uint message[32], uint algorithm, out uint result[16]) {\n'
	for algorithm in schema['algorithms'][6:]:
		index = algorithm['identifier']
		p = next(p for p in hash_schema['poseidon2_profiles'] if 'poseidon2_' + p['name'] == algorithm['name'])
		prime, rate, name = p['prime'], p['width'] - p['capacity'], p['name']
		factor = (-pow(prime, -1, 1 << 32)) % (1 << 32); r2 = (1 << 64) % prime; one = (1 << 32) % prime
		poseidon += f'''\tif (algorithm == {index}u) {{
		uint state[24]; for (uint i = 0u; i < 24u; ++i) state[i] = 0u;
		const uint prime = {prime}u, factor = {factor}u;
		state[{rate}u] = prime31_multiply(0x4f414d32u, {r2}u, prime, factor);
		state[{rate+1}u] = prime31_multiply(16u, {r2}u, prime, factor);
		uint cursor = 0u;
		for (uint i = 0u; i < 16u; ++i) {{
			state[cursor] = prime31_add(state[cursor], prime31_multiply(message[i] % prime, {r2}u, prime, factor), prime);
			if (++cursor == {rate}u) {{ poseidon2_{name}_permutation(state); cursor = 0u; }}
		}}
		state[cursor] = prime31_add(state[cursor], {one}u, prime);
		poseidon2_{name}_permutation(state);
		for (uint i = 0u; i < 16u; ++i) result[i] = 0u;
		for (uint i = 0u; i < 8u; ++i) result[i] = prime31_multiply(state[i], 1u, prime, factor);
		return;
	}}\n'''
	poseidon += '\tfor (uint i = 0u; i < 16u; ++i) result[i] = 0u;\n}\n'
	outputs[root / 'src/slang/core/math/merkle_poseidon2.slang'] = poseidon
	for kernel in schema['kernels']:
		name = kernel['name']
		if name.startswith('merkle_subtree_cooperative_'):
			outputs[root / kernel['source']] = generate_merkle_cooperative(kernel, mark, schema['proof_layout'])
			continue
		poseidon2 = name.endswith('_poseidon2')
		base = name[:-10] if poseidon2 else name
		imports = 'import merkle_general;\nimport merkle_general_hash;' if poseidon2 else 'import merkle_general_hash;'
		guard = 'push.algorithm < 6u || push.algorithm >= 10u' if poseidon2 else 'push.algorithm >= 6u'
		body_text = MERKLE_SHADER_BODIES[base].replace('merkle_general_pair(', 'merkle_general_pair(' if poseidon2 else 'merkle_hash_pair(')
		fields = '\n'.join(f'\tuint {n};' for n, _ in kernel['push_fields'])
		body = mark + f'''// Selected-hash public Merkle graph operation, no host digest copies.
import attributes;
{imports}
static const uint MERKLE_FRAME_VERSION = {schema['proof_layout']['version']}u;
static const uint MERKLE_HEADER_BYTES = {schema['proof_layout']['header_bytes']}u;
static const uint MERKLE_MAX_DEPTH = {schema['proof_layout']['max_depth']}u;
uint merkle_depth(uint count) {{
	uint depth = 0u;
	while (count > 1u) {{ count = (count + 1u) / 2u; ++depth; }}
	return depth;
}}
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
[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {{
	uint index = tid.x + tid.y * push.grid_width;
	if (index >= push.count || {guard} || push.leaf_count == 0u || push.leaf_count > (1u << MERKLE_MAX_DEPTH)) return;
	uint bytes = merkle_digest_bytes(push.algorithm);
{body_text}
}}\n'''
		outputs[root / kernel['source']] = body
	bindings = (FN_ROOT / 'template/crypto/merkle/binding.rs.in').read_text()
	arms = '\n'.join(f'\t\t"{a["name"]}" => Ok(oa::crypto::merkle::MerkleAlgorithm::{a["variant"]}),' for a in schema['algorithms'])
	outputs[root / 'src/py/pyo3/crypto/merkle.gen.rs'] = format_rust(mark + bindings.replace('{{algorithms}}', arms), root)
	stub = banner(schema_hash, '#', 'crypto_merkle.json') + 'from .._native import Matrix\n\nclass DeviceMerkleTree:\n    @property\n    def root(self) -> Matrix: ...\n    @property\n    def leaf_count(self) -> int: ...\n    @property\n    def algorithm(self) -> str: ...\n\ndef crypto_merkle_build(leaves: Matrix, hash: str) -> DeviceMerkleTree: ...\ndef crypto_merkle_subtree_roots(leaves: Matrix, hash: str, subtree_leaves: int) -> Matrix: ...\ndef crypto_merkle_proofs(tree: DeviceMerkleTree, indices: Matrix) -> Matrix: ...\ndef crypto_merkle_verify(leaves: Matrix, frames: Matrix, roots: Matrix, hash: str, leaf_count: int) -> Matrix: ...\n'
	outputs[root / 'src/py/oa/_gen/crypto_merkle_native.pyi'] = stub
	outputs[root / 'src/py/oa/_gen/crypto_merkle.py'] = banner(schema_hash, '#', 'crypto_merkle.json') + 'from .._native import (DeviceMerkleTree, crypto_merkle_build as merkle_build, crypto_merkle_subtree_roots as merkle_subtree_roots, crypto_merkle_proofs as merkle_proofs, crypto_merkle_verify as merkle_verify)\n'
	return outputs
