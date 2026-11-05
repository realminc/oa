"""NTT schema validation, cooperative shaders and output inventory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .paths import FN_ROOT
from .model import validate_physical_write, validate_unique
from .util import banner, format_rust, require

NTT_SHADER_BODIES = {
'ntt_twiddles': '''	uint forward_extent, inverse_extent;
	storage_buffers[push.forward_index].GetDimensions(forward_extent);
	storage_buffers[push.inverse_index].GetDimensions(inverse_extent);
	if (push.count > forward_extent / 8u || push.count > inverse_extent / 8u) return;
	uint2 one = uint2(push.one_low, push.one_high);
	uint2 forward = prime62_power(uint2(push.root_low, push.root_high), index, modulus, push.negative_inverse, one);
	uint2 inverse = prime62_power(uint2(push.inverse_root_low, push.inverse_root_high), index, modulus, push.negative_inverse, one);
	storage_buffers[push.forward_index].Store<uint2>(index * 8u, forward);
	storage_buffers[push.inverse_index].Store<uint2>(index * 8u, inverse);''',
'ntt_reorder': '''	if (push.length < 4u || push.log_length < 2u || push.log_length > 24u || push.length != (1u << push.log_length)
		|| push.count % push.length != 0u) return;
	uint input_extent, output_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	if (push.count > input_extent / 8u || push.count > output_extent / 8u) return;
	uint row = index / push.length;
	uint coefficient = index % push.length;
	uint reversed = reversebits(coefficient) >> (32u - push.log_length);
	uint2 value = prime62_normalize(storage_buffers[push.input_index].Load<uint2>((row * push.length + reversed) * 8u), modulus);
	if (push.montgomery == 0u) value = prime62_multiply(value, uint2(push.r2_low, push.r2_high), modulus, push.negative_inverse);
	storage_buffers[push.output_index].Store<uint2>(index * 8u, value);''',
'ntt_stage': '''	if (push.length < 4u || push.length > (1u << 24u) || (push.length & (push.length - 1u)) != 0u
		|| push.span < 2u || push.span > push.length || (push.span & (push.span - 1u)) != 0u
		|| push.count % (push.length / 2u) != 0u) return;
	uint input_extent, output_extent, twiddle_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	storage_buffers[push.twiddles_index].GetDimensions(twiddle_extent);
	if (push.count > input_extent / 16u || push.count > output_extent / 16u || push.length / 2u > twiddle_extent / 8u) return;
	uint half_span = push.span / 2u;
	uint row = index / (push.length / 2u);
	uint butterfly = index % (push.length / 2u);
	uint offset = butterfly % half_span;
	uint left = row * push.length + (butterfly / half_span) * push.span + offset;
	uint right = left + half_span;
	uint2 a = storage_buffers[push.input_index].Load<uint2>(left * 8u);
	uint2 b = storage_buffers[push.input_index].Load<uint2>(right * 8u);
	uint2 twiddle = storage_buffers[push.twiddles_index].Load<uint2>(offset * (push.length / push.span) * 8u);
	b = prime62_multiply(b, twiddle, modulus, push.negative_inverse);
	uint2 sum = prime62_add(a, b, modulus);
	uint2 difference = prime62_difference(a, b, modulus);
	if (push.span == push.length) {
		uint2 normalization = uint2(push.normalization_low, push.normalization_high);
		sum = prime62_multiply(sum, normalization, modulus, push.negative_inverse);
		difference = prime62_multiply(difference, normalization, modulus, push.negative_inverse);
		if (push.montgomery == 0u) {
			sum = prime62_multiply(sum, uint2(1u, 0u), modulus, push.negative_inverse);
			difference = prime62_multiply(difference, uint2(1u, 0u), modulus, push.negative_inverse);
		}
	}
	storage_buffers[push.output_index].Store<uint2>(left * 8u, sum);
	storage_buffers[push.output_index].Store<uint2>(right * 8u, difference);''',
'ntt_pointwise': '''	uint left_extent, right_extent, output_extent;
	storage_buffers[push.left_index].GetDimensions(left_extent);
	storage_buffers[push.right_index].GetDimensions(right_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	if (push.count > left_extent / 8u || push.count > right_extent / 8u || push.count > output_extent / 8u) return;
	uint2 a = prime62_normalize(storage_buffers[push.left_index].Load<uint2>(index * 8u), modulus);
	uint2 b = prime62_normalize(storage_buffers[push.right_index].Load<uint2>(index * 8u), modulus);
	if (push.montgomery == 0u) {
		uint2 r2 = uint2(push.r2_low, push.r2_high);
		a = prime62_multiply(a, r2, modulus, push.negative_inverse);
		b = prime62_multiply(b, r2, modulus, push.negative_inverse);
	}
	uint2 result = prime62_multiply(a, b, modulus, push.negative_inverse);
	if (push.montgomery == 0u) result = prime62_multiply(result, uint2(1u, 0u), modulus, push.negative_inverse);
	storage_buffers[push.output_index].Store<uint2>(index * 8u, result);''',
}


def generate_ntt_cooperative(kernel: dict[str, Any], mark: str) -> str:
	capacity = kernel['shared_coefficients']
	fields = '\n'.join(f'\tuint {n};' for n, _ in kernel['push_fields'])
	return mark + f'''// Complete shared-memory cyclic NTT; one workgroup owns each output row.
// Radix-two butterfly identities; no OA C++ generic NTT donor exists.
import attributes;
import prime62;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
groupshared uint2 coefficients[{capacity}];
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
	if (row >= push.count || push.length < 4u || push.length > {capacity}u
		|| push.log_length < 2u || push.log_length > 24u || push.length != (1u << push.log_length)
		|| (push.modulus_low & 1u) == 0u || push.modulus_high >= 0x40000000u) return;
	uint input_extent, output_extent, twiddle_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	storage_buffers[push.twiddles_index].GetDimensions(twiddle_extent);
	uint row_bytes = push.length * 8u;
	if (push.count > input_extent / row_bytes || push.count > output_extent / row_bytes
		|| push.length / 2u > twiddle_extent / 8u) return;
	uint2 modulus = uint2(push.modulus_low, push.modulus_high);
	for (uint i = lane; i < push.length; i += 128u) {{
		uint reversed = reversebits(i) >> (32u - push.log_length);
		uint2 value = prime62_normalize(storage_buffers[push.input_index].Load<uint2>(row * row_bytes + reversed * 8u), modulus);
		if (push.montgomery == 0u) value = prime62_multiply(value, uint2(push.r2_low, push.r2_high), modulus, push.negative_inverse);
		coefficients[i] = value;
	}}
	GroupMemoryBarrierWithGroupSync();
	for (uint span = 2u; span <= push.length; span *= 2u) {{
		uint half_span = span / 2u;
		for (uint i = lane; i < push.length / 2u; i += 128u) {{
			uint offset = i % half_span;
			uint left = (i / half_span) * span + offset;
			uint right = left + half_span;
			uint2 a = coefficients[left];
			uint2 twiddle = storage_buffers[push.twiddles_index].Load<uint2>(offset * (push.length / span) * 8u);
			uint2 b = prime62_multiply(coefficients[right], twiddle, modulus, push.negative_inverse);
			coefficients[left] = prime62_add(a, b, modulus);
			coefficients[right] = prime62_difference(a, b, modulus);
		}}
		GroupMemoryBarrierWithGroupSync();
	}}
	uint2 normalization = uint2(push.normalization_low, push.normalization_high);
	for (uint i = lane; i < push.length; i += 128u) {{
		uint2 value = prime62_multiply(coefficients[i], normalization, modulus, push.negative_inverse);
		if (push.montgomery == 0u) value = prime62_multiply(value, uint2(1u, 0u), modulus, push.negative_inverse);
		storage_buffers[push.output_index].Store<uint2>(row * row_bytes + i * 8u, value);
	}}
}}\n'''


def generate_ntt_stage_fused(kernel: dict[str, Any], mark: str) -> str:
	capacity = kernel['shared_coefficients']
	fields = '\n'.join(f'\tuint {n};' for n, _ in kernel['push_fields'])
	return mark + f'''// Shared-memory tile pass performing every stage whose span fits one tile.
// One workgroup owns each aligned tile; radix-two butterfly identities.
import attributes;
import prime62;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
groupshared uint2 coefficients[{capacity}];
[kernel_name("{kernel['name']}")]
[domain("cryptography")]
[variant("generic")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {{
	uint tile = group.x + group.y * push.grid_width;
	// Every early exit is uniform across the workgroup, before any barrier.
	if (tile >= push.count || push.length <= {capacity}u || push.length > (1u << 24u)
		|| (push.length & (push.length - 1u)) != 0u || push.length % {capacity}u != 0u
		|| push.count % (push.length / {capacity}u) != 0u
		|| (push.modulus_low & 1u) == 0u || push.modulus_high >= 0x40000000u) return;
	uint input_extent, output_extent, twiddle_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	storage_buffers[push.twiddles_index].GetDimensions(twiddle_extent);
	if (push.count > input_extent / ({capacity}u * 8u) || push.count > output_extent / ({capacity}u * 8u)
		|| push.length / 2u > twiddle_extent / 8u) return;
	uint tiles_per_row = push.length / {capacity}u;
	uint row = tile / tiles_per_row;
	uint base = row * push.length + (tile % tiles_per_row) * {capacity}u;
	uint2 modulus = uint2(push.modulus_low, push.modulus_high);
	for (uint i = lane; i < {capacity}u; i += 128u) {{
		coefficients[i] = storage_buffers[push.input_index].Load<uint2>((base + i) * 8u);
	}}
	GroupMemoryBarrierWithGroupSync();
	for (uint span = 2u; span <= {capacity}u; span *= 2u) {{
		uint half_span = span / 2u;
		for (uint i = lane; i < {capacity}u / 2u; i += 128u) {{
			uint offset = i % half_span;
			uint left = (i / half_span) * span + offset;
			uint right = left + half_span;
			uint2 a = coefficients[left];
			uint2 twiddle = storage_buffers[push.twiddles_index].Load<uint2>(offset * (push.length / span) * 8u);
			uint2 b = prime62_multiply(coefficients[right], twiddle, modulus, push.negative_inverse);
			coefficients[left] = prime62_add(a, b, modulus);
			coefficients[right] = prime62_difference(a, b, modulus);
		}}
		GroupMemoryBarrierWithGroupSync();
	}}
	for (uint i = lane; i < {capacity}u; i += 128u) {{
		storage_buffers[push.output_index].Store<uint2>((base + i) * 8u, coefficients[i]);
	}}
}}\n'''


def generate_ntt_cooperative_scalar(kernel: dict[str, Any], mark: str) -> str:
	capacity = kernel['shared_coefficients']
	fields = '\n'.join(f'\tuint {n};' for n, _ in kernel['push_fields'])
	return mark + f'''// Complete shared-memory cyclic NTT over scalar coefficient words; one
// workgroup owns each output row. Montgomery R = 2^32 for primes below 2^31.
import attributes;
import prime31;
struct PushConstants {{
{fields}
}};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
groupshared uint coefficients[{capacity}];
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
	if (row >= push.count || push.length < 4u || push.length > {capacity}u
		|| push.log_length < 2u || push.log_length > 24u || push.length != (1u << push.log_length)
		|| push.modulus == 0u || (push.modulus & 1u) == 0u || push.modulus >= 0x80000000u) return;
	uint input_extent, output_extent, twiddle_extent;
	storage_buffers[push.input_index].GetDimensions(input_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	storage_buffers[push.twiddles_index].GetDimensions(twiddle_extent);
	uint row_bytes = push.length * 4u;
	if (push.count > input_extent / row_bytes || push.count > output_extent / row_bytes
		|| push.length / 2u > twiddle_extent / 4u) return;
	for (uint i = lane; i < push.length; i += 128u) {{
		uint reversed = reversebits(i) >> (32u - push.log_length);
		uint value = storage_buffers[push.input_index].Load(row * row_bytes + reversed * 4u) % push.modulus;
		if (push.montgomery == 0u) value = prime31_multiply(value, push.r2, push.modulus, push.negative_inverse);
		coefficients[i] = value;
	}}
	GroupMemoryBarrierWithGroupSync();
	for (uint span = 2u; span <= push.length; span *= 2u) {{
		uint half_span = span / 2u;
		for (uint i = lane; i < push.length / 2u; i += 128u) {{
			uint offset = i % half_span;
			uint left = (i / half_span) * span + offset;
			uint right = left + half_span;
			uint a = coefficients[left];
			uint twiddle = storage_buffers[push.twiddles_index].Load(offset * (push.length / span) * 4u);
			uint b = prime31_multiply(coefficients[right], twiddle, push.modulus, push.negative_inverse);
			coefficients[left] = prime31_add(a, b, push.modulus);
			coefficients[right] = a >= b ? a - b : push.modulus - (b - a);
		}}
		GroupMemoryBarrierWithGroupSync();
	}}
	for (uint i = lane; i < push.length; i += 128u) {{
		uint value = prime31_multiply(coefficients[i], push.normalization, push.modulus, push.negative_inverse);
		if (push.montgomery == 0u) value = prime31_multiply(value, 1u, push.modulus, push.negative_inverse);
		storage_buffers[push.output_index].Store(row * row_bytes + i * 4u, value);
	}}
}}\n'''


NTT_SCALAR_SHADER_BODIES = {
'ntt_twiddles_scalar': '''	uint forward_extent, inverse_extent;
	storage_buffers[push.forward_index].GetDimensions(forward_extent);
	storage_buffers[push.inverse_index].GetDimensions(inverse_extent);
	if (push.count > forward_extent / 4u || push.count > inverse_extent / 4u) return;
	uint forward = prime31_power(push.root, index, push.modulus, push.negative_inverse, push.one);
	uint inverse = prime31_power(push.inverse_root, index, push.modulus, push.negative_inverse, push.one);
	storage_buffers[push.forward_index].Store(index * 4u, forward);
	storage_buffers[push.inverse_index].Store(index * 4u, inverse);''',
'ntt_pointwise_scalar': '''	uint left_extent, right_extent, output_extent;
	storage_buffers[push.left_index].GetDimensions(left_extent);
	storage_buffers[push.right_index].GetDimensions(right_extent);
	storage_buffers[push.output_index].GetDimensions(output_extent);
	if (push.count > left_extent / 4u || push.count > right_extent / 4u || push.count > output_extent / 4u) return;
	uint a = storage_buffers[push.left_index].Load(index * 4u) % push.modulus;
	uint b = storage_buffers[push.right_index].Load(index * 4u) % push.modulus;
	if (push.montgomery == 0u) {
		a = prime31_multiply(a, push.r2, push.modulus, push.negative_inverse);
		b = prime31_multiply(b, push.r2, push.modulus, push.negative_inverse);
	}
	uint result = prime31_multiply(a, b, push.modulus, push.negative_inverse);
	if (push.montgomery == 0u) result = prime31_multiply(result, 1u, push.modulus, push.negative_inverse);
	storage_buffers[push.output_index].Store(index * 4u, result);''',
}


def generate_ntt_scalar(kernel: dict[str, Any], mark: str) -> str:
	fields = '\n'.join(f'\tuint {n};' for n, _ in kernel['push_fields'])
	return mark + f'''// Scalar-word NTT helper; Montgomery R = 2^32 for primes below 2^31.
import attributes;
import prime31;
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
	if (index >= push.count || push.modulus == 0u || (push.modulus & 1u) == 0u || push.modulus >= 0x80000000u) return;
{NTT_SCALAR_SHADER_BODIES[kernel['name']]}
}}\n'''


def validate_ntt_schema(schema: dict[str, Any]) -> None:
	require(set(schema) <= {'schema_version', 'family', 'domain', 'dtype', 'coefficient_layout', 'field_bits', 'min_log_length', 'max_log_length', 'contracts', 'kernels', 'signature_records'} and {'schema_version', 'family', 'domain', 'dtype', 'coefficient_layout', 'field_bits', 'min_log_length', 'max_log_length', 'contracts', 'kernels'} <= set(schema), 'NTT schema fields differ')
	require(schema['schema_version'] == 1 and schema['family'] == 'cryptography_ntt' and schema['domain'] == 'crypto::ntt' and schema['dtype'] == 'u32'
		and schema['coefficient_layout'] == {'word_bits': 32, 'words': 2, 'order': 'little_endian'} and (schema['field_bits'], schema['min_log_length'], schema['max_log_length']) == (62, 2, 24), 'NTT schema identity differs')
	expected_fields = {
		'ntt_twiddles': ['forward_index', 'inverse_index', 'count', 'grid_width', 'modulus_low', 'modulus_high', 'negative_inverse', 'root_low', 'root_high', 'inverse_root_low', 'inverse_root_high', 'one_low', 'one_high'],
		'ntt_reorder': ['input_index', 'output_index', 'count', 'grid_width', 'length', 'log_length', 'modulus_low', 'modulus_high', 'negative_inverse', 'r2_low', 'r2_high', 'montgomery'],
		'ntt_stage': ['input_index', 'twiddles_index', 'output_index', 'count', 'grid_width', 'length', 'span', 'modulus_low', 'modulus_high', 'negative_inverse', 'normalization_low', 'normalization_high', 'montgomery'],
		'ntt_pointwise': ['left_index', 'right_index', 'output_index', 'count', 'grid_width', 'modulus_low', 'modulus_high', 'negative_inverse', 'r2_low', 'r2_high', 'montgomery'],
	}
	require(len(schema['kernels']) == 18 and len(schema['contracts']) == 6, 'NTT operation inventory differs')
	seen = set()
	for kernel in schema['kernels']:
		name = kernel['name']; validate_unique(name, seen, 'NTT kernel')
		if name.startswith('ntt_cooperative_') or name.startswith('ntt_stage_fused_'):
			cap = kernel.get('shared_coefficients')
			require(cap in {256, 1024, 4096, 8192} and kernel['source'] == f'src/slang/crypto/ntt/{name}.gen.slang'
				and kernel['workgroup_size'] == [128, 1, 1], 'NTT shared provider identity differs')
			require(kernel['physical_write'] == {'writes': [{'binding': 2, 'domain': 'rows', 'partition': 'exclusive_per_workgroup', 'extent': 'row_width', 'collision': 'exclusive', 'tail': 'bounds_checked'}], 'workspace': 'none'}, 'NTT shared provider ownership differs')
			if name.startswith('ntt_stage_fused_'):
				require(kernel.get('fused_stages') is True and name == f'ntt_stage_fused_{cap}'
					and kernel['kernel_id'] == f'CryptographyNttStageFused{cap}U32' and kernel['semantic_operation'] == 'ntt_stage_fused'
					and kernel['push_fields'] == [[f, 'uint32'] for f in ['input_index', 'twiddles_index', 'output_index', 'count', 'grid_width', 'length', 'modulus_low', 'modulus_high', 'negative_inverse']], 'Fused NTT ABI differs')
			elif kernel.get('scalar_storage') is True:
				require(name == f'ntt_cooperative_scalar_{cap}'
					and kernel['kernel_id'] == f'CryptographyNttCooperativeScalar{cap}U32' and kernel['semantic_operation'] == 'ntt_transform'
					and kernel['push_fields'] == [[f, 'uint32'] for f in ['input_index', 'twiddles_index', 'output_index', 'count', 'grid_width', 'length', 'log_length', 'modulus', 'negative_inverse', 'r2', 'normalization', 'montgomery']], 'Scalar cooperative NTT ABI differs')
			else:
				require(name == f'ntt_cooperative_{cap}'
					and kernel['kernel_id'] == f'CryptographyNttCooperative{cap}U32' and kernel['semantic_operation'] == 'ntt_transform'
					and kernel['push_fields'] == [[f, 'uint32'] for f in ['input_index', 'twiddles_index', 'output_index', 'count', 'grid_width', 'length', 'log_length', 'modulus_low', 'modulus_high', 'negative_inverse', 'r2_low', 'r2_high', 'normalization_low', 'normalization_high', 'montgomery']], 'Cooperative NTT ABI differs')
			continue
		if kernel.get('scalar_storage') is True:
			scalar_fields = {
				'ntt_twiddles_scalar': (['forward_index', 'inverse_index', 'count', 'grid_width', 'modulus', 'negative_inverse', 'root', 'inverse_root', 'one'], 'ntt_twiddles', [0, 1]),
				'ntt_pointwise_scalar': (['left_index', 'right_index', 'output_index', 'count', 'grid_width', 'modulus', 'negative_inverse', 'r2', 'montgomery'], 'ntt_pointwise', [2]),
			}
			require(name in scalar_fields, 'Unknown scalar NTT kernel')
			fields, semantic, bindings = scalar_fields[name]
			require(kernel['push_fields'] == [[f, 'uint32'] for f in fields] and kernel['semantic_operation'] == semantic
				and kernel['workgroup_size'] == [128, 1, 1] and kernel['source'] == f'src/slang/crypto/ntt/{name}.gen.slang'
				and kernel['kernel_id'] == 'Cryptography' + ''.join(w.title() for w in name.split('_')) + 'U32', 'Scalar NTT ABI differs')
			writes = [{'binding': b, 'domain': 'rows', 'partition': 'exclusive_per_invocation', 'extent': 'row_width', 'collision': 'exclusive', 'tail': 'bounds_checked'} for b in bindings]
			require(kernel['physical_write'] == {'writes': writes, 'workspace': 'none'}, 'Scalar NTT ownership differs')
			continue
		require(name in expected_fields, 'Unknown NTT kernel')
		require(kernel['push_fields'] == [[n, 'uint32'] for n in expected_fields[name]] and kernel['semantic_operation'] == name
			and kernel['workgroup_size'] == [128, 1, 1] and kernel['source'] == f'src/slang/crypto/ntt/{name}.gen.slang'
			and kernel['kernel_id'] == 'Cryptography' + ''.join(w.title() for w in name.split('_')) + 'U32', 'NTT ABI differs')
		contract = next((c for c in schema['contracts'] if c['name'] == name), None)
		inputs = 0 if name == 'ntt_twiddles' else 1 if name == 'ntt_reorder' else 2
		outputs = 2 if name == 'ntt_twiddles' else 1
		attrs = ['modulus', 'length'] + (['span', 'inverse'] if name == 'ntt_stage' else []) + ([] if name == 'ntt_twiddles' else ['montgomery'])
		require(contract == {'name': name, 'input_kinds': ['matrix'] * inputs, 'output_kinds': ['matrix'] * outputs, 'shape_rule': 'explicit', 'dtype_rule': 'u32', 'attributes': [[n, 'unsigned_integer'] for n in attrs]}, 'NTT semantic contract differs')
		validate_physical_write(kernel['physical_write'], 'NTT', required=True)
		writes = [{'binding': inputs + i, 'domain': 'output_elements' if name == 'ntt_stage' else 'rows', 'partition': 'exclusive_per_invocation', 'extent': 'up_to_two_elements' if name == 'ntt_stage' else 'row_width', 'collision': 'exclusive', 'tail': 'bounds_checked'} for i in range(outputs)]
		require(kernel['physical_write'] == {'writes': writes, 'workspace': 'none'}, 'NTT ownership differs')


	transform = next((c for c in schema['contracts'] if c['name'] == 'ntt_transform'), None)
	require(transform == {'name': 'ntt_transform', 'input_kinds': ['matrix', 'matrix'], 'output_kinds': ['matrix'], 'shape_rule': 'explicit', 'dtype_rule': 'u32', 'attributes': [[n, 'unsigned_integer'] for n in ['modulus', 'length', 'inverse', 'montgomery']]}, 'Cooperative NTT semantic contract differs')
	fused = next((c for c in schema['contracts'] if c['name'] == 'ntt_stage_fused'), None)
	require(fused == {'name': 'ntt_stage_fused', 'input_kinds': ['matrix', 'matrix'], 'output_kinds': ['matrix'], 'shape_rule': 'explicit', 'dtype_rule': 'u32', 'attributes': [[n, 'unsigned_integer'] for n in ['modulus', 'length', 'inverse']]}, 'Fused NTT semantic contract differs')
	require(len({k['stable_id'] for k in schema['kernels']}) == 18 and all(type(k['stable_id']) is int and 0 < k['stable_id'] <= 65535 for k in schema['kernels']), 'NTT stable IDs must be unique positive u16 values')


def ntt_outputs(root: Path, schema: dict[str, Any], schema_hash: str) -> dict[Path, str]:
	mark = banner(schema_hash, '//', 'crypto_ntt.json')
	directory = FN_ROOT / 'template/crypto/ntt'
	direction = (directory / 'direction.rs.in').read_text()
	transforms = []
	for name in ('forward', 'inverse'):
		scalar_normalization = '((u128::from(power_mod(config.length as u64, modulus - 2, modulus)) * (1_u128 << 32)) % u128::from(modulus)) as u32' if name == 'inverse' else '((1_u128 << 32) % u128::from(modulus)) as u32'
		values = dict(name=name, normalization='parameters.inverse_length' if name == 'inverse' else '((1_u128 << 64) % u128::from(modulus)) as u64', scalar_normalization=scalar_normalization, inverse_flag='1' if name == 'inverse' else '0')
		require(set(re.findall(r'\{\{([^{}]+)\}\}', direction)) == set(values), 'NTT direction template differs')
		transforms.append(re.sub(r'\{\{([^{}]+)\}\}', lambda m: values[m[1]], direction))
	body = (directory / 'transform.rs.in').read_text().replace('{{transforms}}', '\n'.join(transforms))
	outputs = {root / 'src/rs/crypto/ntt/transform.gen.rs': format_rust(mark + body, root)}
	for kernel in schema['kernels']:
		if kernel.get('fused_stages') is True:
			outputs[root / kernel['source']] = generate_ntt_stage_fused(kernel, mark)
			continue
		if 'shared_coefficients' in kernel:
			outputs[root / kernel['source']] = generate_ntt_cooperative_scalar(kernel, mark) if kernel.get('scalar_storage') is True else generate_ntt_cooperative(kernel, mark)
			continue
		if kernel.get('scalar_storage') is True:
			outputs[root / kernel['source']] = generate_ntt_scalar(kernel, mark)
			continue
		fields = '\n'.join(f'\tuint {n};' for n, _ in kernel['push_fields'])
		body = mark + f'''// Public generic cyclic NTT; radix-two identities, portable paired words.
// No OA C++ generic NTT donor exists; fixed PQC transforms are distinct.
import attributes;
import prime62;
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
	if (index >= push.count || (push.modulus_low & 1u) == 0u || push.modulus_high >= 0x40000000u) return;
	uint2 modulus = uint2(push.modulus_low, push.modulus_high);
{NTT_SHADER_BODIES[kernel['name']]}
}}\n'''
		outputs[root / kernel['source']] = body
	python = (directory / 'binding.rs.in').read_text()
	operations, registration = [], []
	stub = banner(schema_hash, '#', 'crypto_ntt.json') + 'from .._native import Engine, Matrix\n\nclass NttConfig:\n    def __init__(self, modulus: int, root: int, length: int, representation: str = "standard") -> None: ...\n    @property\n    def modulus(self) -> int: ...\n    @property\n    def root(self) -> int: ...\n    @property\n    def length(self) -> int: ...\n\nclass NttParameters:\n    @property\n    def config(self) -> NttConfig: ...\n\ndef crypto_ntt_prepare(engine: Engine, config: NttConfig) -> NttParameters: ...\n'
	facade = banner(schema_hash, '#', 'crypto_ntt.json') + 'from .._native import NttConfig, NttParameters, crypto_ntt_prepare as ntt_prepare\n'
	for name in ('forward', 'inverse', 'pointwise_multiply'):
		arguments = ['left', 'right'] if name == 'pointwise_multiply' else ['input']
		parameters = ', '.join(a + ': &PythonMatrix' for a in arguments) + ', parameters: &PythonNttParameters'
		call = ', '.join('&' + a + '.inner' for a in arguments) + ', &parameters.inner'
		operations.append(f'#[pyfunction]\nfn crypto_ntt_{name}({parameters}) -> PyResult<PythonMatrix> {{\n\toa::crypto::ntt::{name}({call}).map(PythonMatrix::wrap).map_err(python_error)\n}}\n')
		registration.append(f'\tmodule.add_function(wrap_pyfunction!(crypto_ntt_{name}, module)?)?;')
		stub += f'def crypto_ntt_{name}(' + ', '.join(a + ': Matrix' for a in arguments) + ', parameters: NttParameters) -> Matrix: ...\n'
		facade += f'from .._native import crypto_ntt_{name} as ntt_{name}\n'
	outputs[root / 'src/py/pyo3/crypto/ntt.gen.rs'] = format_rust(mark + python.replace('{{operations}}', '\n'.join(operations)).replace('{{registration}}', '\n'.join(registration)), root)
	outputs[root / 'src/py/oa/_gen/crypto_ntt_native.pyi'] = stub
	outputs[root / 'src/py/oa/_gen/crypto_ntt.py'] = facade
	return outputs
