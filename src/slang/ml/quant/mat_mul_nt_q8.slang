// Portable fused Q8 MatMulNt for OA's explicit two-plane weight format.
// One 32-thread workgroup owns one output element and reduces over K without
// a subgroup-width assumption or an intermediate Float32 weight matrix.
//
// Donor: oacpp/source/cpp/lib/oa/ml/shader/compute/quantize/matMulNtQ8.slang

import attributes;
import storage;

struct PushConstants {
	uint input_index;
	uint payload_index;
	uint scale_index;
	uint output_index;
	uint m;
	uint n;
	uint k;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BLOCK_SIZE = 32;
groupshared float partial[32];

[kernel_name("mat_mul_nt_q8")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(32, 1, 1)]
void main(
	uint3 group_id : SV_GroupID,
	uint3 group_thread_id : SV_GroupThreadID
) {
	uint output_column = group_id.x;
	uint output_row = group_id.y;
	if (output_column >= push.n || output_row >= push.m) {
		return;
	}

	uint lane = group_thread_id.x;
	float accumulator = 0.0;
	for (uint input_column = lane; input_column < push.k; input_column += 32) {
		uint weight_index = output_column * push.k + input_column;
		uint block_index = weight_index / BLOCK_SIZE;
		uint block_lane = weight_index % BLOCK_SIZE;
		uint packed = load_u32(
			storage_buffers[push.payload_index],
			block_index * 8 + block_lane / 4);
		uint raw = (packed >> ((block_lane % 4) * 8)) & 0xFF;
		int quantized = raw >= 128 ? int(raw) - 256 : int(raw);
		float scale = load_f32(storage_buffers[push.scale_index], block_index);
		float weight = isfinite(scale) && scale > 0.0
			? float(quantized) * scale
			: 0.0;
		float input_value = load_f32(
			storage_buffers[push.input_index], output_row * push.k + input_column);
		accumulator += input_value * weight;
	}

	partial[lane] = accumulator;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 16; stride > 0; stride >>= 1) {
		if (lane < stride) {
			partial[lane] += partial[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (lane == 0) {
		store_f32(
			storage_buffers[push.output_index],
			output_row * push.n + output_column,
			partial[0]);
	}
}
