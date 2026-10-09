// FP32 RMSNorm forward adapted from OA's donor kernel.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/rmsnorm/rmsNorm.slang
// Shared reduction: source/cpp/lib/oa/core/shader/compute/math/reductions.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint output_index;
	uint rows;
	uint columns;
	float epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[256];

[kernel_name("rms_norm")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	if (row >= push.rows) return;
	uint base = row * push.columns;

	float local_square_sum = 0.0f;
	for (uint column = lane; column < push.columns; column += 256u) {
		float value = load_f32(storage_buffers[push.input_index], base + column);
		local_square_sum += value * value;
	}
	partial[lane] = local_square_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) partial[lane] += partial[lane + stride];
		GroupMemoryBarrierWithGroupSync();
	}
	float inverse_rms = 1.0f / sqrt(partial[0] / float(push.columns) + push.epsilon);

	for (uint column = lane; column < push.columns; column += 256u) {
		float value = load_f32(storage_buffers[push.input_index], base + column);
		float weight = load_f32(storage_buffers[push.weight_index], column);
		store_f32(storage_buffers[push.output_index], base + column,
			value * inverse_rms * weight);
	}
}
