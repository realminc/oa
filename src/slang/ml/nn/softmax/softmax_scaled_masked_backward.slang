// Mechanically adapted from OA C++:
// source/cpp/lib/oa/ml/shader/compute/nn/softmax/softmaxScaledMaskedBwd.slang

import storage;
import attributes;

struct PushConstants {
	uint forward_output_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint rows;
	uint columns;
	float scale;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_values[256];

[kernel_name("softmax_scaled_masked_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	if (row >= push.rows) {
		return;
	}
	uint base = row * push.columns;
	float local_dot = 0.0;
	for (uint column = lane; column < push.columns; column += 256) {
		local_dot +=
			load_f32(storage_buffers[push.output_gradient_index], base + column) *
			load_f32(storage_buffers[push.forward_output_index], base + column);
	}
	shared_values[lane] = local_dot;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[lane] += shared_values[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float dot = shared_values[0];
	for (uint column = lane; column < push.columns; column += 256) {
		float softmax_value = load_f32(storage_buffers[push.forward_output_index], base + column);
		float gradient = load_f32(storage_buffers[push.output_gradient_index], base + column);
		store_f32(
			storage_buffers[push.input_gradient_index],
			base + column,
			softmax_value * (gradient - dot) * push.scale);
	}
}
