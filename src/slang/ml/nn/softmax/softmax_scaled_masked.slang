// Mechanically adapted from OA C++:
// source/cpp/lib/oa/ml/shader/compute/nn/softmax/softmaxScaledMasked.slang

import storage;
import attributes;

struct PushConstants {
	uint scores_index;
	uint mask_index;
	uint output_index;
	uint rows;
	uint columns;
	float scale;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_values[256];

[kernel_name("softmax_scaled_masked")]
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
	float local_maximum = -3.402823466e+38f;
	for (uint column = lane; column < push.columns; column += 256) {
		float value = load_f32(storage_buffers[push.scores_index], base + column) * push.scale;
		value += load_f32(storage_buffers[push.mask_index], base + column);
		local_maximum = max(local_maximum, value);
	}
	shared_values[lane] = local_maximum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[lane] = max(shared_values[lane], shared_values[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float row_maximum = shared_values[0];
	float local_sum = 0.0;
	for (uint column = lane; column < push.columns; column += 256) {
		float value = load_f32(storage_buffers[push.scores_index], base + column) * push.scale;
		value += load_f32(storage_buffers[push.mask_index], base + column);
		local_sum += exp(value - row_maximum);
	}
	shared_values[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[lane] += shared_values[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float inverse_sum = 1.0 / shared_values[0];
	for (uint column = lane; column < push.columns; column += 256) {
		float value = load_f32(storage_buffers[push.scores_index], base + column) * push.scale;
		value += load_f32(storage_buffers[push.mask_index], base + column);
		store_f32(
			storage_buffers[push.output_index],
			base + column,
			exp(value - row_maximum) * inverse_sum);
	}
}
