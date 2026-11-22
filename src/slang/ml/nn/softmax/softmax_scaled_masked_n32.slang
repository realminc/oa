// Mechanically adapted from OA C++:
// source/cpp/lib/oa/ml/shader/compute/nn/softmax/softmaxScaledMaskedN32.slang

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

groupshared float shared_maximum[32];
groupshared float shared_sum[32];

float reduce_maximum_32(float value, uint lane) {
	float accumulator = value;
	shared_maximum[lane] = accumulator;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 16; stride > 0; stride >>= 1) {
		if (lane < stride) {
			accumulator = max(accumulator, shared_maximum[lane + stride]);
			shared_maximum[lane] = accumulator;
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return shared_maximum[0];
}

float reduce_sum_32(float value, uint lane) {
	float accumulator = value;
	shared_sum[lane] = accumulator;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 16; stride > 0; stride >>= 1) {
		if (lane < stride) {
			accumulator += shared_sum[lane + stride];
			shared_sum[lane] = accumulator;
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return shared_sum[0];
}

[kernel_name("softmax_scaled_masked_n32")]
[domain("ml")]
[variant("n32")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	if (row >= push.rows) {
		return;
	}
	uint base = row * push.columns;
	bool active = lane < push.columns;
	float value = -3.402823466e+38f;
	if (active) {
		value = load_f32(storage_buffers[push.scores_index], base + lane) * push.scale;
		value += load_f32(storage_buffers[push.mask_index], base + lane);
	}
	float row_maximum = reduce_maximum_32(value, lane);
	float exponent = active ? exp(value - row_maximum) : 0.0;
	float inverse_sum = 1.0 / reduce_sum_32(exponent, lane);
	if (active) {
		store_f32(storage_buffers[push.output_index], base + lane, exponent * inverse_sum);
	}
}
