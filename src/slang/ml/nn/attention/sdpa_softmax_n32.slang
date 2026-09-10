// Mechanically adapted from OA C++:
// source/cpp/lib/oa/ml/shader/compute/nn/attention/sdpaSoftmaxN32.slang

import storage;
import attributes;

struct PushConstants {
	uint scores_index;
	uint mask_index;
	uint probability_index;
	uint rows;
	uint columns;
	float scale;
	uint has_mask;
	uint causal;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_maximum[32];
groupshared float shared_sum[32];

float reduce_maximum_32(float value, uint lane) {
	shared_maximum[lane] = value;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 16; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_maximum[lane] = max(shared_maximum[lane], shared_maximum[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return shared_maximum[0];
}

float reduce_sum_32(float value, uint lane) {
	shared_sum[lane] = value;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 16; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_sum[lane] += shared_sum[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return shared_sum[0];
}

[kernel_name("sdpa_softmax_n32")]
[domain("ml")]
[variant("standard_n32")]
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
	uint query = row % push.columns;
	bool active = lane < push.columns;
	float value = -3.402823466e+38f;
	if (active) {
		value = load_f32(storage_buffers[push.scores_index], base + lane) * push.scale;
		if (push.has_mask != 0) {
			value += load_f32(storage_buffers[push.mask_index], base + lane);
		}
		if (push.causal != 0 && lane > query) {
			value = -3.402823466e+38f;
		}
	}
	float row_maximum = reduce_maximum_32(value, lane);
	float exponent = active ? exp(value - row_maximum) : 0.0;
	float inverse_sum = 1.0 / reduce_sum_32(exponent, lane);
	if (active) {
		store_f32(
			storage_buffers[push.probability_index],
			base + lane,
			exponent * inverse_sum);
	}
}
