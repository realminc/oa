// Mechanically adapted from OA C++:
// source/cpp/lib/oa/ml/shader/compute/nn/attention/sdpaSoftmax.slang

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

groupshared float shared_reduce[256];

float transformed_score(uint base, uint query, uint column) {
	float value = load_f32(storage_buffers[push.scores_index], base + column) * push.scale;
	if (push.has_mask != 0) {
		value += load_f32(storage_buffers[push.mask_index], base + column);
	}
	if (push.causal != 0 && column > query) {
		value = -3.402823466e+38f;
	}
	return value;
}

[kernel_name("scaled_dot_product_attention")]
[domain("ml")]
[variant("standard")]
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
	uint query = row % push.columns;
	float local_maximum = -3.402823466e+38f;
	for (uint column = lane; column < push.columns; column += 256) {
		local_maximum = max(local_maximum, transformed_score(base, query, column));
	}
	shared_reduce[lane] = local_maximum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_reduce[lane] = max(shared_reduce[lane], shared_reduce[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float row_maximum = shared_reduce[0];
	float local_sum = 0.0;
	for (uint column = lane; column < push.columns; column += 256) {
		local_sum += exp(transformed_score(base, query, column) - row_maximum);
	}
	shared_reduce[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_reduce[lane] += shared_reduce[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float inverse_sum = 1.0 / shared_reduce[0];
	for (uint column = lane; column < push.columns; column += 256) {
		float probability = exp(transformed_score(base, query, column) - row_maximum) * inverse_sum;
		store_f32(storage_buffers[push.probability_index], base + column, probability);
	}
}
