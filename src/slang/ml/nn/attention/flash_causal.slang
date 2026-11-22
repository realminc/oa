// Mechanically adapted from OA C++:
// source/cpp/lib/oa/ml/shader/compute/nn/attention/flashCausal.slang

import storage;
import attributes;

static const uint THREAD_COUNT = 64;
static const uint MAX_SEQUENCE_LENGTH = 1024;

struct PushConstants {
	uint query_index;
	uint key_index;
	uint value_index;
	uint output_index;
	uint log_sum_exp_index;
	uint batch_heads;
	uint sequence_length;
	uint head_dim;
	float scale;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_weight[MAX_SEQUENCE_LENGTH];
groupshared float shared_reduce[THREAD_COUNT];

[kernel_name("flash_attention_causal")]
[domain("ml")]
[variant("flash_causal")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREAD_COUNT, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	uint row_count = push.batch_heads * push.sequence_length;
	if (row >= row_count) {
		return;
	}
	uint batch_head = row / push.sequence_length;
	uint query = row - batch_head * push.sequence_length;
	uint query_base = row * push.head_dim;
	uint key_value_head_base = batch_head * push.sequence_length * push.head_dim;

	float local_maximum = -3.402823466e+38f;
	for (uint key = lane; key <= query; key += THREAD_COUNT) {
		uint key_base = key_value_head_base + key * push.head_dim;
		float score = 0.0f;
		for (uint dimension = 0; dimension < push.head_dim; ++dimension) {
			score += load_f32(storage_buffers[push.query_index], query_base + dimension)
				* load_f32(storage_buffers[push.key_index], key_base + dimension);
		}
		score *= push.scale;
		shared_weight[key] = score;
		local_maximum = max(local_maximum, score);
	}
	shared_reduce[lane] = local_maximum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = THREAD_COUNT / 2; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_reduce[lane] = max(shared_reduce[lane], shared_reduce[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float row_maximum = shared_reduce[0];
	float local_sum = 0.0f;
	for (uint key = lane; key <= query; key += THREAD_COUNT) {
		float weight = exp(shared_weight[key] - row_maximum);
		shared_weight[key] = weight;
		local_sum += weight;
	}
	GroupMemoryBarrierWithGroupSync();
	shared_reduce[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = THREAD_COUNT / 2; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_reduce[lane] += shared_reduce[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float sum = shared_reduce[0];
	if (lane == 0) {
		store_f32(storage_buffers[push.log_sum_exp_index], row, row_maximum + log(sum));
	}
	GroupMemoryBarrierWithGroupSync();

	for (uint dimension = lane; dimension < push.head_dim; dimension += THREAD_COUNT) {
		float accumulator = 0.0f;
		for (uint key = 0; key <= query; ++key) {
			uint value_index = key_value_head_base + key * push.head_dim + dimension;
			accumulator += (shared_weight[key] / sum)
				* load_f32(storage_buffers[push.value_index], value_index);
		}
		store_f32(storage_buffers[push.output_index], query_base + dimension, accumulator);
	}
}
