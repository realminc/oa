// Mechanically adapted from OA C++:
// source/cpp/lib/oa/ml/shader/compute/nn/attention/flashCausalBwdQ.slang

import storage;
import attributes;

static const uint THREAD_COUNT = 64;
static const uint MAX_SEQUENCE_LENGTH = 1024;

struct PushConstants {
	uint query_index;
	uint key_index;
	uint value_index;
	uint output_index;
	uint output_gradient_index;
	uint log_sum_exp_index;
	uint query_gradient_index;
	uint batch_heads;
	uint sequence_length;
	uint head_dim;
	float scale;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_score_gradient[MAX_SEQUENCE_LENGTH];
groupshared float shared_reduce[THREAD_COUNT];

[kernel_name("scaled_dot_product_attention_backward")]
[domain("ml")]
[variant("flash_causal_q")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREAD_COUNT, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	uint lane = group_thread_id.x;
	if (row >= push.batch_heads * push.sequence_length) {
		return;
	}
	uint batch_head = row / push.sequence_length;
	uint query = row - batch_head * push.sequence_length;
	uint query_base = row * push.head_dim;
	uint key_value_head_base = batch_head * push.sequence_length * push.head_dim;

	float local_delta = 0.0f;
	for (uint dimension = lane; dimension < push.head_dim; dimension += THREAD_COUNT) {
		local_delta += load_f32(storage_buffers[push.output_gradient_index], query_base + dimension)
			* load_f32(storage_buffers[push.output_index], query_base + dimension);
	}
	shared_reduce[lane] = local_delta;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = THREAD_COUNT / 2; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_reduce[lane] += shared_reduce[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float delta = shared_reduce[0];
	float log_sum_exp = load_f32(storage_buffers[push.log_sum_exp_index], row);
	for (uint key = lane; key <= query; key += THREAD_COUNT) {
		uint key_value_base = key_value_head_base + key * push.head_dim;
		float score = 0.0f;
		float probability_gradient = 0.0f;
		for (uint dimension = 0; dimension < push.head_dim; ++dimension) {
			score += load_f32(storage_buffers[push.query_index], query_base + dimension)
				* load_f32(storage_buffers[push.key_index], key_value_base + dimension);
			probability_gradient += load_f32(
				storage_buffers[push.output_gradient_index], query_base + dimension)
				* load_f32(storage_buffers[push.value_index], key_value_base + dimension);
		}
		float probability = exp(score * push.scale - log_sum_exp);
		shared_score_gradient[key] = probability * (probability_gradient - delta) * push.scale;
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint dimension = lane; dimension < push.head_dim; dimension += THREAD_COUNT) {
		float accumulator = 0.0f;
		for (uint key = 0; key <= query; ++key) {
			accumulator += shared_score_gradient[key]
				* load_f32(
					storage_buffers[push.key_index],
					key_value_head_base + key * push.head_dim + dimension);
		}
		store_f32(
			storage_buffers[push.query_gradient_index], query_base + dimension, accumulator);
	}
}
