// Mechanically adapted from OA C++:
// source/cpp/lib/oa/ml/shader/compute/nn/attention/flashCausalBwdKV.slang

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
	uint key_gradient_index;
	uint value_gradient_index;
	uint batch_heads;
	uint sequence_length;
	uint head_dim;
	float scale;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_probability[MAX_SEQUENCE_LENGTH];
groupshared float shared_score_gradient[MAX_SEQUENCE_LENGTH];

[kernel_name("flash_attention_causal_backward_kv")]
[domain("ml")]
[variant("flash_causal_kv")]
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
	uint key = row - batch_head * push.sequence_length;
	uint key_value_base = row * push.head_dim;
	uint head_base = batch_head * push.sequence_length * push.head_dim;

	for (uint query = key + lane; query < push.sequence_length; query += THREAD_COUNT) {
		uint query_base = head_base + query * push.head_dim;
		float score = 0.0f;
		float probability_gradient = 0.0f;
		float delta = 0.0f;
		for (uint dimension = 0; dimension < push.head_dim; ++dimension) {
			float output_gradient = load_f32(
				storage_buffers[push.output_gradient_index], query_base + dimension);
			score += load_f32(storage_buffers[push.query_index], query_base + dimension)
				* load_f32(storage_buffers[push.key_index], key_value_base + dimension);
			probability_gradient += output_gradient
				* load_f32(storage_buffers[push.value_index], key_value_base + dimension);
			delta += output_gradient
				* load_f32(storage_buffers[push.output_index], query_base + dimension);
		}
		float probability = exp(
			score * push.scale
				- load_f32(
					storage_buffers[push.log_sum_exp_index],
					batch_head * push.sequence_length + query));
		shared_probability[query] = probability;
		shared_score_gradient[query] = probability * (probability_gradient - delta) * push.scale;
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint dimension = lane; dimension < push.head_dim; dimension += THREAD_COUNT) {
		float key_gradient = 0.0f;
		float value_gradient = 0.0f;
		for (uint query = key; query < push.sequence_length; ++query) {
			uint query_index = head_base + query * push.head_dim + dimension;
			key_gradient += shared_score_gradient[query]
				* load_f32(storage_buffers[push.query_index], query_index);
			value_gradient += shared_probability[query]
				* load_f32(storage_buffers[push.output_gradient_index], query_index);
		}
		store_f32(
			storage_buffers[push.key_gradient_index], key_value_base + dimension, key_gradient);
		store_f32(
			storage_buffers[push.value_gradient_index], key_value_base + dimension, value_gradient);
	}
}
