// Q/K/V adjoints from saved probabilities and precomputed score gradients.

import storage;
import attributes;

struct PushConstants {
	uint query_index;
	uint key_index;
	uint probability_index;
	uint score_gradient_index;
	uint output_gradient_index;
	uint query_gradient_index;
	uint key_gradient_index;
	uint value_gradient_index;
	uint batch;
	uint sequence_length;
	uint model_width;
	uint num_heads;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("scaled_dot_product_attention_causal_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	uint element_count = push.batch * push.sequence_length * push.model_width;
	if (index >= element_count) {
		return;
	}

	uint feature = index % push.model_width;
	uint row = index / push.model_width;
	uint position = row % push.sequence_length;
	uint batch_index = row / push.sequence_length;
	uint head_width = push.model_width / push.num_heads;
	uint head = feature / head_width;
	uint head_feature = feature % head_width;

	float query_gradient = 0.0f;
	uint score_gradient_base = ((batch_index * push.sequence_length + position)
		* push.num_heads + head) * push.sequence_length;
	for (uint key_position = 0; key_position <= position; ++key_position) {
		uint key_element = (batch_index * push.sequence_length + key_position)
			* push.model_width + head * head_width + head_feature;
		query_gradient += load_f32(
			storage_buffers[push.score_gradient_index],
			score_gradient_base + key_position
		) * load_f32(storage_buffers[push.key_index], key_element);
	}

	float key_gradient = 0.0f;
	float value_gradient = 0.0f;
	for (uint query_position = position; query_position < push.sequence_length; ++query_position) {
		uint query_element = (batch_index * push.sequence_length + query_position)
			* push.model_width + head * head_width + head_feature;
		uint probability_element = ((batch_index * push.sequence_length + query_position)
			* push.num_heads + head) * push.sequence_length + position;
		key_gradient += load_f32(
			storage_buffers[push.score_gradient_index],
			probability_element
		) * load_f32(storage_buffers[push.query_index], query_element);
		value_gradient += load_f32(
			storage_buffers[push.probability_index],
			probability_element
		) * load_f32(storage_buffers[push.output_gradient_index], query_element);
	}

	store_f32(storage_buffers[push.query_gradient_index], index, query_gradient);
	store_f32(storage_buffers[push.key_gradient_index], index, key_gradient);
	store_f32(storage_buffers[push.value_gradient_index], index, value_gradient);
}
