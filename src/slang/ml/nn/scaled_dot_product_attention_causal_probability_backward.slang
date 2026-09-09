// Compute the softmax-score adjoint from saved causal attention probabilities.

import storage;
import attributes;

static const uint MAX_SEQUENCE_LENGTH = 1024u;

struct PushConstants {
	uint probability_index;
	uint value_index;
	uint output_gradient_index;
	uint score_gradient_index;
	uint batch;
	uint sequence_length;
	uint model_width;
	uint num_heads;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float probability_gradient[MAX_SEQUENCE_LENGTH];
groupshared float partial[256];

[kernel_name("scaled_dot_product_attention_causal_probability_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row_head = group_id.x;
	uint lane = group_thread_id.x;
	uint attention_row_count = push.batch * push.sequence_length * push.num_heads;
	if (row_head >= attention_row_count) {
		return;
	}

	uint head = row_head % push.num_heads;
	uint row = row_head / push.num_heads;
	uint query_position = row % push.sequence_length;
	uint batch_index = row / push.sequence_length;
	uint head_width = push.model_width / push.num_heads;
	uint output_gradient_base = row * push.model_width + head * head_width;
	uint probability_base = row_head * push.sequence_length;

	float local_expected = 0.0f;
	for (uint key_position = lane; key_position < push.sequence_length; key_position += 256u) {
		float gradient = 0.0f;
		if (key_position <= query_position) {
			uint value_base = (batch_index * push.sequence_length + key_position)
				* push.model_width + head * head_width;
			for (uint feature = 0; feature < head_width; ++feature) {
				gradient += load_f32(
					storage_buffers[push.output_gradient_index],
					output_gradient_base + feature
				) * load_f32(storage_buffers[push.value_index], value_base + feature);
			}
			local_expected += load_f32(
				storage_buffers[push.probability_index],
				probability_base + key_position
			) * gradient;
		}
		probability_gradient[key_position] = gradient;
	}
	partial[lane] = local_expected;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lane] += partial[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	float scale = rsqrt(float(head_width));
	for (uint key_position = lane; key_position < push.sequence_length; key_position += 256u) {
		float probability = load_f32(
			storage_buffers[push.probability_index],
			probability_base + key_position
		);
		float score_gradient = key_position <= query_position
			? probability * (probability_gradient[key_position] - partial[0]) * scale
			: 0.0f;
		store_f32(
			storage_buffers[push.score_gradient_index],
			probability_base + key_position,
			score_gradient
		);
	}
}
