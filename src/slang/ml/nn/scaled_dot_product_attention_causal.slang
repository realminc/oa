// Row-cooperative multi-head causal attention over packed [B*S, D] Q/K/V.

import storage;
import attributes;

static const uint MAX_SEQUENCE_LENGTH = 1024u;

struct PushConstants {
	uint query_index;
	uint key_index;
	uint value_index;
	uint output_index;
	uint probability_index;
	uint batch;
	uint sequence_length;
	uint model_width;
	uint num_heads;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float row_values[MAX_SEQUENCE_LENGTH];
groupshared float partial[256];

[kernel_name("scaled_dot_product_attention_causal")]
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
	uint query_base = row * push.model_width + head * head_width;
	uint probability_base = row_head * push.sequence_length;
	float scale = rsqrt(float(head_width));

	float local_maximum = -3.402823466e+38f;
	for (uint key_position = lane; key_position < push.sequence_length; key_position += 256u) {
		float score = -3.402823466e+38f;
		if (key_position <= query_position) {
			uint key_base = (batch_index * push.sequence_length + key_position)
				* push.model_width + head * head_width;
			score = 0.0f;
			for (uint feature = 0; feature < head_width; ++feature) {
				score += load_f32(storage_buffers[push.query_index], query_base + feature)
					* load_f32(storage_buffers[push.key_index], key_base + feature);
			}
			score *= scale;
			local_maximum = max(local_maximum, score);
		}
		row_values[key_position] = score;
	}
	partial[lane] = local_maximum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lane] = max(partial[lane], partial[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}

	float local_sum = 0.0f;
	for (uint key_position = lane; key_position < push.sequence_length; key_position += 256u) {
		float unnormalized = key_position <= query_position
			? exp(row_values[key_position] - partial[0])
			: 0.0f;
		row_values[key_position] = unnormalized;
		local_sum += unnormalized;
	}
	partial[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lane] += partial[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	float inverse_denominator = 1.0f / partial[0];
	for (uint key_position = lane; key_position < push.sequence_length; key_position += 256u) {
		float probability = row_values[key_position] * inverse_denominator;
		row_values[key_position] = probability;
		store_f32(
			storage_buffers[push.probability_index],
			probability_base + key_position,
			probability
		);
	}
	GroupMemoryBarrierWithGroupSync();

	for (uint feature = lane; feature < head_width; feature += 256u) {
		float output = 0.0f;
		for (uint key_position = 0; key_position <= query_position; ++key_position) {
			uint value_element = (batch_index * push.sequence_length + key_position)
				* push.model_width + head * head_width + feature;
			output += row_values[key_position]
				* load_f32(storage_buffers[push.value_index], value_element);
		}
		store_f32(storage_buffers[push.output_index], query_base + feature, output);
	}
}
