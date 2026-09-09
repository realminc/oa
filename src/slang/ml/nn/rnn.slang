// Single-layer Elman RNN scan. One workgroup owns one batch sequence.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_ih_index;
	uint weight_hh_index;
	uint bias_ih_index;
	uint bias_hh_index;
	uint output_index;
	uint hidden_previous_index;
	uint batch;
	uint sequence_length;
	uint input_size;
	uint hidden_size;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float hidden[1024];
groupshared float next_hidden[1024];

[kernel_name("rnn")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint batch_index = group_id.x;
	uint lane = group_thread_id.x;
	if (batch_index >= push.batch) {
		return;
	}

	for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
		hidden[feature] = 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();

	for (uint time = 0; time < push.sequence_length; ++time) {
		uint input_base = (batch_index * push.sequence_length + time) * push.input_size;
		uint hidden_base = (batch_index * push.sequence_length + time) * push.hidden_size;
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			store_f32(storage_buffers[push.hidden_previous_index], hidden_base + feature, hidden[feature]);
			float value = load_f32(storage_buffers[push.bias_ih_index], feature)
				+ load_f32(storage_buffers[push.bias_hh_index], feature);
			for (uint input_feature = 0; input_feature < push.input_size; ++input_feature) {
				value += load_f32(storage_buffers[push.input_index], input_base + input_feature)
					* load_f32(
						storage_buffers[push.weight_ih_index],
						feature * push.input_size + input_feature
					);
			}
			for (uint hidden_feature = 0; hidden_feature < push.hidden_size; ++hidden_feature) {
				value += hidden[hidden_feature] * load_f32(
					storage_buffers[push.weight_hh_index],
					feature * push.hidden_size + hidden_feature
				);
			}
			next_hidden[feature] = tanh(value);
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			hidden[feature] = next_hidden[feature];
			store_f32(storage_buffers[push.output_index], hidden_base + feature, next_hidden[feature]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
