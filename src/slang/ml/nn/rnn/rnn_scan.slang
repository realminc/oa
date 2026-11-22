// Whole-sequence Elman recurrence. One workgroup owns one batch sequence.

import storage;
import attributes;

struct PushConstants {
	uint gates_i_index;
	uint weight_hh_index;
	uint bias_hh_index;
	uint output_index;
	uint hidden_previous_index;
	uint hidden_size;
	uint sequence_length;
	uint batch;
	uint has_bias;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float hidden[1024];
groupshared float activation[1024];

[kernel_name("rnn_scan")]
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
		uint base = (batch_index * push.sequence_length + time) * push.hidden_size;
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			store_f32(storage_buffers[push.hidden_previous_index], base + feature, hidden[feature]);
			float value = push.has_bias != 0u
				? load_f32(storage_buffers[push.bias_hh_index], feature)
				: 0.0f;
			for (uint input_feature = 0; input_feature < push.hidden_size; ++input_feature) {
				value += hidden[input_feature] * load_f32(
					storage_buffers[push.weight_hh_index],
					feature * push.hidden_size + input_feature
				);
			}
			activation[feature] = value;
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			float value = tanh(
				load_f32(storage_buffers[push.gates_i_index], base + feature)
					+ activation[feature]
			);
			hidden[feature] = value;
			store_f32(storage_buffers[push.output_index], base + feature, value);
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
