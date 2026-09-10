// Whole-sequence GRU recurrence. One workgroup owns one batch sequence.

import activations;
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
groupshared float hidden_gates[3072];

[kernel_name("gru_scan")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint batch_index = group_id.x;
	uint lane = group_thread_id.x;
	uint gate_count = 3u * push.hidden_size;
	if (batch_index >= push.batch) {
		return;
	}

	for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
		hidden[feature] = 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();

	for (uint time = 0; time < push.sequence_length; ++time) {
		uint hidden_base = (batch_index * push.sequence_length + time) * push.hidden_size;
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			store_f32(
				storage_buffers[push.hidden_previous_index],
				hidden_base + feature,
				hidden[feature]
			);
		}

		for (uint gate = lane; gate < gate_count; gate += 256u) {
			float value = push.has_bias != 0u
				? load_f32(storage_buffers[push.bias_hh_index], gate)
				: 0.0f;
			uint weight_base = gate * push.hidden_size;
			for (uint feature = 0; feature < push.hidden_size; ++feature) {
				value += hidden[feature]
					* load_f32(storage_buffers[push.weight_hh_index], weight_base + feature);
			}
			hidden_gates[gate] = value;
		}
		GroupMemoryBarrierWithGroupSync();

		uint gate_base = (batch_index * push.sequence_length + time) * gate_count;
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			float input_reset = load_f32(storage_buffers[push.gates_i_index], gate_base + feature);
			float input_update = load_f32(
				storage_buffers[push.gates_i_index], gate_base + push.hidden_size + feature
			);
			float input_candidate = load_f32(
				storage_buffers[push.gates_i_index], gate_base + 2u * push.hidden_size + feature
			);
			float reset = sigmoid(input_reset + hidden_gates[feature]);
			float update = sigmoid(input_update + hidden_gates[push.hidden_size + feature]);
			float candidate = tanh(
				input_candidate + reset * hidden_gates[2u * push.hidden_size + feature]
			);
			float value = (1.0f - update) * candidate + update * hidden[feature];
			hidden[feature] = value;
			store_f32(storage_buffers[push.output_index], hidden_base + feature, value);
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
