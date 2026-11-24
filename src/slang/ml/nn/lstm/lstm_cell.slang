// Fused recurrent projection and one LSTM cell update.

import activations;
import storage;
import attributes;

struct PushConstants {
	uint hidden_index;
	uint cell_index;
	uint weight_hh_index;
	uint bias_hh_index;
	uint gates_i_index;
	uint hidden_output_index;
	uint cell_output_index;
	uint gates_h_index;
	uint hidden_size;
	uint batch;
	uint has_bias;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float hidden_gates[4096];

[kernel_name("lstm_cell")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint batch_index = group_id.x;
	uint lane = group_thread_id.x;
	uint gate_count = 4u * push.hidden_size;
	if (batch_index >= push.batch) {
		return;
	}
	uint hidden_base = batch_index * push.hidden_size;

	for (uint gate = lane; gate < gate_count; gate += 256u) {
		float value = push.has_bias != 0u
			? load_f32(storage_buffers[push.bias_hh_index], gate)
			: 0.0f;
		uint weight_base = gate * push.hidden_size;
		for (uint feature = 0; feature < push.hidden_size; ++feature) {
			value += load_f32(storage_buffers[push.hidden_index], hidden_base + feature)
				* load_f32(storage_buffers[push.weight_hh_index], weight_base + feature);
		}
		hidden_gates[gate] = value;
	}
	GroupMemoryBarrierWithGroupSync();

	uint gate_base = batch_index * gate_count;
	for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
		float input_i = load_f32(storage_buffers[push.gates_i_index], gate_base + feature);
		float input_f = load_f32(
			storage_buffers[push.gates_i_index], gate_base + push.hidden_size + feature
		);
		float input_g = load_f32(
			storage_buffers[push.gates_i_index], gate_base + 2u * push.hidden_size + feature
		);
		float input_o = load_f32(
			storage_buffers[push.gates_i_index], gate_base + 3u * push.hidden_size + feature
		);
		float input_gate = sigmoid(input_i + hidden_gates[feature]);
		float forget_gate = sigmoid(input_f + hidden_gates[push.hidden_size + feature]);
		float candidate = tanh(input_g + hidden_gates[2u * push.hidden_size + feature]);
		float output_gate = sigmoid(input_o + hidden_gates[3u * push.hidden_size + feature]);
		float previous = load_f32(storage_buffers[push.cell_index], hidden_base + feature);
		float cell_next = forget_gate * previous + input_gate * candidate;
		store_f32(storage_buffers[push.cell_output_index], hidden_base + feature, cell_next);
		store_f32(
			storage_buffers[push.hidden_output_index], hidden_base + feature,
			output_gate * tanh(cell_next)
		);
		for (uint gate_slot = 0u; gate_slot < 4u; ++gate_slot) {
			uint gate = gate_slot * push.hidden_size + feature;
			store_f32(
				storage_buffers[push.gates_h_index], gate_base + gate,
				hidden_gates[gate]
			);
		}
	}
}
