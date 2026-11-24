// Whole-sequence LSTM BPTT. One workgroup owns one batch sequence.

import activations;
import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint gates_i_index;
	uint hidden_previous_index;
	uint cell_previous_index;
	uint cell_state_index;
	uint weight_hh_index;
	uint bias_hh_index;
	uint gates_i_gradient_index;
	uint gates_h_gradient_index;
	uint hidden_size;
	uint sequence_length;
	uint batch;
	uint has_bias;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float hidden_gradient[1024];
groupshared float cell_gradient[1024];
groupshared float hidden_gates[4096];
groupshared float hidden_previous[1024];
groupshared float hidden_gate_gradient[4096];

[kernel_name("lstm_scan_backward")]
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

	for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
		hidden_gradient[feature] = 0.0f;
		cell_gradient[feature] = 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();

	for (int signed_time = int(push.sequence_length) - 1; signed_time >= 0; --signed_time) {
		uint time = uint(signed_time);
		uint hidden_base = (batch_index * push.sequence_length + time) * push.hidden_size;
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			hidden_previous[feature] = load_f32(
				storage_buffers[push.hidden_previous_index], hidden_base + feature
			);
		}
		GroupMemoryBarrierWithGroupSync();

		for (uint gate = lane; gate < gate_count; gate += 256u) {
			float value = push.has_bias != 0u
				? load_f32(storage_buffers[push.bias_hh_index], gate)
				: 0.0f;
			uint weight_base = gate * push.hidden_size;
			for (uint feature = 0; feature < push.hidden_size; ++feature) {
				value += hidden_previous[feature]
					* load_f32(storage_buffers[push.weight_hh_index], weight_base + feature);
			}
			hidden_gates[gate] = value;
		}
		GroupMemoryBarrierWithGroupSync();

		uint gate_base = (batch_index * push.sequence_length + time) * gate_count;
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
			float cell_previous = load_f32(
				storage_buffers[push.cell_previous_index], hidden_base + feature
			);
			float cell_state = load_f32(
				storage_buffers[push.cell_state_index], hidden_base + feature
			);
			float tanh_cell = tanh(cell_state);
			float upstream = load_f32(
				storage_buffers[push.output_gradient_index], hidden_base + feature
			) + hidden_gradient[feature];
			float cell_upstream = cell_gradient[feature]
				+ upstream * output_gate * (1.0f - tanh_cell * tanh_cell);
			float output_pre = upstream * tanh_cell * output_gate * (1.0f - output_gate);
			float forget_pre = cell_upstream * cell_previous * forget_gate * (1.0f - forget_gate);
			float input_pre = cell_upstream * candidate * input_gate * (1.0f - input_gate);
			float candidate_pre = cell_upstream * input_gate * (1.0f - candidate * candidate);
			float cell_previous_gradient = cell_upstream * forget_gate;

			store_f32(
				storage_buffers[push.gates_i_gradient_index], gate_base + feature,
				input_pre
			);
			store_f32(
				storage_buffers[push.gates_i_gradient_index],
				gate_base + push.hidden_size + feature,
				forget_pre
			);
			store_f32(
				storage_buffers[push.gates_i_gradient_index],
				gate_base + 2u * push.hidden_size + feature,
				candidate_pre
			);
			store_f32(
				storage_buffers[push.gates_i_gradient_index],
				gate_base + 3u * push.hidden_size + feature,
				output_pre
			);
			hidden_gate_gradient[feature] = input_pre;
			hidden_gate_gradient[push.hidden_size + feature] = forget_pre;
			hidden_gate_gradient[2u * push.hidden_size + feature] = candidate_pre;
			hidden_gate_gradient[3u * push.hidden_size + feature] = output_pre;
			cell_gradient[feature] = cell_previous_gradient;
			hidden_previous[feature] = 0.0f;
		}
		GroupMemoryBarrierWithGroupSync();

		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			float value = hidden_previous[feature];
			for (uint gate = 0; gate < gate_count; ++gate) {
				value += hidden_gate_gradient[gate] * load_f32(
					storage_buffers[push.weight_hh_index], gate * push.hidden_size + feature
				);
			}
			hidden_gradient[feature] = value;
		}
		for (uint gate = lane; gate < gate_count; gate += 256u) {
			store_f32(
				storage_buffers[push.gates_h_gradient_index], gate_base + gate,
				hidden_gate_gradient[gate]
			);
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
