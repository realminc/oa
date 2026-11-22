// Whole-sequence GRU BPTT. One workgroup owns one batch sequence.

import activations;
import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint gates_i_index;
	uint hidden_previous_index;
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
groupshared float hidden_gates[3072];
groupshared float hidden_previous[1024];
groupshared float hidden_gate_gradient[3072];

[kernel_name("gru_scan_backward")]
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
		hidden_gradient[feature] = 0.0f;
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
			float upstream = load_f32(
				storage_buffers[push.output_gradient_index], hidden_base + feature
			) + hidden_gradient[feature];
			float candidate_gradient = upstream * (1.0f - update);
			float update_gradient = upstream * (hidden_previous[feature] - candidate);
			float direct_hidden_gradient = upstream * update;
			float candidate_logit_gradient = candidate_gradient * (1.0f - candidate * candidate);
			float reset_gradient = candidate_logit_gradient
				* hidden_gates[2u * push.hidden_size + feature];
			float hidden_candidate_gradient = candidate_logit_gradient * reset;
			float reset_logit_gradient = reset_gradient * reset * (1.0f - reset);
			float update_logit_gradient = update_gradient * update * (1.0f - update);

			store_f32(
				storage_buffers[push.gates_i_gradient_index], gate_base + feature,
				reset_logit_gradient
			);
			store_f32(
				storage_buffers[push.gates_i_gradient_index],
				gate_base + push.hidden_size + feature,
				update_logit_gradient
			);
			store_f32(
				storage_buffers[push.gates_i_gradient_index],
				gate_base + 2u * push.hidden_size + feature,
				candidate_logit_gradient
			);
			hidden_gate_gradient[feature] = reset_logit_gradient;
			hidden_gate_gradient[push.hidden_size + feature] = update_logit_gradient;
			hidden_gate_gradient[2u * push.hidden_size + feature] = hidden_candidate_gradient;
			hidden_previous[feature] = direct_hidden_gradient;
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
