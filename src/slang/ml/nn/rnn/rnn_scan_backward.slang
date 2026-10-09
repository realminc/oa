// Whole-sequence Elman BPTT. One workgroup owns one batch sequence.

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
groupshared float output[1024];
groupshared float hidden_previous[1024];

[kernel_name("rnn_scan_backward")]
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
		hidden_gradient[feature] = 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();

	for (int signed_time = int(push.sequence_length) - 1; signed_time >= 0; --signed_time) {
		uint time = uint(signed_time);
		uint base = (batch_index * push.sequence_length + time) * push.hidden_size;
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			hidden_previous[feature] = load_f32(
				storage_buffers[push.hidden_previous_index], base + feature
			);
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			float recurrent = push.has_bias != 0u
				? load_f32(storage_buffers[push.bias_hh_index], feature)
				: 0.0f;
			for (uint input_feature = 0; input_feature < push.hidden_size; ++input_feature) {
				recurrent += hidden_previous[input_feature] * load_f32(
					storage_buffers[push.weight_hh_index],
					feature * push.hidden_size + input_feature
				);
			}
			output[feature] = tanh(
				load_f32(storage_buffers[push.gates_i_index], base + feature) + recurrent
			);
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			float upstream = load_f32(
				storage_buffers[push.output_gradient_index], base + feature
			) + hidden_gradient[feature];
			float delta = upstream * (1.0f - output[feature] * output[feature]);
			store_f32(storage_buffers[push.gates_i_gradient_index], base + feature, delta);
			store_f32(storage_buffers[push.gates_h_gradient_index], base + feature, delta);
			output[feature] = delta;
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			float value = 0.0f;
			for (uint output_feature = 0; output_feature < push.hidden_size; ++output_feature) {
				value += output[output_feature] * load_f32(
					storage_buffers[push.weight_hh_index],
					output_feature * push.hidden_size + feature
				);
			}
			hidden_gradient[feature] = value;
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
