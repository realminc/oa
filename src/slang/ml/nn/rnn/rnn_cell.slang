// Fused recurrent projection and one Elman RNN cell update.

import storage;
import attributes;

struct PushConstants {
	uint hidden_index;
	uint weight_hh_index;
	uint bias_hh_index;
	uint gates_i_index;
	uint output_index;
	uint gates_h_index;
	uint hidden_size;
	uint batch;
	uint has_bias;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float hidden_projection[1024];

[kernel_name("rnn_cell")]
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
	uint hidden_base = batch_index * push.hidden_size;

	for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
		float value = push.has_bias != 0u
			? load_f32(storage_buffers[push.bias_hh_index], feature)
			: 0.0f;
		uint weight_base = feature * push.hidden_size;
		for (uint input_feature = 0; input_feature < push.hidden_size; ++input_feature) {
			value += load_f32(
				storage_buffers[push.hidden_index], hidden_base + input_feature
			) * load_f32(
				storage_buffers[push.weight_hh_index], weight_base + input_feature
			);
		}
		hidden_projection[feature] = value;
	}
	GroupMemoryBarrierWithGroupSync();

	for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
		float gates_h = hidden_projection[feature];
		float value = tanh(
			load_f32(storage_buffers[push.gates_i_index], hidden_base + feature) + gates_h
		);
		store_f32(storage_buffers[push.output_index], hidden_base + feature, value);
		store_f32(storage_buffers[push.gates_h_index], hidden_base + feature, gates_h);
	}
}
