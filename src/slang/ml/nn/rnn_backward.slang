// Deterministic single-layer Elman BPTT. One workgroup owns the complete batch.

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint input_index;
	uint output_index;
	uint hidden_previous_index;
	uint weight_ih_index;
	uint weight_hh_index;
	uint input_gradient_index;
	uint weight_ih_gradient_index;
	uint weight_hh_gradient_index;
	uint bias_ih_gradient_index;
	uint bias_hh_gradient_index;
	uint batch;
	uint sequence_length;
	uint input_size;
	uint hidden_size;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float hidden_gradient[1024];
groupshared float next_hidden_gradient[1024];
groupshared float delta[1024];

[kernel_name("rnn_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	uint weight_ih_count = push.hidden_size * push.input_size;
	uint weight_hh_count = push.hidden_size * push.hidden_size;
	for (uint index = lane; index < weight_ih_count; index += 256u) {
		store_f32(storage_buffers[push.weight_ih_gradient_index], index, 0.0f);
	}
	for (uint index = lane; index < weight_hh_count; index += 256u) {
		store_f32(storage_buffers[push.weight_hh_gradient_index], index, 0.0f);
	}
	for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
		store_f32(storage_buffers[push.bias_ih_gradient_index], feature, 0.0f);
		store_f32(storage_buffers[push.bias_hh_gradient_index], feature, 0.0f);
	}
	GroupMemoryBarrierWithGroupSync();

	for (uint batch_index = 0; batch_index < push.batch; ++batch_index) {
		for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
			hidden_gradient[feature] = 0.0f;
		}
		GroupMemoryBarrierWithGroupSync();

		for (int signed_time = int(push.sequence_length) - 1; signed_time >= 0; --signed_time) {
			uint time = uint(signed_time);
			uint input_base = (batch_index * push.sequence_length + time) * push.input_size;
			uint hidden_base = (batch_index * push.sequence_length + time) * push.hidden_size;
			for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
				float output_value = load_f32(storage_buffers[push.output_index], hidden_base + feature);
				float upstream = load_f32(
					storage_buffers[push.output_gradient_index], hidden_base + feature
				) + hidden_gradient[feature];
				delta[feature] = upstream * (1.0f - output_value * output_value);
			}
			GroupMemoryBarrierWithGroupSync();

			for (uint input_feature = lane; input_feature < push.input_size; input_feature += 256u) {
				float value = 0.0f;
				for (uint feature = 0; feature < push.hidden_size; ++feature) {
					value += delta[feature] * load_f32(
						storage_buffers[push.weight_ih_index],
						feature * push.input_size + input_feature
					);
				}
				store_f32(storage_buffers[push.input_gradient_index], input_base + input_feature, value);
			}
			for (uint index = lane; index < weight_ih_count; index += 256u) {
				uint feature = index / push.input_size;
				uint input_feature = index % push.input_size;
				float value = load_f32(storage_buffers[push.weight_ih_gradient_index], index);
				value += delta[feature]
					* load_f32(storage_buffers[push.input_index], input_base + input_feature);
				store_f32(storage_buffers[push.weight_ih_gradient_index], index, value);
			}
			for (uint index = lane; index < weight_hh_count; index += 256u) {
				uint feature = index / push.hidden_size;
				uint hidden_feature = index % push.hidden_size;
				float value = load_f32(storage_buffers[push.weight_hh_gradient_index], index);
				value += delta[feature] * load_f32(
					storage_buffers[push.hidden_previous_index],
					hidden_base + hidden_feature
				);
				store_f32(storage_buffers[push.weight_hh_gradient_index], index, value);
			}
			for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
				float bias_value = load_f32(storage_buffers[push.bias_ih_gradient_index], feature)
					+ delta[feature];
				store_f32(storage_buffers[push.bias_ih_gradient_index], feature, bias_value);
				store_f32(storage_buffers[push.bias_hh_gradient_index], feature, bias_value);
			}
			GroupMemoryBarrierWithGroupSync();

			for (uint hidden_feature = lane; hidden_feature < push.hidden_size; hidden_feature += 256u) {
				float value = 0.0f;
				for (uint feature = 0; feature < push.hidden_size; ++feature) {
					value += delta[feature] * load_f32(
						storage_buffers[push.weight_hh_index],
						feature * push.hidden_size + hidden_feature
					);
				}
				next_hidden_gradient[hidden_feature] = value;
			}
			GroupMemoryBarrierWithGroupSync();
			for (uint feature = lane; feature < push.hidden_size; feature += 256u) {
				hidden_gradient[feature] = next_hidden_gradient[feature];
			}
			GroupMemoryBarrierWithGroupSync();
		}
	}
}
