// Mechanical adaptation of OA's adaptive pooling window geometry to NCL
// AdaptiveMaxPool1d: floor/ceil bin boundaries with argmax index output.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint indices_index;
	uint batch_size;
	uint channels;
	uint input_length;
	uint output_length;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("adaptive_max_pool_1d")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_x = thread_id.x;
	uint batch_channel = thread_id.y;
	if (
		output_x >= push.output_length
		|| batch_channel >= push.batch_size * push.channels
	) {
		return;
	}

	uint input_x_start = output_x * push.input_length / push.output_length;
	uint input_x_end = ((output_x + 1) * push.input_length + push.output_length - 1)
		/ push.output_length;
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	uint channel_base =
		batch * push.channels * push.input_length + channel * push.input_length;
	float maximum = -3.402823466e+38f;
	uint maximum_index = 0;
	for (uint input_x = input_x_start; input_x < input_x_end; ++input_x) {
		uint input_offset = channel_base + input_x;
		float value = load_f32(storage_buffers[push.input_index], input_offset);
		if (value > maximum) {
			maximum = value;
			maximum_index = input_offset;
		}
	}
	uint output_offset =
		batch * push.channels * push.output_length
		+ channel * push.output_length
		+ output_x;
	store_f32(storage_buffers[push.output_index], output_offset, maximum);
	store_u32(storage_buffers[push.indices_index], output_offset, maximum_index);
}
