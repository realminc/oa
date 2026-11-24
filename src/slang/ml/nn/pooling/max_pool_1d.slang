// Mechanical adaptation of OA's NCHW MaxPool2d forward kernel to NCL MaxPool1d.

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
	uint kernel_size;
	uint stride;
	uint padding;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("max_pool_1d")]
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

	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	int input_x_start = int(output_x * push.stride) - int(push.padding);
	float maximum = -3.402823466e+38f;
	uint maximum_index = 0;
	for (uint kernel_x = 0; kernel_x < push.kernel_size; ++kernel_x) {
		int input_x = input_x_start + int(kernel_x);
		if (input_x >= 0 && input_x < int(push.input_length)) {
			uint input_offset =
				batch * push.channels * push.input_length
				+ channel * push.input_length
				+ uint(input_x);
			float value = load_f32(storage_buffers[push.input_index], input_offset);
			if (value > maximum) {
				maximum = value;
				maximum_index = input_offset;
			}
		}
	}

	uint output_offset =
		batch * push.channels * push.output_length
		+ channel * push.output_length
		+ output_x;
	store_f32(storage_buffers[push.output_index], output_offset, maximum);
	store_u32(storage_buffers[push.indices_index], output_offset, maximum_index);
}
