// Mechanical adaptation of OA's direct NCHW grouped Conv2d kernel.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/conv2d.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint bias_index;
	uint output_index;
	uint batch_size;
	uint input_channels;
	uint output_channels;
	uint input_height;
	uint input_width;
	uint kernel_size;
	uint stride;
	uint padding;
	uint output_height;
	uint output_width;
	uint groups;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_2d")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_index = thread_id.x;
	uint output_count =
		push.batch_size * push.output_channels * push.output_height * push.output_width;
	if (output_index >= output_count) {
		return;
	}

	uint output_x = output_index % push.output_width;
	uint quotient = output_index / push.output_width;
	uint output_y = quotient % push.output_height;
	quotient /= push.output_height;
	uint output_channel = quotient % push.output_channels;
	uint batch = quotient / push.output_channels;

	uint input_channels_per_group = push.input_channels / push.groups;
	uint output_channels_per_group = push.output_channels / push.groups;
	uint group = output_channel / output_channels_per_group;
	uint input_channel_base = group * input_channels_per_group;
	float value = load_f32(storage_buffers[push.bias_index], output_channel);

	for (uint local_input_channel = 0; local_input_channel < input_channels_per_group; ++local_input_channel) {
		uint input_channel = input_channel_base + local_input_channel;
		for (uint kernel_y = 0; kernel_y < push.kernel_size; ++kernel_y) {
			for (uint kernel_x = 0; kernel_x < push.kernel_size; ++kernel_x) {
				int input_y = int(output_y * push.stride + kernel_y) - int(push.padding);
				int input_x = int(output_x * push.stride + kernel_x) - int(push.padding);
				if (
					input_y >= 0 && input_y < int(push.input_height)
					&& input_x >= 0 && input_x < int(push.input_width)
				) {
					uint input_offset =
						((batch * push.input_channels + input_channel) * push.input_height + uint(input_y))
						* push.input_width + uint(input_x);
					uint weight_offset =
						((output_channel * input_channels_per_group + local_input_channel)
						* push.kernel_size + kernel_y) * push.kernel_size + kernel_x;
					value += load_f32(storage_buffers[push.input_index], input_offset)
						* load_f32(storage_buffers[push.weight_index], weight_offset);
				}
			}
		}
	}

	store_f32(storage_buffers[push.output_index], output_index, value);
}
