// Mechanical adaptation of OA's grouped Conv2d input adjoint.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/conv2dBwdData.slang

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint weight_index;
	uint input_gradient_index;
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

[kernel_name("conv_2d_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint input_index = thread_id.x;
	uint input_count =
		push.batch_size * push.input_channels * push.input_height * push.input_width;
	if (input_index >= input_count) {
		return;
	}

	uint input_x = input_index % push.input_width;
	uint quotient = input_index / push.input_width;
	uint input_y = quotient % push.input_height;
	quotient /= push.input_height;
	uint input_channel = quotient % push.input_channels;
	uint batch = quotient / push.input_channels;

	uint input_channels_per_group = push.input_channels / push.groups;
	uint output_channels_per_group = push.output_channels / push.groups;
	uint group = input_channel / input_channels_per_group;
	uint local_input_channel = input_channel - group * input_channels_per_group;
	uint output_channel_base = group * output_channels_per_group;
	float gradient = 0.0f;

	for (uint local_output_channel = 0; local_output_channel < output_channels_per_group; ++local_output_channel) {
		uint output_channel = output_channel_base + local_output_channel;
		for (uint kernel_y = 0; kernel_y < push.kernel_size; ++kernel_y) {
			for (uint kernel_x = 0; kernel_x < push.kernel_size; ++kernel_x) {
				int output_y_numerator = int(input_y) - int(kernel_y) + int(push.padding);
				int output_x_numerator = int(input_x) - int(kernel_x) + int(push.padding);
				if (
					output_y_numerator < 0 || output_x_numerator < 0
					|| output_y_numerator % int(push.stride) != 0
					|| output_x_numerator % int(push.stride) != 0
				) {
					continue;
				}
				int output_y = output_y_numerator / int(push.stride);
				int output_x = output_x_numerator / int(push.stride);
				if (
					output_y >= int(push.output_height)
					|| output_x >= int(push.output_width)
				) {
					continue;
				}
				uint output_offset =
					((batch * push.output_channels + output_channel) * push.output_height
					+ uint(output_y)) * push.output_width + uint(output_x);
				uint weight_offset =
					((output_channel * input_channels_per_group + local_input_channel)
					* push.kernel_size + kernel_y) * push.kernel_size + kernel_x;
				gradient += load_f32(storage_buffers[push.output_gradient_index], output_offset)
					* load_f32(storage_buffers[push.weight_index], weight_offset);
			}
		}
	}

	store_f32(storage_buffers[push.input_gradient_index], input_index, gradient);
}
