// Mechanical adaptation of OA's regular Conv2d input adjoint for ConvTranspose2d.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/conv2d.slang

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
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_transpose_2d_backward")]
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
	float gradient = 0.0f;
	for (uint output_channel = 0; output_channel < push.output_channels; ++output_channel) {
		for (uint kernel_y = 0; kernel_y < push.kernel_size; ++kernel_y) {
			int output_y = int(input_y * push.stride + kernel_y) - int(push.padding);
			if (output_y < 0 || output_y >= int(push.output_height)) {
				continue;
			}
			for (uint kernel_x = 0; kernel_x < push.kernel_size; ++kernel_x) {
				int output_x = int(input_x * push.stride + kernel_x) - int(push.padding);
				if (output_x < 0 || output_x >= int(push.output_width)) {
					continue;
				}
				uint output_offset =
					((batch * push.output_channels + output_channel) * push.output_height
					+ uint(output_y)) * push.output_width + uint(output_x);
				uint weight_offset =
					((input_channel * push.output_channels + output_channel) * push.kernel_size
					+ kernel_y) * push.kernel_size + kernel_x;
				gradient += load_f32(
					storage_buffers[push.output_gradient_index],
					output_offset
				) * load_f32(storage_buffers[push.weight_index], weight_offset);
			}
		}
	}
	store_f32(storage_buffers[push.input_gradient_index], input_index, gradient);
}
