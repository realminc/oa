// Mechanical adaptation of OA's NCHW Conv2dTranspose kernel to NCDHW
// Conv3dTranspose: per-output-element gather over the strided IODHW weight.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/convTranspose2d.slang

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
	uint input_depth;
	uint input_height;
	uint input_width;
	uint kernel_size;
	uint stride;
	uint padding;
	uint output_depth;
	uint output_height;
	uint output_width;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_3d_transpose")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_index = thread_id.x;
	uint out_plane = push.output_height * push.output_width;
	uint output_count = push.batch_size * push.output_channels
		* push.output_depth * out_plane;
	if (output_index >= output_count) {
		return;
	}

	uint output_x = output_index % push.output_width;
	uint quotient = output_index / push.output_width;
	uint output_y = quotient % push.output_height;
	quotient /= push.output_height;
	uint output_z = quotient % push.output_depth;
	quotient /= push.output_depth;
	uint output_channel = quotient % push.output_channels;
	uint batch = quotient / push.output_channels;
	uint in_plane = push.input_height * push.input_width;
	uint kernel_volume = push.kernel_size * push.kernel_size * push.kernel_size;
	float value = load_f32(storage_buffers[push.bias_index], output_channel);
	for (uint input_channel = 0; input_channel < push.input_channels; ++input_channel) {
		for (uint kernel_z = 0; kernel_z < push.kernel_size; ++kernel_z) {
			int input_z_numerator =
				int(output_z) - int(kernel_z) + int(push.padding);
			if (input_z_numerator < 0 || input_z_numerator % int(push.stride) != 0) {
				continue;
			}
			int input_z = input_z_numerator / int(push.stride);
			if (input_z >= int(push.input_depth)) {
				continue;
			}
			for (uint kernel_y = 0; kernel_y < push.kernel_size; ++kernel_y) {
				int input_y_numerator =
					int(output_y) - int(kernel_y) + int(push.padding);
				if (input_y_numerator < 0 || input_y_numerator % int(push.stride) != 0) {
					continue;
				}
				int input_y = input_y_numerator / int(push.stride);
				if (input_y >= int(push.input_height)) {
					continue;
				}
				for (uint kernel_x = 0; kernel_x < push.kernel_size; ++kernel_x) {
					int input_x_numerator =
						int(output_x) - int(kernel_x) + int(push.padding);
					if (
						input_x_numerator < 0
						|| input_x_numerator % int(push.stride) != 0
					) {
						continue;
					}
					int input_x = input_x_numerator / int(push.stride);
					if (input_x >= int(push.input_width)) {
						continue;
					}
					uint input_offset =
						((batch * push.input_channels + input_channel)
							* push.input_depth + uint(input_z)) * in_plane
						+ uint(input_y) * push.input_width
						+ uint(input_x);
					uint weight_offset =
						(input_channel * push.output_channels + output_channel)
							* kernel_volume
						+ kernel_z * push.kernel_size * push.kernel_size
						+ kernel_y * push.kernel_size
						+ kernel_x;
					value += load_f32(storage_buffers[push.input_index], input_offset)
						* load_f32(storage_buffers[push.weight_index], weight_offset);
				}
			}
		}
	}
	store_f32(storage_buffers[push.output_index], output_index, value);
}
