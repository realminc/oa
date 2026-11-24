// Mechanical adaptation of the NCHW adaptive-max kernel to NCDHW AdaptiveMaxPool3d:
// floor/ceil bins per dimension, maximum plus recorded argmax index.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint indices_index;
	uint batch_size;
	uint channels;
	uint input_depth;
	uint input_height;
	uint input_width;
	uint output_depth;
	uint output_height;
	uint output_width;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("adaptive_max_pool_3d")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_y = thread_id.x;
	uint output_x = thread_id.y;
	uint batch_channel_depth = thread_id.z;
	if (
		output_y >= push.output_height
		|| output_x >= push.output_width
		|| batch_channel_depth >= push.batch_size * push.channels * push.output_depth
	) {
		return;
	}

	uint output_z = batch_channel_depth % push.output_depth;
	uint batch_channel = batch_channel_depth / push.output_depth;
	uint input_z_start = output_z * push.input_depth / push.output_depth;
	uint input_y_start = output_y * push.input_height / push.output_height;
	uint input_x_start = output_x * push.input_width / push.output_width;
	uint input_z_end = ((output_z + 1) * push.input_depth + push.output_depth - 1)
		/ push.output_depth;
	uint input_y_end = ((output_y + 1) * push.input_height + push.output_height - 1)
		/ push.output_height;
	uint input_x_end = ((output_x + 1) * push.input_width + push.output_width - 1)
		/ push.output_width;
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	uint plane = push.input_height * push.input_width;
	float maximum = -3.402823466e+38f;
	uint maximum_index = 0;
	for (uint input_z = input_z_start; input_z < input_z_end; ++input_z) {
		for (uint input_y = input_y_start; input_y < input_y_end; ++input_y) {
			for (uint input_x = input_x_start; input_x < input_x_end; ++input_x) {
				uint input_offset =
					(batch * push.channels + channel) * push.input_depth * plane
					+ input_z * plane
					+ input_y * push.input_width
					+ input_x;
				float value = load_f32(storage_buffers[push.input_index], input_offset);
				if (value > maximum) {
					maximum = value;
					maximum_index = input_offset;
				}
			}
		}
	}
	uint out_plane = push.output_height * push.output_width;
	uint output_offset =
		(batch * push.channels + channel) * push.output_depth * out_plane
		+ output_z * out_plane
		+ output_y * push.output_width
		+ output_x;
	store_f32(storage_buffers[push.output_index], output_offset, maximum);
	store_u32(storage_buffers[push.indices_index], output_offset, maximum_index);
}
