// NCHW MaxUnpool2d forward: the recorded argmax scatter adjoint of MaxPool2d.
// Each output element scans the windows covering it; the window whose index
// names this element contributes its pooled value. Overlapping windows resolve
// to the highest pooled index, matching a sequential scatter order.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint indices_index;
	uint output_index;
	uint batch_size;
	uint channels;
	uint output_height;
	uint output_width;
	uint input_height;
	uint input_width;
	uint kernel_size;
	uint stride;
	uint padding;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("max_unpool_2d")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_y = thread_id.x;
	uint output_x = thread_id.y;
	uint batch_channel = thread_id.z;
	if (
		output_y >= push.output_height
		|| output_x >= push.output_width
		|| batch_channel >= push.batch_size * push.channels
	) {
		return;
	}

	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	uint plane = push.output_height * push.output_width;
	uint input_plane = push.input_height * push.input_width;
	uint output_offset =
		(batch * push.channels + channel) * plane
		+ output_y * push.output_width
		+ output_x;
	float value = 0.0f;
	for (uint input_y = 0; input_y < push.input_height; ++input_y) {
		int output_y_start = int(input_y * push.stride) - int(push.padding);
		if (
			int(output_y) < output_y_start
			|| int(output_y) >= output_y_start + int(push.kernel_size)
		) {
			continue;
		}
		for (uint input_x = 0; input_x < push.input_width; ++input_x) {
			int output_x_start = int(input_x * push.stride) - int(push.padding);
			if (
				int(output_x) < output_x_start
				|| int(output_x) >= output_x_start + int(push.kernel_size)
			) {
				continue;
			}
			uint input_offset =
				(batch * push.channels + channel) * input_plane
				+ input_y * push.input_width
				+ input_x;
			if (
				load_u32(storage_buffers[push.indices_index], input_offset)
				== output_offset
			) {
				value = load_f32(storage_buffers[push.input_index], input_offset);
			}
		}
	}
	store_f32(storage_buffers[push.output_index], output_offset, value);
}
