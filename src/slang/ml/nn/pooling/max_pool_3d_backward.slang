// Mechanical adaptation of OA's NCHW MaxPool2d input adjoint to NCDHW:
// each input element accumulates the gradient of every output window that
// selected it, identified through the recorded argmax indices.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/pooling/maxPool2dBwd.slang

import storage;
import attributes;

struct PushConstants {
	uint indices_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint batch_size;
	uint channels;
	uint input_depth;
	uint input_height;
	uint input_width;
	uint output_depth;
	uint output_height;
	uint output_width;
	uint kernel_size;
	uint stride;
	uint padding;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("max_pool_3d_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint input_y = thread_id.x;
	uint input_x = thread_id.y;
	uint batch_channel_depth = thread_id.z;
	if (
		input_y >= push.input_height
		|| input_x >= push.input_width
		|| batch_channel_depth >= push.batch_size * push.channels * push.input_depth
	) {
		return;
	}

	uint input_z = batch_channel_depth % push.input_depth;
	uint batch_channel = batch_channel_depth / push.input_depth;
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	uint plane = push.input_height * push.input_width;
	uint input_offset =
		(batch * push.channels + channel) * push.input_depth * plane
		+ input_z * plane
		+ input_y * push.input_width
		+ input_x;
	uint out_plane = push.output_height * push.output_width;
	float gradient_sum = 0.0f;
	for (uint output_z = 0; output_z < push.output_depth; ++output_z) {
		int input_z_start = int(output_z * push.stride) - int(push.padding);
		if (int(input_z) < input_z_start || int(input_z) >= input_z_start + int(push.kernel_size)) {
			continue;
		}
		for (uint output_y = 0; output_y < push.output_height; ++output_y) {
			int input_y_start = int(output_y * push.stride) - int(push.padding);
			if (
				int(input_y) < input_y_start
				|| int(input_y) >= input_y_start + int(push.kernel_size)
			) {
				continue;
			}
			for (uint output_x = 0; output_x < push.output_width; ++output_x) {
				int input_x_start = int(output_x * push.stride) - int(push.padding);
				if (
					int(input_x) < input_x_start
					|| int(input_x) >= input_x_start + int(push.kernel_size)
				) {
					continue;
				}
				uint output_offset =
					(batch * push.channels + channel) * push.output_depth * out_plane
					+ output_z * out_plane
					+ output_y * push.output_width
					+ output_x;
				if (
					load_u32(storage_buffers[push.indices_index], output_offset)
					== input_offset
				) {
					gradient_sum += load_f32(
						storage_buffers[push.output_gradient_index], output_offset
					);
				}
			}
		}
	}
	store_f32(storage_buffers[push.input_gradient_index], input_offset, gradient_sum);
}
