// Mechanical adaptation of OA's NCHW AvgPool2d forward kernel to NCDHW AvgPool3d.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/pooling/avgPool2d.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
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

[kernel_name("avg_pool_3d")]
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
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	int input_z_start = int(output_z * push.stride) - int(push.padding);
	int input_y_start = int(output_y * push.stride) - int(push.padding);
	int input_x_start = int(output_x * push.stride) - int(push.padding);
	uint plane = push.input_height * push.input_width;
	float sum = 0.0f;
	uint count = 0;
	for (uint kernel_z = 0; kernel_z < push.kernel_size; ++kernel_z) {
		for (uint kernel_y = 0; kernel_y < push.kernel_size; ++kernel_y) {
			for (uint kernel_x = 0; kernel_x < push.kernel_size; ++kernel_x) {
				int input_z = input_z_start + int(kernel_z);
				int input_y = input_y_start + int(kernel_y);
				int input_x = input_x_start + int(kernel_x);
				if (
					input_z >= 0
					&& input_z < int(push.input_depth)
					&& input_y >= 0
					&& input_y < int(push.input_height)
					&& input_x >= 0
					&& input_x < int(push.input_width)
				) {
					uint input_offset =
						(batch * push.channels + channel) * push.input_depth * plane
						+ uint(input_z) * plane
						+ uint(input_y) * push.input_width
						+ uint(input_x);
					sum += load_f32(storage_buffers[push.input_index], input_offset);
					count += 1;
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
	float average = count > 0 ? sum / float(count) : 0.0f;
	store_f32(storage_buffers[push.output_index], output_offset, average);
}
