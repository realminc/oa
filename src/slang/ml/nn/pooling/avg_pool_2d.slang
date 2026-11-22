// Mechanical adaptation of OA's NCHW AvgPool2d forward kernel.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/pooling/avgPool2d.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint batch_size;
	uint channels;
	uint input_height;
	uint input_width;
	uint output_height;
	uint output_width;
	uint kernel_size;
	uint stride;
	uint padding;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("avg_pool_2d")]
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
	int input_y_start = int(output_y * push.stride) - int(push.padding);
	int input_x_start = int(output_x * push.stride) - int(push.padding);
	float sum = 0.0f;
	uint count = 0;
	for (uint kernel_y = 0; kernel_y < push.kernel_size; ++kernel_y) {
		for (uint kernel_x = 0; kernel_x < push.kernel_size; ++kernel_x) {
			int input_y = input_y_start + int(kernel_y);
			int input_x = input_x_start + int(kernel_x);
			if (
				input_y >= 0
				&& input_y < int(push.input_height)
				&& input_x >= 0
				&& input_x < int(push.input_width)
			) {
				uint input_offset =
					batch * push.channels * push.input_height * push.input_width
					+ channel * push.input_height * push.input_width
					+ uint(input_y) * push.input_width
					+ uint(input_x);
				sum += load_f32(storage_buffers[push.input_index], input_offset);
				count += 1;
			}
		}
	}

	uint output_offset =
		batch * push.channels * push.output_height * push.output_width
		+ channel * push.output_height * push.output_width
		+ output_y * push.output_width
		+ output_x;
	float average = count > 0 ? sum / float(count) : 0.0f;
	store_f32(storage_buffers[push.output_index], output_offset, average);
}
