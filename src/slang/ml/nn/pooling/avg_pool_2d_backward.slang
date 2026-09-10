// Mechanical adaptation of OA's NCHW AvgPool2d input adjoint.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/pooling/avgPool2dBwd.slang

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint input_gradient_index;
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

[kernel_name("avg_pool_2d_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint input_y = thread_id.x;
	uint input_x = thread_id.y;
	uint batch_channel = thread_id.z;
	if (
		input_y >= push.input_height
		|| input_x >= push.input_width
		|| batch_channel >= push.batch_size * push.channels
	) {
		return;
	}

	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	float gradient_sum = 0.0f;
	for (uint output_y = 0; output_y < push.output_height; ++output_y) {
		for (uint output_x = 0; output_x < push.output_width; ++output_x) {
			int input_y_start = int(output_y * push.stride) - int(push.padding);
			int input_x_start = int(output_x * push.stride) - int(push.padding);
			int input_y_end = input_y_start + int(push.kernel_size);
			int input_x_end = input_x_start + int(push.kernel_size);
			if (
				int(input_y) >= input_y_start
				&& int(input_y) < input_y_end
				&& int(input_x) >= input_x_start
				&& int(input_x) < input_x_end
			) {
				uint count = 0;
				for (uint kernel_y = 0; kernel_y < push.kernel_size; ++kernel_y) {
					for (uint kernel_x = 0; kernel_x < push.kernel_size; ++kernel_x) {
						int y = input_y_start + int(kernel_y);
						int x = input_x_start + int(kernel_x);
						if (
							y >= 0
							&& y < int(push.input_height)
							&& x >= 0
							&& x < int(push.input_width)
						) {
							count += 1;
						}
					}
				}
				uint output_offset =
					batch * push.channels * push.output_height * push.output_width
					+ channel * push.output_height * push.output_width
					+ output_y * push.output_width
					+ output_x;
				gradient_sum += load_f32(
					storage_buffers[push.output_gradient_index], output_offset
				) / float(count);
			}
		}
	}

	uint input_offset =
		batch * push.channels * push.input_height * push.input_width
		+ channel * push.input_height * push.input_width
		+ input_y * push.input_width
		+ input_x;
	store_f32(storage_buffers[push.input_gradient_index], input_offset, gradient_sum);
}
