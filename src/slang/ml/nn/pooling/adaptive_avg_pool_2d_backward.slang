// OARS input adjoint for corrected AdaptiveAvgPool2d geometry.
// Donor: source/cpp/lib/oa/ml/nn/adaptiveavgpool2d/adaptiveAvgPool2d.cpp

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
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("adaptive_avg_pool_2d_backward")]
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
		uint input_y_start = output_y * push.input_height / push.output_height;
		uint input_y_end = ((output_y + 1) * push.input_height + push.output_height - 1)
			/ push.output_height;
		if (input_y < input_y_start || input_y >= input_y_end) {
			continue;
		}
		for (uint output_x = 0; output_x < push.output_width; ++output_x) {
			uint input_x_start = output_x * push.input_width / push.output_width;
			uint input_x_end = ((output_x + 1) * push.input_width + push.output_width - 1)
				/ push.output_width;
			if (input_x >= input_x_start && input_x < input_x_end) {
				uint output_offset =
					batch * push.channels * push.output_height * push.output_width
					+ channel * push.output_height * push.output_width
					+ output_y * push.output_width
					+ output_x;
				uint count = (input_y_end - input_y_start) * (input_x_end - input_x_start);
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
