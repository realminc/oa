// OARS correction of OA's generated AdaptiveAvgPool2d composition.
// Donor: source/cpp/lib/oa/ml/nn/adaptiveavgpool2d/adaptiveAvgPool2d.cpp

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
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("adaptive_avg_pool_2d")]
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

	uint input_y_start = output_y * push.input_height / push.output_height;
	uint input_x_start = output_x * push.input_width / push.output_width;
	uint input_y_end = ((output_y + 1) * push.input_height + push.output_height - 1)
		/ push.output_height;
	uint input_x_end = ((output_x + 1) * push.input_width + push.output_width - 1)
		/ push.output_width;
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	float sum = 0.0f;
	for (uint input_y = input_y_start; input_y < input_y_end; ++input_y) {
		for (uint input_x = input_x_start; input_x < input_x_end; ++input_x) {
			uint input_offset =
				batch * push.channels * push.input_height * push.input_width
				+ channel * push.input_height * push.input_width
				+ input_y * push.input_width
				+ input_x;
			sum += load_f32(storage_buffers[push.input_index], input_offset);
		}
	}
	uint count = (input_y_end - input_y_start) * (input_x_end - input_x_start);
	uint output_offset =
		batch * push.channels * push.output_height * push.output_width
		+ channel * push.output_height * push.output_width
		+ output_y * push.output_width
		+ output_x;
	store_f32(storage_buffers[push.output_index], output_offset, sum / float(count));
}
