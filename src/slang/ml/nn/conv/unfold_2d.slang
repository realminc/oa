// Mechanical extension of OA's Conv1d im2col lowering to NCHW unfold columns.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/im2Col1d.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint columns_index;
	uint batch_size;
	uint input_channels;
	uint input_height;
	uint input_width;
	uint kernel_size;
	uint stride;
	uint padding;
	uint dilation;
	uint output_height;
	uint output_width;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("unfold")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint kernel_area = push.kernel_size * push.kernel_size;
	uint column_size = push.input_channels * kernel_area;
	uint output_length = push.output_height * push.output_width;
	uint element_count = push.batch_size * column_size * output_length;
	if (index >= element_count) {
		return;
	}

	uint output_position = index % output_length;
	uint rest = index / output_length;
	uint column = rest % column_size;
	uint batch = rest / column_size;
	uint kernel_x = column % push.kernel_size;
	uint kernel_y = (column / push.kernel_size) % push.kernel_size;
	uint input_channel = column / kernel_area;
	uint output_y = output_position / push.output_width;
	uint output_x = output_position % push.output_width;
	int input_y = int(output_y) * int(push.stride)
		+ int(kernel_y) * int(push.dilation) - int(push.padding);
	int input_x = int(output_x) * int(push.stride)
		+ int(kernel_x) * int(push.dilation) - int(push.padding);
	float value = 0.0f;
	if (input_y >= 0 && input_y < int(push.input_height)
		&& input_x >= 0 && input_x < int(push.input_width)) {
		uint input_offset =
			((batch * push.input_channels + input_channel) * push.input_height
				+ uint(input_y)) * push.input_width
			+ uint(input_x);
		value = load_f32(storage_buffers[push.input_index], input_offset);
	}
	store_f32(storage_buffers[push.columns_index], index, value);
}
