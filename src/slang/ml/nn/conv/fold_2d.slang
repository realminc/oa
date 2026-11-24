// Deterministic col2im adjoint of the NCHW unfold lowering: each output
// element accumulates every column tap that lands on it; no atomics.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/im2Col1d.slang

import storage;
import attributes;

struct PushConstants {
	uint columns_index;
	uint output_index;
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

[kernel_name("fold")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint image_area = push.input_height * push.input_width;
	uint element_count = push.batch_size * push.input_channels * image_area;
	if (index >= element_count) {
		return;
	}

	uint input_x = index % push.input_width;
	uint input_y = (index / push.input_width) % push.input_height;
	uint rest = index / image_area;
	uint input_channel = rest % push.input_channels;
	uint batch = rest / push.input_channels;
	uint kernel_area = push.kernel_size * push.kernel_size;
	uint column_size = push.input_channels * kernel_area;
	uint output_length = push.output_height * push.output_width;

	float value = 0.0f;
	for (uint kernel_y = 0; kernel_y < push.kernel_size; ++kernel_y) {
		int output_num_y = int(input_y) + int(push.padding)
			- int(kernel_y) * int(push.dilation);
		if (output_num_y < 0 || output_num_y % int(push.stride) != 0) {
			continue;
		}
		uint output_y = uint(output_num_y) / push.stride;
		if (output_y >= push.output_height) {
			continue;
		}
		for (uint kernel_x = 0; kernel_x < push.kernel_size; ++kernel_x) {
			int output_num_x = int(input_x) + int(push.padding)
				- int(kernel_x) * int(push.dilation);
			if (output_num_x < 0 || output_num_x % int(push.stride) != 0) {
				continue;
			}
			uint output_x = uint(output_num_x) / push.stride;
			if (output_x >= push.output_width) {
				continue;
			}
			uint column = input_channel * kernel_area
				+ kernel_y * push.kernel_size + kernel_x;
			uint columns_offset =
				(batch * column_size + column) * output_length
				+ output_y * push.output_width + output_x;
			value += load_f32(storage_buffers[push.columns_index], columns_offset);
		}
	}
	store_f32(storage_buffers[push.output_index], index, value);
}
