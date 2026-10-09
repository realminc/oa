// Mechanical adaptation of OA's Conv1d im2col lowering.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/im2Col1d.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint columns_index;
	uint batch_size;
	uint input_channels;
	uint input_length;
	uint kernel_size;
	uint stride;
	uint padding;
	uint dilation;
	uint output_length;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_1d")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint inner_size = push.input_channels * push.kernel_size;
	uint element_count = push.batch_size * push.output_length * inner_size;
	if (index >= element_count) {
		return;
	}

	uint column = index % inner_size;
	uint row = index / inner_size;
	uint kernel = column % push.kernel_size;
	uint input_channel = column / push.kernel_size;
	uint output_position = row % push.output_length;
	uint batch = row / push.output_length;
	int input_position = int(output_position) * int(push.stride)
		+ int(kernel) * int(push.dilation) - int(push.padding);
	float value = 0.0f;
	if (input_position >= 0 && input_position < int(push.input_length)) {
		uint input_offset =
			(batch * push.input_channels + input_channel) * push.input_length
			+ uint(input_position);
		value = load_f32(storage_buffers[push.input_index], input_offset);
	}
	store_f32(storage_buffers[push.columns_index], index, value);
}
