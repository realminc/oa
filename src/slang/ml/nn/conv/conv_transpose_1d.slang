// Mechanical adaptation of OA's Conv1d data adjoint for ConvTranspose1d.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/conv1dBwdData.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint weight_index;
	uint output_index;
	uint batch_size;
	uint input_channels;
	uint output_channels;
	uint input_length;
	uint kernel_size;
	uint stride;
	uint padding;
	uint dilation;
	uint output_length;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_transpose_1d")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_index = thread_id.x;
	uint output_count = push.batch_size * push.output_channels * push.output_length;
	if (output_index >= output_count) {
		return;
	}

	uint output_position = output_index % push.output_length;
	uint quotient = output_index / push.output_length;
	uint output_channel = quotient % push.output_channels;
	uint batch = quotient / push.output_channels;
	float value = 0.0f;
	for (uint input_channel = 0; input_channel < push.input_channels; ++input_channel) {
		for (uint kernel = 0; kernel < push.kernel_size; ++kernel) {
			int input_numerator = int(output_position)
				- int(kernel) * int(push.dilation) + int(push.padding);
			if (input_numerator < 0 || input_numerator % int(push.stride) != 0) {
				continue;
			}
			int input_position = input_numerator / int(push.stride);
			if (input_position >= int(push.input_length)) {
				continue;
			}
			uint input_offset =
				(batch * push.input_channels + input_channel) * push.input_length
				+ uint(input_position);
			uint weight_offset =
				(input_channel * push.output_channels + output_channel) * push.kernel_size
				+ kernel;
			value += load_f32(storage_buffers[push.input_index], input_offset)
				* load_f32(storage_buffers[push.weight_index], weight_offset);
		}
	}
	store_f32(storage_buffers[push.output_index], output_index, value);
}
