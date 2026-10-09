// Mechanical adaptation of OA's Conv1d input adjoint.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/conv1dBwdData.slang

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint weight_index;
	uint input_gradient_index;
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

[kernel_name("conv_1d_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint input_index = thread_id.x;
	uint input_count = push.batch_size * push.input_channels * push.input_length;
	if (input_index >= input_count) {
		return;
	}

	uint input_position = input_index % push.input_length;
	uint quotient = input_index / push.input_length;
	uint input_channel = quotient % push.input_channels;
	uint batch = quotient / push.input_channels;
	float gradient = 0.0f;
	for (uint output_channel = 0; output_channel < push.output_channels; ++output_channel) {
		for (uint kernel = 0; kernel < push.kernel_size; ++kernel) {
			int output_numerator = int(input_position)
				- int(kernel) * int(push.dilation) + int(push.padding);
			if (
				output_numerator < 0
				|| output_numerator % int(push.stride) != 0
			) {
				continue;
			}
			int output_position = output_numerator / int(push.stride);
			if (output_position >= int(push.output_length)) {
				continue;
			}
			uint output_offset =
				(batch * push.output_channels + output_channel) * push.output_length
				+ uint(output_position);
			uint weight_offset =
				(output_channel * push.input_channels + input_channel) * push.kernel_size
				+ kernel;
			gradient += load_f32(storage_buffers[push.output_gradient_index], output_offset)
				* load_f32(storage_buffers[push.weight_index], weight_offset);
		}
	}
	store_f32(storage_buffers[push.input_gradient_index], input_index, gradient);
}
