// Mechanical adaptation of OA's deterministic Conv1d weight/bias adjoint.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/conv1dBwdWeight.slang

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_gradient_index;
	uint weight_gradient_index;
	uint bias_gradient_index;
	uint batch_size;
	uint input_channels;
	uint output_channels;
	uint input_length;
	uint kernel_size;
	uint stride;
	uint padding;
	uint dilation;
	uint output_length;
	uint parameter_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_1d_parameter_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint parameter_index = thread_id.x;
	if (parameter_index >= push.parameter_count) {
		return;
	}
	uint weight_count = push.output_channels * push.input_channels * push.kernel_size;
	if (parameter_index >= weight_count) {
		uint output_channel = parameter_index - weight_count;
		float bias_gradient = 0.0f;
		for (uint batch = 0; batch < push.batch_size; ++batch) {
			for (uint output_position = 0; output_position < push.output_length; ++output_position) {
				uint output_offset =
					(batch * push.output_channels + output_channel) * push.output_length
					+ output_position;
				bias_gradient += load_f32(
					storage_buffers[push.output_gradient_index],
					output_offset
				);
			}
		}
		store_f32(storage_buffers[push.bias_gradient_index], output_channel, bias_gradient);
		return;
	}

	uint kernel = parameter_index % push.kernel_size;
	uint quotient = parameter_index / push.kernel_size;
	uint input_channel = quotient % push.input_channels;
	uint output_channel = quotient / push.input_channels;
	float weight_gradient = 0.0f;
	for (uint batch = 0; batch < push.batch_size; ++batch) {
		for (uint output_position = 0; output_position < push.output_length; ++output_position) {
			int input_position = int(output_position) * int(push.stride)
				+ int(kernel) * int(push.dilation) - int(push.padding);
			if (input_position < 0 || input_position >= int(push.input_length)) {
				continue;
			}
			uint input_offset =
				(batch * push.input_channels + input_channel) * push.input_length
				+ uint(input_position);
			uint output_offset =
				(batch * push.output_channels + output_channel) * push.output_length
				+ output_position;
			weight_gradient += load_f32(storage_buffers[push.input_index], input_offset)
				* load_f32(storage_buffers[push.output_gradient_index], output_offset);
		}
	}
	store_f32(
		storage_buffers[push.weight_gradient_index],
		parameter_index,
		weight_gradient
	);
}
