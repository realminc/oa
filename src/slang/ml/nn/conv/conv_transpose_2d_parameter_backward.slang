// Mechanical adaptation of OA's deterministic ConvTranspose2d parameter adjoint.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/conv/convTranspose2dBwdWeight.slang

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
	uint input_height;
	uint input_width;
	uint kernel_size;
	uint stride;
	uint padding;
	uint output_height;
	uint output_width;
	uint parameter_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_transpose_2d_parameter_backward")]
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
	uint weight_count = push.input_channels * push.output_channels
		* push.kernel_size * push.kernel_size;
	if (parameter_index >= weight_count) {
		uint output_channel = parameter_index - weight_count;
		float bias_gradient = 0.0f;
		for (uint batch = 0; batch < push.batch_size; ++batch) {
			for (uint output_y = 0; output_y < push.output_height; ++output_y) {
				for (uint output_x = 0; output_x < push.output_width; ++output_x) {
					uint output_offset =
						((batch * push.output_channels + output_channel) * push.output_height
						+ output_y) * push.output_width + output_x;
					bias_gradient += load_f32(
						storage_buffers[push.output_gradient_index],
						output_offset
					);
				}
			}
		}
		store_f32(storage_buffers[push.bias_gradient_index], output_channel, bias_gradient);
		return;
	}

	uint kernel_x = parameter_index % push.kernel_size;
	uint quotient = parameter_index / push.kernel_size;
	uint kernel_y = quotient % push.kernel_size;
	quotient /= push.kernel_size;
	uint output_channel = quotient % push.output_channels;
	uint input_channel = quotient / push.output_channels;
	float weight_gradient = 0.0f;
	for (uint batch = 0; batch < push.batch_size; ++batch) {
		for (uint input_y = 0; input_y < push.input_height; ++input_y) {
			int output_y = int(input_y * push.stride + kernel_y) - int(push.padding);
			if (output_y < 0 || output_y >= int(push.output_height)) {
				continue;
			}
			for (uint input_x = 0; input_x < push.input_width; ++input_x) {
				int output_x = int(input_x * push.stride + kernel_x) - int(push.padding);
				if (output_x < 0 || output_x >= int(push.output_width)) {
					continue;
				}
				uint input_offset =
					((batch * push.input_channels + input_channel) * push.input_height
					+ input_y) * push.input_width + input_x;
				uint output_offset =
					((batch * push.output_channels + output_channel) * push.output_height
					+ uint(output_y)) * push.output_width + uint(output_x);
				weight_gradient += load_f32(storage_buffers[push.input_index], input_offset)
					* load_f32(storage_buffers[push.output_gradient_index], output_offset);
			}
		}
	}
	store_f32(
		storage_buffers[push.weight_gradient_index],
		parameter_index,
		weight_gradient
	);
}
