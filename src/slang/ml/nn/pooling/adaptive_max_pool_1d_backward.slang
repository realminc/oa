// Mechanical adaptation of OA's adaptive max pooling adjoint to NCL: each
// input element accumulates the gradient of every adaptive bin that selected
// it, identified through the recorded argmax indices.

import storage;
import attributes;

struct PushConstants {
	uint indices_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint batch_size;
	uint channels;
	uint input_length;
	uint output_length;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("adaptive_max_pool_1d_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint input_x = thread_id.x;
	uint batch_channel = thread_id.y;
	if (
		input_x >= push.input_length
		|| batch_channel >= push.batch_size * push.channels
	) {
		return;
	}

	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	uint input_offset =
		batch * push.channels * push.input_length
		+ channel * push.input_length
		+ input_x;
	float gradient_sum = 0.0f;
	for (uint output_x = 0; output_x < push.output_length; ++output_x) {
		uint input_x_start = output_x * push.input_length / push.output_length;
		uint input_x_end = ((output_x + 1) * push.input_length + push.output_length - 1)
			/ push.output_length;
		if (input_x >= input_x_start && input_x < input_x_end) {
			uint output_offset =
				batch * push.channels * push.output_length
				+ channel * push.output_length
				+ output_x;
			if (load_u32(storage_buffers[push.indices_index], output_offset) == input_offset) {
				gradient_sum += load_f32(
					storage_buffers[push.output_gradient_index], output_offset
				);
			}
		}
	}
	store_f32(storage_buffers[push.input_gradient_index], input_offset, gradient_sum);
}
