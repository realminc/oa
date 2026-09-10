// Deterministic nearest-neighbor upsampling adjoint for FP32 NCHW matrices.

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
	uint scale_factor;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("upsample_2d_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint input_x = group_id.x * 16u + group_thread_id.x;
	uint input_y = group_id.y * 16u + group_thread_id.y;
	uint batch_channel = group_id.z;
	if (input_x >= push.input_width || input_y >= push.input_height) {
		return;
	}
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	if (batch >= push.batch_size) {
		return;
	}
	float gradient = 0.0f;
	for (uint offset_y = 0u; offset_y < push.scale_factor; ++offset_y) {
		uint output_y = input_y * push.scale_factor + offset_y;
		for (uint offset_x = 0u; offset_x < push.scale_factor; ++offset_x) {
			uint output_x = input_x * push.scale_factor + offset_x;
			uint output_index = ((batch * push.channels + channel) * push.output_height + output_y)
				* push.output_width + output_x;
			gradient += load_f32(storage_buffers[push.output_gradient_index], output_index);
		}
	}
	uint input_index = ((batch * push.channels + channel) * push.input_height + input_y)
		* push.input_width + input_x;
	store_f32(storage_buffers[push.input_gradient_index], input_index, gradient);
}
