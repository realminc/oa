// Integer-scale nearest-neighbor upsampling for FP32 NCHW matrices.

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
	uint scale_factor;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("upsample_2d")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint output_x = group_id.x * 16u + group_thread_id.x;
	uint output_y = group_id.y * 16u + group_thread_id.y;
	uint batch_channel = group_id.z;
	if (output_x >= push.output_width || output_y >= push.output_height) {
		return;
	}
	uint batch = batch_channel / push.channels;
	uint channel = batch_channel % push.channels;
	if (batch >= push.batch_size) {
		return;
	}
	uint input_x = output_x / push.scale_factor;
	uint input_y = output_y / push.scale_factor;
	uint input_index = ((batch * push.channels + channel) * push.input_height + input_y)
		* push.input_width + input_x;
	uint output_index = ((batch * push.channels + channel) * push.output_height + output_y)
		* push.output_width + output_x;
	store_f32(
		storage_buffers[push.output_index],
		output_index,
		load_f32(storage_buffers[push.input_index], input_index)
	);
}
