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
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("resize_nearest")]
[domain("image")]
[variant("nearest")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint output_x = thread_id.x;
	uint output_y = thread_id.y;
	uint batch_channel = thread_id.z;
	uint batch_channels = push.batch_size * push.channels;
	if (output_x >= push.output_width || output_y >= push.output_height ||
		batch_channel >= batch_channels) return;

	uint input_x = min(
		uint(floor(float(output_x) * float(push.input_width) / float(push.output_width))),
		push.input_width - 1u);
	uint input_y = min(
		uint(floor(float(output_y) * float(push.input_height) / float(push.output_height))),
		push.input_height - 1u);
	uint input_index = (batch_channel * push.input_height + input_y) * push.input_width + input_x;
	uint output_index =
		(batch_channel * push.output_height + output_y) * push.output_width + output_x;
	store_f32(
		storage_buffers[push.output_index],
		output_index,
		load_f32(storage_buffers[push.input_index], input_index));
}
