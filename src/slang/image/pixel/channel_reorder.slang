import storage;
import attributes;

struct PushConstants {
	uint input_index, output_index, batch_size, channels, height, width;
	uint channel_0, channel_1, channel_2, channel_3;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("channel_reorder")]
[domain("image")]
[variant("explicit_semantic_format")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane_index = thread_id.z;
	if (x >= push.width || y >= push.height || plane_index >= push.batch_size * push.channels) return;
	uint batch = plane_index / push.channels, channel = plane_index % push.channels;
	uint order[4] = {push.channel_0, push.channel_1, push.channel_2, push.channel_3};
	uint plane = push.height * push.width;
	uint pixel = y * push.width + x;
	uint source = batch * push.channels * plane + order[channel] * plane + pixel;
	uint target = batch * push.channels * plane + channel * plane + pixel;
	store_f32(storage_buffers[push.output_index], target,
		load_f32(storage_buffers[push.input_index], source));
}
