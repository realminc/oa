import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, batch_size, channels, height, width;
	float mean_0, mean_1, mean_2, std_0, std_1, std_2;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("normalize")]
[domain("image")]
[variant("per_channel_three_parameter")]
[dtype("f32")]
[layout("nchw_or_chw")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	uint channel = plane % push.channels;
	float mean = channel == 0u ? push.mean_0 : channel == 1u ? push.mean_1 : push.mean_2;
	float stddev = channel == 0u ? push.std_0 : channel == 1u ? push.std_1 : push.std_2;
	uint index = (plane * push.height + y) * push.width + x;
	store_f32(storage_buffers[push.output_index], index,
		(load_f32(storage_buffers[push.input_index], index) - mean) / stddev);
}
