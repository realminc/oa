import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, marker_index;
	uint batch_size, channels, height, width;
	uint seed_x, seed_y;
	float value;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("flood_fill_init")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x;
	uint y = thread_id.y;
	uint plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	uint index = plane * push.height * push.width + y * push.width + x;
	bool seed = x == push.seed_x && y == push.seed_y;
	float source = load_f32(storage_buffers[push.input_index], index);
	store_f32(storage_buffers[push.output_index], index, seed ? push.value : source);
	store_u32(storage_buffers[push.marker_index], index, seed ? 1u : 0u);
}
