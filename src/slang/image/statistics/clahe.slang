import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index;
	uint batch_size, channels, height, width, tile_width, tile_height, tiles_x;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("clahe")]
[domain("image")]
[variant("atomic_u32")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint x = thread_id.x, y = thread_id.y, plane = thread_id.z;
	if (x >= push.width || y >= push.height || plane >= push.batch_size * push.channels) return;
	float value = load_f32(storage_buffers[push.input_index],
		(plane * push.height + y) * push.width + x);
	uint bucket = min(uint(clamp(value, 0.0, 1.0) * 256.0), 255u);
	uint tiles_y = (push.height + push.tile_height - 1u) / push.tile_height;
	uint tile = (plane * tiles_y + y / push.tile_height) * push.tiles_x
		+ x / push.tile_width;
	uint ignored;
	storage_buffers[push.output_index].InterlockedAdd(
		(tile * 256u + bucket) * 4u, 1u, ignored);
}
