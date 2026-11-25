import attributes;
import storage;

struct PushConstants {
	uint input_index, output_index, marker_index;
	uint batch_size, channels, height, width;
	uint connectivity;
	float lo_diff, up_diff, value;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("flood_fill_relax")]
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
	uint plane_size = push.height * push.width;
	uint index = plane * plane_size + y * push.width + x;
	RWByteAddressBuffer marker_buffer = storage_buffers[push.marker_index];
	if (load_u32(marker_buffer, index) != 0u) return;
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	float candidate = load_f32(input_buffer, index);
	bool joins = false;
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			if (dx == 0 && dy == 0) continue;
			if (push.connectivity == 4u && dx != 0 && dy != 0) continue;
			int nx = int(x) + dx;
			int ny = int(y) + dy;
			if (nx < 0 || ny < 0 || nx >= int(push.width) || ny >= int(push.height)) continue;
			uint neighbor = plane * plane_size + uint(ny) * push.width + uint(nx);
			if (load_u32(marker_buffer, neighbor) == 0u) continue;
			float source = load_f32(input_buffer, neighbor);
			float delta = candidate - source;
			if (delta >= -push.lo_diff && delta <= push.up_diff) joins = true;
		}
	}
	if (!joins) return;
	store_u32(marker_buffer, index, 1u);
	store_f32(storage_buffers[push.output_index], index, push.value);
}
