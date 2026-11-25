import attributes;
import storage;

struct PushConstants {
	uint output_index;
	uint batch_size, channels, height, width;
	uint distance;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("distance_transform_relax")]
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
	RWByteAddressBuffer output_buffer = storage_buffers[push.output_index];
	float best = load_f32(output_buffer, index);
	for (int dy = -1; dy <= 1; ++dy) {
		for (int dx = -1; dx <= 1; ++dx) {
			if (dx == 0 && dy == 0) continue;
			bool diagonal = dx != 0 && dy != 0;
			if (push.distance == 1u && diagonal) continue;
			float weight = diagonal ? 1.4142135 : 1.0;
			int nx = int(x) + dx;
			int ny = int(y) + dy;
			if (nx < 0 || ny < 0 || nx >= int(push.width) || ny >= int(push.height)) continue;
			uint neighbor = plane * plane_size + uint(ny) * push.width + uint(nx);
			best = min(best, load_f32(output_buffer, neighbor) + weight);
		}
	}
	store_f32(output_buffer, index, best);
}
