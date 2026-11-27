import attributes;
import storage;

struct PushConstants {
	uint labels_index, seeds_index, points_index, lengths_index;
	uint plane_count, max_contours, height, width;
	uint max_points;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const int DX[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
static const int DY[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };

[kernel_name("find_contours_trace")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint slot = thread_id.x;
	if (slot >= push.plane_count * push.max_contours) return;
	uint plane = slot / push.max_contours;
	uint entry = slot - plane * push.max_contours;
	uint stride = push.max_contours + 1u;
	uint seed_base = plane * stride;
	uint count = load_u32(storage_buffers[push.seeds_index], seed_base);
	if (entry >= count) return;
	uint seed = load_u32(storage_buffers[push.seeds_index], seed_base + 1u + entry);
	if (seed == 0xFFFFFFFFu) return;
	uint plane_size = push.height * push.width;
	uint local = seed - plane * plane_size;
	int sx = int(local % push.width);
	int sy = int(local / push.width);
	uint point_base = slot * push.max_points * 2u;
	RWByteAddressBuffer points_buffer = storage_buffers[push.points_index];
	RWByteAddressBuffer labels_buffer = storage_buffers[push.labels_index];
	store_u32(points_buffer, point_base, asuint(float(sx)));
	store_u32(points_buffer, point_base + 1u, asuint(float(sy)));
	int cx = sx;
	int cy = sy;
	uint previous_direction = 0u;
	uint written = 1u;
	bool closed = false;
	for (uint step = 0u; step + 1u < push.max_points; ++step) {
		uint backtrack = (previous_direction + 4u) % 8u;
		int next_x = cx;
		int next_y = cy;
		uint next_direction = previous_direction;
		bool advanced = false;
		for (uint probe = 0u; probe < 8u; ++probe) {
			uint direction = (backtrack + 1u + probe) % 8u;
			int nx = cx + DX[direction];
			int ny = cy + DY[direction];
			if (nx < 0 || ny < 0 || nx >= int(push.width) || ny >= int(push.height)) continue;
			uint neighbor = plane * plane_size + uint(ny) * push.width + uint(nx);
			if (load_u32(labels_buffer, neighbor) == seed) {
				next_x = nx;
				next_y = ny;
				next_direction = direction;
				advanced = true;
				break;
			}
		}
		if (!advanced) break;
		cx = next_x;
		cy = next_y;
		previous_direction = next_direction;
		if (cx == sx && cy == sy && written >= 2u) {
			closed = true;
			break;
		}
		store_u32(points_buffer, point_base + written * 2u, asuint(float(cx)));
		store_u32(points_buffer, point_base + written * 2u + 1u, asuint(float(cy)));
		written += 1u;
	}
	store_u32(storage_buffers[push.lengths_index], slot, written);
}
