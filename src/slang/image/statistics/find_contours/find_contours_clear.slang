import attributes;
import storage;

struct PushConstants {
	uint seeds_index, points_index, lengths_index;
	uint seed_elements, point_elements, length_elements;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("find_contours_clear")]
[domain("image")]
[variant("replay_safe_zero")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index < push.seed_elements) {
		store_u32(storage_buffers[push.seeds_index], index, 0u);
	}
	if (index < push.point_elements) {
		store_u32(storage_buffers[push.points_index], index, 0u);
	}
	if (index < push.length_elements) {
		store_u32(storage_buffers[push.lengths_index], index, 0u);
	}
}
