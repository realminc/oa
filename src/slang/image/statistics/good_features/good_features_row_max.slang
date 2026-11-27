import attributes;
import storage;

struct PushConstants {
	uint input_index, row_max_index;
	uint batch_size, channels, height, width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// Per-row response maximum for good_features_to_track: one thread per
// (plane, row) writes the row's largest score.
[kernel_name("good_features_row_max")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint row = thread_id.x;
	uint plane_count = push.batch_size * push.channels;
	uint plane_size = push.height * push.width;
	uint total = plane_count * push.height;
	if (row >= total) return;
	uint plane = row / push.height;
	uint y = row - plane * push.height;
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	uint base = plane * plane_size + y * push.width;
	float best = load_f32(input_buffer, base);
	for (uint x = 1u; x < push.width; x++) {
		best = max(best, load_f32(input_buffer, base + x));
	}
	store_f32(storage_buffers[push.row_max_index], row, best);
}
