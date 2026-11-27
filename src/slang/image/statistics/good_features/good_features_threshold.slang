import attributes;
import storage;

struct PushConstants {
	uint row_max_index, thresholds_index;
	uint batch_size, channels, height;
	float quality;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// Per-plane quality threshold for good_features_to_track: one thread per
// plane reduces the row maxima and emits quality * plane_max.
[kernel_name("good_features_threshold")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint plane = thread_id.x;
	uint plane_count = push.batch_size * push.channels;
	if (plane >= plane_count) return;
	RWByteAddressBuffer row_max = storage_buffers[push.row_max_index];
	uint base = plane * push.height;
	float best = load_f32(row_max, base);
	for (uint y = 1u; y < push.height; y++) {
		best = max(best, load_f32(row_max, base + y));
	}
	store_f32(storage_buffers[push.thresholds_index], plane, best * push.quality);
}
