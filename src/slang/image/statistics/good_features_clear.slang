import attributes;
import storage;

struct PushConstants {
	uint keypoints_index, counts_index;
	uint batch_size, channels, height, width;
	uint max_corners;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// Zero-fill the keypoint rows and per-plane survivor counters before
// good_features_to_track compaction.
[kernel_name("good_features_clear")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint plane_count = push.batch_size * push.channels;
	uint total = plane_count * push.max_corners * 3u;
	if (index >= total) return;
	store_f32(storage_buffers[push.keypoints_index], index, 0.0);
	if (index < plane_count) {
		store_u32(storage_buffers[push.counts_index], index, 0u);
	}
}
