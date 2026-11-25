import attributes;
import storage;

struct PushConstants {
	uint input_index, thresholds_index, keypoints_index, counts_index;
	uint batch_size, channels, height, width;
	uint max_corners;
	float min_distance;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// Radius-suppressed keypoint selection: a pixel above the plane threshold
// survives when no strictly stronger candidate lies within min_distance
// (equal scores lose to the lower linear index). Survivors claim atomic
// slots capped at max_corners per plane.
[kernel_name("good_features_select")]
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
	uint plane_count = push.batch_size * push.channels;
	uint plane_size = push.height * push.width;
	if (x >= push.width || y >= push.height || plane >= plane_count) return;
	RWByteAddressBuffer input_buffer = storage_buffers[push.input_index];
	uint index = plane * plane_size + y * push.width + x;
	float score = load_f32(input_buffer, index);
	float threshold = load_f32(storage_buffers[push.thresholds_index], plane);
	if (score < threshold || score == 0.0) return;
	uint radius = uint(ceil(push.min_distance));
	uint base = plane * plane_size;
	bool suppressed = false;
	for (uint dy = 0u; dy <= radius && !suppressed; dy++) {
		int sy = int(y) - int(radius) + int(dy);
		if (sy < 0 || sy >= int(push.height)) continue;
		for (uint dx = 0u; dx <= radius; dx++) {
			int sx = int(x) - int(radius) + int(dx);
			if (sx < 0 || sx >= int(push.width)) continue;
			uint neighbor = uint(sy) * push.width + uint(sx);
			if (neighbor == y * push.width + x) continue;
			float other = load_f32(input_buffer, base + neighbor);
			if (other < threshold) continue;
			float dist = sqrt(float(sx - int(x)) * float(sx - int(x)) + float(sy - int(y)) * float(sy - int(y)));
			if (dist > push.min_distance) continue;
			if (other > score || (other == score && neighbor < y * push.width + x)) {
				suppressed = true;
				break;
			}
		}
	}
	if (suppressed) return;
	RWByteAddressBuffer counts = storage_buffers[push.counts_index];
	uint slot;
	counts.InterlockedAdd(plane * 4, 1u, slot);
	if (slot >= push.max_corners) return;
	RWByteAddressBuffer keypoints = storage_buffers[push.keypoints_index];
	uint out_base = (plane * push.max_corners + slot) * 3;
	store_f32(keypoints, out_base, float(x));
	store_f32(keypoints, out_base + 1, float(y));
	store_f32(keypoints, out_base + 2, score);
}
