import attributes;
import storage;

// Steered rBRIEF test-point pairs, byte-parity with OpenCV bit_pattern_31_ (BSD-3).
static const int orb_pattern[1024] = {
	8, -3, 9, 5, 4, 2, 7, -12, -11, 9, -8, 2, 7, -12, 12, -13,
	2, -13, 2, 12, 1, -7, 1, 6, -2, -10, -2, -4, -13, -13, -11, -8,
	-13, -3, -12, -9, 10, 4, 11, 9, -13, -8, -8, -9, -11, 7, -9, 12,
	7, 7, 12, 6, -4, -5, -3, 0, -13, 2, -12, -3, -9, 0, -7, 5,
	12, -6, 12, -1, -3, 6, -2, 12, -6, -13, -4, -8, 11, -13, 12, -8,
	4, 7, 5, 1, 5, -3, 10, -3, 3, -7, 6, 12, -8, -7, -6, -2,
	-2, 11, -1, -10, -13, 12, -8, 10, -7, 3, -5, -3, -4, 2, -3, 7,
	-10, -12, -6, 11, 5, -12, 6, -7, 5, -6, 7, -1, 1, 0, 4, -5,
	9, 11, 11, -13, 4, 7, 4, 12, 2, -1, 4, 4, -4, -12, -2, 7,
	-8, -5, -7, -10, 4, 11, 9, 12, 0, -8, 1, -13, -13, -2, -8, 2,
	-3, -2, -2, 3, -6, 9, -4, -9, 8, 12, 10, 7, 0, 9, 1, 3,
	7, -5, 11, -10, -13, -6, -11, 0, 10, 7, 12, 1, -6, -3, -6, 12,
	10, -9, 12, -4, -13, 8, -8, -12, -13, 0, -8, -4, 3, 3, 7, 8,
	5, 7, 10, -7, -1, 7, 1, -12, 3, -10, 5, 6, 2, -4, 3, -10,
	-13, 0, -13, 5, -13, -7, -12, 12, -13, 3, -11, 8, -7, 12, -4, 7,
	6, -10, 12, 8, -9, -1, -7, -6, -2, -5, 0, 12, -12, 5, -7, 5,
	3, -10, 8, -13, -7, -7, -4, 5, -3, -2, -1, -7, 2, 9, 5, -11,
	-11, -13, -5, -13, -1, 6, 0, -1, 5, -3, 5, 2, -4, -13, -4, 12,
	-9, -6, -9, 6, -12, -10, -8, -4, 10, 2, 12, -3, 7, 12, 12, 12,
	-7, -13, -6, 5, -4, 9, -3, 4, 7, -1, 12, 2, -7, 6, -5, 1,
	-13, 11, -12, 5, -3, 7, -2, -6, 7, -8, 12, -7, -13, -7, -11, -12,
	1, -3, 12, 12, 2, -6, 3, 0, -4, 3, -2, -13, -1, -13, 1, 9,
	7, 1, 8, -6, 1, -1, 3, 12, 9, 1, 12, 6, -1, -9, -1, 3,
	-13, -13, -10, 5, 7, 7, 10, 12, 12, -5, 12, 9, 6, 3, 7, 11,
	5, -13, 6, 10, 2, -12, 2, 3, 3, 8, 4, -6, 2, 6, 12, -13,
	9, -12, 10, 3, -8, 4, -7, 9, -11, 12, -4, -6, 1, 12, 2, -8,
	6, -9, 7, -4, 2, 3, 3, -2, 6, 3, 11, 0, 3, -3, 8, -8,
	7, 8, 9, 3, -11, -5, -6, -4, -10, 11, -5, 10, -5, -8, -3, 12,
	-10, 5, -9, 0, 8, -1, 12, -6, 4, -6, 6, -11, -10, 12, -8, 7,
	4, -2, 6, 7, -2, 0, -2, 12, -5, -8, -5, 2, 7, -6, 10, 12,
	-9, -13, -8, -8, -5, -13, -5, -2, 8, -8, 9, -13, -9, -11, -9, 0,
	1, -8, 1, -2, 7, -4, 9, 1, -2, 1, -1, -4, 11, -6, 12, -11,
	-12, -9, -6, 4, 3, 7, 7, 12, 5, 5, 10, 8, 0, -4, 2, 8,
	-9, 12, -5, -13, 0, 7, 2, 12, -1, 2, 1, 7, 5, 11, 7, -9,
	3, 5, 6, -8, -13, -4, -8, 9, -5, 9, -3, -3, -4, -7, -3, -12,
	6, 5, 8, 0, -7, 6, -6, 12, -13, 6, -5, -2, 1, -10, 3, 10,
	4, 1, 8, -4, -2, -2, 2, -13, 2, -12, 12, 12, -2, -13, 0, -6,
	4, 1, 9, 3, -6, -10, -3, -5, -3, -13, -1, 1, 7, 5, 12, -11,
	4, -2, 5, -7, -13, 9, -9, -5, 7, 1, 8, 6, 7, -8, 7, 6,
	-7, -4, -7, 1, -8, 11, -7, -8, -13, 6, -12, -8, 2, 4, 3, 9,
	10, -5, 12, 3, -6, -5, -6, 7, 8, -3, 9, -8, 2, -12, 2, 8,
	-11, -2, -10, 3, -12, -13, -7, -9, -11, 0, -10, -5, 5, -3, 11, 8,
	-2, -13, -1, 12, -1, -8, 0, 9, -13, -11, -12, -5, -10, -2, -10, 11,
	-3, 9, -2, -13, 2, -3, 3, 2, -9, -13, -4, 0, -4, 6, -3, -10,
	-4, 12, -2, -7, -6, -11, -4, 9, 6, -3, 6, 11, -13, 11, -5, 5,
	11, 11, 12, 6, 7, -5, 12, -2, -1, 12, 0, 7, -4, -8, -3, -2,
	-7, 1, -6, 7, -13, -12, -8, -13, -7, -2, -6, -8, -8, 5, -6, -9,
	-5, -1, -4, 5, -13, 7, -8, 10, 1, 5, 5, -13, 1, 0, 10, -13,
	9, 12, 10, -1, 5, -8, 10, -9, -1, 11, 1, -13, -9, -3, -6, 2,
	-1, -10, 1, 12, -13, 1, -8, -10, 8, -11, 10, -6, 2, -13, 3, -6,
	7, -13, 12, -9, -10, -10, -5, -7, -10, -8, -8, -13, 4, -6, 8, 5,
	3, 12, 8, -13, -4, 2, -3, -3, 5, -13, 10, -12, 4, -13, 5, -1,
	-9, 9, -4, 3, 0, 3, 3, -9, -12, 1, -6, 1, 3, 2, 4, -8,
	-10, -10, -10, 9, 8, -13, 12, 12, -8, -12, -6, -5, 2, 2, 3, 7,
	10, 6, 11, -8, 6, 8, 8, -12, -7, 10, -6, 5, -3, -9, -3, 9,
	-1, -13, -1, 5, -3, -7, -3, 4, -8, -2, -8, 3, 4, 2, 12, 12,
	2, -5, 3, 11, 6, -9, 11, -13, 3, -1, 7, 12, 11, -1, 12, 4,
	-3, 0, -3, 6, 4, -11, 4, 12, 2, -4, 2, 1, -10, -6, -8, 1,
	-13, 7, -11, 1, -13, 12, -11, -13, 6, 0, 11, -13, 0, -1, 1, 4,
	-13, 3, -9, -2, -9, 8, -6, -3, -13, -6, -8, -2, 5, -9, 8, 10,
	2, 7, 3, -9, -1, -6, -1, -1, 9, 5, 11, -2, 11, -3, 12, -8,
	3, 0, 3, 5, -1, 4, 0, 10, 3, -6, 4, 5, -13, 0, -10, 5,
	5, 8, 12, 11, 8, 9, 9, -6, 7, -4, 8, -12, -10, 4, -10, 9,
	7, 3, 12, 4, 9, -7, 10, -2, 7, 0, 12, -2, -1, -6, 0, -11,
};


struct PushConstants {
	uint level_image_index, level_keypoints_index, level_counts_index;
	uint keypoints_index, descriptors_index, counts_index;
	uint batch_size, channels, height, width;
	uint cap, edge, max_features, level;
	float scale;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// Per-level keypoint description: one thread per candidate row computes the
// 31x31 intensity-centroid orientation and the 256-bit steered rBRIEF
// descriptor on the level image, then claims a shared output slot. Row order
// within a level is atomic-claim order, not score order.
[kernel_name("orb_describe")]
[domain("image")]
[variant("nchw_or_chw")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint plane_count = push.batch_size * push.channels;
	uint plane = index / push.cap;
	uint entry = index % push.cap;
	if (plane >= plane_count) return;
	RWByteAddressBuffer level_counts = storage_buffers[push.level_counts_index];
	if (entry >= load_u32(level_counts, plane)) return;
	if (push.width <= 2u * push.edge || push.height <= 2u * push.edge) return;
	RWByteAddressBuffer level_keypoints = storage_buffers[push.level_keypoints_index];
	uint in_base = (plane * push.cap + entry) * 3;
	float cx = load_f32(level_keypoints, in_base);
	float cy = load_f32(level_keypoints, in_base + 1);
	float score = load_f32(level_keypoints, in_base + 2);
	int ix = int(round(cx));
	int iy = int(round(cy));
	if (ix < int(push.edge) || ix >= int(push.width - push.edge) ||
		iy < int(push.edge) || iy >= int(push.height - push.edge)) return;
	RWByteAddressBuffer level_image = storage_buffers[push.level_image_index];
	uint plane_base = plane * push.height * push.width;
	// Intensity centroid over a square 31x31 patch.
	float m10 = 0.0;
	float m01 = 0.0;
	for (int dy = -15; dy <= 15; dy++) {
		for (int dx = -15; dx <= 15; dx++) {
			float value = load_f32(level_image, plane_base + uint(iy + dy) * push.width + uint(ix + dx));
			m10 += float(dx) * value;
			m01 += float(dy) * value;
		}
	}
	float angle = atan2(m01, m10);
	float cos_a = cos(angle);
	float sin_a = sin(angle);
	uint descriptor[8];
	for (uint word = 0u; word < 8u; word++) descriptor[word] = 0u;
	for (uint pair = 0u; pair < 256u; pair++) {
		int x1 = orb_pattern[pair * 4];
		int y1 = orb_pattern[pair * 4 + 1];
		int x2 = orb_pattern[pair * 4 + 2];
		int y2 = orb_pattern[pair * 4 + 3];
		int rx1 = int(round(float(x1) * cos_a - float(y1) * sin_a));
		int ry1 = int(round(float(x1) * sin_a + float(y1) * cos_a));
		int rx2 = int(round(float(x2) * cos_a - float(y2) * sin_a));
		int ry2 = int(round(float(x2) * sin_a + float(y2) * cos_a));
		float first = load_f32(level_image, plane_base + uint(iy + ry1) * push.width + uint(ix + rx1));
		float second = load_f32(level_image, plane_base + uint(iy + ry2) * push.width + uint(ix + rx2));
		if (first < second) descriptor[pair >> 5] |= 1u << (pair & 31u);
	}
	uint slot;
	RWByteAddressBuffer counts = storage_buffers[push.counts_index];
	counts.InterlockedAdd(plane * 4, 1u, slot);
	if (slot >= push.max_features) return;
	RWByteAddressBuffer keypoints = storage_buffers[push.keypoints_index];
	uint out_base = (plane * push.max_features + slot) * 6;
	store_f32(keypoints, out_base, cx * push.scale);
	store_f32(keypoints, out_base + 1, cy * push.scale);
	store_f32(keypoints, out_base + 2, angle);
	store_f32(keypoints, out_base + 3, score);
	store_f32(keypoints, out_base + 4, float(push.level));
	store_f32(keypoints, out_base + 5, 31.0 * push.scale);
	RWByteAddressBuffer descriptors = storage_buffers[push.descriptors_index];
	uint desc_base = (plane * push.max_features + slot) * 8;
	for (uint word = 0u; word < 8u; word++)
		store_u32(descriptors, desc_base + word, descriptor[word]);
}
