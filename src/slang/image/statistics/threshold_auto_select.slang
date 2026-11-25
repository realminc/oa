import attributes;
import storage;

struct PushConstants {
	uint histogram_index, thresholds_index, plane_count, method;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

uint otsu_bin(uint base, float total) {
	float weight_back = 0.0;
	float sum_back = 0.0;
	float sum_all = 0.0;
	for (uint i = 0; i < 256u; ++i) {
		sum_all += float(i) * float(load_u32(storage_buffers[push.histogram_index], base + i));
	}
	float best = -1.0;
	uint best_bin = 0;
	for (uint t = 0; t < 256u; ++t) {
		float count = float(load_u32(storage_buffers[push.histogram_index], base + t));
		weight_back += count;
		sum_back += float(t) * count;
		float weight_fore = total - weight_back;
		if (weight_back <= 0.0 || weight_fore <= 0.0) continue;
		float mean_back = sum_back / weight_back;
		float mean_fore = (sum_all - sum_back) / weight_fore;
		float delta = mean_back - mean_fore;
		float variance = weight_back * weight_fore * delta * delta;
		if (variance > best) {
			best = variance;
			best_bin = t;
		}
	}
	return best_bin;
}

// OpenCV THRESH_TRIANGLE: line from the first nonzero bound to the
// histogram peak, maximizing a*i + b*h[i] where a = peak count and
// b = left_bound - max_ind. Mirrored when the peak sits closer to
// the left bound so the long tail always extends rightward.
uint triangle_bin(uint base) {
	float values[256];
	for (uint i = 0; i < 256u; ++i) {
		values[i] = float(load_u32(storage_buffers[push.histogram_index], base + i));
	}
	uint left = 0;
	while (left < 256u && values[left] <= 0.0) left += 1u;
	if (left >= 256u) return 0;
	if (left > 0u) left -= 1u;
	uint right = 255;
	while (right > 0u && values[right] <= 0.0) right -= 1u;
	if (right < 255u) right += 1u;
	uint max_ind = 0;
	float max_val = 0.0;
	for (uint i = 0; i < 256u; ++i) {
		if (values[i] > max_val) {
			max_val = values[i];
			max_ind = i;
		}
	}
	bool flipped = int(max_ind) - int(left) < int(right) - int(max_ind);
	uint segment_left = flipped ? 255u - right : left;
	uint segment_peak = flipped ? 255u - max_ind : max_ind;
	float a = max_val;
	float b = float(int(segment_left) - int(segment_peak));
	float dist = 0.0;
	uint thresh = segment_left;
	for (uint i = segment_left + 1u; i <= segment_peak; ++i) {
		float count = flipped ? values[255u - i] : values[i];
		float distance = a * float(i) + b * count;
		if (distance > dist) {
			dist = distance;
			thresh = i;
		}
	}
	if (thresh > 0u) thresh -= 1u;
	if (flipped) thresh = 255u - thresh;
	return thresh;
}

[kernel_name("threshold_auto_select")]
[domain("image")]
[variant("row_reduce")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint plane = thread_id.x;
	if (plane >= push.plane_count) return;
	uint base = plane * 256u;
	float total = 0.0;
	for (uint i = 0; i < 256u; ++i) {
		total += float(load_u32(storage_buffers[push.histogram_index], base + i));
	}
	uint bin = 0;
	if (total > 0.0) {
		bin = push.method == 0u ? otsu_bin(base, total) : triangle_bin(base);
	}
	store_f32(storage_buffers[push.thresholds_index], plane, float(bin));
}
