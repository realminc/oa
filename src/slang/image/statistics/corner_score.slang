import attributes;
import storage;

struct PushConstants {
	uint products_index, output_index;
	uint batch_size, channels, height, width;
	uint block_size;
	float k;
	uint score_type;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

int reflect101(int coordinate, int extent) {
	if (coordinate < 0) return -coordinate;
	if (coordinate >= extent) return 2 * extent - 2 - coordinate;
	return coordinate;
}

// Corner response from windowed second moments. Harris: det - k*trace^2.
// MinEigen: smaller eigenvalue of the moment matrix (Shi-Tomasi).
[kernel_name("corner_score")]
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
	RWByteAddressBuffer products = storage_buffers[push.products_index];
	int half = int(push.block_size) / 2;
	float sxx = 0.0, syy = 0.0, sxy = 0.0;
	int ix = int(x), iy = int(y);
	for (int dy = -half; dy <= half; dy++) {
		for (int dx = -half; dx <= half; dx++) {
			uint sx = uint(reflect101(ix + dx, int(push.width)));
			uint sy = uint(reflect101(iy + dy, int(push.height)));
			uint base = (plane * push.height * push.width + sy * push.width + sx) * 3;
			sxx += load_f32(products, base);
			syy += load_f32(products, base + 1);
			sxy += load_f32(products, base + 2);
		}
	}
	float score;
	if (push.score_type == 0u) {
		float det = sxx * syy - sxy * sxy;
		float trace = sxx + syy;
		score = det - push.k * trace * trace;
	} else {
		float diff = sxx - syy;
		float root = sqrt(max(diff * diff + 4.0 * sxy * sxy, 0.0));
		score = 0.5 * (sxx + syy - root);
	}
	uint index = plane * push.height * push.width + y * push.width + x;
	store_f32(storage_buffers[push.output_index], index, score);
}
