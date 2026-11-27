import attributes;
import storage;

struct PushConstants {
	uint histogram_index, lut_index, tile_count;
	float clip_limit, tile_area;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("clahe_lut")]
[domain("image")]
[variant("row_reduce")]
[dtype("u32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint tile = thread_id.x;
	if (tile >= push.tile_count) return;
	RWByteAddressBuffer histogram = storage_buffers[push.histogram_index];
	RWByteAddressBuffer lut = storage_buffers[push.lut_index];
	uint base = tile * 256u;
	float clip = max(push.clip_limit * push.tile_area / 256.0, 1.0);
	float excess = 0.0;
	for (uint bin = 0u; bin < 256u; ++bin) {
		float count = float(load_u32(histogram, base + bin));
		excess += max(count - clip, 0.0);
	}
	float spread = excess / 256.0;
	float sum = 0.0;
	for (uint bin = 0u; bin < 256u; ++bin) {
		float count = min(float(load_u32(histogram, base + bin)), clip) + spread;
		sum += count;
		store_f32(lut, base + bin, clamp(sum / push.tile_area, 0.0, 1.0));
	}
}
