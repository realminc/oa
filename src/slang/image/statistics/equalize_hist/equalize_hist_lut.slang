import attributes;
import storage;

struct PushConstants {
	uint histogram_index, lut_index, element_count;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("equalize_hist_lut")]
[domain("image")]
[variant("cdf_lut")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint cell = thread_id.x;
	if (cell >= push.element_count) return;
	uint base = (cell / 256u) * 256u;
	uint bucket = cell - base;
	uint cumulative = 0u, total = 0u, first = 0u;
	for (uint i = 0u; i < 256u; ++i) {
		uint count = load_u32(storage_buffers[push.histogram_index], base + i);
		total += count;
		if (first == i && count == 0u) ++first;
		if (i == bucket) cumulative = total;
	}
	uint cdf_min = first < 256u ? load_u32(storage_buffers[push.histogram_index], base + first) : 0u;
	float spread = max(float(total - cdf_min), 1.0);
	store_f32(storage_buffers[push.lut_index], cell,
		clamp((float(cumulative) - float(cdf_min)) / spread, 0.0, 1.0));
}
