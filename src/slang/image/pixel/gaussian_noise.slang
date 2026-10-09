import attributes;
import philox;
import storage;

struct PushConstants {
	uint input_index, output_index, element_count, seed_low, seed_high;
	float mean, stddev;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("gaussian_noise")]
[domain("image")]
[variant("philox_explicit_seed")]
[dtype("f32")]
[layout("dense")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint base = thread_id.x * 2u;
	if (base >= push.element_count) return;
	uint4 bits = philox_rng(uint4(thread_id.x, 0u, 0u, 0u), uint2(push.seed_low, push.seed_high));
	float first, second;
	philox_to_normal(bits, first, second);
	store_f32(storage_buffers[push.output_index], base,
		load_f32(storage_buffers[push.input_index], base) + first * push.stddev + push.mean);
	if (base + 1u < push.element_count) {
		store_f32(storage_buffers[push.output_index], base + 1u,
			load_f32(storage_buffers[push.input_index], base + 1u) + second * push.stddev + push.mean);
	}
}
