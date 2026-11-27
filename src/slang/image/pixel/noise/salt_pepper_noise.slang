import attributes;
import philox;
import storage;

struct PushConstants {
	uint input_index, output_index, element_count, seed_low, seed_high;
	float probability, salt_value, pepper_value;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("salt_pepper_noise")]
[domain("image")]
[variant("philox_explicit_seed")]
[dtype("f32")]
[layout("dense")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint base = thread_id.x * 4u;
	if (base >= push.element_count) return;
	uint4 bits = philox_rng(uint4(thread_id.x, 0u, 0u, 0u), uint2(push.seed_low, push.seed_high));
	float half_probability = push.probability * 0.5f;
	[unroll] for (uint lane = 0u; lane < 4u; ++lane) {
		uint index = base + lane;
		if (index < push.element_count) {
			float random = philox_to_float01(bits[lane]);
			float value = load_f32(storage_buffers[push.input_index], index);
			if (random < half_probability) value = push.pepper_value;
			else if (random >= 1.0f - half_probability) value = push.salt_value;
			store_f32(storage_buffers[push.output_index], index, value);
		}
	}
}
