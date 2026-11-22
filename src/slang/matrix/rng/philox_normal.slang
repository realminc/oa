import attributes;
import philox;
import storage;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint element_count;
	uint seed_low;
	uint seed_high;
	uint offset;
	float mean;
	float stddev;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("philox_normal")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint thread = group_id.x * 256u + group_thread_id.x;
	uint base = thread * 2u;
	if (base >= push.element_count) {
		return;
	}
	uint4 bits = philox_rng(
		uint4(thread + push.offset, 0u, 0u, 0u),
		uint2(push.seed_low, push.seed_high)
	);
	float first;
	float second;
	philox_to_normal(bits, first, second);
	store_f32(storage_buffers[push.output_index], base, first * push.stddev + push.mean);
	if (base + 1u < push.element_count) {
		store_f32(storage_buffers[push.output_index], base + 1u, second * push.stddev + push.mean);
	}
}
