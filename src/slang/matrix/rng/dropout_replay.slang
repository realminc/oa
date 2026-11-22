import attributes;
import dropout_rng;
import philox;
import storage;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint state_index;
	uint element_count;
	uint seed_low;
	uint seed_high;
	uint offset;
	float probability;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("dropout_replay")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint thread = group_id.x * 256u + group_thread_id.x;
	uint base = thread * 4u;
	if (base >= push.element_count) {
		return;
	}
	uint replay = storage_buffers[push.state_index].Load(0u);
	uint counters_per_replay = (push.element_count + 3u) / 4u;
	uint4 bits = philox_rng(
		uint4(thread + push.offset + replay * counters_per_replay, 0u, 0u, 0u),
		uint2(push.seed_low, push.seed_high)
	);
	[unroll] for (uint lane = 0u; lane < 4u; ++lane) {
		uint index = base + lane;
		if (index < push.element_count) {
			float input = load_f32(storage_buffers[push.input_index], index);
			store_f32(storage_buffers[push.output_index], index, apply_dropout(input, bits[lane], push.probability));
		}
	}
}
