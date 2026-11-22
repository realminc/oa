// FP32 global sum adapted from OA flat/sum.slang.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[2][256];

[kernel_name("cross_entropy_sum")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	float accumulator = 0.0f;
	for (uint index = lane; index < push.element_count; index += 256u) {
		accumulator += load_f32(storage_buffers[push.input_index], index);
	}
	uint read_bank = 0u;
	uint write_bank = 1u;
	partial[read_bank][lane] = accumulator;
	GroupMemoryBarrierWithGroupSync();
	[[unroll]] for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[write_bank][lane] = partial[read_bank][lane] + partial[read_bank][lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
		uint previous_read_bank = read_bank;
		read_bank = write_bank;
		write_bank = previous_read_bank;
	}
	if (lane == 0u) {
		store_f32(storage_buffers[push.output_index], 0u, partial[read_bank][0]);
	}
}
