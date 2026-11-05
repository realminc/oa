// Mechanical adaptation of OA's subgroup-independent full Max reduction.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_values[256];

[kernel_name("max")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	float best = -3.402823466e+38f;
	for (uint index = lane; index < push.element_count; index += 256) {
		best = max(best, load_f32(storage_buffers[push.input_index], index));
	}
	shared_values[lane] = best;
	GroupMemoryBarrierWithGroupSync();

	[unroll] for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[lane] = max(shared_values[lane], shared_values[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lane == 0) {
		store_f32(storage_buffers[push.output_index], 0, shared_values[0]);
	}
}
