// Mechanical adaptation of OA's subgroup-independent full Sum reduction.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_values[2][256];

[kernel_name("sum")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	float accumulator = 0.0f;
	for (uint index = lane; index < push.element_count; index += 256) {
		accumulator += load_f32(storage_buffers[push.input_index], index);
	}
	uint read_bank = 0;
	uint write_bank = 1;
	shared_values[read_bank][lane] = accumulator;
	GroupMemoryBarrierWithGroupSync();

	[unroll] for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[write_bank][lane] =
				shared_values[read_bank][lane] + shared_values[read_bank][lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
		uint previous_read_bank = read_bank;
		read_bank = write_bank;
		write_bank = previous_read_bank;
	}

	if (lane == 0) {
		store_f32(storage_buffers[push.output_index], 0, shared_values[read_bank][0]);
	}
}
