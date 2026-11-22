// Mechanical adaptation of OA's flat Argmax; the output is a single u32 index.

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
groupshared uint shared_indices[256];

[kernel_name("argmax")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	float best_value = -3.402823466e+38f;
	uint best_index = 0;
	for (uint index = lane; index < push.element_count; index += 256) {
		float value = load_f32(storage_buffers[push.input_index], index);
		if (value > best_value) {
			best_value = value;
			best_index = index;
		}
	}
	shared_values[lane] = best_value;
	shared_indices[lane] = best_index;
	GroupMemoryBarrierWithGroupSync();

	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			float other_value = shared_values[lane + stride];
			uint other_index = shared_indices[lane + stride];
			if (other_value > shared_values[lane]
				|| (other_value == shared_values[lane] && other_index < shared_indices[lane])) {
				shared_values[lane] = other_value;
				shared_indices[lane] = other_index;
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lane == 0) {
		storage_buffers[push.output_index].Store(0, shared_indices[0]);
	}
}
