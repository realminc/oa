// One workgroup reduces one gradient tensor to its sum of squares.

import storage;
import attributes;

struct PushConstants {
	uint params_index;
	uint partials_index;
	uint gradient_0_index;
	uint gradient_1_index;
	uint gradient_2_index;
	uint gradient_3_index;
	uint gradient_4_index;
	uint gradient_5_index;
	uint gradient_6_index;
	uint gradient_7_index;
	uint gradient_8_index;
	uint gradient_9_index;
	uint gradient_10_index;
	uint gradient_11_index;
	uint gradient_12_index;
	uint gradient_13_index;
	uint gradient_14_index;
	uint gradient_15_index;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial_sum[256];

uint gradient_storage_index(uint index) {
	switch (index) {
		case 0: return push.gradient_0_index;
		case 1: return push.gradient_1_index;
		case 2: return push.gradient_2_index;
		case 3: return push.gradient_3_index;
		case 4: return push.gradient_4_index;
		case 5: return push.gradient_5_index;
		case 6: return push.gradient_6_index;
		case 7: return push.gradient_7_index;
		case 8: return push.gradient_8_index;
		case 9: return push.gradient_9_index;
		case 10: return push.gradient_10_index;
		case 11: return push.gradient_11_index;
		case 12: return push.gradient_12_index;
		case 13: return push.gradient_13_index;
		case 14: return push.gradient_14_index;
		default: return push.gradient_15_index;
	}
}

[kernel_name("clip_grad_norm")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID, uint3 group_id : SV_GroupID) {
	uint gradient_count = load_u32(storage_buffers[push.params_index], 0);
	uint gradient = group_id.x;
	if (gradient >= gradient_count) {
		return;
	}

	uint local_index = group_thread_id.x;
	uint element_count = load_u32(storage_buffers[push.params_index], gradient + 1);
	uint gradient_index = gradient_storage_index(gradient);
	float sum = 0.0f;
	for (uint index = local_index; index < element_count; index += 256) {
		float value = load_f32(storage_buffers[gradient_index], index);
		sum += value * value;
	}

	partial_sum[local_index] = sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (local_index < stride) {
			partial_sum[local_index] += partial_sum[local_index + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (local_index == 0) {
		store_f32(storage_buffers[push.partials_index], gradient, partial_sum[0]);
	}
}
