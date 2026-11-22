// Fused scalar mean mechanically adapted from OA smoothL1Mean.slang.

import storage;
import attributes;

struct PushConstants {
	uint prediction_index;
	uint target_index;
	uint output_index;
	uint element_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[256];

float smooth_l1(float difference) {
	float magnitude = abs(difference);
	if (magnitude < 1.0f) {
		return 0.5f * difference * difference;
	}
	return magnitude - 0.5f;
}

float reduce_sum(float value, uint lane) {
	partial[lane] = value;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lane] += partial[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	return partial[0];
}

[kernel_name("smooth_l1_mean")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint lane = dispatch_thread_id.x;
	float local_sum = 0.0f;
	for (uint index = lane; index < push.element_count; index += 256u) {
		float prediction = load_f32(storage_buffers[push.prediction_index], index);
		float target = load_f32(storage_buffers[push.target_index], index);
		local_sum += smooth_l1(prediction - target);
	}

	float total = reduce_sum(local_sum, lane);
	if (lane == 0u) {
		store_f32(
			storage_buffers[push.output_index],
			0u,
			total / float(push.element_count)
		);
	}
}
