// Mechanical adaptation of OA's axis-aware stable Softmax kernel.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float shared_values[256];

[kernel_name("softmax")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint group = group_id.x;
	uint lane = group_thread_id.x;
	if (group >= push.outer_size * push.inner_size) {
		return;
	}

	uint outer = group / push.inner_size;
	uint inner = group % push.inner_size;
	uint base = outer * push.dim_size * push.inner_size + inner;

	float local_maximum = -3.402823466e+38f;
	for (uint index = lane; index < push.dim_size; index += 256) {
		local_maximum = max(
			local_maximum,
			load_f32(storage_buffers[push.input_index], base + index * push.inner_size)
		);
	}
	shared_values[lane] = local_maximum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[lane] = max(shared_values[lane], shared_values[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float row_maximum = shared_values[0];
	GroupMemoryBarrierWithGroupSync();

	float local_sum = 0.0f;
	for (uint index = lane; index < push.dim_size; index += 256) {
		local_sum += exp(
			load_f32(storage_buffers[push.input_index], base + index * push.inner_size)
				- row_maximum
		);
	}
	shared_values[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (lane < stride) {
			shared_values[lane] += shared_values[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float inverse_sum = 1.0f / shared_values[0];
	GroupMemoryBarrierWithGroupSync();

	for (uint index = lane; index < push.dim_size; index += 256) {
		uint element = base + index * push.inner_size;
		store_f32(
			storage_buffers[push.output_index],
			element,
			exp(load_f32(storage_buffers[push.input_index], element) - row_maximum)
				* inverse_sum
		);
	}
}
