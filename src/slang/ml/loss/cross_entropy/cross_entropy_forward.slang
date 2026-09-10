// Per-row cross-entropy adapted from OA crossEntropy.slang.

import storage;
import attributes;

struct PushConstants {
	uint logits_index;
	uint targets_index;
	uint output_index;
	uint rows;
	uint classes;
	uint target_dtype;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[256];
static const uint PORTABLE_DISPATCH_WIDTH = 65535u;

[kernel_name("cross_entropy")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x + group_id.y * PORTABLE_DISPATCH_WIDTH;
	if (row >= push.rows) {
		return;
	}
	uint lane = group_thread_id.x;
	uint base = row * push.classes;

	float local_maximum = -3.402823466e+38f;
	for (uint column = lane; column < push.classes; column += 256u) {
		local_maximum = max(local_maximum, load_f32(storage_buffers[push.logits_index], base + column));
	}
	partial[lane] = local_maximum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lane] = max(partial[lane], partial[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float row_maximum = partial[0];
	GroupMemoryBarrierWithGroupSync();

	float local_sum = 0.0f;
	for (uint column = lane; column < push.classes; column += 256u) {
		local_sum += exp(load_f32(storage_buffers[push.logits_index], base + column) - row_maximum);
	}
	partial[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lane] += partial[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (lane == 0u) {
		uint target = storage_buffers[push.targets_index].Load(row * 4u);
		float loss = 0.0f / 0.0f;
		if (push.target_dtype == 1u && target < push.classes) {
			loss = log(partial[0]) + row_maximum
				- load_f32(storage_buffers[push.logits_index], base + target);
		}
		store_f32(storage_buffers[push.output_index], row, loss);
	}
}
