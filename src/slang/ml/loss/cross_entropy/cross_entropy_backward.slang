// Row-cooperative FP32 gradient of mean cross-entropy.

import storage;
import attributes;

struct PushConstants {
	uint logits_index;
	uint targets_index;
	uint logits_gradient_index;
	uint rows;
	uint classes;
	uint target_dtype;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial[256];
groupshared float row_maximum;
static const uint PORTABLE_DISPATCH_WIDTH = 65535u;

[kernel_name("cross_entropy_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x + group_id.y * PORTABLE_DISPATCH_WIDTH;
	uint lane = group_thread_id.x;
	if (row >= push.rows) {
		return;
	}
	uint base = row * push.classes;
	uint target = storage_buffers[push.targets_index].Load(row * 4u);

	float local_maximum = -3.402823466e+38f;
	for (uint column = lane; column < push.classes; column += 256u) {
		local_maximum = max(
			local_maximum,
			load_f32(storage_buffers[push.logits_index], base + column)
		);
	}
	partial[lane] = local_maximum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lane] = max(partial[lane], partial[lane + stride]);
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (lane == 0u) {
		row_maximum = partial[0];
	}
	GroupMemoryBarrierWithGroupSync();

	float local_sum = 0.0f;
	for (uint column = lane; column < push.classes; column += 256u) {
		local_sum += exp(
			load_f32(storage_buffers[push.logits_index], base + column) - row_maximum
		);
	}
	partial[lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial[lane] += partial[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	float inverse_sum = 1.0f / partial[0];
	float inverse_rows = 1.0f / float(push.rows);
	for (uint column = lane; column < push.classes; column += 256u) {
		uint index = base + column;
		float gradient = 0.0f / 0.0f;
		if (push.target_dtype == 1u && target < push.classes) {
			gradient = (exp(
				load_f32(storage_buffers[push.logits_index], index) - row_maximum
			) * inverse_sum - (column == target ? 1.0f : 0.0f)) * inverse_rows;
		}
		store_f32(storage_buffers[push.logits_gradient_index], index, gradient);
	}
}
