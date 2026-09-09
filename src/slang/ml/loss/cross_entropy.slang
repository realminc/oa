// Mean FP32 cross-entropy with rows distributed across one workgroup.

import storage;
import attributes;

struct PushConstants {
	uint logits_index;
	uint targets_index;
	uint output_index;
	uint rows;
	uint classes;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float partial_loss[256];

[kernel_name("cross_entropy")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	float local_loss = 0.0f;
	for (uint row = lane; row < push.rows; row += 256u) {
		uint target = storage_buffers[push.targets_index].Load(row * 4u);
		if (target >= push.classes) {
			local_loss = 0.0f / 0.0f;
			continue;
		}
		uint base = row * push.classes;
		float maximum = load_f32(storage_buffers[push.logits_index], base);
		for (uint column = 1; column < push.classes; ++column) {
			maximum = max(
				maximum,
				load_f32(storage_buffers[push.logits_index], base + column)
			);
		}
		float exponential_sum = 0.0f;
		for (uint column = 0; column < push.classes; ++column) {
			exponential_sum += exp(
				load_f32(storage_buffers[push.logits_index], base + column) - maximum
			);
		}
		local_loss += maximum + log(exponential_sum)
			- load_f32(storage_buffers[push.logits_index], base + target);
	}
	partial_loss[lane] = local_loss;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (lane < stride) {
			partial_loss[lane] += partial_loss[lane + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (lane == 0u) {
		store_f32(storage_buffers[push.output_index], 0u, partial_loss[0] / float(push.rows));
	}
}
