// Muon-owned Frobenius normalization. One workgroup owns the complete matrix.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint norm_index;
	uint rows;
	uint columns;
	float epsilon;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];
groupshared float partial_sums[256];

[kernel_name("muon_normalize")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint element_count = push.rows * push.columns;
	uint local_index = group_thread_id.x;
	float sum = 0.0f;
	for (uint index = local_index; index < element_count; index += 256) {
		float value = load_f32(storage_buffers[push.input_index], index);
		sum += value * value;
	}
	partial_sums[local_index] = sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint stride = 128; stride > 0; stride >>= 1) {
		if (local_index < stride) {
			partial_sums[local_index] += partial_sums[local_index + stride];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	float norm = sqrt(partial_sums[0] + push.epsilon);
	if (local_index == 0) {
		store_f32(storage_buffers[push.norm_index], 0, norm);
	}
	for (uint index = local_index; index < element_count; index += 256) {
		float value = load_f32(storage_buffers[push.input_index], index);
		store_f32(storage_buffers[push.output_index], index, value / norm);
	}
}
