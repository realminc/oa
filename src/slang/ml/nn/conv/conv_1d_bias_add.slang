// Mechanical adaptation of OA's in-place last-axis bias broadcast.
// Donor: source/cpp/lib/oa/ml/shader/compute/ops/biasAdd.slang

import storage;
import attributes;

struct PushConstants {
	uint values_index;
	uint bias_index;
	uint rows;
	uint columns;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_1d_bias_add")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	if (index >= push.rows * push.columns) {
		return;
	}
	uint column = index % push.columns;
	store_f32(
		storage_buffers[push.values_index],
		index,
		load_f32(storage_buffers[push.values_index], index)
			+ load_f32(storage_buffers[push.bias_index], column)
	);
}
