// Mechanically adapted from OA
// source/cpp/lib/oa/ml/shader/compute/nn/gatherBwd.slang.

import storage;
import attributes;

struct PushConstants {
	uint indices_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint num_indices;
	uint num_rows;
	uint row_size;
	uint index_dtype;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("gather_backward")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint row = group_id.x;
	if (row >= push.num_rows) {
		return;
	}
	for (uint column = group_thread_id.x; column < push.row_size; column += 256u) {
		float gradient = 0.0f;
		for (uint position = 0; position < push.num_indices; ++position) {
			uint index = push.index_dtype == 0
				? uint(storage_buffers[push.indices_index].Load<uint8_t>(position))
				: storage_buffers[push.indices_index].Load(position * 4u);
			if (index == row) {
				gradient += load_f32(
					storage_buffers[push.output_gradient_index],
					position * push.row_size + column
				);
			}
		}
		store_f32(
			storage_buffers[push.input_gradient_index],
			row * push.row_size + column,
			gradient
		);
	}
}
