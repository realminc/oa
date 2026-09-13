// Mechanically adapted from OA
// source/cpp/lib/oa/ml/shader/compute/ops/gather.slang.
// OARS adds an explicit table-row bound so invalid indices cannot read outside storage.

import storage;
import attributes;

struct PushConstants {
	uint table_index;
	uint indices_index;
	uint output_index;
	uint num_indices;
	uint num_rows;
	uint row_size;
	uint index_dtype;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("gather")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint output_index = dispatch_thread_id.x;
	uint output_count = push.num_indices * push.row_size;
	if (output_index >= output_count) {
		return;
	}
	uint index_position = output_index / push.row_size;
	uint column = output_index % push.row_size;
	uint row = push.index_dtype == 0
		? uint(storage_buffers[push.indices_index].Load<uint8_t>(index_position))
		: storage_buffers[push.indices_index].Load(index_position * 4u);
	float value = row < push.num_rows
		? load_f32(storage_buffers[push.table_index], row * push.row_size + column)
		: asfloat(0x7fc00000u);
	store_f32(storage_buffers[push.output_index], output_index, value);
}
