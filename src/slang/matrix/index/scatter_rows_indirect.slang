// Mechanical adaptation of OA's ScatterRowsIndirect; CompactRows authored the
// workgroup count so zero selected rows is a legal zero-work indirect dispatch.
// Only the packed selected rows are added into the copied input values.

import storage;
import attributes;

struct PushConstants {
	uint source_index;
	uint row_map_index;
	uint count_index;
	uint dispatch_args_index;
	uint output_index;
	uint t;
	uint d;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("scatter_rows_indirect")]
[domain("matrix")]
[variant("indirect")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	uint count = load_u32(storage_buffers[push.count_index], 0);
	if (index >= count * push.d) {
		return;
	}
	uint packed_row = index / push.d;
	uint column = index % push.d;
	uint target_row = load_u32(storage_buffers[push.row_map_index], packed_row);
	uint target = target_row * push.d + column;
	float value = load_f32(storage_buffers[push.output_index], target)
		+ load_f32(storage_buffers[push.source_index], index);
	store_f32(storage_buffers[push.output_index], target, value);
}
