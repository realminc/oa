// Mechanical adaptation of OA's ScatterRowsBwdIndirect; CompactRows authored
// the workgroup count so zero selected rows is a legal zero-work indirect
// dispatch. Only the packed source-gradient rows gather from the mapped
// output-gradient rows.

import storage;
import attributes;

struct PushConstants {
	uint gradient_out_index;
	uint row_map_index;
	uint count_index;
	uint dispatch_args_index;
	uint gradient_source_index;
	uint t;
	uint d;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("scatter_rows_backward_source_indirect")]
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
	store_f32(
		storage_buffers[push.gradient_source_index],
		index,
		load_f32(storage_buffers[push.gradient_out_index], target_row * push.d + column));
}
