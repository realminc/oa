// Mechanical adaptation of OA's ScatterRowsBwd; each packed row gathers its
// gradient from the mapped output row into the zero-initialized source gradient.

import storage;
import attributes;

struct PushConstants {
	uint gradient_out_index;
	uint row_map_index;
	uint count_index;
	uint gradient_source_index;
	uint t;
	uint d;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("scatter_rows_backward_source")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.t * push.d) {
		return;
	}
	uint packed_row = index / push.d;
	uint count = load_u32(storage_buffers[push.count_index], 0);
	if (packed_row >= count) {
		return;
	}
	uint column = index % push.d;
	uint target_row = load_u32(storage_buffers[push.row_map_index], packed_row);
	store_f32(
		storage_buffers[push.gradient_source_index],
		index,
		load_f32(storage_buffers[push.gradient_out_index], target_row * push.d + column));
}
