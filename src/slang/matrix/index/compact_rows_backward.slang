// Mechanical adaptation of OA's CompactRowsBwd; selected packed rows atomic-add
// into their source rows of the zero-initialized input gradient.

import storage;
import attributes;

struct PushConstants {
	uint gradient_out_index;
	uint row_map_index;
	uint count_index;
	uint gradient_in_index;
	uint t;
	uint d;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("compact_rows_backward")]
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
	uint source_row = load_u32(storage_buffers[push.row_map_index], packed_row);
	atomic_add_f32(
		storage_buffers[push.gradient_in_index],
		source_row * push.d + column,
		load_f32(storage_buffers[push.gradient_out_index], index));
}
