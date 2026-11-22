// Mechanical adaptation of OA's ScatterRows; each output element is the input
// plus the packed source row when the row is selected by the row map.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint source_index;
	uint row_map_index;
	uint count_index;
	uint output_index;
	uint t;
	uint d;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("scatter_rows")]
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
	uint row = index / push.d;
	uint column = index % push.d;
	uint count = load_u32(storage_buffers[push.count_index], 0);
	uint low = 0;
	uint high = count;
	while (low < high) {
		uint mid = (low + high) >> 1;
		if (load_u32(storage_buffers[push.row_map_index], mid) < row) {
			low = mid + 1;
		} else {
			high = mid;
		}
	}
	float value = load_f32(storage_buffers[push.input_index], index);
	if (low < count && load_u32(storage_buffers[push.row_map_index], low) == row) {
		value += load_f32(storage_buffers[push.source_index], low * push.d + column);
	}
	store_f32(storage_buffers[push.output_index], index, value);
}
