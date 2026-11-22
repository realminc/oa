// Mechanical adaptation of OA's CompactRows; one workgroup performs a chunked
// prefix scan, and count plus dispatch args stay device-resident.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint mask_index;
	uint values_index;
	uint row_map_index;
	uint count_index;
	uint dispatch_args_index;
	uint t;
	uint d;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared uint scan[256];
groupshared uint base_offset;

[kernel_name("compact_rows")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_thread_id : SV_GroupThreadID) {
	uint lane = group_thread_id.x;
	if (lane == 0) {
		base_offset = 0;
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint base = 0; base < push.t; base += 256) {
		uint row = base + lane;
		uint selected =
			row < push.t && load_f32(storage_buffers[push.mask_index], row) != 0.0f ? 1 : 0;
		scan[lane] = selected;
		GroupMemoryBarrierWithGroupSync();
		for (uint offset = 1; offset < 256; offset <<= 1) {
			uint addend = lane >= offset ? scan[lane - offset] : 0;
			GroupMemoryBarrierWithGroupSync();
			scan[lane] += addend;
			GroupMemoryBarrierWithGroupSync();
		}
		uint slot = base_offset + scan[lane] - selected;
		if (selected != 0) {
			store_u32(storage_buffers[push.row_map_index], slot, row);
			for (uint column = 0; column < push.d; ++column) {
				store_f32(
					storage_buffers[push.values_index],
					slot * push.d + column,
					load_f32(storage_buffers[push.input_index], row * push.d + column));
			}
		}
		GroupMemoryBarrierWithGroupSync();
		if (lane == 0) {
			base_offset += scan[255];
		}
		GroupMemoryBarrierWithGroupSync();
	}
	if (lane == 0) {
		store_u32(storage_buffers[push.count_index], 0, base_offset);
		store_u32(storage_buffers[push.dispatch_args_index], 0, (base_offset * push.d + 255) / 256);
		store_u32(storage_buffers[push.dispatch_args_index], 1, 1);
		store_u32(storage_buffers[push.dispatch_args_index], 2, 1);
	}
}
