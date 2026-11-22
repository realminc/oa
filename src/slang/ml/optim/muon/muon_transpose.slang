// Muon-owned dense rank-two FP32 transpose used only inside lowering.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint output_index;
	uint rows;
	uint columns;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("muon_transpose")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	uint element_count = push.rows * push.columns;
	if (index >= element_count) {
		return;
	}
	uint row = index / push.columns;
	uint column = index - row * push.columns;
	store_f32(storage_buffers[push.output_index], column * push.rows + row,
		load_f32(storage_buffers[push.input_index], index));
}
