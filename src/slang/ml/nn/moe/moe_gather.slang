// Donor: source/cpp/lib/oa/ml/shader/compute/ops/gather.slang
// FP32 specialization for stable expert-major MoE packing.

import storage;
import attributes;

struct PushConstants {
	uint input_index;
	uint packed_token_index;
	uint output_index;
	uint routes;
	uint row_width;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("moe_gather")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.routes * push.row_width) return;
	uint packed_row = index / push.row_width;
	uint column = index % push.row_width;
	uint token = storage_buffers[push.packed_token_index].Load(packed_row * 4);
	store_f32(
		storage_buffers[push.output_index],
		index,
		load_f32(storage_buffers[push.input_index], token * push.row_width + column));
}
