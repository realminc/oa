// Donor: source/cpp/lib/oa/ml/shader/compute/nn/moe/moeCombine.slang

import storage;
import attributes;

struct PushConstants {
	uint packed_index;
	uint route_gate_index;
	uint inverse_index;
	uint output_index;
	uint tokens;
	uint routes_per_token;
	uint row_width;
	uint routes;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("moe_combine")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.tokens * push.row_width) return;
	uint token = index / push.row_width;
	uint column = index % push.row_width;
	float sum = 0.0f;
	for (uint route_slot = 0; route_slot < push.routes_per_token; ++route_slot) {
		uint slot = token * push.routes_per_token + route_slot;
		uint packed = storage_buffers[push.inverse_index].Load(slot * 4);
		if (packed < push.routes) {
			sum += load_f32(
				storage_buffers[push.packed_index], packed * push.row_width + column)
				* load_f32(storage_buffers[push.route_gate_index], slot);
		}
	}
	store_f32(storage_buffers[push.output_index], index, sum);
}
