// Donor: source/cpp/lib/oa/ml/shader/compute/nn/moe/moeCombineBwd.slang

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint packed_index;
	uint route_gate_index;
	uint inverse_index;
	uint packed_slot_index;
	uint packed_gradient_index;
	uint route_gate_gradient_index;
	uint tokens;
	uint routes_per_token;
	uint row_width;
	uint routes;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("moe_combine_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index < push.routes * push.row_width) {
		uint packed = index / push.row_width;
		uint column = index % push.row_width;
		uint slot = storage_buffers[push.packed_slot_index].Load(packed * 4);
		uint token = slot / push.routes_per_token;
		float value = load_f32(
			storage_buffers[push.output_gradient_index], token * push.row_width + column)
			* load_f32(storage_buffers[push.route_gate_index], slot);
		store_f32(storage_buffers[push.packed_gradient_index], index, value);
	}
	if (index < push.routes) {
		uint slot = index;
		uint token = slot / push.routes_per_token;
		uint packed = storage_buffers[push.inverse_index].Load(slot * 4);
		float sum = 0.0f;
		for (uint column = 0; column < push.row_width; ++column) {
			sum += load_f32(
				storage_buffers[push.output_gradient_index], token * push.row_width + column)
				* load_f32(storage_buffers[push.packed_index], packed * push.row_width + column);
		}
		store_f32(storage_buffers[push.route_gate_gradient_index], slot, sum);
	}
}
