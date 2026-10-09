// Donor: source/cpp/lib/oa/core/shader/compute/flat/topKMask.slang

import storage;
import attributes;

struct PushConstants {
	uint indices_index;
	uint mask_index;
	uint tokens;
	uint experts;
	uint routes_per_token;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("top_k_mask")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint linear = dispatch_thread_id.x;
	uint count = push.tokens * push.experts;
	if (linear >= count) return;
	uint token = linear / push.experts;
	uint expert = linear - token * push.experts;
	float selected = 0.0f;
	for (uint route_slot = 0; route_slot < push.routes_per_token; ++route_slot) {
		uint index = storage_buffers[push.indices_index].Load(
			(token * push.routes_per_token + route_slot) * 4);
		if (index == expert) { selected = 1.0f; break; }
	}
	store_f32(storage_buffers[push.mask_index], linear, selected);
}
