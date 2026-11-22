// Donor: source/cpp/lib/oa/ml/shader/compute/nn/moe/moeRoutingBiasUpdate.slang
// Device-resident auxiliary-loss-free expert routing-bias update.

import attributes;
import storage;

struct PushConstants {
	uint selection_mask_index;
	uint bias_index;
	uint tokens;
	uint experts;
	uint experts_per_token;
	float gamma;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("moe_routing_bias_update")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint expert = dispatch_thread_id.x;
	if (expert >= push.experts) {
		return;
	}

	float selected = 0.0f;
	for (uint token = 0; token < push.tokens; ++token) {
		selected += load_f32(
			storage_buffers[push.selection_mask_index],
			token * push.experts + expert);
	}
	float target_count = float(push.tokens * push.experts_per_token) / float(push.experts);
	float step = selected > target_count ? 1.0f : (selected < target_count ? -1.0f : 0.0f);
	float current = load_f32(storage_buffers[push.bias_index], expert);
	store_f32(storage_buffers[push.bias_index], expert, current - push.gamma * step);
}
