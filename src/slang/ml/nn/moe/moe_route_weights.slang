// Donor: source/cpp/lib/oa/ml/shader/compute/nn/moe/moeRouteWeights.slang
// Adaptation: Rust naming, attributes, storage helpers, and bindless ABI spelling.

import storage;
import attributes;

struct PushConstants {
	uint probabilities_index;
	uint expert_indices_index;
	uint output_index;
	uint tokens;
	uint experts;
	uint routes_per_token;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("moe_route_weights")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint token = dispatch_thread_id.x;
	if (token >= push.tokens) {
		return;
	}

	float denominator = 0.0;
	for (uint route_slot = 0; route_slot < push.routes_per_token; ++route_slot) {
		uint route = token * push.routes_per_token + route_slot;
		int expert = asint(storage_buffers[push.expert_indices_index].Load(route * 4));
		if (expert >= 0 && uint(expert) < push.experts) {
			denominator += load_f32(
				storage_buffers[push.probabilities_index],
				token * push.experts + uint(expert));
		}
	}

	float inverse_denominator = denominator > 0.0 ? 1.0 / denominator : 0.0;
	for (uint route_slot = 0; route_slot < push.routes_per_token; ++route_slot) {
		uint route = token * push.routes_per_token + route_slot;
		int expert = asint(storage_buffers[push.expert_indices_index].Load(route * 4));
		float value = 0.0;
		if (expert >= 0 && uint(expert) < push.experts) {
			value = load_f32(
				storage_buffers[push.probabilities_index],
				token * push.experts + uint(expert)) * inverse_denominator;
		}
		store_f32(storage_buffers[push.output_index], route, value);
	}
}
