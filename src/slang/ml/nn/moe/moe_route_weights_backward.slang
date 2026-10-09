// Donor: source/cpp/lib/oa/ml/shader/compute/nn/moe/moeRouteWeightsBwd.slang
// Adaptation: Rust naming, attributes, storage helpers, and bindless ABI spelling.

import storage;
import attributes;

struct PushConstants {
	uint output_gradient_index;
	uint probabilities_index;
	uint expert_indices_index;
	uint route_weights_index;
	uint probability_gradient_index;
	uint tokens;
	uint experts;
	uint routes_per_token;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("moe_route_weights_backward")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.tokens * push.experts) {
		return;
	}
	uint token = index / push.experts;
	uint expert_index = index % push.experts;

	float denominator = 0.0;
	float dot = 0.0;
	int selected_slot = -1;
	for (uint route_slot = 0; route_slot < push.routes_per_token; ++route_slot) {
		uint route = token * push.routes_per_token + route_slot;
		int expert = asint(storage_buffers[push.expert_indices_index].Load(route * 4));
		if (expert >= 0 && uint(expert) < push.experts) {
			denominator += load_f32(
				storage_buffers[push.probabilities_index],
				token * push.experts + uint(expert));
			dot += load_f32(storage_buffers[push.output_gradient_index], route)
				* load_f32(storage_buffers[push.route_weights_index], route);
			if (uint(expert) == expert_index) {
				selected_slot = int(route_slot);
			}
		}
	}

	float gradient = 0.0;
	if (selected_slot >= 0 && denominator > 0.0) {
		uint route = token * push.routes_per_token + uint(selected_slot);
		gradient =
			(load_f32(storage_buffers[push.output_gradient_index], route) - dot)
			/ denominator;
	}
	store_f32(storage_buffers[push.probability_gradient_index], index, gradient);
}
