// Mechanical adaptation of OA SDK rlCartPoleReset.slang.
// Full reset reproduces the configured seed; completed-lane reset advances
// that lane's episode index before sampling its next deterministic state.

import storage;
import attributes;

struct PushConstants {
	uint done_index;
	uint state_index;
	uint episode_steps_index;
	uint episode_index_index;
	uint environments;
	uint seed_low;
	uint seed_high;
	uint only_done;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

uint mix32(uint value) {
	value ^= value >> 16u;
	value *= 0x7feb352du;
	value ^= value >> 15u;
	value *= 0x846ca68bu;
	value ^= value >> 16u;
	return value;
}

float initial_state(uint environment, uint episode, uint component) {
	uint value = mix32(push.seed_low ^ mix32(push.seed_high + 0x9e3779b9u));
	value = mix32(value ^ mix32(environment + 0x85ebca6bu));
	value = mix32(value ^ mix32(episode + 0xc2b2ae35u));
	value = mix32(value ^ mix32(component + 0x27d4eb2fu));
	float unit = float(value >> 8u) * (1.0 / 16777216.0);
	return unit * 0.1 - 0.05;
}

[kernel_name("cart_pole_reset")]
[domain("ml")]
[variant("vectorized")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint environment = dispatch_thread_id.x;
	if (environment >= push.environments) {
		return;
	}
	if (push.only_done != 0u
		&& load_u8(storage_buffers[push.done_index], environment) == 0u) {
		return;
	}

	uint episode = 0u;
	if (push.only_done != 0u) {
		episode = load_u32(storage_buffers[push.episode_index_index], environment) + 1u;
	}
	store_u32(storage_buffers[push.episode_index_index], environment, episode);
	store_u32(storage_buffers[push.episode_steps_index], environment, 0u);
	store_u8(storage_buffers[push.done_index], environment, 0u);

	uint state_base = environment * 4u;
	for (uint component = 0u; component < 4u; ++component) {
		store_f32(
			storage_buffers[push.state_index],
			state_base + component,
			initial_state(environment, episode, component));
	}
}
