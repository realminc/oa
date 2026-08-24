// Deterministic vectorized CartPole reset. Full reset reproduces the same
// initial states; ResetDone advances a per-lane episode index before sampling.

struct PushConstants {
	uint done_idx;
	uint state_idx;
	uint episode_steps_idx;
	uint episode_index_idx;
	uint environments;
	uint seed_low;
	uint seed_high;
	uint only_done;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

uint mix32(uint value) {
	value ^= value >> 16u;
	value *= 0x7feb352du;
	value ^= value >> 15u;
	value *= 0x846ca68bu;
	value ^= value >> 16u;
	return value;
}

float initialState(uint environment, uint episode, uint component) {
	uint value = mix32(pc.seed_low ^ mix32(pc.seed_high + 0x9e3779b9u));
	value = mix32(value ^ mix32(environment + 0x85ebca6bu));
	value = mix32(value ^ mix32(episode + 0xc2b2ae35u));
	value = mix32(value ^ mix32(component + 0x27d4eb2fu));
	// The high 24 bits map exactly to [0,1) in FP32.
	float unit = float(value >> 8u) * (1.0 / 16777216.0);
	return unit * 0.1 - 0.05;
}

void storeF32(RWByteAddressBuffer buffer, uint index, float value) {
	buffer.Store(index * 4u, asuint(value));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint environment = tid.x;
	if (environment >= pc.environments) return;
	if (pc.only_done != 0u
		&& heap[pc.done_idx].Load<uint8_t>(environment) == uint8_t(0)) {
		return;
	}

	uint episode = 0u;
	if (pc.only_done != 0u) {
		episode = heap[pc.episode_index_idx].Load(environment * 4u) + 1u;
	}
	heap[pc.episode_index_idx].Store(environment * 4u, episode);
	heap[pc.episode_steps_idx].Store(environment * 4u, 0u);
	heap[pc.done_idx].Store<uint8_t>(environment, uint8_t(0));

	uint stateBase = environment * 4u;
	for (uint component = 0u; component < 4u; ++component) {
		storeF32(heap[pc.state_idx], stateBase + component,
			initialState(environment, episode, component));
	}
}
