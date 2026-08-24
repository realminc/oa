struct PushConstants {
	uint observation_idx, action_idx, next_observation_idx;
	uint reward_idx, terminated_idx, truncated_idx;
	uint out_observation_idx, out_action_idx, out_next_observation_idx;
	uint out_reward_idx, out_terminated_idx, out_truncated_idx, out_index_idx;
	uint size, batch, observation_elements, action_elements, seed_lo, seed_hi;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

uint hash32(uint value) {
	value ^= value >> 16u;
	value *= 0x7feb352du;
	value ^= value >> 15u;
	value *= 0x846ca68bu;
	return value ^ (value >> 16u);
}

uint sourceIndex(uint lane) {
	return hash32(pc.seed_lo ^ hash32(pc.seed_hi + lane * 0x9e3779b9u)) % pc.size;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	uint observationCount = pc.batch * pc.observation_elements;
	if (i < observationCount) {
		uint lane = i / pc.observation_elements;
		uint component = i - lane * pc.observation_elements;
		uint source = sourceIndex(lane) * pc.observation_elements + component;
		heap[pc.out_observation_idx].Store(i * 4u,
			heap[pc.observation_idx].Load(source * 4u));
		heap[pc.out_next_observation_idx].Store(i * 4u,
			heap[pc.next_observation_idx].Load(source * 4u));
	}
	uint actionCount = pc.batch * pc.action_elements;
	if (i < actionCount) {
		uint lane = i / pc.action_elements;
		uint component = i - lane * pc.action_elements;
		uint source = sourceIndex(lane) * pc.action_elements + component;
		heap[pc.out_action_idx].Store(i * 4u,
			heap[pc.action_idx].Load(source * 4u));
	}
	if (i >= pc.batch) return;
	uint source = sourceIndex(i);
	heap[pc.out_reward_idx].Store(i * 4u,
		heap[pc.reward_idx].Load(source * 4u));
	heap[pc.out_terminated_idx].Store<uint8_t>(i,
		heap[pc.terminated_idx].Load<uint8_t>(source));
	heap[pc.out_truncated_idx].Store<uint8_t>(i,
		heap[pc.truncated_idx].Load<uint8_t>(source));
	heap[pc.out_index_idx].Store(i * 4u, source);
}
