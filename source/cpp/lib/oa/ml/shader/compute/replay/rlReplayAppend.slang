struct PushConstants {
	uint observation_idx, action_idx, next_observation_idx;
	uint reward_idx, terminated_idx, truncated_idx;
	uint out_observation_idx, out_action_idx, out_next_observation_idx;
	uint out_reward_idx, out_terminated_idx, out_truncated_idx;
	uint cursor, capacity, batch, observation_elements, action_elements;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x;
	uint observationCount = pc.batch * pc.observation_elements;
	if (i < observationCount) {
		uint lane = i / pc.observation_elements;
		uint component = i - lane * pc.observation_elements;
		uint destination = ((pc.cursor + lane) % pc.capacity)
			* pc.observation_elements + component;
		heap[pc.out_observation_idx].Store(destination * 4u,
			heap[pc.observation_idx].Load(i * 4u));
		heap[pc.out_next_observation_idx].Store(destination * 4u,
			heap[pc.next_observation_idx].Load(i * 4u));
	}
	uint actionCount = pc.batch * pc.action_elements;
	if (i < actionCount) {
		uint lane = i / pc.action_elements;
		uint component = i - lane * pc.action_elements;
		uint destination = ((pc.cursor + lane) % pc.capacity)
			* pc.action_elements + component;
		heap[pc.out_action_idx].Store(destination * 4u,
			heap[pc.action_idx].Load(i * 4u));
	}
	if (i >= pc.batch) return;
	uint destination = (pc.cursor + i) % pc.capacity;
	heap[pc.out_reward_idx].Store(destination * 4u,
		heap[pc.reward_idx].Load(i * 4u));
	heap[pc.out_terminated_idx].Store<uint8_t>(destination,
		heap[pc.terminated_idx].Load<uint8_t>(i));
	heap[pc.out_truncated_idx].Store<uint8_t>(destination,
		heap[pc.truncated_idx].Load<uint8_t>(i));
}
