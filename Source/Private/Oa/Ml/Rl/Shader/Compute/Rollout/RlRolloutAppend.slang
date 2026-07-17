// Fused categorical rollout append. One dispatch writes every field for one
// vector-environment timestep into fixed time-major storage.

struct PushConstants {
	uint observation_idx;
	uint action_idx;
	uint reward_idx;
	uint value_idx;
	uint next_value_idx;
	uint log_probability_idx;
	uint terminated_idx;
	uint truncated_idx;
	uint out_observation_idx;
	uint out_action_idx;
	uint out_reward_idx;
	uint out_value_idx;
	uint out_next_value_idx;
	uint out_log_probability_idx;
	uint out_terminated_idx;
	uint out_truncated_idx;
	uint out_valid_idx;
	uint step;
	uint environments;
	uint observation_elements;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

float loadF32(RWByteAddressBuffer buffer, uint index) {
	return asfloat(buffer.Load(index * 4u));
}

void storeF32(RWByteAddressBuffer buffer, uint index, float value) {
	buffer.Store(index * 4u, asuint(value));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint index = tid.x;
	uint observationCount = pc.environments * pc.observation_elements;
	if (index < observationCount) {
		uint outputIndex = pc.step * observationCount + index;
		storeF32(heap[pc.out_observation_idx], outputIndex,
			loadF32(heap[pc.observation_idx], index));
	}
	if (index >= pc.environments) return;

	uint outputIndex = pc.step * pc.environments + index;
	heap[pc.out_action_idx].Store(
		outputIndex * 4u, heap[pc.action_idx].Load(index * 4u));
	storeF32(heap[pc.out_reward_idx], outputIndex,
		loadF32(heap[pc.reward_idx], index));
	storeF32(heap[pc.out_value_idx], outputIndex,
		loadF32(heap[pc.value_idx], index));
	storeF32(heap[pc.out_next_value_idx], outputIndex,
		loadF32(heap[pc.next_value_idx], index));
	storeF32(heap[pc.out_log_probability_idx], outputIndex,
		loadF32(heap[pc.log_probability_idx], index));
	heap[pc.out_terminated_idx].Store<uint8_t>(outputIndex,
		heap[pc.terminated_idx].Load<uint8_t>(index));
	heap[pc.out_truncated_idx].Store<uint8_t>(outputIndex,
		heap[pc.truncated_idx].Load<uint8_t>(index));
	heap[pc.out_valid_idx].Store<uint8_t>(outputIndex, uint8_t(1));
}
