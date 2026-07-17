struct PushConstants {
	uint reward_idx, next_q1_idx, next_q2_idx, next_log_probability_idx;
	uint terminated_idx, truncated_idx, out_idx, batch;
	float discount, entropy_coefficient;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint index = tid.x;
	if (index >= pc.batch) return;
	float reward = asfloat(heap[pc.reward_idx].Load(index * 4u));
	float q1 = asfloat(heap[pc.next_q1_idx].Load(index * 4u));
	float q2 = asfloat(heap[pc.next_q2_idx].Load(index * 4u));
	float logProbability = asfloat(
		heap[pc.next_log_probability_idx].Load(index * 4u));
	bool terminated = heap[pc.terminated_idx].Load<uint8_t>(index) != 0;
	float bootstrap = min(q1, q2) - pc.entropy_coefficient * logProbability;
	float target = reward + (terminated ? 0.0f : pc.discount * bootstrap);
	heap[pc.out_idx].Store(index * 4u, asuint(target));
}
