struct PushConstants {
	uint reward_idx, next_q_idx, terminated_idx, truncated_idx, out_idx;
	uint batch, actions;
	float discount;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint row = tid.x;
	if (row >= pc.batch) return;
	float next = -3.402823466e+38f;
	for (uint action = 0; action < pc.actions; ++action) {
		next = max(next, asfloat(heap[pc.next_q_idx].Load(
			(row * pc.actions + action) * 4u)));
	}
	float reward = asfloat(heap[pc.reward_idx].Load(row * 4u));
	bool terminated = heap[pc.terminated_idx].Load<uint8_t>(row) != 0;
	// pc.truncated_idx is intentionally not part of target suppression:
	// truncation stops an episode trace but retains bootstrap value.
	float target = reward + (terminated ? 0.0f : pc.discount * next);
	heap[pc.out_idx].Store(row * 4u, asuint(target));
}
