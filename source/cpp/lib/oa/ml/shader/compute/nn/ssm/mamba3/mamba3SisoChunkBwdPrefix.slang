// Reverse-prefix chunk summaries and publish each chunk's exit adjoint.

import storage;

static const uint THREADS = 128u;

struct PushConstants {
	uint summary_idx;
	uint product_idx;
	uint exit_idx;
	uint BH;
	uint P;
	uint N;
	uint chunks;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint bh = groupId.x;
	if (bh >= pc.BH) return;
	for (uint flat = threadId.x; flat < pc.P * pc.N; flat += THREADS) {
		uint p = flat / pc.N;
		uint n = flat - p * pc.N;
		float adjoint = 0.0f;
		for (uint rr = 0u; rr < pc.chunks; ++rr) {
			uint chunk = pc.chunks - 1u - rr;
			uint index = ((bh * pc.chunks + chunk) * pc.P + p) * pc.N + n;
			oaStoreF32(heap[pc.exit_idx], index, adjoint);
			adjoint = oaLoadF32(heap[pc.product_idx], bh * pc.chunks + chunk) * adjoint
				+ oaLoadF32(heap[pc.summary_idx], index);
		}
	}
}
