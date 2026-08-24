// Reverse affine summaries for the state adjoint of every Mamba-3 chunk.

import storage;

static const uint THREADS = 128u;
static const uint CHUNK = 16u;

struct PushConstants {
	uint crot_idx;
	uint decay_idx;
	uint dpre_idx;
	uint summary_idx;
	uint B;
	uint L;
	uint H;
	uint P;
	uint N;
	uint chunks;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint group = groupId.x;
	uint bh = group / pc.chunks;
	uint chunk = group - bh * pc.chunks;
	if (bh >= pc.B * pc.H) return;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	uint begin = chunk * CHUNK;
	uint end = min(begin + CHUNK, pc.L);

	for (uint flat = threadId.x; flat < pc.P * pc.N; flat += THREADS) {
		uint p = flat / pc.N;
		uint n = flat - p * pc.N;
		float adjoint = 0.0f;
		for (uint rr = 0u; rr < end - begin; ++rr) {
			uint t = end - 1u - rr;
			uint scalar = (b * pc.L + t) * pc.H + h;
			uint vector = scalar * pc.P + p;
			uint state = scalar * pc.N + n;
			adjoint = oaLoadF32(heap[pc.decay_idx], scalar)
				* (oaLoadF32(heap[pc.dpre_idx], vector)
					* oaLoadF32(heap[pc.crot_idx], state) + adjoint);
		}
		uint output = ((bh * pc.chunks + chunk) * pc.P + p) * pc.N + n;
		oaStoreF32(heap[pc.summary_idx], output, adjoint);
	}
}
