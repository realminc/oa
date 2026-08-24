// Parallel Mamba-3 forward chunk summaries.
//
// Each invocation owns one (value row, state column) in one sequence chunk and
// reduces that chunk from a zero entry state. The resulting affine summary
// maps a chunk-entry state H to product*H + update.

import storage;

static const uint THREADS = 128u;
static const uint CHUNK = 16u;

struct PushConstants {
	uint brot_idx;
	uint x_idx;
	uint decay_idx;
	uint scale_idx;
	uint summary_idx;
	uint product_idx;
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
		float update = 0.0f;
		for (uint t = begin; t < end; ++t) {
			uint scalar = (b * pc.L + t) * pc.H + h;
			uint vector = scalar * pc.P + p;
			uint state = scalar * pc.N + n;
			update = oaLoadF32(heap[pc.decay_idx], scalar) * update
				+ oaLoadF32(heap[pc.scale_idx], scalar)
					* oaLoadF32(heap[pc.x_idx], vector)
					* oaLoadF32(heap[pc.brot_idx], state);
		}
		uint output = ((bh * pc.chunks + chunk) * pc.P + p) * pc.N + n;
		oaStoreF32(heap[pc.summary_idx], output, update);
	}

	if (threadId.x == 0u) {
		float product = 1.0f;
		for (uint t = begin; t < end; ++t) {
			uint scalar = (b * pc.L + t) * pc.H + h;
			product *= oaLoadF32(heap[pc.decay_idx], scalar);
		}
		oaStoreF32(heap[pc.product_idx], bh * pc.chunks + chunk, product);
	}
}
