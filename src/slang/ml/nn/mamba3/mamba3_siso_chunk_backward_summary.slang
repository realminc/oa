// Reverse affine summaries for the state adjoint of every Mamba-3 chunk.

import storage;
import attributes;

static const uint THREADS = 128u;
static const uint CHUNK = 16u;

struct PushConstants {
	uint crot_idx;
	uint decay_idx;
	uint dpre_idx;
	uint summary_idx;
	uint b;
	uint l;
	uint h;
	uint p;
	uint n;
	uint chunks;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_chunk_backward_summary")]
[domain("ml")]
[variant("chunk16")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint group = groupId.x;
	uint bh = group / push.chunks;
	uint chunk = group - bh * push.chunks;
	if (bh >= push.b * push.h) return;
	uint h = bh % push.h;
	uint b = bh / push.h;
	uint begin = chunk * CHUNK;
	uint end = min(begin + CHUNK, push.l);

	for (uint flat = threadId.x; flat < push.p * push.n; flat += THREADS) {
		uint p = flat / push.n;
		uint n = flat - p * push.n;
		float adjoint = 0.0f;
		for (uint rr = 0u; rr < end - begin; ++rr) {
			uint t = end - 1u - rr;
			uint scalar = (b * push.l + t) * push.h + h;
			uint vector = scalar * push.p + p;
			uint state = scalar * push.n + n;
			adjoint = load_f32(storage_buffers[push.decay_idx], scalar)
				* (load_f32(storage_buffers[push.dpre_idx], vector)
					* load_f32(storage_buffers[push.crot_idx], state) + adjoint);
		}
		uint output = ((bh * push.chunks + chunk) * push.p + p) * push.n + n;
		store_f32(storage_buffers[push.summary_idx], output, adjoint);
	}
}
