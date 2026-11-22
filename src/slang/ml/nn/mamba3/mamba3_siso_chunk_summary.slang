// Parallel Mamba-3 forward chunk summaries.
//
// Each invocation owns one (value row, state column) in one sequence chunk and
// reduces that chunk from a zero entry state. The resulting affine summary
// maps a chunk-entry state H to product*H + update.

import storage;
import attributes;

static const uint THREADS = 128u;
static const uint CHUNK = 16u;

struct PushConstants {
	uint brot_idx;
	uint x_idx;
	uint decay_idx;
	uint scale_idx;
	uint summary_idx;
	uint product_idx;
	uint b;
	uint l;
	uint h;
	uint p;
	uint n;
	uint chunks;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_chunk_summary")]
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
		float update = 0.0f;
		for (uint t = begin; t < end; ++t) {
			uint scalar = (b * push.l + t) * push.h + h;
			uint vector = scalar * push.p + p;
			uint state = scalar * push.n + n;
			update = load_f32(storage_buffers[push.decay_idx], scalar) * update
				+ load_f32(storage_buffers[push.scale_idx], scalar)
					* load_f32(storage_buffers[push.x_idx], vector)
					* load_f32(storage_buffers[push.brot_idx], state);
		}
		uint output = ((bh * push.chunks + chunk) * push.p + p) * push.n + n;
		store_f32(storage_buffers[push.summary_idx], output, update);
	}

	if (threadId.x == 0u) {
		float product = 1.0f;
		for (uint t = begin; t < end; ++t) {
			uint scalar = (b * push.l + t) * push.h + h;
			product *= load_f32(storage_buffers[push.decay_idx], scalar);
		}
		store_f32(storage_buffers[push.product_idx], bh * push.chunks + chunk, product);
	}
}
