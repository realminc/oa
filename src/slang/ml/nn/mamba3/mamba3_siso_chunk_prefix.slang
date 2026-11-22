// Prefix the affine forward summaries and publish every chunk-entry state.

import storage;
import attributes;

static const uint THREADS = 128u;

struct PushConstants {
	uint summary_idx;
	uint product_idx;
	uint entry_idx;
	uint bh;
	uint p;
	uint n;
	uint chunks;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_siso_chunk_prefix")]
[domain("ml")]
[variant("chunk16")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
	uint bh = groupId.x;
	if (bh >= push.bh) return;
	for (uint flat = threadId.x; flat < push.p * push.n; flat += THREADS) {
		uint p = flat / push.n;
		uint n = flat - p * push.n;
		float state = 0.0f;
		for (uint chunk = 0u; chunk < push.chunks; ++chunk) {
			uint index = ((bh * push.chunks + chunk) * push.p + p) * push.n + n;
			store_f32(storage_buffers[push.entry_idx], index, state);
			state = load_f32(storage_buffers[push.product_idx], bh * push.chunks + chunk) * state
				+ load_f32(storage_buffers[push.summary_idx], index);
		}
	}
}
