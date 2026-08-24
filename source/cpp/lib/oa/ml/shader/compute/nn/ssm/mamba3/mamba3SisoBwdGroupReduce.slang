// Deterministic Mamba-3 Q/K gradient reduction from heads back to groups.
//
// The forward scan maps each head to one shared Q/K group.  Backward computes
// independent per-head contributions, then this kernel restores the grouped
// input contract without atomics or a generic broadcast-autograd subgraph.

import storage;

static const uint THREADS = 256u;

struct PushConstants {
	uint dc_head_idx;
	uint db_head_idx;
	uint dc_group_idx;
	uint db_group_idx;
	uint B;
	uint L;
	uint H;
	uint G;
	uint N;
	uint count;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
	uint index = dispatchId.x;
	if (index >= pc.count) return;
	uint n = index % pc.N;
	uint tokenGroup = index / pc.N;
	uint g = tokenGroup % pc.G;
	uint token = tokenGroup / pc.G;
	uint headsPerGroup = pc.H / pc.G;
	uint firstHead = g * headsPerGroup;
	float dc = 0.0f;
	float db = 0.0f;
	for (uint localHead = 0u; localHead < headsPerGroup; ++localHead) {
		uint h = firstHead + localHead;
		uint headIndex = (token * pc.H + h) * pc.N + n;
		dc += oaLoadF32(heap[pc.dc_head_idx], headIndex);
		db += oaLoadF32(heap[pc.db_head_idx], headIndex);
	}
	oaStoreF32(heap[pc.dc_group_idx], index, dc);
	oaStoreF32(heap[pc.db_group_idx], index, db);
}
